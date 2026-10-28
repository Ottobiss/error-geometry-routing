"""Признаки патчей для обученного роутера (routing.tile_router): строка на патч.

    python -m features.tile_features --dataset cityscapes --tiles image_t256_m64
    python -m features.tile_features --dataset acdc --tiles small_t256_m64

Признаки патча (по пикселям патча, из выхода B0):
  ent_mean, ent_p90, ent_frac (доля пикселей с энтропией > 0.5), margin_mean (1 − (p1 − p2)), pmax_mean,
  boundary (доля пикселей, у которых сосед справа или снизу другого класса), n_comp (компонент в патче),
  cls_<c> (доля класса c в предсказании B0);
  m_<метод> — оценка метода роутера (routing.summary, fit на CS P0), перенесённая с компонент на патч:
  entropy, md (MD по выходу B0), ddu, knn, proto, maha.
Цель: delta (Δ патча в пикселях, B2 по патчам — features.tiles), n_valid (размеченных пикселей).
Выход: results/<dataset>/tile_features_<cfg>.parquet.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from core.components import IGNORE, NC
from core.data import DATASETS, FIT_DATASET, FIT_POP, load_units, populations
from core.evaluation import entropy_score, routing_cache
from core.paths import DatasetResults
from core.tiles import comp_to_tiles, grid, tile_effects, tile_index, tile_scores
from features.logits import LogitStore
from routing.methods import METHODS
from routing.summary import configs

METHOD_COLUMNS = {"entropy": "entropy", "md": "md_softmax_e1e-4", "ddu": "ddu_logp_mean",
                  "knn": "knn_K_A_dnp_mean", "proto": "proto_A_spred_mean", "maha": "maha_emb_class"}


def pixel_stats(logits, pred, boxes, tidx, n_tiles):
    """Статистики уверенности и формы по патчам: dict колонок (n_tiles,)."""
    p = logits.softmax(0)
    ent = entropy_score(p)
    top = p.topk(2, dim=0).values
    t = torch.from_numpy(tidx).cuda().ravel()
    area = torch.bincount(t, minlength=n_tiles).double()

    def mean(x):
        return (torch.zeros(n_tiles, dtype=torch.float64, device="cuda")
                .index_add_(0, t, x.ravel().double()) / area).cpu().numpy()

    edge = np.zeros(pred.shape, bool)
    edge[:, :-1] |= pred[:, :-1] != pred[:, 1:]
    edge[:-1, :] |= pred[:-1, :] != pred[1:, :]
    cols = {"ent_mean": mean(ent), "ent_frac": mean((ent > 0.5).float()),
            "margin_mean": mean(1 - (top[0] - top[1])), "pmax_mean": mean(top[0]),
            "ent_p90": np.array([torch.quantile(ent[y0:y1, x0:x1].float(), 0.9).item()
                                 for x0, y0, x1, y1 in boxes]),
            "boundary": np.bincount(tidx.ravel(), weights=edge.ravel(), minlength=n_tiles) / area.cpu().numpy()}
    cls = np.bincount(tidx.ravel() * NC + pred.ravel(), minlength=n_tiles * NC).reshape(n_tiles, NC)
    cols.update({f"cls_{c}": cls[:, c] / area.cpu().numpy() for c in range(NC)})
    return cols


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    ap.add_argument("--tiles", default="image_t256_m64")
    args = ap.parse_args()

    res = DatasetResults(args.dataset)
    info = json.loads(res.tiles(args.tiles, "json").read_text(encoding="utf-8"))
    size, tile = DATASETS[args.dataset].size, info["tile"]
    z = np.load(res.tiles(args.tiles))
    tile_pos, tiled = {i: k for k, i in enumerate(z["image_id"])}, z["tiled"]   # распаковать один раз
    boxes, tidx = grid(size, tile), tile_index(size, tile)
    n_tiles = len(boxes)
    logits = LogitStore(args.dataset)

    cfgs = configs()
    train = load_units(FIT_DATASET, FIT_POP, labeled_only=True)
    methods = {col: METHODS[cfgs[name][0]](**cfgs[name][1]).fit(train) for col, name in METHOD_COLUMNS.items()}

    rows = []
    pops = ["p0", "p1"] if args.dataset == FIT_DATASET else populations(args.dataset)
    for pop in pops:
        units, cache = load_units(args.dataset, pop), routing_cache(args.dataset, pop)
        scores = {col: m.score(units) for col, m in methods.items()}
        for image_id, idx in units.table.groupby("image_id").indices.items():
            k = cache["pos"][image_id]
            comp, gt, b0 = cache["comp"][k], cache["gt"][k], cache["b0"][k]
            overlap = comp_to_tiles(comp, len(idx), tidx, n_tiles)
            b2t = tiled[tile_pos[image_id]]
            _, _, delta = tile_effects(gt, b0, b2t, tidx, n_tiles)
            valid = gt != IGNORE
            # These two costs make the post-hoc deferral loss possible.  They
            # are labels only: a router never receives them at inference time.
            b0_error = np.bincount(tidx.ravel(), weights=(valid & (b0 != gt)).ravel(), minlength=n_tiles)
            b2_error = np.bincount(tidx.ravel(), weights=(valid & (b2t != gt)).ravel(), minlength=n_tiles)
            cols = pixel_stats(logits[image_id], b0.astype(np.int64), boxes, tidx, n_tiles)
            cols.update({f"m_{c}": tile_scores(s[idx], overlap) for c, s in scores.items()})
            rows.append(pd.DataFrame({
                "image_id": image_id, "population": pop, "tile": np.arange(n_tiles), **cols,
                "n_comp": (overlap > 0).sum(0), "delta": delta,
                "n_valid": np.bincount(tidx.ravel(), weights=valid.ravel(), minlength=n_tiles),
                "b0_error": b0_error, "b2_error": b2_error}))
        print(f"{args.dataset}:{pop}: {len(units.table.image_id.unique())} кадров")
    table = pd.concat(rows, ignore_index=True)
    out = res.dir / f"tile_features_{args.tiles}.parquet"
    table.to_parquet(out)
    print(f"{out}: {table.shape}")


if __name__ == "__main__":
    main()
