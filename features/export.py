"""Таблицы признаков для обучения: одна на компоненты, одна на патчи (строка — единица, все датасеты).

    python -m features.export              # results/tabular/components.parquet и tiles.parquet
    python -m features.export --emb        # + 256 столбцов вектора декодера B0 (emb_*) у компонент

Компоненты — CS P0 / P1 и ACDC fog / night / rain / snow:
  ключи   dataset, population, image_id, comp_id, class_id;
  цели    delta, n_valid, has_valid, y_fail — по разметке, в признаки не брать;
  conf_   уверенность B0 (поля p Kang et al.), shape_ — форма, cls_ — класс B0 (one-hot), ms_ — MetaSeg;
  md_     MD по выходу B0 (softmax, eps=1e-4 — как md_softmax_e1e-4): mean / p90 / frac (> 95%-квантиля
          MD верных пикселей класса на P0); maha_ — Махаланобис по вектору компоненты, shared / class
          (eps=1e-4). Оба считаются здесь, fit на CS P0 — как в routing.summary;
  ddu_, knn_, proto_ — готовые таблицы results/<dataset>/features/ (features.build).
Патчи — features.tile_features (CS — окна из полного кадра, ACDC — из 1024×576) с теми же префиксами:
  conf_, shape_, cls_; m_ — оценки методов, перенесённые с компонент на патч; цели delta, n_valid.
Признаки, подогнанные на CS P0, на строках P0 посчитаны на выборке подгонки.
"""
import argparse

import numpy as np
import pandas as pd
import torch

from core.components import NC, P_NUMERIC
from core.data import FIT_DATASET, FIT_POP, load_units, populations
from core.evaluation import routing_cache
from core.paths import RESULTS, DatasetResults, write_json
from features.logits import LogitStore
from routing.methods import METHODS
from routing.methods.density import OutputStats, aggregate, fit_samples

DATASETS = ("cityscapes", "acdc")
TILES = {"cityscapes": "image_t256_m64", "acdc": "small_t256_m64"}
SHAPE = ["area", "boundary_size", "interior_size", "relative_component_area", "relative_boundary_size",
         "relative_interior_size", "touches_image_boundary", "has_interior"]
FEATURE_TABLES = {"ddu": "ddu_", "knn": "knn_", "prototypes": "proto_"}
MD_Q = 0.95


def rename_components(t):
    cols = {c: ("shape_" if c in SHAPE else "conf_") + c for c in P_NUMERIC}
    cols.update({f"class_{c}": f"cls_{c}" for c in range(NC)})
    return t.rename(columns=cols)


@torch.inference_mode()
def md_columns(units, stats, thr):
    """MD по выходу B0 по компонентам популяции: одна карта на кадр, три агрегации."""
    cache, logits = routing_cache(units.dataset, units.population), LogitStore(units.dataset)
    out = {f"md_{a}": np.zeros(len(units.table)) for a in ("mean", "p90", "frac")}
    for image_id, idx in units.table.groupby("image_id").indices.items():
        k = cache["pos"][image_id]
        pred, comp = cache["b0"][k], cache["comp"][k].astype(np.int64)
        md = stats.md_map(logits[image_id], pred)
        out["md_mean"][idx] = aggregate(md, comp, len(idx), "mean")
        out["md_p90"][idx] = aggregate(md, comp, len(idx), "p90")
        out["md_frac"][idx] = aggregate(md, comp, len(idx), "frac", thr[pred])
    return out


def components(emb=False):
    train = load_units(FIT_DATASET, FIT_POP, labeled_only=True)
    maha = {f"maha_{cov}": METHODS["mahalanobis"](cov=cov, eps=1e-4).fit(train) for cov in ("shared", "class")}
    samples = fit_samples(routing_cache(FIT_DATASET, FIT_POP), LogitStore(FIT_DATASET))
    stats = OutputStats("softmax", eps=1e-4).fit(samples)
    thr = stats.quantiles(samples, MD_Q)

    parts = []
    for dataset in DATASETS:
        res = DatasetResults(dataset)
        feats = [pd.read_parquet(res.feature(name)).set_index(["image_id", "comp_id"]).add_prefix(prefix)
                 for name, prefix in FEATURE_TABLES.items()]
        for pop in populations(dataset):
            units = load_units(dataset, pop)
            t = rename_components(units.table.drop(columns="density"))
            t.insert(0, "population", pop)
            t.insert(0, "dataset", dataset)
            for name, m in maha.items():
                t[name] = m.score(units)
            for name, col in md_columns(units, stats, thr).items():
                t[name] = col
            t = t.join(pd.concat(feats, axis=1), on=["image_id", "comp_id"])
            if emb:
                t = pd.concat([t, pd.DataFrame(units.emb, columns=[f"emb_{i}" for i in range(units.emb.shape[1])])],
                              axis=1)
            parts.append(t)
            print(f"{dataset}:{pop}: {len(t)} компонент")
    out = pd.concat(parts, ignore_index=True)
    assert out.notna().all().all(), out.columns[out.isna().any()].tolist()
    return out


def tiles():
    parts = []
    for dataset in DATASETS:
        t = pd.read_parquet(DatasetResults(dataset).dir / f"tile_features_{TILES[dataset]}.parquet")
        t = t.rename(columns={c: "conf_" + c for c in ("ent_mean", "ent_frac", "ent_p90", "margin_mean", "pmax_mean")}
                     | {"boundary": "shape_boundary", "n_comp": "shape_n_comp"})
        t.insert(0, "tiles", TILES[dataset])
        t.insert(0, "dataset", dataset)
        parts.append(t)
    out = pd.concat(parts, ignore_index=True)
    assert out.notna().all().all()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", action="store_true", help="добавить вектор декодера B0 компоненты (emb_*)")
    args = ap.parse_args()

    out_dir = RESULTS / "tabular"
    out_dir.mkdir(exist_ok=True)
    for name, table in (("tiles", tiles()), ("components", components(args.emb))):
        table.to_parquet(out_dir / f"{name}.parquet")
        print(f"{out_dir / name}.parquet: {table.shape}")
    write_json(out_dir / "config.json", {"script": "features.export", "emb": args.emb, "tiles": TILES,
                                         "md": {"variant": "softmax", "eps": 1e-4, "q": MD_Q},
                                         "maha": {"eps": 1e-4}})


if __name__ == "__main__":
    main()
