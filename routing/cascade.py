"""Каскад B0 → B2 по патчам вживую: реальное время на кадр и mIoU.

    python -m routing.cascade                       # CS P1, fp32 и fp16, несколько порогов
    python -m routing.cascade --test acdc:all --q 0.9,0.8

Кадр: B0 на всём кадре → средняя энтропия B0 по каждому патчу (core.tiles, 256 + контекст 64) →
патчи с энтропией выше порога → B2 на окнах с контекстом (пачками) → в этих патчах ответ B2.
Порог — квантиль q энтропии патчей на CS P0 (один для всех кадров: трудные кадры получают больше патчей).
Время — от кадра на GPU до готовой маски (чтение и уменьшение кадра не входят), медиана и p95.
Опорная точка — B2 на всём кадре, замеренная так же.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from core.components import IGNORE, NC
from core.data import DATASETS, FIT_DATASET, FIT_POP, get_dataset, loader, population
from core.evaluation import entropy_score
from core.models import load_model
from core.paths import RESULTS
from core.tiles import grid, tile_index
from features.tiles import tiled_prediction


class Cascade:
    def __init__(self, b0, b2, size, heavy_full, tile=256, margin=64, max_batch=2):
        self.b0, self.b2, self.size, self.heavy_full = b0, b2, size, heavy_full
        self.tile, self.margin, self.max_batch = tile, margin, max_batch
        self.boxes = grid(size, tile)
        tidx = torch.from_numpy(tile_index(size, tile)).cuda().ravel()
        self.tidx, self.area = tidx, torch.bincount(tidx).double()

    def tile_entropy(self, small):
        """B0 на кадре: (pred (H, W) на GPU, средняя энтропия по патчам (n_tiles,))."""
        probs = self.b0(small[None], self.size).softmax(1)[0]
        ent = entropy_score(probs.float()).ravel().double()
        return probs.argmax(0), torch.zeros_like(self.area).index_add_(0, self.tidx, ent) / self.area

    def __call__(self, sample, thr):
        """Маска каскада (H, W) uint8 и число патчей, отданных B2."""
        pred, score = self.tile_entropy(sample["small"])
        sel = torch.nonzero(score > thr).ravel().tolist()
        out = pred.to(torch.uint8).cpu().numpy()
        if sel:
            src = sample["image" if self.heavy_full else "small"]
            boxes = [self.boxes[i] for i in sel]
            b2 = tiled_prediction(self.b2, src, self.size, boxes, self.tile, self.margin, self.max_batch)
            for x0, y0, x1, y1 in boxes:
                out[y0:y1, x0:x1] = b2[y0:y1, x0:x1]
        return out, len(sel)


def miou(conf):
    d = np.diag(conf)
    u = conf.sum(0) + conf.sum(1) - d
    return 100 * np.nanmean(np.where(u > 0, d / np.maximum(u, 1), np.nan))


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="cityscapes:p1")
    ap.add_argument("--q", default="0.95,0.9,0.8,0.7", help="квантили энтропии патчей CS P0 — пороги")
    ap.add_argument("--precision", default="fp32,fp16")
    args = ap.parse_args()

    dataset, pop = args.test.split(":")
    cls = DATASETS[dataset]
    b0, b2 = load_model("b0"), load_model("b2")
    rows = []
    for precision in args.precision.split(","):
        with torch.autocast("cuda", dtype=torch.float16, enabled=precision == "fp16"):
            # Пороги — по энтропии патчей CS P0 (сетка CS).
            fit = Cascade(b0, b2, DATASETS[FIT_DATASET].size, True)
            fit_scores = np.concatenate([fit.tile_entropy(s["small"].cuda())[1].cpu().numpy() for s in
                                         loader(get_dataset(FIT_DATASET, population(FIT_DATASET, FIT_POP), full=False))])
            cascade = Cascade(b0, b2, cls.size, cls.heavy_full)
            ds = get_dataset(dataset, population(dataset, pop), full=cls.heavy_full)
            variants = {"B0": None, "B2 на кадре": None,
                        **{f"каскад q={q}": np.quantile(fit_scores, float(q)) for q in args.q.split(",")}}
            stats = {v: {"conf": np.zeros((NC, NC), np.int64), "ms": [], "tiles": []} for v in variants}
            for k, sample in enumerate(loader(ds)):
                sample = {key: v.cuda() if torch.is_tensor(v) and key != "label" else v for key, v in sample.items()}
                gt = sample["label"].numpy()
                valid = gt != IGNORE
                for v, thr in variants.items():
                    if v == "B0":
                        (pred, n), t = timed(lambda: (b0(sample["small"][None], cls.size).argmax(1)[0]
                                                      .to(torch.uint8).cpu().numpy(), 0))
                    elif v == "B2 на кадре":
                        src = sample["image" if cls.heavy_full else "small"]
                        (pred, n), t = timed(lambda: (b2(src[None], cls.size).argmax(1)[0]
                                                      .to(torch.uint8).cpu().numpy(), 0))
                    else:
                        (pred, n), t = timed(lambda: cascade(sample, thr))
                    if k >= 3:                                    # первые кадры — прогрев
                        stats[v]["ms"].append(t)
                    stats[v]["tiles"].append(n)
                    stats[v]["conf"] += np.bincount(gt[valid].astype(np.int64) * NC + pred[valid],
                                                    minlength=NC * NC).reshape(NC, NC)
            for v, s in stats.items():
                rows.append({"точность": precision, "вариант": v, "патчей": np.mean(s["tiles"]),
                             "мс_медиана": np.median(s["ms"]), "мс_среднее": np.mean(s["ms"]),
                             "мс_p95": np.percentile(s["ms"], 95), "mIoU": miou(s["conf"])})

    table = pd.DataFrame(rows)
    out = RESULTS / "summary"
    out.mkdir(exist_ok=True)
    table.to_csv(out / f"cascade_{dataset}_{pop}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{args.test}: {len(ds)} кадров, патч {cascade.tile} + контекст {cascade.margin}")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
