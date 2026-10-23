"""Каскад по патчам с учётом времени: все методы роутера, кривые «mIoU — мс на кадр».

    python -m routing.tiles --test acdc:all --tiles small_t256_m64
    python -m routing.tiles --test cityscapes:p1 --tiles image_t256_m64 --methods entropy,md_softmax_e1e-4

Каскад: B0 на всём кадре, затем B2 на окнах с контекстом вокруг выбранных патчей (features.tiles);
в выбранных патчах берётся ответ B2, в остальных — B0. Время кадра = B0 + B2 на k окнах (пачками,
по замерам features.tiles, fp32 на этом ноутбуке).
Оценка патча — среднее оценок компонент метода (как в routing.summary, fit на CS P0) по пикселям
патча; оракул — истинная Δ патча (только Δ > 0).
Режимы:
  fixed — k патчей на каждый кадр: время у всех методов одинаковое, bootstrap по кадрам против entropy;
  thr   — порог по оценке патча (квантили оценок патчей CS P0): число патчей на кадр разное, трудные
          кадры получают больше; точка кривой — среднее и 95-й перцентиль времени, mIoU.
Опорные точки: только B0, B2 на всём кадре того же входа, B2 на всех окнах.
Выход: results/summary/tiles_<датасет>_<популяция>_<cfg>.csv (кривые) и ..._boot.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

from core.components import IGNORE, NC
from core.data import DATASETS, FIT_DATASET, FIT_POP, load_units
from core.evaluation import _miou, _miou_batch, routing_cache
from core.paths import RESULTS, DatasetResults
from core.tiles import batch_time, comp_to_tiles, tile_effects, tile_index, tile_scores
from routing.methods import METHODS
from routing.summary import configs

QUANTILES = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def frame_tile_scores(units, cache, scores, tile, size):
    """{image_id: {метод: оценки патчей (n_tiles,)}} и площади патчей."""
    tidx = tile_index(size, tile)
    n_tiles = tidx.max() + 1
    out = {}
    for image_id, idx in units.table.groupby("image_id").indices.items():
        comp = cache["comp"][cache["pos"][image_id]]
        overlap = comp_to_tiles(comp, len(idx), tidx, n_tiles)
        out[image_id] = {m: tile_scores(s[idx], overlap) for m, s in scores.items()}
    return out, tidx, np.bincount(tidx.ravel())


def select(score, k=None, thr=None):
    """Номера выбранных патчей: k лучших (−inf не берутся) или все с оценкой > thr."""
    if thr is not None:
        return np.flatnonzero(score > thr)
    order = np.argsort(-score, kind="stable")[:k]
    return order[score[order] > -np.inf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="acdc:all")
    ap.add_argument("--tiles", default="small_t256_m64", help="конфигурация features.tiles")
    ap.add_argument("--methods", default="", help="через запятую; по умолчанию все из routing.summary")
    ap.add_argument("--boot-k", default="1,2,3")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    dataset, pop = args.test.split(":")
    res = DatasetResults(dataset)
    info = json.loads(res.tiles(args.tiles, "json").read_text(encoding="utf-8"))
    tile, ms = info["tile"], info["ms"]
    z = np.load(res.tiles(args.tiles))
    tile_pos = {i: k for k, i in enumerate(z["image_id"])}
    tiled, full = z["tiled"], z["full"]

    cfgs = configs()
    if args.methods:
        cfgs = {k: v for k, v in cfgs.items() if k in args.methods.split(",") + ["entropy"]}
    cfgs.pop("oracle", None)
    train = load_units(FIT_DATASET, FIT_POP, labeled_only=True)
    methods = {name: METHODS[m](**p).fit(train) for name, (m, p) in cfgs.items()}

    # Пороги — квантили оценок патчей на всех компонентах CS P0 (сетка CS того же размера патча).
    fit_units = load_units(FIT_DATASET, FIT_POP)
    fit_scores, _, _ = frame_tile_scores(fit_units, routing_cache(FIT_DATASET, FIT_POP),
                                         {n: m.score(fit_units) for n, m in methods.items()},
                                         tile, DATASETS[FIT_DATASET].size)
    thresholds = {n: np.quantile(np.concatenate([f[n] for f in fit_scores.values()]), QUANTILES)
                  for n in methods}

    test = load_units(dataset, pop)
    cache = routing_cache(dataset, pop)
    scores, tidx, area = frame_tile_scores(test, cache, {n: m.score(test) for n, m in methods.items()},
                                           tile, DATASETS[dataset].size)
    n_tiles = len(area)
    frames = list(scores)
    effects = {}
    for image_id in frames:
        k = cache["pos"][image_id]
        effects[image_id] = tile_effects(cache["gt"][k], cache["b0"][k], tiled[tile_pos[image_id]],
                                         tidx, n_tiles)
        delta = effects[image_id][2]
        scores[image_id]["oracle"] = np.where(delta > 0, delta / area, -np.inf)
    names = list(methods) + ["oracle"]

    def frame_time(k):
        return ms["b0_frame"] + batch_time(k, ms["b2_batch"], info["max_batch"])

    def run(name, **kw):
        """Матрицы ошибок по кадрам и число патчей на кадр при правиле выбора kw."""
        confs, counts = [], []
        for image_id in frames:
            d, c0, _ = effects[image_id]
            sel = select(scores[image_id][name], **kw)
            confs.append(c0 + d[sel].sum(0))
            counts.append(len(sel))
        return np.stack(confs), np.array(counts)

    c_b0 = sum(effects[i][1] for i in frames)
    c_full = np.zeros((NC, NC), np.int64)
    for image_id in frames:
        k = cache["pos"][image_id]
        gt, f = cache["gt"][k], full[tile_pos[image_id]]
        v = gt != IGNORE
        c_full += np.bincount(gt[v].astype(np.int64) * NC + f[v], minlength=NC * NC).reshape(NC, NC)
    rows = [{"метод": "только B0", "режим": "ref", "параметр": "", "мс": ms["b0_frame"], "мс_p95": ms["b0_frame"],
             "патчей": 0.0, "mIoU": _miou(c_b0)},
            {"метод": "B2 на кадре", "режим": "ref", "параметр": info["input"], "мс": ms["b2_frame"],
             "мс_p95": ms["b2_frame"], "патчей": 0.0, "mIoU": _miou(c_full)}]

    per_frame = {}
    for name in names:
        for k in range(1, n_tiles + 1):
            confs, counts = run(name, k=k)
            t = np.array([frame_time(c) for c in counts])
            rows.append({"метод": name, "режим": "fixed", "параметр": k, "мс": t.mean(),
                         "мс_p95": np.percentile(t, 95), "патчей": counts.mean(), "mIoU": _miou(confs.sum(0))})
            per_frame[name, k] = confs
        for q, thr in zip(QUANTILES, thresholds.get(name, [])):
            confs, counts = run(name, thr=thr)
            t = np.array([frame_time(c) for c in counts])
            rows.append({"метод": name, "режим": "thr", "параметр": q, "мс": t.mean(),
                         "мс_p95": np.percentile(t, 95), "патчей": counts.mean(), "mIoU": _miou(confs.sum(0))})
    curves = pd.DataFrame(rows)

    # Bootstrap по кадрам: разница с entropy при одинаковом k (одинаковое время).
    rng = np.random.default_rng(0)
    w = rng.multinomial(len(frames), np.full(len(frames), 1 / len(frames)), size=args.n_boot)
    boot = []
    for k in map(int, args.boot_k.split(",")):
        ref = per_frame["entropy", k].reshape(len(frames), -1)
        ref_b = _miou_batch((w @ ref).reshape(-1, NC, NC))
        for name in names:
            m = per_frame[name, k].reshape(len(frames), -1)
            diff = _miou_batch((w @ m).reshape(-1, NC, NC)) - ref_b
            boot.append({"метод": name, "k": k, "мс": frame_time(k),
                         "mIoU": _miou(m.sum(0).reshape(NC, NC)),
                         "Δentropy": _miou(m.sum(0).reshape(NC, NC)) - _miou(ref.sum(0).reshape(NC, NC)),
                         "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5)})
    boot = pd.DataFrame(boot)

    out = RESULTS / "summary"
    out.mkdir(exist_ok=True)
    stem = f"tiles_{dataset}_{pop}_{args.tiles}"
    curves.to_csv(out / f"{stem}.csv", index=False)
    boot.to_csv(out / f"{stem}_boot.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(f"{args.test}, {args.tiles}: {len(frames)} кадров, {n_tiles} патчей на кадр")
    print(curves[curves.режим == "ref"].round(2).to_string(index=False))
    print("\nk патчей на кадр (время одинаково для всех методов), mIoU:")
    fx = curves[curves.режим == "fixed"]
    t = fx.pivot(index="метод", columns="параметр", values="mIoU").round(2)
    t.columns = [f"k={k} ({frame_time(k):.0f} мс)" for k in t.columns]
    print(t.sort_values(t.columns[1], ascending=False).to_string())
    print("\nРазница с entropy при одинаковом k, bootstrap по кадрам [95%]:")
    b = boot.assign(cell=boot.apply(lambda r: f"{r['Δentropy']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]",
                                    axis=1))
    print(b.pivot(index="метод", columns="k", values="cell").to_string())


if __name__ == "__main__":
    main()
