"""Политика на уровне кадра для каскада по патчам: только B0 / B2 на патчах / B2 на всём кадре.

    python -m routing.frame_policy                    # CS P1 (окна из полного кадра) и весь ACDC
    python -m routing.frame_policy --tests cityscapes:p1:image_t256_m64

После B0 на кадре берутся патчи с энтропией выше порога thr (оценки и пороги — как в routing.tiles,
квантили энтропии патчей CS P0). Если таких патчей больше n_max, B2 запускается на всём кадре, иначе —
на окнах выбранных патчей. n_max = число патчей — обычный каскад с порогом; n_max = 0 — «B0 или B2 на
кадре» по наличию хоть одного трудного патча. Отдельная политика «кадр» — B2 на всём кадре, если средняя
энтропия кадра выше порога (квантили по кадрам CS P0), иначе B0.
Время кадра — по замерам features.tiles: fp32 (ms) и fp16 (ms_fp16: B2 под autocast; B0 всегда fp32,
под autocast он медленнее). Время на решение и склейку не входит — как в routing.tiles.
Критерий (выбран до запуска): самая быстрая по среднему времени точка с mIoU ≥ B2 на кадре − 0.5.
На CS точка выбирается по P0 (те же патчи и замеры) и проверяется на тесте, bootstrap по кадрам теста
разницы с B2 на кадре; на ACDC P0 нет — точка выбирается по самому тесту (описательно).
Выход: results/summary/frame_policy_<датасет>_<популяция>.csv (все точки) и ..._pick.csv.
"""
import argparse
import json

import numpy as np
import pandas as pd

from core.components import IGNORE, NC
from core.data import DATASETS, FIT_DATASET, FIT_POP, load_units
from core.evaluation import _miou, _miou_batch, _conf, routing_cache
from core.paths import RESULTS, DatasetResults
from core.tiles import batch_time, tile_effects
from routing.methods import METHODS
from routing.tiles import QUANTILES, frame_tile_scores

TESTS = "cityscapes:p1:image_t256_m64,acdc:all:small_t256_m64"
SLACK = 0.5


def frame_data(dataset, pop, cfg, method):
    """Оценки патчей (F, T), вклад патчей (F, T, NC, NC), матрицы ошибок B0 и B2 на кадре (F, NC, NC),
    площади патчей (T,) и описание конфигурации патчей."""
    res = DatasetResults(dataset)
    info = json.loads(res.tiles(cfg, "json").read_text(encoding="utf-8"))
    z = dict(np.load(res.tiles(cfg)))                 # npz распаковывает массив при каждом обращении
    tile_pos = {i: k for k, i in enumerate(z["image_id"])}
    units, cache = load_units(dataset, pop), routing_cache(dataset, pop)
    scores, tidx, area = frame_tile_scores(units, cache, {"s": method.score(units)}, info["tile"],
                                           DATASETS[dataset].size)
    S, D, C0, C2 = [], [], [], []
    for image_id, s in scores.items():
        k, t = cache["pos"][image_id], tile_pos[image_id]
        gt = cache["gt"][k]
        d, c0, _ = tile_effects(gt, cache["b0"][k], z["tiled"][t], tidx, len(area))
        v = gt != IGNORE
        S.append(s["s"]), D.append(d), C0.append(c0)
        C2.append(_conf(gt[v].astype(np.int64), z["full"][t][v].astype(np.int64)))
    return np.stack(S), np.stack(D), np.stack(C0), np.stack(C2), area, info


def frame_costs(info, precision):
    """(B0 на кадре, B2 на кадре, функция «время B2 на k окнах») в мс."""
    ms = info["ms"] if precision == "fp32" else info["ms_fp16"]
    b0 = info["ms"]["b0_frame"]
    return b0, ms["b2_frame"], lambda k: batch_time(k, ms["b2_batch"], info["max_batch"])


def apply(S, D, C0, C2, thr, n_max):
    """Матрицы ошибок по кадрам (F, NC, NC), число патчей на кадр и флаг «B2 на всём кадре»."""
    sel = S > thr
    n = sel.sum(1)
    full = n > n_max
    confs = C0 + np.einsum("ft,ftij->fij", (sel & ~full[:, None]).astype(np.int64), D)
    confs[full] = C2[full]
    return confs, n, full


def curve(S, D, C0, C2, area, info, tile_thr, frame_thr):
    """Все точки политик на популяции для fp32 и fp16."""
    T = S.shape[1]
    fscore = S @ area / area.sum()
    rows = []
    for precision in ("fp32", "fp16"):
        b0, b2, win = frame_costs(info, precision)
        base = {"точность": precision}
        rows += [{**base, "политика": "только B0", "мс": b0, "мс_p95": b0, "mIoU": _miou(C0.sum(0))},
                 {**base, "политика": "B2 на кадре", "мс": b2, "мс_p95": b2, "mIoU": _miou(C2.sum(0))}]
        for q, thr in zip(QUANTILES, tile_thr):
            for n_max in range(T + 1):
                confs, n, full = apply(S, D, C0, C2, thr, n_max)
                t = b0 + np.where(full, b2, [win(k) for k in n])
                rows.append({**base, "политика": "патчи" if n_max == T else "гибрид", "q": q, "n_max": n_max,
                             "мс": t.mean(), "мс_p95": np.percentile(t, 95), "патчей": np.where(full, 0, n).mean(),
                             "доля_B2_кадр": full.mean(), "mIoU": _miou(confs.sum(0))})
        for q, thr in zip(QUANTILES, frame_thr):
            full = fscore > thr
            confs = np.where(full[:, None, None], C2, C0)
            t = b0 + np.where(full, b2, 0.0)
            rows.append({**base, "политика": "кадр", "q": q, "мс": t.mean(), "мс_p95": np.percentile(t, 95),
                         "доля_B2_кадр": full.mean(), "mIoU": _miou(confs.sum(0))})
    return pd.DataFrame(rows)


def pareto(df):
    """Точки, которые не хуже по mIoU и времени никакой другой точки той же точности."""
    df = df.sort_values(["мс", "mIoU"], ascending=[True, False])
    return df[df.mIoU > df.mIoU.cummax().shift(fill_value=-np.inf)]


def pick(df, precision):
    """Самая быстрая точка с mIoU ≥ B2 на кадре − SLACK (без опорных строк)."""
    d = df[df.точность == precision]
    target = d.loc[d.политика == "B2 на кадре", "mIoU"].item() - SLACK
    ok = d[(d.mIoU >= target) & ~d.политика.isin(["только B0", "B2 на кадре"])]
    return ok.sort_values("мс").iloc[0] if len(ok) else None


def boot_vs_full(confs, C2, n_boot=2000, seed=0):
    """Разница mIoU «политика − B2 на кадре», bootstrap по кадрам: (точка, низ, верх 95%)."""
    F = len(confs)
    w = np.random.default_rng(seed).multinomial(F, np.full(F, 1 / F), size=n_boot)
    diff = _miou_batch((w @ confs.reshape(F, -1)).reshape(-1, NC, NC)) \
        - _miou_batch((w @ C2.reshape(F, -1)).reshape(-1, NC, NC))
    return _miou(confs.sum(0)) - _miou(C2.sum(0)), *np.percentile(diff, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=TESTS, help="датасет:популяция:конфигурация features.tiles")
    args = ap.parse_args()

    method = METHODS["entropy"]().fit(load_units(FIT_DATASET, FIT_POP, labeled_only=True))
    out = RESULTS / "summary"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["политика", "q", "n_max", "мс", "мс_p95", "патчей", "доля_B2_кадр", "mIoU"]
    for spec in args.tests.split(","):
        dataset, pop, cfg = spec.split(":")
        # Пороги — по CS P0 на той же конфигурации патчей (у ACDC — по CS с тем же размером патча и окна).
        fit_cfg = cfg if dataset == FIT_DATASET else cfg.replace("small", "image")
        fit = frame_data(FIT_DATASET, FIT_POP, fit_cfg, method)
        S0, area0 = fit[0], fit[4]
        tile_thr = np.quantile(S0, QUANTILES)
        frame_thr = np.quantile(S0 @ area0 / area0.sum(), QUANTILES)

        S, D, C0, C2, area, info = data = frame_data(dataset, pop, cfg, method)
        table = curve(*data, tile_thr, frame_thr)
        table.to_csv(out / f"frame_policy_{dataset}_{pop}.csv", index=False)
        fit_table = curve(*fit, tile_thr, frame_thr) if dataset == FIT_DATASET else None

        print(f"\n{dataset}:{pop}, {cfg}: {len(S)} кадров, {S.shape[1]} патчей на кадр")
        picks = []
        for precision in ("fp32", "fp16"):
            d = table[table.точность == precision]
            print(f"\n{precision}: фронт Парето (все политики)")
            print(pareto(d)[cols].round(3).to_string(index=False))
            p = pick(fit_table if fit_table is not None else table, precision)
            if p is None:
                print(f"{precision}: нет точки с mIoU ≥ B2 − {SLACK}")
                continue
            if p.политика == "кадр":
                full = (S @ area / area.sum()) > frame_thr[list(QUANTILES).index(p.q)]
                confs = np.where(full[:, None, None], C2, C0)
            else:
                confs, _, _ = apply(S, D, C0, C2, tile_thr[list(QUANTILES).index(p.q)], p.n_max)
            key = (d.политика == p.политика) & (d.q == p.q) & ((d.n_max == p.n_max) | d.n_max.isna())
            row = d[key].iloc[0]
            est, lo, hi = boot_vs_full(confs, C2)
            b2_ms = d.loc[d.политика == "B2 на кадре", "мс"].item()
            picks.append({"тест": spec, "точность": precision, "выбор_по": "CS P0" if fit_table is not None
                          else "тест", "политика": p.политика, "q": p.q, "n_max": p.n_max,
                          "мс": row.мс, "мс_p95": row.мс_p95, "мс_B2": b2_ms, "ускорение": b2_ms / row.мс,
                          "mIoU": row.mIoU, "Δ_B2": est, "ci_lo": lo, "ci_hi": hi})
        picks = pd.DataFrame(picks)
        picks.to_csv(out / f"frame_policy_{dataset}_{pop}_pick.csv", index=False)
        print(f"\nКритерий: самая быстрая точка с mIoU ≥ B2 на кадре − {SLACK}, Δ к B2 [95% bootstrap]:")
        print(picks.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
