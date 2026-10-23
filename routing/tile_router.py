"""Обученный роутер на патчах: регрессия Δ патча по его признакам против энтропии при одинаковом k.

    python -m routing.tile_router

Признаки — features.tile_features. Регрессор — HistGBR (как routing.methods.learned): цель — Δ патча на
размеченный пиксель, вес — число размеченных пикселей. Наборы признаков:
  энтропия      — только m_entropy (контроль: то же обучение на одной энтропии);
  уверенность   — статистики уверенности B0, доля границ, число компонент, доли классов;
  +MD           — уверенность + энтропия и MD по выходу B0 (Henriksson et al.);
  +геометрия    — +MD + DDU, kNN, прототипы, Махаланобис по эмбеддингам.
Обучение и тест:
  CS P0 → CS P1;  CS P0 → ACDC;  ACDC LOCO (три условия → четвёртое);  CS P0 + ACDC LOCO.
Оценка: k патчей на кадр (одинаковое время у всех), mIoU, разница с энтропией без обучения (m_entropy)
и парный bootstrap по кадрам (2000, сид 0). Выход: results/summary/tile_router.csv.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from core.components import NC
from core.evaluation import _miou, _miou_batch, routing_cache
from core.paths import RESULTS, DatasetResults
from core.tiles import tile_effects, tile_index
from core.data import DATASETS
from routing.tiles import select

UNCERTAINTY = ["ent_mean", "ent_p90", "ent_frac", "margin_mean", "pmax_mean", "boundary", "n_comp"] \
    + [f"cls_{c}" for c in range(NC)]
GROUPS = {"энтропия": ["m_entropy"],
          "уверенность": UNCERTAINTY,
          "+MD": UNCERTAINTY + ["m_entropy", "m_md"],
          "+геометрия": UNCERTAINTY + ["m_entropy", "m_md", "m_ddu", "m_knn", "m_proto", "m_maha"]}
TESTS = {"cityscapes:p1": "image_t256_m64", "acdc:all": "small_t256_m64"}
KS = (1, 2, 3)


def fit_predict(train, test, cols):
    train = train[train.n_valid > 0]
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=50, random_state=0)
    model.fit(train[cols], train.delta / train.n_valid, sample_weight=train.n_valid)
    # У бустинга выход кусочно-постоянный: равные оценки в кадре разбиваются энтропией, а не номером патча.
    return model.predict(test[cols]) + 1e-9 * test.m_entropy.values


def loco(acdc, cols, extra=None):
    """Предсказания для ACDC вне выборки: каждое условие — моделью, обученной на трёх других (+ extra)."""
    out = np.zeros(len(acdc))
    for cond in acdc.population.unique():
        test = acdc.population.values == cond
        train = acdc[~test] if extra is None else pd.concat([acdc[~test], extra])
        out[test] = fit_predict(train, acdc[test], cols)
    return out


def effects(test_spec, tiles_cfg, image_ids):
    """Вклад патчей при замене B0 на B2 по кадрам: {image_id: (d, c0)}."""
    dataset, pop = test_spec.split(":")
    res, cache = DatasetResults(dataset), routing_cache(dataset, pop)
    z = np.load(res.tiles(tiles_cfg))
    pos, tiled = {i: k for k, i in enumerate(z["image_id"])}, z["tiled"]       # распаковать один раз
    size = DATASETS[dataset].size
    tile = int(tiles_cfg.split("_t")[1].split("_")[0])
    tidx = tile_index(size, tile)
    n = tidx.max() + 1
    out = {}
    for image_id in image_ids:
        k = cache["pos"][image_id]
        d, c0, _ = tile_effects(cache["gt"][k], cache["b0"][k], tiled[pos[image_id]], tidx, n)
        out[image_id] = (d, c0)
    return out


def evaluate(test_spec, table, scores, eff, n_boot=2000):
    """Строки: метод × k — mIoU, Δ к entropy и 95%-интервал bootstrap по кадрам."""
    frames = list(table.groupby("image_id", sort=True).indices.items())
    rng = np.random.default_rng(0)
    w = rng.multinomial(len(frames), np.full(len(frames), 1 / len(frames)), size=n_boot)
    rows = []
    for k in KS:
        per = {}
        for name, s in scores.items():
            per[name] = np.stack([(eff[i][1] + eff[i][0][select(s[idx], k=k)].sum(0)).ravel()
                                  for i, idx in frames])
        ref = per["entropy"]
        ref_b = _miou_batch((w @ ref).reshape(-1, NC, NC))
        for name, m in per.items():
            diff = _miou_batch((w @ m).reshape(-1, NC, NC)) - ref_b
            miou = _miou(m.sum(0).reshape(NC, NC))
            rows.append({"тест": test_spec, "метод": name, "k": k, "mIoU": miou,
                         "Δentropy": miou - _miou(ref.sum(0).reshape(NC, NC)),
                         "ci_lo": np.percentile(diff, 2.5), "ci_hi": np.percentile(diff, 97.5)})
    return rows


def main():
    cs = pd.read_parquet(DatasetResults("cityscapes").dir / f"tile_features_{TESTS['cityscapes:p1']}.parquet")
    acdc = pd.read_parquet(DatasetResults("acdc").dir / f"tile_features_{TESTS['acdc:all']}.parquet")
    cs_p0, cs_p1 = cs[cs.population == "p0"], cs[cs.population == "p1"]
    acdc = acdc.sort_values(["image_id", "tile"]).reset_index(drop=True)
    cs_p1 = cs_p1.sort_values(["image_id", "tile"]).reset_index(drop=True)

    def base_scores(t):
        oracle = np.where(t.delta > 0, t.delta / t.n_valid.clip(lower=1), -np.inf)
        return {"entropy": t.m_entropy.values, "oracle": oracle}

    rows = []
    scores = base_scores(cs_p1)
    for g, cols in GROUPS.items():
        scores[f"CS P0 · {g}"] = fit_predict(cs_p0, cs_p1, cols)
    rows += evaluate("cityscapes:p1", cs_p1, scores,
                     effects("cityscapes:p1", TESTS["cityscapes:p1"], cs_p1.image_id.unique()))

    scores = base_scores(acdc)
    for g, cols in GROUPS.items():
        scores[f"CS P0 · {g}"] = fit_predict(cs_p0, acdc, cols)
        scores[f"ACDC LOCO · {g}"] = loco(acdc, cols)
        scores[f"CS P0 + ACDC LOCO · {g}"] = loco(acdc, cols, extra=cs_p0)
    rows += evaluate("acdc:all", acdc, scores, effects("acdc:all", TESTS["acdc:all"], acdc.image_id.unique()))

    out = pd.DataFrame(rows)
    (RESULTS / "summary").mkdir(exist_ok=True)
    out.to_csv(RESULTS / "summary" / "tile_router.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cell = out.apply(lambda r: f"{r.mIoU:.2f} ({r.Δentropy:+.2f} [{r.ci_lo:+.2f},{r.ci_hi:+.2f}])", axis=1)
    for test_spec in TESTS:
        t = out[out.тест == test_spec].assign(cell=cell)
        print(f"\n{test_spec}: mIoU (Δ к энтропии [95%]) при k патчах на кадр")
        print(t.pivot(index="метод", columns="k", values="cell").to_string())


if __name__ == "__main__":
    main()
