"""Роутинг по выигрышу Δ: какие компоненты B0 отдать B2 при бюджете в пикселях. P0 → P1.

    python -m baseline.delta

Методы ранжируют компоненты кадра; B2 получает компоненты по порядку, пока их суммарная
площадь не превысит бюджет (доля пикселей кадра). Итоговая маска: B0, в отданных
компонентах — B2. Метрики на P1: mIoU итоговой маски и доля выигрыша оракула.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from baseline.failure import M_FIELDS, P_FIELDS, fit_p
from common.components import IGNORE, extract_components
from common.data import cityscapes_pairs, load_pair

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
NC = 19


def fit_density(train, fields):
    """Регрессия Δ на пиксель; вес строки — число валидных пикселей (шумные крошки весят мало)."""
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=50, random_state=0)
    model.fit(train[fields], train.density, sample_weight=train.n_valid)
    return model


def route(table, score, budget, hw):
    """comp_id компонент кадра, отданных B2 при бюджете budget·hw пикселей."""
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(table.area.values[order])
    return table.comp_id.values[order[cum <= budget * hw]]


def miou(conf):
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf))
    return 100 * np.nanmean(iou)


def main():
    df = pd.read_parquet("results/components_cityscapes_val.parquet")
    split = {i: name for name in ("p0", "p1")
             for i in Path(f"splits/cityscapes_{name}.txt").read_text().split()}
    df["pop"] = df.image_id.map(split)
    df["density"] = np.where(df.n_valid > 0, df.delta / df.n_valid.clip(lower=1), 0.0)
    p0 = df[(df["pop"] == "p0") & df.has_valid].reset_index(drop=True)
    p1 = df[df["pop"] == "p1"].reset_index(drop=True)          # роутим все компоненты

    rng = np.random.default_rng(0)
    p_fail = fit_p(p0)
    p_den, m_den = fit_density(p0, P_FIELDS), fit_density(p0, M_FIELDS)
    scores = {
        "random": rng.random(len(p1)),
        "entropy": p1.mean_entropy.values,
        "p (провал)": p_fail.predict_proba(p1[P_FIELDS])[:, 1],
        "p-поля → Δ": p_den.predict(p1[P_FIELDS]),
        "MetaSeg-поля → Δ": m_den.predict(p1[M_FIELDS]),
        "оракул": p1.density.values,
    }

    # Ранжирующие метрики на компонентах P1 с меткой
    lab = p1.has_valid.values
    print("Ранжирование на P1 (компоненты с меткой):")
    for name, s in scores.items():
        auc = roc_auc_score(p1.delta[lab] > 0, s[lab])
        rho = spearmanr(s[lab], p1.density[lab]).statistic
        print(f"  {name:18s} AUROC(Δ>0) {auc:.3f}   Spearman с Δ/пиксель {rho:.3f}")

    # Роутинг: mIoU итоговой маски и доля выигрыша оракула
    conf = {(n, b): np.zeros((NC, NC), np.int64) for n in scores for b in BUDGETS}
    conf_b0, conf_b2 = np.zeros((NC, NC), np.int64), np.zeros((NC, NC), np.int64)
    gain = {(n, b): 0 for n in scores for b in BUDGETS}
    paths = {i: (img, gt) for i, img, gt in cityscapes_pairs()}
    groups = p1.groupby("image_id").indices
    for image_id, idx in groups.items():
        masks = np.load(f"results/masks/{image_id}.npz")
        b0, b2 = masks["pred_b0"].astype(np.int64), masks["pred_b2"].astype(np.int64)
        _, gt = load_pair(*paths[image_id])
        comp, _ = extract_components(b0)
        assert comp.max() + 1 == len(idx), image_id
        valid = gt != IGNORE
        g = gt[valid].astype(np.int64)
        conf_b0 += np.bincount(g * NC + b0[valid], minlength=NC * NC).reshape(NC, NC)
        conf_b2 += np.bincount(g * NC + b2[valid], minlength=NC * NC).reshape(NC, NC)
        table = p1.iloc[idx]
        for name, s in scores.items():
            for b in BUDGETS:
                chosen = route(table, s[idx], b, gt.size)
                sel = np.isin(comp, chosen)
                merged = np.where(sel, b2, b0)
                conf[name, b] += np.bincount(g * NC + merged[valid],
                                             minlength=NC * NC).reshape(NC, NC)
                gain[name, b] += table.delta.values[np.isin(table.comp_id.values, chosen)].sum()

    print(f"\nmIoU на P1: только B0 {miou(conf_b0):.2f}, только B2 {miou(conf_b2):.2f}")
    rows = []
    for name in scores:
        row = {"метод": name}
        for b in BUDGETS:
            row[f"mIoU@{int(b * 100)}%"] = round(miou(conf[name, b]), 2)
        for b in BUDGETS:
            row[f"выигр@{int(b * 100)}%"] = round(gain[name, b] / gain["оракул", b], 3)
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print("выигр@b — суммарная Δ отданных компонент как доля Δ оракула при том же бюджете")


if __name__ == "__main__":
    main()
