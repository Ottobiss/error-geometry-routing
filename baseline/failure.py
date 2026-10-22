"""Сверка с таблицей 1 Kang et al.: p, MetaSeg m и энтропия на метке провала, P0 → P1.

    python -m baseline.failure

p (S2.1): L1-логистическая регрессия, liblinear, C=1, веса кадров 1/n_j, 37 полей.
m (S2.6): L1-логистическая регрессия, saga, C из 5 кандидатов по фолдам кадров, 34 поля,
          обучение только на компонентах с непустым интерьером.
На P0 — out-of-fold по 5 фолдам кадров; модель на всём P0 применяется к P1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.evaluation import auroc, image_macro_ce, retained_mean

NUMERIC = [
    "area", "boundary_size", "interior_size", "mean_max_probability", "mean_entropy",
    "mean_probability_margin", "mean_predicted_class_probability", "mean_entropy_boundary",
    "mean_entropy_interior", "mean_max_probability_boundary", "mean_max_probability_interior",
    "mean_probability_margin_boundary", "mean_probability_margin_interior",
    "relative_component_area", "relative_boundary_size", "relative_interior_size",
    "touches_image_boundary", "has_interior",
]
P_FIELDS = NUMERIC + [f"class_{c}" for c in range(19)]
M_FIELDS = [f"ms_{s}_{t}" for s in ("E", "D") for t in ("mean", "in", "bd", "rel", "rel_in")] + \
    ["ms_S", "ms_S_in", "ms_S_bd", "ms_S_rel", "ms_S_rel_in"] + [f"ms_prob_mean_{c}" for c in range(19)]
M_CS = [0.01, 0.1, 1.0, 10.0, 100.0]  # ДОПУЩЕНИЕ: статья не называет пять кандидатов C
FOLDS = 5


def image_weights(df):
    return 1.0 / df.groupby("image_id").image_id.transform("size").values


def make_p():
    # ДОПУЩЕНИЕ: стандартизация признаков (статья не уточняет; без неё L1 зависит от масштаба).
    return make_pipeline(StandardScaler(),
                         LogisticRegression(l1_ratio=1.0, solver="liblinear", C=1.0))


def make_m(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=5000))


def fit_p(train):
    model = make_p()
    model.fit(train[P_FIELDS], train.y_fail, logisticregression__sample_weight=image_weights(train))
    return model


def fit_m(train):
    train = train[train.has_interior == 1]
    best, best_ll = None, np.inf
    for C in M_CS:                     # выбор C по внутренним фолдам кадров
        ll = 0.0
        for tr, va in GroupKFold(FOLDS).split(train, groups=train.image_id):
            mod = make_m(C).fit(train.iloc[tr][M_FIELDS], train.y_fail.iloc[tr])
            q = np.clip(mod.predict_proba(train.iloc[va][M_FIELDS])[:, 1], 1e-6, 1 - 1e-6)
            y = train.y_fail.iloc[va].values
            ll -= (y * np.log(q) + (1 - y) * np.log(1 - q)).sum()
        if ll < best_ll:
            best, best_ll = C, ll
    return make_m(best).fit(train[M_FIELDS], train.y_fail), best


def out_of_fold(p0, fit, fields):
    pred = np.zeros(len(p0))
    for tr, va in GroupKFold(FOLDS).split(p0, groups=p0.image_id):
        model = fit(p0.iloc[tr])
        model = model[0] if isinstance(model, tuple) else model
        pred[va] = model.predict_proba(p0.iloc[va][fields])[:, 1]
    return pred


def report(name, df, scores):
    y = df.y_fail.values
    row = {"pop": name, "images": df.image_id.nunique(), "N": len(df)}
    for key, s in scores.items():
        row[f"AUROC {key}"] = auroc(y, s)
    for key in ("p", "m"):
        row[f"CE {key}"] = image_macro_ce(df, y, scores[key])
    for key, s in scores.items():
        row[f"fail90 {key}"] = 100 * retained_mean(df, s, y)
    return row


def main():
    df = pd.read_parquet("results/components_cityscapes_val.parquet")
    df = df[df.has_valid].reset_index(drop=True)
    split = {i: name for name in ("p0", "p1")
             for i in Path(f"splits/cityscapes_{name}.txt").read_text().split()}
    df["pop"] = df.image_id.map(split)
    p0 = df[df["pop"] == "p0"].reset_index(drop=True)
    p1 = df[df["pop"] == "p1"].reset_index(drop=True)

    p_model = fit_p(p0)
    m_model, m_C = fit_m(p0)
    rows = [
        report("P0 (OOF)", p0, {"entropy": p0.mean_entropy.values,
                                "p": out_of_fold(p0, fit_p, P_FIELDS),
                                "m": out_of_fold(p0, fit_m, M_FIELDS)}),
        report("P1", p1, {"entropy": p1.mean_entropy.values,
                          "p": p_model.predict_proba(p1[P_FIELDS])[:, 1],
                          "m": m_model.predict_proba(p1[M_FIELDS])[:, 1]}),
    ]
    pd.set_option("display.width", 200)
    print(f"m: выбранный C = {m_C}")
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
