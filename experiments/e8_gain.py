"""E8: predict per-frame B5 gain instead of uncertainty, tested on unseen conditions.

Frame features: mean over tiles of the tile mean and tile p90 of entropy, knn, maha (from E3).
Target: frame gain / labeled pixels. Model: standardized ridge regression.
Leave-one-condition-out: for each ACDC condition the model is fit on cs_fit + the other three;
cs_test is predicted by the model fit on cs_fit + all four conditions.
  ridge_ent - entropy features only
  ridge_all - entropy + geometric features
Evaluated as E7 "pool": top `budget` share of the mixed stream goes to B5, 95% bootstrap CIs.
"""
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, RESULTS, miou
from e7_stream import BOOT, BUDGETS, TEST, per_image

ACDC = ["fog", "night", "rain", "snow"]
FEATS = {"ridge_ent": ["entropy"], "ridge_all": ["entropy", "knn", "maha"]}


def frame_features(domain, names):
    tiles = np.load(CACHE / "e3_tiles.npz")
    return np.stack([tiles[f"{domain}__{k}__{s}"].mean((1, 2)) for k in names for s in ("mean", "p90")], 1)


def fit(domains, names, target):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return model.fit(np.concatenate([frame_features(d, names) for d in domains]),
                     np.concatenate([target[d] for d in domains]))


def main():
    data = {d: per_image(d) for d in ["cs_fit"] + TEST}
    target = {d: g / n for d, (_, _, g, n) in data.items()}
    c0 = np.concatenate([data[d][0] for d in TEST])
    c5 = np.concatenate([data[d][1] for d in TEST])
    domain = np.concatenate([[d] * len(data[d][2]) for d in TEST])

    scores = {"entropy": np.concatenate([frame_features(d, ["entropy"])[:, 0] for d in TEST])}
    for method, names in FEATS.items():
        preds = []
        for d in TEST:
            train = ["cs_fit"] + [c for c in ACDC if c != d]
            preds.append(fit(train, names, target).predict(frame_features(d, names)))
        scores[method] = np.concatenate(preds)
    scores["oracle"] = np.concatenate([data[d][2] for d in TEST])

    full = fit(["cs_fit"] + ACDC, FEATS["ridge_all"], target)
    cols = [f"{k}_{s}" for k in FEATS["ridge_all"] for s in ("mean", "p90")]
    print("ridge_all coefs (standardized, fit on cs_fit + all ACDC):",
          dict(zip(cols, full[-1].coef_.round(4))))

    def closed(sel, idx=slice(None)):
        m0, m5 = miou(c0[idx].sum(0)), miou(c5[idx].sum(0))
        return 100 * (miou(np.where(sel[idx, None, None], c5[idx], c0[idx]).sum(0)) - m0) / (m5 - m0)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(c0), len(c0)) for _ in range(BOOT)]
    ci = lambda a, b: np.percentile([closed(a, i) - closed(b, i) for i in boots], [2.5, 97.5])

    report = {}
    print(f"{'method':10} {'budget':>6} {'closed':>7}   diff vs entropy   diff vs ridge_ent "
          + "".join(f"{d:>8}" for d in TEST))
    for budget in BUDGETS:
        sel = {}
        for k, s in scores.items():
            sel[k] = np.zeros(len(s), bool)
            sel[k][np.argsort(-s)[:round(budget * len(s))]] = True
        for k, s in sel.items():
            d_ent, d_ridge = ci(s, sel["entropy"]), ci(s, sel["ridge_ent"])
            shares = [100 * s[domain == d].mean() for d in TEST]
            report[f"{k}@{budget}"] = {"closed": round(closed(s), 1), "diff_entropy": d_ent.round(1).tolist(),
                                       "diff_ridge_ent": d_ridge.round(1).tolist(),
                                       "by_domain": dict(zip(TEST, np.round(shares, 1).tolist()))}
            print(f"{k:10} {budget:6.0%} {closed(s):7.1f}   [{d_ent[0]:6.1f},{d_ent[1]:6.1f}]  "
                  f"[{d_ridge[0]:6.1f},{d_ridge[1]:6.1f}]  " + "".join(f"{v:8.1f}" for v in shares))

    (RESULTS / "e8.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
