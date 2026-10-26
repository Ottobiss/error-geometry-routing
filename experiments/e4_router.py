"""E4: tile-level router trained to predict tile gain, with bootstrap CIs.

Tile features: mean and p90 of entropy, maha, geo_margin, knn (from E3, cache/e3_tiles.npz).
Target: tile gain (pixels B5 fixes minus pixels it breaks), i.e. what the oracle ranks by.
Router is fit on cs_fit, everything else is test.
rank: training-free mean rank of entropy and knn within each domain.
CIs: 95% bootstrap over images at the 10% budget, also for the paired difference vs entropy.
"""
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from common import CACHE, DOMAINS, RESULTS, gain, load, miou, route_confs, tile_sum
from e2_confidence import BUDGETS

FEATS = ["entropy", "maha", "geo_margin", "knn"]
STATS = ["mean", "p90"]
BOOT = 200
AT = BUDGETS.index(0.1)


def features(tiles, domain):
    return np.stack([tiles[f"{domain}__{k}__{s}"] for k in FEATS for s in STATS], -1)


def rank(scores):
    """Rank of each tile within the domain, so differently scaled scores can be averaged."""
    return scores.ravel().argsort().argsort().reshape(scores.shape)


def gap_closed(confs, idx):
    """confs: (budgets incl. 0 and 1, images, C, C) -> % gap closed per inner budget."""
    m = [miou(c[idx].sum(0)) for c in confs]
    return np.array([100 * (x - m[0]) / (m[-1] - m[0]) for x in m[1:-1]])


def main():
    tiles = np.load(CACHE / "e3_tiles.npz")
    gains = {d: np.stack([tile_sum(gain(*a)) for a in load(d, "gt", "pred_b0", "pred_b5")]) for d in DOMAINS}

    x_fit = features(tiles, "cs_fit")
    router = HistGradientBoostingRegressor(random_state=0)
    router.fit(x_fit.reshape(-1, x_fit.shape[-1]), gains["cs_fit"].ravel())

    rng = np.random.default_rng(0)
    report = {}
    print(f"{'domain':8} {'method':8}" + "".join(f"{b:>7.0%}" for b in BUDGETS) + "   CI@10%        diff vs entropy@10%")
    for domain in DOMAINS:
        x = features(tiles, domain)
        methods = {
            "entropy": tiles[f"{domain}__entropy__mean"],
            "knn": tiles[f"{domain}__knn__mean"],
            "router": router.predict(x.reshape(-1, x.shape[-1])).reshape(x.shape[:-1]),
        }
        methods["rank"] = rank(methods["entropy"]) + rank(methods["knn"])
        confs = {k: route_confs(domain, s, [0] + BUDGETS + [1]) for k, s in methods.items()}
        n = len(x)
        boots = [rng.integers(0, n, n) for _ in range(BOOT)]
        boot10 = {k: np.array([gap_closed(c, idx)[AT] for idx in boots]) for k, c in confs.items()}

        report[domain] = {}
        for k, c in confs.items():
            closed = gap_closed(c, np.arange(n))
            ci = np.percentile(boot10[k], [2.5, 97.5])
            diff = np.percentile(boot10[k] - boot10["entropy"], [2.5, 97.5])
            report[domain][k] = {"closed": closed.round(1).tolist(), "ci10": ci.round(1).tolist(),
                                 "diff10": diff.round(1).tolist()}
            print(f"{domain:8} {k:8}" + "".join(f"{v:7.1f}" for v in closed)
                  + f"   [{ci[0]:5.1f},{ci[1]:5.1f}]   [{diff[0]:6.1f},{diff[1]:6.1f}]")

    (RESULTS / "e4.json").write_text(json.dumps({"budgets": BUDGETS, **report}, indent=2))


if __name__ == "__main__":
    main()
