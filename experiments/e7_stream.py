"""E7: whole-frame routing on a mixed stream (cs_test + ACDC) under one shared budget.

Image scores as in E6 (entropy, knn, cs_fit-ECDF rank); oracle = per-image gain.
1. pool:   top `budget` share of the whole pool goes to B5 (offline allocation across domains).
           95% bootstrap CI over images for the difference vs entropy.
2. online: frames arrive domain by domain (cs_test, fog, night, rain, snow, sequence order);
           a frame goes to B5 if its score exceeds the (1 - budget) quantile of the previous
           WINDOW scores. The window starts filled with cs_fit scores.
Metrics on the pool: % of the B5-B0 gap closed and share of each domain's frames routed.
"""
import json

import numpy as np

from common import DOMAINS, IGNORE, RESULTS, confusion, gain, load, miou
from e6_threshold import image_scores

TEST = [d for d in DOMAINS if d != "cs_fit"]
BUDGETS = [0.25, 0.5]
WINDOW = 50
BOOT = 200


def per_image(domain):
    """Per-image B0 and B5 confusions, gain and number of labeled pixels."""
    c0, c5, g, n = [], [], [], []
    for gt, b0, b5 in load(domain, "gt", "pred_b0", "pred_b5"):
        c0.append(confusion(b0, gt))
        c5.append(confusion(b5, gt))
        g.append(gain(gt, b0, b5).sum())
        n.append((gt != IGNORE).sum())
    return np.stack(c0), np.stack(c5), np.array(g), np.array(n)


def online(scores, budget, warmup):
    history, routed = list(warmup[-WINDOW:]), []
    for s in scores:
        routed.append(s > np.quantile(history[-WINDOW:], 1 - budget))
        history.append(s)
    return np.array(routed)


def main():
    parts = [per_image(d) for d in TEST]
    c0, c5 = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    domain = np.concatenate([[d] * len(p[2]) for d, p in zip(TEST, parts)])
    scores = {k: np.concatenate([image_scores(d)[k] for d in TEST]) for k in ("entropy", "knn", "rank")}
    scores["oracle"] = np.concatenate([p[2] for p in parts])
    fit = image_scores("cs_fit")

    def closed(sel, idx=slice(None)):
        m0, m5 = miou(c0[idx].sum(0)), miou(c5[idx].sum(0))
        m = miou(np.where(sel[idx, None, None], c5[idx], c0[idx]).sum(0))
        return 100 * (m - m0) / (m5 - m0)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(c0), len(c0)) for _ in range(BOOT)]
    report = {}

    print(f"pool: B0 {miou(c0.sum(0)):.2f}, B5 {miou(c5.sum(0)):.2f}, {len(c0)} frames")
    print(f"{'mode':7} {'method':8} {'budget':>6} {'routed':>7} {'closed':>7}   diff vs entropy   "
          + "".join(f"{d:>8}" for d in TEST))
    for budget in BUDGETS:
        selections = {}
        for k, s in scores.items():
            top = np.zeros(len(s), bool)
            top[np.argsort(-s)[:round(budget * len(s))]] = True
            selections[("pool", k)] = top
        for k in ("entropy", "knn", "rank"):
            selections[("online", k)] = online(scores[k], budget, fit[k])

        for (mode, k), sel in selections.items():
            base_sel = selections[(mode, "entropy")]
            diff = np.percentile([closed(sel, i) - closed(base_sel, i) for i in boots], [2.5, 97.5])
            shares = [100 * sel[domain == d].mean() for d in TEST]
            report[f"{mode}/{k}@{budget}"] = {"routed": round(100 * sel.mean(), 1), "closed": round(closed(sel), 1),
                                              "diff": diff.round(1).tolist(),
                                              "by_domain": dict(zip(TEST, np.round(shares, 1).tolist()))}
            print(f"{mode:7} {k:8} {budget:6.0%} {100 * sel.mean():7.1f} {closed(sel):7.1f}"
                  f"   [{diff[0]:6.1f},{diff[1]:6.1f}]  " + "".join(f"{v:8.1f}" for v in shares))

    (RESULTS / "e7.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
