"""E10: per-frame choice between several compute levels under one budget (test pool).

Levels (B5 cost relative to full-frame B5): b0 0, ens@0.5 0.25, ens@0.75 0.56, b5 1.
ens = mean of B0 and low-res B5 probabilities (E9).
  oracle_2  - true per-frame gain, only b0 / b5 (as the E7 oracle)
  oracle_4  - true per-frame gain, all levels
  entropy   - top frames by entropy get b5 (E7 baseline)
  knn_rule  - frames with knn above the cs_fit 95th percentile ("not Cityscapes") get ens@s,
              the rest of the budget upgrades frames to b5 in order of entropy
Oracles allocate with a Lagrangian: each frame takes argmax(gain - lam * cost), lam bisected
so that the mean cost fits the budget. Gain = correctly labeled pixels over B0.
"""
import json

import numpy as np
import torch

from common import DOMAINS, CACHE, RESULTS, confusion, load, miou
from e6_threshold import image_scores
from e7_stream import BOOT, TEST
from e9_lowres import SCALES, predictions

LEVELS = ["b0", "ens@0.5", "ens@0.75", "b5"]
COSTS = np.array([0, 0.25, 0.5625, 1])
BUDGETS = [0.25, 0.5]


def level_confusions(domain):
    """(images, levels, C, C)"""
    out = []
    lowres = sorted((CACHE / "e9" / domain).glob("*.npz"))
    for (gt, b0, b5, logits0), path in zip(load(domain, "gt", "pred_b0", "pred_b5", "logits_b0"), lowres):
        with np.load(path) as z:
            ens = [predictions(torch.from_numpy(logits0).float(),
                               torch.from_numpy(z[f"s{round(s * 100)}"]).float())["ens"] for s in SCALES]
        out.append([confusion(p, gt) for p in [b0, *ens, b5]])
    return np.array(out)


def allocate(gains, budget):
    lo, hi = 0.0, gains.max() / COSTS[1] + 1
    for _ in range(60):
        lam = (lo + hi) / 2
        lo, hi = (lam, hi) if COSTS[(gains - lam * COSTS).argmax(1)].mean() > budget else (lo, lam)
    return (gains - hi * COSTS).argmax(1)


def knn_rule(entropy, ood, low_level, budget):
    choice = np.where(ood, low_level, 0)
    for i in np.argsort(-entropy):
        extra = (COSTS[3] - COSTS[choice[i]]) / len(choice)
        if COSTS[choice].mean() + extra <= budget:
            choice[i] = 3
    return choice if COSTS[choice].mean() <= budget + 1e-9 else None


def main():
    confs = np.concatenate([level_confusions(d) for d in TEST])
    domain = np.concatenate([[d] * len(image_scores(d)["entropy"]) for d in TEST])
    scores = {k: np.concatenate([image_scores(d)[k] for d in TEST]) for k in ("entropy", "knn")}
    ood = scores["knn"] > np.quantile(image_scores("cs_fit")["knn"], 0.95)

    acc = np.trace(confs, axis1=2, axis2=3).astype(float)
    gains = acc - acc[:, :1]
    two_level = np.where(np.isin(np.arange(4), [0, 3]), gains, -1e9)

    def closed(choice, idx=slice(None)):
        c = confs[idx]
        pick = c[np.arange(len(c)), choice[idx]]
        m0, m5 = miou(c[:, 0].sum(0)), miou(c[:, 3].sum(0))
        return 100 * (miou(pick.sum(0)) - m0) / (m5 - m0)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(confs), len(confs)) for _ in range(BOOT)]
    report = {}
    print(f"OOD by knn: " + ", ".join(f"{d} {100 * ood[domain == d].mean():.0f}%" for d in TEST))
    print(f"{'method':16} {'budget':>6} {'cost':>5} {'closed':>7}   diff vs entropy  "
          + "".join(f"{d:>8}" for d in TEST) + "   (mean cost per domain)")
    for budget in BUDGETS:
        entropy = np.zeros(len(confs), int)
        entropy[np.argsort(-scores["entropy"])[:round(budget * len(confs))]] = 3
        policies = {"oracle_2": allocate(two_level, budget), "oracle_4": allocate(gains, budget), "entropy": entropy}
        for level in (1, 2):
            choice = knn_rule(scores["entropy"], ood, level, budget)
            if choice is not None:
                policies[f"knn_rule {LEVELS[level]}"] = choice

        for k, choice in policies.items():
            diff = np.percentile([closed(choice, i) - closed(entropy, i) for i in boots], [2.5, 97.5])
            per_domain = [COSTS[choice[domain == d]].mean() for d in TEST]
            report[f"{k}@{budget}"] = {"cost": round(COSTS[choice].mean(), 3), "closed": round(closed(choice), 1),
                                       "diff": diff.round(1).tolist(),
                                       "cost_by_domain": dict(zip(TEST, np.round(per_domain, 2).tolist()))}
            print(f"{k:16} {budget:6.0%} {COSTS[choice].mean():5.2f} {closed(choice):7.1f}   "
                  f"[{diff[0]:6.1f},{diff[1]:6.1f}] " + "".join(f"{v:8.2f}" for v in per_domain))

    (RESULTS / "e10.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
