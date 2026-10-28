"""E10b: is the E10 knn_rule gain robust, and does it need geometry at all?

Rule as in E10: frames with feature above the cs_fit percentile get ens@s, the rest of the budget
upgrades frames to b5 by entropy. Feature: knn, or entropy itself (no geometry).
Grid: percentile x budget x level. Cell = closed-gap difference vs the entropy baseline (E7),
'*' when the 95% bootstrap CI excludes zero, '-' when the rule does not fit the budget.
"""
import json

import numpy as np

from common import RESULTS, miou
from e6_threshold import image_scores
from e7_stream import BOOT, TEST
from e10_levels import LEVELS, level_confusions, knn_rule

PERCENTILES = [80, 90, 95, 99]
BUDGETS = [0.35, 0.4, 0.5, 0.6, 0.7]


def main():
    confs = np.concatenate([level_confusions(d) for d in TEST])
    scores = {k: np.concatenate([image_scores(d)[k] for d in TEST]) for k in ("entropy", "knn")}
    fit = image_scores("cs_fit")

    def closed(choice, idx):
        c = confs[idx]
        m0, m5 = miou(c[:, 0].sum(0)), miou(c[:, 3].sum(0))
        return 100 * (miou(c[np.arange(len(c)), choice[idx]].sum(0)) - m0) / (m5 - m0)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(confs), len(confs)) for _ in range(BOOT)]
    everything = np.arange(len(confs))

    baseline = {}
    for b in BUDGETS:
        choice = np.zeros(len(confs), int)
        choice[np.argsort(-scores["entropy"])[:round(b * len(confs))]] = 3
        baseline[b] = (choice, np.array([closed(choice, i) for i in boots]))

    report = {}
    print(f"{'feature':8} {'level':9} {'pct':>4}" + "".join(f"{b:>10}" for b in BUDGETS))
    for feature in ("knn", "entropy"):
        for level in (1, 2):
            for p in PERCENTILES:
                flagged = scores[feature] > np.percentile(fit[feature], p)
                cells = []
                for b in BUDGETS:
                    choice = knn_rule(scores["entropy"], flagged, level, b)
                    if choice is None:
                        cells.append("-")
                        continue
                    diffs = np.array([closed(choice, i) for i in boots]) - baseline[b][1]
                    point = closed(choice, everything) - closed(baseline[b][0], everything)
                    lo, hi = np.percentile(diffs, [2.5, 97.5])
                    report[f"{feature}/{LEVELS[level]}/p{p}/{b}"] = [round(point, 1), round(lo, 1), round(hi, 1)]
                    cells.append(f"{point:+.1f}" + ("*" if lo > 0 or hi < 0 else " "))
                print(f"{feature:8} {LEVELS[level]:9} {p:4}" + "".join(f"{c:>10}" for c in cells))

    (RESULTS / "e10b.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
