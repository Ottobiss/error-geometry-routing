"""E1: how much of the B5-B0 gap can routing a share of tiles to B5 recover at best?

Oracles rank by true gain (pixels B5 fixes minus pixels it breaks), random is the floor.
Reported as % of the gap closed: (mIoU - B0) / (B5 - B0).
"""
import json

import numpy as np

from common import DOMAINS, RESULTS, gain, load, route_curve, tile_sum

BUDGETS = [0.05, 0.1, 0.2, 0.3, 0.5]


def main():
    base = json.loads((RESULTS / "e0.json").read_text())
    rng = np.random.default_rng(0)
    report = {}

    print(f"{'domain':8} {'method':12}" + "".join(f"{b:>7.0%}" for b in BUDGETS))
    for domain in DOMAINS:
        gains = np.stack([tile_sum(gain(*arrs)) for arrs in load(domain, "gt", "pred_b0", "pred_b5")])
        methods = {
            "random": rng.random(gains.shape),
            "oracle_img": np.broadcast_to(gains.sum((1, 2), keepdims=True), gains.shape),
            "oracle_tile": gains,
        }
        b0, b5 = base[domain]["b0"], base[domain]["b5"]
        report[domain] = {}
        for name, scores in methods.items():
            closed = [100 * (m - b0) / (b5 - b0) for m in route_curve(domain, scores, BUDGETS)]
            report[domain][name] = [round(c, 1) for c in closed]
            print(f"{domain:8} {name:12}" + "".join(f"{c:7.1f}" for c in closed))

    (RESULTS / "e1.json").write_text(json.dumps({"budgets": BUDGETS, **report}, indent=2))


if __name__ == "__main__":
    main()
