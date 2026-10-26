"""E2: standard confidence scores of B0 as routing signals.

Pixel AUROC (logit resolution, sampled pixels) for two targets:
  err  - B0 is wrong
  help - B0 is wrong and B5 is right
Tile routing: mean score per tile, % of the B5-B0 gap closed (as in E1).
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from common import DOMAINS, IGNORE, RESULTS, TILE, load, route_curve

BUDGETS = [0.05, 0.1, 0.2, 0.3]
PIXELS_PER_IMAGE = 2000
STRIDE = 4  # logits are at 1/4 resolution


def uncertainty(logits):
    """Per-pixel scores, higher = less confident."""
    p = logits.softmax(0)
    top2 = p.topk(2, dim=0).values
    return {
        "entropy": -(p * p.clamp_min(1e-12).log()).sum(0),
        "msp": 1 - top2[0],
        "margin": top2[1] - top2[0],
        "energy": -logits.logsumexp(0),
    }


def main():
    base = json.loads((RESULTS / "e0.json").read_text())
    oracle = json.loads((RESULTS / "e1.json").read_text())
    rng = np.random.default_rng(0)
    auroc, closed = {}, {}

    for domain in DOMAINS:
        tiles, pix, err, helps = {}, {}, [], []
        for gt, b0, b5, logits in load(domain, "gt", "pred_b0", "pred_b5", "logits_b0"):
            scores = uncertainty(torch.from_numpy(logits).float())
            for k, s in scores.items():
                tiles.setdefault(k, []).append(F.avg_pool2d(s[None], TILE // STRIDE)[0].numpy())

            gt, b0, b5 = (a[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE] for a in (gt, b0, b5))
            valid = np.flatnonzero(gt != IGNORE)
            idx = rng.choice(valid, min(PIXELS_PER_IMAGE, valid.size), replace=False)
            for k, s in scores.items():
                pix.setdefault(k, []).append(s.numpy().ravel()[idx])
            wrong = b0.ravel()[idx] != gt.ravel()[idx]
            err.append(wrong)
            helps.append(wrong & (b5.ravel()[idx] == gt.ravel()[idx]))

        err, helps = np.concatenate(err), np.concatenate(helps)
        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        auroc[domain], closed[domain] = {}, {}
        for k in tiles:
            s = np.concatenate(pix[k])
            auroc[domain][k] = [round(roc_auc_score(err, s), 3), round(roc_auc_score(helps, s), 3)]
            curve = route_curve(domain, np.stack(tiles[k]), BUDGETS)
            closed[domain][k] = [round(100 * (m - b0_miou) / (b5_miou - b0_miou), 1) for m in curve]

    print(f"{'domain':8} {'method':8} {'AUC err':>8} {'AUC help':>9}" + "".join(f"{b:>7.0%}" for b in BUDGETS))
    for domain in DOMAINS:
        for k, (a_err, a_help) in auroc[domain].items():
            print(f"{domain:8} {k:8} {a_err:8.3f} {a_help:9.3f}" + "".join(f"{c:7.1f}" for c in closed[domain][k]))
        print(f"{domain:8} {'oracle':8} {'':8} {'':9}" + "".join(f"{c:7.1f}" for c in oracle[domain]["oracle_tile"][:len(BUDGETS)]))

    (RESULTS / "e2.json").write_text(json.dumps({"budgets": BUDGETS, "auroc": auroc, "closed": closed}, indent=2))


if __name__ == "__main__":
    main()
