"""E6: whole-frame routing with a threshold fixed on cs_fit - does the budget hold under shift?

Image score = mean over tiles of:
  entropy, knn
  rank - ECDF(entropy) + ECDF(knn), ECDFs taken on cs_fit tiles (no test-domain statistics)
Threshold: the cs_fit quantile that routes TARGET share of cs_fit images to B5.
For each domain: share of images actually routed and % of the B5-B0 gap closed.
"""
import json

import numpy as np

from common import CACHE, DOMAINS, NUM_CLASSES, RESULTS, confusion, load, miou

TARGETS = [0.25, 0.5]


def image_confusions(domain):
    c0, c5 = [], []
    for gt, b0, b5 in load(domain, "gt", "pred_b0", "pred_b5"):
        c0.append(confusion(b0, gt))
        c5.append(confusion(b5, gt))
    return np.stack(c0), np.stack(c5)


def image_scores(domain):
    """Per-image entropy, knn and cs_fit-ECDF rank (see module docstring)."""
    feats = np.load(CACHE / "e3_tiles.npz")
    tiles = lambda d, k: feats[f"{d}__{k}__mean"]
    fit_rank = lambda k: np.searchsorted(np.sort(tiles("cs_fit", k).ravel()), tiles(domain, k)) / tiles("cs_fit", k).size
    return {"entropy": tiles(domain, "entropy").mean((1, 2)), "knn": tiles(domain, "knn").mean((1, 2)),
            "rank": (fit_rank("entropy") + fit_rank("knn")).mean((1, 2))}


def main():
    fit = image_scores("cs_fit")
    thresholds = {(k, t): np.quantile(s, 1 - t) for k, s in fit.items() for t in TARGETS}

    report = {}
    print(f"{'domain':8} {'method':8} {'target':>6} {'routed':>7} {'closed':>7}")
    for domain in DOMAINS:
        c0, c5 = image_confusions(domain)
        m0, m5 = miou(c0.sum(0)), miou(c5.sum(0))
        closed = lambda sel: 100 * (miou(np.where(sel[:, None, None], c5, c0).sum(0)) - m0) / (m5 - m0)
        report[domain] = {}
        for k, s in image_scores(domain).items():
            for t in TARGETS:
                sel = s > thresholds[(k, t)]
                row = [round(100 * sel.mean(), 1), round(closed(sel), 1)]
                report[domain][f"{k}@{t}"] = dict(zip(["routed", "closed"], row))
                print(f"{domain:8} {k:8} {t:6.0%}" + "".join(f"{v:7.1f}" for v in row))

    (RESULTS / "e6.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
