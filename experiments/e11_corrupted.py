"""E11: confirm the E10 knn rule on synthetic shift (cs_test corrupted Cityscapes-C style).

Domains: CORRUPTIONS = fog / dark / blur / noise x severity 1-2 over the 233 cs_test scenes.
Caches are built with E0 and E9 code; knn uses the E3 bank fitted on cs_fit.
Pre-registered test (fixed before the run): knn above the cs_fit 95th percentile -> ens@0.75,
the rest of the budget upgrades frames to b5 by entropy, budget 0.5, vs entropy routing.
The bootstrap resamples scenes (each scene appears in all corruptions).
Control: the same share of frames flagged at random instead of by knn.
Supplementary: the E10b grid for knn + ens@0.75.
"""
import json

import numpy as np

import e0_cache
from common import CACHE, CORRUPTIONS, RESULTS, load, miou
from e10_levels import COSTS, knn_rule, level_confusions
from e3_geometry import fit_geometry, score_maps
from e6_threshold import image_scores
from e7_stream import BOOT
from e9_lowres import cache_lowres

PERCENTILES = [80, 90, 95, 99]
BUDGETS = [0.35, 0.4, 0.5, 0.6, 0.7]
LEVEL = 2  # ens@0.75


def build_caches():
    missing = [d for d in CORRUPTIONS if not (CACHE / d).exists()]
    if missing:
        models = e0_cache.load_models()
        for d in missing:
            print(d, e0_cache.cache_domain(d, *models))
        del models
    missing = [d for d in CORRUPTIONS if not (CACHE / "e9" / d).exists()]
    if missing:
        cache_lowres(missing)


def frame_scores(domain, geo):
    maps = [score_maps(feat, logits, geo) for logits, feat in load(domain, "logits_b0", "feat_b0")]
    return np.array([m["entropy"].mean() for m in maps]), np.array([m["knn"].mean() for m in maps])


def main():
    geo = fit_geometry()  # first draw from the seeded RNGs, same bank as in E3
    build_caches()

    path = CACHE / "e11.npz"  # level confusions take ~15 min, keep them
    if not path.exists():
        confs = np.concatenate([level_confusions(d) for d in CORRUPTIONS])
        entropy, knn = map(np.concatenate, zip(*[frame_scores(d, geo) for d in CORRUPTIONS]))
        np.savez(path, confs=confs, entropy=entropy, knn=knn)
    with np.load(path) as z:
        confs, entropy, knn = z["confs"], z["entropy"], z["knn"]
    n_scenes = len(confs) // len(CORRUPTIONS)
    domain = np.repeat(CORRUPTIONS, n_scenes)
    fit_knn = image_scores("cs_fit")["knn"]

    def closed(choice, idx=slice(None)):
        c = confs[idx]
        m0, m5 = miou(c[:, 0].sum(0)), miou(c[:, 3].sum(0))
        return 100 * (miou(c[np.arange(len(c)), choice[idx]].sum(0)) - m0) / (m5 - m0)

    def entropy_routing(budget):
        choice = np.zeros(len(confs), int)
        choice[np.argsort(-entropy)[:round(budget * len(confs))]] = 3
        return choice

    rng = np.random.default_rng(0)
    offsets = n_scenes * np.arange(len(CORRUPTIONS))[:, None]
    boots = [(rng.integers(0, n_scenes, n_scenes)[None] + offsets).ravel() for _ in range(BOOT)]

    def diff(choice, base):
        lo, hi = np.percentile([closed(choice, i) - closed(base, i) for i in boots], [2.5, 97.5])
        return closed(choice) - closed(base), lo, hi

    print(f"{'domain':9} {'b0':>6} {'b5':>6} {'flagged':>8}")
    for d in CORRUPTIONS:
        c = confs[domain == d]
        print(f"{d:9} {miou(c[:, 0].sum(0)):6.2f} {miou(c[:, 3].sum(0)):6.2f} "
              f"{100 * (knn[domain == d] > np.percentile(fit_knn, 95)).mean():7.0f}%")

    base = entropy_routing(0.5)
    rule = knn_rule(entropy, knn > np.percentile(fit_knn, 95), LEVEL, 0.5)
    point, lo, hi = diff(rule, base)
    report = {"primary": {"entropy": round(closed(base), 1), "knn_rule": round(closed(rule), 1),
                          "cost": round(COSTS[rule].mean(), 3), "diff": [round(point, 1), round(lo, 1), round(hi, 1)]}}
    print(f"\nprimary (knn p95, ens@0.75, budget 0.5): entropy {closed(base):.1f}, knn_rule {closed(rule):.1f}, "
          f"diff {point:+.1f} [{lo:+.1f}, {hi:+.1f}]")

    # control: the same number of frames flagged at random, a fresh draw per bootstrap replicate
    rand_rng = np.random.default_rng(1)
    n_flagged = (knn > np.percentile(fit_knn, 95)).sum()
    randoms = [knn_rule(entropy, np.isin(np.arange(len(confs)), rand_rng.permutation(len(confs))[:n_flagged]),
                        LEVEL, 0.5) for _ in boots]
    rand_closed = np.mean([closed(r) for r in randoms])
    lo, hi = np.percentile([closed(rule, i) - closed(r, i) for r, i in zip(randoms, boots)], [2.5, 97.5])
    report["random_control"] = {"random_rule": round(rand_closed, 1),
                                "knn_minus_random": [round(closed(rule) - rand_closed, 1), round(lo, 1), round(hi, 1)]}
    print(f"control (random flags, same share): random_rule {rand_closed:.1f}, "
          f"knn - random {closed(rule) - rand_closed:+.1f} [{lo:+.1f}, {hi:+.1f}]")

    print(f"\n{'pct':>4}" + "".join(f"{b:>10}" for b in BUDGETS))
    for p in PERCENTILES:
        cells = []
        for b in BUDGETS:
            rule = knn_rule(entropy, knn > np.percentile(fit_knn, p), LEVEL, b)
            if rule is None:
                cells.append("-")
                continue
            point, lo, hi = diff(rule, entropy_routing(b))
            report[f"p{p}/{b}"] = [round(point, 1), round(lo, 1), round(hi, 1)]
            cells.append(f"{point:+.1f}" + ("*" if lo > 0 or hi < 0 else " "))
        print(f"{p:4}" + "".join(f"{c:>10}" for c in cells))

    (RESULTS / "e11.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
