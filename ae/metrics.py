"""Patch metrics of v2 (PLANv2.md §9) on top of sae.classify (recall_at, auroc, entropy_bins).

Changes from v1:
  auroc_y50_robust                 mean y50 AUROC over the frozen entropy deciles ROBUST_DECILES (P0 deciles
                                   with ≥ 200 y50 positives on P0); replaces auroc_y50_entropy_deciles
  auroc_y50_deciles_posweighted    per-decile y50 AUROC over all deciles, weighted by the decile's y50
                                   positives in the population (descriptive)
  auroc_y_any_deciles              mean y_any AUROC over all deciles (descriptive), and _posweighted
  wrong_pixels_below_p0_entropy_median   share of wrong pixels in patches below the P0 entropy median
The lowest-entropy-quintile AUROC is no longer reported.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from sae.classify import BUDGETS, auroc, recall_at

ROBUST_DECILES = (4, 5, 6, 7, 8, 9)      # 0-based P0 entropy deciles with ≥ MIN_POSITIVES y50 positives
MIN_POSITIVES = 200


def deciles(ent, edges):
    """Entropy decile (0–9) of every patch; edges = the 9 inner P0 deciles (sae.classify.entropy_bins)."""
    return np.searchsorted(edges, ent, side="right")


def robust_deciles(p0, edges):
    """Deciles with ≥ MIN_POSITIVES y50 positives among the P0 patches p0, and the counts of all deciles."""
    pos = np.bincount(deciles(p0.ent_mean.to_numpy(), edges), weights=(p0.err_frac.to_numpy() >= 0.5).astype(float), minlength=10)
    return tuple(int(j) for j in np.flatnonzero(pos >= MIN_POSITIVES)), pos.astype(int).tolist()


def _decile_means(score, y, dec):
    aucs = np.array([auroc(score[dec == j], y[dec == j]) for j in range(10)])
    pos = np.bincount(dec, weights=y.astype(float), minlength=10)
    ok = np.isfinite(aucs)
    return aucs, (aucs[ok] * pos[ok]).sum() / pos[ok].sum() if pos[ok].sum() > 0 else np.nan


def metrics(df, edges):
    """Patch metrics of df[score] pooled over frames (PLANv2.md §9)."""
    s, e = df.score.to_numpy(), df.n_err.to_numpy()
    y50, yany = df.err_frac.to_numpy() >= 0.5, df.err_frac.to_numpy() >= 0.1
    ent = df.ent_mean.to_numpy()
    order = np.argsort(-s, kind="stable")
    out = {"n_frames": int(df.image_id.nunique()), "n_patches": len(df), "wrong_pixels": int(e.sum()),
           "share_y50": float(y50.mean()), "share_y_any": float(yany.mean())}
    for b in BUDGETS:
        out[f"recall@{round(100 * b)}%"] = recall_at(order, e, b)
    for name, y in (("y50", y50), ("y_any", yany)):
        out[f"auroc_{name}"] = auroc(s, y)
        out[f"ap_{name}"] = average_precision_score(y, s) if y.any() else np.nan
    dec = deciles(ent, edges)
    a50, out["auroc_y50_deciles_posweighted"] = _decile_means(s, y50, dec)
    robust = a50[list(ROBUST_DECILES)]
    out["auroc_y50_robust"] = float(np.nanmean(robust)) if np.isfinite(robust).any() else np.nan
    out["robust_deciles_used"] = int(np.isfinite(robust).sum())
    aany, out["auroc_y_any_deciles_posweighted"] = _decile_means(s, yany, dec)
    out["auroc_y_any_deciles"] = float(np.nanmean(aany)) if np.isfinite(aany).any() else np.nan
    out["wrong_pixels_below_p0_entropy_median"] = float(e[ent < edges[4]].sum() / e.sum())
    return out


def population_metrics(scores, edges):
    """Metrics of the run score and of raw patch mean entropy for every population and each dataset's union."""
    res = {}
    groups = [(f"{d}:{p}", (scores.dataset == d) & (scores.population == p))
              for d, p in scores[["dataset", "population"]].drop_duplicates().itertuples(index=False)]
    for d in scores.dataset.unique():
        if scores[scores.dataset == d].population.nunique() > 1:
            groups.append((f"{d}:all", scores.dataset == d))
    for name, m in groups:
        df = scores[m]
        res[name] = {"model": metrics(df, edges), "entropy": metrics(df.assign(score=df.ent_mean), edges)}
    return res


def paired_bootstrap(a, b, metric, edges, rng_counts):
    """Point difference a − b of the metric (recall@<x>% or auroc_y50_robust), 95% percentile interval and
    two-sided bootstrap p over frame resamples rng_counts (multiplicities per frame, frames in order of first
    appearance in a)."""
    frame_idx = pd.factorize(a.image_id)[0]
    e, y50 = a.n_err.to_numpy(), a.err_frac.to_numpy() >= 0.5
    if metric.startswith("recall@"):
        budget = float(metric[len("recall@"):-1]) / 100
        orders = [np.argsort(-x.score.to_numpy(), kind="stable") for x in (a, b)]
        stat = lambda k, w: recall_at(orders[k], e, budget, w)
    elif metric == "auroc_y50_robust":
        dec = deciles(a.ent_mean.to_numpy(), edges)
        groups = [np.flatnonzero(dec == j) for j in ROBUST_DECILES]
        invs = [[np.unique(x.score.to_numpy()[g], return_inverse=True)[1] for g in groups] for x in (a, b)]

        def stat(k, w):
            w = np.ones(len(e)) if w is None else w
            return np.nanmean([auroc(None, y50[g], w[g], invs[k][j]) for j, g in enumerate(groups)])
    else:
        raise ValueError(metric)
    point = stat(0, None) - stat(1, None)
    diffs = np.array([stat(0, c[frame_idx].astype(float)) - stat(1, c[frame_idx].astype(float)) for c in rng_counts])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return point, lo, hi, p


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adj = np.empty_like(p)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        adj[j] = running
    return adj
