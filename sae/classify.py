"""Steps 4–5: CatBoost on the patch table, GroupKFold on P0, frozen tests on P1 and ACDC (PLAN.md §6, §7).

    uv run python -m sae.classify cv --features F1 --sae D2k8       # P0: 5 folds by frame, out-of-fold scores
    uv run python -m sae.classify test --features F1 --sae D2k8     # final fit on P0, scores on P1 and ACDC
    uv run python -m sae.classify test --features F1 --sae D2k8 --tests cityscapes:p1     # P1 only
    uv run python -m sae.classify compare --sae D2k8                # pre-registered comparisons (paired bootstrap)

Feature sets (the CatBoost setup is the same for all of them):
  F0    output features of B0 (control): entropy mean / max / p90, MSP mean / min, margin mean / min,
        majority predicted class (native categorical), boundary fraction, patch row / column
  F1    F0 + SAE codes of the 8 Mix-FFN hidden layers
  F2    SAE codes only
  F3    raw Mix-FFN hidden activations, same selection and pooling (4,096 columns)
  F3F0  F3 + F0, the control for F1
SAE codes go in as sparse pandas columns: CatBoost rejects a categorical column in a float scipy matrix, but
accepts sparse columns next to it in a DataFrame. Latents that fire in < 0.1% of P0 patches are dropped.

Objective: CrossEntropy on err_frac, weight n_valid. Training patches: all with err_frac > 0 and a random 25%
of clean ones (seed 0) with weight ×4; the P0 pool is quantised once (label-free borders) and sliced per fold.
Early stopping on the held-out fold (its training patches); out-of-fold scores cover all its patches. The
final model is fitted on all P0 training patches with the mean best iteration count of the folds.

Output: results/sae/runs/<F>[_<sae>]/
  cv_config.json, cv_scores.parquet, cv_metrics.json, latents.npy     (cv)
  config.json, scores.parquet, metrics.json, pixel_auroc.csv, model.cbm (test)
results/sae/compare_<sae>.csv (compare): paired bootstrap over frames (2,000, seed 0, the same resamples for
every comparison on a test), difference run − control, 95% interval, two-sided bootstrap p; no multiplicity
correction, so an interval that excludes 0 with p ≥ 0.01 is marked borderline.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from catboost import CatBoost, Pool
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from core.data import DATASETS, FIT_DATASET, FIT_POP, population, populations
from core.paths import write_json
from sae import ROOT
from sae.hooks import PATCH
from sae.patch_features import F0_NUM, base_path, codes_path

FEATURE_SETS = {"F0": ("f0",), "F1": ("f0", "codes"), "F2": ("codes",), "F3": ("raw",), "F3F0": ("raw", "f0")}
PARAMS = dict(loss_function="CrossEntropy", depth=6, learning_rate=0.05, l2_leaf_reg=3, iterations=3000,
              od_type="Iter", od_wait=200, use_best_model=True, random_seed=0, thread_count=-1,
              allow_writing_files=False, verbose=200)
FOLDS, SEED = 5, 0
CLEAN_KEEP = 0.25
MIN_FIRE = 0.001
BUDGETS = (0.05, 0.10, 0.20, 0.30)
TESTS = ("cityscapes:p1", "acdc:all")
KEYS = ["image_id", "patch", "n_valid", "n_err", "err_frac", "ent_mean"]
# Pre-registered main comparisons (PLAN.md §7): test, metric, run, control.
COMPARISONS = [("cityscapes:p1", "recall@10%", "F1", "F0"),
               ("acdc:all", "recall@10%", "F1", "F0"),
               ("cityscapes:p1", "recall@10%", "F2", "F3"),
               ("cityscapes:p1", "auroc_y50_entropy_deciles", "F1", "F0")]


def run_name(features, sae):
    return f"{features}_{sae}" if "codes" in FEATURE_SETS[features] else features


def run_dir(features, sae):
    return ROOT / "runs" / run_name(features, sae)


# --- Data ---

def load_meta(dataset, ids):
    """Patches with valid pixels of frames ids, in frame order: keys, labels and F0."""
    parts = []
    for image_id in ids:
        z = np.load(base_path(dataset, image_id))
        keep = z["n_valid"] > 0
        cols = {"image_id": image_id, "patch": np.flatnonzero(keep).astype(np.int32)}
        for c in ["n_valid", "n_err", "cls", *F0_NUM]:
            cols[c] = z[c][keep]
        parts.append(pd.DataFrame(cols))
    meta = pd.concat(parts, ignore_index=True)
    meta["err_frac"] = meta.n_err / meta.n_valid
    meta["cls"] = meta.cls.astype(np.int32)
    return meta


def frame_runs(meta):
    """(image_id, row slice) of each frame; rows of one frame must be contiguous."""
    ids = meta.image_id.to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    assert len(set(ids[starts])) == len(starts), "rows of a frame are not contiguous"
    return [(ids[s], slice(s, e)) for s, e in zip(starts, ends)]


def load_parts(dataset, meta, sae, parts):
    """SAE codes (CSR float32) and raw activations (float32) of the rows of meta, or None."""
    codes, raw = [], None
    patch = meta.patch.to_numpy()
    if "raw" in parts:
        d = sum(json.loads((base_path(dataset, "_info").with_suffix(".json")).read_text())["d"])
        raw = np.empty((len(meta), d), dtype=np.float32)
    for image_id, rows in frame_runs(meta):
        if "codes" in parts:
            z = np.load(codes_path(sae, dataset, image_id))
            c = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            codes.append(c[patch[rows]])
        if "raw" in parts:
            raw[rows] = np.load(base_path(dataset, image_id))["raw"][patch[rows]]
    return (sp.vstack(codes, format="csr") if codes else None), raw


def column_names(dataset, sae, kind):
    """Names <block>_z<latent> of the code columns or <block>_h<unit> of the raw columns."""
    if kind == "codes":
        info = json.loads((codes_path(sae, dataset, "_info").with_suffix(".json")).read_text())
        dims, tag = info["D"], "z"
    else:
        info = json.loads((base_path(dataset, "_info").with_suffix(".json")).read_text())
        dims, tag = info["d"], "h"
    return np.array([f"{b}_{tag}{j}" for b, n in zip(info["blocks"], dims) for j in range(n)])


def sparse_frame(m, columns):
    """CSR → DataFrame of sparse columns with fill value 0 (pandas 3 DataFrame.sparse.from_spmatrix fills with
    NaN, which CatBoost would read as missing)."""
    m = m.tocsc()
    df = pd.DataFrame({c: pd.arrays.SparseArray.from_spmatrix(m[:, [j]]) for j, c in enumerate(columns)})
    assert all(dt.fill_value == 0 for dt in df.dtypes)
    return df


def design(dataset, meta, sae, parts, latents=None):
    """Feature DataFrame of the rows of meta and the names of its categorical columns."""
    codes, raw = load_parts(dataset, meta, sae, parts)
    frames = []
    if "raw" in parts:
        frames.append(pd.DataFrame(raw, columns=column_names(dataset, sae, "raw"), copy=False))
    if "f0" in parts:
        frames.append(meta[F0_NUM + ["cls"]].reset_index(drop=True))
    if "codes" in parts:
        frames.append(sparse_frame(codes[:, latents], column_names(dataset, sae, "codes")[latents]))
    X = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    return X, (["cls"] if "f0" in parts else [])


def training_rows(meta, seed=SEED):
    """P0 training patches: all with err_frac > 0 and a random 25% of clean ones. → (mask, weight)."""
    clean = np.flatnonzero(meta.n_err.to_numpy() == 0)
    keep = meta.n_err.to_numpy() > 0
    keep[np.random.default_rng(seed).choice(clean, int(round(CLEAN_KEEP * len(clean))), replace=False)] = True
    weight = meta.n_valid.to_numpy() * np.where(meta.n_err.to_numpy() == 0, 1 / CLEAN_KEEP, 1.0)
    return keep, weight


def firing_latents(meta, sae):
    """Code columns that fire in at least MIN_FIRE of the patches of meta (P0, label-free)."""
    counts, n_cols = None, None
    for image_id, rows in frame_runs(meta):
        z = np.load(codes_path(sae, FIT_DATASET, image_id))
        c = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))[meta.patch.to_numpy()[rows]]
        n_cols = c.shape[1]
        counts = np.bincount(c.indices[c.data > 0], minlength=n_cols) + (0 if counts is None else counts)
    return np.flatnonzero(counts / len(meta) >= MIN_FIRE), n_cols


def p0_meta():
    return load_meta(FIT_DATASET, population(FIT_DATASET, FIT_POP))


def p0_pool(meta, features, sae, latents):
    """Training mask of the P0 patches meta and the quantised CatBoost pool of the training patches."""
    keep, weight = training_rows(meta)
    t0 = time.time()
    X, cat = design(FIT_DATASET, meta[keep], sae, FEATURE_SETS[features], latents)
    pool = Pool(X, label=meta.err_frac.to_numpy()[keep], weight=weight[keep], cat_features=cat)
    del X
    pool.quantize()
    print(f"P0 pool: {pool.num_row()} × {pool.num_col()} ({keep.sum()} of {len(meta)} patches), "
          f"{time.time() - t0:.0f} s")
    return keep, pool


def predict(model, dataset, meta, sae, features, latents, chunk=40):
    """Scores of all rows of meta, frames in chunks (memory)."""
    runs = frame_runs(meta)
    out = []
    for i in range(0, len(runs), chunk):
        m = meta.iloc[runs[i][1].start:runs[min(i + chunk, len(runs)) - 1][1].stop]
        X, cat = design(dataset, m, sae, FEATURE_SETS[features], latents)
        out.append(model.predict(Pool(X, cat_features=cat), prediction_type="Probability")[:, 1])
    return np.concatenate(out)


# --- Metrics ---

def recall_at(order, n_err, budget, w=None):
    """Share of wrong valid pixels inside the top budget share of patches (global budget); order — patches by
    descending score; w — patch multiplicities (bootstrap)."""
    w = np.ones(len(order)) if w is None else w[order]
    e = n_err[order]
    cum = np.cumsum(w)
    take = np.clip(budget * cum[-1] - (cum - w), 0, w)  # whole patches, the last one partly
    return (take * e).sum() / (w * e).sum()


def auroc(score, y, w=None, inv=None):
    """AUROC with ties counted ½ and optional sample weights; inv — np.unique(score, return_inverse)[1]."""
    if inv is None:
        inv = np.unique(score, return_inverse=True)[1]
    w = np.ones(len(y)) if w is None else w
    y = np.asarray(y, dtype=float)
    pos, neg = np.bincount(inv, weights=w * y), np.bincount(inv, weights=w * (1 - y))
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return np.nan
    return (pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * N)


def entropy_bins(p0_meta):
    """Edges of P0 patch mean entropy deciles and the lowest quintile cut."""
    ent = p0_meta.ent_mean.to_numpy()
    return np.quantile(ent, np.arange(1, 10) / 10), float(np.quantile(ent, 0.2))


def decile_auroc(score, y50, ent, edges, w=None):
    dec = np.searchsorted(edges, ent, side="right")
    aucs = [auroc(score[dec == j], y50[dec == j], None if w is None else w[dec == j]) for j in range(10)]
    return np.nanmean(aucs), int(np.isfinite(aucs).sum())


def metrics(df, edges, q20):
    """Patch metrics of df[score] pooled over frames (PLAN.md §7)."""
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
        out[f"ap_{name}"] = average_precision_score(y, s)
    out["auroc_y50_entropy_deciles"], out["entropy_deciles_used"] = decile_auroc(s, y50, ent, edges)
    low = ent <= q20
    out["auroc_y50_low_entropy_quintile"] = auroc(s[low], y50[low])
    out["n_low_entropy_quintile"] = int(low.sum())
    return out


def population_metrics(scores, edges, q20):
    """Metrics of the run score and of raw patch mean entropy for every test population and their union."""
    res = {}
    groups = [(f"{d}:{p}", (scores.dataset == d) & (scores.population == p))
              for d, p in scores[["dataset", "population"]].drop_duplicates().itertuples(index=False)]
    for d in scores.dataset.unique():
        if scores[scores.dataset == d].population.nunique() > 1:
            groups.append((f"{d}:all", scores.dataset == d))
    for name, m in groups:
        df = scores[m]
        res[name] = {"model": metrics(df, edges, q20), "entropy": metrics(df.assign(score=df.ent_mean), edges, q20)}
    return res


def pixel_auroc(scores):
    """Patch score broadcast to the sampled test pixels (the existing 5% protocol) next to MSP and entropy."""
    rows = []
    for dataset in scores.dataset.unique():
        W, H = DATASETS[dataset].size
        Wp = W // PATCH
        cols = {"score": [], "MSP": [], "entropy": [], "_err": [], "_pop": []}
        sd = scores[scores.dataset == dataset]
        pops = sd.population.to_numpy()
        for image_id, rows_ in frame_runs(sd):
            pop = pops[rows_.start]
            z = np.load(base_path(dataset, image_id))
            if "pix_idx" not in z:
                continue
            per_patch = np.full((H // PATCH) * Wp, np.nan)
            per_patch[sd.patch.to_numpy()[rows_]] = sd.score.to_numpy()[rows_]
            pix = z["pix_idx"].astype(np.int64)
            s = per_patch[(pix // W) // PATCH * Wp + (pix % W) // PATCH]
            assert np.isfinite(s).all(), image_id
            cols["score"].append(s)
            cols["MSP"].append(z["pix_msp"])
            cols["entropy"].append(z["pix_ent"])
            cols["_err"].append(z["pix_err"])
            cols["_pop"].append(np.full(len(s), pop))
        if not cols["score"]:
            continue
        cols = {k: np.concatenate(v) for k, v in cols.items()}
        groups = {f"{dataset}_{p}": cols["_pop"] == p for p in np.unique(cols["_pop"])}
        if len(groups) > 1:
            groups[f"{dataset}_all"] = np.ones(len(cols["_err"]), bool)
        for g, m in groups.items():
            for k in ("score", "MSP", "entropy"):
                rows.append({"domain": g, "score": k, "auroc": auroc(cols[k][m], cols["_err"][m]),
                             "n_pixels": int(m.sum()), "error_rate": float(cols["_err"][m].mean())})
    return pd.DataFrame(rows)


# --- Commands ---

def model_params(args, **over):
    return {**PARAMS, "task_type": "GPU" if args.gpu else "CPU", **over}


def cmd_cv(args):
    out = run_dir(args.features, args.sae)
    out.mkdir(parents=True, exist_ok=True)
    parts = FEATURE_SETS[args.features]
    meta = p0_meta()
    latents, n_cols = firing_latents(meta, args.sae) if "codes" in parts else (None, None)
    if latents is not None:
        np.save(out / "latents.npy", latents)
        print(f"latents firing in ≥ {MIN_FIRE:.1%} of P0 patches: {len(latents)} of {n_cols}")
    keep, pool = p0_pool(meta, args.features, args.sae, latents)
    frames = np.array(sorted(meta.image_id.unique()))
    fold_of = {frames[i]: f for f, (_, va) in
               enumerate(GroupKFold(FOLDS, shuffle=True, random_state=SEED).split(frames, groups=frames)) for i in va}
    fold = meta.image_id.map(fold_of).to_numpy()
    train_fold = fold[keep]
    oof, best = np.full(len(meta), np.nan), []
    for f in range(FOLDS):
        t0 = time.time()
        model = CatBoost(model_params(args, iterations=args.iterations))
        model.fit(pool.slice(np.flatnonzero(train_fold != f)), eval_set=pool.slice(np.flatnonzero(train_fold == f)))
        best.append(model.get_best_iteration() + 1)
        rows = np.flatnonzero(fold == f)
        oof[rows] = predict(model, FIT_DATASET, meta.iloc[rows], args.sae, args.features, latents)
        print(f"fold {f}: best iteration {best[-1]}, {time.time() - t0:.0f} s")
    assert np.isfinite(oof).all()
    scores = meta[KEYS].assign(score=oof, fold=fold, dataset=FIT_DATASET, population=FIT_POP)
    scores.to_parquet(out / "cv_scores.parquet")
    edges, q20 = entropy_bins(meta)
    res = population_metrics(scores, edges, q20)
    write_json(out / "cv_metrics.json", res, stamp=False)
    write_json(out / "cv_config.json",
               {"features": args.features, "sae": args.sae if "codes" in parts else None, "params": model_params(args),
                "folds": FOLDS, "seed": SEED, "clean_keep": CLEAN_KEEP, "min_fire": MIN_FIRE,
                "n_latents": None if latents is None else len(latents), "n_code_columns": n_cols,
                "n_train_patches": int(keep.sum()), "n_patches": len(meta), "best_iterations": best,
                "final_iterations": int(round(np.mean(best)))})
    m = res[f"{FIT_DATASET}:{FIT_POP}"]
    print(f"P0 CV {run_name(args.features, args.sae)}: recall@10% {m['model']['recall@10%']:.4f} "
          f"(entropy {m['entropy']['recall@10%']:.4f}), decile AUROC y50 {m['model']['auroc_y50_entropy_deciles']:.4f} "
          f"(entropy {m['entropy']['auroc_y50_entropy_deciles']:.4f})")


def cmd_test(args):
    out = run_dir(args.features, args.sae)
    cv_cfg = json.loads((out / "cv_config.json").read_text(encoding="utf-8"))
    latents = np.load(out / "latents.npy") if "codes" in FEATURE_SETS[args.features] else None
    meta = p0_meta()
    keep, pool = p0_pool(meta, args.features, args.sae, latents)
    params = model_params(args, iterations=cv_cfg["final_iterations"])
    for k in ("od_type", "od_wait", "use_best_model"):
        params.pop(k)
    model = CatBoost(params)
    model.fit(pool)
    model.save_model(str(out / "model.cbm"))
    del pool
    edges, q20 = entropy_bins(meta)

    parts = []
    for spec in args.tests:
        dataset, pop = spec.split(":")
        pops = populations(dataset) if pop == "all" else [pop]
        for p in pops:
            m = load_meta(dataset, population(dataset, p))
            s = predict(model, dataset, m, args.sae, args.features, latents)
            parts.append(m[KEYS].assign(score=s, dataset=dataset, population=p))
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out / "scores.parquet")
    res = population_metrics(scores, edges, q20)
    write_json(out / "metrics.json", res, stamp=False)
    pix = pixel_auroc(scores)
    pix.to_csv(out / "pixel_auroc.csv", index=False)
    write_json(out / "config.json",
               {**{k: cv_cfg[k] for k in ("features", "sae", "n_latents", "clean_keep", "min_fire")},
                "params": params, "tests": args.tests, "entropy_decile_edges": edges.tolist(),
                "entropy_q20": q20, "n_train_patches": int(keep.sum())})
    rows = [{"test": t, "score": k, **{c: v[c] for c in ("recall@10%", "auroc_y50", "ap_y50",
                                                         "auroc_y50_entropy_deciles")}}
            for t, r in res.items() for k, v in r.items()]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    if len(pix):
        print(pix.round(4).to_string(index=False))


def paired_bootstrap(a, b, metric, edges, rng_counts):
    """Point difference a − b of the metric, 95% interval and two-sided bootstrap p over frame resamples."""
    frame_idx = pd.factorize(a.image_id)[0]
    e, y50, ent = a.n_err.to_numpy(), a.err_frac.to_numpy() >= 0.5, a.ent_mean.to_numpy()
    if metric.startswith("recall@"):
        budget = float(metric[len("recall@"):-1]) / 100
        orders = [np.argsort(-x.score.to_numpy(), kind="stable") for x in (a, b)]
        stat = lambda k, w: recall_at(orders[k], e, budget, w)
    elif metric == "auroc_y50_entropy_deciles":
        dec = np.searchsorted(edges, ent, side="right")
        groups = [np.flatnonzero(dec == j) for j in range(10)]
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


def cmd_compare(args):
    rows, counts = [], {}
    for test, metric, run, control in COMPARISONS:
        sa = pd.read_parquet(run_dir(run, args.sae) / "scores.parquet")
        sb = pd.read_parquet(run_dir(control, args.sae) / "scores.parquet")
        edges = np.array(json.loads((run_dir(run, args.sae) / "config.json").read_text(encoding="utf-8"))
                         ["entropy_decile_edges"])
        dataset, pop = test.split(":")
        sel = lambda s: s[(s.dataset == dataset) & ((s.population == pop) if pop != "all" else True)].reset_index(drop=True)
        a, b = sel(sa), sel(sb)
        assert len(a) and a[["image_id", "patch"]].equals(b[["image_id", "patch"]]), (test, run, control)
        if test not in counts:                  # the same frame resamples for every comparison on a test
            n = a.image_id.nunique()
            counts[test] = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=args.n_boot)
        point, lo, hi, p = paired_bootstrap(a, b, metric, edges, counts[test])
        rows.append({"test": test, "metric": metric, "run": run_name(run, args.sae),
                     "control": run_name(control, args.sae), "diff": point, "ci_lo": lo, "ci_hi": hi, "p_boot": p,
                     "excludes_0": bool(lo > 0 or hi < 0), "borderline": bool((lo > 0 or hi < 0) and p >= 0.01)})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / f"compare_{args.sae}.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cv", "test", "compare"])
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="F1")
    ap.add_argument("--sae", default="D2k8", help="SAE config of the code columns")
    ap.add_argument("--tests", nargs="+", default=list(TESTS), help="<dataset>:<population|all>")
    ap.add_argument("--iterations", type=int, default=PARAMS["iterations"], help="upper bound for CV")
    ap.add_argument("--gpu", action="store_true", help="CatBoost task_type=GPU")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    {"cv": cmd_cv, "test": cmd_test, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
