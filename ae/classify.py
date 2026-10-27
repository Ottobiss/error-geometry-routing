"""Steps 5 and 7: classifiers on the patch features, GroupKFold on P0, frozen tests on P1 and ACDC
(PLANv2.md §6, §7, §9, §10).

    uv run python -m ae.classify cv --features R --encoder ae_b8 --model catboost      # P0, out-of-fold scores
    uv run python -m ae.classify cv --features R --encoder ae_b8 --model logreg
    uv run python -m ae.classify test --features R --encoder ae_b8 --model catboost    # final fit, P1 and ACDC
    uv run python -m ae.classify compare                                               # pre-registered (§10)

Feature sets (parts: f0 = v1 output features incl. the categorical majority class, n = N, r = R, z = Z of the
encoder, raw = v1 raw activations):
  F0 f0 · F0N f0+n · R f0+n+r · Z f0+z · ZR f0+n+z+r · R_only n+r · F3F0 raw+f0
R_pca, Z_pca, ZR_pca are R, Z, ZR with --encoder pca_b8. N does not depend on the encoder.

Both classifiers use the v1 P0 training rows (all patches with errors, 25% of clean ones with weight ×4,
sae.classify.training_rows) and the v1 folds (GroupKFold(5) by frame, shuffle, seed 0).
  catboost  v1 setup (CrossEntropy on err_frac, weight n_valid, depth 6, lr 0.05, l2 3, early stopping after
            200 rounds) with the iteration cap raised to 10,000; final model = mean best iteration of the folds.
  logreg    L2 logistic regression on err_frac: weighted cross-entropy with fractional targets, which equals
            sklearn LogisticRegression on two rows per patch (y = 1 with weight w·err_frac, y = 0 with weight
            w·(1 − err_frac)); fitted in torch with full-batch L-BFGS, so that the raw sets fit in memory.
            Objective C·Σ wᵢ ℓᵢ + ½‖β‖² (intercept not penalised) with the weights scaled to mean 1 over the
            fitted rows, as with unit weights in sklearn. Numeric columns standardised on the fitted rows; the
            predicted class one-hot (19); patch row and column one-hot in 8 bands each (band = ⌊8·row / Hp⌋,
            per dataset). C from {1e-3, 1e-2, 1e-1, 1, 10} by P0 CV recall@10%, then fixed.

Output: results/ae/runs/<set>[_<encoder>]_<model>/
  cv_config.json, cv_scores.parquet, cv_metrics.json (+ cv_c_grid.csv for logreg)                 (cv)
  config.json, scores.parquet, metrics.json, pixel_auroc.csv, model.cbm | model.npz               (test)
results/ae/compare.csv (compare).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from catboost import CatBoost, Pool
from sklearn.model_selection import GroupKFold

from ae import ROOT
from ae.metrics import ROBUST_DECILES, holm, paired_bootstrap, population_metrics, robust_deciles
from ae.patch_features import NORM, patch_path
from core.components import NC
from core.data import DATASETS, FIT_DATASET, FIT_POP, population, populations
from core.paths import write_json
from sae.classify import (FOLDS, SEED, CLEAN_KEEP, PARAMS as V1_PARAMS, column_names, entropy_bins, frame_runs,
                          load_meta, load_parts, pixel_auroc, recall_at, training_rows)
from sae.hooks import PATCH
from sae.patch_features import F0_NUM

FEATURE_SETS = {"F0": ("f0",), "F0N": ("f0", "n"), "R": ("f0", "n", "r"), "Z": ("f0", "z"),
                "ZR": ("f0", "n", "z", "r"), "R_only": ("n", "r"), "F3F0": ("raw", "f0")}
MODELS = ("catboost", "logreg")
PARAMS = {**V1_PARAMS, "iterations": 10000}
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
BANDS = 8
KEYS = ["image_id", "patch", "n_valid", "n_err", "err_frac", "ent_mean"]
TESTS = ("cityscapes:p1", "acdc:all")
# Pre-registered comparisons (PLANv2.md §10): family, test, metric, (set, encoder), (set, encoder); CatBoost.
COMPARISONS = [("primary", "acdc:all", "recall@10%", ("R", "ae"), ("F0N", None)),
               ("primary", "acdc:all", "auroc_y50_robust", ("R", "ae"), ("F0N", None)),
               ("primary", "cityscapes:p1", "recall@10%", ("R", "ae"), ("F0N", None)),
               ("secondary", "acdc:all", "recall@10%", ("ZR", "ae"), ("F0", None)),
               ("secondary", "cityscapes:p1", "recall@10%", ("ZR", "ae"), ("F0", None)),
               ("secondary", "acdc:all", "recall@10%", ("R", "ae"), ("R", "pca")),
               ("secondary", "cityscapes:p1", "recall@10%", ("Z", "ae"), ("F3F0", None))]


def uses_encoder(features):
    return bool({"r", "z"} & set(FEATURE_SETS[features]))


def run_name(features, encoder, model):
    return f"{features}_{encoder}_{model}" if uses_encoder(features) else f"{features}_{model}"


def run_dir(features, encoder, model):
    return ROOT / "runs" / run_name(features, encoder, model)


# --- Data ---

def _info(name, dataset):
    return json.loads((ROOT / "patches" / name / dataset / "_info.json").read_text())


def design(dataset, meta, parts, encoder):
    """Feature DataFrame of the rows of meta (float32 columns, cls int32) and its categorical columns."""
    frames = []
    if "raw" in parts:
        raw = load_parts(dataset, meta, None, ("raw",))[1]
        frames.append(pd.DataFrame(raw, columns=column_names(dataset, None, "raw"), copy=False))
    if "f0" in parts:
        frames.append(meta[F0_NUM + ["cls"]].reset_index(drop=True))
    patch = meta.patch.to_numpy()
    for part, name, key in (("n", NORM, "N"), ("z", encoder, "Z"), ("r", encoder, "R")):
        if part not in parts:
            continue
        cols = _info(name, dataset)[key]
        x = np.empty((len(meta), len(cols)), dtype=np.float32)
        for image_id, rows in frame_runs(meta):
            x[rows] = np.load(patch_path(name, dataset, image_id))[key][patch[rows]]
        frames.append(pd.DataFrame(x, columns=cols, copy=False))
    X = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    assert X.columns.is_unique
    return X, (["cls"] if "f0" in parts else [])


def p0_meta():
    return load_meta(FIT_DATASET, population(FIT_DATASET, FIT_POP))


def folds(meta):
    """v1 folds: GroupKFold(5) by frame, shuffle, seed 0 → fold of every row of meta."""
    frames = np.array(sorted(meta.image_id.unique()))
    fold_of = {frames[i]: f for f, (_, va) in
               enumerate(GroupKFold(FOLDS, shuffle=True, random_state=SEED).split(frames, groups=frames)) for i in va}
    return meta.image_id.map(fold_of).to_numpy()


def chunks(meta, chunk=40):
    """Row slices of meta covering whole frames, chunk frames each (memory)."""
    runs = frame_runs(meta)
    return [slice(runs[i][1].start, runs[min(i + chunk, len(runs)) - 1][1].stop) for i in range(0, len(runs), chunk)]


# --- Logistic regression ---

def linear_inputs(X, dataset, device):
    """Numeric block (float32 tensor), one-hot block (class, row band, column band) and numeric names."""
    one_hot = []
    if "cls" in X:
        W, H = DATASETS[dataset].size
        Hp, Wp = H // PATCH, W // PATCH
        cls, row, col = (X[c].to_numpy().astype(np.int64) for c in ("cls", "row", "col"))
        assert cls.max() < NC and row.max() < Hp and col.max() < Wp
        one_hot = [np.eye(NC, dtype=np.float32)[cls], np.eye(BANDS, dtype=np.float32)[BANDS * row // Hp],
                   np.eye(BANDS, dtype=np.float32)[BANDS * col // Wp]]
        X = X.drop(columns=["cls", "row", "col"])
    num = torch.from_numpy(X.to_numpy(np.float32)).to(device)
    oh = torch.from_numpy(np.concatenate(one_hot, 1) if one_hot else np.zeros((len(X), 0), np.float32)).to(device)
    return num, oh, list(X.columns)


class Logistic:
    """L2 logistic regression on fractional targets (see the module docstring)."""

    def __init__(self, C):
        self.C = C

    def fit(self, num, oh, y, w, rows, max_iter=2000):
        """Fit on the rows (index tensor) of num / oh; y err_frac and w weights as float tensors."""
        dev = num.device
        mask = torch.zeros(len(num), dtype=torch.bool, device=dev)
        mask[rows] = True
        n = int(mask.sum())
        ww = torch.where(mask, w, torch.zeros_like(w))
        ww = ww / ww.sum() * n                                   # mean 1 over the fitted rows
        s = torch.zeros(num.shape[1], dtype=torch.float64, device=dev)
        ss = torch.zeros_like(s)
        for i in range(0, len(num), 1 << 16):                    # in chunks: no copy of the whole matrix
            x = num[i:i + (1 << 16)][mask[i:i + (1 << 16)]].double()
            s += x.sum(0)
            ss += x.pow(2).sum(0)
        mu = s / n
        sd = (ss / n - mu.pow(2)).clamp(min=0).sqrt()
        sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
        self.mu, self.sd = mu.float(), sd.float()
        self.beta = torch.zeros(num.shape[1], device=dev, requires_grad=True)
        self.gamma = torch.zeros(oh.shape[1], device=dev, requires_grad=True)
        self.b0 = torch.zeros((), device=dev, requires_grad=True)
        params = [self.beta, self.gamma, self.b0]
        opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-12,
                                history_size=20, line_search_fn="strong_wolfe")

        def objective():
            t = self._logit(num, oh)
            data = (ww * (torch.nn.functional.softplus(t) - y * t)).double().sum()
            pen = 0.5 * (self.beta.double().pow(2).sum() + self.gamma.double().pow(2).sum()) / self.C
            return ((data + pen) / n).float()

        def closure():
            opt.zero_grad()
            loss = objective()
            loss.backward()
            return loss

        opt.step(closure)
        loss = closure()
        self.n_iter = opt.state[opt._params[0]]["n_iter"]
        self.grad_norm = float(torch.cat([p.grad.reshape(-1) for p in params]).abs().max())
        self.loss = loss.item()
        for p in params:
            p.requires_grad_(False)
        return self

    def _logit(self, num, oh):
        bs = self.beta / self.sd
        return num @ bs - self.mu @ bs + oh @ self.gamma + self.b0

    @torch.no_grad()
    def predict(self, num, oh):
        return torch.sigmoid(self._logit(num, oh)).cpu().numpy()

    def state(self):
        return {k: v.detach().cpu().numpy() for k, v in
                (("beta", self.beta), ("gamma", self.gamma), ("b0", self.b0), ("mu", self.mu), ("sd", self.sd))}


# --- Model wrappers ---

def model_params(args, **over):
    return {**PARAMS, "task_type": "GPU" if args.gpu else "CPU", **over}


def catboost_pool(meta, keep, weight, parts, encoder):
    t0 = time.time()
    X, cat = design(FIT_DATASET, meta[keep], parts, encoder)
    pool = Pool(X, label=meta.err_frac.to_numpy()[keep], weight=weight[keep], cat_features=cat)
    del X
    pool.quantize()
    print(f"P0 pool: {pool.num_row()} x {pool.num_col()} ({keep.sum()} of {len(meta)} patches), {time.time() - t0:.0f} s")
    return pool


def catboost_predict(model, dataset, meta, parts, encoder):
    out = []
    for sl in chunks(meta):
        X, cat = design(dataset, meta.iloc[sl], parts, encoder)
        out.append(model.predict(Pool(X, cat_features=cat), prediction_type="Probability")[:, 1])
    return np.concatenate(out)


def logreg_train_inputs(meta, keep, weight, parts, encoder, device):
    X, _ = design(FIT_DATASET, meta[keep], parts, encoder)
    num, oh, names = linear_inputs(X, FIT_DATASET, device)
    del X
    y = torch.from_numpy(meta.err_frac.to_numpy()[keep].astype(np.float32)).to(device)
    w = torch.from_numpy(weight[keep].astype(np.float32)).to(device)
    print(f"P0 linear inputs: {num.shape[0]} x ({num.shape[1]} numeric + {oh.shape[1]} one-hot)")
    return num, oh, y, w, names


def logreg_predict(models, dataset, meta, parts, encoder, device):
    """Scores of all rows of meta for each model of the list."""
    out = [[] for _ in models]
    for sl in chunks(meta):
        X, _ = design(dataset, meta.iloc[sl], parts, encoder)
        num, oh, _ = linear_inputs(X, dataset, device)
        for k, m in enumerate(models):
            out[k].append(m.predict(num, oh))
    return [np.concatenate(o) for o in out]


# --- Commands ---

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def recall10(meta, score):
    return recall_at(np.argsort(-score, kind="stable"), meta.n_err.to_numpy(), 0.10)


def cmd_cv(args):
    parts = FEATURE_SETS[args.features]
    out = run_dir(args.features, args.encoder, args.model)
    out.mkdir(parents=True, exist_ok=True)
    meta = p0_meta()
    edges, _ = entropy_bins(meta)
    robust, pos = robust_deciles(meta, edges)
    if robust != ROBUST_DECILES:
        print(f"WARNING: P0 deciles with >= 200 y50 positives are {robust}, frozen list {ROBUST_DECILES}")
    keep, weight = training_rows(meta)
    fold = folds(meta)
    train_fold = fold[keep]
    cfg = {"features": args.features, "parts": parts, "encoder": args.encoder if uses_encoder(args.features) else None,
           "model": args.model, "folds": FOLDS, "seed": SEED, "clean_keep": CLEAN_KEEP,
           "n_train_patches": int(keep.sum()), "n_patches": len(meta), "entropy_decile_edges": edges.tolist(),
           "robust_deciles": list(ROBUST_DECILES), "y50_positives_per_decile": pos}

    if args.model == "catboost":
        pool = catboost_pool(meta, keep, weight, parts, args.encoder)
        oof, best, cap = np.full(len(meta), np.nan), [], []
        for f in range(FOLDS):
            t0 = time.time()
            model = CatBoost(model_params(args, iterations=args.iterations))
            model.fit(pool.slice(np.flatnonzero(train_fold != f)), eval_set=pool.slice(np.flatnonzero(train_fold == f)))
            best.append(model.get_best_iteration() + 1)
            run = len(next(iter(model.get_evals_result()["validation"].values())))   # iterations actually run
            cap.append(run >= args.iterations)                  # stopped by the cap, not by early stopping
            rows = np.flatnonzero(fold == f)
            oof[rows] = catboost_predict(model, FIT_DATASET, meta.iloc[rows], parts, args.encoder)
            print(f"fold {f}: best iteration {best[-1]} of {run} run, {time.time() - t0:.0f} s")
        if any(cap):
            print(f"WARNING: the iteration cap {args.iterations} was reached in {sum(cap)} fold(s)")
        cfg.update({"params": model_params(args, iterations=args.iterations), "best_iterations": best,
                    "cap_reached": cap, "final_iterations": int(round(np.mean(best)))})
    else:
        dev = device()
        num, oh, y, w, names = logreg_train_inputs(meta, keep, weight, parts, args.encoder, dev)
        oof = {C: np.full(len(meta), np.nan) for C in C_GRID}
        fits = []
        for f in range(FOLDS):
            t0 = time.time()
            rows = torch.from_numpy(np.flatnonzero(train_fold != f)).to(dev)
            models = [Logistic(C).fit(num, oh, y, w, rows) for C in C_GRID]
            fits += [{"fold": f, "C": m.C, "n_iter": m.n_iter, "grad_max": m.grad_norm, "loss": m.loss} for m in models]
            va = np.flatnonzero(fold == f)
            for C, s in zip(C_GRID, logreg_predict(models, FIT_DATASET, meta.iloc[va], parts, args.encoder, dev)):
                oof[C][va] = s
            print(f"fold {f}: L-BFGS iterations {[m.n_iter for m in models]}, {time.time() - t0:.0f} s")
        grid = pd.DataFrame([{"C": C, "recall@10%": recall10(meta, oof[C])} for C in C_GRID])
        best_C = float(grid.C[grid["recall@10%"].idxmax()])
        grid.to_csv(out / "cv_c_grid.csv", index=False)
        print(grid.round(4).to_string(index=False), f"\nchosen C = {best_C:g}")
        oof = oof[best_C]
        cfg.update({"C_grid": list(C_GRID), "C": best_C, "numeric_columns": len(names),
                    "one_hot_columns": int(oh.shape[1]), "fits": fits})
    assert np.isfinite(oof).all()
    scores = meta[KEYS].assign(score=oof, fold=fold, dataset=FIT_DATASET, population=FIT_POP)
    scores.to_parquet(out / "cv_scores.parquet")
    res = population_metrics(scores, edges)
    write_json(out / "cv_metrics.json", res, stamp=False)
    write_json(out / "cv_config.json", cfg)
    m = res[f"{FIT_DATASET}:{FIT_POP}"]
    print(f"P0 CV {run_name(args.features, args.encoder, args.model)}: recall@10% {m['model']['recall@10%']:.4f} "
          f"(entropy {m['entropy']['recall@10%']:.4f}), robust AUROC y50 {m['model']['auroc_y50_robust']:.4f} "
          f"(entropy {m['entropy']['auroc_y50_robust']:.4f})")


def cmd_test(args):
    parts = FEATURE_SETS[args.features]
    out = run_dir(args.features, args.encoder, args.model)
    cv_cfg = json.loads((out / "cv_config.json").read_text(encoding="utf-8"))
    assert cv_cfg["model"] == args.model and tuple(cv_cfg["parts"]) == parts
    meta = p0_meta()
    edges = np.array(cv_cfg["entropy_decile_edges"])
    assert np.array_equal(edges, entropy_bins(meta)[0])
    keep, weight = training_rows(meta)

    if args.model == "catboost":
        pool = catboost_pool(meta, keep, weight, parts, args.encoder)
        params = model_params(args, iterations=cv_cfg["final_iterations"])
        for k in ("od_type", "od_wait", "use_best_model"):
            params.pop(k)
        model = CatBoost(params)
        model.fit(pool)
        model.save_model(str(out / "model.cbm"))
        del pool
        predict = lambda dataset, m: catboost_predict(model, dataset, m, parts, args.encoder)
        fit_info = {"params": params}
    else:
        dev = device()
        num, oh, y, w, names = logreg_train_inputs(meta, keep, weight, parts, args.encoder, dev)
        model = Logistic(cv_cfg["C"]).fit(num, oh, y, w, torch.arange(len(num), device=dev))
        del num, oh, y, w
        np.savez(out / "model.npz", **model.state(), numeric_columns=np.array(names), C=model.C)
        predict = lambda dataset, m: logreg_predict([model], dataset, m, parts, args.encoder, dev)[0]
        fit_info = {"C": model.C, "n_iter": model.n_iter, "grad_max": model.grad_norm}

    scores = []
    for spec in args.tests:
        dataset, pop = spec.split(":")
        for p in (populations(dataset) if pop == "all" else [pop]):
            m = load_meta(dataset, population(dataset, p))
            scores.append(m[KEYS].assign(score=predict(dataset, m), dataset=dataset, population=p))
    scores = pd.concat(scores, ignore_index=True)
    assert np.isfinite(scores.score).all()
    scores.to_parquet(out / "scores.parquet")
    res = population_metrics(scores, edges)
    write_json(out / "metrics.json", res, stamp=False)
    pix = pixel_auroc(scores)
    pix.to_csv(out / "pixel_auroc.csv", index=False)
    write_json(out / "config.json", {**{k: cv_cfg[k] for k in ("features", "parts", "encoder", "model", "clean_keep",
                                                                "robust_deciles")},
                                     **fit_info, "tests": args.tests, "entropy_decile_edges": edges.tolist(),
                                     "n_train_patches": int(keep.sum())})
    rows = [{"test": t, "score": k, **{c: v[c] for c in ("recall@10%", "auroc_y50", "ap_y50", "auroc_y50_robust")}}
            for t, r in res.items() for k, v in r.items()]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    if len(pix):
        print(pix.round(4).to_string(index=False))


def cmd_compare(args):
    enc = {"ae": args.encoder, "pca": args.pca, None: None}
    rows, counts = [], {}
    for family, test, metric, (run, re), (control, ce) in COMPARISONS:
        da, db = run_dir(run, enc[re], "catboost"), run_dir(control, enc[ce], "catboost")
        sa, sb = pd.read_parquet(da / "scores.parquet"), pd.read_parquet(db / "scores.parquet")
        edges = np.array(json.loads((da / "config.json").read_text(encoding="utf-8"))["entropy_decile_edges"])
        assert np.array_equal(edges, json.loads((db / "config.json").read_text(encoding="utf-8"))["entropy_decile_edges"])
        dataset, pop = test.split(":")
        sel = lambda s: s[(s.dataset == dataset) & ((s.population == pop) if pop != "all" else True)].reset_index(drop=True)
        a, b = sel(sa), sel(sb)
        assert len(a) and a[["image_id", "patch"]].equals(b[["image_id", "patch"]]), (test, run, control)
        if test not in counts:                  # the same frame resamples for every comparison on a test
            n = a.image_id.nunique()
            counts[test] = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=args.n_boot)
        point, lo, hi, p = paired_bootstrap(a, b, metric, edges, counts[test])
        rows.append({"family": family, "test": test, "metric": metric, "run": da.name, "control": db.name,
                     "diff": point, "ci_lo": lo, "ci_hi": hi, "p_boot": p, "excludes_0": bool(lo > 0 or hi < 0)})
    table = pd.DataFrame(rows)
    prim = table.family == "primary"
    table["p_holm"] = np.nan
    table.loc[prim, "p_holm"] = holm(table.p_boot[prim].to_numpy())
    # primary: Holm-adjusted p < 0.05; secondary (no correction): interval excludes 0 with p ≥ 0.01 → borderline
    table["significant"] = pd.Series([bool(p < 0.05) if pr else None for p, pr in zip(table.p_holm, prim)], dtype=object)
    table["borderline"] = pd.Series([None if pr else bool(ex and p >= 0.01)
                                     for ex, p, pr in zip(table.excludes_0, table.p_boot, prim)], dtype=object)
    table.to_csv(ROOT / "compare.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["cv", "test", "compare"])
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="R")
    ap.add_argument("--encoder", default="ae_b8", help="encoder of the Z / R columns (compare: the AE)")
    ap.add_argument("--pca", default="pca_b8", help="compare: the PCA encoder")
    ap.add_argument("--model", choices=MODELS, default="catboost")
    ap.add_argument("--tests", nargs="+", default=list(TESTS), help="<dataset>:<population|all>")
    ap.add_argument("--iterations", type=int, default=PARAMS["iterations"], help="CatBoost upper bound for CV")
    ap.add_argument("--gpu", action="store_true", help="CatBoost task_type=GPU")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    {"cv": cmd_cv, "test": cmd_test, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
