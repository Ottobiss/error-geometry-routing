"""Step 4: label-free checks of the residual before any classifier (PLANv2.md §8).

    uv run python -m ae.checks --encoders ae_b8 pca_b8
    uv run python -m ae.checks --encoders ae_b16 ae_b4          # bottleneck ablation (sensitivity only)

Per encoder, for every R column (and the N control columns) of the patch features (ae.patch_features):
  1. shift.csv     mean over all patches of P0, P1, every ACDC condition and ACDC all; ratio to P0
  2. spearman.csv  Spearman correlation on the valid P0 patches with patch mean entropy, and of R with the
                   matching N column (above ≈ 0.8, R is entropy / norm again)
  3. auroc.csv     AUROC for y_any (err_frac ≥ 0.1) on the valid P0 patches, overall and averaged over the
                   frozen robust entropy deciles (ae.metrics.ROBUST_DECILES); patch mean entropy as reference
No fitting on labels: AUROC uses the raw column as the score (higher = more likely wrong).

Output: results/ae/checks/<encoder>_{shift,spearman,auroc}.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ae import ROOT
from ae.metrics import ROBUST_DECILES, deciles
from ae.patch_features import NORM, patch_path
from core.data import FIT_DATASET, FIT_POP, population, populations
from sae.classify import auroc, entropy_bins, load_meta


def info(encoder, dataset):
    return json.loads((ROOT / "patches" / encoder / dataset / "_info.json").read_text())


def load_columns(encoder, dataset, ids, valid_only):
    """Base meta (sae.classify.load_meta: valid patches) or all patches, with the R and N columns."""
    r_cols, n_cols = info(encoder, dataset)["R"], info(NORM, dataset)["N"]
    if valid_only:
        meta = load_meta(dataset, ids)
        patch = meta.patch.to_numpy()
        starts = np.flatnonzero(np.r_[True, meta.image_id.to_numpy()[1:] != meta.image_id.to_numpy()[:-1]])
        bounds = list(zip(starts, np.r_[starts[1:], len(meta)]))
        assert [meta.image_id.iloc[s] for s, _ in bounds] == list(ids)
        R = np.concatenate([np.load(patch_path(encoder, dataset, i))["R"][patch[s:e]] for i, (s, e) in zip(ids, bounds)])
        N = np.concatenate([np.load(patch_path(NORM, dataset, i))["N"][patch[s:e]] for i, (s, e) in zip(ids, bounds)])
    else:
        meta = None
        R = np.concatenate([np.load(patch_path(encoder, dataset, i))["R"] for i in ids])
        N = np.concatenate([np.load(patch_path(NORM, dataset, i))["N"] for i in ids])
    return meta, pd.DataFrame(R, columns=r_cols), pd.DataFrame(N, columns=n_cols)


def run(encoder, out):
    # 1. Shift: all patches of every population.
    rows = []
    for dataset in ("cityscapes", "acdc"):
        pops = [p for p in populations(dataset)]
        parts = {}
        for p in pops:
            _, R, N = load_columns(encoder, dataset, population(dataset, p), valid_only=False)
            parts[p] = (R, N)
            rows.append({"population": f"{dataset}:{p}", **R.mean().to_dict(), **N.mean().to_dict()})
        if dataset == "acdc":
            R = pd.concat([v[0] for v in parts.values()])
            N = pd.concat([v[1] for v in parts.values()])
            rows.append({"population": "acdc:all", **R.mean().to_dict(), **N.mean().to_dict()})
    shift = pd.DataFrame(rows).set_index("population").T
    ref = shift[f"{FIT_DATASET}:{FIT_POP}"]
    for c in list(shift.columns):
        shift[f"{c}/p0"] = shift[c] / ref
    shift.index.name = "column"
    shift.to_csv(out / f"{encoder}_shift.csv")

    # 2–3. Valid P0 patches.
    meta, R, N = load_columns(encoder, FIT_DATASET, population(FIT_DATASET, FIT_POP), valid_only=True)
    ent = meta.ent_mean.to_numpy()
    yany = meta.err_frac.to_numpy() >= 0.1
    edges, _ = entropy_bins(meta)
    dec = deciles(ent, edges)

    def robust(score):
        return float(np.nanmean([auroc(score[dec == j], yany[dec == j]) for j in ROBUST_DECILES]))

    sp_rows, au_rows = [], []
    for c in [*R.columns, *N.columns]:
        v = (R[c] if c in R else N[c]).to_numpy()
        row = {"column": c, "spearman_entropy": spearmanr(v, ent).statistic}
        if c in R:
            row["spearman_norm"] = spearmanr(v, N[c.replace("_r", "_n", 1)].to_numpy()).statistic
        sp_rows.append(row)
        au_rows.append({"column": c, "auroc_y_any": auroc(v, yany), "auroc_y_any_robust": robust(v)})
    au_rows.append({"column": "entropy (ent_mean)", "auroc_y_any": auroc(ent, yany), "auroc_y_any_robust": robust(ent)})
    pd.DataFrame(sp_rows).to_csv(out / f"{encoder}_spearman.csv", index=False)
    pd.DataFrame(au_rows).to_csv(out / f"{encoder}_auroc.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"== {encoder}: shift (mean per population)")
    print(shift[[c for c in shift.columns if not c.endswith("/p0")]].round(4).to_string())
    print(f"== {encoder}: valid P0 patches, y_any share {yany.mean():.4f}")
    print(pd.DataFrame(sp_rows).merge(pd.DataFrame(au_rows), on="column", how="outer").round(4).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    args = ap.parse_args()
    out = ROOT / "checks"
    out.mkdir(parents=True, exist_ok=True)
    for e in args.encoders:
        run(e, out)


if __name__ == "__main__":
    main()
