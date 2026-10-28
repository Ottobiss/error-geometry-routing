# PLAN v2: patch-level error prediction from a bottleneck autoencoder of Mix-FFN activations

Draft from 2026-10-27, branch `SAE`. Follows PLAN.md (v1) and RESULTSV1.md. Everything in v1 not mentioned
here stays as it was: the hooks, the patch grid and selection rule (§5), labels, F0, the CatBoost setup, P0
CV and the training-row sampling.

## 1. Why v2

v1 results on P0 CV (RESULTSV1.md §4):
- Internal features add nothing to the global ranking. Recall@10% is 0.566 for F0, 0.566 for F1, 0.569 for
  F3+F0 and 0.560 for entropy.
- **SAE codes do worse than raw activations** (F2 < F3, F1 < F3+F0). The sparse bottleneck loses information
  that CatBoost could use. Sparsity is there for human interpretation, which this question does not need.
- The F1 > F0 within-entropy gain (decile AUROC 0.739 vs 0.676) comes from deciles with 2–95 positive patches.
  Averaged over deciles with at least 200 positives, F1 ≈ F0 (0.661 vs 0.662), while F3+F0 still leads (0.738).
  The v1 decile metric is therefore too noisy to pre-register as it is.

v2 changes the question from "do interpretable latents detect errors" to two questions:
1. **Anomaly:** is a patch whose activations are poorly reconstructed by a model of *normal* B0 activations
   more likely to be wrong? The signal is the reconstruction error, not the code.
2. **Compression:** do dense bottleneck codes carry the error signal of the raw activations in fewer columns?

The bottleneck size matters more than reconstruction quality. An autoencoder that reconstructs everything
also reconstructs anomalies, and its residual then carries no signal.

## 2. Data

| Role | Data | Labels used |
|---|---|---|
| Autoencoder / PCA fit | **Cityscapes train**, 2,975 frames, images only | none |
| Classifier fit and selection | **P0**, 300 Cityscapes val frames, GroupKFold(5) by frame, seed 0 | yes |
| Test, in domain | **P1**, the other 200 Cityscapes val frames | evaluation only |
| Test, shift | **ACDC val**, fog / night / rain / snow, 406 frames, each condition and all together | evaluation only |

- ACDC test labels are not public, so ACDC val is the shift test set. It is never used for any choice.
- Why Cityscapes train for the AE: (a) P0 is no longer used to fit the AE, so P0 residuals are out-of-sample
  exactly like P1 and ACDC residuals, and no fold-wise AEs are needed; (b) stage 4 gets 1.5M positions
  instead of 154k (v1 risk §12).
- Caveat: B0 was fine-tuned on these frames, so "normal" means B0's own training distribution. All val frames
  are therefore slightly off-distribution. This is the same for P0, P1 and ACDC, so it does not bias the
  comparisons. Report the train → P0 FVE gap per block (§4.3).
- Download: `python -m download.cityscapes --prefix leftImg8bit/train/`. Only `gtFine/train` is on disk now.
- **Do not add `splits/cityscapes_train.txt`.** `core.data.test_populations` treats every split file except
  P0 as a test population. Pass `split="train"` to `Cityscapes` instead (`get_dataset` does not forward
  `split` yet).

## 3. Activations

Same layer as v1: the input of `fc2` in each of the 8 blocks (`sae.hooks.MixFFNTap`), one sample per
position, native grid.

- **Collection:** at most 2M random positions per block from Cityscapes train. Stage 4 has 512 per frame,
  so it uses all ≈ 1.52M positions. Hold out 5% of the train frames (seed 0) for AE early stopping.
- **Normalisation:** per block, mean and scale with E‖x‖² = d, fitted on the training frames of Cityscapes
  train (`sae.collect.fit_norm`). P0, P1 and ACDC use the same constants.
- **Disk:** about 14 GB fp16 (2M × {128, 256, 640} and 1.52M × 1,024, two blocks per stage).
- **Memory:** stage 3 is 2M × 640 fp16 ≈ 2.6 GB and stage 4 ≈ 3.1 GB, so each block fits on the 8 GB GPU
  on its own, as in `sae.train.load_block`.

## 4. Encoders

### 4.1 Bottleneck autoencoder (main)

One per block, on normalised activations x:

`x → Linear(d, h) → GELU → Linear(h, b) = z → Linear(b, h) → GELU → Linear(h, d) = x̂`, with h = d.

| Block | d | b = d/8 (main) | Ablation b |
|---|---|---|---|
| s1b1, s1b2 | 128 | 16 | 8, 32 |
| s2b1, s2b2 | 256 | 32 | 16, 64 |
| s3b1, s3b2 | 640 | 80 | 40, 160 |
| s4b1, s4b2 | 1,024 | 128 | 64, 256 |

- Loss: MSE on x. AdamW, lr 1e-3, weight decay 1e-4, cosine schedule, batch 4,096, seed 0.
- Up to 30 epochs with early stopping on the held-out train frames (patience 3 epochs). Record the number
  of steps.
- b = d/8 is fixed in advance and not tuned on labels. The ablation sizes are used only in §8.
- Expose the same `normalize / encode / decode / denormalize` interface as `BatchTopKSAE`, so that
  `sae.train.evaluate` and `splice_drop` work unchanged. L0 and the dead share do not apply.

### 4.2 PCA (linear alternative)

Per block, PCA on the same normalised Cityscapes-train activations, keeping the same b components.
z = top-b projections, x̂ = reconstruction from them. A linear autoencoder spans the same subspace as PCA,
so **AE vs PCA measures what the nonlinearity adds**. Fit with incremental or randomised PCA in float64
statistics: d ≤ 1,024, so the covariance fits in memory.

### 4.3 Quality (reported per block and encoder, no gates)

- FVE on held-out train frames, on P0, on P1 and on ACDC (per condition). The train → P0 → ACDC drop is the
  first sign of whether the residual reacts to shift.
- Splice test on P0 (v1 definition): mIoU drop when the block's fc2 input is replaced by x̂. This is
  descriptive only. A low-FVE bottleneck is expected to fail the v1 gate of 1.0 point; that is intended.

## 5. Patch features

All from the v1 closest-position rule (`sae.hooks.selection_index`). Stages 1–2 pool 16 or 4 positions per
patch; stages 3–4 gather one position.

| Group | Per position | Patch pooling (stages 1–2) | Columns (8 blocks) |
|---|---|---|---|
| **Z** codes | z (b values, signed, dense) | mean (max over signed dense codes does not mean much) | Σ b = 512 at b = d/8 |
| **R** reconstruction error | r = ‖x − x̂‖² / d | mean and max | 12 (2 + 2 + 2 + 2 for stages 1–2, 1 each for stages 3–4) |
| **N** activation norm (control) | ‖x‖² / d | mean and max | 12 |

- **N is the control for R.** Residuals often grow with activation magnitude. The claim "anomaly predicts
  error" is supported only if R adds something over N.
- Optional descriptive extra: relative residual r / (‖x‖² / d).
- Storage: one npz per frame under `results/ae/patches/<encoder>/<dataset>/<image_id>.npz`, holding Z, R and
  N as float32. Base tables (F0, labels, pixel samples) are reused from `results/sae/patches/base/`.

## 6. Feature sets

| Set | Contents | Role |
|---|---|---|
| F0 | v1 output features | control |
| F0N | F0 + N | control for R |
| **R** | F0 + N + R (AE) | main: anomaly |
| **Z** | F0 + Z (AE) | compression |
| **ZR** | F0 + N + Z + R (AE) | both |
| R_pca, Z_pca, ZR_pca | the same with PCA | linear alternative |
| R_only | N + R (AE), no F0 | descriptive: residual alone |
| F3F0 | v1 raw activations + F0 | reference: what the raw activations hold |

## 7. Classifiers

Every feature set is run with both classifiers, on the same P0 training rows (all patches with errors plus
25% of clean ones weighted ×4) and the same folds.

### 7.1 CatBoost

As in v1 (CrossEntropy on err_frac, weight n_valid, depth 6, lr 0.05, l2 3, early stopping 200 rounds),
except **the iteration cap is raised to 10,000**. In v1 the 3,000 cap was binding for F2, F3 and one F1
fold. With at most 536 columns, v2 sets should stop well below the cap; record it if they do not.

### 7.2 Linear model

**Logistic regression on err_frac** (the same cross-entropy as CatBoost, i.e. linear regression on the logit
scale), L2 penalty.
- Exact fractional targets: each patch becomes two rows, y = 1 with weight n_err and y = 0 with weight
  n_valid − n_err (times the ×4 clean-patch weight). This equals weighted cross-entropy on err_frac and works
  with `sklearn.linear_model.LogisticRegression`.
- Preprocessing fitted on P0 training rows only: standardise numeric columns, one-hot the predicted class,
  one-hot patch row in 8 bands (row and column enter a linear model badly as raw numbers).
- C from {1e-3, 1e-2, 1e-1, 1, 10} by P0 CV recall@10% for each feature set, then fixed.
- Plain least squares on err_frac is a possible variant, but it does not keep scores in [0, 1] and it treats
  the heavy tail of error-free patches badly. Run it only as a descriptive row if wanted.

The linear model answers "is the signal linearly readable". The paper's Tab. 4 analogue is CatBoost vs linear
on the same features.

## 8. Unsupervised checks (before any classifier, all populations)

These need no labels for fitting and decide whether R is an anomaly signal at all:
1. **Shift test:** mean R per block on P0, P1 and each ACDC condition. If R does not rise on ACDC, and most
   on night, R is not detecting shift.
2. **Redundancy:** Spearman correlation of each block's patch R with patch mean entropy on P0. Above about
   0.8, R is entropy again.
3. **Error signal without training (P0 only):** AUROC of each block's R for y_any, overall and inside the
   robust entropy deciles (§9).
4. Bottleneck ablation (b = d/16, d/8, d/4): repeat 1–3. Use it only to report sensitivity. The main b stays
   d/8.

## 9. Metrics

Computed on P0 CV (selection), then once on P1 and ACDC (all and per condition), pooled over frames, using
`sae.classify.metrics`, with these changes:

- **Primary:** wrong-pixel recall at a 10% global patch budget (unchanged). Also 5 / 20 / 30%.
- AUROC and AP for y50 and y_any (unchanged).
- **Robust within-entropy AUROC (replaces the v1 decile metric):** entropy deciles are fit on P0 as before.
  Average the AUROC only over deciles with at least 200 y50 positives on P0. In RESULTSV1 these are deciles
  4–9 (0-based); freeze this list in the config. Report the same for y_any over all deciles, and the
  positive-weighted mean, as descriptive rows.
- Drop "AUROC in the lowest-entropy quintile" as a reported number: it rests on 2 positives in P0.
- Descriptive: share of wrong pixels in patches below the P0 entropy median, per population. This bounds how
  much any within-entropy signal can move recall@budget.
- Pixel AUROC (descriptive, unchanged).

## 10. Pre-registered comparisons

Paired bootstrap over frames, 2,000 resamples, seed 0, 95% interval of the difference, CatBoost unless
stated.

Primary (the anomaly hypothesis; Holm correction over these three):
1. ACDC all, recall@10%: R − F0N.
2. ACDC all, robust within-entropy AUROC (y50): R − F0N.
3. P1, recall@10%: R − F0N.

Secondary (no correction; an interval at the edge counts as borderline):
4. ACDC all, recall@10%: ZR − F0.
5. P1, recall@10%: ZR − F0.
6. ACDC all, recall@10%: R − R_pca (does the nonlinear AE beat PCA).
7. P1, recall@10%: Z − F3F0 (do the codes keep the raw activations' signal).

Everything else is descriptive: the linear model, the per-condition ACDC numbers, 5 / 20 / 30% budgets,
R_only and the bottleneck ablation.

Expectation, stated in advance: most B0 errors on Cityscapes are ambiguous boundary and small-object pixels
that are in distribution, not anomalous. Earlier density-type scores (DDU, kNN, Mahalanobis) stayed at or
below entropy on Cityscapes. A null result on P1 is likely; ACDC is where R can help.

## 11. Steps and outputs

| Step | Module (proposed) | Output |
|---|---|---|
| 0. Data | `download.cityscapes --prefix leftImg8bit/train/`; `core.data`: forward `split` | Cityscapes train frames on disk |
| 1. Collect | `ae/collect.py` (from `sae.collect`, Cityscapes train) | `results/ae/acts/<block>.npy`, `_frame.npy`, `_norm.npz`, `meta.json` |
| 2. Fit encoders | `ae/model.py`, `ae/train.py` (AE), `ae/pca.py` | `results/ae/ckpt/<ae_b8 \| pca_b8>/<block>.pt`; `results/ae/quality.csv` (FVE per population, splice drop) |
| 3. Unsupervised checks | `ae/checks.py` | `results/ae/checks/`: R by population, Spearman with entropy, P0 AUROC per block |
| 4. Patch features | `ae/patch_features.py` (reuses `sae.hooks.pool_to_patches`, base tables from v1) | `results/ae/patches/<encoder>/<dataset>/<image_id>.npz` |
| 5. P0 CV | `ae/classify.py cv --features R --encoder ae_b8 --model catboost \| logreg` | `results/ae/runs/<set>_<encoder>_<model>/cv_*` |
| 6. Tests | `ae/classify.py test`, then `compare` | `scores.parquet`, `metrics.json`, `results/ae/compare.csv` |
| 7. Write-up | RESULTSV2.md | — |

`ae/classify.py` should reuse `sae.classify` (`load_meta`, `training_rows`, `metrics`, `paired_bootstrap`)
rather than copy it; only the feature loading and the model wrapper are new.

## 12. Risks

- **Bottleneck too loose:** the AE reconstructs well everywhere and R is flat. Check 1 in §8 catches this.
  The fallback is the smaller b from the ablation, chosen on the shift test (label-free) before any
  classifier.
- **R is just activation norm:** this is why F0N is the control in every primary comparison.
- **Cityscapes train looks too "normal":** if FVE already drops sharply from train to P0, R mostly separates
  "seen during B0 training" from "not seen". That is still consistent across the test sets, but worth
  reporting.
- **Stage-1 pooling:** mean over 16 positions can wash out a single anomalous position; max is included for
  this reason.
- **Correlated samples:** neighbouring patches are correlated. All intervals are bootstrapped over frames,
  never over patches.
