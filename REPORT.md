# Report: SAE and AE experiments (branch `SAE`)

Written in ASD-STE100 (Simplified Technical English). Status: 2026-10-28. This report gives the SAE (v1) and
AE (v2) experiments of branch `SAE`. The full numbers are in `RESULTSV1.md`, `RESULTSV2.md` and
`RESULTSv2.1.md`.

## 1. Question

SegFormer-B0 makes errors. Entropy of the B0 output finds most of them. The question: do the Mix-FFN
activations inside B0 find B0 errors that entropy does not find?

The unit is a 16 × 16 patch of the B0 output map. The target is the fraction of wrong valid pixels in the patch.

## 2. Common setup

| Item | Value |
|---|---|
| Model | `nvidia/segformer-b0-finetuned-cityscapes-512-1024`, frozen, fp32 |
| Resolution | Cityscapes 1024 × 512, ACDC 1024 × 576 |
| Activations | Input of `fc2` in the 8 encoder blocks (s1b1 … s4b2), 128 to 1,024 channels |
| Patch selection | Stages 1–2: max over the positions in the patch. Stages 3–4: the nearest position. |
| F0 (control) | 11 output features: entropy, MSP, margin, boundary fraction, row, column, majority class |
| Classifier | CatBoost, CrossEntropy on the error fraction, weight = valid pixels, depth 6, lr 0.05 |
| Fit and selection | Cityscapes P0 (300 val frames), GroupKFold(5) by frame |
| Tests | Cityscapes P1 (200 val frames) and ACDC val (406 frames: fog, night, rain, snow) |
| Statistics | Paired bootstrap over frames, 2,000 resamples, 95% CI |

Metrics:
- **recall@10%:** the share of wrong pixels in the top 10% of patches, with one budget for the full population.
- **Robust AUROC:** the y50 AUROC (patch at least 50% wrong) inside entropy deciles 4–9 of P0. Each of these
  deciles has 200 or more positive patches on P0. v2 uses this metric.
- **Decile AUROC (v1):** the same, but over all deciles. Some deciles have only 2–95 positives.

A hook check confirmed that the hooks do not change the B0 output (max |Δ logits| = 0, P1 mIoU 71.57).

## 3. SAE v1 (2026-10-26 and 2026-10-27)

### 3.1 Setup

- One BatchTopK SAE for each block: D = 2d, k = 8, Adam lr 1e-4, AuxK for dead latents.
- Fit on P0 activations: 2M positions for stages 1–2, all positions for stages 3–4. 5% of frames held out.
- Gates: dead latents < 10%, splice drop ≤ 1.0 mIoU (B0 with the SAE reconstruction in one block).

### 3.2 SAE quality

| Block | Epochs | FVE (held out) | Dead | Splice drop |
|---|---|---|---|---|
| s1b1 | 40 | 0.917 | 0.0% | 1.09 (fail) |
| s1b2 | 20 | 0.855 | 0.0% | 0.44 |
| s2b1 | 20 | 0.736 | 0.2% | 0.38 |
| s2b2 | 20 | 0.745 | 0.4% | 0.54 |
| s3b1 | 50 | 0.602 | 0.0% | 0.94 |
| s3b2 | 50 | 0.688 | 0.9% | 0.40 |
| s4b1 | 200 | 0.654 | 23.0% (fail) | 0.53 |
| s4b2 | 200 | 0.726 | 4.6% | 0.06 |

- The first run (5 epochs) was undertrained. All blocks failed the splice gate.
- We accepted the two borderline failures for v1. The s4b1 dead share uses only 7,680 held-out positions.

### 3.3 P0 cross-validation

| Set | Features | recall@10% | AUROC y50 | Decile AUROC (v1) |
|---|---|---|---|---|
| Entropy | 1 | 0.560 | 0.907 | 0.616 |
| F0 | 11 | 0.566 | 0.914 | 0.676 |
| F1: F0 + SAE codes | 5,494 | 0.566 | 0.913 | 0.739 |
| F2: SAE codes | 5,483 | 0.463 | 0.875 | 0.723 |
| F3: raw activations | 4,096 | 0.483 | 0.889 | 0.769 |
| F3F0: raw + F0 | 4,107 | 0.569 | 0.919 | 0.814 |

### 3.4 Findings

1. The internal features do not change the global ranking. recall@10% stays at 0.566–0.569.
2. SAE codes are worse than raw activations (F2 < F3, F1 < F3F0).
3. The F1 gain in decile AUROC comes from deciles with very few positives. In deciles with 200 or more
   positives, F1 = F0 (0.661 against 0.662). F3F0 keeps a gain (0.738).
4. The 3,000-iteration cap stopped F2 and F3 too early.
5. We did not do the P1 and ACDC tests for v1.

Decision: sparsity is for human interpretation, and it loses signal. v2 uses a dense bottleneck AE.

## 4. AE v2 (2026-10-27 and 2026-10-28)

### 4.1 Setup

- **AE:** one for each block, d → d (GELU) → b → d (GELU) → d, b = d / 8. AdamW lr 1e-3, cosine schedule,
  batch 4,096, 30 epochs.
- **PCA:** the same b, on the same data. This is the linear alternative.
- **Fit data:** Cityscapes train (2,975 frames, no labels). Thus, P0, P1 and ACDC are all out of sample for
  the encoders.
- **Patch features:**
  - Z: AE codes, 512 columns, mean over the patch;
  - R: reconstruction error ‖x − x̂‖² / d, 12 columns (mean and max for stages 1–2);
  - N: activation norm ‖x‖² / d, 12 columns. N is the control for R.
- **Classifier:** CatBoost as in v1, with an iteration cap of 10,000. No fold reached the cap.

### 4.2 Encoder quality

| Block | AE FVE (P0) | PCA FVE (P0) | AE FVE (ACDC night) | AE splice drop | PCA splice drop |
|---|---|---|---|---|---|
| s1b1 | 0.874 | 0.686 | 0.768 | 2.06 | 22.65 |
| s2b1 | 0.808 | 0.562 | 0.736 | 0.31 | 2.74 |
| s3b2 | 0.862 | 0.673 | 0.693 | 0.20 | 2.63 |
| s4b1 | 0.857 | 0.696 | 0.585 | 0.57 | 3.01 |
| s4b2 | 0.884 | 0.763 | 0.689 | 0.07 | 1.19 |

(5 of 8 blocks. The full table is in `RESULTSV2.md` §2.)

- There is no gap between Cityscapes train, P0 and P1. The encoders do not overfit.
- On ACDC, FVE decreases in all conditions. It decreases most at night in stage 4.
- The AE explains 12–25 points more variance than PCA. The AE splice drop is ≤ 2.1 mIoU.
- The AEs did not converge: the best epoch was the last epoch in all blocks.

### 4.3 Label-free checks

- **R detects the shift.** On ACDC, mean R of s4b2 is 2.17 times the P0 value (night: 2.40). N does not
  follow (night: 0.98).
- **R is not the same as entropy.** Spearman R vs entropy: 0.55–0.71.
- **Early-stage R is mostly the norm.** Spearman R vs N: 0.67–0.91 in stages 1–2.
- **Only stage 4 R has its own error signal.** P0 y_any AUROC inside entropy deciles: s4b2 0.629, s4b1 0.617,
  entropy 0.601.

### 4.4 Results

| Set | P0 CV recall@10% | P1 recall@10% | ACDC recall@10% | P0 CV robust AUROC | P1 robust AUROC | ACDC robust AUROC |
|---|---|---|---|---|---|---|
| Entropy | 0.560 | 0.552 | 0.300 | 0.567 | 0.571 | 0.561 |
| F0 | 0.566 | 0.563 | 0.293 | 0.662 | 0.638 | 0.521 |
| F0N | 0.567 | 0.560 | 0.306 | 0.692 | 0.652 | 0.559 |
| R (AE) | 0.571 | 0.564 | 0.313 | **0.766** | 0.743 | **0.737** |
| R (PCA) | **0.573** | 0.564 | 0.301 | 0.763 | 0.730 | 0.669 |
| Z (AE) | 0.569 | 0.563 | 0.304 | 0.747 | 0.731 | 0.642 |
| ZR (AE) | 0.571 | **0.566** | **0.319** | 0.765 | **0.744** | 0.687 |
| F3F0 | 0.569 | 0.561 | 0.295 | 0.738 | 0.722 | 0.568 |

Robust AUROC on ACDC for each condition (R − F0N): fog +0.06, night +0.12, rain +0.10, snow +0.08.

### 4.5 Pre-registered comparisons

Holm correction over the three primary comparisons.

| Family | Comparison | Test | Metric | Δ [95% CI] | Result |
|---|---|---|---|---|---|
| primary | R − F0N | ACDC | recall@10% | +0.007 [+0.002, +0.011] | pass |
| primary | R − F0N | ACDC | robust AUROC | +0.178 [+0.163, +0.193] | pass |
| primary | R − F0N | P1 | recall@10% | +0.004 [−0.001, +0.010] | fail |
| secondary | ZR − F0 | ACDC | recall@10% | +0.026 [+0.021, +0.031] | CI does not include 0 |
| secondary | ZR − F0 | P1 | recall@10% | +0.004 [−0.002, +0.011] | CI includes 0 |
| secondary | R (AE) − R (PCA) | ACDC | recall@10% | +0.012 [+0.009, +0.016] | CI does not include 0 |
| secondary | Z − F3F0 | P1 | recall@10% | +0.002 [−0.003, +0.008] | CI includes 0 |

### 4.6 Findings

1. **On Cityscapes, the expected null result occurred.** No set changes recall@10% on P1. Only 1.9% of wrong
   pixels on P1 are below the median entropy. This limits all re-ranking inside entropy.
2. **On ACDC, R adds a large signal inside the same entropy level.** Robust AUROC increases from 0.56 to 0.74.
   This is the main result of v2.
3. **The pooled ACDC recall gain of R is small.** For each condition, R − F0N recall@10% is +0.3 (fog),
   −0.6 (night), −1.3 (rain) and −1.2 (snow) points. Thus, the pooled gain can come from a budget shift
   between conditions. We did not check this directly.
4. **The nonlinear AE is better than PCA under shift.** The two are equal on P0. On ACDC, the AE is better in
   recall (+1.2 points) and robust AUROC (0.74 against 0.67).
5. **AE codes keep the raw-activation signal.** Z is equal to F3F0 on P1, better on ACDC, and four times
   faster to fit.
6. **ZR is the best set for recall.** It is first on P1, fog, night and ACDC all.
7. **Rain and snow are different.** Sets with codes or raw activations give the best recall there. R alone is
   below entropy.
8. **Entropy keeps the best global AUROC and AP at night.** The learned scores mostly re-rank inside the same
   entropy level.

## 5. Short items on branch `SAE`

- **v1 metrics on v2 runs (`RESULTSv2.1.md`):** with the v1 decile metric, ZR is first on P0 (0.834). F0 and
  F3F0 v2 scores are identical to v1. On P1, F0 is worse than entropy inside entropy levels (0.545 against
  0.570). Thus, the v1 "feeling of error" from F0 does not transfer.
- **F3F0NR, raw + F0 + N + R (descriptive, added after the tests):** equal to R and ZR on P1. Worse on ACDC
  (robust AUROC 0.59 against 0.74 for R). With raw columns, CatBoost gives only 5% of importance to N and R
  (36% in set R).
- **Integration (`INTEGRATION.md`):** a scorer package for a router. Default set ZR. Time per frame on
  RTX 3060 Ti: B0 17 ms, R scorer 55 ms, ZR scorer 68 ms, F3F0NR scorer 253 ms. The scorer predicts B0 errors,
  not the gain from a heavy model.

## 6. Limits

- One seed for each run.
- The AEs had not converged at epoch 30.
- F3F0NR was added after we saw the test results.
- The secondary comparisons have no multiple-testing correction.

## 7. Open items

- Logistic-regression runs, R only, Z and ZR with PCA (PLANv2 §10).
- Bottleneck ablation: b = d / 4 and d / 16.
- Measure how the ACDC budget is divided between conditions for R and F0N (finding 3).
- Test whether R and ZR predict the B2/B5 gain, not only the B0 error.

## 8. Other experiments in the project (not in this report)

The other branches contain these series: baseline component routing (`baseline`), feature geometry and the
ACDC shift test (`geometry`), the frozen B0/B2 router (`experiments/frozen-router-results-20261025`), the
frozen SAE error head (`experiments/frozen-sae-errors-20261026`), B0 → B5 routing E0–E11
(`baseline-experiments`), the B5 context adapter (`experiments/frozen-context-adapter-20261027`), the spatial
B0 router (`results/spatial-router-acdc-20261028`) and HyperUL (`double_hyperul_segformer`).
