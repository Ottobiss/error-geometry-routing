# Report: SAE and AE experiments (branch `SAE`)

Written in ASD-STE100 (Simplified Technical English). Status: 2026-10-28. This report gives the SAE (v1) and
AE (v2) experiments of branch `SAE`. All sets use the same metrics (section 5). The values come from
`RESULTSV1.md`, `RESULTSV2.md` and `RESULTSv2.1.md`.

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

A hook check confirmed that the hooks do not change the B0 output (max |Δ logits| = 0, P1 mIoU 71.57).

## 3. SAE v1 (2026-10-26 and 2026-10-27)

### 3.1 Setup

- One BatchTopK SAE for each block: D = 2d, k = 8, Adam lr 1e-4, AuxK for dead latents.
- Fit on P0 activations: 2M positions for stages 1–2, all positions for stages 3–4. 5% of frames held out.
- Gates: dead latents < 10%, splice drop ≤ 1.0 mIoU (B0 with the SAE reconstruction in one block).
- Feature sets: F1 = F0 + SAE codes (5,494 columns), F2 = SAE codes only (5,483 columns).
- CatBoost iteration cap: 3,000.

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

### 3.3 Findings (P0 numbers in section 5.1)

1. The internal features do not change the global ranking. recall@10% stays at 0.566–0.569.
2. SAE codes are worse than raw activations (F2 < F3, F1 < F3F0).
3. The F1 gain in the Deciles column comes from deciles with very few positives. Thus, v2 uses the Robust
   column.
4. The 3,000-iteration cap stopped F2 and F3 too early. Their values are a little low.
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
- **Feature sets:** F0N = F0 + N; R = F0 + N + R; Z = F0 + Z; ZR = F0 + N + Z + R;
  F3F0NR = raw + F0 + N + R (descriptive, added after the tests).
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

### 4.4 Pre-registered comparisons

Holm correction over the three primary comparisons.

| Family | Comparison | Test | Metric | Δ [95% CI] | Result |
|---|---|---|---|---|---|
| primary | R − F0N | ACDC all | rec@10% | +0.007 [+0.002, +0.011] | pass |
| primary | R − F0N | ACDC all | Robust | +0.178 [+0.163, +0.193] | pass |
| primary | R − F0N | P1 | rec@10% | +0.004 [−0.001, +0.010] | fail |
| secondary | ZR − F0 | ACDC all | rec@10% | +0.026 [+0.021, +0.031] | CI does not include 0 |
| secondary | ZR − F0 | P1 | rec@10% | +0.004 [−0.002, +0.011] | CI includes 0 |
| secondary | R (AE) − R (PCA) | ACDC all | rec@10% | +0.012 [+0.009, +0.016] | CI does not include 0 |
| secondary | Z − F3F0 | P1 | rec@10% | +0.002 [−0.003, +0.008] | CI includes 0 |

### 4.5 Findings (numbers in section 5)

1. **On Cityscapes, the expected null result occurred.** No set changes rec@10% on P1. Only 1.9% of wrong
   pixels on P1 are below the median entropy. This limits all re-ranking inside entropy.
2. **On ACDC, R adds a large signal inside the same entropy level.** Robust increases from 0.56 to 0.74. It
   increases in all four conditions. This is the main result of v2.
3. **The pooled ACDC recall gain of R is small.** For each condition, R − F0N rec@10% is +0.3 (fog),
   −0.6 (night), −1.3 (rain) and −1.2 (snow) points. Thus, the pooled gain can come from a budget shift
   between conditions. We did not check this directly.
4. **The nonlinear AE is better than PCA under shift.** The two are equal on P0. On ACDC, the AE is better in
   rec@10% (+1.2 points) and Robust (0.74 against 0.67).
5. **AE codes keep the raw-activation signal.** Z is equal to F3F0 on P1, better on ACDC, and four times
   faster to fit.
6. **ZR is the best set for recall.** It is first on P1, fog, night and ACDC all.
7. **Rain and snow are different.** Sets with codes or raw activations give the best recall there. R alone is
   below entropy.
8. **Raw activations crowd out R.** In F3F0NR, CatBoost gives only 5% of importance to N and R (36% in set R).
   F3F0NR is equal to R and ZR on P1 and worse on ACDC.
9. **Entropy keeps the best global AUROC and AP at night.** The learned scores mostly re-rank inside the same
   entropy level.

## 5. Results: one metric set for all sets

### 5.0 Metrics

All tables use the v1 metric set (`RESULTSV1.md` §4.1, `RESULTSv2.1.md`) and one more column, Robust:

| Column | Meaning |
|---|---|
| rec@5% … rec@30% | Share of wrong pixels in the top 5 / 10 / 20 / 30% of patches. One budget for the full population. |
| AUROC y50, AP y50 | Patch is at least 50% wrong. |
| AUROC y_any, AP y_any | Patch is at least 10% wrong. |
| Deciles | y50 AUROC, mean over all P0 entropy deciles with positives (v1 metric). |
| Low ⅕ | y50 AUROC in the lowest-entropy fifth (v1 metric). |
| Robust | y50 AUROC, mean over P0 entropy deciles 4–9 only (v2 metric). |

How to compare:
- Use **rec@10%** for the global ranking. This is the primary metric of PLAN and PLANv2.
- Use **Robust** for the ranking inside the same entropy level. It replaces Deciles and Low ⅕ because these
  use deciles with very few positives:
  - Deciles: on P0, deciles 1–3 have 2, 28 and 95 positive patches.
  - Low ⅕: positives are P0 2, P1 68, fog 4, night 248, rain 0, snow 298, ACDC all 550. Read Low ⅕ only for
    night, snow and ACDC all.
- Robust is not available for the v1-only sets (F1, F2, F3), because v2 did not run them again. These sets
  have no tests.
- F0 and F3F0 are the same models in v1 and v2. Their scores are identical.

Row groups: control (entropy, F0, F0N); SAE v1 (F1, F2); raw activations (F3, F3F0); AE v2 (R, Z, ZR,
F3F0NR); PCA (R PCA).

### 5.1 P0 (out-of-fold CV)

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.333 | 0.560 | 0.818 | 0.936 | 0.907 | 0.283 | 0.952 | 0.744 | 0.616 | 0.944 | 0.567 |
| F0 | 0.348 | 0.566 | 0.823 | 0.939 | 0.914 | 0.307 | 0.953 | 0.743 | 0.676 | 0.881 | 0.662 |
| F0N | 0.349 | 0.567 | 0.822 | 0.940 | 0.916 | 0.308 | 0.953 | 0.739 | 0.728 | 0.915 | 0.692 |
| SAE F1 | 0.349 | 0.566 | 0.823 | 0.937 | 0.913 | 0.302 | 0.954 | 0.736 | 0.739 | 0.963 | — |
| SAE F2 | 0.267 | 0.463 | 0.733 | 0.886 | 0.875 | 0.169 | 0.924 | 0.603 | 0.723 | 0.940 | — |
| F3 | 0.283 | 0.483 | 0.753 | 0.901 | 0.889 | 0.193 | 0.930 | 0.641 | 0.769 | 0.970 | — |
| F3F0 | 0.350 | 0.569 | 0.825 | 0.942 | 0.919 | 0.317 | 0.955 | 0.739 | 0.814 | 0.994 | 0.738 |
| AE R | 0.352 | 0.571 | 0.828 | 0.949 | 0.923 | 0.319 | 0.953 | 0.730 | 0.819 | 0.988 | 0.766 |
| AE Z | 0.350 | 0.569 | 0.825 | 0.942 | 0.920 | 0.316 | 0.954 | 0.733 | 0.814 | 0.987 | 0.747 |
| AE ZR | 0.352 | 0.571 | 0.826 | 0.945 | 0.922 | 0.319 | 0.954 | 0.730 | 0.834 | 0.996 | 0.765 |
| AE F3F0NR | 0.350 | 0.571 | 0.827 | 0.943 | 0.920 | 0.321 | 0.955 | 0.738 | 0.821 | 0.987 | 0.752 |
| PCA R | 0.353 | 0.573 | 0.829 | 0.948 | 0.923 | 0.319 | 0.953 | 0.730 | 0.816 | 0.970 | 0.763 |

### 5.2 P1

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.329 | 0.552 | 0.813 | 0.933 | 0.902 | 0.298 | 0.950 | 0.743 | 0.570 | 0.610 | 0.571 |
| F0 | 0.348 | 0.563 | 0.817 | 0.935 | 0.909 | 0.326 | 0.949 | 0.729 | 0.545 | 0.404 | 0.638 |
| F0N | 0.343 | 0.560 | 0.815 | 0.936 | 0.909 | 0.323 | 0.949 | 0.727 | 0.591 | 0.468 | 0.652 |
| F3F0 | 0.348 | 0.561 | 0.820 | 0.940 | 0.914 | 0.327 | 0.951 | 0.729 | 0.762 | 0.795 | 0.722 |
| AE R | 0.347 | 0.564 | 0.820 | 0.939 | 0.917 | 0.334 | 0.948 | 0.717 | 0.722 | 0.611 | 0.743 |
| AE Z | 0.349 | 0.563 | 0.822 | 0.940 | 0.917 | 0.332 | 0.950 | 0.720 | 0.725 | 0.644 | 0.731 |
| AE ZR | 0.351 | 0.566 | 0.822 | 0.940 | 0.917 | 0.343 | 0.950 | 0.723 | 0.714 | 0.544 | 0.744 |
| AE F3F0NR | 0.347 | 0.562 | 0.821 | 0.940 | 0.915 | 0.332 | 0.951 | 0.728 | 0.750 | 0.715 | 0.733 |
| PCA R | 0.347 | 0.563 | 0.821 | 0.938 | 0.917 | 0.336 | 0.949 | 0.717 | 0.698 | 0.562 | 0.730 |

### 5.3 ACDC all

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.158 | 0.300 | 0.525 | 0.679 | 0.814 | 0.484 | 0.881 | 0.742 | 0.547 | 0.580 | 0.561 |
| F0 | 0.156 | 0.293 | 0.507 | 0.651 | 0.793 | 0.464 | 0.864 | 0.717 | 0.496 | 0.551 | 0.521 |
| F0N | 0.165 | 0.306 | 0.528 | 0.678 | 0.808 | 0.488 | 0.871 | 0.718 | 0.555 | 0.602 | 0.559 |
| F3F0 | 0.159 | 0.295 | 0.511 | 0.659 | 0.806 | 0.465 | 0.878 | 0.731 | 0.616 | 0.746 | 0.568 |
| AE R | 0.166 | 0.313 | 0.550 | 0.715 | 0.844 | 0.511 | 0.884 | 0.698 | 0.728 | 0.711 | 0.737 |
| AE Z | 0.163 | 0.304 | 0.526 | 0.680 | 0.821 | 0.485 | 0.882 | 0.724 | 0.647 | 0.686 | 0.642 |
| AE ZR | 0.175 | 0.319 | 0.546 | 0.702 | 0.833 | 0.515 | 0.884 | 0.716 | 0.686 | 0.698 | 0.687 |
| AE F3F0NR | 0.162 | 0.299 | 0.515 | 0.666 | 0.810 | 0.471 | 0.879 | 0.728 | 0.638 | 0.755 | 0.590 |
| PCA R | 0.160 | 0.301 | 0.525 | 0.685 | 0.824 | 0.483 | 0.876 | 0.696 | 0.688 | 0.747 | 0.669 |

### 5.4 ACDC fog

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.228 | 0.420 | 0.668 | 0.806 | 0.850 | 0.339 | 0.913 | 0.679 | 0.620 | 0.821 | 0.563 |
| F0 | 0.234 | 0.413 | 0.651 | 0.790 | 0.855 | 0.353 | 0.907 | 0.656 | 0.693 | 0.951 | 0.637 |
| F0N | 0.244 | 0.424 | 0.670 | 0.806 | 0.856 | 0.359 | 0.908 | 0.655 | 0.708 | 0.991 | 0.622 |
| F3F0 | 0.238 | 0.419 | 0.666 | 0.811 | 0.857 | 0.331 | 0.920 | 0.685 | 0.753 | 1.000 | 0.660 |
| AE R | 0.244 | 0.428 | 0.677 | 0.818 | 0.864 | 0.359 | 0.911 | 0.644 | 0.752 | 0.995 | 0.686 |
| AE Z | 0.241 | 0.424 | 0.674 | 0.820 | 0.864 | 0.336 | 0.917 | 0.663 | 0.762 | 0.997 | 0.688 |
| AE ZR | 0.240 | 0.428 | 0.679 | 0.822 | 0.863 | 0.339 | 0.915 | 0.652 | 0.757 | 0.999 | 0.683 |
| AE F3F0NR | 0.245 | 0.426 | 0.671 | 0.815 | 0.860 | 0.336 | 0.920 | 0.680 | 0.761 | 1.000 | 0.674 |
| PCA R | 0.243 | 0.426 | 0.675 | 0.813 | 0.860 | 0.363 | 0.911 | 0.654 | 0.738 | 0.995 | 0.667 |

### 5.5 ACDC night

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.087 | 0.171 | 0.330 | 0.473 | 0.767 | 0.645 | 0.834 | 0.805 | 0.550 | 0.550 | 0.552 |
| F0 | 0.085 | 0.167 | 0.319 | 0.457 | 0.727 | 0.611 | 0.797 | 0.777 | 0.405 | 0.581 | 0.343 |
| F0N | 0.090 | 0.174 | 0.328 | 0.471 | 0.747 | 0.632 | 0.811 | 0.779 | 0.502 | 0.633 | 0.438 |
| F3F0 | 0.085 | 0.165 | 0.318 | 0.457 | 0.738 | 0.614 | 0.811 | 0.784 | 0.552 | 0.777 | 0.443 |
| AE R | 0.085 | 0.168 | 0.327 | 0.475 | 0.760 | 0.630 | 0.811 | 0.756 | 0.619 | 0.730 | 0.554 |
| AE Z | 0.087 | 0.168 | 0.323 | 0.464 | 0.747 | 0.622 | 0.812 | 0.778 | 0.559 | 0.746 | 0.469 |
| AE ZR | 0.093 | 0.177 | 0.335 | 0.478 | 0.758 | 0.645 | 0.815 | 0.777 | 0.598 | 0.704 | 0.529 |
| AE F3F0NR | 0.086 | 0.166 | 0.319 | 0.459 | 0.742 | 0.617 | 0.814 | 0.783 | 0.565 | 0.767 | 0.458 |
| PCA R | 0.086 | 0.167 | 0.321 | 0.461 | 0.739 | 0.613 | 0.800 | 0.757 | 0.553 | 0.699 | 0.469 |

### 5.6 ACDC rain

Low ⅕ has no positives in rain (—).

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.228 | 0.422 | 0.682 | 0.825 | 0.875 | 0.423 | 0.927 | 0.744 | 0.544 | — | 0.579 |
| F0 | 0.244 | 0.432 | 0.687 | 0.827 | 0.883 | 0.460 | 0.923 | 0.736 | 0.610 | — | 0.692 |
| F0N | 0.236 | 0.431 | 0.697 | 0.844 | 0.887 | 0.450 | 0.925 | 0.725 | 0.630 | — | 0.699 |
| F3F0 | 0.247 | 0.438 | 0.694 | 0.836 | 0.887 | 0.446 | 0.931 | 0.748 | 0.746 | — | 0.724 |
| AE R | 0.229 | 0.418 | 0.694 | 0.851 | 0.899 | 0.429 | 0.918 | 0.663 | 0.787 | — | 0.801 |
| AE Z | 0.247 | 0.442 | 0.708 | 0.853 | 0.898 | 0.468 | 0.930 | 0.734 | 0.768 | — | 0.786 |
| AE ZR | 0.241 | 0.435 | 0.708 | 0.858 | 0.900 | 0.450 | 0.926 | 0.698 | 0.761 | — | 0.797 |
| AE F3F0NR | 0.253 | 0.446 | 0.701 | 0.843 | 0.892 | 0.457 | 0.932 | 0.744 | 0.770 | — | 0.759 |
| PCA R | 0.223 | 0.415 | 0.693 | 0.850 | 0.896 | 0.427 | 0.918 | 0.668 | 0.748 | — | 0.787 |

### 5.7 ACDC snow

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ | Robust |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.194 | 0.354 | 0.591 | 0.735 | 0.808 | 0.417 | 0.882 | 0.725 | 0.530 | 0.558 | 0.556 |
| F0 | 0.196 | 0.357 | 0.593 | 0.738 | 0.822 | 0.430 | 0.887 | 0.720 | 0.574 | 0.489 | 0.667 |
| F0N | 0.193 | 0.354 | 0.590 | 0.738 | 0.816 | 0.415 | 0.882 | 0.701 | 0.590 | 0.524 | 0.633 |
| F3F0 | 0.203 | 0.363 | 0.600 | 0.743 | 0.822 | 0.420 | 0.890 | 0.728 | 0.611 | 0.539 | 0.647 |
| AE R | 0.189 | 0.342 | 0.582 | 0.738 | 0.828 | 0.398 | 0.877 | 0.650 | 0.665 | 0.510 | 0.712 |
| AE Z | 0.198 | 0.360 | 0.604 | 0.758 | 0.832 | 0.422 | 0.891 | 0.711 | 0.636 | 0.471 | 0.711 |
| AE ZR | 0.198 | 0.356 | 0.602 | 0.756 | 0.834 | 0.413 | 0.887 | 0.677 | 0.664 | 0.530 | 0.721 |
| AE F3F0NR | 0.206 | 0.365 | 0.602 | 0.749 | 0.829 | 0.425 | 0.893 | 0.722 | 0.666 | 0.579 | 0.690 |
| PCA R | 0.189 | 0.350 | 0.592 | 0.745 | 0.832 | 0.409 | 0.882 | 0.668 | 0.687 | 0.573 | 0.719 |

## 6. Integration (`INTEGRATION.md`)

A scorer package for a router. Default set ZR. Time per frame on RTX 3060 Ti: B0 17 ms, R scorer 55 ms,
ZR scorer 68 ms, F3F0NR scorer 253 ms. The scorer predicts B0 errors, not the gain from a heavy model.

## 7. Limits

- One seed for each run.
- The AEs had not converged at epoch 30.
- SAE F1, F2 and F3 used the 3,000-iteration cap. F2 and F3 reached it.
- F3F0NR was added after we saw the test results.
- The secondary comparisons have no multiple-testing correction.

## 8. Open items

- Logistic-regression runs, R only, Z and ZR with PCA (PLANv2 §10).
- Bottleneck ablation: b = d / 4 and d / 16.
- Measure how the ACDC budget is divided between conditions for R and F0N (finding 3 in 4.5).
- Test whether R and ZR predict the B2/B5 gain, not only the B0 error.

## 9. Other experiments in the project (not in this report)

The other branches contain these series: baseline component routing (`baseline`), feature geometry and the
ACDC shift test (`geometry`), the frozen B0/B2 router (`experiments/frozen-router-results-20261025`), the
frozen SAE error head (`experiments/frozen-sae-errors-20261026`), B0 → B5 routing E0–E11
(`baseline-experiments`), the B5 context adapter (`experiments/frozen-context-adapter-20261027`), the spatial
B0 router (`results/spatial-router-acdc-20261028`) and HyperUL (`double_hyperul_segformer`).
