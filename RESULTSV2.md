# RESULTS v2: bottleneck AE residuals for patch-level error prediction

Status as of 2026-10-27. This covers PLANv2.md steps 0–6 for the CatBoost runs: encoders, quality, label-free
checks, P0 CV, P1/ACDC tests and the pre-registered comparisons. **Not run yet:** the logistic-regression runs,
R_only, Z/ZR with PCA, and the bottleneck ablation (all descriptive in §10 of the plan). Code is at commit `023c7b1`
plus uncommitted `ae/` changes. Hardware and versions as in RESULTSV1.

## 1. Encoders (`ae.collect`, `ae.train`, `ae.pca`)

Fitted on Cityscapes train (2,975 frames): 1,999,200 positions per block (1,523,200 for s4), 5% of frames held
out. AE: d→d (GELU)→b→d (GELU)→d with b = d/8, AdamW lr 1e-3, wd 1e-4, cosine schedule, batch 4096, 30 epochs,
patience 3. PCA: the same b, fitted on the same normalised data.

| Block | d | b | AE steps | AE FVE held | PCA FVE held |
|---|---|---|---|---|---|
| s1b1 | 128 | 16 | 13,920 | 0.876 | 0.686 |
| s1b2 | 128 | 16 | 13,920 | 0.784 | 0.546 |
| s2b1 | 256 | 32 | 13,920 | 0.810 | 0.562 |
| s2b2 | 256 | 32 | 13,920 | 0.794 | 0.592 |
| s3b1 | 640 | 80 | 13,920 | 0.842 | 0.590 |
| s3b2 | 640 | 80 | 13,920 | 0.861 | 0.673 |
| s4b1 | 1024 | 128 | 10,620 | 0.858 | 0.696 |
| s4b2 | 1024 | 128 | 10,620 | 0.889 | 0.763 |

Training: early stopping never fired. Every AE ran all 30 epochs and the best epoch was the last one. Held-out
MSE was still going down when the cosine schedule ran out, so longer training would probably raise FVE a
little. The AE beats PCA by 12–25 FVE points at the same b.

## 2. Quality per population (`ae.quality`)

FVE, AE (PCA in brackets). Splice drop is in P0 mIoU points (P0 mIoU 70.97).

| Block | Train held | P0 | P1 | ACDC fog | night | rain | snow | Splice drop |
|---|---|---|---|---|---|---|---|---|
| s1b1 | 0.876 | 0.874 | 0.874 | 0.822 | 0.768 | 0.819 | 0.802 | 2.06 (22.65) |
| s1b2 | 0.784 | 0.781 | 0.781 | 0.759 | 0.748 | 0.753 | 0.738 | 1.38 (2.87) |
| s2b1 | 0.810 | 0.808 | 0.808 | 0.768 | 0.736 | 0.765 | 0.738 | 0.31 (2.74) |
| s2b2 | 0.794 | 0.792 | 0.791 | 0.795 | 0.746 | 0.784 | 0.755 | 0.61 (2.04) |
| s3b1 | 0.842 | 0.840 | 0.839 | 0.802 | 0.760 | 0.796 | 0.786 | 0.43 (2.07) |
| s3b2 | 0.861 | 0.862 | 0.859 | 0.814 | 0.693 | 0.800 | 0.773 | 0.20 (2.63) |
| s4b1 | 0.858 | 0.857 | 0.851 | 0.747 | 0.585 | 0.731 | 0.700 | 0.57 (3.01) |
| s4b2 | 0.889 | 0.884 | 0.880 | 0.811 | 0.689 | 0.783 | 0.771 | 0.07 (1.19) |

- **Cityscapes:** there is no gap between train and P0/P1, so the encoders do not overfit.
- **ACDC:** FVE drops on every condition. The drop is largest at night and in stages 3–4 (s4b1 0.86 → 0.59).
- **Splice drop:** the AE reconstructions keep the segmentation within 2 mIoU points everywhere. PCA does not
  (s1b1: 22.7 points).

## 3. Label-free checks (`ae.checks`, AE)

Mean R on each population, divided by its P0 mean:

| Column | P1 | fog | night | rain | snow | ACDC all |
|---|---|---|---|---|---|---|
| s1b1_r_mean | 1.01 | 1.33 | 1.00 | 1.62 | 2.12 | 1.51 |
| s3b2_r | 1.01 | 1.23 | 1.41 | 1.32 | 1.46 | 1.36 |
| s4b1_r | 1.04 | 1.52 | 2.08 | 1.65 | 1.85 | 1.78 |
| s4b2_r | 1.05 | 1.81 | 2.40 | 2.09 | 2.37 | 2.17 |
| s4b2_n (norm control) | 1.02 | 1.23 | 0.98 | 1.25 | 1.33 | 1.20 |

- **R tracks the shift:** it rises on ACDC, most at night in stage 4. The norm N does not follow, and at night it
  goes down.
- **Correlation with entropy and N (Spearman, valid P0 patches):**
  - R vs entropy: 0.55–0.71.
  - R vs N: 0.67–0.91 in stages 1–2, 0.30 for s3b2 and about 0.78 for stage 4. The early-stage R is largely a
    norm proxy.
- **P0 y_any AUROC, within entropy deciles:**
  - s4b2_r 0.629 and s4b1_r 0.617, above entropy's 0.601. The other R columns are at 0.48–0.55.
  - N columns are at 0.46–0.54.
  - So the only label-free R signal beyond entropy sits in stage 4.

## 4. P0 cross-validation (CatBoost, 5 group folds)

CatBoost v1 parameters with an iteration cap of 10,000. No fold reached the cap. "Final" is the iteration count
used for the model refit on all of P0.

| Run | Best iteration per fold | Final | CV time | recall@10% | AUROC y50 | AP y50 | Robust AUROC y50 | y_any decile AUROC |
|---|---|---|---|---|---|---|---|---|
| entropy | — | — | — | 0.5600 | 0.9070 | 0.2833 | 0.5668 | 0.6329 |
| F0 | 308, 290, 74, 742, 155 | 314 | 2 min | 0.5663 | 0.9135 | 0.3072 | 0.6623 | 0.6696 |
| F0N | 299, 556, 93, 730, 237 | 383 | 3 min | 0.5674 | 0.9156 | 0.3080 | 0.6920 | 0.7171 |
| R (AE) | 317, 834, 354, 739, 242 | 497 | — | 0.5710 | 0.9232 | 0.3192 | **0.7659** | 0.7812 |
| R (PCA) | 354, 856, 1115, 1083, 196 | 721 | 4 min | **0.5732** | 0.9233 | 0.3187 | 0.7625 | 0.7771 |
| Z (AE) | 1512, 1872, 280, 1547, 1164 | 1275 | 14 min | 0.5693 | 0.9199 | 0.3161 | 0.7470 | 0.7959 |
| ZR (AE) | 1465, 2403, 370, 1015, 873 | 1225 | 14 min | 0.5709 | 0.9221 | 0.3187 | 0.7646 | **0.8089** |
| F3F0 | 1339, 1482, 294, 445, 934 | 899 | 54 min | 0.5691 | 0.9189 | 0.3167 | 0.7378 | 0.8025 |

The R (AE) CV time is missing because that run was started by hand and its log is not under results/ae/logs.

### Learning

- **Stopping is noisy across folds:** fold 2 always stops early (74–1115) and fold 1 always late.
- **Feature-set size:** the scalar sets (F0, F0N, R) stop at 300–800 iterations. The code sets (Z, ZR, F3F0) need
  1,000–2,400.
- **Cost:** F3F0 (raw activations) costs four times as much as Z for the same result.

### Reading

- **Recall@10%:** on P0 it barely moves, from 0.560 for entropy to 0.567–0.573 for every set. The share of wrong
  pixels below the entropy median is 1.9% on P1, which bounds what any re-ranking inside entropy can gain.
- **Robust AUROC:** the within-entropy ranking is where the features differ:
  - R adds 0.07 over F0N.
  - F0N adds 0.03 over F0, so the norm alone is part of the gain.
  - AE and PCA residuals are equal on P0.

## 5. Tests: P1 and ACDC

Each P0 refit is scored on P1 (200 frames, 376k patches, y50 share 3.2%) and on ACDC val (y50 share: fog 7.2%,
night 40.6%, rain 8.9%, snow 11.7%, all 17.4%).

### recall@10%

| Run | P1 | fog | night | rain | snow | ACDC all |
|---|---|---|---|---|---|---|
| entropy | 0.5523 | 0.4201 | 0.1711 | 0.4221 | 0.3541 | 0.3001 |
| F0 | 0.5627 | 0.4131 | 0.1667 | 0.4325 | 0.3574 | 0.2929 |
| F0N | 0.5596 | 0.4241 | 0.1737 | 0.4309 | 0.3543 | 0.3064 |
| R (AE) | 0.5639 | 0.4275 | 0.1682 | 0.4181 | 0.3422 | 0.3133 |
| R (PCA) | 0.5635 | 0.4259 | 0.1670 | 0.4149 | 0.3502 | 0.3008 |
| Z (AE) | 0.5631 | 0.4239 | 0.1677 | 0.4425 | 0.3597 | 0.3040 |
| ZR (AE) | **0.5663** | **0.4283** | **0.1768** | 0.4351 | 0.3558 | **0.3191** |
| F3F0 | 0.5611 | 0.4193 | 0.1651 | 0.4385 | **0.3634** | 0.2948 |

### Robust within-entropy AUROC (y50, P0 deciles 4–9)

| Run | P1 | fog | night | rain | snow | ACDC all |
|---|---|---|---|---|---|---|
| entropy | 0.5714 | 0.5634 | 0.5521 | 0.5785 | 0.5564 | 0.5605 |
| F0 | 0.6383 | 0.6373 | 0.3431 | 0.6922 | 0.6673 | 0.5213 |
| F0N | 0.6522 | 0.6216 | 0.4377 | 0.6985 | 0.6331 | 0.5585 |
| R (AE) | 0.7429 | 0.6862 | **0.5540** | **0.8011** | 0.7117 | **0.7365** |
| R (PCA) | 0.7299 | 0.6667 | 0.4690 | 0.7874 | 0.7185 | 0.6687 |
| Z (AE) | 0.7310 | **0.6879** | 0.4690 | 0.7861 | 0.7113 | 0.6416 |
| ZR (AE) | **0.7437** | 0.6827 | 0.5290 | 0.7965 | **0.7205** | 0.6873 |
| F3F0 | 0.7220 | 0.6597 | 0.4434 | 0.7240 | 0.6468 | 0.5682 |

### Global AUROC / AP (y50)

| Run | P1 AUROC | P1 AP | ACDC all AUROC | ACDC all AP | night AUROC |
|---|---|---|---|---|---|
| entropy | 0.9020 | 0.2984 | 0.8137 | 0.4837 | **0.7666** |
| F0N | 0.9090 | 0.3227 | 0.8083 | 0.4879 | 0.7471 |
| R (AE) | 0.9169 | 0.3336 | **0.8443** | 0.5111 | 0.7601 |
| ZR (AE) | **0.9173** | **0.3433** | 0.8332 | **0.5152** | 0.7585 |

### Pixel AUROC (descriptive; patch score broadcast to pixels vs the per-pixel score)

| Run | P1 | fog | night | rain | snow | ACDC all |
|---|---|---|---|---|---|---|
| entropy (pixel) | 0.9336 | 0.8756 | 0.7779 | 0.8924 | 0.8242 | 0.8341 |
| R (AE) | 0.9118 | 0.8657 | 0.7646 | 0.8912 | 0.8308 | 0.8472 |
| ZR (AE) | 0.9130 | 0.8671 | 0.7634 | 0.8955 | 0.8383 | 0.8388 |

Pixel entropy stays ahead on Cityscapes and fog/night. That is expected: the patch score cannot rank pixels inside a patch.

## 6. Pre-registered comparisons (`ae.classify compare`)

Paired frame bootstrap, 2,000 resamples, seed 0. Holm correction over the three primaries.

| Family | Test | Metric | Comparison | Diff | 95% CI | p | p (Holm) | Result |
|---|---|---|---|---|---|---|---|---|
| primary | ACDC all | recall@10% | R − F0N | +0.0068 | [0.0024, 0.0110] | 0.002 | 0.004 | **significant** |
| primary | ACDC all | robust AUROC y50 | R − F0N | +0.1781 | [0.1632, 0.1928] | <0.001 | <0.001 | **significant** |
| primary | P1 | recall@10% | R − F0N | +0.0043 | [−0.0008, 0.0102] | 0.110 | 0.110 | not significant |
| secondary | ACDC all | recall@10% | ZR − F0 | +0.0262 | [0.0212, 0.0314] | <0.001 | — | excludes 0 (not borderline) |
| secondary | P1 | recall@10% | ZR − F0 | +0.0036 | [−0.0018, 0.0106] | 0.230 | — | includes 0 |
| secondary | ACDC all | recall@10% | R(AE) − R(PCA) | +0.0125 | [0.0091, 0.0156] | <0.001 | — | excludes 0 (not borderline) |
| secondary | P1 | recall@10% | Z − F3F0 | +0.0020 | [−0.0028, 0.0076] | 0.440 | — | includes 0 |

## 7. Reading

1. **The pre-registered expectation held.** Neither R nor ZR moves recall@10% on P1 measurably. On Cityscapes,
   B0 errors are in-distribution boundary and small-object errors, and only 1.9% of wrong pixels sit below the
   entropy median.
2. **On ACDC, R adds a large within-entropy signal.** Robust AUROC rises from 0.56 (F0N) to 0.74, and it rises
   inside every condition: fog +0.06, night +0.12, rain +0.10, snow +0.08 over F0N. This is the clearest result
   of v2. F0 and F0N rank *against* errors inside entropy deciles at night (0.34 and 0.44), and R repairs that.
3. **The pooled ACDC recall@10% gain for R (+0.7 points) is small and likely between conditions.** Per condition,
   R − F0N recall@10% is +0.3 (fog), −0.6 (night), −1.3 (rain) and −1.2 (snow) points. A positive pooled
   difference with mostly negative per-condition differences means the global budget moved between conditions.
   Night R is highest, so night patches (40% y50) likely got more of the budget. This is an inference from the
   numbers and has not been checked directly. It should be stated with the primary result, because it is not
   "R finds the wrong patches inside a frame".
4. **AE vs PCA.** The two are equal on P0 CV (PCA is even 0.002 ahead on recall). On ACDC the AE is clearly
   better: +1.25 recall points, and robust AUROC 0.74 vs 0.67, with the largest gap at night (0.55 vs 0.47). The
   nonlinear encoder generalises better under shift, which matches its much higher FVE and far smaller splice
   drop.
5. **The codes are enough.** Z matches F3F0 on P1 (+0.002, CI includes 0) and beats it on ACDC robust AUROC
   (0.64 vs 0.57), at a quarter of the CV cost.
6. **ZR is the best overall set.** It has the best recall@10% on P1, fog, night and ACDC all, and the best ACDC
   AP. Its within-entropy gain on ACDC is smaller than R's (0.69 vs 0.74), so the codes help the global ranking
   more than the within-entropy one.
7. **Rain and snow are the exception.** The sets with codes or raw activations (Z, ZR, F3F0) give the best
   recall there, and R alone is below entropy.

## 8. Open items

- Logistic-regression runs for the main sets (descriptive §10), R_only, and Z/ZR with PCA.
- Bottleneck ablation (`ae.train --div 16`, `--div 4`).
- Check point 3 directly: the share of the 10% ACDC-all budget spent per condition, R vs F0N.
- The AEs had not converged by epoch 30 (best epoch = last for all blocks). More epochs are an option, but they
  would change the frozen encoders, so they belong in a separate run.
- Log naming: `results/ae/logs/cv_ZR.log` holds the ZR (AE) run, and `cv_ZRencoderae_b8.log` is an empty
  leftover.
