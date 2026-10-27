# RESULTS v1: SAE training and P0 cross-validation

Status as of 2026-10-27. This covers PLAN.md steps 0–4: sanity check, activation collection, SAE training,
patch tables and CatBoost CV on Cityscapes P0. **No test population (P1, ACDC) has been scored yet.**
Code is at commit `24d4f74` plus uncommitted `sae/` changes. Hardware: RTX 3060 Ti 8 GB, 16 GB RAM,
torch 2.14.1+cu130, transformers 5.20.2, catboost 1.2.10.

## 0. Sanity check (`sae.hooks`)

| Check | Result |
|---|---|
| Logits with all 8 hooks vs without | max \|Δ\| = 0, argmax identical |
| fc2-input shapes (H/r × W/r × 4C) | all 8 blocks as expected (128 / 256 / 640 / 1024 channels) |
| P1 B0 mIoU with hooks | 71.57 (existing protocol: 71.57) |
| Patch → position overlay | `results/sae/sanity_overlay.png`, checked by eye |

## 1. Activations (`sae.collect`)

300 P0 frames; 15 of them (5%, seed 0) held out for SAE validation.

| Stage | Positions per frame | Total positions | Held-out positions |
|---|---|---|---|
| s1 (stride 4) | 6,666 (random sample) | 1,999,800 | 99,990 |
| s2 (stride 8) | 6,666 (random sample) | 1,999,800 | 99,990 |
| s3 (stride 16) | 2,048 (all) | 614,400 | 30,720 |
| s4 (stride 32) | 512 (all) | 153,600 | 7,680 |

Per-block normalisation (mean, plus a scale giving E‖x‖² = d) was fitted on the training frames only.

## 2. SAEs (`sae.train`), config D2k8

BatchTopK, D = 2·4C, k = 8, Adam lr 1e-4, batch 4096, AuxK (1/32, 512 latents, dead after 5 batches).
θ is the mean BatchTopK cut-off over training batches. All metrics below are on held-out frames, encoded with θ.

### 2.1 First run: 5 epochs for every block (rejected)

The SAEs were clearly undertrained. Held-out FVE was still gaining 2–5 points per epoch, and every block failed the splice gate.

| Block | Steps | FVE held | Dead | Splice drop |
|---|---|---|---|---|
| s1b1 | 2,320 | 0.788 | 0.0% | 3.99 |
| s1b2 | 2,320 | 0.679 | 0.0% | 1.49 |
| s2b1 | 2,320 | 0.608 | 0.0% | 1.07 |
| s2b2 | 2,320 | 0.615 | 0.0% | 1.35 |
| s3b1 | 715 | 0.415 | 0.1% | 1.78 |
| s3b2 | 715 | 0.507 | 7.7% | 2.99 |
| s4b1 | 180 | 0.322 | 48.5% | 3.87 |
| s4b2 | 180 | 0.344 | 61.3% | 4.52 |

Table: `results/sae/quality_5ep.csv`.

### 2.2 Final run

Epochs were raised so that every block gets about 7–9k steps or more. s1b1 was then retrained for 40 epochs to try to pass the splice gate.
The table below is the frozen D2k8 set used by all downstream steps (`results/sae/quality.csv`).
Splice drop is the fall in P0 mIoU (base 70.96) when the block's fc2 input is replaced by its SAE reconstruction.
Gates: dead < 10%, splice drop ≤ 1.0.

| Block | d → D | Epochs | Steps | FVE held | FVE train | L0 | Dead | Last-epoch FVE gain | Splice drop | Gates |
|---|---|---|---|---|---|---|---|---|---|---|
| s1b1 | 128 → 256 | 40 | 18,560 | 0.917 | 0.913 | 8.34 | 0.0% | +0.0002 | 1.09 | **splice ✗** |
| s1b2 | 128 → 256 | 20 | 9,280 | 0.855 | 0.852 | 8.26 | 0.0% | +0.0028 | 0.44 | ✓ |
| s2b1 | 256 → 512 | 20 | 9,280 | 0.736 | 0.733 | 8.23 | 0.2% | +0.0025 | 0.38 | ✓ |
| s2b2 | 256 → 512 | 20 | 9,280 | 0.745 | 0.748 | 8.12 | 0.4% | +0.0032 | 0.54 | ✓ |
| s3b1 | 640 → 1280 | 50 | 7,150 | 0.602 | 0.617 | 8.11 | 0.0% | +0.0010 | 0.94 | ✓ |
| s3b2 | 640 → 1280 | 50 | 7,150 | 0.688 | 0.705 | 8.19 | 0.9% | +0.0010 | 0.40 | ✓ |
| s4b1 | 1024 → 2048 | 200 | 7,200 | 0.654 | 0.701 | 8.90 | 23.0% | +0.0002 | 0.53 | **dead ✗** |
| s4b2 | 1024 → 2048 | 200 | 7,200 | 0.726 | 0.801 | 9.36 | 4.6% | +0.0001 | 0.06 | ✓ |

**Gate failures. Both are borderline and both are accepted for v1.**
- **s1b1 splice drop of 1.09 (limit 1.0).** At 20 epochs the drop was 1.61 (FVE 0.904; table in `results/sae/quality_s1b1_20ep.csv`). At 40 epochs FVE has plateaued at 0.917 (last-epoch gain +0.0002). More epochs won't close the gap; the remaining options are a larger D or k. Log: `results/sae/train_s1b1_40ep.log`.
- **s4b1 dead share of 23%.** This is measured on only 7,680 held-out positions, while D = 2,048. On all 146k training positions just 0.2% of latents never fire, so most of the "dead" latents are rare rather than dead. The 4.6-point train–held FVE gap at s4 is the largest of any stage and shows mild overfitting, which is expected with 154k positions.

## 3. Patch tables (`sae.patch_features`)

16×16 patches, about 2,048 per frame. Targets are err_frac = n_err / n_valid, with weight n_valid, where an error is a valid GT pixel with B0 argmax ≠ GT.

| Population | Frames | Non-zero SAE codes per patch (all 8 blocks) |
|---|---|---|
| Cityscapes (P0 + P1) | 500 | ≈ 172 |
| ACDC | 406 | ≈ 166 |

On P0 there are 569,346 valid patches:
- 24.3% have at least one wrong pixel;
- 13.7% have err_frac ≥ 0.1 (y_any);
- 3.0% have err_frac ≥ 0.5 (y50);
- the pixel error rate is 5.0%.

On disk the tables take about 16 GB (base, including the fp16 raw activations at 4,096 dimensions) plus 2.5 GB (CSR codes).

## 4. CatBoost CV on P0 (`sae.classify cv`)

Setup:
- CrossEntropy on err_frac, weighted by n_valid. Depth 6, lr 0.05, l2 3, at most 3,000 iterations, early stopping after 200 rounds without improvement. CPU, seed 0.
- GroupKFold(5) by frame.
- Training rows: every patch with an error, plus 25% of clean patches weighted ×4. That gives 246,107 of 569,346 patches.
- SAE latents firing in under 0.1% of P0 patches are dropped, leaving 5,483 of 8,192.
- Scores are out-of-fold over all 569,346 patches.

### 4.1 Metrics (out-of-fold, all P0 patches)

| Set | Features | recall@5% | recall@10% | recall@20% | recall@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | AUROC y50, averaged over entropy deciles | AUROC y50, lowest-entropy fifth |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 1 | 0.333 | 0.560 | 0.818 | 0.936 | 0.907 | 0.283 | 0.952 | 0.744 | 0.616 | 0.944 |
| F0 | 11 | 0.348 | 0.566 | 0.823 | 0.939 | 0.914 | 0.307 | 0.953 | 0.743 | 0.676 | 0.881 |
| F1 | 11 + 5,483 | 0.349 | 0.566 | 0.823 | 0.937 | 0.913 | 0.302 | 0.954 | 0.736 | 0.739 | 0.963 |
| F2 | 5,483 | 0.267 | 0.463 | 0.733 | 0.886 | 0.875 | 0.169 | 0.924 | 0.603 | 0.723 | 0.940 |
| F3 | 4,096 | 0.283 | 0.483 | 0.753 | 0.901 | 0.889 | 0.193 | 0.930 | 0.641 | 0.769 | 0.970 |
| F3+F0 | 4,107 | 0.350 | 0.569 | 0.825 | 0.942 | 0.919 | 0.317 | 0.955 | 0.739 | **0.814** | **0.994** |

Recall is the share of wrong pixels caught when the top x% of patches are flagged under a global budget.
The decile column averages the y50 AUROC computed inside each P0 entropy decile; 9 of the 10 deciles contain positives.

### 4.2 Iterations and run time

| Set | Best iteration per fold | Final iterations | Time per fold |
|---|---|---|---|
| F0 | 308, 290, 74, 742, 155 | 314 | 13–44 s |
| F1 | 912, **2967**, 251, 772, 861 | 1,153 | 3.5–23 min |
| F2 | **2995, 2992, 2936, 2994, 2993** | 2,982 | ≈ 18 min |
| F3 | **3000, 2993**, 1204, **2410, 2498** | 2,421 | 10–22 min |
| F3+F0 | 1339, 1482, 294, 445, 934 | 899 | 5–16 min |

Final iterations = the mean best iteration, rounded.

### 4.3 Reading

1. **Global ranking.** Internal features add almost nothing on top of the output features. F0, F1 and F3+F0 reach recall@10% of 0.566–0.569, against 0.560 for entropy alone. The B0 output features already settle most of the global ordering. F2 and F3 lack the output features and fall well below entropy.
2. **Ranking within an entropy level.** Here internal features clearly carry signal. Averaged over entropy deciles, AUROC goes from 0.616 (entropy) and 0.676 (F0) to 0.739 (F1) and 0.814 (F3+F0). The same holds for confidently predicted patches: in the lowest-entropy fifth, AUROC is 0.881 for F0, 0.963 for F1 and 0.994 for F3+F0. This is the "feeling of error" signal. On P0 it doesn't carry over into recall@budget, because errors at a fixed budget are dominated by high-entropy patches.
3. **SAE codes vs raw activations.** Codes do worse than raw activations: F2 0.723 vs F3 0.769, and F1 0.739 vs F3+F0 0.814. Possible causes include the SAE reconstruction loss (FVE 0.60–0.92), max-pooling of sparse codes across stride-4/8 positions, and the iteration cap (point 4).
4. **The iteration cap is binding.**
   - F2 hit the 3,000 cap in all five folds.
   - F3 came within about 600 iterations of it in four folds.
   - F1 reached 2,967 in one fold.

   So F2 and F3 are undertrained relative to F0 and F3+F0, which stopped early. That biases the pre-registered F2 vs F3 comparison, and part of the gap in point 3 may come from training budget rather than from the features.

## 5. Open items before the pre-registered tests

- Decide on the iteration cap: either re-run F1, F2 and F3 CV with a higher limit (e.g. 10,000) or freeze at 3,000 and record the limitation.
- Freeze the configuration and write the EXPERIMENTS.md pre-registration entry before the first P1/ACDC score.
- Then run `test` for F0, F1, F2 and F3 (P1, then ACDC) and `compare` (paired frame bootstrap, 2,000 resamples).
- Deferred: ablations (PLAN §8), interpretation (§9), routing (§10).

## Artifacts

| Path | Content |
|---|---|
| `results/sae/sanity_overlay.png` | Patch-to-position overlay |
| `results/sae/acts/` | Activations, normalisation, meta.json |
| `results/sae/ckpt/D2k8/<block>.pt` | SAE weights, θ, normalisation, training history |
| `results/sae/quality.csv`, `quality_5ep.csv`, `quality_s1b1_20ep.csv` | SAE quality tables |
| `results/sae/patches/` | Patch tables (base and D2k8 codes) |
| `results/sae/runs/<set>/cv_*` | CV config, out-of-fold scores, metrics |
| `results/sae/logs/cv_<set>.log` | CV logs |
