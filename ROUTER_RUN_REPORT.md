# Report: Z / R / ZR scorers in the main router (external run, 2026-10-29)

Written in ASD-STE100 (Simplified Technical English).

## 1. Scope

This report tells you what an external experiment did with the scorers of branch `SAE`. It also gives the
results and the limits of the experiment. A different person did the run in Google Colab. We did not run it
again. All numbers come from the saved outputs of the run (section 2).

## 2. Source materials

| Material | Content |
|---|---|
| `ottobiss_z_r_zr_final_metrics_(1).ipynb` | The Colab notebook: text, code and cell outputs. |
| `archive stuff/source/` | The code that the run used: `experiment.py`, `router.py`, `scorer.py`, `sae_lib.py`, `features/`, `p0_ids.txt`, and a copy of [INTEGRATION.md](INTEGRATION.md) (`guide.md`). |
| `archive stuff/budget_sweep/`, `archive stuff/entropy_deciles/` | The scripts of the two extra analyses. |
| `archive stuff/results/` | The saved outputs: protocol, thresholds, mIoU tables, bootstrap intervals, timing, confusion matrices, logs. |
| `archive stuff/artifacts/` | The 8 AE checkpoints of `ae_b8`. The run used them without changes. |

The level counts in section 6.4 come from the saved `results/blocked_confusions.npz` (array `levels`). The run
did not save these counts as a table. All other numbers are in the saved CSV and JSON files.

## 3. Experiment setup

### 3.1 Question

The run asks this question: is the frame mean of a Z, R or ZR score map a better signal for the main router
than the kNN distance?

### 3.2 The router

The router comes without changes from branch `main`, commit `8eb70ca`. It gives each frame one of four levels:

| Level | What runs | B5 area cost |
|---|---|---|
| light | B0 only | 0 |
| ens@0.5 | mean of B0 and B5 probabilities, B5 on a 0.5× frame | 0.25 |
| ens@0.75 | the same, B5 on a 0.75× frame | 0.5625 |
| heavy | B5 on the full working frame | 1 |

The router uses two scores for each frame:

- **Flag.** If the flag is above the 95th percentile (q95) of clean frames, the frame gets ens@0.75. If the
  credit is too low, the frame gets ens@0.5 or light.
- **Priority.** If the priority is in the top share of the last 50 frames, the frame gets heavy. The top share
  is the budget that stays after the flagged frames.

A credit keeps the mean B5 area cost near the budget. Each frame adds the budget to the credit, and each level
removes its cost. The budget is 0.5.

NOTE: Only a flag can put a frame at ens@0.5 or ens@0.75. A frame without a flag gets light or heavy.

### 3.3 The five variants

The priority is the mean B0 entropy for all variants. Only the flag changes.

| Variant | Flag |
|---|---|
| `entropy` | no flag |
| `knn_entropy` | kNN cosine distance of the B0 decoder features to a bank of clean features (the main baseline) |
| `Z_entropy` | mean of the Z score map over all patches of the frame |
| `R_entropy` | mean of the R score map |
| `ZR_entropy` | mean of the ZR score map |

The run also gives two reference points: B0 alone and B5 alone.

### 3.4 Models

- The run used the 8 AE checkpoints and the Z, R and ZR CatBoost models of branch `SAE`. It did not train them
  again. It did not use F3F0NR.
- B0 and B5 run in fp32. Both get the frame at the working resolution: Cityscapes 1024 × 512, ACDC 1024 × 576.
- One B0 pass with the hooks gives the features for all three scorers. The Z and R CatBoost models use the
  features of the ZR pass.

### 3.5 Data

| Set | Frames | Use |
|---|---|---|
| Fit | 267 Cityscapes val frames, all of Frankfurt | kNN bank (20,000 vectors, k = 10, seed 0) and the q95 flag thresholds. No labels. |
| Test, Cityscapes | 92 val frames: Lindau 28, Munster 64 | Test. All 300 P0 frames are removed. |
| Test, ACDC | 406 val frames: fog 100, night 106, rain 100, snow 100 | Test. |

The pool is the 498 test frames.

The frames go through the router as one stream. The **blocked** order is Cityscapes, fog, night, rain, snow.
The run also uses 5 random orders (`shuffle0` … `shuffle4`).

The fitted flag thresholds are:

| Flag | q95 on the fit set |
|---|---|
| kNN | 0.1654 |
| Z | 0.0979 |
| R | 0.0983 |
| ZR | 0.0984 |

### 3.6 Metrics

- **mIoU:** from the sum of the confusion matrices of all frames, not the mean of frame mIoU.
- **B5 area cost:** the mean of the level costs (scale²). It is not latency.
- **Gap closed:** 100 × (mIoU − mIoU of B0) / (mIoU of B5 − mIoU of B0).
- **Intervals:** paired bootstrap, 2,000 resamples, by sequence and by frame, no correction for multiple
  comparisons. The routing decisions stay fixed inside the bootstrap.
- **Time:** the full time from the PNG decode on the CPU to the mask on the CPU. The run measures 40 frames
  (8 from each test set), 2 repeats each, on a Tesla T4.

The primary comparison was fixed before the run: `ZR_entropy` minus `knn_entropy`, blocked order, pool.

## 4. Run log

1. The run installed the package versions of [INTEGRATION.md](INTEGRATION.md): Python 3.14, torch 2.14.1+cu130,
   transformers 5.20.2, catboost 1.2.10.
2. It downloaded the AE checkpoints, the CatBoost models and the reference `scores.parquet` files.
3. It did the acceptance test of INTEGRATION.md §9. **The test failed** (section 5).
4. The run then started a second study, `ottobiss_sae_local_diagnostic_v1`. The code of this study is the same
   as before, with two changes:
   - The gate now accepts the run only if the saved acceptance record shows the failure.
   - All output tables have the columns `reference_reproduced = False` and
     `evaluation_mode = local_scorer_after_failed_acceptance`.
5. **Fit** (2.1 min): the kNN bank and the four q95 thresholds on the 267 Frankfurt frames.
6. **Cache** (6.4 min): for each test frame, the run saved the confusion matrices of all four levels and all
   flag and priority scores.
7. **Analysis:** mIoU for 6 orders and 7 domains, and the bootstrap intervals.
8. **Timing** (2.8 min): 40 frames with the live path of each variant. The run makes sure that the live level
   and the live mask are the same as in the cache.
9. **Budget sweep:** budgets 0.1, 0.25, 0.5, 0.75 and 1.0 from the cache, with live timing for each point.
10. **Entropy deciles:** mIoU inside 10 bins of patch B0 entropy, at budget 0.5.
11. The run put all outputs into one archive. This archive is the `archive stuff` folder.

## 5. Acceptance failure

### 5.1 Data

The test compares the local Z, R and ZR scores with the reference `scores.parquet`. The limit is a maximum
difference of 1e-6.

| Frame | Set | Maximum difference | Mean difference |
|---|---|---|---|
| `frankfurt_000000_000576` | Z / R / ZR | 0.0046 / 0.0109 / 0.0031 | 3e-5 / 3e-5 / 2e-5 |
| `fog_GOPR0476_frame_000761` | Z / R / ZR | 0.0185 / 0.0108 / 0.0173 | 5e-5 / 5e-5 / 4e-5 |
| `night_GOPR0351_frame_000033` | Z / R / ZR | 0.0400 / 0.0433 / 0.0454 | 9e-4 / 9e-4 / 6e-4 |
| `rain_GOPR0400_frame_000058` | Z / R / ZR | 0.0202 / 0.0359 / 0.0540 | 2e-4 / 3e-4 / 3e-4 |
| `snow_GOPR0122_frame_000001` | Z / R / ZR | 0.0143 / 0.0327 / 0.0166 | 2e-4 / 2e-4 / 1e-4 |

The hooks did not change the B0 output. The run checked this on each frame.

### 5.2 Probable cause

This cause is probable. We did not do a test to prove it.

- The run used a Tesla T4. The reference scores come from an Ampere GPU (INTEGRATION.md §10 gives an
  RTX 3060 Ti).
- On Ampere GPUs, PyTorch runs cuDNN convolutions in TF32 by default (`torch.backends.cudnn.allow_tf32 = True`).
  The repository does not change this setting. The T4 does not have TF32. Thus, it calculates the
  convolutions of B0 in true fp32.
- The activations are then a little different. CatBoost uses thresholds on the features. Thus, a small
  difference can move a patch to a different leaf and change its score by a larger value.
- The difference is largest on the night frame. This frame is far from the training data.

The INTEGRATION.md result (≤ 3e-8) comes from the same machine that made the reference scores. Thus, it does not
show that the scores are the same on a different GPU.

### 5.3 Effect on this experiment

The effect is small. The flag is the mean of the score map over approximately 2,000 patches. The mean
differences are 2e-5 to 9e-4, and the thresholds are near 0.098. Only a frame very near a threshold can get a
different flag.

## 6. Results

### 6.1 Main results (blocked order, budget 0.5)

| Variant | Pool mIoU | ACDC mIoU | Pool B5 cost | ACDC B5 cost | Pool gap closed | ACDC gap closed |
|---|---|---|---|---|---|---|
| B0 alone | 53.30 | 47.27 | 0 | 0 | 0% | 0% |
| `entropy` | 59.17 | 53.77 | 0.422 | 0.416 | 63.0% | 63.6% |
| `knn_entropy` | 60.76 | 55.68 | 0.472 | 0.484 | 80.0% | 82.3% |
| `Z_entropy` | 60.54 | 55.53 | 0.439 | 0.443 | 77.7% | 80.9% |
| `R_entropy` | **60.82** | **55.93** | 0.462 | 0.473 | **80.6%** | **84.8%** |
| `ZR_entropy` | 60.51 | 55.49 | 0.437 | 0.441 | 77.4% | 80.5% |
| B5 alone | 62.62 | 57.49 | 1 | 1 | 100% | 100% |

### 6.2 mIoU for each domain (blocked order, budget 0.5)

| Variant | Cityscapes | fog | night | rain | snow |
|---|---|---|---|---|---|
| B0 alone | 68.23 | 64.43 | 22.99 | 50.32 | 50.03 |
| `entropy` | 70.88 | 70.87 | 28.50 | 56.87 | 53.81 |
| `knn_entropy` | **71.35** | **71.59** | 32.46 | **57.20** | 56.21 |
| `Z_entropy` | 70.75 | 71.03 | 32.19 | 55.82 | **56.97** |
| `R_entropy` | 70.75 | 70.49 | **33.09** | 56.95 | 56.66 |
| `ZR_entropy` | 70.75 | 71.00 | 32.19 | 55.83 | 56.91 |
| B5 alone | 73.41 | 73.07 | 34.56 | 60.13 | 57.89 |

### 6.3 Paired comparisons (blocked order, budget 0.5)

The values are mIoU differences in percentage points, with 95% intervals. The unit is the sequence. The pool
has 113 sequences: 92 Cityscapes frames (one frame per sequence) and 21 ACDC sequences.

| Comparison | Pool | ACDC all |
|---|---|---|
| `ZR_entropy` − `knn_entropy` (primary) | −0.25 [−0.98, +0.32] | −0.18 [−0.88, +0.42] |
| `Z_entropy` − `knn_entropy` | −0.22 [−0.98, +0.37] | −0.15 [−0.88, +0.50] |
| `R_entropy` − `knn_entropy` | +0.06 [−0.59, +0.50] | +0.25 [−0.34, +0.75] |
| `knn_entropy` − `entropy` | +1.59 [−0.39, +3.19] | +1.91 [−0.26, +3.75] |
| `Z_entropy` − `entropy` | +1.37 [−0.66, +2.93] | +1.76 [−0.41, +3.52] |
| `R_entropy` − `entropy` | +1.64 [−0.52, +3.35] | +2.16 [−0.23, +4.18] |
| `ZR_entropy` − `entropy` | +1.34 [−0.67, +2.88] | +1.73 [−0.38, +3.46] |

With the frame as the unit, the primary interval on the pool is [−0.95, +0.32].

The primary comparison in the 5 random orders (pool, sequence unit):

| Order | `ZR_entropy` − `knn_entropy` |
|---|---|
| shuffle0 | −0.72 [−2.01, +0.38] |
| shuffle1 | −0.70 [−2.10, +0.38] |
| shuffle2 | −0.77 [−2.18, +0.27] |
| shuffle3 | −0.49 [−1.88, +0.58] |
| shuffle4 | −0.70 [−2.14, +0.27] |

All intervals contain 0. The difference is negative in all 6 orders.

Pool mIoU for each order:

| Variant | blocked | shuffle0 | shuffle1 | shuffle2 | shuffle3 | shuffle4 |
|---|---|---|---|---|---|---|
| `entropy` | 59.17 | 59.14 | 58.48 | 59.95 | 59.70 | 58.61 |
| `knn_entropy` | 60.76 | 60.44 | 60.71 | 60.75 | 60.32 | 60.59 |
| `Z_entropy` | 60.54 | 60.22 | 60.51 | 60.46 | 60.31 | 60.34 |
| `R_entropy` | 60.82 | 60.30 | 60.85 | 60.95 | 60.55 | 60.61 |
| `ZR_entropy` | 60.51 | 59.72 | 60.01 | 59.98 | 59.82 | 59.89 |

The order changes `entropy` by 1.5 pp. This is more than the differences between the flag variants.

### 6.4 Router levels for each domain (blocked order, budget 0.5)

Each cell gives the number of frames at the levels light / ensemble (ens@0.5 + ens@0.75) / heavy. Only a flag
can put a frame at an ensemble level (section 3.2).

| Domain (frames) | `entropy` | `knn_entropy` | `Z_entropy` | `R_entropy` | `ZR_entropy` |
|---|---|---|---|---|---|
| Cityscapes (92) | 51 / 0 / 41 | 49 / 10 / 33 | 52 / 3 / 37 | 52 / 3 / 37 | 52 / 3 / 37 |
| fog (100) | 56 / 0 / 44 | 38 / 37 / 25 | 55 / 4 / 41 | 48 / 19 / 33 | 53 / 8 / 39 |
| night (106) | 56 / 0 / 50 | 0 / 104 / 2 | 0 / 104 / 2 | 0 / 104 / 2 | 0 / 104 / 2 |
| rain (100) | 70 / 0 / 30 | 19 / 79 / 2 | 40 / 49 / 11 | 16 / 83 / 1 | 41 / 47 / 12 |
| snow (100) | 55 / 0 / 45 | 5 / 92 / 3 | 33 / 47 / 20 | 18 / 73 / 9 | 37 / 41 / 22 |

On the 267 fit frames, the frame means of Z and ZR have a correlation of 0.995 (saved `fit_scores.npz`).

### 6.5 Budget sweep (blocked order)

mIoU for each nominal budget:

| Variant | Pool 0.1 | Pool 0.25 | Pool 0.5 | Pool 0.75 | ACDC 0.1 | ACDC 0.25 | ACDC 0.5 | ACDC 0.75 |
|---|---|---|---|---|---|---|---|---|
| `entropy` | 55.17 | 55.96 | 59.17 | 60.37 | 49.10 | 49.98 | 53.77 | 55.01 |
| `knn_entropy` | 55.46 | 59.06 | 60.76 | 61.86 | 49.80 | 53.60 | 55.68 | 56.84 |
| `Z_entropy` | 55.63 | **59.34** | 60.54 | 61.62 | **49.82** | 53.91 | 55.53 | 56.58 |
| `R_entropy` | 54.76 | 59.19 | **60.82** | **61.95** | 49.01 | 53.75 | **55.93** | **57.00** |
| `ZR_entropy` | **55.73** | 58.98 | 60.51 | 61.67 | 49.77 | **54.06** | 55.49 | 56.66 |

At budget 1.0, all variants are equal to B5 alone. No variant is the best at all budgets.

### 6.6 Time (Tesla T4, budget 0.5)

| Path | Mean time | p95 |
|---|---|---|
| B0 alone | 101 ms | 126 ms |
| `entropy` | 249 ms | 379 ms |
| `knn_entropy` | 263 ms | 386 ms |
| `R_entropy` | 315 ms | 427 ms |
| `Z_entropy` | 322 ms | 451 ms |
| `ZR_entropy` | 332 ms | 462 ms |
| B5 alone | 323 ms | 346 ms |

The B0-alone value comes from the budget sweep. The other values come from `timing.csv`.

- The scorers add 52–69 ms to the kNN path.
- On the T4, the `ZR_entropy` path is slower than B5 alone. Its pool mIoU is 2.1 pp lower.
- The time includes the PNG decode and the resize on the CPU. This part is a large share of the 101 ms of B0
  alone.

### 6.7 Entropy deciles (budget 0.5)

The run put each 16 × 16 patch into one of 10 bins of mean B0 entropy. The bin limits come from the 267 clean
Frankfurt frames. Then it calculated mIoU in each bin and the equal-weight mean over the bins.

| Variant | Pool mean of bins | ACDC mean of bins |
|---|---|---|
| B0 alone | 68.34 | 63.08 |
| `entropy` | 70.90 | 64.65 |
| `knn_entropy` | 71.33 | 65.91 |
| `Z_entropy` | 71.02 | 65.72 |
| `R_entropy` | 71.09 | 65.83 |
| `ZR_entropy` | 71.02 | 65.72 |
| B5 alone | 72.37 | 66.34 |

The flag variants are almost equal in each bin. The router selects a level for the full frame. Thus, this
analysis does not test how the scorers rank patches.

## 7. Interpretation

1. **As a frame flag, ZR is equal to kNN or a little worse.** The primary difference is −0.25 pp, and the
   interval contains 0. In the random orders, the difference is −0.5 to −0.8 pp.
2. **R is the only scorer that is equal to kNN in all orders.**
3. **Z and ZR give almost the same flag.** Their frame means have a correlation of 0.995. Their levels are almost
   the same in all domains.
4. **Night is easy for all flags.** Each flag variant puts 104 of 106 night frames at an ensemble level. Thus,
   the flag variants are almost equal on night.
5. **Z and ZR react less to fog, rain and snow than kNN.** In rain and snow, they put approximately half as many
   frames at an ensemble level as kNN. In fog, they put 4–8 frames there, and kNN puts 37. The CatBoost score predicts the B0 error fraction, and it learned from Cityscapes only. Under
   domain shift it increases less than the kNN distance. The mean over the full frame makes the signal weaker
   again. R reacts more, and its counts are near the kNN counts.
6. **The equal budget does not give an equal cost.** `ZR_entropy` used 0.437 of B5 area, and `knn_entropy` used
   0.472. Thus, a part of the mIoU difference comes from less B5 use.
7. **The scorers add time.** On the T4, they add 52–69 ms for each frame.

## 8. Limits of the experiment

CAUTION: The experiment does not use the patch information of the scorers. It reduces each score map to one
frame mean, and the priority is B0 entropy for all variants. The main result of branch `SAE` is the ranking of
patches inside the same entropy level (robust AUROC, recall@k). This experiment does not test it.

- **Acceptance.** The local scores are not the same as the reference scores (section 5). The effect on the flags
  is probably small.
- **Thresholds.** 159 of the 267 Frankfurt fit frames are in P0. CatBoost learned from P0. Thus, the q95
  thresholds of Z, R and ZR come partly from training frames.
- **Intervals for each condition.** The sequence bootstrap has only 3 groups for fog, 2 for night and 8 for rain
  and snow. These intervals are not reliable. The pool and ACDC-all intervals are better.
- **B5 resolution.** B5 gets the frame at the working resolution (1024 × 512 or 1024 × 576), not at full
  resolution. Thus, the B0-to-B5 gap is only 9.3 pp on the pool (10.2 pp on ACDC).
- **Test size.** Only 92 Cityscapes frames are in the test, because all P0 frames are removed.
- **Time.** The measurement uses 40 frames, 2 repeats, on one GPU type (T4). The CPU decode is in the
  measurement.
- **Reuse of data.** The notebook says that the Cityscapes and ACDC frames were used in earlier studies. Thus,
  the results are exploratory.

## 9. Conclusion

In the main router, the frame mean of the ZR score is not better than the kNN distance as a domain-shift flag.
The R score is equal to kNN. Z and ZR react too little to fog, rain and snow. All three scorers add 52–69 ms on a
T4. The experiment does not tell if the scorers help routing at patch or tile level.

## 10. Possible next steps

1. **Check the acceptance cause.** On the Ampere GPU, score `frankfurt_000000_000576` with
   `torch.backends.cudnn.allow_tf32` set to `True` and then to `False`. Compare both with `scores.parquet`.
2. **Update the acceptance test in INTEGRATION.md.** Write the GPU type and the TF32 setting of the reference
   scores. Give a tolerance that is correct for a different GPU, or a rank test.
3. **Test the patch information.** Use ZR in a router that runs B5 on selected tiles or regions, or use the ZR
   frame score as the priority in place of entropy. Both tests need a new GPU run, because the archive does not
   contain the cache of each frame.
