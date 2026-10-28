# Project report: error geometry routing

Written in ASD-STE100 (Simplified Technical English). Status: 2026-10-28. This report gives a short summary of
all experiments in the repository, on all branches. The source documents have the full numbers (section 15).

## 1. Project question

A light model (SegFormer-B0) segments all frames. A heavy model (SegFormer-B2 or B5) is more accurate but slow.
The project tries to find where the heavy model must run. It also tries to find where B0 makes errors.

The main question: do signals from the inside of B0 (feature geometry, activations, autoencoders) give
more than the entropy of the B0 output?

Terms in this report:
- **Entropy:** the entropy of the B0 softmax. It is the baseline in all series.
- **Oracle:** a selector that uses the ground truth. It shows the upper limit.
- **Δ [95% CI]:** a difference and its 95% interval from a paired bootstrap over frames.
- **CI > 0:** the interval does not include 0.

## 2. Data and splits

| Split | Frames | Used for |
|---|---|---|
| Cityscapes P0 | 300 val | Fit and selection (series 3–6, 11, 13, 14) |
| Cityscapes P1 | 200 val | Test, in domain (series 3–7, 11, 14) |
| Cityscapes train / internal val | 240 / 60 val frames | Fit / selection (series 7, 8) |
| Cityscapes `cs_fit` / `cs_test` | 267 (Frankfurt) / 233 (Lindau, Münster) | Fit / test (series 9, 10) |
| Cityscapes train | 2,975 | AE fit only, no labels (series 14) |
| ACDC val | 406: fog 100, night 106, rain 100, snow 100 | Test under domain shift |

CAUTION: The series use different heavy models (B2 or B5), splits, tile sizes and metrics. Do not compare
numbers between series directly.

## 3. Baseline: components and learned fields (2026-10-22, branch `baseline`)

The unit is a connected component of the B0 prediction. A router sends components to B2 under a pixel budget.

- Entropy on P1: 76.73 mIoU at 10% of pixels and 77.76 at 20%. B0 only: 71.57. B2 only: 79.95.
- Oracle at 20%: 80.35. Random at 20%: 75.91.
- Learned failure fields (p-fields, MetaSeg) are equal to entropy: Δ +0.27 [−0.28, +1.00] and −0.02 [−0.19, +0.15].

## 4. Feature geometry of B0 (2026-10-23, branch `geometry`)

We tried these component scores on P1 at a 20% budget, compared with entropy:

| Method | Δ to entropy at 20% |
|---|---|
| Mahalanobis distance on B0 output (softmax) | +0.48 [−0.15, +1.21] |
| Prototypes (Williams et al.), best variant | +0.19 [−0.41, +0.94] |
| DDU density (no spectral norm in B0) | +0.07 [−0.58, +0.78] |
| Mahalanobis on decoder embeddings | −0.72 [−1.01, −0.42] |
| kNN (DNP, cDNP, CSLS), best variant | −0.05 [−0.74, +0.66] |
| Regressor: entropy + DDU, against entropy-only regressor | +0.38 [−0.06, +1.06] |

Result: no geometry score is better than entropy on Cityscapes. Many scores are worse. Pixel AUROC of entropy
(0.934) is higher than all geometry scores (0.72–0.92).

## 5. Domain shift Cityscapes → ACDC (2026-10-23, branch `geometry`)

We wrote four main comparisons before we looked at ACDC. B2 on ACDC uses 1024 × 576 input.

- The oracle gain over entropy on ACDC is +5.55 points at 20% (Cityscapes: +2.60).
- Entropy gets 31% of the oracle gain on ACDC. On Cityscapes, it gets 64%.
- Mahalanobis on B0 output: +0.65 [+0.09, +1.22]. This is the only pass, and it is at the limit.
- Prototypes, DDU and the DDU regressor did not pass.

CAUTION: We selected the B2 input resolution with ACDC val labels. This choice changes only the B2 labels,
not the routers.

## 6. Tile cascade with time (2026-10-23 and 2026-10-24, branch `geometry`)

B0 runs on the full frame. B2 runs on 256 × 256 tiles with 64 pixels of context.

- Context is necessary. A 128 tile without context gives 47.2 mIoU. With 96 pixels of context, it gives 70.6.
- Live cascade on P1, fp32: 79.59 mIoU at 211 ms. B2 on the full frame: 79.95 at 358 ms (1.7 times slower).
- In fp16, B2 is two times faster. Then the cascade is not faster at B2 quality.
- A threshold that gives more tiles to hard frames adds +0.6 to +1.2 over a fixed number of tiles.
- Learned tile routers (HistGBR on confidence, shape, geometry) are not better than entropy.
- When we selected the operating point on P0, the speedup on P1 fell to 1.14 (fp32) and 1.05 (fp16).
- A spatial prior (B2 helps near the horizon) adds nothing on Cityscapes. On ACDC it adds 2–3 points of
  oracle share.

## 7. Frozen router B0/B2 (2026-10-25, branch `experiments/frozen-router-results-20261025`)

A trained head on B0 stage 2–3 features (F2/F3) and component geometry, T4 GPU, P1:

| Method | mIoU | Tiles per frame | Mean time |
|---|---|---|---|
| Entropy | 79.90 | 2.74 | 381 ms |
| Ridge on confidence | 79.79 | 2.83 | 463 ms |
| Mahalanobis F3 | 79.67 | 2.72 | 967 ms |

- The F2/F3 head against entropy: +0.21 and −0.20 points. Both intervals include 0.
- Mahalanobis F3 was +0.57 over ridge on the internal validation set. This gain did not transfer to P1.

## 8. Frozen SAE error head (2026-10-26, branch `experiments/frozen-sae-errors-20261026`)

A small convolutional SAE (16 latents) on B0 stage 2–3 features, trained together with an error classifier.

- AP for all B0 errors: entropy 0.434, SAE head 0.428, activations only 0.275.
- Result: no method was better than entropy. This is one seed on 60 validation frames.

## 9. Routing B0 → B5, E0–E11 (2026-10-26 to 2026-10-28, branch `baseline-experiments`)

Metric: the share of the B0–B5 gap that the router closes. Tiles are 64 × 64.

| Experiment | Result |
|---|---|
| E0 gap | B5 is 8–13 mIoU better on ACDC, most at night. |
| E1 oracle | The gain is in 5–10% of tiles. The tile oracle is 2–4 times stronger than the frame oracle. |
| E2 confidence | Entropy, MSP and margin are equal. Energy is worse. Night is the weak point. |
| E3–E4 geometry | Entropy wins at pixel level. At tile level, kNN is better at night. The rank sum of entropy and kNN is better at night and in rain (CI > 0). A learned router overfits Cityscapes. |
| E5, E5b crops | Without context, B5 on crops makes the result worse. At equal compute, windows are not better than full frames. |
| E6 fixed threshold | A threshold from clean data does not hold the budget under shift: 35–100% of ACDC frames go to B5. |
| E7 mixed stream | Geometry is not better than entropy. An online quantile holds the budget (+4 to +8 points over). |
| E8 gain model | A frame-gain regressor is not better than entropy. Geometry adds nothing to it. |
| E9 low-resolution B5 | Frame routing is better than B5 at 0.5 scale at the same compute. At night, B5 at 0.5 scale closes 80% of the gap. |
| E10 compute levels | A 4-level oracle is +20 points over a 2-level oracle. Rule "kNN-abnormal frame → ensemble at 0.75 scale, rest → B5 by entropy": +4.6 at budget 0.5, interval [+0.2, +10.2]. |
| E10b robustness | The rule gain is positive in all 15 cells at budgets 0.35–0.6, but only 2 cells have CI > 0. |
| E11 synthetic shift | Fixed in advance on corrupted Cityscapes: +15.0 [+11.8, +18.9] over entropy. A random flag of the same frames gives no gain. |

Result: kNN works as a detector of strong domain shift at frame level. It helps to select the compute level.
It does not help to select tiles inside a frame.

## 10. Frozen B5 context adapter, E8–E9-k4 (2026-10-27, branch `experiments/frozen-context-adapter-20261027`)

B5 computes deep blocks only for selected tokens. An adapter predicts the B5 states outside the region from
B0 features. Test set: `cs_test`. B0 73.65, full B5 78.29.

- True B5 states at 4 points keep 97–98% of the tile gain.
- Predicted states (E9-k4) give 74.97 (10% tiles) and 76.33 (20% tiles). This is 1.4 and 1.2 points below the
  true states.
- Entropy frame routing at the same estimated cost gives 77.48 and 77.62. The adapter is worse.
- There is no measured speedup. The B5 blocks still compute densely. E12 (output distillation) is ready but
  did not run.

## 11. Spatial B0 router for B5 tiles (2026-10-28, branch `results/spatial-router-acdc-20261028`)

A small CNN reads B0 output maps of a 256 tile and frame context. It predicts the B5 gain for each tile.
It is trained on P0. A P0 out-of-fold check approved budgets of 1 and 3 tiles before the test.

| Test | Tiles per frame | Δ to entropy |
|---|---|---|
| P1 | 1 | +0.35 [−1.83, +0.78] |
| P1 | 3 | **+0.38 [+0.01, +0.73]** |
| ACDC all (zero-shot) | 2 | +0.73 [−0.14, +1.69] |
| ACDC all (zero-shot) | 3 | **+0.97 [+0.19, +1.84]** |

Result: the router is a little better than entropy at 3 tiles per frame, on P1 and on ACDC. The P1 lower limit
is near 0. Per condition, only fog (1 tile) and rain (3 tiles) have CI > 0.

## 12. HyperUL double SegFormer (2026-10-27, branch `double_hyperul_segformer`)

B0 predicts its own hyperbolic uncertainty and the uncertainty of B2 (Chen et al., HyperUL). A frame goes to
B2 if the difference is above a threshold. The full pipeline (train, export, benchmark, report) is in the
branch. There are no results in git.

## 13. SAE v1: sparse codes of Mix-FFN activations (2026-10-26 and 2026-10-27, branch `SAE`)

One BatchTopK SAE for each of the 8 B0 blocks (D = 2d, k = 8), fitted on P0. CatBoost predicts the error
fraction of each 16 × 16 patch. P0 cross-validation only. No test was done.

| Set | recall@10% | Decile AUROC (v1 metric) |
|---|---|---|
| Entropy | 0.560 | 0.616 |
| F0: B0 output features | 0.566 | 0.676 |
| F1: F0 + SAE codes | 0.566 | 0.739 |
| F3F0: raw activations + F0 | 0.569 | 0.814 |

- The internal features do not change the global ranking (recall@10%).
- SAE codes are worse than raw activations.
- The v1 decile metric is not reliable. The low-entropy deciles contain only 2–95 positive patches.

## 14. AE v2: bottleneck autoencoder (2026-10-27 and 2026-10-28, branch `SAE`)

One dense autoencoder for each block (b = d / 8), fitted on Cityscapes train. Features: codes (Z),
reconstruction error (R), activation norm (N, control). PCA with the same b is the linear alternative.
Fit on P0. Test on P1 and ACDC val. The robust AUROC uses only entropy deciles 4–9.

### 14.1 Encoders

- The AE explains 78–89% of the variance. PCA explains 55–76%.
- On ACDC, the reconstruction error increases. It is largest at night in stage 4 (2.4 times the P0 value).
  The norm N does not follow. Thus, R detects the shift.

### 14.2 Tests

| Set | P1 recall@10% | ACDC recall@10% | P1 robust AUROC | ACDC robust AUROC |
|---|---|---|---|---|
| Entropy | 0.552 | 0.300 | 0.571 | 0.561 |
| F0N: F0 + norm (control) | 0.560 | 0.306 | 0.652 | 0.559 |
| R: F0 + N + R | 0.564 | 0.313 | 0.743 | **0.737** |
| ZR: F0 + N + Z + R | **0.566** | **0.319** | **0.744** | 0.687 |
| F3F0NR: raw + F0 + N + R | 0.562 | 0.299 | 0.733 | 0.590 |

Pre-registered comparisons (Holm correction over the three main comparisons):

| Comparison | Test | Metric | Δ [95% CI] | Result |
|---|---|---|---|---|
| R − F0N | ACDC | recall@10% | +0.007 [+0.002, +0.011] | pass |
| R − F0N | ACDC | robust AUROC | +0.178 [+0.163, +0.193] | pass |
| R − F0N | P1 | recall@10% | +0.004 [−0.001, +0.010] | fail |
| R (AE) − R (PCA) | ACDC | recall@10% | +0.012 [+0.009, +0.016] | CI > 0 |
| ZR − F0 | ACDC | recall@10% | +0.026 [+0.021, +0.031] | CI > 0 |

- On Cityscapes, the new features do not change recall. Only 1.9% of wrong pixels are below the median entropy.
- On ACDC, R gives a large gain inside the same entropy level, in all four conditions.
- The pooled ACDC recall gain of R is small. It can come from a budget shift between conditions. We did not
  check this directly.
- The nonlinear AE is better than PCA under shift.
- With raw activations in the set (F3F0NR), CatBoost does not use N and R. The result on ACDC is worse.

### 14.3 Integration

`INTEGRATION.md` tells how to install the scorer in a router. ZR is the default set. The full scorer takes
68 ms per frame (R: 55 ms, B0 only: 17 ms) on an RTX 3060 Ti.

NOTE: The scorer predicts B0 errors. It does not predict the gain from the heavy model.

## 15. Summary

1. **On Cityscapes, entropy is a very strong baseline.** No geometry, learned field, activation or SAE/AE
   score gives a clear gain for routing or error recall (series 3, 4, 6–8, 13, 14).
2. **Under domain shift, entropy is weaker.** On ACDC it gets 31% of the oracle gain (64% on Cityscapes).
   This is where the internal signals help.
3. **Signals of "abnormal input" help under shift.**
   - kNN at frame level selects the compute level: +15 points on synthetic shift, +4.6 at the limit on ACDC
     (series 9).
   - The AE reconstruction error ranks B0 errors inside the same entropy level: robust AUROC 0.56 → 0.74 on ACDC
     (series 14).
4. **Learned spatial routers give small gains.** The spatial B0 router is +0.4 (P1) and +1.0 (ACDC) points at
   3 tiles (series 11). Other learned routers overfit Cityscapes (series 6, 9).
5. **Budget control is a stronger lever than tile selection.** A frame threshold, an online quantile and the
   choice of compute level each give more than a better tile score (series 6, 9).
6. **Speed depends on precision.** The tile cascade is 1.7 times faster than B2 in fp32. In fp16 the gain
   disappears (series 6). The B5 context adapter has no measured speedup (series 10).
7. **B0 error is not the same as heavy-model gain.** A B0 error detector does not tell where B2 or B5 is
   correct (series 7, 8, 14).

## 16. Limits of the evidence

- P1 and `cs_test` were used many times. Results on them are exploratory, not independent tests.
- The B2 input resolution for ACDC was selected with ACDC labels (series 5).
- Many results use one seed.
- The intervals do not include a correction for the number of comparisons, except in series 14.

## 17. Open items

- AE v2: logistic-regression runs, R only, Z and ZR with PCA, bottleneck ablation (b = d/4, d/16).
- AE v2: measure the share of the ACDC budget that goes to each condition for R and F0N.
- Test whether the AE score (R, ZR) predicts the B2/B5 gain, not only the B0 error.
- E12 output distillation for the B5 context adapter.
- HyperUL: run the benchmark and add the results to git.
- kNN compute-level rule: test on ACDC train as an additional test set.

## 18. Source documents

| Series | Document | Branch |
|---|---|---|
| 3–6 | `EXPERIMENTS.md`, `METHODOLOGY.md` | `geometry` |
| 7 | `experiments/frozen_router_20261025/README.md` | `experiments/frozen-router-results-20261025` |
| 8 | `experiments/sae_error_head_20261026/README.md` | `experiments/frozen-sae-errors-20261026` |
| 9 | `experiments/README.md` | `baseline-experiments` |
| 10 | `experiments/context_adapter_20261027/README.md` | `experiments/frozen-context-adapter-20261027` |
| 11 | `artifacts/*/README*`, `acdc_router_report.csv` | `results/spatial-router-acdc-20261028` |
| 12 | `README.md` | `double_hyperul_segformer` |
| 13 | `PLAN.md`, `RESULTSV1.md` | `SAE` |
| 14 | `PLANv2.md`, `RESULTSV2.md`, `RESULTSv2.1.md`, `INTEGRATION.md` | `SAE` |
