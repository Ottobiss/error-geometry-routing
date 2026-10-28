# RESULTS v2.1: v1 metrics on the v2 runs

Status as of 2026-10-28. This recomputes the RESULTSV1 §4.1 metric set for every v2 CatBoost run (RESULTSV2.md),
so v1 and v2 can be read side by side. Nothing was retrained. Code is at commit `023c7b1` plus uncommitted
`ae/` changes.

## How it was computed

- **Code:** the v1 metric code, `sae.classify.population_metrics`, applied to the saved v2 scores:
  - `results/ae/runs/<run>/cv_scores.parquet` gives out-of-fold P0;
  - `results/ae/runs/<run>/scores.parquet` gives the P1 and ACDC tests.
- **Entropy cuts:** the P0 patch mean-entropy deciles and the lowest-fifth cut are those of v1, fitted on all 569,346 P0 patches.
- **Output:** the table is saved as `results/ae/v1_metrics.csv`.
- **Columns:**
  - "Deciles": the v1 metric, the y50 AUROC averaged over every P0 entropy decile that contains positives.
  - "Low ⅕": the y50 AUROC in the lowest-entropy fifth.
  - Bold marks the best value in each column, rounded to 3 decimals.
- **v1-only rows:** v1 F1 (SAE codes + F0) and v1 F3 (raw activations only) were never re-run in v2. Their P0
  numbers come from `results/sae/runs/<set>/cv_metrics.json`, and they have no test scores.

**Reproducibility check.** F0 and F3F0 are the same models in v1 and v2, because the v1 cap of 3,000 iterations
never bound for them. Their v2 out-of-fold scores are byte-identical to v1 (max |Δ| = 0). The recomputed P0
numbers match RESULTSV1 §4.1: decile AUROC 0.616 (entropy), 0.676 (F0), 0.739 (F1), 0.814 (F3F0), and Low ⅕ 0.994
(F3F0).

**Low ⅕ positives.** The lowest-entropy fifth holds few y50 positives in most populations:

| Population | Patches in lowest fifth | y50 positives |
|---|---|---|
| P0 | 113,870 | 2 |
| P1 | 72,940 | 68 |
| ACDC fog | 18,178 | 4 |
| ACDC night | 14,834 | 248 |
| ACDC rain | 6,662 | 0 (undefined) |
| ACDC snow | 9,240 | 298 |
| ACDC all | 48,914 | 550 |

Read the Low ⅕ column only for night, snow and ACDC all. P1 is shaky, and P0 and fog are noise. This is why v2
replaced it with the robust decile AUROC (RESULTSV2 §5).

## Tables

### P0 (out-of-fold CV; same layout as RESULTSV1 §4.1)

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.333 | 0.560 | 0.818 | 0.936 | 0.907 | 0.283 | 0.952 | **0.744** | 0.616 | 0.944 |
| F0 | 0.348 | 0.566 | 0.823 | 0.939 | 0.914 | 0.307 | 0.953 | 0.743 | 0.676 | 0.881 |
| v1 F1 (SAE codes) | 0.349 | 0.566 | 0.823 | 0.937 | 0.913 | 0.302 | 0.954 | 0.736 | 0.739 | 0.963 |
| v1 F3 (raw only) | 0.283 | 0.483 | 0.753 | 0.901 | 0.889 | 0.193 | 0.930 | 0.641 | 0.769 | 0.970 |
| F3F0 | 0.350 | 0.569 | 0.825 | 0.942 | 0.919 | 0.317 | **0.955** | 0.739 | 0.814 | 0.994 |
| F0N | 0.349 | 0.567 | 0.822 | 0.940 | 0.916 | 0.308 | 0.953 | 0.739 | 0.728 | 0.915 |
| R (AE) | 0.352 | 0.571 | 0.828 | **0.949** | **0.923** | **0.319** | 0.953 | 0.730 | 0.819 | 0.988 |
| R (PCA) | **0.353** | **0.573** | **0.829** | 0.948 | **0.923** | **0.319** | 0.953 | 0.730 | 0.816 | 0.970 |
| Z (AE) | 0.350 | 0.569 | 0.825 | 0.942 | 0.920 | 0.316 | 0.954 | 0.733 | 0.814 | 0.987 |
| ZR (AE) | 0.352 | 0.571 | 0.826 | 0.945 | 0.922 | **0.319** | 0.954 | 0.730 | **0.834** | **0.996** |

### P1

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.329 | 0.552 | 0.813 | 0.933 | 0.902 | 0.298 | 0.950 | **0.743** | 0.570 | 0.610 |
| F0 | 0.348 | 0.563 | 0.817 | 0.935 | 0.909 | 0.326 | 0.949 | 0.729 | 0.545 | 0.404 |
| F3F0 | 0.348 | 0.561 | 0.820 | **0.940** | 0.914 | 0.327 | **0.951** | 0.729 | **0.762** | **0.795** |
| F0N | 0.343 | 0.560 | 0.815 | 0.936 | 0.909 | 0.323 | 0.949 | 0.727 | 0.591 | 0.468 |
| R (AE) | 0.347 | 0.564 | 0.820 | 0.939 | **0.917** | 0.334 | 0.948 | 0.717 | 0.722 | 0.611 |
| R (PCA) | 0.347 | 0.563 | 0.821 | 0.938 | **0.917** | 0.336 | 0.949 | 0.717 | 0.698 | 0.562 |
| Z (AE) | 0.349 | 0.563 | **0.822** | **0.940** | **0.917** | 0.332 | 0.950 | 0.720 | 0.725 | 0.644 |
| ZR (AE) | **0.351** | **0.566** | **0.822** | **0.940** | **0.917** | **0.343** | 0.950 | 0.723 | 0.714 | 0.544 |

### ACDC fog

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.228 | 0.420 | 0.668 | 0.806 | 0.850 | 0.339 | 0.913 | 0.679 | 0.620 | 0.821 |
| F0 | 0.234 | 0.413 | 0.651 | 0.790 | 0.855 | 0.353 | 0.907 | 0.656 | 0.693 | 0.951 |
| F3F0 | 0.238 | 0.419 | 0.666 | 0.811 | 0.857 | 0.331 | **0.920** | **0.685** | 0.753 | **1.000** |
| F0N | **0.244** | 0.424 | 0.670 | 0.806 | 0.856 | 0.359 | 0.908 | 0.655 | 0.708 | 0.991 |
| R (AE) | **0.244** | **0.428** | 0.677 | 0.818 | **0.864** | 0.359 | 0.911 | 0.644 | 0.752 | 0.995 |
| R (PCA) | 0.243 | 0.426 | 0.675 | 0.813 | 0.860 | **0.363** | 0.911 | 0.654 | 0.738 | 0.995 |
| Z (AE) | 0.241 | 0.424 | 0.674 | 0.820 | **0.864** | 0.336 | 0.917 | 0.663 | **0.762** | 0.997 |
| ZR (AE) | 0.240 | **0.428** | **0.679** | **0.822** | 0.863 | 0.339 | 0.915 | 0.652 | 0.757 | 0.999 |

### ACDC night

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.087 | 0.171 | 0.330 | 0.473 | **0.767** | **0.645** | **0.834** | **0.805** | 0.550 | 0.550 |
| F0 | 0.085 | 0.167 | 0.319 | 0.457 | 0.727 | 0.611 | 0.797 | 0.777 | 0.405 | 0.581 |
| F3F0 | 0.085 | 0.165 | 0.318 | 0.457 | 0.738 | 0.614 | 0.811 | 0.784 | 0.552 | **0.777** |
| F0N | 0.090 | 0.174 | 0.328 | 0.471 | 0.747 | 0.632 | 0.811 | 0.779 | 0.502 | 0.633 |
| R (AE) | 0.085 | 0.168 | 0.327 | 0.475 | 0.760 | 0.630 | 0.811 | 0.756 | **0.619** | 0.730 |
| R (PCA) | 0.086 | 0.167 | 0.321 | 0.461 | 0.739 | 0.613 | 0.800 | 0.757 | 0.553 | 0.699 |
| Z (AE) | 0.087 | 0.168 | 0.323 | 0.464 | 0.747 | 0.622 | 0.812 | 0.778 | 0.559 | 0.746 |
| ZR (AE) | **0.093** | **0.177** | **0.335** | **0.478** | 0.758 | **0.645** | 0.815 | 0.777 | 0.598 | 0.704 |

### ACDC rain

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.228 | 0.422 | 0.682 | 0.825 | 0.875 | 0.423 | 0.927 | 0.744 | 0.544 | — |
| F0 | 0.244 | 0.432 | 0.687 | 0.827 | 0.883 | 0.460 | 0.923 | 0.736 | 0.610 | — |
| F3F0 | **0.247** | 0.438 | 0.694 | 0.836 | 0.887 | 0.446 | **0.931** | **0.748** | 0.746 | — |
| F0N | 0.236 | 0.431 | 0.697 | 0.844 | 0.887 | 0.450 | 0.925 | 0.725 | 0.630 | — |
| R (AE) | 0.229 | 0.418 | 0.694 | 0.851 | 0.899 | 0.429 | 0.918 | 0.663 | **0.787** | — |
| R (PCA) | 0.223 | 0.415 | 0.693 | 0.850 | 0.896 | 0.427 | 0.918 | 0.668 | 0.748 | — |
| Z (AE) | **0.247** | **0.442** | **0.708** | 0.853 | 0.898 | **0.468** | 0.930 | 0.734 | 0.768 | — |
| ZR (AE) | 0.241 | 0.435 | **0.708** | **0.858** | **0.900** | 0.450 | 0.926 | 0.698 | 0.761 | — |

### ACDC snow

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.194 | 0.354 | 0.591 | 0.735 | 0.808 | 0.417 | 0.882 | 0.725 | 0.530 | 0.558 |
| F0 | 0.196 | 0.357 | 0.593 | 0.738 | 0.822 | **0.430** | 0.887 | 0.720 | 0.574 | 0.489 |
| F3F0 | **0.203** | **0.363** | 0.600 | 0.743 | 0.822 | 0.420 | 0.890 | **0.728** | 0.611 | 0.539 |
| F0N | 0.193 | 0.354 | 0.590 | 0.738 | 0.816 | 0.415 | 0.882 | 0.701 | 0.590 | 0.524 |
| R (AE) | 0.189 | 0.342 | 0.582 | 0.738 | 0.828 | 0.398 | 0.877 | 0.650 | 0.665 | 0.510 |
| R (PCA) | 0.189 | 0.350 | 0.592 | 0.745 | 0.832 | 0.409 | 0.882 | 0.668 | **0.687** | **0.573** |
| Z (AE) | 0.198 | 0.360 | **0.604** | **0.758** | 0.832 | 0.422 | **0.891** | 0.711 | 0.636 | 0.471 |
| ZR (AE) | 0.198 | 0.356 | 0.602 | 0.756 | **0.834** | 0.413 | 0.887 | 0.677 | 0.664 | 0.530 |

### ACDC all

| Set | rec@5% | rec@10% | rec@20% | rec@30% | AUROC y50 | AP y50 | AUROC y_any | AP y_any | Deciles | Low ⅕ |
|---|---|---|---|---|---|---|---|---|---|---|
| entropy | 0.158 | 0.300 | 0.525 | 0.679 | 0.814 | 0.484 | 0.881 | **0.742** | 0.547 | 0.580 |
| F0 | 0.156 | 0.293 | 0.507 | 0.651 | 0.793 | 0.464 | 0.864 | 0.717 | 0.496 | 0.551 |
| F3F0 | 0.159 | 0.295 | 0.511 | 0.659 | 0.806 | 0.465 | 0.878 | 0.731 | 0.616 | 0.746 |
| F0N | 0.165 | 0.306 | 0.528 | 0.678 | 0.808 | 0.488 | 0.871 | 0.718 | 0.555 | 0.602 |
| R (AE) | 0.166 | 0.313 | **0.550** | **0.715** | **0.844** | 0.511 | **0.884** | 0.698 | **0.728** | 0.711 |
| R (PCA) | 0.160 | 0.301 | 0.525 | 0.685 | 0.824 | 0.483 | 0.876 | 0.696 | 0.688 | **0.747** |
| Z (AE) | 0.163 | 0.304 | 0.526 | 0.680 | 0.821 | 0.485 | 0.882 | 0.724 | 0.647 | 0.686 |
| ZR (AE) | **0.175** | **0.319** | 0.546 | 0.702 | 0.833 | **0.515** | **0.884** | 0.716 | 0.686 | 0.698 |

## Reading

1. **On P0, the AE features close the gap to raw activations.**
   - Raw activations (F3F0) led the v1 decile AUROC at 0.814. R (AE) reaches 0.819 and ZR (AE) 0.834 with a
     fraction of the features.
   - The v1 SAE codes (F1) stayed at 0.739.
   - Global metrics (recall@budget, AUROC, AP) move by at most about 0.01 across all sets, as in v1.
2. **F3F0 keeps the best within-entropy ranking on P1** (Deciles 0.762, Low ⅕ 0.795), with Z (AE) and R (AE) at
   0.72–0.73. The raw-activation signal generalises within Cityscapes.
3. **On ACDC, the AE residuals take over.** On ACDC all, the decile AUROC is:

   | Set | Deciles (ACDC all) |
   |---|---|
   | R (AE) | 0.728 |
   | R (PCA) | 0.688 |
   | ZR (AE) | 0.686 |
   | F3F0 | 0.616 |
   | F0N | 0.555 |
   | entropy | 0.547 |
   | F0 | 0.496 |

   R (AE) is first or close to first on Deciles in every condition. The gap is largest at night: 0.619 against
   0.552 for F3F0.
4. **The v1 "feeling of error" from F0 alone does not survive testing.** F0 ranks below entropy inside entropy
   levels on P1 (Deciles 0.545 vs 0.570, Low ⅕ 0.404 vs 0.610), at night (0.405) and on ACDC all (0.496).
5. **Recall@budget gains are small everywhere.**
   - The largest is on ACDC all for R (AE): recall@30% 0.715 vs 0.679 (entropy) and 0.678 (F0N).
   - ZR (AE) leads recall@5/10% on ACDC all and at night.
   - In rain and snow, sets with codes (Z, ZR, F3F0) have the best recall. R alone is below entropy at 10% in
     rain and at 5–10% in snow.
6. **Entropy keeps the best global AUROC/AP at night (ZR ties the AP) and the best y_any AP on P0, P1 and ACDC all.** The
   learned scores mostly re-rank inside entropy levels rather than beat it globally.
