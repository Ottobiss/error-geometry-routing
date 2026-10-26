"""Patch-level error prediction for SegFormer-B0 from SAE codes of Mix-FFN activations (PLAN.md).

    uv run python -m sae.hooks                          # 0. hooks + sanity
    uv run python -m sae.collect                        # 1. P0 activations for SAE training
    uv run python -m sae.train                          # 2. one BatchTopK SAE per block, results/sae/quality.csv
    uv run python -m sae.patch_features --dataset cityscapes --sae D2k8   # 3. patch table
    uv run python -m sae.patch_features --dataset acdc --sae D2k8
    uv run python -m sae.classify cv --features F1 --sae D2k8             # 4. CatBoost, GroupKFold on P0
    uv run python -m sae.classify test --features F1 --sae D2k8           # 5. final fit on P0, P1 and ACDC
    uv run python -m sae.classify compare --sae D2k8                      #    pre-registered comparisons

Steps 4–5 run once per feature set: F0, F1, F2, F3, F3F0 (compare needs F0, F1, F2 and F3). The raw sets
(F3, F3F0) hold the P0 training patches as dense float32 (~0.3M × 4,096, ~5 GB) plus CatBoost's copy until the
pool is quantised.

Everything is written under results/sae/.
"""
from core.paths import RESULTS

ROOT = RESULTS / "sae"
