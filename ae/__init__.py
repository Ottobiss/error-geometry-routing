"""Patch-level error prediction from a bottleneck autoencoder of Mix-FFN activations (PLANv2.md).

    uv run python -m download.cityscapes --prefix leftImg8bit/train/    # 0. Cityscapes train frames
    uv run python -m ae.collect                                         # 1. Cityscapes-train activations
    uv run python -m ae.train                                           # 2. AE per block, b = d/8 (ae_b8)
    uv run python -m ae.pca                                             #    PCA per block, b = d/8 (pca_b8)
    uv run python -m ae.quality --encoders ae_b8 pca_b8                 #    FVE per population, splice drop
    uv run python -m ae.patch_features --dataset cityscapes --encoders ae_b8 pca_b8   # 3. Z, R, N per patch
    uv run python -m ae.patch_features --dataset acdc --encoders ae_b8 pca_b8
    uv run python -m ae.checks --encoders ae_b8 pca_b8                  # 4. label-free checks (PLANv2 §8)
    uv run python -m ae.classify cv --features R --encoder ae_b8 --model catboost     # 5. P0 CV
    uv run python -m ae.classify test --features R --encoder ae_b8 --model catboost   # 6. P1 and ACDC
    uv run python -m ae.classify compare                                #    pre-registered comparisons

Bottleneck ablation (b = d/16, d/4; sensitivity only, §8.4): `ae.train --div 16`, `ae.train --div 4`, then
quality, patch_features and checks with --encoders ae_b16 ae_b4.

Steps 5 and 6 run once per feature set and classifier (catboost, logreg): F0, F0N, R, Z, ZR, R_only, F3F0 with
the AE, and R, Z, ZR with --encoder pca_b8. Everything is written under results/ae/; the base patch tables
(F0, labels, pixel samples, raw activations) are reused from results/sae/patches/base/.
"""
from core.paths import RESULTS

ROOT = RESULTS / "ae"
