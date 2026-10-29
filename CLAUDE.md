# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research project that predicts where SegFormer-B0 (`nvidia/segformer-b0-finetuned-cityscapes-512-1024`, frozen, fp32) makes errors. It works per 16 × 16 patch of the B0 output, and the features come from the Mix-FFN hidden activations: the input of `fc2` in the 8 encoder blocks `s1b1 … s4b2`, caught by a forward pre-hook. The target is the fraction of wrong valid pixels in the patch, and a CatBoost model (CrossEntropy, weighted by valid pixels) predicts it. The baseline to beat is the B0 output entropy plus the F0 output statistics.

There are two experiment series on branch `SAE`:
- **v1, `sae/`** ([PLAN.md](PLAN.md)): BatchTopK sparse autoencoders per block. Results are in [RESULTSV1.md](RESULTSV1.md).
- **v2, `ae/`** ([PLANv2.md](PLANv2.md)): a bottleneck AE (d → d → d/8 → d → d) and PCA per block, giving features Z (codes), R (reconstruction residual) and N (activation norm). Results are in [RESULTSV2.md](RESULTSV2.md) and [RESULTSv2.1.md](RESULTSv2.1.md).

[REPORT.md](REPORT.md) summarizes both series. [DESCRIPTION.md](DESCRIPTION.md) describes the ZR pipeline end to end, and [INTEGRATION.md](INTEGRATION.md) is the guide for installing the scorer in a router. These docs are written in ASD-STE100 (Simplified Technical English), so keep that style when you edit them.

## Commands

The project uses uv with Python 3.14. torch comes from the `pytorch-cu130` index (CUDA ≥ 13.0 driver). Run every module from the repo root as `python -m`, because modules import `core`, `sae` and `ae` as top-level packages and use relative `results/`, `data/` and `splits/` paths.

```
uv sync
uv run pytest tests                                   # all tests (tiny random B0, no data, no GPU)
uv run pytest tests/test_ae.py::test_holm             # a single test
```

The **run order** for each pipeline is the module docstring of [sae/__init__.py](sae/__init__.py) and [ae/__init__.py](ae/__init__.py), which are the reference for the steps and flags. In short:
- v1: `sae.hooks` → `sae.collect` → `sae.train` → `sae.patch_features --dataset {cityscapes,acdc} --sae D2k8` → `sae.classify {cv,test,compare} --features F0|F1|F2|F3|F3F0`
- v2: `download.cityscapes --prefix leftImg8bit/train/` → `ae.collect` → `ae.train` / `ae.pca` → `ae.quality` → `ae.patch_features --encoders ae_b8 pca_b8` → `ae.checks` → `ae.classify {cv,test,compare} --features … --encoder … --model catboost|logreg`

Run `cv` before `test` for each feature set, because `test` reuses the number of trees chosen in CV. `compare` needs all the feature sets that it compares. Most collection and feature steps take `--limit N` for a smoke run on the first N frames. `classify` takes `--gpu` to run CatBoost on the GPU.

Data goes in `data/cityscapes` (official layout) and `data/acdc`. Get it with `python -m download.cityscapes` (asks for Cityscapes credentials through cityscapesscripts) and `python -m download.acdc`. These scripts stream only the files they need from the remote ZIPs, and you can restart them after a break.

## Architecture

- **`core/`** holds the shared infrastructure: datasets and populations (`core.data`), SegFormer loading and preprocessing (`core.models`), paths and run-config JSON with date and git-commit stamps (`core.paths`), and routing evaluation. A large part of `core` (connected components, MetaSeg fields, B2 tiles, `Units`, routing caches) comes from the component-routing experiments on other branches. The `sae/` and `ae/` pipelines do not use it. Comments in `core/` are in Russian, and comments in `sae/` and `ae/` are in English.
- **Populations** are defined by `splits/<dataset>_<pop>.txt`. Cityscapes `p0` (300 val frames) is the only fit and selection population (`FIT_DATASET, FIT_POP`). Cityscapes `p1` (200 val) and ACDC `fog/night/rain/snow` (406 val frames in total) are test populations, and `acdc:all` is their union. `test_populations()` takes every split file except P0, so do not add a split file for training data. v2 encoders are fit on Cityscapes **train** through `get_dataset(..., split="train")`.
- **Working resolution:** Cityscapes is 1024 × 512 and ACDC is 1024 × 576, both made by a PIL bilinear resize. Labels are resized with nearest neighbour into trainIds with ignore = 255. A patch grid is 32 × 64 (Cityscapes) or 36 × 64 (ACDC).
- **`sae/hooks.py` is shared by both series:** `BLOCKS`, `stride`, `MixFFNTap` and `pool_to_patches`/`selection_index`. They map native activation positions to patches. Stages 1–2 (stride < 16) pool all positions inside the patch, and stages 3–4 take the one position nearest to the patch centre. `ae/` also reuses `sae.classify` (training-row subsampling, metrics, GroupKFold by frame) and `sae.patch_features` (F0 columns and the base patch table).
- **Output layout:** v1 writes to `results/sae/` and v2 writes to `results/ae/`. Each has `acts/`, `ckpt/<encoder>/`, `patches/` and `runs/<features>_<encoder>_<model>/` (model, `cv_scores.parquet`, `scores.parquet`, metrics JSON). The base patch tables (F0, labels, pixel samples, raw activations) live only in `results/sae/patches/base/`, and v2 reads them from there. So `sae.patch_features --sae none` (base only) must run before the `ae.patch_features` step.
- **Pre-registered comparisons** (`COMPARISONS` in each `classify.py`) use a paired bootstrap over frames (2,000 resamples), and v2 adds a Holm correction. These were fixed before the tests ran, so do not change them in a way that would amount to tuning on test populations.

## Conventions

- Report numbers must come from the saved `RESULTS*.md` files or from saved run outputs (`results/**/runs/*`, `results/ae/v1_metrics.csv`, `compare.csv`). Do not recompute them.
- `data/` and `results/` are gitignored. Large runs keep dense float32 activations in memory (for F3/F3F0, about 5 GB plus CatBoost's copy).
