# Integration guide: B0 patch error score (R, Z, ZR)

Written in ASD-STE100 (Simplified Technical English). Status: 2026-10-28, branch `SAE`, commit `023c7b1`
plus the working tree.

## 1. Scope

This guide tells you how to install the patch error scorer in a router. The guide does not know the router.
Thus, the scorer has one input (a frame) and one main output (a score map). You connect the score map to
your router units with an adapter (section 8).

The scorer finds 16 × 16 pixel patches where SegFormer-B0 makes errors. It uses three feature sets:

| Set | Features | Columns |
|---|---|---|
| R | B0 output statistics (F0), activation norm (N), AE reconstruction residual (R) | 11 + 12 + 12 = 35 |
| Z | F0, AE bottleneck codes (Z) | 11 + 512 = 523 |
| ZR | F0, N, Z, R | 11 + 12 + 512 + 12 = 547 |

A CatBoost model gives the score from these features. The score is the predicted fraction of wrong B0
pixels in the patch.

## 2. Data that the scorer gets

The scorer gets these data for each frame:

1. **The input frame.** RGB, uint8, shape (3, H, W), at the working resolution of B0.
   - Cityscapes: 1024 × 512 (the 2048 × 1024 frame, bilinear resize).
   - ACDC: 1024 × 576 (the 1920 × 1080 frame, bilinear resize).
2. **The B0 softmax.** Shape (19, H, W). B0 is `nvidia/segformer-b0-finetuned-cityscapes-512-1024`, fp32.
3. **The Mix-FFN hidden activations of B0.** This is the input of `fc2` in each of the 8 encoder blocks
   (s1b1 … s4b2). A forward pre-hook catches them during the usual B0 forward pass. The hook does not change
   the B0 output (the logits difference is 0).

| Stage | Blocks | Stride | Channels d | Code size b = d / 8 |
|---|---|---|---|---|
| 1 | s1b1, s1b2 | 4 | 128 | 16 |
| 2 | s2b1, s2b2 | 8 | 256 | 32 |
| 3 | s3b1, s3b2 | 16 | 640 | 80 |
| 4 | s4b1, s4b2 | 32 | 1024 | 128 |

The scorer does not use ground truth. It does not use a heavy model.

## 3. How the scorer processes the data

The scorer does these steps for each frame:

1. It runs B0 one time with the hooks. It keeps the softmax and the 8 activation maps.
2. It normalises each activation x = (h − mean) / scale. The constants are in the encoder checkpoints.
3. It encodes and decodes x with the bottleneck autoencoder of the block. The AE is
   d → d (GELU) → b → d (GELU) → d.
4. For each native position, it calculates:
   - Z: the code z (b values).
   - R: the residual ‖x − x̂‖² / d.
   - N: the norm ‖x‖² / d.
5. It moves the values to the 16 × 16 patch grid:
   - Stages 1–2: N and R use the mean and the max over the positions in the patch. Z uses the mean.
   - Stages 3–4: the scorer uses the one position nearest to the patch centre.
6. It calculates F0 from the B0 softmax for each patch:
   - entropy: mean, max, p90;
   - max softmax: mean, min;
   - top-2 margin: mean, min;
   - boundary fraction;
   - row and column of the patch;
   - majority class `cls` (a categorical column).
7. It puts the columns in the model order: F0, then N, then Z, then R.
8. It applies the CatBoost model and returns the probability.

## 4. Output

`ErrorScorer(small)` returns a dict:

| Key | Shape | Type | Content |
|---|---|---|---|
| `score` | (Hp, Wp) | float32 | Predicted fraction of wrong B0 pixels in the patch, 0…1. Higher = more errors. |
| `pred` | (H, W) | int64, GPU | B0 prediction (argmax). This is the usual B0 output. |
| `probs` | (19, H, W) | float32, GPU | B0 softmax. |
| `N` | (Hp·Wp, 12) | float32 | Activation norms (section 3, step 4). |
| `Z` | (Hp·Wp, 512) | float32 | AE codes. |
| `R` | (Hp·Wp, 12) | float32 | AE residuals. |

Hp = H / 16 and Wp = W / 16. Patches are in raster order: patch = row · Wp + col. For Cityscapes, the grid
is 32 × 64. For ACDC, it is 36 × 64.

Use `score` if your router needs one number for each region. Use `N`, `Z` and `R` if your router trains its
own model.

The column names are:
- N: `<block>_n_mean`, `<block>_n_max` (stages 1–2), `<block>_n` (stages 3–4);
- R: the same with `r`;
- Z: `<block>_z<j>`.

All columns are in the block order s1b1 … s4b2.

## 5. Files to copy

### 5.1 Code

Copy these objects from branch `SAE` into one module, `ae_error/lib.py`. Copy them without changes. Copy
only the objects in the list. The full files import training code that the scorer does not need.

| From | Objects |
|---|---|
| `core/components.py` | `NC` (= 19) |
| `core/models.py` | `CHECKPOINTS`, `FULL_RES`, `SegModel`, `load_model` |
| `core/evaluation.py` | `entropy_score` |
| `sae/hooks.py` | `BLOCKS`, `STRIDES`, `PATCH`, `stage`, `stride`, `fc2`, `MixFFNTap`, `selection_index`, `pool_to_patches` |
| `sae/patch_features.py` | `per_patch`, `boundary_map`, `output_features` |
| `ae/model.py` | `_Normalized`, `BottleneckAE`, `PCA`, `KINDS`, `load_encoder` |
| `ae/patch_features.py` | `scalar_columns`, `pool_scalar`, `frame_features` |

Put these imports at the top of `ae_error/lib.py`:

```python
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation
```

NOTE: The router can already have `core/models.py` and `core/evaluation.py`. If so, import `SegModel`,
`load_model` and `entropy_score` from there. Make sure that `SegModel` is the same as on branch `SAE`.

Put an empty `ae_error/__init__.py` next to it. Then add the scorer, `ae_error/scorer.py`. This is new code.
Copy it from this guide:

```python
"""Patch error score of SegFormer-B0 from bottleneck-AE features (R, Z, ZR)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoost, Pool

from ae_error.lib import BLOCKS, PATCH, MixFFNTap, frame_features, load_encoder, load_model, output_features

F0 = ["ent_mean", "ent_max", "ent_p90", "msp_mean", "msp_min", "margin_mean", "margin_min",
      "boundary_frac", "row", "col", "cls"]
PARTS = {"R": ("N", "R"), "Z": ("Z",), "ZR": ("N", "Z", "R")}


class ErrorScorer:
    """features: "R", "Z" or "ZR"; root: folder with ckpt/ae_b8/<block>.pt and <features>.cbm."""

    def __init__(self, features, root, device="cuda", b0=None):
        root = Path(root)
        self.parts = PARTS[features]
        self.b0 = b0 if b0 is not None else load_model("b0", device)
        self.enc = {b: load_encoder(root / "ckpt" / "ae_b8" / f"{b}.pt", device)[0] for b in BLOCKS}
        self.norms = {b: (e.mean, e.scale.item()) for b, e in self.enc.items()}
        self.model = CatBoost()
        self.model.load_model(str(root / f"{features}.cbm"))

    @torch.inference_mode()
    def __call__(self, small):
        """small: (3, H, W) uint8 RGB frame at the working resolution, H and W multiples of 16.
        Returns score (Hp, Wp), B0 pred (H, W), probs (19, H, W) and the per-patch features N, Z, R."""
        H, W = small.shape[1:]
        hw = (H // PATCH, W // PATCH)
        with MixFFNTap(self.b0) as tap:
            probs = self.b0(small[None].to(self.b0.mean.device), (W, H)).softmax(1)[0]
            pred = probs.argmax(0)
            enc_out, N = frame_features(tap.acts, {"ae": self.enc}, self.norms, hw)
        Z, R = enc_out["ae"]
        f0 = output_features(probs, pred)
        feats = {"N": N, "Z": Z, "R": R}
        X = pd.concat([pd.DataFrame({c: f0[c] for c in F0})] + [pd.DataFrame(feats[p]) for p in self.parts], axis=1)
        X.columns = self.model.feature_names_
        X["cls"] = X["cls"].astype(np.int32)
        score = self.model.predict(Pool(X, cat_features=["cls"]), prediction_type="Probability")[:, 1]
        return {"score": score.reshape(hw).astype(np.float32), "pred": pred, "probs": probs, **feats}
```

### 5.2 Fitted artifacts

Copy these files into one artifact folder (30 MB in total):

| Copy from (branch `SAE`) | Copy to | Size |
|---|---|---|
| `results/ae/ckpt/ae_b8/<block>.pt`, 8 files | `<root>/ckpt/ae_b8/<block>.pt` | 27 MB |
| `results/ae/runs/R_ae_b8_catboost/model.cbm` | `<root>/R.cbm` | 0.6 MB |
| `results/ae/runs/Z_ae_b8_catboost/model.cbm` | `<root>/Z.cbm` | 1.5 MB |
| `results/ae/runs/ZR_ae_b8_catboost/model.cbm` | `<root>/ZR.cbm` | 1.4 MB |

The encoder checkpoints contain the normalisation constants. You do not need `results/ae/acts/`.

### 5.3 Python packages

`torch`, `transformers`, `catboost`, `numpy`, `pandas`, `pillow`. The tested versions are torch 2.14.1+cu130,
transformers 5.20.2, catboost 1.2.10 and Python 3.14.

## 6. Procedure

1. Copy the code (section 5.1) into the router repository.
2. Copy the artifacts (section 5.2) into a folder `<root>`.
3. Install the Python packages (section 5.3).
4. Make one scorer for each process:

   ```python
   from ae_error.scorer import ErrorScorer
   scorer = ErrorScorer("ZR", root="<root>")          # or "R", or "Z"
   ```

5. If the router already has a B0 `SegModel`, give it to the scorer. This prevents a second copy of B0:

   ```python
   scorer = ErrorScorer("ZR", root="<root>", b0=router_b0)
   ```

6. Replace the router's B0 call with the scorer call:

   ```python
   out = scorer(small)              # small: (3, H, W) uint8, working resolution
   pred, score = out["pred"], out["score"]
   ```

7. Use `out["pred"]` and `out["probs"]` as the B0 result. Do not run B0 a second time.
8. Do the acceptance test (section 9).

## 7. Cautions and limits

CAUTION: Give the scorer one frame at a time. `MixFFNTap` accepts only a batch of 1. It stops with an
assertion if the batch is larger.

CAUTION: Use the working resolution of section 2. The models know only 1024 × 512 (Cityscapes) and
1024 × 576 (ACDC). The `row` and `col` features are patch positions. Thus, a different resolution changes
their meaning.

CAUTION: H and W must be multiples of 16. If not, `per_patch` stops with an assertion.

CAUTION: Use B0 in fp32. We did not test fp16 or other B0 checkpoints. If you change B0, you must train the
encoders and CatBoost again (section 11).

- The score predicts the **B0 error** in the patch. It does not predict the gain from a heavy model. If the
  router needs the gain, train a small model on top of the score or the features.
- The models are fitted on Cityscapes only: AE on 2,975 train frames, CatBoost on 300 val frames (P0). Do
  not use the Cityscapes P0 frames to test the router. The list is in `splits/cityscapes_p0.txt`.
- The scorer gives a score to all patches, also to patches without ground truth.
- We tested the scorer only on CUDA.

## 8. Adapters to router units

The router can use a different unit than the 16 × 16 patch. Use these adapters (tested):

```python
import numpy as np

def to_pixels(score, patch=16):
    """(Hp, Wp) patch scores -> (H, W) pixel map."""
    return np.repeat(np.repeat(score, patch, axis=0), patch, axis=1)

def region_stats(pixel_map, regions, n):
    """regions: (H, W) int unit ids 0..n-1 (components, tiles, ...). Mean, max and expected wrong pixels."""
    r, v = regions.ravel(), pixel_map.ravel()
    area = np.bincount(r, minlength=n)
    total = np.bincount(r, weights=v, minlength=n)
    peak = np.full(n, -np.inf)
    np.maximum.at(peak, r, v)
    return {"mean": total / np.maximum(area, 1), "max": peak, "expected_errors": total}
```

| Router unit | What to do |
|---|---|
| 16 × 16 patch | Use `score` directly. |
| Square tile (multiple of 16, for example 256) | Make a tile id map. Apply `region_stats`. A 256 tile holds 16 × 16 patches. |
| Connected component or other region | Apply `region_stats` with the component map. |
| Frame | Use `score.mean()` or the sum of `expected_errors`. |

`expected_errors` is the predicted number of wrong B0 pixels in the unit. Use it if the router sorts units
under a pixel budget. Use `mean` if the router sorts units of equal size.

To give the features to a learned router, aggregate the N, Z and R columns the same way. Reshape each column
to (Hp, Wp). Then apply `to_pixels` and `region_stats`.

## 9. Acceptance test

Do this test after the installation. It compares the scorer with the saved test scores of branch `SAE`.

1. Copy `results/ae/runs/<set>_ae_b8_catboost/scores.parquet` for R, Z and ZR to a test folder.
2. Load a frame and make it the same way as the B0 pipeline:

   ```python
   img = Image.open(path).convert("RGB").resize((1024, 512), Image.BILINEAR)   # ACDC: (1024, 576)
   small = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
   ```

3. Get `out = scorer(small)`.
4. Get the rows of `scores.parquet` for this `image_id`. For ACDC, `image_id` is `<condition>_<frame>`.
5. Compare `out["score"].ravel()[rows.patch]` with `rows.score`.
6. Make sure that the maximum difference is less than 1e-6.

Our result (2026-10-28, outside the repository, with only the copied code): the maximum difference was
≤ 3e-8 for R, Z and ZR. We tested Cityscapes frame `frankfurt_000000_000576` and ACDC frame
`night_GOPR0351_frame_000033`. The difference comes from the float32 output.

## 10. Which set to use, and what it costs

Quality: the scorers are fitted on Cityscapes P0 and tested on P1 and ACDC val (RESULTSV2.md, RESULTSv2.1.md).
- Recall@10% is the share of wrong pixels in the top 10% of patches.
- Robust AUROC is the y50 AUROC inside entropy deciles 4–9.

| Set | P1 recall@10% | ACDC recall@10% | P1 robust AUROC | ACDC robust AUROC |
|---|---|---|---|---|
| entropy only | 0.552 | 0.300 | 0.571 | 0.561 |
| R | 0.564 | 0.313 | 0.743 | 0.737 |
| Z | 0.563 | 0.304 | 0.731 | 0.642 |
| ZR | 0.566 | 0.319 | 0.744 | 0.687 |

Time per frame: RTX 3060 Ti, fp32, batch 1, 1024 × 512, mean of 20 frames:

| Part | R | Z | ZR |
|---|---|---|---|
| B0 only | 17 ms | 17 ms | 17 ms |
| B0 + hooks + features | 45 ms | 45 ms | 45 ms |
| Full scorer (+ CatBoost) | 49 ms | 64 ms | 65 ms |

Recommendation:
- **ZR**: the best recall on P1 and ACDC. Use it as the default.
- **R**: the best ranking inside the same entropy level under domain shift. It is also the fastest.
- **Z**: no advantage over ZR at the same cost.

## 11. Train again (only if B0, the resolution or the data change)

Run these commands on branch `SAE`. Then copy the new artifacts (section 5.2).

```bash
uv run python -m ae.collect
uv run python -m ae.train
uv run python -m sae.patch_features --dataset cityscapes --sae none
uv run python -m ae.patch_features --dataset cityscapes --encoders ae_b8
uv run python -m ae.classify cv --features ZR --encoder ae_b8 --model catboost
uv run python -m ae.classify test --features ZR --encoder ae_b8 --model catboost
```

NOTE: `ae.classify` reads F0 and the labels from the v1 base tables. `sae.patch_features --sae none` writes
only these tables (it does not need the SAE). Run it before `ae.patch_features`. Repeat the last two commands
for R and Z.
