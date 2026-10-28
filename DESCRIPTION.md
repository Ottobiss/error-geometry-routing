# ZR pipeline: description

Written in ASD-STE100 (Simplified Technical English). Status: 2026-10-28, branch `SAE`. This document tells
how the ZR patch error score is made: the data, the transforms, the models and the data flow. For the
installation in a router, see `INTEGRATION.md`. For the results, see `REPORT.md`.

## 1. Scope

The ZR pipeline gives one score for each 16 × 16 pixel patch of the SegFormer-B0 output. The score is the
predicted fraction of wrong B0 pixels in the patch. It goes from 0 to 1.

The name ZR comes from the features: F0 (B0 output), N (activation norm), Z (autoencoder codes) and
R (autoencoder reconstruction error). One CatBoost model uses the 547 features.

The pipeline has four stages:

| Stage | Name | When | Labels | Data |
|---|---|---|---|---|
| A | Encoder fit | Offline, one time | No | Cityscapes train |
| B | Patch tables | Offline, one time | Yes (for targets) | Cityscapes val (P0, P1), ACDC val |
| C | Classifier fit | Offline, one time | Yes | Cityscapes P0 |
| D | Inference | For each frame | No | Any frame |

## 2. Data flow

```
STAGE A: ENCODER FIT (Cityscapes train, 2,975 frames, no labels)
  frame 2048 × 1024 ──resize──> 1024 × 512 ──> B0 + hooks ──> 8 activation maps (h, w, d)
    ──random positions──> results/ae/acts/<block>.npy (fp16)
    ──fit──> normalisation (mean, scale) for each block
    ──train──> bottleneck AE for each block ──> results/ae/ckpt/ae_b8/<block>.pt

STAGE B: PATCH TABLES (P0, P1, ACDC val)
  frame ──resize──> B0 + hooks ─┬─> softmax (19, H, W) ──> F0 (11 per patch)  ─┐
                                │   GT ──> n_valid, n_err (labels)            ├─> results/sae/patches/base/
                                └─> 8 activation maps ──normalise──> x        ─┘
                                      ├─> N = ‖x‖² / d ──pool──> 12 per patch   ──> results/ae/patches/norm/
                                      └─> AE: z = enc(x), x̂ = dec(z)
                                            ├─> Z = z ──pool──> 512 per patch    ─┐
                                            └─> R = ‖x − x̂‖² / d ──pool──> 12   ─┴─> results/ae/patches/ae_b8/

STAGE C: CLASSIFIER FIT (P0, 300 frames)
  P0 patch table ──rows with n_valid > 0──> 569,346 patches
    ──training rows──> 246,107 patches (all with errors, 25% of clean, weights)
    ──5-fold CV by frame──> number of trees (1,225)
    ──fit on all training rows──> results/ae/runs/ZR_ae_b8_catboost/model.cbm

STAGE D: INFERENCE (one frame)
  frame ──resize──> B0 + hooks ──> F0, N, Z, R (547 columns × 2,048 patches) ──> CatBoost ──> score (32 × 64)
```

## 3. Input data

### 3.1 Datasets

| Data | Frames | Use | Code |
|---|---|---|---|
| Cityscapes train | 2,975 | Stage A only. The pipeline does not read the train labels. | `Cityscapes(split="train")` |
| Cityscapes P0 | 300 (val) | Stage C: fit and selection | `splits/cityscapes_p0.txt` |
| Cityscapes P1 | 200 (val) | Test, in domain | `splits/cityscapes_p1.txt` |
| ACDC val | 406: fog 100, night 106, rain 100, snow 100 | Test, domain shift | `splits/acdc_<condition>.txt` |

NOTE: B0 was fine-tuned on Cityscapes train. Thus, the encoders learn the activations of the B0 training
distribution. P0, P1 and ACDC are all out of sample for the encoders.

NOTE: There is no `splits/cityscapes_train.txt`. `core.data.test_populations` uses all split files except P0
as test populations. Thus, stage A reads the train frames with `split="train"`.

### 3.2 Frame transform

1. Read the frame as RGB uint8.
2. Resize it to the working resolution with PIL bilinear:
   - Cityscapes: 2048 × 1024 → 1024 × 512;
   - ACDC: 1920 × 1080 → 1024 × 576.
3. In B0, subtract the ImageNet mean (0.485, 0.456, 0.406) × 255 and divide by the ImageNet standard
   deviation (0.229, 0.224, 0.225) × 255.

### 3.3 Label transform (stages B and C only)

1. Resize the ground truth to the working resolution with nearest neighbour.
2. Cityscapes: change `labelIds` to the 19 `trainIds`. Other labels become 255 (ignore). ACDC: the files
   already contain `trainIds`.
3. A pixel is valid if its label is not 255.
4. A pixel is wrong if it is valid and the B0 argmax is not equal to the label.

## 4. B0 and the hooks

### 4.1 Model

`nvidia/segformer-b0-finetuned-cityscapes-512-1024`, transformers 5.x, fp32, eval mode. The weights are
frozen. The pipeline does not train B0.

B0 gives logits at 1/4 of the working resolution. `SegModel` resizes them to the working resolution with
bilinear interpolation (`align_corners=False`). Then softmax gives `probs` (19, H, W).

### 4.2 Activations

The pipeline uses the Mix-FFN hidden activation of each of the 8 encoder blocks. Each block has
Mix-FFN = fc2 ∘ GELU ∘ DWConv 3 × 3 ∘ fc1. The activation is the input of `fc2`: after the depthwise
convolution and GELU.

- `MixFFNTap` puts a forward pre-hook on `stages[s].blocks[j].mlp.fc2`. The hook keeps the input tensor
  (1, N, 4C).
- A forward hook on `stages[s].patch_embeddings` gives the grid size (h, w) of the stage.
- The hook changes the tensor to (h, w, d), d = 4C, in raster order.
- The hooks do not change the B0 output (max |Δ logits| = 0).

| Block | Stride | d | Grid, Cityscapes | Grid, ACDC | AE code size b |
|---|---|---|---|---|---|
| s1b1, s1b2 | 4 | 128 | 128 × 256 | 144 × 256 | 16 |
| s2b1, s2b2 | 8 | 256 | 64 × 128 | 72 × 128 | 32 |
| s3b1, s3b2 | 16 | 640 | 32 × 64 | 36 × 64 | 80 |
| s4b1, s4b2 | 32 | 1,024 | 16 × 32 | 18 × 32 | 128 |

## 5. Stage A: encoder fit

### 5.1 Activation collection (`ae.collect`)

1. B0 with hooks runs on all 2,975 train frames, in sorted `image_id` order.
2. For each frame and block, the pipeline takes positions of the native grid:
   - stages 1–3: 672 random positions (2,000,000 / 2,975, seed 0);
   - stage 4: all 512 positions.
3. It stores the activations as fp16 in `results/ae/acts/<block>.npy`, with the frame number of each position.
4. It holds out 149 frames (5%, seed 0). The other 2,826 frames are the training frames.

Result: 1,999,200 positions for each block of stages 1–3 and 1,523,200 for stage 4.

### 5.2 Normalisation

For each block, the pipeline fits two constants on the training frames:
- `mean`: the mean activation (d values);
- `scale`: one number, so that E‖x‖² = d for x = (h − mean) / scale.

The constants go into each encoder checkpoint. Stages B and D use the same constants for all data.

### 5.3 Bottleneck autoencoder (`ae.model.BottleneckAE`, `ae.train`)

One AE for each block:

```
x (d) ──Linear(d, d)──GELU──Linear(d, b)──> z (b) ──Linear(b, d)──GELU──Linear(d, d)──> x̂ (d)
```

| Parameter | Value |
|---|---|
| Code size b | d / 8 |
| Loss | MSE between x̂ and x |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | Cosine over 30 epochs, one step for each batch |
| Batch | 4,096 positions, random order for each epoch |
| Stop | 3 epochs without held-out improvement (it did not occur) |
| Kept weights | The epoch with the lowest held-out MSE (epoch 30 in all blocks) |
| Seed | 0 |

The code z is dense and signed. The AE uses no sparsity.

Held-out FVE (fraction of variance explained): 0.78–0.89. The table for each block is in `RESULTSV2.md` §1.

Output: `results/ae/ckpt/ae_b8/<block>.pt`. Each file contains the weights, `mean`, `scale`, the
configuration and the training history. The 8 files are 27 MB.

## 6. Stage B: patch tables

### 6.1 Patch grid

- A patch is 16 × 16 pixels of the working resolution.
- Cityscapes: 32 × 64 = 2,048 patches for each frame. ACDC: 36 × 64 = 2,304.
- Patch index = row · Wp + column (raster order). Wp = 64.

### 6.2 Activation-to-patch rule (`sae.hooks.selection_index`, `pool_to_patches`)

Position k of a stage with stride r has its centre at output pixel r · k. For patch row u (columns are the same):

| Stage | Stride | Positions for one patch | Rule |
|---|---|---|---|
| 1 | 4 | 4 × 4 = 16 | All positions with the centre in the patch: k = 4u … 4u + 3 |
| 2 | 8 | 2 × 2 = 4 | k = 2u, 2u + 1 |
| 3 | 16 | 1 | k = u |
| 4 | 32 | 1 | k = round((16u + 7.5) / 32), clamped to the grid. One position serves two patches. |

Stages 1–2 then pool the positions (mean or max, see 6.4). Stages 3–4 use the single value.

### 6.3 F0 and labels (`sae.patch_features`, base tables)

For each patch, from `probs` and the B0 argmax `pred`:

| Column | Definition |
|---|---|
| `ent_mean`, `ent_max`, `ent_p90` | Entropy −Σ p log p (natural log) of each pixel: mean, max, 90th percentile |
| `msp_mean`, `msp_min` | Max softmax of each pixel: mean, min |
| `margin_mean`, `margin_min` | Top-1 minus top-2 probability: mean, min |
| `boundary_frac` | Fraction of pixels with more than one predicted class in the 3 × 3 neighbourhood (edge padding) |
| `row`, `col` | Patch row and column |
| `cls` | Majority predicted class (0–18). Ties go to the smallest class number. Categorical. |
| `n_valid`, `n_err` | Number of valid pixels and of wrong pixels (labels, not features) |

Output: `results/sae/patches/base/<dataset>/<image_id>.npz`. This file also holds the raw activations and
the pixel samples for the pixel AUROC. ZR does not use them.

### 6.4 N, Z and R (`ae.patch_features.frame_features`)

For each block, on all native positions of the frame:

1. Normalise: x = (h − mean) / scale.
2. N for each position: ‖x‖² / d.
3. Encode: z = enc(x). Decode: x̂ = dec(z).
4. R for each position: ‖x − x̂‖² / d.
5. Move the values to the patch grid (6.2):

| Feature | Stages 1–2 | Stages 3–4 | Columns |
|---|---|---|---|
| N | Mean and max over the patch positions | The one value | 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1 = 12 |
| R | Mean and max over the patch positions | The one value | 12 |
| Z | Mean over the patch positions (b values) | The one code | 16 + 16 + 32 + 32 + 80 + 80 + 128 + 128 = 512 |

Column names: `<block>_n_mean`, `<block>_n_max` (stages 1–2), `<block>_n` (stages 3–4); the same with `r`;
`<block>_z<j>`. All columns are in the block order s1b1 … s4b2.

NOTE: N does not use the encoder. The pipeline stores it one time for all encoders.

Output:
- `results/ae/patches/ae_b8/<dataset>/<image_id>.npz`: Z (n_patches, 512) and R (n_patches, 12), float32;
- `results/ae/patches/norm/<dataset>/<image_id>.npz`: N (n_patches, 12), float32.

## 7. Stage C: classifier fit (`ae.classify`)

### 7.1 Rows

1. Load the P0 patch tables. Remove the patches without valid pixels. Result: 569,346 patches.
2. Target: `err_frac` = n_err / n_valid.
3. Training rows (`sae.classify.training_rows`, seed 0):
   - all patches with at least one wrong pixel;
   - a random 25% of the clean patches.

   Result: 246,107 patches.
4. Weight: n_valid. Clean patches get an additional weight × 4. This compensates for the 25% sample.

### 7.2 Feature matrix

The columns are in this order: F0 (11), N (12), Z (512), R (12). Total: 547 columns.
`cls` is a categorical column (int32). All other columns are float32.

### 7.3 CatBoost

| Parameter | Value |
|---|---|
| Loss | CrossEntropy. It accepts targets from 0 to 1. |
| Trees | Symmetric (oblivious), depth 6 |
| Learning rate | 0.05 |
| L2 regularisation | 3 |
| Categorical column | `cls`, with the CatBoost ordered target statistics |
| Seed | 0 |
| Device | CPU |

CatBoost quantises the P0 pool one time. The quantisation does not use labels.

### 7.4 Number of trees

1. Split the P0 frames into 5 folds: GroupKFold by frame, shuffle, seed 0. All patches of one frame are in one
   fold.
2. For each fold, train on the other 4 folds. Stop after 200 iterations without improvement on the held-out
   fold, with a limit of 10,000.
3. Best iterations: 1,465, 2,403, 370, 1,015 and 873. The mean is 1,225.
4. The out-of-fold scores give the P0 metrics.

### 7.5 Final model

Train one model on all 246,107 training rows with 1,225 trees and no early stopping.
Output: `results/ae/runs/ZR_ae_b8_catboost/model.cbm` (1.4 MB).

## 8. Stage D: inference

For one frame (`ErrorScorer` in `INTEGRATION.md`):

1. Resize the frame to the working resolution (3.2).
2. Run B0 one time with the hooks. Keep `probs` and the 8 activation maps.
3. Calculate F0 from `probs` and `pred` (6.3).
4. Calculate N, Z and R from the activations (6.4).
5. Put the 547 columns in the order of 7.2.
6. Apply the CatBoost model. Get the probability.
7. Change the shape of the scores to (Hp, Wp): 32 × 64 for Cityscapes, 36 × 64 for ACDC.

The inference does not use labels and does not use a heavy model. B0 runs only one time. Its prediction is
the usual B0 output.

Data for one Cityscapes frame:

| Step | Shape | Type |
|---|---|---|
| Input frame | (3, 512, 1024) | uint8 |
| B0 logits (1/4) | (19, 128, 256) | float32 |
| `probs` | (19, 512, 1024) | float32 |
| Activations | (128, 256, 128) … (16, 32, 1024), 8 maps | float32 |
| Feature matrix | (2,048, 547) | float32 + int32 |
| Score | (32, 64) | float32 |

Time on RTX 3060 Ti, fp32, batch 1: B0 only 17 ms; B0 with hooks and features 46 ms; full ZR scorer 68 ms.

## 9. Evaluation

The evaluation uses only patches with valid pixels.

- Global metrics: recall at a patch budget of 5 / 10 / 20 / 30%, AUROC and AP for y50 (patch at least 50%
  wrong) and y_any (patch at least 10% wrong).
- Inside-entropy metric (Robust): the y50 AUROC inside each P0 entropy decile 4–9, then the mean.
- Tests: P1 and ACDC val, with the model of 7.5. The pipeline does not use P1 or ACDC for any choice.
- Comparisons: paired bootstrap over frames, 2,000 resamples, seed 0.

ZR results:

| Population | rec@10% | AUROC y50 | AP y50 | Robust |
|---|---|---|---|---|
| P0 (out-of-fold) | 0.571 | 0.922 | 0.319 | 0.765 |
| P1 | 0.566 | 0.917 | 0.343 | 0.744 |
| ACDC all | 0.319 | 0.833 | 0.515 | 0.687 |

For entropy alone: P1 rec@10% 0.552, ACDC 0.300. ZR − F0 on ACDC rec@10%: +0.026 [+0.021, +0.031].
All tables are in `REPORT.md` §5.

## 10. Files

| Path | Stage | Content |
|---|---|---|
| `results/ae/acts/<block>.npy`, `_frame.npy`, `_norm.npz`, `meta.json` | A | Training activations, frame numbers, normalisation, held-out frames |
| `results/ae/ckpt/ae_b8/<block>.pt` | A | AE weights and normalisation |
| `results/ae/ckpt/ae_b8/train.csv` | A | Training summary |
| `results/sae/patches/base/<dataset>/<image_id>.npz` | B | F0, labels, raw activations, pixel samples |
| `results/ae/patches/norm/<dataset>/<image_id>.npz` | B | N |
| `results/ae/patches/ae_b8/<dataset>/<image_id>.npz` | B | Z, R |
| `results/ae/runs/ZR_ae_b8_catboost/cv_*` | C | CV configuration, out-of-fold scores, P0 metrics |
| `results/ae/runs/ZR_ae_b8_catboost/model.cbm`, `config.json` | C | Final model and its configuration |
| `results/ae/runs/ZR_ae_b8_catboost/scores.parquet`, `metrics.json` | C | P1 and ACDC scores and metrics |

`results/` is in `.gitignore`. The files are not in git.

## 11. Commands

Run them from the repository root, on branch `SAE`, in this sequence:

```bash
uv run python -m download.cityscapes --prefix leftImg8bit/train/
uv run python -m ae.collect
uv run python -m ae.train
uv run python -m sae.patch_features --dataset cityscapes --sae none
uv run python -m sae.patch_features --dataset acdc --sae none
uv run python -m ae.patch_features --dataset cityscapes --encoders ae_b8
uv run python -m ae.patch_features --dataset acdc --encoders ae_b8
uv run python -m ae.classify cv --features ZR --encoder ae_b8 --model catboost
uv run python -m ae.classify test --features ZR --encoder ae_b8 --model catboost
```

## 12. Cautions and limits

CAUTION: Use the working resolution of 3.2. The features `row` and `col` are patch positions. A different
resolution changes their meaning.

CAUTION: ACDC has 36 patch rows. Cityscapes has 32. Thus, ACDC rows 32–35 are outside the training range of
`row`. CatBoost puts them into its highest `row` bin.

CAUTION: Give B0 one frame at a time. `MixFFNTap` accepts only a batch of 1.

CAUTION: If you change B0, the resolution or the transformers version, do stages A to C again.

- The score predicts the B0 error. It does not predict the gain from a heavy model.
- The AE was trained on fp16 activations. Stages B and D give fp32 activations to the AE.
- The AEs did not converge: the best epoch was the last epoch in all blocks.
- All fits use one seed.
- The score is a weighted estimate of the error fraction. We did not check its calibration.
