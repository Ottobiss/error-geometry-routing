# PLAN: patch-level error prediction for SegFormer-B0 from SAE codes of Mix-FFN activations

Draft from 2026-10-26, branch `SAE`.

## 1. Question

Can sparse-autoencoder (SAE) codes of B0's inner activations predict **which patches of the final
segmentation map are wrong** better than B0's output confidence? This includes the patches where B0 is
confidently wrong, which entropy and MSP miss by construction.

**The inner activations are the Mix-FFN hidden activations after GELU**, one vector per spatial
position. **Each patch is described by the activations located closest to it on the output map.**

This is a dense, patch-level version of Ma et al., "Medical AI Encodes a 'Feeling of Error'" (ECCV 2026,
arXiv 2609.08879). Their pipeline is: one SAE per layer on patch tokens of a frozen ViT, then the layer codes
are concatenated, then gradient-boosted trees. Their classifier is image-level (Dice ≤ τ). The patch-level
step there is used only for false-positive removal and is not validated. Here the classifier is trained and
evaluated directly on patch labels.

What we already know (EXPERIMENTS.md, METHODOLOGY.md §1): in-domain (CS P1) no B0 score beats entropy,
including trained regressors on geometry features. Pixel AUROC of "B0 pixel wrong" on P1 is 0.934 for MSP
and entropy. The bar for this experiment is therefore **the same classifier on output features only**,
not raw entropy.

## 2. Setup

| | Value |
|---|---|
| Model | `nvidia/segformer-b0-finetuned-cityscapes-512-1024` (`core.models.CHECKPOINTS["b0"]`): ImageNet-1K pretrained MiT-B0 encoder, fine-tuned on Cityscapes train. Frozen, eval mode |
| Working resolution | CS 1024×512, ACDC 1024×576 (`core/data.py`), same preprocessing as `SegModel`. The final segmentation map is at this resolution: 1/4 logits upsampled bilinearly |
| Fit data | **P0** = 300 random Cityscapes val frames (`splits/cityscapes_p0.txt`). Used for the SAEs, the classifier and all model selection |
| Selection | GroupKFold by `image_id` inside P0, 5 folds, seed 0 |
| Tests | **P1** = the other 200 Cityscapes val frames (`splits/cityscapes_p1.txt`, in domain); ACDC fog / night / rain / snow / all (406 frames, shift) |
| Ignore | GT label 255 pixels are excluded from labels and metrics; patches with no valid pixels are dropped |

All fitting uses P0 only. P1 and ACDC are not used for any choice. B0 never saw any val frame in training.
The SAEs are unsupervised and see all P0 frames, including the classifier's held-out folds. They never see
labels, so this is accepted and noted.

## 3. Activations: Mix-FFN hidden layer after GELU

Each B0 block (4 stages × 2 blocks) contains:
`x = x + MixFFN(LN2(x))`, where `MixFFN = fc2 ∘ GELU ∘ DWConv3×3 ∘ fc1`.

The activation used is **the input of `fc2`**: the hidden layer after the 3×3 depthwise conv and GELU.
Dropout is identity in eval mode. For every block it is a `(B, N, 4C)` tensor on the stage's token grid.
One **sample is one spatial position: a 1×1×4C vector**.

Why this layer:
- It is 4× wider than the residual stream, so even stages 1–2 give 128 / 256 dimensions.
- It is post-nonlinearity: the block's "neurons".
- The depthwise conv mixes each channel with its 3×3 neighbours, so it carries local texture and edge
  context.

| Block | Stride | Hidden d = 4C | Grid CS (H×W) | Positions / CS frame | SAE D = 2d |
|---|---|---|---|---|---|
| s1b1, s1b2 | 4 | 128 | 128×256 | 32,768 | 256 |
| s2b1, s2b2 | 8 | 256 | 64×128 | 8,192 | 512 |
| s3b1, s3b2 | 16 | 640 | 32×64 | 2,048 | 1,280 |
| s4b1, s4b2 | 32 | 1,024 | 16×32 | 512 | 2,048 |

ACDC grids are 144/72/36/18 × 256/128/64/32. Total SAE features over the 8 blocks at D = 2d: **8,192**.
Raw hidden activations over the 8 blocks: 4,096.

**Hooks.** Use a forward **pre**-hook on each block's Mix-FFN output linear. It is called `fc2` in the
NVIDIA code and `dense2` in transformers 4.46; check the name in transformers 5.20.2 (the repo already uses
the 5.x names `stages[s].blocks[j]`, `attention.q_proj`). The hook receives `(B, N, 4C)` in raster order;
reshape with the stage's (H, W).

**Comparison hook sets (ablations only):**
- the residual stream after each block, output of `stages[s].blocks[j]` (d = C);
- the decoder fused map `C`, the input of `decode_head.classifier` (256-d at 1/4).

## 4. SAEs

### Reference configuration

The reference configuration follows Ma et al. 2026 (§4.1, Tab. 2).

| Parameter | Value |
|---|---|
| Architecture | BatchTopK (Bussmann et al. 2024): ReLU encoder, linear decoder, `z = E(h)`, `ĥ = W_dec z + b_dec` |
| Dictionary size | D = 2·d (paper: 1,536 = 768 × 2) |
| Sparsity | k = 8 active latents per position on average (BatchTopK over the batch) |
| Optimiser | Adam, lr 1e-4 |
| Epochs | 5 |
| Ablation grid (paper) | D ∈ {2d, 4d} × k ∈ {4, 8, 16}; paper optimum D = 2d, k = 8 |

### Additions (standard practice, not in the paper)

- **One SAE per block,** trained independently on that block's **native positions**. Coarse positions are
  never duplicated onto a finer grid.
- **Input normalisation per block:** subtract the P0 mean and scale so that E‖h‖² = d. Store the constants
  with the checkpoint. GELU outputs are bounded below (≈ −0.17), so they are not sign-symmetric; the mean
  subtraction handles this.
- **Decoder:** columns are kept at unit norm. `b_dec` is initialised at the activation mean.
- **Dead latents:** AuxK loss (Gao et al. 2024).
- **Inference:** use the learned global threshold θ of BatchTopK. Each position is then encoded
  independently, and the number of active latents varies per position.
- **Batch size:** 4,096 positions.
- **Training positions:** at most 2M random positions per block from P0. In practice this caps stages 1–2.
  Stage 3 has 614k positions and stage 4 has 154k, all of which are used. Hold out 5% of P0 frames for SAE
  validation.
- **Stages 3–4:** these combine large d (640 / 1,024) with few positions. An extra variant with D = 1·d is
  added for them.

### Quality gates

These are checked before features are built and reported for each block:

- Fraction of variance explained (FVE) on held-out positions, mean L0, and the share of dead latents (never
  fire on held-out positions). Gate: dead latents < 10%.
- **Splice test:** replace one block's `fc2` input with its SAE reconstruction (the pre-hook returns ĥ) and
  measure mIoU on P0. Gate: drop ≤ 1.0 point.
- Stage 4 has few positions, so 5 epochs is only about 190 steps. If held-out FVE is still rising, train
  longer and record it.

## 5. Patches of the output map and the closest activations

### Patch grid

Patches are 16×16 px on the final segmentation map at working resolution: 32×64 = 2,048 patches per CS
frame and 36×64 = 2,304 per ACDC frame. Patch (u, v) covers pixels `[16u, 16u+16) × [16v, 16v+16)`.
Ablation: 32 px.

### Where an activation sits on the output map

Input and final map share the same pixel grid (working resolution). With B0's conv paddings (7×7/s4/p3,
then 3×3/s2/p1), the position k of a stage with stride r is centred at output pixel `r·k`. This is the
top-left corner of its stride cell, not the middle. The Mix-FFN depthwise conv is 3×3/s1/p1, so it keeps
these centres.

### Selection rule: closest activations only

For each patch and each block, take the activations whose centres are **closest to the patch**:

- **Stride < patch** (stages 1–2): all positions whose centre lies inside the patch — 4×4 for stage 1, 2×2
  for stage 2. Their SAE codes are **max-pooled**.
- **Stride = patch** (stage 3): the one position at k = u, centred at the patch's top-left corner (7.5 px
  from its centre).
- **Stride > patch** (stage 4): the single position nearest to the patch centre,
  `k = round((16u + 7.5) / 32)`, clamped to the grid. Stage-4 position k then serves patches 2k−1 and 2k.

The patch's context is whatever these closest activations already encode. Their receptive fields are large:
every block has global attention over a 16×32 key/value grid. **No neighbouring patches' activations are
added in the main configuration.**

Ablations:
- **Floor alignment** (`k = u // 2` for stage 4), to quantify the half-cell offset.
- **+ ring:** also max-pool the codes of the 8 neighbouring patches, to test whether explicit wider context
  helps.

Implementation: encode each block once per frame at native resolution. Then max-pool or gather by index into
the patch grid. Never materialise upsampled tensors.

### Labels

Per patch:
- `n_valid`: number of non-ignore pixels.
- `err_frac`: fraction of valid pixels where the B0 prediction ≠ GT. GT is resized with nearest-neighbour to
  working resolution, as in `core/data.py`.
- Derived binaries: `y50 = err_frac ≥ 0.5` (mostly wrong) and `y_any = err_frac ≥ 0.1`.

## 6. Features and classifier

### Feature sets

Feature sets are nested where possible, following the METHODOLOGY §5 convention.

| Set | Contents |
|---|---|
| **F0, control** | Output features of B0 per patch: entropy mean / max / p90, MSP mean / min, margin mean / min, majority predicted class (**categorical**), boundary fraction (pixels whose 3×3 neighbourhood has > 1 predicted class), patch row / column |
| **F1 = F0 + SAE** | F0 plus SAE codes of the 8 Mix-FFN hidden layers, selected as in §5 (8,192 columns at D = 2d) |
| **F2, SAE only** | SAE codes only |
| **F3, raw only** | Raw Mix-FFN hidden activations, same selection and pooling, no SAE (4,096 columns) |
| **F3 + F0** | Raw activations plus output features: the control for F1 |

### Classifier: CatBoost

- **Objective:** `loss_function="CrossEntropy"` (accepts targets in [0, 1]) on `err_frac`, with sample
  weight `n_valid`.
- **Categorical feature:** the predicted class goes in as a native categorical feature (`cat_features`).
  CatBoost handles it with ordered target statistics, so no one-hot encoding is needed.
- **Hyperparameters:** depth 6, learning_rate 0.05, l2_leaf_reg 3, up to 3,000 iterations with early
  stopping on the CV fold (`od_type="Iter"`, `od_wait=200`, `use_best_model=True`), `random_seed=0`,
  `thread_count=-1`. Start on CPU and switch to `task_type="GPU"` only if the pool fits in 8 GB.
- **Data size and format:**
  - P0 has about 614k patches. For training, keep all patches with `err_frac > 0` and a random 25% of
    clean patches reweighted ×4.
  - Codes are stored as scipy CSR and passed to `Pool` as sparse data. Max-pooled codes have about 300
    non-zeros out of 8,192 per patch.
  - Latents that fire in < 0.1% of P0 patches are dropped before training. This is unsupervised, so it does
    not leak labels.
  - If the sparse `Pool` path rejects the categorical column, pass the predicted class as one-hot columns
    instead and record it.
- **The same CatBoost setup is used for every feature set,** so differences come from features only.

## 7. Evaluation

### Metrics

Metrics are computed on P1 and ACDC (each condition and all together), pooled over frames:

- **Primary:** wrong-pixel recall at a 10% patch budget, i.e. the share of all wrong valid pixels inside the
  top-10% scored patches of the population (global budget, as in the routing protocol). Also reported at
  5 / 20 / 30%.
- AUROC and AP for `y50` and `y_any`.
- **Confidently wrong:** AUROC for `y50` inside each decile of patch mean entropy, using deciles fit on P0,
  averaged over deciles. Also AUROC on the lowest-entropy quintile only. This answers "does it find errors
  entropy cannot see" (METHODOLOGY §6.1).
- **Pixel AUROC (descriptive):** broadcast the patch score to its pixels and evaluate on the same 5% of P1
  pixels as the existing protocol, beside MSP / entropy 0.934.

### Pre-registered main comparisons

Paired bootstrap over frames, 2,000 replicates, seed 0, 95% interval of the difference. There is no
multiplicity correction, so results at the edge of the interval are marked as borderline.

1. P1, recall@10%: F1 − F0.
2. ACDC all, recall@10%: F1 − F0.
3. P1, recall@10%: F2 − F3 (does the SAE add anything over raw Mix-FFN activations).
4. P1, entropy-decile-averaged AUROC (y50): F1 − F0.

Everything else is descriptive. Before the first ACDC run, write a short EXPERIMENTS.md entry fixing the
chosen configuration and these comparisons.

## 8. Ablations (P0 CV only, used for selection)

- **SAE:** D ∈ {2d, 4d} × k ∈ {4, 8, 16}, plus D = 1d for stages 3–4.
- **Hook set:** Mix-FFN hidden (main) / residual stream after each block / Mix-FFN + decoder `C`.
- **Depth groups:** stages 1–2 / stage 3 / stage 4 / all 8 blocks.
- **Pooling for stride < patch:** max (main) / mean / max + mean.
- **Selection:** closest (main) / floor alignment / closest + 8-neighbour ring.
- **Patch size:** 16 (main) / 32.
- **Classifier:** CatBoost / logistic regression on standardised features (paper Tab. 4 analogue).

The configuration with the best P0-CV recall@10% for F1 is frozen and then evaluated once on P1 and ACDC.

## 9. Interpretation (descriptive)

- CatBoost importance (`LossFunctionChange`) mapped back to (block, latent). Top-20 latents shown as
  activation heatmaps on P1 frames next to the B0 error map.
- Per-patch SHAP values (CatBoost's built-in TreeSHAP) for selected confidently-wrong patches: which
  latents in which blocks drive the error score.
- The paper's Finding 1 at patch level: number of active latents and mean activation per patch for wrong vs
  right patches, per block (Cohen's d, P1 and ACDC).
- Error types (METHODOLOGY §6.4): importance and recall separately for boundary patches (boundary fraction
  > 0) and interior patches.

## 10. Optional downstream: routing

Do this only if comparison 1 or 2 is positive. Aggregate patch scores to the existing 256 + 64 tiles (mean
and max over the tile's 16×16 patches) and run `routing.tile_router` against entropy at 10 / 20% budget,
using the standard bootstrap.

## 11. Steps and outputs

| Step | Module (proposed) | Output |
|---|---|---|
| 0. Hooks + sanity | `sae/hooks.py` | `fc2` pre-hook shapes match §3; logits unchanged with hooks; P1 B0 mIoU 71.57 reproduced; overlay figure of patch grid and selected positions for one frame |
| 1. Collect activations | `sae/collect.py` | `results/sae/acts/<block>.npy` (fp16, ≤ 2M positions × 4C), normalisation constants |
| 2. Train SAEs | `sae/model.py`, `sae/train.py` | `results/sae/ckpt/<config>/<block>.pt`; `results/sae/quality.csv` (FVE, L0, dead %, splice ΔmIoU) |
| 3. Patch table | `sae/patch_features.py` | `results/sae/patches/<dataset>/<image_id>.npz`: CSR codes, F0, labels |
| 4. Classifier + CV | `sae/classify.py` | `results/sae/runs/<name>/` with `config.json`, `scores.parquet`, `metrics.json` |
| 5. Frozen tests | same | Main comparisons; entry in EXPERIMENTS.md, new section "SAE" |
| 6. Interpretation | `sae/explain.py` | Figures and tables for §9 |

Dependencies: add `catboost` to `requirements.txt`.

## 12. Risks

- **Stage 3–4 SAEs are underdetermined:** large d (640 / 1,024) with only 614k / 154k positions. Watch the
  train vs held-out FVE gap. Fallback: download CS train frames for unsupervised SAE training only (labels
  are never needed). B0 was trained on those frames, so compare held-out FVE on P0 for both SAEs.
- **Classifier size:** 8,192 SAE columns over ~0.2M training rows. Mitigations are sparse input, rare-latent
  filtering, clean-patch subsampling and the depth-group ablation. If CatBoost still does not fit in memory,
  train on the best two depth groups.
- **Max-pooling saturation:** max over 16 stage-1 positions can make many latents fire in most patches. The
  pooling ablation and the active-latent count will show whether max-pooled codes still separate patches.
- **Position and class confound:** errors cluster by class and image location in Cityscapes. F0 contains
  class and position, so any SAE gain is measured on top of them.
- **Correlated samples:** neighbouring patches are correlated. All intervals are bootstrapped over frames,
  never over patches.
