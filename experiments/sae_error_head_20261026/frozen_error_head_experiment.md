# Error detection from frozen B0 activations

## Question and scope

Do intermediate B0 activations improve detection of B0 errors, especially
low-entropy errors inside predicted components, beyond output-confidence maps?
This is distinct from the earlier selector that regressed signed B2 replacement
gain. Detecting an error does not establish that B2 can fix it.

B0, its segmentation decoder and B2 remain unchanged. The training executable
does not instantiate either segmentator. It reads only cached activations,
confidence maps, B0 predictions and GT. B2 is used only by optional cached routing
evaluation. No dropout, suppression or masking is inserted into B0 itself.
This is a deterministic auxiliary model, not Bayesian inference.

All additions are in `routing/error_head/`, `tests/test_error_head.py`, this
document and `notebooks/frozen_error_head_v1.ipynb`. Previous routing modules,
checkpoints and outputs are not modified.

## Data

Reuse the exported Cityscapes validation cache: 240 images for selector training,
60 for internal validation, 200 for historical P1. Normalization is fitted on
the 240 training images only. Image overlap is rejected; train/validation
sequence overlap is rejected by default. The supplied split manifests have no
train/validation sequence overlap. Historical P1 shares one sequence with train,
and has been inspected many times: it is exploratory and cannot be a new
independent confirmation. No P1 evaluation is run automatically by the notebook.
Use previously unseen labeled images for a final evaluation.

## Training target

For valid pixel p, e_p = 1[B0_p != GT_p]. Ignore ID 255 is excluded.
The prediction grid is the cached confidence grid (128 x 256 for the existing
512 x 1024 masks). Each cell's target is the fraction of erroneous valid pixels;
its loss weight is the fraction of valid pixels. Pool errors and valid pixels
separately before dividing. This retains small errors as soft targets rather
than discarding them with nearest-neighbor label resizing.

Use ordinary binary cross-entropy with logits as the baseline. This is not
segmentation cross-entropy fine-tuning. The best checkpoint minimizes ordinary
validation BCE; early stopping patience is 3, maximum 8 epochs, AdamW lr 0.001,
weight decay 0.0001, seed 0 for the initial screen. Seed-0 screening is not a
robustness conclusion. If a candidate improves validation, repeat the candidate
and learned confidence control with seeds 1 and 2 before spending on live timing.

## Prespecified comparisons

| Run | Inputs | Additional objective |
|---|---|---|
| raw_entropy | Full-resolution cached B0 entropy | No training |
| confidence_bce | Three output-confidence maps | None |
| activations_bce | F2 and F3 | None |
| both_bce | Confidence plus F2/F3 | None |
| both_sparse_ae | Confidence, activation latent code, reconstruction residual | Reconstruction + sparsity |

Each ordinary head uses width 16, per-layer 1x1 activation projections and a
shared 3x3 local classifier. The activation maps are normalized per channel
using training-only statistics. Confidence entropy is divided by log(19).
Input tensors are detached. Parameter counts are recorded; they differ between
input variants, so a successful result should also receive a capacity control
such as `confidence_bce --width 32`. This comparison does not isolate all effects
of model size automatically.

The sparse autoencoder uses a 16-channel nonnegative latent code to reconstruct
the concatenated standardized F2/F3 maps on the common grid. Its error classifier
receives latent features and a per-location squared reconstruction residual,
plus confidence. Total loss is BCE + 0.1*reconstruction_MSE + 0.001*mean_abs_latent.
It is jointly supervised by errors, not purely unsupervised pretraining. Good
reconstruction alone is not evidence of good error detection. No Mahalanobis or
kNN is fitted in this first stage. Geometry of the learned latent representation
is a follow-up only if the learned representation adds error-detection signal.

Optional `--hard-fraction 0.25` selects the hardest quarter of cells separately
among cells with nonzero error target and cells with zero error target. Hardness
is the auxiliary head's BCE, never B0 entropy. This is OHEM-style cell mining,
not an exact implementation of region-based detector OHEM. Ordinary BCE is still
used for validation/checkpoint selection. OHEM outputs should not be interpreted
as calibrated probabilities without checking calibration. It is not part of
the four initial training runs.

## Metrics and decision

Primary: pixel error average precision across all valid pixels. Compare both
activation heads with the SAME trained confidence-only head, and report raw
entropy separately. Secondary: recall at 1%, 5%, 10%, 20% selected pixel coverage,
pixel Brier score for learned heads, and component-level AP. A bad component is
prespecified as >=20% erroneous valid pixels. All 8-connected B0 components are
eligible; entropy never filters candidates. A small erroneous subregion may
not make its whole component bad, so component AP does not replace pixel metrics.

Report separately the interior mask (distance >3 pixels from B0 class boundaries
and image border), low-entropy pixels (H/log(19) <=0.1), and their intersection.
These masks are diagnostics only and never the training sampling gate. Low
entropy is the definition of confidence here; it is not a claim that exact
full-resolution pmax was cached. Interior excludes boundaries of B0's predicted
classes, not GT object boundaries.

AP and recall are streaming approximations with 4096 fixed score bins; ties are
handled as score groups, with uniform-in-bin selection for coverage recall.
Scores are interpolated back to the mask resolution; entropy baseline uses its
original full-resolution map. Report subgroup error prevalence and counts.
Very small subgroup counts should not support a broad claim.

`evaluate --routing` adds a separate cached diagnostic: average the predicted
error score in each B2 window; choose thresholds on internal validation for 0.8
and 2.4 mean windows/image. Component estimates also remain in the report.
Ground truth is never used to choose a window. P1 requires `--exploratory` and
`--calibration` with the immutable validation report and identical checkpoint.
Ties may prevent exact target coverage; actual window counts are reported.
Cached mIoU has NO live-latency measurement. It is not evidence of a speedup or
an equal-total-time comparison. Higher error AP without routing improvement is
a valid, narrower outcome.

Stop this stage if activations do not improve validation error detection over
the trained confidence control. If there is a signal, check seeds/capacity,
then evaluate on unseen data and finally measure full cascade latency. Do not
select a new model, threshold or epoch on P1.

## Commands

```bash
python -m routing.error_head.prepare --train CACHE/train --out NEW/normalization
python -m routing.error_head.train --train CACHE/train --val CACHE/val --out NEW/both_bce --mode both --normalization NEW/normalization --device cuda
python -m routing.error_head.evaluate --cache CACHE/val --run NEW/both_bce --out NEW/both_bce/validation.json --routing --device cpu
python -m unittest discover -s tests -p test_error_head.py -v
```

Full notebook: `notebooks/frozen_error_head_v1.ipynb`. It checks old caches,
prints progress, writes separate outputs and never starts a cloud VM itself.
No external model weights or dataset downloads are needed for this stage.
