# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

Research code on routing between a small and a large segmentation model by predicted errors, built on HyperUL from "Hyperbolic Uncertainty Aware Semantic Segmentation" (Chen, Peng, Cao, Röning, IEEE T-ITS 2023, doi 10.1109/TITS.2023.3312290): SegFormer on Cityscapes. The paper PDF is not in the repo; "Eq. N" / "Table N" in docstrings and configs refer to it. The user writes in Russian — answer in Russian.

Routing pipeline (user's design; it replaced first a joint B0+B2 training, then YOLO-style confidence heads):

1. The large model (B2) is trained with HyperUL as in the paper (segmentation weighted by its hyperbolic uncertainty).
2. The trained large model predicts its hyperbolic uncertainty (Eq. 15) on the whole training set once; the maps are saved and stay fixed.
3. The small model (B0) is trained with HyperUL (its own uncertainty) plus a loss between its prediction of the large model's uncertainty and the saved maps.

At inference the small model outputs segmentation, its own uncertainty and the predicted uncertainty of the large model; where its own uncertainty is relatively high and the large one relatively low, the large model is run.

The whole network is not trained (user's requirement): on top of a Cityscapes-trained SegFormer only the layers HyperUL acts on are fine-tuned, and uncertainties are averaged over patches.

## Environment

- Python: `/home/artyom/LIBS/ML/bin/python` (3.12, torch 2.12, torchvision 0.27, transformers 4.46.3, omegaconf 2.3, scipy).
- This machine has no GPU. The user runs training/inference on a separate GPU machine: write code for CUDA, smoke-test on CPU here.
- Never install packages or download weights/data yourself. `tensorboard` is not installed locally; it is listed in `requirements.txt` for the GPU machine.

## Commands

Run from the repo root: the code is the `src` package (imports are `from src.x import ...`), `out` paths are relative to the working directory.

```bash
pip install -r requirements.txt   # GPU machine
python -m src.train configs/train.yaml data.root=/data/cityscapes                    # HyperUL baseline (B0)
python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes  # stage 1 -> runs/large
python -m src.export_uncertainty configs/export_uncertainty.yaml root=/data/cityscapes  # stage 2 -> runs/large/uncertainty
python -m src.train configs/train.yaml configs/small.yaml data.root=/data/cityscapes  # stage 3 -> runs/small
python -m src.eval_router configs/eval_router.yaml root=/data/cityscapes             # threshold sweep on val
python -m src.infer_router configs/infer_router.yaml input=<image or dir> [threshold=0.3]
python -m src.infer configs/infer.yaml input=<image or dir> [checkpoint=runs/hyperul/best.pt]
tensorboard --logdir runs --samples_per_plugin images=100
```

`src.train` also takes `--resume <ckpt>` and `--eval-only`. The model of hyperbolic space is switched with `model.manifold=poincare|lorentz`. Arguments after the first config are applied in order: more YAML files (`large.yaml` / `small.yaml` only override parts of `train.yaml`) and dotted `key=value` pairs. OmegaConf runs in struct mode: unknown keys raise, `???` marks mandatory values. A training run writes TensorBoard events, `config.yaml`, `last.pt`, `best.pt` to `out`; checkpoints store the resolved config and `model.load_model` rebuilds the network from it (`variant`, `manifold`, large-uncertainty head, no weight download).

There is no test suite. CPU smoke test without downloads: build a tiny fake Cityscapes tree (`leftImg8bit/{train,val}/<city>/*_leftImg8bit.png`, `gtFine/{train,val}/<city>/*_gtFine_labelIds.png`) and run every stage with small settings, e.g.

```bash
HF_HUB_OFFLINE=1 python -m src.train configs/train.yaml configs/large.yaml data.root=<fake> out=<tmp> model.pretrained=null \
  train.iters=4 data.batch_size=2 'data.crop=[128,256]' log.every=2 log.val_every=2 data.workers=0
```

Because tensorboard is missing locally, put a stub module into `sys.modules["torch.utils.tensorboard"]` with a `SummaryWriter` that records/validates `add_scalar/add_image/add_histogram/add_pr_curve/add_text`, and start the module with `runpy.run_module(name, run_name="__main__")`.

## Architecture

`model.HyperSegFormer` (paper Fig. 2) wraps HF `SegformerForSemanticSegmentation` built from `VARIANTS` (b0/b2 presets; `model.pretrained` weights are loaded into that config with `ignore_mismatched_sizes`, so the 19-class classifier of a segmentation checkpoint is initialized anew). The last decoder 1×1 conv outputs `embed_dim` channels = Euclidean embeddings v (1/4 resolution) → native `expmap0` of the chosen model of hyperbolic space → `hyperbolic.HyperbolicMLR` logits; with `large_head` (set by `model.build_model` when `data.large_uncertainty` is given) one more channel is the logit of the large model's uncertainty. `forward` returns `(logits, dist, large)` upsampled to the input size (dist = hyperbolic distance to the origin, large is None without the head), dist and large averaged over `model.patch` × `model.patch` squares (`patch_mean`, partial squares at the right/bottom edges); the hyperbolic part runs in fp32 even under bf16 autocast. With `model.freeze` only `net.decode_head.classifier` and `mlr` require grad and the overridden `train()` keeps `net` in eval mode.

- `hyperbolic.MANIFOLDS`: `Poincare` and `Lorentz` (curvature −c), implemented natively after Peng et al., "Hyperbolic Deep Neural Networks: A Survey" (TPAMI 2022). Each defines `origin`, `geodesic` (used by the shared `expmap0` with `scale` = Riemannian length of a Euclidean unit vector at the origin: 2 for Poincaré, 1 for Lorentz) and `dist` (the survey's distance formula; `dist0` feeds HyperUL and uncertainty); `max_rho` caps √c · distance.
- `loss.HyperUL` (Eq. 10–14): per-pixel CE × `1 / log(t + d)`, d = dist / per-image max (detached); OHUEM keeps pixels with `d <= d_min + hr * (1 - d_min)`.
- `loss.RoutingLoss` = HyperUL + `large_unc_weight` × `large_unc_loss` (`LARGE_UNC_LOSSES`: bce | mse | l1 between the sigmoid of the large-head logit and the exported large-model uncertainty) on labelled pixels; `large_unc_mae` is logged for every choice. Without the exported map it is plain HyperUL.
- `export_uncertainty` (stage 2) runs the large model on un-augmented full-resolution images of a split and saves `hyperbolic_uncertainty` as 8-bit PNGs at 1/`stride` resolution, `<out>/<split>/<city>/<image name>`.
- `data.Cityscapes` maps labelIds → trainIds on the fly; train augmentation is torchvision v2 `ScaleJitter` (its `target_size` is (W, H)) + `RandomCrop` (mask fill 255) + flip. With `large_uncertainty` it loads the exported map, upsamples it to the label size and augments it as a second mask; items become `(image, label, large_unc)`.
- `train.SegMetrics`: mIoU and, per named uncertainty map, ECE of 1 − uncertainty (as in Table XIV) and mean uncertainty of correct vs misclassified pixels. `train.py` also holds `make_loaders`, `param_groups` (frozen parameters left out), `poly_scheduler`, `endless`.
- `routing.Router(small, large)` takes two loaded models (the small one must have the large head). `routing.segment_gain` averages gain = own hyperbolic uncertainty − predicted large uncertainty over segments (connected regions of one class in the small prediction, scipy, at 1/`stride` resolution); `Router.cascade` runs the large model only if some segment's gain exceeds the threshold and copies its prediction into those segments.
- `eval_router` runs both models on every val image and logs small/large mIoU and hyperbolic ECE, the MAE between the predicted and the real large uncertainty, a PR curve of segment gain vs fixable pixels (small wrong, large right) and, per threshold (TensorBoard step = 100 × threshold), routed mIoU, launch rate, routed pixels and fix precision/recall.

## Decisions where the paper or the notes are ambiguous

Keep these unless the user asks otherwise.

HyperUL:

- Logits come from hyperbolic MLR (Ganea et al. 2018) as in refs [6], [15]; the paper does not name the classifier. `HyperbolicMLR.gyroplanes` is the native Poincaré form with the Möbius addition expanded through inner products (no (B, K, D, H, W) tensor; matches the naive formula to ~1e-13). `HyperbolicMLR.hyperboloid` builds the same construction natively in the Lorentz model (p_k by the Lorentz exp map, a_k by Lorentz parallel transport, hyperplane <a_k, x>_L = 0).
- Eq. 13 as printed is always true; it is implemented as a per-image threshold between the least and the most confident pixel.
- Uncertainty weights are computed from `dist.detach()`.
- `embed_dim=64` is not from the paper. d_c(expmap0(v), 0) = 2‖v‖, so a large classifier init saturates embeddings at the ball boundary where radial gradients vanish; HF init (std 0.02) keeps them inside.
- Defaults c=0.7, t=1.869, hr=1.0 are the best SegFormer-B0/Cityscapes setting (Tables XI–XII); lr 1.5e-4 (decoder ×10), 80k iterations, crop 512×1024 follow the paper; batch 8 on one GPU (paper: 24 on 4×A100). The same values are used for B2. They were tuned for full per-pixel training: with patches and a frozen network t and hr may need retuning.

Fine-tuning and patches:

- `model.freeze: true` (default) trains only the layers HyperUL acts on: the last 1×1 decoder conv (embeddings and the large-uncertainty logit) and the hyperbolic MLR, at lr × head_lr_mult = 1.5e-3. The rest comes frozen from `nvidia/segformer-{b0,b2}-finetuned-cityscapes-1024-1024` (configs match `VARIANTS`) and stays in eval mode (BatchNorm statistics, no drop path / dropout). Full training as in the paper: `model.freeze=false model.pretrained=nvidia/mit-b0`.
- `model.patch` (default 32 px, 1 = per pixel as in the paper) averages the hyperbolic distance, so Eq. 15 is per patch everywhere (HyperUL weights, OHUEM, metrics, export, routing) and normalized by the most confident patch, and the large-uncertainty logit, one logit per patch: BCE makes its sigmoid the patch mean of the target. Patches are aligned to the top-left corner of the input (the training crop or the full image).

Models of hyperbolic space:

- Only the Poincaré ball (default; the HyperUL paper and its tuned c, t, hr use it) and the Lorentz model are kept. Klein, hemisphere and half-space were implemented and then removed at the user's request: having a unit metric at the origin, they coincide with Lorentz in exact arithmetic, and Klein saturates earliest (√c · distance ≤ 3.8).
- Both models are native after the survey: own origin, metric, exponential map at the origin with respect to the own metric and the survey's distance formula. An earlier version forced one shared convention, so all models gave identical results; the user did not want that.
- They are isometric: the ball has metric 4I at the origin, so Poincaré with v, p, a equals Lorentz with 2v, 2p, 2a (checked to ~1e-15). Other differences come from float32: the ball clips at √c · distance 7.6 (`BALL_EPS`), the hyperboloid at 15 (`MAX_RHO`, beyond which its inner products lose precision). Radial gradients stop at the cap, which changes HyperUL weights and the per-image uncertainty normalization; with Lorentz, t and hr may need retuning.
- Truly different behaviour would need non-isometric choices (e.g. model-specific parametrizations or operations); ask the user before adding any.

Routing:

- "Segmentation loss + uncertainty loss as in the paper" = HyperUL: the paper has one loss, CE weighted by the hyperbolic uncertainty. Each model's own uncertainty is its hyperbolic uncertainty (Eq. 15), nothing extra is learned for it.
- The small model predicts the large model's uncertainty with a separate output channel (sigmoid) trained toward the exported maps; its own hyperbolic uncertainty is not pulled toward the large one, since routing needs both.
- Default `loss.large_unc_loss: bce` (soft targets). Like MSE it estimates the conditional mean, which the segment averaging in routing needs (L1 estimates the median). Unlike MSE and L1 through a sigmoid, its logit gradient û − u does not vanish when the prediction saturates near 0, where most targets lie. Its value has the target entropy as a floor, so compare runs by `large_unc_mae`.
- The large model's uncertainty is exported once on un-augmented full-resolution training images (normalized by each full image's max patch-mean distance) and augmented together with the label when the small model trains.
- "Relatively high own / relatively low large uncertainty" = segment mean of own − predicted large uncertainty above the threshold.
- The large model runs on the whole image; its prediction replaces the small one only on routed segments.

## Conventions

- Concise, readable code with few comments; don't overload with features.
- Tunable parameters belong in `configs/*.yaml`, dependencies in `requirements.txt` (`transformers<5`, tested on 4.46.3), monitoring in TensorBoard.
- `torch.load` relies on the default `weights_only=True`: checkpoints may contain only tensors and plain Python types (configs via `OmegaConf.to_container`).
