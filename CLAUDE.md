# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context

The repo currently holds only `hyperul/`: an implementation of HyperUL from "Hyperbolic Uncertainty Aware Semantic Segmentation" (Chen, Peng, Cao, Röning, IEEE T-ITS 2023, doi 10.1109/TITS.2023.3312290) for SegFormer-B0 on Cityscapes. The paper PDF is not in the repo; "Eq. N" / "Table N" in docstrings and configs refer to it. The user writes in Russian — answer in Russian.

## Environment

- Python: `/home/artyom/LIBS/ML/bin/python` (3.12, torch 2.12, torchvision 0.27, transformers 4.46.3, omegaconf 2.3).
- This machine has no GPU. The user runs training/inference on a separate GPU machine: write code for CUDA, smoke-test on CPU here.
- Never install packages or download weights/data yourself. `tensorboard` is not installed locally; it is listed in `hyperul/requirements.txt` for the GPU machine.

## Commands

Run from `hyperul/` (scripts import sibling modules; `out` paths are relative to the working directory):

```bash
pip install -r requirements.txt   # GPU machine
python train.py configs/train.yaml data.root=/data/cityscapes [train.amp=true] [--resume runs/hyperul/last.pt] [--eval-only]
python infer.py configs/infer.yaml input=<image or dir> [checkpoint=runs/hyperul/best.pt]
tensorboard --logdir runs --samples_per_plugin images=100
```

Parameters are overridden with dotted `key=value` args (OmegaConf in struct mode: unknown keys raise, `???` marks mandatory values). A training run writes TensorBoard events, `config.yaml`, `last.pt`, `best.pt` to `out`; checkpoints store the resolved config and `infer.py` rebuilds the model from it (`pretrained=None`).

There is no test suite. CPU smoke test without downloads: build a tiny fake Cityscapes tree (`leftImg8bit/{train,val}/<city>/*_leftImg8bit.png`, `gtFine/{train,val}/<city>/*_gtFine_labelIds.png`), then

```bash
HF_HUB_OFFLINE=1 python train.py configs/train.yaml data.root=<fake> out=<tmp> model.pretrained=null \
  train.iters=4 data.batch_size=2 'data.crop=[128,256]' log.every=2 log.val_every=2 data.workers=0
```

Because tensorboard is missing locally, put a stub module into `sys.modules["torch.utils.tensorboard"]` with a `SummaryWriter` that records/validates calls, and run the script through `runpy.run_path`.

## Architecture

Data flow (paper Fig. 2): `model.HyperSegFormer` wraps HF `SegformerForSemanticSegmentation` whose decoder classifier outputs `embed_dim` channels = Euclidean embeddings v (1/4 resolution) → `hyperbolic.expmap0` onto the Poincaré ball → `hyperbolic.HyperbolicMLR` logits. `forward` returns `(logits, dist)` (dist = hyperbolic distance to the origin), both upsampled to the input size; the hyperbolic part runs in fp32 even under bf16 autocast.

- `loss.HyperUL` (Eq. 10–14): per-pixel CE × `1 / log(t + d)`, d = dist / per-image max (detached); OHUEM keeps pixels with `d <= d_min + hr * (1 - d_min)`.
- `hyperbolic.hyperbolic_uncertainty` (Eq. 15) = 1 − dist / per-image max; feeds validation metrics, TensorBoard panels and inference maps.
- `train.SegMetrics`: mIoU, ECE of confidence = 1 − uncertainty (as in Table XIV), mean uncertainty of correct vs misclassified pixels.
- `data.Cityscapes` maps labelIds → trainIds on the fly; train augmentation is torchvision v2 `ScaleJitter` (its `target_size` is (W, H)) + `RandomCrop` (mask fill 255) + flip.
- Training is iteration-based: AdamW with encoder lr and `head_lr_mult` for decoder + hyperbolic MLR, linear warmup × poly decay via `LambdaLR`.

## Decisions where the paper is ambiguous

Keep these unless the user asks otherwise:

- Logits come from hyperbolic MLR (Ganea et al. 2018) as in refs [6], [15]; the paper does not name the classifier. `HyperbolicMLR` expands the Möbius addition through inner products (no (B, K, D, H, W) tensor); it matches the naive formula to ~1e-13.
- Eq. 13 as printed is always true; it is implemented as a per-image threshold between the least and the most confident pixel.
- Uncertainty weights are computed from `dist.detach()`.
- `embed_dim=64` is not from the paper. d_c(expmap0(v), 0) = 2‖v‖, so a large classifier init saturates embeddings at the ball boundary where radial gradients vanish; HF init (std 0.02) keeps them inside.
- Defaults c=0.7, t=1.869, hr=1.0 are the best SegFormer-B0/Cityscapes setting (Tables XI–XII); lr 1.5e-4 (decoder ×10), 80k iterations, crop 512×1024 follow the paper; batch 8 on one GPU (paper: 24 on 4×A100).

## Conventions

- Concise, readable code with few comments; don't overload with features.
- Tunable parameters belong in `hyperul/configs/*.yaml`, dependencies in `hyperul/requirements.txt` (`transformers<5`, tested on 4.46.3), monitoring in TensorBoard.
- `torch.load` relies on the default `weights_only=True`: checkpoints may contain only tensors and plain Python types (configs via `OmegaConf.to_container`).
