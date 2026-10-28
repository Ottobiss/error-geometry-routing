# Integration guide: hyperbolic uncertainty of B0 and B2 (HyperUL head)

## 1. Scope

This guide tells how the hyperbolic method of this branch works, how to merge it into the common project, how to
run its stages again after the merge, and how to attach it to B0 and B2.

The method uses the image parameters of this branch:
- B0 `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` and B2 `nvidia/segformer-b2-finetuned-cityscapes-1024-1024`;
- the full frame: Cityscapes 2048 × 1024, ACDC 1920 × 1080;
- patches of 32 × 32 pixels.

The method adds one small head to a model. The head is a 1 × 1 convolution. It reads the same tensor as the last
layer of the default network: the input of `decode_head.classifier`. It does not change the default network or
its outputs.

The head of B0 gives three maps, one value for each 32 × 32 patch of the frame:
- `own`: the hyperbolic uncertainty of B0;
- `large`: the predicted hyperbolic uncertainty of B2 on the same frame (B2 does not run);
- `gain` = `own` − `large`.

The head of B2 gives the hyperbolic uncertainty of B2. The stages use it to make the training targets of the B0
head.

## 2. How the method works

### 2.1 Idea

HyperUL (Chen, Peng, Cao, Röning, "Hyperbolic Uncertainty Aware Semantic Segmentation", IEEE T-ITS 2023) maps the
class outputs of a segmentation network into hyperbolic space. The distance of the point from the origin shows how
certain the network is. A short distance shows high uncertainty.

In this branch:
1. The SegFormer is frozen. Only a copy of its last 1 × 1 convolution trains. The copy starts from the default
   weights.
2. B2 learns its own uncertainty.
3. B0 learns its own uncertainty. One more output channel of B0 learns the uncertainty that B2 will have on the
   same frame. Thus B0 can estimate it without B2.
4. A patch where B0 is uncertain and B2 is expected to be certain has a high `gain`.

### 2.2 Formulas

Let f be the input of `decode_head.classifier` at one position of the 1/4 resolution map (D = 256 channels for B0,
768 for B2). Let W and b be the weights of the head, and s = 0.2 (`model.classifier_scale`). Then

$$o = W f + b, \qquad v = s \, o_{1 \ldots 19}$$

The exponential map at the origin of the Lorentz model with curvature −c gives the point

$$x = \frac{1}{\sqrt{c}} \left( \cosh(\sqrt{c} \lVert v \rVert), \; \sinh(\sqrt{c} \lVert v \rVert) \frac{v}{\lVert v \rVert} \right)$$

- The 19 space-like coordinates of x are the class logits.
- The distance from the origin is d = ‖v‖. The code clips √c ‖v‖ at 15.
- d is upsampled (bilinear) to the resolution of the frame and averaged over 32 × 32 patches: d̄.
- The uncertainty of a patch (Eq. 15 of the paper) is below. The maximum is over the frame. The most certain patch
  of the frame has u = 0.

$$u = 1 - \frac{\bar{d}}{\max \bar{d}}$$

- The 20th output of B0 is the predicted uncertainty of B2: û₂ = σ(patch mean of o₂₀).
- The gain of a patch is g = u₀ − û₂.

Training loss (HyperUL, Eq. 14): the cross-entropy of the hyperbolic logits. The weight of each pixel is
1 / ln(t + d̄ / max d̄), t = 1.869. OHUEM keeps the share hr (1.0) of the least certain labelled pixels. The
20th output of B0 trains with binary cross-entropy against the exported uncertainty of B2.

### 2.3 Stages

| Stage | Model | What trains | Data | Output |
|---|---|---|---|---|
| 1 | B2 | Head, HyperUL loss | Cityscapes train (2,975) | `runs/large/best.pt` |
| 2 | B2 | Nothing: export of u₂ | Cityscapes val (500), not seen by B2 | `runs/large/uncertainty/val/` |
| 3 | B0 | Head: HyperUL on train, output 20 on val | Cityscapes train and val | `runs/small/best.pt` |

```
STAGE 1  Cityscapes train ──> B2 (frozen) ──C──> head 768→19 ──Lorentz──> logits, d ──HyperUL──> runs/large/best.pt
STAGE 2  Cityscapes val ────> B2 + head ──> u₂ per patch ──PNG, 1/4 resolution──> runs/large/uncertainty/val/
STAGE 3  Cityscapes train ──> B0 (frozen) ──C──> head 256→20 ── outputs 1..19 ──HyperUL──┐
         Cityscapes val ────> B0 (frozen) ──C──> head ─────────── output 20 ──BCE(u₂)────┴──> runs/small/best.pt
```

NOTE: In stage 3, train frames teach only the segmentation, and val frames teach only the prediction of the B2
uncertainty. The class labels of val are not used for training. Thus val mIoU can still select the checkpoint.

NOTE: `best.pt` is the checkpoint of the validation with the best val mIoU. `src.sweep` can search c,
`classifier_scale`, t, hr, weight decay and the learning rate.

The branch also evaluates the heads (`src.benchmark`, `src.report`: mIoU, ECE, MAE of û₂, patch error metrics on
ACDC). These scripts are optional for the integration.

## 3. Where the head attaches in B0 and B2

| Model | Default checkpoint | Frame given to the model | Head input C | Head |
|---|---|---|---|---|
| B0 | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | Full frame: Cityscapes 2048 × 1024, ACDC 1920 × 1080 | (1, 256, H/4, W/4) | 256 → 20 |
| B2 | `nvidia/segformer-b2-finetuned-cityscapes-1024-1024` | Full frame | (1, 768, H/4, W/4) | 768 → 19 |

C is the output of `linear_fuse → batch_norm → ReLU` in the decode head. Dropout is an identity in eval mode. A
forward pre-hook on `decode_head.classifier` catches C during the usual forward pass of the default model. The head
does not write to the network. The default logits do not change (section 9).

`core.models.SegModel("b0")` of the common project uses B0 `…-512-1024` on the reduced frame. Its `features["C"]`
does not fit the head. Use a pass of B0 `…-1024-1024` on the full frame (section 8).

CAUTION: A head fits only the checkpoint and the frame size that it was trained on. Do not give it C of
B0 `…-512-1024` or of a reduced frame.

## 4. Data that the head gets

1. C from the forward pass of the default B0 `…-1024-1024` on one full frame.
2. `size` = (W, H) of this frame.
3. The checkpoint of stage 3 (`runs/small/best.pt`). The head reads from it only `decode_head.classifier` and the
   config (`manifold`, `c`, `classifier_scale`, `patch`).

The head does not use ground truth, B2 or the default logits.

## 5. Output

`HyperbolicHead(checkpoint)(C, size)` returns a dict:

| Key | Shape | Type | Content |
|---|---|---|---|
| `own` | (Hp, Wp) | float32 | Hyperbolic uncertainty of the model, 0…1. 0 = the most certain patch of the frame. |
| `large` | (Hp, Wp) | float32 | B0 only. Predicted uncertainty of B2, 0…1. |
| `gain` | (Hp, Wp) | float32 | B0 only. `own` − `large`. Higher = B2 is expected to be more certain than B0. |

Hp = ⌈H / 32⌉ and Wp = ⌈W / 32⌉: 32 × 64 for Cityscapes and 34 × 60 for ACDC (the last row is 24 pixels high).
Patches are in raster order: patch = row · Wp + col. The tensors are on the device of the head.

On Cityscapes, the patches cover the same areas as the 16 × 16 patches of branch `SAE` at 1024 × 512. On ACDC,
the grids are different: `SAE` uses 36 × 64 at 1024 × 576.

## 6. Merge into the common project

### 6.1 Files

Put these files of this branch into the common project:

| File | What to do |
|---|---|
| `src/`, `configs/` | Copy without changes. The base code has no folders with these names. |
| `src/head.py` | New file. Copy it from this guide (below). |
| `README.md` | Put it as `docs/hyperbolic.md`. It tells the details of the stages: data layout, two GPUs, sweep, time. |
| `requirements.txt`, `.gitignore` | Do not copy. Change the files of the common project (section 6.2). |

`src/head.py`:

```python
"""Hyperbolic uncertainty of SegFormer next to its default classifier (branch double_hyperul_segformer)."""
import torch
import torch.nn.functional as F
from torch import nn

from src.hyperbolic import MANIFOLDS

NC = 19


class HyperbolicHead(nn.Module):
    """The trained last 1x1 conv of a src.train checkpoint, applied to the input of the default classifier.
    The default classifier and its outputs do not change. checkpoint: runs/small/best.pt (B0, 20 outputs) or
    runs/large/best.pt (B2, 19 outputs) of the branch, or a file with only their classifier tensors."""

    def __init__(self, checkpoint):
        super().__init__()
        ckpt = torch.load(checkpoint, map_location="cpu")
        m, state = ckpt["config"]["model"], ckpt["model"]
        weight, bias = state["net.decode_head.classifier.weight"], state["net.decode_head.classifier.bias"]
        self.conv = nn.Conv2d(weight.shape[1], weight.shape[0], 1)
        self.conv.load_state_dict({"weight": weight, "bias": bias})
        self.manifold = MANIFOLDS[m["manifold"]](m["c"])
        self.scale, self.patch = m["classifier_scale"], m["patch"]

    @torch.inference_mode()
    def forward(self, features, size):
        """features: (1, D, h, w), the input of decode_head.classifier (SegModel.features["C"]); size: (W, H) of the
        frame given to the model. Returns maps (Hp, Wp) per patch: own, the uncertainty of the model; for B0 also
        large, the predicted uncertainty of B2, and gain = own - large."""
        out = self.conv(features.float())
        z = self.manifold.expmap0(self.scale * out[:, :NC])
        hw = (size[1], size[0])
        dist = F.interpolate(self.manifold.dist0(z)[:, None], hw, mode="bilinear", align_corners=False)
        dist = F.avg_pool2d(dist, self.patch, ceil_mode=True)[0, 0]
        result = {"own": 1 - dist / dist.max().clamp_min(1e-5)}  # Eq. 15
        if out.shape[1] > NC:
            large = F.interpolate(out[:, NC:], hw, mode="bilinear", align_corners=False)
            result["large"] = F.avg_pool2d(large, self.patch, ceil_mode=True)[0, 0].sigmoid()
            result["gain"] = result["own"] - result["large"]
        return result
```

### 6.2 Packages and ignored files

The common `requirements.txt` already has torch, torchvision, transformers, scipy and numpy. Add the packages that
the stages need: `omegaconf`, `tensorboard`, `tqdm`, `matplotlib`.

Add `runs/` to the common `.gitignore`. The stages write all their results there.

## 7. Run the stages again after the merge

### 7.1 Data

The stages read the official Cityscapes layout from `data/cityscapes`. This is the same folder as in the base code:

```
data/cityscapes/
├── leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
└── gtFine/{train,val}/<city>/*_gtFine_labelIds.png
```

CAUTION: The base code gets only the val part by default. The stages also need train with labels. Unpack
`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` into `data/cityscapes`.

ACDC is necessary only for the optional evaluation. The base code keeps it in `data/acdc` (`rgb_anon/`, `gt/`).
Make the Cityscapes layout with symbolic links: use the script of `docs/hyperbolic.md` (section "ACDC") with
`ACDC=$PWD/data/acdc` and `DST=$PWD/data/acdc_cityscapes`. If you have only the val part of ACDC, use only `val`
in the loop of the script, and `split=val` in the benchmark.

### 7.2 Commands

Run all commands from the root of the common project. The commands are for two GPUs with bf16. For one GPU, use
`python -m src.train` instead of `torchrun --standalone --nproc_per_node=2 -m src.train`. `data.workers` is the
total number of loader processes.

0. Trial run. It checks the environment, the data and the speed:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/large.yaml \
     data.root=data/cityscapes out=runs/try train.iters=200 log.every=20 log.val_every=200 train.amp=true
   ```

   Make sure that val mIoU at step 200 is approximately 0.8. Then remove `runs/try`.

1. Stage 1, the B2 head:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/large.yaml \
     data.root=data/cityscapes train.amp=true
   ```

   Or search the hyperparameters, then copy the best trial into `runs/large`:

   ```bash
   python -m src.sweep configs/sweep.yaml stage=large root=data/cityscapes 'overrides=["train.amp=true"]'
   mkdir -p runs/large && cp -r runs/sweep/large/<trial>/. runs/large/
   ```

2. Stage 2, the export of the B2 uncertainty:

   ```bash
   python -m src.export_uncertainty configs/export_uncertainty.yaml root=data/cityscapes
   ```

3. Stage 3, the B0 head:

   ```bash
   torchrun --standalone --nproc_per_node=2 -m src.train configs/train.yaml configs/small.yaml \
     data.root=data/cityscapes train.amp=true
   ```

   Or search the hyperparameters, then copy the best trial into `runs/small`:

   ```bash
   python -m src.sweep configs/sweep.yaml stage=small root=data/cityscapes 'overrides=["train.amp=true"]'
   mkdir -p runs/small && cp -r runs/sweep/small/<trial>/. runs/small/
   ```

4. Optional evaluation (the details are in `docs/hyperbolic.md`):

   ```bash
   python -m src.benchmark configs/benchmark.yaml root=data/cityscapes split=val out=runs/benchmark_val
   python -m src.benchmark configs/benchmark.yaml root=data/acdc_cityscapes
   python -m src.report configs/report.yaml
   ```

## 8. Use the head

Make the default B0 `…-1024-1024`, the hook and the head one time for each process. Then, for each frame, run B0
one time on the full frame and give its C to the head:

```python
import torch
from transformers import SegformerForSemanticSegmentation

from src.head import HyperbolicHead

device = "cuda"
b0 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
b0 = b0.to(device).eval()                                  # default B0 1024-1024
features = {}
b0.decode_head.classifier.register_forward_pre_hook(lambda m, a: features.__setitem__("C", a[0]))
head = HyperbolicHead("runs/small/best.pt").to(device)
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255

image = sample["image"]                                    # uint8 (3, H, W): the full frame
with torch.inference_mode():
    logits = b0(pixel_values=(image[None].to(device).float() - mean) / std).logits   # default output of B0
hyper = head(features["C"], (image.shape[2], image.shape[1]))                      # own, large, gain: (Hp, Wp)
```

`sample["image"]` is the full frame of the datasets of `core.data` (with `full=True`). The normalisation is the
same as in `core.models.SegModel`.

Call the head before the next B0 pass: `features` keeps only the last pass. Give it one frame at a time.

NOTE: You can also add the key `"b0_1024": "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"` to `CHECKPOINTS`
in `core/models.py`. Then `load_model("b0_1024")` sets the same hook. Call it with the full frame
(`sample["image"]`), not with `probs()`: `probs()` gives B0 the reduced frame.

B2 (optional): to get the uncertainty of B2 on a full frame, do the same with
`nvidia/segformer-b2-finetuned-cityscapes-1024-1024` and `HyperbolicHead("runs/large/best.pt")`. The B2 head gives
only `own`.

## 9. Acceptance test

Do this test after stage 3. `path` is a Cityscapes frame:

```python
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from src.head import HyperbolicHead
from src.hyperbolic import hyperbolic_uncertainty
from src.model import load_model

device, ckpt = "cuda", "runs/small/best.pt"
b0 = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
b0 = b0.to(device).eval()                                  # default B0
head = HyperbolicHead(ckpt).to(device)
reference = load_model(ckpt, 19).to(device).eval()         # the model of stage 3
image = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)   # full frame
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1) * 255
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1) * 255
x = (image[None].to(device).float() - mean) / std

features = {}
with torch.inference_mode():
    before = b0(pixel_values=x).logits
    b0.decode_head.classifier.register_forward_pre_hook(lambda m, a: features.__setitem__("C", a[0]))
    after = b0(pixel_values=x).logits
    _, dist, large = reference(x)
out = head(features["C"], (image.shape[2], image.shape[1]))
grid = lambda m: F.avg_pool2d(m[None, None], head.patch, ceil_mode=True)[0, 0]
print("default logits", (after - before).abs().max().item())
print("own", (out["own"] - grid(hyperbolic_uncertainty(dist)[0])).abs().max().item())
print("large", (out["large"] - grid(large[0].sigmoid())).abs().max().item())
```

1. Make sure that the difference of the default logits is 0. The head only reads C.
2. Make sure that the differences of `own` and `large` are less than 1e-4.
