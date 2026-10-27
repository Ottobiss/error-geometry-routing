"""B0 Mix-FFN hidden activations and their place on the output map (PLAN.md §3, §5).

    uv run python -m sae.hooks          # step 0: shapes, logits unchanged, P1 B0 mIoU, overlay figure

The activation is the input of fc2 in every block: Mix-FFN = fc2 ∘ GELU ∘ DWConv3×3 ∘ fc1, dropout is
identity in eval. It is caught by a forward pre-hook on `stages[s].blocks[j].mlp.fc2` (transformers 5.x):
(B, N, 4C) in raster order on the stage's token grid, whose (H, W) comes from the stage's patch embeddings.

Position k of a stage with stride r is centred at output pixel r·k (7×7/s4/p3, then 3×3/s2/p1 convs).
A patch (u, v) of size P takes, per axis:
  r < P: all positions whose centre lies inside the patch, k = u·P/r … u·P/r + P/r − 1 (max-pooled);
  r ≥ P: the one position nearest to the patch centre, k = round((P·u + (P − 1)/2) / r), clamped
         (r = P gives k = u; r = 2P gives k = ⌈u/2⌉).
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

from sae import ROOT

BLOCKS = [f"s{s}b{j}" for s in range(1, 5) for j in (1, 2)]
STRIDES = {1: 4, 2: 8, 3: 16, 4: 32}
PATCH = 16


def stage(block):
    return int(block[1])


def stride(block):
    return STRIDES[stage(block)]


def fc2(model, block):
    """fc2 of the block in SegModel model ("s1b1" → stages[0].blocks[0])."""
    return model.net.segformer.stages[stage(block) - 1].blocks[int(block[3]) - 1].mlp.fc2


def hidden_dim(model, block):
    cfg = model.net.config
    s = stage(block) - 1
    return int(cfg.hidden_sizes[s] * cfg.mlp_ratios[s])


class MixFFNTap:
    """Context manager over SegModel: during forward passes (batch of 1) self.acts[block] holds the
    block's fc2 input as (H, W, 4C) for the last frame.

    replace = {block: f}: f((N, 4C)) → (N, 4C) is passed to fc2 instead (splice test); acts keeps the
    original activations.
    """

    def __init__(self, model, blocks=BLOCKS, replace=None):
        self.model, self.blocks, self.replace = model, list(blocks), dict(replace or {})
        self.acts, self.grid, self.handles = {}, {}, []

    def __enter__(self):
        stages = self.model.net.segformer.stages
        for s in sorted({stage(b) for b in self.blocks + list(self.replace)}):
            self.handles.append(stages[s - 1].patch_embeddings.register_forward_hook(
                lambda m, i, o, s=s: self.grid.__setitem__(s, (o[1], o[2]))))
        for b in sorted(set(self.blocks) | set(self.replace)):
            self.handles.append(fc2(self.model, b).register_forward_pre_hook(
                lambda m, args, b=b: self._pre(b, args)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre(self, block, args):
        x = args[0]
        h, w = self.grid[stage(block)]
        assert x.shape[0] == 1 and x.shape[1] == h * w, (block, tuple(x.shape), (h, w))
        if block in self.blocks:
            self.acts[block] = x[0].reshape(h, w, x.shape[2])
        if block in self.replace:
            return (self.replace[block](x[0]).to(x.dtype)[None],) + tuple(args[1:])
        return None


def selection_index(n_patches, n_grid, block, patch=PATCH):
    """Positions of the block's grid closest to each patch along one axis: (n_patches, m), m = P/r for
    r < P (all of them are max-pooled), otherwise m = 1."""
    r = stride(block)
    u = np.arange(n_patches)
    if r < patch:
        assert patch % r == 0
        m = patch // r
        idx = u[:, None] * m + np.arange(m)[None, :]
    else:
        idx = np.floor((patch * u + (patch - 1) / 2) / r + 0.5).astype(np.int64)[:, None]
        idx = np.clip(idx, 0, n_grid - 1)
    assert idx.max() < n_grid, (block, n_patches, n_grid)
    return idx


def pool_to_patches(x, block, n_patches_hw, patch=PATCH, reduce="max"):
    """x (H, W, D) on the block's grid → (Hp·Wp, D): max (or mean) over the closest positions of each patch,
    raster order of patches."""
    Hp, Wp = n_patches_hw
    rows = torch.as_tensor(selection_index(Hp, x.shape[0], block, patch), device=x.device)
    cols = torch.as_tensor(selection_index(Wp, x.shape[1], block, patch), device=x.device)
    y = x[rows][:, :, cols]                       # (Hp, m, Wp, m, D)
    y = y.amax(dim=(1, 3)) if reduce == "max" else y.mean(dim=(1, 3)) if reduce == "mean" else None
    assert y is not None, reduce
    return y.reshape(Hp * Wp, x.shape[2])


def draw_overlay(image, size, grids, examples, path, patch=PATCH, zoom=4):
    """Figure for step 0: the patch grid over the frame and the centres of the positions selected for a few
    example patches, coloured by stage; next to it a zoomed crop around the first example.
    image (3, H, W) uint8, grids {block: (h, w)}, examples [(u, v)]."""
    colors = {1: (255, 40, 40), 2: (255, 160, 0), 3: (40, 220, 40), 4: (40, 120, 255)}
    W, H = size
    img = Image.fromarray(image.permute(1, 2, 0).numpy()).convert("RGB")
    d = ImageDraw.Draw(img)
    for x in range(0, W, patch):
        d.line([(x, 0), (x, H)], fill=(200, 200, 200), width=1)
    for y in range(0, H, patch):
        d.line([(0, y), (W, y)], fill=(200, 200, 200), width=1)
    Hp, Wp = H // patch, W // patch
    for u, v in examples:
        d.rectangle([v * patch, u * patch, v * patch + patch - 1, u * patch + patch - 1], outline=(255, 255, 0))
        for b in BLOCKS[::2]:                     # both blocks of a stage share the grid
            r, (h, w) = stride(b), grids[b]
            for ky in selection_index(Hp, h, b, patch)[u]:
                for kx in selection_index(Wp, w, b, patch)[v]:
                    cy, cx = r * ky, r * kx
                    d.ellipse([cx - 1, cy - 1, cx + 1, cy + 1], fill=colors[stage(b)])
    u, v = examples[0]
    x0, y0 = max(v * patch - 2 * patch, 0), max(u * patch - 2 * patch, 0)
    crop = img.crop((x0, y0, x0 + 5 * patch + 16, y0 + 5 * patch + 16))
    crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.NEAREST)
    out = Image.new("RGB", (W + crop.width, max(H, crop.height)), (0, 0, 0))
    out.paste(img, (0, 0))
    out.paste(crop, (W, 0))
    out.save(path)


@torch.inference_mode()
def main():
    from core.data import get_dataset, loader, population
    from core.evaluation import _conf, _miou
    from core.components import IGNORE, NC
    from core.models import load_model

    b0 = load_model("b0")
    dev = b0.mean.device
    ds = get_dataset("cityscapes", population("cityscapes", "p1"), full=False)
    W, H = ds.size

    # Shapes and logits with / without hooks on one frame.
    sample = ds[0]
    x = sample["small"][None].to(dev)
    ref = b0(x, ds.size)
    with MixFFNTap(b0) as tap:
        out = b0(x, ds.size)
    diff = (ref - out).abs().max().item()
    same_argmax = torch.equal(ref.argmax(1), out.argmax(1))
    print(f"logits with hooks: max |Δ| = {diff:.3g}, argmax identical: {same_argmax}")
    assert diff < 1e-4 and same_argmax
    print("block  grid      4C    expected")
    for b in BLOCKS:
        r = stride(b)
        exp = (H // r, W // r, hidden_dim(b0, b))
        got = tuple(tap.acts[b].shape)
        print(f"{b}   {got[0]}×{got[1]}  {got[2]:5d}  {exp}")
        assert got == exp, (b, got, exp)
    grids = {b: tuple(tap.acts[b].shape[:2]) for b in BLOCKS}

    # P1 B0 mIoU with all hooks on (the existing protocol gives 71.57).
    conf = np.zeros((NC, NC), np.int64)
    with MixFFNTap(b0):
        for s in loader(ds):
            pred = b0.probs(s, ds.size).argmax(0).cpu().numpy()
            gt = s["label"].numpy()
            v = gt != IGNORE
            conf += _conf(gt[v].astype(np.int64), pred[v].astype(np.int64))
    print(f"P1 B0 mIoU with hooks: {_miou(conf):.2f} (expected 71.57)")

    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / "sanity_overlay.png"
    Hp, Wp = H // PATCH, W // PATCH
    draw_overlay(sample["small"], ds.size, grids, [(Hp // 2 - 1, Wp // 2), (Hp // 2, Wp // 2 + 1),
                                                    (Hp - 1, Wp - 1)], path)
    print(f"overlay: {path}")


if __name__ == "__main__":
    main()
