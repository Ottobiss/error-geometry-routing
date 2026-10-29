import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation
NC = 19


CHECKPOINTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

FULL_RES = {"b2", "b5"}

def upsample_logits(logits, size):
    """Логиты (19, h, w) → (19, H, W) на рабочем разрешении size = (W, H), билинейно."""
    x = torch.as_tensor(logits, device="cuda").float()[None]
    return F.interpolate(x, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0]

class SegModel(nn.Module):
    """SegFormer с предобработкой. forward(image uint8 (B, 3, H, W), size) → логиты (B, 19, h, w)
    на рабочем разрешении size = (W, h). У лёгкой модели после каждого прохода в self.features —
    внутренние признаки (у тяжёлых хуков нет, чтобы не держать их тензоры в памяти GPU):

    C      — вход decode_head.classifier: (1, 256, h/4, w/4) (после linear_fuse → BN → ReLU;
             dropout в eval — тождество);
    S4     — выход стадии 4 энкодера (вход decode_head): (n, 256) токенов сетки 1/32;
    Q/K/V  — выходы q_proj / k_proj / v_proj последнего блока стадии 4: (n, 256), 8 голов × 32 склеены;
    logits — выход decode_head: (19, h/4, w/4) до подъёма разрешения.
    """

    def __init__(self, name):
        super().__init__()
        self.full_res = name in FULL_RES
        self.net = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINTS[name])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
        self.features = {}
        if self.full_res:
            return
        att =self.net.segformer.stages[3].blocks[-1].attention
        for key, mod in (("Q", att.q_proj), ("K", att.k_proj), ("V", att.v_proj)):
            mod.register_forward_hook(lambda m, i, o, k=key: self.features.__setitem__(k, o[0]))
        head = self.net.decode_head
        head.register_forward_pre_hook(lambda m, a: self.features.__setitem__("S4", a[0][-1][0].flatten(1).T))
        head.classifier.register_forward_pre_hook(lambda m, a: self.features.__setitem__("C", a[0]))
        head.register_forward_hook(lambda m, i, o: self.features.__setitem__("logits", o[0]))

    def forward(self, image, size):
        x = (image.float() - self.mean) / self.std
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=(size[1], size[0]), mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def probs(self, sample, size, full=True):
        """Элемент датасета → softmax (19, H, W) на GPU на рабочем разрешении size.
        full=False — тяжёлая модель тоже получает уменьшенный кадр (SegDataset.heavy_full)."""
        image = sample["image" if self.full_res and full else "small"]
        return self(image[None].to(self.mean.device), size).softmax(1)[0]

    def tokens(self, key):
        """Признак как строки (n, 256) float32."""
        x = self.features[key]
        if key == "C":
            x = x[0].flatten(1).T
        return x.float().contiguous()

    def c_map(self, size, normalize=False):
        """C, поднятый билинейно до рабочего разрешения: (256, H·W); normalize — L2 по каналам."""
        up = F.interpolate(self.features["C"].float(), size=(size[1], size[0]), mode="bilinear",
                           align_corners=False)[0]
        if normalize:
            up = F.normalize(up, dim=0)
        return up.reshape(up.shape[0], -1)

    def neg_lse(self, size):
        """−logsumexp логитов, поднятых до рабочего разрешения: (H·W,)."""
        return -upsample_logits(self.features["logits"], size).logsumexp(0).reshape(-1)

def load_model(name, device="cuda"):
    return SegModel(name).to(device).eval()

def entropy_score(probs, dim=0):
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim)

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

def per_patch(x, patch=PATCH):
    """(H, W) map → (Hp·Wp, P²): pixels of each patch, raster order of patches."""
    H, W = x.shape
    assert H % patch == 0 and W % patch == 0, (H, W)
    return x.reshape(H // patch, patch, W // patch, patch).swapaxes(1, 2).reshape(-1, patch * patch)

def boundary_map(pred):
    """Pixels whose 3×3 neighbourhood (inside the frame) has more than one predicted class."""
    p = np.pad(pred, 1, mode="edge")
    H, W = pred.shape
    out = np.zeros((H, W), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] != pred
    return out

def output_features(probs, pred, patch=PATCH):
    """F0 of every patch from the B0 softmax probs (19, H, W) and pred (H, W) tensors: {name: (n,) numpy}."""
    ent = per_patch(entropy_score(probs), patch)
    top2 = probs.topk(2, dim=0).values
    msp = per_patch(top2[0], patch)
    margin = per_patch(top2[0] - top2[1], patch)
    counts = F.one_hot(per_patch(pred, patch), NC).sum(1)
    boundary = per_patch(torch.from_numpy(boundary_map(pred.cpu().numpy())), patch)
    Hp, Wp = pred.shape[0] // patch, pred.shape[1] // patch
    row, col = np.divmod(np.arange(Hp * Wp), Wp)
    f = {"ent_mean": ent.mean(1), "ent_max": ent.amax(1), "ent_p90": torch.quantile(ent, 0.9, dim=1),
         "msp_mean": msp.mean(1), "msp_min": msp.amin(1), "margin_mean": margin.mean(1),
         "margin_min": margin.amin(1), "boundary_frac": boundary.float().mean(1)}
    f = {k: v.float().cpu().numpy() for k, v in f.items()}
    f["cls"] = counts.argmax(1).cpu().numpy().astype(np.int16)     # ties → smallest class id
    f["row"], f["col"] = row.astype(np.int16), col.astype(np.int16)
    return f

class _Normalized(nn.Module):
    def __init__(self, d, mean, scale):
        super().__init__()
        self.d = d
        self.register_buffer("mean", torch.zeros(d) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("scale", torch.tensor(float(scale)))

    def normalize(self, h):
        return (h.float() - self.mean) / self.scale

    def denormalize(self, x):
        return x * self.scale + self.mean

class BottleneckAE(_Normalized):
    kind = "ae"

    def __init__(self, d, b, h=None, mean=None, scale=1.0):
        super().__init__(d, mean, scale)
        self.b, self.h = b, h or d
        self.enc = nn.Sequential(nn.Linear(d, self.h), nn.GELU(), nn.Linear(self.h, b))
        self.dec = nn.Sequential(nn.Linear(b, self.h), nn.GELU(), nn.Linear(self.h, d))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def config(self):
        return {"d": self.d, "b": self.b, "h": self.h}

class PCA(_Normalized):
    kind = "pca"

    def __init__(self, d, b, mean=None, scale=1.0):
        super().__init__(d, mean, scale)
        self.b = b
        self.register_buffer("center", torch.zeros(d))          # μ: mean of the normalised training x
        self.register_buffer("components", torch.zeros(b, d))   # rows: eigenvectors, decreasing eigenvalue

    def encode(self, x):
        return (x - self.center) @ self.components.T

    def decode(self, z):
        return z @ self.components + self.center

    def config(self):
        return {"d": self.d, "b": self.b}

KINDS = {"ae": BottleneckAE, "pca": PCA}

def load_encoder(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    enc = KINDS[ckpt["kind"]](**ckpt["config"]).to(device)
    enc.load_state_dict(ckpt["state"])
    return enc.eval(), ckpt

def scalar_columns(tag):
    """Names of the R or N columns: <block>_<tag>_mean, <block>_<tag>_max (stages 1–2), <block>_<tag>."""
    return [c for b in BLOCKS for c in ([f"{b}_{tag}_mean", f"{b}_{tag}_max"] if stride(b) < PATCH else [f"{b}_{tag}"])]

def pool_scalar(v, block, hw):
    """Per-position values (h, w) → (n_patches, 2) mean and max (stages 1–2) or (n_patches, 1)."""
    v = v[..., None]
    if stride(block) < PATCH:
        return torch.cat([pool_to_patches(v, block, hw, reduce="mean"), pool_to_patches(v, block, hw, reduce="max")], 1)
    return pool_to_patches(v, block, hw, reduce="mean")

def frame_features(acts, encs, norms, hw):
    """{encoder: (Z, R)} and N of one frame, numpy float32."""
    Z = {n: [] for n in encs}
    R = {n: [] for n in encs}
    N = []
    for b in BLOCKS:
        a = acts[b]
        h, w, d = a.shape
        x = (a.reshape(-1, d).float() - norms[b][0]) / norms[b][1]
        N.append(pool_scalar((x.pow(2).sum(1) / d).reshape(h, w), b, hw))
        for n, e in encs.items():
            enc = e[b]
            z = enc.encode(x)
            r = (enc.decode(z) - x).pow(2).sum(1) / d
            Z[n].append(pool_to_patches(z.reshape(h, w, -1), b, hw, reduce="mean"))
            R[n].append(pool_scalar(r.reshape(h, w), b, hw))
    out = {n: (torch.cat(Z[n], 1).cpu().numpy(), torch.cat(R[n], 1).cpu().numpy()) for n in encs}
    return out, torch.cat(N, 1).cpu().numpy()
