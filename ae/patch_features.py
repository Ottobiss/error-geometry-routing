"""Step 3: encoder features of the 16×16 patches (PLANv2.md §5).

    uv run python -m ae.patch_features --dataset cityscapes --encoders ae_b8 pca_b8
    uv run python -m ae.patch_features --dataset acdc --encoders ae_b8 pca_b8
    uv run python -m ae.patch_features --dataset cityscapes --encoders ae_b8 --limit 3      # smoke test

Every block is encoded once per frame on its native grid (normalised x, ae.collect constants), then brought
to the patch grid with the v1 closest-position rule (sae.hooks.pool_to_patches): stages 1–2 pool the 16 or
4 positions of a patch, stages 3–4 gather one position. Per block:
  Z  codes z (b values, dense, signed): mean over the positions of the patch
  R  residual r = ‖x − x̂‖² / d: mean and max over the positions (stages 1–2), the value itself (stages 3–4)
  N  activation norm ‖x‖² / d, pooled like R; it does not depend on the encoder, so it is stored once
Patches are in raster order, patch = u·Wp + v, all patches of the frame, as in the v1 base tables
(results/sae/patches/base/, which hold F0, labels and pixel samples and must exist for the same frames).

Output per frame (float32):
  results/ae/patches/<encoder>/<dataset>/<image_id>.npz    Z (n_patches, Σ b), R (n_patches, 12)
  results/ae/patches/norm/<dataset>/<image_id>.npz         N (n_patches, 12)
and _info.json in every folder with the column names.
"""
import argparse
import json
import time

import numpy as np
import torch

from ae import ROOT
from ae.collect import ACTS
from ae.quality import load_encoders
from core.data import get_dataset, loader, population, populations
from core.models import load_model
from sae.hooks import BLOCKS, PATCH, MixFFNTap, pool_to_patches, stride
from sae.patch_features import base_path

NORM = "norm"


def patch_path(encoder, dataset, image_id):
    return ROOT / "patches" / encoder / dataset / f"{image_id}.npz"


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


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N frames of every population")
    args = ap.parse_args()
    assert NORM not in args.encoders

    ids = sorted({i for p in populations(args.dataset) for i in population(args.dataset, p)[: args.limit]})
    ds = get_dataset(args.dataset, ids, full=False)
    assert ds.image_ids == ids, "frames must come in sorted image_id order"
    W, H = ds.size
    hw = (H // PATCH, W // PATCH)
    for image_id in ids:
        assert base_path(args.dataset, image_id).exists(), f"v1 base table missing: {image_id}"

    b0 = load_model("b0")
    dev = b0.mean.device
    encs = load_encoders(args.encoders, dev)
    norms = {}
    for b in BLOCKS:
        z = np.load(ACTS / f"{b}_norm.npz")
        norms[b] = (torch.from_numpy(z["mean"]).to(dev), float(z["scale"]))
        assert torch.equal(norms[b][0], encs[args.encoders[0]][b].mean), b
        assert norms[b][1] == encs[args.encoders[0]][b].scale.item(), b
    for n in [*args.encoders, NORM]:
        patch_path(n, args.dataset, "x").parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with MixFFNTap(b0) as tap:
        for k, sample in enumerate(loader(ds), 1):
            image_id = sample["image_id"]
            assert image_id == ids[k - 1]
            b0.probs(sample, ds.size)
            out, N = frame_features(tap.acts, encs, norms, hw)
            assert len(N) == len(np.load(base_path(args.dataset, image_id))["n_valid"]), image_id
            np.savez(patch_path(NORM, args.dataset, image_id), N=N)
            for n, (Z, R) in out.items():
                np.savez(patch_path(n, args.dataset, image_id), Z=Z, R=R)
            if k % 50 == 0 or k == len(ds):
                print(f"{k}/{len(ds)}  {time.time() - t0:.0f} s")
    info = {"frames": len(ds), "patch": PATCH, "grid": hw, "blocks": BLOCKS, "limit": args.limit}
    (ROOT / "patches" / NORM / args.dataset / "_info.json").write_text(json.dumps(
        {**info, "N": scalar_columns("n")}, indent=1))
    for n, e in encs.items():
        (ROOT / "patches" / n / args.dataset / "_info.json").write_text(json.dumps(
            {**info, "b": [e[b].b for b in BLOCKS], "R": scalar_columns("r"),
             "Z": [f"{b}_z{j}" for b in BLOCKS for j in range(e[b].b)]}, indent=1))


if __name__ == "__main__":
    main()
