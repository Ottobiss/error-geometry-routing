"""Step 3: patch table — 16×16 patches of the final B0 map with output features (F0), labels, SAE codes and
raw Mix-FFN activations of the closest positions (PLAN.md §5, §6).

    uv run python -m sae.patch_features --dataset cityscapes --sae D2k8
    uv run python -m sae.patch_features --dataset acdc --sae D2k8
    uv run python -m sae.patch_features --dataset cityscapes --sae D2k8 --limit 3     # smoke test

Every block is encoded once per frame on its native grid; codes and raw activations are then max-pooled or
gathered into the patch grid (sae.hooks.pool_to_patches). Patches are in raster order, patch = u·Wp + v.

Output per frame (all patches, including those without valid pixels):
  results/sae/patches/base/<dataset>/<image_id>.npz
      n_valid, n_err (int32)        valid (GT ≠ 255) and wrong valid pixels of B0
      ent_mean, ent_max, ent_p90, msp_mean, msp_min, margin_mean, margin_min, boundary_frac (float32),
      cls (int16) majority predicted class, row, col (int16)
      raw (n_patches, 4096) fp16    Mix-FFN hidden activations of the 8 blocks, BLOCKS order
      pix_idx, pix_err, pix_msp, pix_ent   test frames only: the same 5% of valid pixels as the existing
                                    pixel protocol (core.evaluation.PixelSampler, seed 0, sorted image_id),
                                    flat index on the working-resolution map, B0 error, 1 − MSP, entropy
  results/sae/patches/<sae>/<dataset>/<image_id>.npz
      data, indices, indptr, shape  CSR (n_patches, Σ D) float32 SAE codes, blocks in BLOCKS order
"""
import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from core.components import IGNORE, NC
from core.data import get_dataset, loader, population, populations, test_populations
from core.evaluation import PixelSampler, entropy_score
from core.models import load_model
from sae import ROOT
from sae.hooks import BLOCKS, PATCH, MixFFNTap, pool_to_patches
from sae.model import load_sae
from sae.train import ckpt_path

F0_NUM = ["ent_mean", "ent_max", "ent_p90", "msp_mean", "msp_min", "margin_mean", "margin_min",
          "boundary_frac", "row", "col"]


def base_path(dataset, image_id):
    return ROOT / "patches" / "base" / dataset / f"{image_id}.npz"


def codes_path(sae, dataset, image_id):
    return ROOT / "patches" / sae / dataset / f"{image_id}.npz"


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


def labels(gt, pred, patch=PATCH):
    """n_valid, n_err per patch; gt (H, W) uint8 numpy, pred (H, W) numpy."""
    valid = gt != IGNORE
    err = valid & (pred != gt)
    return (per_patch(valid, patch).sum(1).astype(np.int32), per_patch(err, patch).sum(1).astype(np.int32))


def frame_patches(acts, saes, n_patches_hw):
    """SAE codes (CSR, n × ΣD) and raw activations (n × Σ4C, fp16) of the closest positions."""
    codes, raw = [], []
    for b in BLOCKS:
        a = acts[b].float()                                   # (h, w, 4C)
        raw.append(pool_to_patches(a, b, n_patches_hw).half().cpu().numpy())
        if saes is not None:
            z = saes[b].encode(saes[b].normalize(a.reshape(-1, a.shape[2])))
            z = pool_to_patches(z.reshape(a.shape[0], a.shape[1], -1), b, n_patches_hw)
            codes.append(sp.csr_matrix(z.cpu().numpy()))
    return (sp.hstack(codes, format="csr", dtype=np.float32) if saes is not None else None,
            np.concatenate(raw, axis=1))


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    ap.add_argument("--sae", default="D2k8", help="SAE config (results/sae/ckpt/<sae>); 'none' — base only")
    ap.add_argument("--pixel-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N frames of every population")
    args = ap.parse_args()

    ids = sorted({i for p in populations(args.dataset) for i in population(args.dataset, p)[: args.limit]})
    ds = get_dataset(args.dataset, ids, full=False)
    assert ds.image_ids == sorted(ds.image_ids) == ids, "frames must come in sorted image_id order"
    test = {i: p for p in test_populations(args.dataset) for i in population(args.dataset, p)}
    W, H = ds.size
    hw = (H // PATCH, W // PATCH)

    b0 = load_model("b0")
    dev = b0.mean.device
    saes = None if args.sae == "none" else {b: load_sae(ckpt_path(args.sae, b), dev)[0] for b in BLOCKS}
    sampler = PixelSampler(args.pixel_frac, args.seed)
    base_path(args.dataset, "x").parent.mkdir(parents=True, exist_ok=True)
    if saes is not None:
        codes_path(args.sae, args.dataset, "x").parent.mkdir(parents=True, exist_ok=True)
    t0, nnz = time.time(), []
    with MixFFNTap(b0) as tap:
        for k, sample in enumerate(loader(ds), 1):
            image_id = sample["image_id"]
            probs = b0.probs(sample, ds.size)                   # (19, H, W)
            pred_t = probs.argmax(0)
            pred, gt = pred_t.cpu().numpy(), sample["label"].numpy()
            out = output_features(probs, pred_t)
            out["n_valid"], out["n_err"] = labels(gt, pred)
            codes, out["raw"] = frame_patches(tap.acts, saes, hw)
            if image_id in test:
                sel = sampler.frame(f"{args.dataset}_{test[image_id]}", gt, pred)
                ps = probs.reshape(NC, -1)[:, torch.from_numpy(sel).to(dev)]
                out["pix_idx"] = sel.astype(np.int32)
                out["pix_err"] = pred.ravel()[sel] != gt.ravel()[sel]
                out["pix_msp"] = (1 - ps.max(0).values).cpu().numpy()
                out["pix_ent"] = entropy_score(ps).cpu().numpy()
            np.savez(base_path(args.dataset, image_id), **out)
            if codes is not None:
                np.savez(codes_path(args.sae, args.dataset, image_id), data=codes.data, indices=codes.indices,
                         indptr=codes.indptr, shape=np.array(codes.shape))
                nnz.append(codes.nnz / codes.shape[0])
            if k % 50 == 0 or k == len(ds):
                print(f"{k}/{len(ds)}  {time.time() - t0:.0f} s"
                      + (f", non-zero codes per patch {np.mean(nnz):.0f}" if nnz else ""))
    (ROOT / "patches" / "base" / args.dataset / "_info.json").write_text(json.dumps(
        {"frames": len(ds), "patch": PATCH, "grid": hw, "blocks": BLOCKS,
         "d": [tap.acts[b].shape[2] for b in BLOCKS], "pixel_frac": args.pixel_frac, "seed": args.seed,
         "limit": args.limit}, indent=1))
    if saes is not None:
        (ROOT / "patches" / args.sae / args.dataset / "_info.json").write_text(json.dumps(
            {"frames": len(ds), "blocks": BLOCKS, "D": [saes[b].D for b in BLOCKS]}, indent=1))


if __name__ == "__main__":
    main()
