"""Step 1: Mix-FFN hidden activations of B0 on Cityscapes P0 for SAE training (PLAN.md §4).

    uv run python -m sae.collect
    uv run python -m sae.collect --limit 5             # smoke test: first 5 P0 frames

Every P0 frame gives each block min(N, max_positions // n_frames) random positions of its native grid
(N = positions per frame): about 2M for stages 1–2, all positions for stages 3–4. 5% of P0 frames
(seed 0) are held out for SAE validation. Normalisation per block is fitted on the training frames:
x = (h − mean) / scale with E‖x‖² = 4C.

Output in results/sae/acts/:
  <block>.npy          (n, 4C) fp16 activations
  <block>_frame.npy    (n,) int16, number of the frame in meta.json "frames"
  <block>_norm.npz     mean (4C,) float32, scale () float32
  meta.json            frames, held-out frame numbers, positions per frame and block
"""
import argparse
import json
import time

import numpy as np
import torch

from core.data import FIT_DATASET, FIT_POP, get_dataset, loader, population
from core.models import load_model
from sae import ROOT
from sae.hooks import BLOCKS, MixFFNTap

HOLDOUT = 0.05


ACTS = ROOT / "acts"


def fit_norm(arr, rows, chunk=1 << 18):
    """mean (d,) and scale with E‖(h − mean) / scale‖² = d over arr[rows], accumulated in float64."""
    d = arr.shape[1]
    s, ss, n = np.zeros(d), 0.0, 0
    for i in range(0, len(rows), chunk):
        x = arr[rows[i:i + chunk]].astype(np.float64)
        s += x.sum(0)
        ss += (x * x).sum()
        n += len(x)
    mean = s / n
    var = ss / n - mean @ mean                  # E‖h − mean‖²
    return mean.astype(np.float32), np.float32(np.sqrt(var / d))


@torch.inference_mode()
def collect(ds, out, max_positions, seed):
    """Activations of all frames of ds (in ds order) into out/: see the module docstring."""
    ids = ds.image_ids
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    held = np.sort(rng.choice(len(ids), max(1, round(HOLDOUT * len(ids))), replace=False))

    b0 = load_model("b0")
    arrs, frames, quota, t0 = {}, {}, {}, time.time()
    with MixFFNTap(b0) as tap:
        for f, sample in enumerate(loader(ds)):
            assert sample["image_id"] == ids[f]
            b0.probs(sample, ds.size)
            for b in BLOCKS:
                a = tap.acts[b].reshape(-1, tap.acts[b].shape[-1])
                if b not in arrs:
                    quota[b] = min(len(a), max_positions // len(ids))
                    arrs[b] = np.lib.format.open_memmap(out / f"{b}.npy", mode="w+", dtype=np.float16,
                                                        shape=(quota[b] * len(ids), a.shape[1]))
                    frames[b] = np.repeat(np.arange(len(ids), dtype=np.int16), quota[b])
                sel = np.arange(len(a)) if quota[b] == len(a) else np.sort(rng.choice(len(a), quota[b], replace=False))
                arrs[b][f * quota[b]:(f + 1) * quota[b]] = a[torch.from_numpy(sel).to(a.device)].half().cpu().numpy()
            if (f + 1) % 50 == 0 or f + 1 == len(ids):
                print(f"{f + 1}/{len(ids)}  {time.time() - t0:.0f} s")

    train_frames = np.setdiff1d(np.arange(len(ids)), held)
    for b in BLOCKS:
        arrs[b].flush()
        np.save(out / f"{b}_frame.npy", frames[b])
        rows = np.flatnonzero(np.isin(frames[b], train_frames))
        mean, scale = fit_norm(arrs[b], rows)
        np.savez(out / f"{b}_norm.npz", mean=mean, scale=scale)
        print(f"{b}: {arrs[b].shape[0]} positions ({quota[b]} per frame), d = {arrs[b].shape[1]}, "
              f"scale {scale:.3f}, |mean| {np.linalg.norm(mean):.3f}")
    meta = {"frames": ids, "held_out": held.tolist(), "per_frame": quota,
            "max_positions": max_positions, "seed": seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positions", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N P0 frames")
    args = ap.parse_args()

    ids = population(FIT_DATASET, FIT_POP)[: args.limit]
    ds = get_dataset(FIT_DATASET, ids, full=False)
    assert ds.image_ids == ids
    collect(ds, ACTS, args.max_positions, args.seed)


if __name__ == "__main__":
    main()
