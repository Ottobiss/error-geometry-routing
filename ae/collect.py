"""Step 1: Mix-FFN hidden activations of B0 on Cityscapes train for the encoders (PLANv2.md §2, §3).

    uv run python -m ae.collect
    uv run python -m ae.collect --limit 20             # smoke test: first 20 train frames

The same procedure as sae.collect, on the 2,975 Cityscapes train frames instead of P0: every frame gives each
block min(N, 2M // n_frames) random positions of its native grid (672 per frame for stages 1–3, i.e. ≈ 2M;
all 512 for stage 4, ≈ 1.52M). 5% of the frames (seed 0) are held out for early stopping. Normalisation per
block (mean, scale with E‖x‖² = 4C) is fitted on the training frames and used for P0, P1 and ACDC as well.

The train split is read through Cityscapes(split="train"); there is deliberately no splits/ file for it
(core.data.test_populations would treat it as a test population).

Output in results/ae/acts/ (same layout as results/sae/acts/).
"""
import argparse

from core.data import get_dataset
from ae import ROOT
from sae.collect import collect

ACTS = ROOT / "acts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positions", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N train frames")
    args = ap.parse_args()

    ds = get_dataset("cityscapes", full=False, split="train")
    assert len(ds) == 2975 or args.limit, f"{len(ds)} Cityscapes train frames on disk, expected 2975"
    ids = ds.image_ids[: args.limit]
    ds = get_dataset("cityscapes", ids, full=False, split="train")
    assert ds.image_ids == ids == sorted(ids)
    collect(ds, ACTS, args.max_positions, args.seed)


if __name__ == "__main__":
    main()
