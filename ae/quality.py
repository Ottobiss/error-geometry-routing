"""Step 2: encoder quality per block and population, no gates (PLANv2.md §4.3).

    uv run python -m ae.quality --encoders ae_b8 pca_b8
    uv run python -m ae.quality --encoders ae_b16 ae_b4 --no-splice
    uv run python -m ae.quality --encoders ae_b8 --limit 3 --splice-limit 3      # smoke test

For every encoder and block, on normalised x:
  fve_<pop>   1 − SSE / SST, SST around the population's own mean
  mse_<pop>   mean squared reconstruction error per dimension, i.e. the mean position residual r = ‖x − x̂‖² / d
  norm_<pop>  mean ‖x‖² / d (the same for every encoder)
Populations: train_held (held-out Cityscapes-train frames, from results/ae/acts), cityscapes_p0,
cityscapes_p1, acdc_<condition> and acdc_all (pooled positions); all native positions of every frame.
Splice test on P0 (v1 definition, sae.train.splice_drop): mIoU drop when the block's fc2 input is replaced
by its reconstruction. Descriptive only; a tight bottleneck is expected to fail the v1 gate of 1.0.

Output: results/ae/quality.csv, one row per encoder and block (rows of other encoders are kept).
"""
import argparse

import pandas as pd
import torch

from ae import ROOT
from ae.collect import ACTS
from ae.model import load_encoder
from ae.train import ckpt_path, evaluate, upsert_csv
from core.data import get_dataset, loader, population, populations
from core.models import load_model
from sae.hooks import BLOCKS, MixFFNTap
from sae.train import load_block, splice_drop

QUALITY = ROOT / "quality.csv"


def load_encoders(names, device):
    """{name: {block: encoder}}; all encoders must share the normalisation of results/ae/acts."""
    encs = {n: {b: load_encoder(ckpt_path(n, b), device)[0] for b in BLOCKS} for n in names}
    for b in BLOCKS:
        ref = encs[names[0]][b]
        for n in names[1:]:
            assert torch.equal(encs[n][b].mean, ref.mean) and torch.equal(encs[n][b].scale, ref.scale), (n, b)
    return encs


@torch.inference_mode()
def population_stats(b0, encs, dataset, ids):
    """Sums over all native positions of the frames ids: {(encoder, block): {sse, ss, s, n}}."""
    ds = get_dataset(dataset, ids, full=False)
    assert ds.image_ids == sorted(ids)
    acc = {}
    with MixFFNTap(b0) as tap:
        for sample in loader(ds):
            b0.probs(sample, ds.size)
            for blk in BLOCKS:
                a = tap.acts[blk].reshape(-1, tap.acts[blk].shape[-1])
                for name, e in encs.items():
                    enc = e[blk]
                    x = enc.normalize(a)
                    st = acc.setdefault((name, blk), {"sse": 0.0, "ss": 0.0, "s": 0.0, "n": 0})
                    st["sse"] += (enc.decode(enc.encode(x)) - x).double().pow(2).sum().item()
                    st["ss"] += x.double().pow(2).sum().item()
                    st["s"] = st["s"] + x.double().sum(0)
                    st["n"] += len(x)
    return acc


def summarize(st, d):
    s = st["s"]
    return {"fve": 1 - st["sse"] / (st["ss"] - (s @ s).item() / st["n"]), "mse": st["sse"] / (st["n"] * d),
            "norm": st["ss"] / (st["n"] * d)}


def add(a, b):
    return {k: a[k] + b[k] for k in a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", required=True)
    ap.add_argument("--no-splice", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N frames of every population")
    ap.add_argument("--splice-limit", type=int, default=None, help="smoke test: splice on first N P0 frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encs = load_encoders(args.encoders, device)
    rows = {(n, b): {"encoder": n, "block": b, "d": encs[n][b].d, "b": encs[n][b].b} for n in encs for b in BLOCKS}

    for blk in BLOCKS:                                       # held-out Cityscapes-train frames
        _, X_held, _, _ = load_block(blk, device, ACTS)
        for n in encs:
            h = evaluate(encs[n][blk], X_held)
            x = encs[n][blk].normalize(X_held).double()      # fits on the GPU: ≤ 76k × 1,024
            rows[(n, blk)].update({"fve_train_held": h["fve"], "mse_train_held": h["mse"],
                                   "norm_train_held": x.pow(2).mean().item()})
        del X_held

    b0 = load_model("b0")
    for dataset, pops in (("cityscapes", ["p0", "p1"]), ("acdc", populations("acdc"))):
        total = {}
        for p in pops:
            acc = population_stats(b0, encs, dataset, population(dataset, p)[: args.limit])
            for key, st in acc.items():
                rows[key].update({f"{k}_{dataset}_{p}": v for k, v in summarize(st, rows[key]["d"]).items()})
                total[key] = add(total[key], st) if key in total else st
            print(f"{dataset}:{p} done")
        if len(pops) > 1 and dataset == "acdc":
            for key, st in total.items():
                rows[key].update({f"{k}_{dataset}_all": v for k, v in summarize(st, rows[key]["d"]).items()})
    del b0

    if not args.no_splice:
        for n in encs:
            base, drop = splice_drop(encs[n], args.splice_limit)
            for blk in BLOCKS:
                rows[(n, blk)].update({"miou_p0": base, "splice_drop": drop[blk]})
    table = upsert_csv(QUALITY, list(rows.values()), ["encoder", "block"])
    pd.set_option("display.width", 250)
    cols = ["encoder", "block", "b"] + [c for c in table.columns if c.startswith("fve_")] + \
           [c for c in ("splice_drop",) if c in table.columns]
    print(table[table.encoder.isin(args.encoders)][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
