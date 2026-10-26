"""Step 2: one BatchTopK SAE per block on its native positions (sae.collect), then quality gates.

    uv run python -m sae.train                                   # reference: D = 2·4C, k = 8, 5 epochs, all 8 blocks
    uv run python -m sae.train --expansion 1 --blocks s3b1 s3b2 s4b1 s4b2
    uv run python -m sae.train --blocks s4b1 s4b2 --epochs 10    # if held-out FVE is still rising (recorded)
    uv run python -m sae.train --quality-only                    # recompute quality rows of existing checkpoints

Adam, lr 1e-4, batches of 4096 positions; config name D<expansion>k<k> (or --name).
Quality per block (results/sae/quality.csv, one row per config and block), on the held-out P0 frames with
the inference threshold θ: FVE, mean L0, dead share (latents that never fire), the same FVE on as many
training positions (gap), the held-out FVE gain of the last epoch (BatchTopK) and the splice test —
P0 mIoU drop when this block's fc2 input is replaced by its SAE reconstruction.
Gates: dead < 10%, splice drop ≤ 1.0 point.

Checkpoint: results/sae/ckpt/<config>/<block>.pt (weights, θ, normalisation, history).
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from core.components import IGNORE, NC
from core.data import FIT_DATASET, FIT_POP, get_dataset, loader, population
from core.evaluation import _conf, _miou
from core.models import load_model
from sae import ROOT
from sae.collect import ACTS
from sae.hooks import BLOCKS, MixFFNTap
from sae.model import BatchTopKSAE, load_sae

BATCH = 4096
QUALITY = ROOT / "quality.csv"
MAX_DEAD, MAX_SPLICE_DROP = 0.10, 1.0


def config_name(expansion, k):
    return f"D{expansion:g}k{k}"


def ckpt_path(config, block):
    return ROOT / "ckpt" / config / f"{block}.pt"


def load_block(block, device):
    """Training and held-out activations of the block (fp16 on device) and its normalisation."""
    meta = json.loads((ACTS / "meta.json").read_text())
    arr = np.load(ACTS / f"{block}.npy", mmap_mode="r")
    held = np.isin(np.load(ACTS / f"{block}_frame.npy"), meta["held_out"])
    norm = np.load(ACTS / f"{block}_norm.npz")
    X_train = torch.from_numpy(np.ascontiguousarray(arr[np.flatnonzero(~held)])).to(device)
    X_held = torch.from_numpy(np.ascontiguousarray(arr[np.flatnonzero(held)])).to(device)
    return X_train, X_held, norm["mean"], float(norm["scale"])


@torch.no_grad()
def evaluate(sae, X, topk=False, batch=BATCH):
    """FVE, mean L0 and share of latents that never fire over positions X (fp16, raw scale);
    topk — training-time BatchTopK encoding instead of the threshold."""
    sse, ss, s, n, l0 = 0.0, 0.0, torch.zeros(sae.d, dtype=torch.float64, device=X.device), 0, 0
    fired = torch.zeros(sae.D, dtype=torch.bool, device=X.device)
    for i in range(0, len(X), batch):
        x = sae.normalize(X[i:i + batch])
        z = sae.batch_topk(sae.pre_acts(x)) if topk else sae.encode(x)
        sse += (sae.decode(z) - x).double().pow(2).sum().item()
        ss += x.double().pow(2).sum().item()
        s += x.double().sum(0)
        n += len(x)
        l0 += (z > 0).sum().item()
        fired |= (z > 0).any(0)
    sst = ss - (s @ s).item() / n
    return {"fve": 1 - sse / sst, "l0": l0 / n, "dead": 1 - fired.float().mean().item()}


def train_block(block, X_train, X_held, mean, scale, expansion, k, epochs, lr, seed):
    device = X_train.device
    d = X_train.shape[1]
    torch.manual_seed(seed)
    sae = BatchTopKSAE(d, int(expansion * d), k, mean, scale).to(device)
    with torch.no_grad():                       # b_dec at the mean of the (normalised) training activations
        s = torch.zeros(d, dtype=torch.float64, device=device)
        for i in range(0, len(X_train), BATCH):
            s += sae.normalize(X_train[i:i + BATCH]).double().sum(0)
        sae.b_dec.copy_((s / len(X_train)).float())
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    history, t0 = [], time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_train), generator=gen).to(device)
        mse_sum, steps = 0.0, 0
        for i in range(0, len(perm), BATCH):
            loss, mse, _ = sae.loss(sae.normalize(X_train[perm[i:i + BATCH]]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.project_decoder_grad()
            opt.step()
            sae.normalize_decoder()
            mse_sum += mse.item()
            steps += 1
        h = evaluate(sae, X_held, topk=True)
        history.append({"epoch": epoch, "steps": steps, "train_mse": mse_sum / steps, "held_fve_topk": h["fve"]})
        print(f"  {block} epoch {epoch}: train mse {mse_sum / steps:.4f}, held-out FVE (BatchTopK) "
              f"{h['fve']:.4f}  {time.time() - t0:.0f} s")
    # θ: mean BatchTopK cut-off over the training batches (random batches as in training).
    perm = torch.randperm(len(X_train), generator=gen).to(device)
    cuts = [sae.batch_cutoff(sae.normalize(X_train[perm[i:i + BATCH]])) for i in range(0, len(perm), BATCH)]
    sae.threshold.fill_(torch.stack(cuts).mean().item())
    return sae.eval(), history


@torch.inference_mode()
def splice_drop(saes, limit=None):
    """P0 mIoU without SAEs and its drop when one block's fc2 input is replaced by the SAE reconstruction."""
    b0 = load_model("b0")
    ids = population(FIT_DATASET, FIT_POP)[:limit]
    ds = get_dataset(FIT_DATASET, ids, full=False)
    conf = {b: np.zeros((NC, NC), np.int64) for b in [None, *saes]}
    for sample in loader(ds):
        gt = sample["label"].numpy()
        v = gt != IGNORE
        g = gt[v].astype(np.int64)
        for b in conf:
            replace = {} if b is None else {b: lambda h, sae=saes[b]: sae.denormalize(sae.decode(sae.encode(sae.normalize(h))))}
            with MixFFNTap(b0, blocks=[], replace=replace):
                pred = b0.probs(sample, ds.size).argmax(0).cpu().numpy()
            conf[b] += _conf(g, pred[v].astype(np.int64))
    base = _miou(conf[None])
    return base, {b: base - _miou(conf[b]) for b in saes}


def write_quality(rows):
    new = pd.DataFrame(rows)
    if QUALITY.exists():
        old = pd.read_csv(QUALITY)
        keep = ~old.set_index(["config", "block"]).index.isin(new.set_index(["config", "block"]).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(["config", "block"]).reset_index(drop=True)
    QUALITY.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(QUALITY, index=False)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expansion", type=float, default=2, help="D = expansion · 4C")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--blocks", nargs="+", default=BLOCKS, choices=BLOCKS)
    ap.add_argument("--name", default=None, help="config name (default D<expansion>k<k>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--no-splice", action="store_true")
    ap.add_argument("--splice-limit", type=int, default=None, help="smoke test: splice on first N P0 frames")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.name or config_name(args.expansion, args.k)
    saes, rows = {}, []
    for b in args.blocks:
        X_train, X_held, mean, scale = load_block(b, device)
        path = ckpt_path(name, b)
        if args.quality_only:
            sae, ckpt = load_sae(path, device)
            history = ckpt["history"]
        else:
            print(f"{b}: {len(X_train)} training / {len(X_held)} held-out positions, d = {X_train.shape[1]}")
            sae, history = train_block(b, X_train, X_held, mean, scale, args.expansion, args.k,
                                       args.epochs, args.lr, args.seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": sae.config(), "state": sae.state_dict(), "block": b, "epochs": args.epochs,
                        "lr": args.lr, "seed": args.seed, "batch": BATCH, "history": history,
                        "n_train": len(X_train), "n_held": len(X_held)}, path)
        held = evaluate(sae, X_held)
        sub = torch.randperm(len(X_train), generator=torch.Generator().manual_seed(args.seed))[:len(X_held)]
        train = evaluate(sae, X_train[sub.to(device)])
        gain = history[-1]["held_fve_topk"] - history[-2]["held_fve_topk"] if len(history) > 1 else float("nan")
        rows.append({"config": name, "block": b, "d": sae.d, "D": sae.D, "k": sae.k, "epochs": len(history),
                     "steps": sum(h["steps"] for h in history), "threshold": sae.threshold.item(),
                     "fve_held": held["fve"], "fve_train": train["fve"], "l0_held": held["l0"],
                     "dead_held": held["dead"], "fve_last_epoch_gain": gain})
        print(f"{b}: held-out FVE {held['fve']:.4f} (train {train['fve']:.4f}), L0 {held['l0']:.2f}, "
              f"dead {100 * held['dead']:.1f}%, θ {sae.threshold.item():.4f}, last-epoch FVE gain {gain:+.4f}")
        saes[b] = sae
        del X_train, X_held
    if not args.no_splice:
        base, drop = splice_drop(saes, args.splice_limit)
        print(f"P0 mIoU without splice: {base:.2f}")
        for r in rows:
            r["miou_p0"], r["splice_drop"] = base, drop[r["block"]]
    for r in rows:
        r["gate_dead"] = r["dead_held"] < MAX_DEAD
        r["gate_splice"] = r["splice_drop"] <= MAX_SPLICE_DROP if "splice_drop" in r else None
    table = write_quality(rows)
    pd.set_option("display.width", 200)
    print(table[table.config == name].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
