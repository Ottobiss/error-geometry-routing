"""Step 2: one bottleneck autoencoder per block on Cityscapes-train activations (PLANv2.md §4.1).

    uv run python -m ae.train                          # main: b = d/8 → results/ae/ckpt/ae_b8/
    uv run python -m ae.train --div 16                 # ablation b = d/16 (§8.4)
    uv run python -m ae.train --blocks s1b1 --epochs 2 # smoke test

MSE on normalised x, AdamW (lr 1e-3, weight decay 1e-4 on all parameters), cosine schedule over the planned
epochs (per step), batch 4,096, seed 0. Up to 30 epochs; after every epoch the MSE on the held-out train
frames is measured, training stops after 3 epochs without improvement and the best epoch's weights are kept.

Checkpoint: results/ae/ckpt/<name>/<block>.pt (weights, normalisation, history); summary of the fit in
results/ae/ckpt/<name>/train.csv. Quality per population and the splice test: ae.quality.
"""
import argparse
import copy
import math
import time

import pandas as pd
import torch

from ae import ROOT
from ae.collect import ACTS
from ae.model import BottleneckAE
from sae.hooks import BLOCKS
from sae.train import load_block

BATCH = 4096


def encoder_name(kind, div):
    return f"{kind}_b{div}"


def ckpt_path(name, block):
    return ROOT / "ckpt" / name / f"{block}.pt"


@torch.no_grad()
def evaluate(enc, X, batch=BATCH):
    """FVE and mean squared error per dimension of the reconstruction of positions X (raw scale, on device),
    measured on normalised x; FVE is relative to the variance of X around its own mean."""
    sse, ss, s, n = 0.0, 0.0, torch.zeros(enc.d, dtype=torch.float64, device=X.device), 0
    for i in range(0, len(X), batch):
        x = enc.normalize(X[i:i + batch])
        sse += (enc.decode(enc.encode(x)) - x).double().pow(2).sum().item()
        ss += x.double().pow(2).sum().item()
        s += x.double().sum(0)
        n += len(x)
    return {"fve": 1 - sse / (ss - (s @ s).item() / n), "mse": sse / (n * enc.d)}


def upsert_csv(path, rows, keys):
    """Write rows into the CSV at path, replacing rows with the same keys."""
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        keep = ~old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[keep], new], ignore_index=True)
    new = new.sort_values(keys).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return new


def train_block(block, X_train, X_held, mean, scale, b, epochs, patience, lr, wd, seed):
    device = X_train.device
    d = X_train.shape[1]
    torch.manual_seed(seed)
    ae = BottleneckAE(d, b, mean=mean, scale=scale).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = math.ceil(len(X_train) / BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)
    gen = torch.Generator().manual_seed(seed)
    history, best, best_state, wait, t0 = [], math.inf, None, 0, time.time()
    for epoch in range(1, epochs + 1):
        ae.train()
        perm = torch.randperm(len(X_train), generator=gen).to(device)
        mse_sum = 0.0
        for i in range(0, len(perm), BATCH):
            x = ae.normalize(X_train[perm[i:i + BATCH]])
            loss = (ae.decode(ae.encode(x)) - x).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            mse_sum += loss.item()
        ae.eval()
        h = evaluate(ae, X_held)
        history.append({"epoch": epoch, "steps": steps_per_epoch, "lr_end": sched.get_last_lr()[0],
                        "train_mse": mse_sum / steps_per_epoch, "held_mse": h["mse"], "held_fve": h["fve"]})
        print(f"  {block} epoch {epoch}: train mse {mse_sum / steps_per_epoch:.4f}, held-out mse {h['mse']:.4f}, "
              f"FVE {h['fve']:.4f}  {time.time() - t0:.0f} s")
        if h["mse"] < best:
            best, best_state, wait = h["mse"], copy.deepcopy(ae.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    ae.load_state_dict(best_state)
    return ae.eval(), history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--div", type=int, default=8, help="bottleneck b = d / div")
    ap.add_argument("--blocks", nargs="+", default=BLOCKS, choices=BLOCKS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = encoder_name("ae", args.div)
    rows = []
    for blk in args.blocks:
        X_train, X_held, mean, scale = load_block(blk, device, ACTS)
        d = X_train.shape[1]
        assert d % args.div == 0, (blk, d, args.div)
        b = d // args.div
        print(f"{blk}: {len(X_train)} training / {len(X_held)} held-out positions, d = {d}, b = {b}")
        ae, history = train_block(blk, X_train, X_held, mean, scale, b, args.epochs, args.patience, args.lr,
                                  args.wd, args.seed)
        best = min(history, key=lambda h: h["held_mse"])
        path = ckpt_path(name, blk)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "ae", "config": ae.config(), "state": ae.state_dict(), "block": blk,
                    "epochs_max": args.epochs, "patience": args.patience, "lr": args.lr, "wd": args.wd,
                    "seed": args.seed, "batch": BATCH, "history": history, "best_epoch": best["epoch"],
                    "n_train": len(X_train), "n_held": len(X_held)}, path)
        rows.append({"encoder": name, "block": blk, "d": d, "b": b, "epochs_run": len(history),
                     "best_epoch": best["epoch"], "steps_run": sum(h["steps"] for h in history),
                     "steps_to_best": sum(h["steps"] for h in history[:best["epoch"]]),
                     "held_mse": best["held_mse"], "held_fve": best["held_fve"],
                     "stopped_early": len(history) < args.epochs})
        print(f"{blk}: best epoch {best['epoch']} of {len(history)}, held-out FVE {best['held_fve']:.4f}")
        del X_train, X_held
    table = upsert_csv(ROOT / "ckpt" / name / "train.csv", rows, ["encoder", "block"])
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
