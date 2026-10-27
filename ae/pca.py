"""Step 2: PCA per block on the same normalised Cityscapes-train activations (PLANv2.md §4.2).

    uv run python -m ae.pca                  # b = d/8 → results/ae/ckpt/pca_b8/
    uv run python -m ae.pca --div 16

Exact PCA: mean and covariance of x over the training frames (the held-out frames are left out, as for the
AE) accumulated in float64, then the eigendecomposition; the top-b eigenvectors are kept. The sign of each
component is fixed so that its largest-magnitude coordinate is positive.
"""
import argparse

import pandas as pd
import torch

from ae import ROOT
from ae.collect import ACTS
from ae.model import PCA
from ae.train import BATCH, ckpt_path, encoder_name, evaluate, upsert_csv
from sae.hooks import BLOCKS
from sae.train import load_block


@torch.no_grad()
def fit_pca(X, mean, scale, b, batch=BATCH * 16):
    """PCA of the normalised positions X (raw scale, on device) with b components and the explained
    variance share of each of them (of all d)."""
    d = X.shape[1]
    pca = PCA(d, b, mean=mean, scale=scale).to(X.device)
    s = torch.zeros(d, dtype=torch.float64, device=X.device)
    xtx = torch.zeros(d, d, dtype=torch.float64, device=X.device)
    for i in range(0, len(X), batch):
        x = pca.normalize(X[i:i + batch]).double()
        s += x.sum(0)
        xtx += x.T @ x
    mu = s / len(X)
    cov = xtx / len(X) - torch.outer(mu, mu)
    evals, evecs = torch.linalg.eigh(cov)                   # ascending
    evals, evecs = evals.flip(0), evecs.flip(1)
    V = evecs[:, :b].T                                      # (b, d)
    sign = torch.sign(V.gather(1, V.abs().argmax(1, keepdim=True)))
    pca.components.copy_((V * sign).float())
    pca.center.copy_(mu.float())
    return pca.eval(), (evals[:b] / evals.sum()).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--div", type=int, default=8, help="components b = d / div")
    ap.add_argument("--blocks", nargs="+", default=BLOCKS, choices=BLOCKS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = encoder_name("pca", args.div)
    rows = []
    for blk in args.blocks:
        X_train, X_held, mean, scale = load_block(blk, device, ACTS)
        d = X_train.shape[1]
        assert d % args.div == 0, (blk, d, args.div)
        pca, explained = fit_pca(X_train, mean, scale, d // args.div)
        h = evaluate(pca, X_held)
        path = ckpt_path(name, blk)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"kind": "pca", "config": pca.config(), "state": pca.state_dict(), "block": blk,
                    "explained": explained, "n_train": len(X_train), "n_held": len(X_held)}, path)
        rows.append({"encoder": name, "block": blk, "d": d, "b": pca.b, "explained_train": float(explained.sum()),
                     "held_mse": h["mse"], "held_fve": h["fve"]})
        print(f"{blk}: b = {pca.b}, explained variance (train) {explained.sum():.4f}, held-out FVE {h['fve']:.4f}")
        del X_train, X_held
    table = upsert_csv(ROOT / "ckpt" / name / "train.csv", rows, ["encoder", "block"])
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
