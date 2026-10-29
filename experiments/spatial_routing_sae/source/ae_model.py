"""Encoders of normalised Mix-FFN activations (PLANv2.md §4): bottleneck autoencoder and PCA.

Both work on x = (h − mean) / scale (ae.collect) and expose the interface of sae.model.BatchTopKSAE that
sae.train.splice_drop uses: normalize, encode, decode, denormalize. z is dense and signed.

  BottleneckAE  x → Linear(d, h) → GELU → Linear(h, b) = z → Linear(b, h) → GELU → Linear(h, d) = x̂, h = d
  PCA           z = (x − μ) V_b, x̂ = μ + z V_bᵀ, V_b the top-b eigenvectors of the covariance of x
"""
import torch
from torch import nn


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

