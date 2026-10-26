"""BatchTopK sparse autoencoder (Bussmann et al. 2024) with AuxK for dead latents (Gao et al. 2024).

z = ReLU((x − b_dec) W_enc + b_enc), x̂ = z W_dec + b_dec; decoder rows (one per latent) have unit norm.
Training keeps the k·B largest activations over the batch (k per position on average); inference keeps
activations above the global threshold θ, estimated after training as the mean BatchTopK cut-off over the
training batches, so every position is encoded on its own. The SAE works on normalised activations
x = (h − mean) / scale (sae.collect); mean and scale are stored as buffers with the weights.
"""
import torch
import torch.nn.functional as F
from torch import nn

AUX_COEF = 1 / 32              # AuxK weight (Gao et al. 2024)
AUX_K = 512                    # dead latents used by AuxK per position (reference BatchTopK code)
DEAD_BATCHES = 5               # latent counts as dead for AuxK after this many batches without firing


class BatchTopKSAE(nn.Module):
    def __init__(self, d, D, k, mean=None, scale=1.0):
        super().__init__()
        self.d, self.D, self.k = d, D, k
        W = torch.empty(D, d)
        nn.init.kaiming_uniform_(W)
        W = F.normalize(W, dim=1)
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(D))
        self.b_dec = nn.Parameter(torch.zeros(d))
        self.register_buffer("threshold", torch.tensor(0.0))
        self.register_buffer("mean", torch.zeros(d) if mean is None else torch.as_tensor(mean).float())
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("inactive", torch.zeros(D))

    def normalize(self, h):
        return (h.float() - self.mean) / self.scale

    def denormalize(self, x):
        return x * self.scale + self.mean

    def pre_acts(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def batch_topk(self, acts):
        flat = acts.flatten()
        top = flat.topk(min(self.k * acts.shape[0], flat.numel()))
        return torch.zeros_like(flat).scatter(0, top.indices, top.values).view_as(acts)

    def encode(self, x):
        """Inference: each position on its own, activations above θ."""
        acts = self.pre_acts(x)
        return acts * (acts > self.threshold)

    def decode(self, z):
        return z @ self.W_dec + self.b_dec

    def loss(self, x):
        """Training step on a batch of normalised positions: (loss, mse, z)."""
        acts = self.pre_acts(x)
        z = self.batch_topk(acts)
        x_hat = self.decode(z)
        mse = (x_hat - x).pow(2).mean()
        with torch.no_grad():
            self.inactive += 1
            self.inactive[(z > 0).any(0)] = 0
        dead = self.inactive >= DEAD_BATCHES
        aux = torch.zeros((), device=x.device)
        if dead.any():
            residual = (x - x_hat).detach()
            a = acts[:, dead]
            top = a.topk(min(AUX_K, a.shape[1]), dim=1)
            z_aux = torch.zeros_like(a).scatter(1, top.indices, top.values)
            aux = (z_aux @ self.W_dec[dead] - residual).pow(2).mean()
        return mse + AUX_COEF * aux, mse, z

    @torch.no_grad()
    def project_decoder_grad(self):
        """Remove the gradient component along each decoder row (rows stay on the unit sphere)."""
        W, g = self.W_dec, self.W_dec.grad
        if g is not None:
            g -= (g * W).sum(1, keepdim=True) * W

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))

    @torch.no_grad()
    def batch_cutoff(self, x):
        """Smallest positive activation kept by BatchTopK on batch x (for θ)."""
        z = self.batch_topk(self.pre_acts(x))
        pos = z[z > 0]
        return pos.min() if len(pos) else torch.zeros((), device=x.device)

    def config(self):
        return {"d": self.d, "D": self.D, "k": self.k}


def load_sae(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sae = BatchTopKSAE(**ckpt["config"]).to(device)
    sae.load_state_dict(ckpt["state"])
    return sae.eval(), ckpt
