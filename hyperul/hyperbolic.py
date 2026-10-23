import torch
import torch.nn as nn

EPS = 1e-5
BALL_EPS = 1e-3


def project(x, c, dim=1):
    max_norm = (1 - BALL_EPS) / c ** 0.5
    norm = x.norm(dim=dim, keepdim=True).clamp_min(EPS)
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def expmap0(v, c, dim=1):
    """Eq. 2 at x = 0: Euclidean (tangent) vectors -> Poincaré ball."""
    sqrt_c = c ** 0.5
    norm = v.norm(dim=dim, keepdim=True).clamp_min(EPS)
    return project(torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm), c, dim)


def dist0(x, c, dim=1):
    """Eq. 9: hyperbolic distance to the origin."""
    sqrt_c = c ** 0.5
    return 2 / sqrt_c * torch.atanh((sqrt_c * x.norm(dim=dim)).clamp(max=1 - BALL_EPS))


def hyperbolic_uncertainty(dist):
    """Eq. 15: 1 - d_c(x, 0) / d_c(u, 0), u is the most confident pixel of the image. dist: (B, H, W)."""
    d_max = dist.flatten(1).amax(1).clamp_min(EPS)
    return 1 - dist / d_max[:, None, None]


class HyperbolicMLR(nn.Module):
    """Hyperbolic multinomial logistic regression (Ganea et al., 2018) for (B, D, H, W) points on the ball.

    The Möbius addition -p_k ⊕ z (Eq. 1) is expanded through inner products,
    so no (B, K, D, H, W) tensor is materialized.
    """

    def __init__(self, dim, num_classes, c):
        super().__init__()
        self.c = c
        self.p = nn.Parameter(torch.empty(num_classes, dim))
        self.a = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.kaiming_uniform_(self.p, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, z):
        c, sqrt_c = self.c, self.c ** 0.5
        p = expmap0(self.p, c)
        a = self.a * (1 - c * p.pow(2).sum(1, keepdim=True))
        pp = p.pow(2).sum(1)[:, None, None]
        pa = (p * a).sum(1)[:, None, None]
        a_norm = a.norm(dim=1).clamp_min(EPS)[:, None, None]

        zz = z.pow(2).sum(1, keepdim=True)
        pz = torch.einsum("bdhw,kd->bkhw", z, p)
        az = torch.einsum("bdhw,kd->bkhw", z, a)

        # w = -p ⊕ z = (beta * z - alpha * p) / gamma
        alpha = 1 - 2 * c * pz + c * zz
        beta = 1 - c * pp
        gamma = 1 - 2 * c * pz + c ** 2 * pp * zz
        wa = (beta * az - alpha * pa) / gamma
        ww = (beta ** 2 * zz - 2 * alpha * beta * pz + alpha ** 2 * pp) / gamma ** 2

        lam = 2 / (1 - c * pp)
        return lam * a_norm / sqrt_c * torch.asinh(2 * sqrt_c * wa / ((1 - c * ww).clamp_min(EPS) * a_norm))
