"""Two models of hyperbolic space with curvature -c, implemented natively after Peng et al., "Hyperbolic Deep Neural
Networks: A Survey" (TPAMI 2022): the Poincaré ball (used by the HyperUL paper) and the Lorentz (hyperboloid) model.

Each model has its own origin, metric, exponential map at the origin (with respect to its own metric) and distance
formula. The models are isometric: the ball has metric 4I at the origin, so it equals the Lorentz model with v, p, a
doubled. In float32 they differ in where points saturate: sqrt(c) * distance <= max_rho, 7.6 for the ball and 15 for
the hyperboloid, which has no boundary.
"""
import math

import torch
import torch.nn as nn

EPS = 1e-5
ACOSH_EPS = 1e-6  # acosh arguments are kept >= 1 + ACOSH_EPS for finite gradients
BALL_EPS = 1e-3  # the ball keeps sqrt(c) * |x| <= 1 - BALL_EPS
MAX_RHO = 15.0  # hyperboloid: farther out, float32 inner products lose precision


def acosh(z):
    return torch.acosh(z.clamp_min(1 + ACOSH_EPS))


def sqnorm(x):
    return x.pow(2).sum(1)


class Manifold:
    scale = 1.0  # Riemannian length of a Euclidean unit vector at the origin
    max_rho = MAX_RHO

    def __init__(self, c):
        self.c, self.sqrt_c = c, c ** 0.5

    def expmap0(self, v):
        """Exponential map at the origin; v are Euclidean coordinates of a tangent vector there."""
        norm = v.norm(dim=1, keepdim=True).clamp_min(EPS)
        rho = (self.scale * self.sqrt_c * norm).clamp(max=self.max_rho)
        return self.geodesic(rho, v / norm)

    def dist0(self, x):
        """Hyperbolic distance to the origin."""
        return self.dist(x, self.origin(x))

    def origin(self, x):
        return torch.zeros_like(x[:1])

    def geodesic(self, rho, u):
        """Point at distance rho / sqrt(c) from the origin on the geodesic with unit direction u."""
        raise NotImplementedError

    def dist(self, x, y):
        raise NotImplementedError


class Poincare(Manifold):
    """Ball |x| < 1/sqrt(c) with metric (2 / (1 - c|x|^2))^2 I, origin 0."""
    scale = 2.0
    max_rho = 2 * math.atanh(1 - BALL_EPS)

    def geodesic(self, rho, u):
        return torch.tanh(rho / 2) / self.sqrt_c * u

    def dist(self, x, y):
        c = self.c
        return acosh(1 + 2 * c * sqnorm(x - y) / ((1 - c * sqnorm(x)) * (1 - c * sqnorm(y)))) / self.sqrt_c


class Lorentz(Manifold):
    """Hyperboloid <x, x>_L = -1/c, x_0 > 0 in Minkowski space, origin (1/sqrt(c), 0, ..., 0)."""

    def geodesic(self, rho, u):
        return torch.cat([torch.cosh(rho), torch.sinh(rho) * u], 1) / self.sqrt_c

    def origin(self, x):
        o = torch.zeros_like(x[:1])
        o[:, 0] = 1 / self.sqrt_c
        return o

    def dist(self, x, y):
        minkowski = (x[:, 1:] * y[:, 1:]).sum(1) - x[:, 0] * y[:, 0]
        return acosh(-self.c * minkowski) / self.sqrt_c


MANIFOLDS = {"poincare": Poincare, "lorentz": Lorentz}


def hyperbolic_uncertainty(dist):
    """Eq. 15: 1 - d_c(x, 0) / d_c(u, 0), u is the most confident pixel (patch, for patch means) of the image.
    dist: (B, H, W)."""
    d_max = dist.flatten(1).amax(1).clamp_min(EPS)
    return 1 - dist / d_max[:, None, None]


class HyperbolicMLR(nn.Module):
    """Hyperbolic MLR (Ganea et al., 2018): logit_k = |a_k| * signed distance to the hyperplane through
    p_k = exp_0(p_k') orthogonal to a_k, the parallel transport of a_k' from the origin to p_k.

    Computed natively in each model: on gyroplanes (Möbius addition) in the Poincaré ball, with Lorentz parallel
    transport and hyperplanes <a_k, x>_L = 0 on the hyperboloid.
    """

    def __init__(self, dim, num_classes, manifold):
        super().__init__()
        self.manifold = manifold
        self.p = nn.Parameter(torch.empty(num_classes, dim))
        self.a = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.kaiming_uniform_(self.p, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.a, a=5 ** 0.5)

    def forward(self, x):
        return self.gyroplanes(x) if isinstance(self.manifold, Poincare) else self.hyperboloid(x)

    def gyroplanes(self, z):
        """(B, D, H, W) ball points; -p ⊕ z is expanded through inner products, no (B, K, D, H, W) tensor."""
        c, sqrt_c = self.manifold.c, self.manifold.sqrt_c
        p = self.manifold.expmap0(self.p)
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

    def hyperboloid(self, x):
        """(B, D + 1, H, W) points on the hyperboloid; hyperplane k is <a_k, x>_L = 0."""
        c, sqrt_c = self.manifold.c, self.manifold.sqrt_c
        p = self.manifold.expmap0(self.p)
        p_time, p_space = p[:, :1], p[:, 1:]
        transport = c * (p_space * self.a).sum(1, keepdim=True) / (1 + sqrt_c * p_time)
        a_time, a_space = transport * (1 / sqrt_c + p_time), self.a + transport * p_space
        inner = torch.einsum("bdhw,kd->bkhw", x[:, 1:], a_space) - x[:, :1] * a_time[:, 0, None, None]
        norm = self.a.norm(dim=1).clamp_min(EPS)[:, None, None]  # parallel transport keeps |a_k|
        return norm / sqrt_c * torch.asinh(sqrt_c * inner / norm)
