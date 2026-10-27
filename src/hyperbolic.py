"""Two models of hyperbolic space with curvature -c, implemented natively after Peng et al., "Hyperbolic Deep Neural
Networks: A Survey" (TPAMI 2022): the Poincaré ball (used by the HyperUL paper) and the Lorentz (hyperboloid) model.

Each model has its own origin, metric, exponential map at the origin (with respect to its own metric) and distance
formula. The models are isometric: the ball has metric 4I at the origin, so it equals the Lorentz model with v doubled,
and distances to the origin agree up to where points saturate in float32: sqrt(c) * distance <= max_rho, 7.6 for the
ball and 15 for the hyperboloid, which has no boundary. Their coordinates differ: the ball's are bounded by 1/sqrt(c),
the hyperboloid's space-like ones grow like sinh of the distance.
"""
import math

import torch

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

    def coordinates(self, x):
        """n coordinates of a point of the n-dimensional space (the class logits of HyperSegFormer)."""
        return x

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

    def coordinates(self, x):
        return x[:, 1:]  # space-like; the time-like x_0 follows from them

    def dist(self, x, y):
        minkowski = (x[:, 1:] * y[:, 1:]).sum(1) - x[:, 0] * y[:, 0]
        return acosh(-self.c * minkowski) / self.sqrt_c


MANIFOLDS = {"poincare": Poincare, "lorentz": Lorentz}


def hyperbolic_uncertainty(dist):
    """Eq. 15: 1 - d_c(x, 0) / d_c(u, 0), u is the most confident pixel (patch, for patch means) of the image.
    dist: (B, H, W)."""
    d_max = dist.flatten(1).amax(1).clamp_min(EPS)
    return 1 - dist / d_max[:, None, None]
