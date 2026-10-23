import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hyperbolic import hyperbolic_uncertainty


class HyperUL(nn.Module):
    """Hyperbolic Uncertainty Loss (Eq. 14): cross-entropy weighted by uw = 1 / log(t + d) (Eq. 10-11)
    and computed only on the pixels selected by OHUEM (Eq. 13)."""

    def __init__(self, t=1.869, hr=1.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.t, self.hr, self.ignore_index = t, hr, ignore_index
        weights = None if class_weights is None else torch.tensor(list(class_weights), dtype=torch.float)
        self.register_buffer("class_weights", weights)

    def forward(self, logits, dist, target):
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=self.ignore_index,
                             reduction="none")
        d = 1 - hyperbolic_uncertainty(dist.detach())
        weight = 1 / torch.log(self.t + d)

        # Eq. 13: keep pixels below the threshold between the least and the most confident pixel of the image
        d_min = d.flatten(1).amin(1)[:, None, None]
        valid = target != self.ignore_index
        keep = valid & (d <= d_min + self.hr * (1 - d_min))

        n = keep.sum().clamp_min(1)
        loss = (weight * ce * keep).sum() / n
        stats = {
            "weight": (weight * keep).sum() / n,
            "selected": keep.sum() / valid.sum().clamp_min(1),
            "dist": dist.detach().mean(),
        }
        return loss, stats


LARGE_UNC_LOSSES = {  # logit z of the predicted large-model uncertainty, target u in [0, 1] -> per-pixel loss
    "bce": lambda z, u: F.binary_cross_entropy_with_logits(z, u, reduction="none"),
    "mse": lambda z, u: (z.sigmoid() - u) ** 2,
    "l1": lambda z, u: (z.sigmoid() - u).abs(),
}


class RoutingLoss(nn.Module):
    """HyperUL as in the paper (segmentation weighted by the own hyperbolic uncertainty). Given the large model's
    uncertainty exported by src.export_uncertainty, also a loss (bce | mse | l1) between it and the small model's
    prediction of it, the sigmoid of the extra output."""

    def __init__(self, hyperul, large_weight=1.0, large_loss="bce"):
        super().__init__()
        self.hyperul, self.large_weight, self.large_loss = hyperul, large_weight, LARGE_UNC_LOSSES[large_loss]

    def forward(self, logits, dist, large_pred, target, large_unc=None):
        loss, stats = self.hyperul(logits, dist, target)
        if large_unc is None:
            return loss, stats
        valid = target != self.hyperul.ignore_index
        n = valid.sum().clamp_min(1)
        stats["large_unc"] = (self.large_loss(large_pred, large_unc) * valid).sum() / n
        stats["large_unc_mae"] = ((large_pred.detach().sigmoid() - large_unc).abs() * valid).sum() / n
        return loss + self.large_weight * stats["large_unc"], stats
