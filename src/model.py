import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.hyperbolic import HyperbolicMLR, dist0, expmap0


class HyperSegFormer(nn.Module):
    """SegFormer-B0 -> Euclidean embeddings v -> exp map to the Poincaré ball -> hyperbolic MLR (Fig. 2)."""

    def __init__(self, num_classes, embed_dim=64, c=0.7, pretrained="nvidia/mit-b0"):
        super().__init__()
        self.c = c
        # the last 1x1 conv of the decoder outputs embed_dim channels: the Euclidean embeddings v
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                pretrained, num_labels=embed_dim, ignore_mismatched_sizes=True)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=embed_dim))
        self.mlr = HyperbolicMLR(embed_dim, num_classes, c)

    def forward(self, x):
        """-> class logits (B, K, H, W), hyperbolic distances to the origin (B, H, W)."""
        v = self.net(pixel_values=x).logits
        with torch.autocast(x.device.type, enabled=False):
            z = expmap0(v.float(), self.c)
            logits, dist = self.mlr(z), dist0(z, self.c)
        size = x.shape[-2:]
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        dist = F.interpolate(dist[:, None], size, mode="bilinear", align_corners=False)[:, 0]
        return logits, dist
