import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.hyperbolic import MANIFOLDS, HyperbolicMLR

VARIANTS = {  # SegformerConfig fields that differ between the MiT encoders
    "b0": dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256),
    "b2": dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768),
}


class HyperSegFormer(nn.Module):
    """SegFormer -> Euclidean embeddings v -> exp map into a model of hyperbolic space -> hyperbolic MLR (Fig. 2).

    manifold: poincare | lorentz (see src.hyperbolic). With large_head the last 1x1 conv of the decoder also outputs
    a logit of the large model's hyperbolic uncertainty (the small model of the routing pipeline).
    """

    def __init__(self, num_classes, embed_dim=64, c=0.7, pretrained=None, variant="b0", manifold="poincare",
                 large_head=False):
        super().__init__()
        self.embed_dim, self.large_head = embed_dim, large_head
        self.manifold = MANIFOLDS[manifold](c)
        config = SegformerConfig(num_labels=embed_dim + int(large_head), **VARIANTS[variant])
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(pretrained, config=config)
        else:
            self.net = SegformerForSemanticSegmentation(config)
        self.mlr = HyperbolicMLR(embed_dim, num_classes, self.manifold)

    def forward(self, x):
        """-> class logits (B, K, H, W), hyperbolic distances to the origin (B, H, W),
        logits of the large model's uncertainty (B, H, W) or None."""
        out = self.net(pixel_values=x).logits
        with torch.autocast(x.device.type, enabled=False):
            z = self.manifold.expmap0(out[:, :self.embed_dim].float())
            logits, dist = self.mlr(z), self.manifold.dist0(z)
        size = x.shape[-2:]
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        dist = F.interpolate(dist[:, None], size, mode="bilinear", align_corners=False)[:, 0]
        large = None
        if self.large_head:
            large = F.interpolate(out[:, self.embed_dim:].float(), size, mode="bilinear", align_corners=False)[:, 0]
        return logits, dist, large


def build_model(cfg, num_classes, pretrained=True):
    """HyperSegFormer from a training config; data.large_uncertainty adds the large-uncertainty head."""
    m = cfg.model
    return HyperSegFormer(num_classes, m.embed_dim, m.c, m.pretrained if pretrained else None, m.get("variant", "b0"),
                          m.get("manifold", "poincare"), bool(cfg.data.get("large_uncertainty")))


def load_model(path, num_classes):
    """Model from a src.train checkpoint; the architecture comes from its config, nothing is downloaded."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(OmegaConf.create(ckpt["config"]), num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model
