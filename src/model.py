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


def patch_mean(x, patch):
    """(B, C, H, W) -> mean over each patch x patch square (smaller at the right and bottom edges), broadcast back."""
    mean = F.avg_pool2d(x, patch, ceil_mode=True)
    return mean.repeat_interleave(patch, 2).repeat_interleave(patch, 3)[..., :x.shape[2], :x.shape[3]]


class HyperSegFormer(nn.Module):
    """SegFormer -> Euclidean embeddings v -> exp map into a model of hyperbolic space -> hyperbolic MLR (Fig. 2).

    manifold: poincare | lorentz (see src.hyperbolic). With large_head the last 1x1 conv of the decoder also outputs
    a logit of the large model's hyperbolic uncertainty (the small model of the routing pipeline).
    freeze: only the layers HyperUL acts on are trained, the last 1x1 conv of the decoder and the hyperbolic MLR;
    the rest of the pretrained SegFormer is fixed. Uncertainties are averaged over patch x patch squares of the input.
    """

    def __init__(self, num_classes, embed_dim=64, c=0.7, pretrained=None, variant="b0", manifold="poincare",
                 large_head=False, freeze=False, patch=1):
        super().__init__()
        self.embed_dim, self.large_head, self.frozen, self.patch = embed_dim, large_head, freeze, patch
        self.manifold = MANIFOLDS[manifold](c)
        config = SegformerConfig(num_labels=embed_dim + int(large_head), **VARIANTS[variant])
        if pretrained:  # the classifier of a segmentation checkpoint does not fit and is initialized anew
            self.net = SegformerForSemanticSegmentation.from_pretrained(pretrained, config=config,
                                                                        ignore_mismatched_sizes=True)
        else:
            self.net = SegformerForSemanticSegmentation(config)
        self.mlr = HyperbolicMLR(embed_dim, num_classes, self.manifold)
        if freeze:
            self.net.requires_grad_(False)
            self.net.decode_head.classifier.requires_grad_(True)
        self.train()  # from_pretrained leaves the network in eval mode

    def train(self, mode=True):
        """A frozen SegFormer stays in eval mode: fixed BatchNorm statistics, no drop path and dropout."""
        super().train(mode)
        if self.frozen:
            self.net.eval()
        return self

    def forward(self, x):
        """-> class logits (B, K, H, W), hyperbolic distances to the origin (B, H, W),
        logits of the large model's uncertainty (B, H, W) or None; the last two are patch means."""
        out = self.net(pixel_values=x).logits
        with torch.autocast(x.device.type, enabled=False):
            z = self.manifold.expmap0(out[:, :self.embed_dim].float())
            logits, dist = self.mlr(z), self.manifold.dist0(z)
        size = x.shape[-2:]
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        dist = patch_mean(F.interpolate(dist[:, None], size, mode="bilinear", align_corners=False), self.patch)[:, 0]
        large = None
        if self.large_head:
            large = F.interpolate(out[:, self.embed_dim:].float(), size, mode="bilinear", align_corners=False)
            large = patch_mean(large, self.patch)[:, 0]
        return logits, dist, large


def build_model(cfg, num_classes, pretrained=True):
    """HyperSegFormer from a training config; data.large_uncertainty adds the large-uncertainty head."""
    m = cfg.model
    return HyperSegFormer(num_classes, m.embed_dim, m.c, m.pretrained if pretrained else None, m.get("variant", "b0"),
                          m.get("manifold", "poincare"), bool(cfg.data.get("large_uncertainty")),
                          m.get("freeze", False), m.get("patch", 1))


def load_model(path, num_classes):
    """Model from a src.train checkpoint; the architecture comes from its config, nothing is downloaded."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(OmegaConf.create(ckpt["config"]), num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model
