import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.hyperbolic import MANIFOLDS

VARIANTS = {  # SegformerConfig fields that differ between the MiT encoders
    "b0": dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256], decoder_hidden_size=256),
    "b2": dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768),
}
CLASSIFIER_SCALE = 0.02  # trained logits (tens) would saturate the ball: a pretrained classifier is scaled down


def patch_mean(x, patch):
    """(B, C, H, W) -> mean over each patch x patch square (smaller at the right and bottom edges), broadcast back."""
    mean = F.avg_pool2d(x, patch, ceil_mode=True)
    return mean.repeat_interleave(patch, 2).repeat_interleave(patch, 3)[..., :x.shape[2], :x.shape[3]]


def add_output(conv):
    """The 1x1 conv with one more output channel (HF init: normal, std 0.02); the others keep their weights."""
    new = nn.Conv2d(conv.in_channels, conv.out_channels + 1, 1)
    with torch.no_grad():
        new.weight.normal_(std=0.02)
        new.bias.zero_()
        new.weight[:-1], new.bias[:-1] = conv.weight, conv.bias
    return new


class HyperSegFormer(nn.Module):
    """SegFormer -> Euclidean embeddings v, the output of its last layer with one channel per class -> exp map into
    a model of hyperbolic space; the coordinates of the point are the class logits (Fig. 2, Eq. 12) and its distance
    to the origin gives the uncertainty (Eq. 15).

    manifold: poincare | lorentz (see src.hyperbolic). With large_head the last 1x1 conv of the decoder also outputs
    a logit of the large model's hyperbolic uncertainty (the small model of the routing pipeline). A trained classifier
    of a segmentation checkpoint is kept, scaled by CLASSIFIER_SCALE: the same predictions, embeddings inside the ball.
    Only the last 1x1 conv of the decoder, which HyperUL acts on, is trained; the rest of SegFormer is fixed.
    Uncertainties are averaged over patch x patch squares of the input. pretrained=None builds the architecture only
    (for load_model).
    """

    def __init__(self, num_classes, c=0.7, pretrained=None, variant="b0", manifold="poincare", large_head=False,
                 patch=1):
        super().__init__()
        self.num_classes, self.large_head, self.patch = num_classes, large_head, patch
        self.manifold = MANIFOLDS[manifold](c)
        config = SegformerConfig(num_labels=num_classes, **VARIANTS[variant])
        if pretrained:
            self.net, info = SegformerForSemanticSegmentation.from_pretrained(
                pretrained, config=config, ignore_mismatched_sizes=True, output_loading_info=True)
            initialized = info["missing_keys"] + [key for key, *_ in info["mismatched_keys"]]
            if "decode_head.classifier.weight" not in initialized:  # loaded: a trained classifier of these classes
                with torch.no_grad():
                    for p in self.net.decode_head.classifier.parameters():
                        p *= CLASSIFIER_SCALE
        else:
            self.net = SegformerForSemanticSegmentation(config)
        if large_head:
            self.net.decode_head.classifier = add_output(self.net.decode_head.classifier)
        self.net.requires_grad_(False)
        self.net.decode_head.classifier.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        """The frozen SegFormer stays in eval mode: fixed BatchNorm statistics, no drop path and dropout."""
        super().train(mode)
        self.net.eval()
        return self

    def forward(self, x):
        """-> class logits (B, K, H, W), hyperbolic distances to the origin (B, H, W),
        logits of the large model's uncertainty (B, H, W) or None; the last two are patch means."""
        out = self.net(pixel_values=x).logits
        with torch.autocast(x.device.type, enabled=False):
            z = self.manifold.expmap0(out[:, :self.num_classes].float())
            logits, dist = self.manifold.coordinates(z), self.manifold.dist0(z)
        size = x.shape[-2:]
        logits = F.interpolate(logits, size, mode="bilinear", align_corners=False)
        dist = patch_mean(F.interpolate(dist[:, None], size, mode="bilinear", align_corners=False), self.patch)[:, 0]
        large = None
        if self.large_head:
            large = F.interpolate(out[:, self.num_classes:].float(), size, mode="bilinear", align_corners=False)
            large = patch_mean(large, self.patch)[:, 0]
        return logits, dist, large


def build_model(cfg, num_classes, pretrained=True):
    """HyperSegFormer from a training config; data.large_uncertainty adds the large-uncertainty head."""
    m = cfg.model
    return HyperSegFormer(num_classes, m.c, m.pretrained if pretrained else None, m.get("variant", "b0"),
                          m.get("manifold", "poincare"), bool(cfg.data.get("large_uncertainty")), m.get("patch", 1))


def load_model(path, num_classes):
    """Model from a src.train checkpoint; the architecture comes from its config, nothing is downloaded."""
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(OmegaConf.create(ckpt["config"]), num_classes, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model
