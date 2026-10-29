import torch.nn.functional as F
from torch import nn
from torchvision.transforms.v2 import functional as TF
from transformers import SegformerForSemanticSegmentation

from models.common import Out, capture


class SegFormer(nn.Module):
    """SegFormer fine-tuned on Cityscapes (Hugging Face): size "b0", "b2" or "b5"."""
    CHECKPOINTS = {"b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
                   "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                   "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"}
    TAPS = {"decoder": "net.decode_head.classifier"}  # fused decoder features, 256 channels at 1/4
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    num_classes = 19

    def __init__(self, size):
        super().__init__()
        self.net = SegformerForSemanticSegmentation.from_pretrained(self.CHECKPOINTS[size])

    def forward(self, images, scale=1.0, taps=()):
        p = next(self.parameters())
        x = images.to(p.device)
        if scale != 1.0:
            x = F.interpolate(x, scale_factor=scale, mode="bilinear", antialias=True)
        with capture(self, taps, self.TAPS) as captured:
            logits = self.net(pixel_values=TF.normalize(x, self.MEAN, self.STD).to(p.dtype)).logits.float()
        return Out(logits, captured)
