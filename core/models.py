"""SegFormer из HuggingFace как nn.Module: нормализация, прогон, логиты на рабочем разрешении,
хуки внутренних признаков.

Лёгкая B0 получает кадр, уменьшенный до рабочего разрешения датасета (элемент «small»), тяжёлые
B2 / B5 — кадр в исходном разрешении («image»), на котором они обучены, если у датасета
heavy_full = True (Cityscapes), иначе тоже уменьшенный (ACDC).
"""
import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerForSemanticSegmentation

CHECKPOINTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
FULL_RES = {"b2", "b5"}


def upsample_logits(logits, size):
    """Логиты (19, h, w) → (19, H, W) на рабочем разрешении size = (W, H), билинейно."""
    x = torch.as_tensor(logits, device="cuda").float()[None]
    return F.interpolate(x, size=(size[1], size[0]), mode="bilinear", align_corners=False)[0]


class SegModel(nn.Module):
    """SegFormer с предобработкой. forward(image uint8 (B, 3, H, W), size) → логиты (B, 19, h, w)
    на рабочем разрешении size = (W, h). У лёгкой модели после каждого прохода в self.features —
    внутренние признаки (у тяжёлых хуков нет, чтобы не держать их тензоры в памяти GPU):

    C      — вход decode_head.classifier: (1, 256, h/4, w/4) (после linear_fuse → BN → ReLU;
             dropout в eval — тождество);
    S4     — выход стадии 4 энкодера (вход decode_head): (n, 256) токенов сетки 1/32;
    Q/K/V  — выходы q_proj / k_proj / v_proj последнего блока стадии 4: (n, 256), 8 голов × 32 склеены;
    logits — выход decode_head: (19, h/4, w/4) до подъёма разрешения.
    """

    def __init__(self, name):
        super().__init__()
        self.full_res = name in FULL_RES
        self.net = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINTS[name])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)
        self.features = {}
        if self.full_res:
            return
        att =self.net.segformer.stages[3].blocks[-1].attention
        for key, mod in (("Q", att.q_proj), ("K", att.k_proj), ("V", att.v_proj)):
            mod.register_forward_hook(lambda m, i, o, k=key: self.features.__setitem__(k, o[0]))
        head = self.net.decode_head
        head.register_forward_pre_hook(lambda m, a: self.features.__setitem__("S4", a[0][-1][0].flatten(1).T))
        head.classifier.register_forward_pre_hook(lambda m, a: self.features.__setitem__("C", a[0]))
        head.register_forward_hook(lambda m, i, o: self.features.__setitem__("logits", o[0]))

    def forward(self, image, size):
        x = (image.float() - self.mean) / self.std
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=(size[1], size[0]), mode="bilinear", align_corners=False)

    @torch.inference_mode()
    def probs(self, sample, size, full=True):
        """Элемент датасета → softmax (19, H, W) на GPU на рабочем разрешении size.
        full=False — тяжёлая модель тоже получает уменьшенный кадр (SegDataset.heavy_full)."""
        image = sample["image" if self.full_res and full else "small"]
        return self(image[None].to(self.mean.device), size).softmax(1)[0]

    def tokens(self, key):
        """Признак как строки (n, 256) float32."""
        x = self.features[key]
        if key == "C":
            x = x[0].flatten(1).T
        return x.float().contiguous()

    def c_map(self, size, normalize=False):
        """C, поднятый билинейно до рабочего разрешения: (256, H·W); normalize — L2 по каналам."""
        up = F.interpolate(self.features["C"].float(), size=(size[1], size[0]), mode="bilinear",
                           align_corners=False)[0]
        if normalize:
            up = F.normalize(up, dim=0)
        return up.reshape(up.shape[0], -1)

    def neg_lse(self, size):
        """−logsumexp логитов, поднятых до рабочего разрешения: (H·W,)."""
        return -upsample_logits(self.features["logits"], size).logsumexp(0).reshape(-1)


def load_model(name, device="cuda"):
    return SegModel(name).to(device).eval()
