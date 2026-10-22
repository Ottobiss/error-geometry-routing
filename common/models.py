"""Загрузка SegFormer из HuggingFace и предсказание softmax на рабочем разрешении."""
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from common.data import SIZE, resize_image

CHECKPOINTS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
# Входное разрешение (W, H): B0 обучена на 512×1024, B2/B5 — на полном кадре.
INPUT_SIZE = {"b0": (1024, 512), "b2": (2048, 1024), "b5": (2048, 1024)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def load_model(name, device="cuda"):
    model = SegformerForSemanticSegmentation.from_pretrained(CHECKPOINTS[name]).to(device).eval()
    model.input_size = INPUT_SIZE[name]
    return model


@torch.inference_mode()
def predict_probs(model, img, out_size=SIZE):
    """img (H, W, 3) uint8 → softmax (19, H', W') float32 на GPU.

    Кадр приводится к входному размеру модели, логиты — к рабочему out_size = (W, H).
    """
    device = next(model.parameters()).device
    img = resize_image(img, model.input_size)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float()
    x = ((x - MEAN) / STD).to(device)
    logits = model(pixel_values=x).logits
    logits = F.interpolate(logits, size=(out_size[1], out_size[0]), mode="bilinear",
                           align_corners=False)
    return logits.softmax(1)[0]
