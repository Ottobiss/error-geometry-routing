"""Загрузка Cityscapes в официальной структуре: пары кадр — разметка, перевод в trainIds."""
from pathlib import Path

import numpy as np
from cityscapesscripts.helpers.labels import labels as CS_LABELS
from PIL import Image

from common.components import IGNORE

# labelId (0..33, -1) → trainId (0..18), всё остальное → ignore
ID_TO_TRAIN = np.full(256, IGNORE, dtype=np.uint8)
for lab in CS_LABELS:
    if 0 <= lab.id < 256 and 0 <= lab.trainId < 19:
        ID_TO_TRAIN[lab.id] = lab.trainId

SIZE = (1024, 512)  # (W, H) — рабочее разрешение


def cityscapes_pairs(root="data/cityscapes", split="val"):
    """Список (image_id, путь к кадру, путь к labelIds), отсортированный по id."""
    root = Path(root)
    out = []
    for img in sorted((root / "leftImg8bit" / split).rglob("*_leftImg8bit.png")):
        image_id = img.name[: -len("_leftImg8bit.png")]
        gt = root / "gtFine" / split / img.parent.name / f"{image_id}_gtFine_labelIds.png"
        out.append((image_id, img, gt))
    return out


def load_pair(img_path, gt_path, size=SIZE):
    """Кадр в исходном разрешении (H, W, 3) uint8 и GT в trainIds на рабочем разрешении.

    Кадр не уменьшается: каждая модель приводит его к своему входному размеру сама.
    """
    img = Image.open(img_path).convert("RGB")
    gt = Image.open(gt_path).resize(size, Image.NEAREST)
    return np.asarray(img), ID_TO_TRAIN[np.asarray(gt)]


def resize_image(img, size):
    """Уменьшение кадра (H, W, 3) uint8 до size = (W, H), билинейно."""
    if (img.shape[1], img.shape[0]) == tuple(size):
        return img
    return np.asarray(Image.fromarray(img).resize(size, Image.BILINEAR))
