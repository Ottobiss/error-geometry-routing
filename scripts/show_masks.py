"""Картинка для глаз: кадр, GT, маски B0 и B2 и карта «кто прав».

    python scripts/show_masks.py --index 0
Сохраняет results/masks_<image_id>.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from cityscapesscripts.helpers.labels import trainId2label
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.components import IGNORE  # noqa: E402
from common.data import SIZE, cityscapes_pairs, load_pair, resize_image  # noqa: E402
from common.models import load_model, predict_probs  # noqa: E402

PALETTE = np.zeros((256, 3), dtype=np.uint8)
for t in range(19):
    PALETTE[t] = trainId2label[t].color


def colorize(mask):
    return PALETTE[mask]


def who_is_right(gt, light, heavy):
    """Зелёный — права только тяжёлая, красный — права только лёгкая,
    серый — обе правы, чёрный — обе неправы или ignore."""
    out = np.zeros(gt.shape + (3,), dtype=np.uint8)
    valid = gt != IGNORE
    ok_l, ok_h = (light == gt) & valid, (heavy == gt) & valid
    out[ok_l & ok_h] = (90, 90, 90)
    out[ok_h & ~ok_l] = (0, 220, 0)
    out[ok_l & ~ok_h] = (230, 0, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    image_id, img_path, gt_path = cityscapes_pairs()[args.index]
    img, gt = load_pair(img_path, gt_path)
    preds = {m: predict_probs(load_model(m), img).argmax(0).cpu().numpy().astype(np.uint8)
             for m in ("b0", "b2")}
    valid = gt != IGNORE
    acc = {m: 100 * (p[valid] == gt[valid]).mean() for m, p in preds.items()}

    panels = [
        ("Кадр", resize_image(img, SIZE)),
        ("GT", colorize(gt)),
        (f"B0  (pixel acc {acc['b0']:.1f}%)", colorize(preds["b0"])),
        (f"B2  (pixel acc {acc['b2']:.1f}%)", colorize(preds["b2"])),
        ("Кто прав: зелёный — только B2, красный — только B0", who_is_right(gt, preds["b0"], preds["b2"])),
    ]
    w, h = SIZE
    pad = 28
    canvas = Image.new("RGB", (2 * w, 3 * (h + pad)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("arial.ttf", 18)  # шрифт по умолчанию без кириллицы
    for k, (title, arr) in enumerate(panels):
        x, y = (k % 2) * w, (k // 2) * (h + pad)
        draw.text((x + 6, y + 4), title, fill="black", font=font)
        canvas.paste(Image.fromarray(arr), (x, y + pad))
    out = Path("results") / f"masks_{image_id}.png"
    out.parent.mkdir(exist_ok=True)
    canvas.save(out)
    print(out, acc)


if __name__ == "__main__":
    main()
