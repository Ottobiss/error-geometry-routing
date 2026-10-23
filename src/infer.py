"""Segmentation + hyperbolic uncertainty maps for an image or a folder of images.

    python -m src.infer configs/infer.yaml input=/data/cityscapes/leftImg8bit/val
    tensorboard --logdir runs        # parallel monitoring
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2
from tqdm import tqdm

from src.config import load_config
from src.data import CLASSES, MEAN, STD, colorize, heatmap, make_panel, save_rgb
from src.hyperbolic import hyperbolic_uncertainty
from src.model import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. input=/data/images amp=true")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.checkpoint, len(CLASSES)).to(device).eval()

    src, out = Path(cfg.input), Path(cfg.out)
    paths = (sorted(f for f in src.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
             if src.is_dir() else [src])
    for name in ("pred", "uncertainty"):
        (out / name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out, flush_secs=30)
    normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    for i, path in enumerate(tqdm(paths)):
        image = read_image(str(path), ImageReadMode.RGB)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            logits, dist, large = model(normalize(image)[None].to(device))
            pred, uncertainty = logits.argmax(1)[0].cpu(), hyperbolic_uncertainty(dist)[0].cpu()
            extra = [] if large is None else [large[0].sigmoid().cpu()]  # predicted uncertainty of the large model
        latency = (time.perf_counter() - start) * 1000

        save_rgb(colorize(pred), out / "pred" / f"{path.stem}.png")
        save_rgb(heatmap(uncertainty), out / "uncertainty" / f"{path.stem}.png")
        writer.add_scalar("infer/latency_ms", latency, i)
        writer.add_scalar("infer/uncertainty", uncertainty.mean().item(), i)
        if i % cfg.image_every == 0:
            tiles = [image, colorize(pred), heatmap(uncertainty), *map(heatmap, extra)]
            writer.add_image("infer/sample", make_panel(tiles, cols=3), i)
    writer.close()


if __name__ == "__main__":
    main()
