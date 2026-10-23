"""Cascade inference: the small model runs on every image, the large one only when some segment is routed.

Outputs per image: segmentation map, the small model's own hyperbolic uncertainty, its prediction of the large
model's uncertainty and the mask of segments taken from the large model.

    python -m src.infer_router configs/infer_router.yaml input=/data/cityscapes/leftImg8bit/val
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
from src.model import load_model
from src.routing import Router


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. input=/data/images threshold=0.3")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = Router(load_model(cfg.small, len(CLASSES)), load_model(cfg.large, len(CLASSES))).to(device).eval()

    src, out = Path(cfg.input), Path(cfg.out)
    paths = (sorted(f for f in src.rglob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg"))
             if src.is_dir() else [src])
    outputs = ("pred", "uncertainty", "large_uncertainty", "routed")
    for name in outputs:
        (out / name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out, flush_secs=30)
    normalize = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    for i, path in enumerate(tqdm(paths)):
        image = read_image(str(path), ImageReadMode.RGB)
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            result = router.cascade(normalize(image)[None].to(device), cfg.threshold, cfg.stride)
        pred, own, large, routed = (t.cpu() for t in result)
        latency = (time.perf_counter() - start) * 1000

        maps = (colorize(pred), heatmap(own), heatmap(large), heatmap(routed.float()))
        for name, rgb in zip(outputs, maps):
            save_rgb(rgb, out / name / f"{path.stem}.png")
        writer.add_scalar("infer/latency_ms", latency, i)
        writer.add_scalar("infer/large_launched", float(routed.any()), i)
        writer.add_scalar("infer/routed_pixels", routed.float().mean().item(), i)
        if i % cfg.image_every == 0:
            writer.add_image("infer/sample", make_panel([image, *maps], cols=3), i)
    writer.close()


if __name__ == "__main__":
    main()
