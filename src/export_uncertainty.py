"""Stage 2: the trained large model predicts its hyperbolic uncertainty (Eq. 15) for every image of a split (val by
default: images it has not seen). The maps are saved once as 8-bit PNGs, <out>/<split>/<city>/<image name>, and
become fixed targets for the small model (data.large_uncertainty, data.large_uncertainty_split). <out>/summary.json
keeps their histogram, <out>/export.log the console output.

    python -m src.export_uncertainty configs/export_uncertainty.yaml root=/data/cityscapes
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import CLASSES, Cityscapes
from src.hyperbolic import hyperbolic_uncertainty
from src.logs import keep_log
from src.model import load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. root=/data/cityscapes split=val")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    keep_log(out / "export.log")
    OmegaConf.save(cfg, out / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.checkpoint, len(CLASSES)).to(device).eval()
    dataset = Cityscapes(cfg.root, cfg.split)
    loader = DataLoader(dataset, 1, num_workers=cfg.workers)

    histogram, total = torch.zeros(20, dtype=torch.float64), 0.0
    for path, (image, _) in zip(dataset.images, tqdm(loader)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            _, dist, _ = model(image.to(device))
        unc = hyperbolic_uncertainty(dist)[0, ::cfg.stride, ::cfg.stride]
        target = Path(out, *path.parts[-3:])  # <split>/<city>/<image name>
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((unc * 255).round().byte().cpu().numpy()).save(target)
        histogram += torch.histc(unc.float(), len(histogram), 0, 1).cpu()
        total += unc.sum().item()
    summary = {"split": cfg.split, "images": len(dataset), "mean": total / max(histogram.sum().item(), 1),
               "histogram": histogram.tolist()}  # pixel counts in equal bins over [0, 1]
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(dataset)} maps of {cfg.split} in {out}, mean uncertainty {summary['mean']:.4f}")


if __name__ == "__main__":
    main()
