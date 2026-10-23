"""Stage 2: the trained large model predicts its hyperbolic uncertainty (Eq. 15) for every image of a split. The maps
are saved once as 8-bit PNGs, <out>/<split>/<city>/<image name>, and become fixed targets for the small model
(data.large_uncertainty).

    python -m src.export_uncertainty configs/export_uncertainty.yaml root=/data/cityscapes
"""
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data import CLASSES, Cityscapes
from src.hyperbolic import hyperbolic_uncertainty
from src.model import load_model


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. root=/data/cityscapes split=val")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.checkpoint, len(CLASSES)).to(device).eval()
    dataset = Cityscapes(cfg.root, cfg.split)
    loader = DataLoader(dataset, 1, num_workers=cfg.workers)

    for path, (image, _) in zip(dataset.images, tqdm(loader)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            _, dist, _ = model(image.to(device))
        unc = hyperbolic_uncertainty(dist)[0, ::cfg.stride, ::cfg.stride]
        target = Path(cfg.out, cfg.split, path.parent.name, path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((unc * 255).round().byte().cpu().numpy()).save(target)


if __name__ == "__main__":
    main()
