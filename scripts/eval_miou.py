"""Санити-проверка: mIoU моделей на Cityscapes val при рабочем разрешении.

    python scripts/eval_miou.py --models b0 b2
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.components import IGNORE  # noqa: E402
from common.data import cityscapes_pairs, load_pair  # noqa: E402
from common.models import load_model, predict_probs  # noqa: E402

NC = 19


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["b0", "b2"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pairs = cityscapes_pairs()[: args.limit]
    models = {m: load_model(m) for m in args.models}
    conf = {m: torch.zeros(NC * NC, dtype=torch.int64, device="cuda") for m in models}
    t0 = time.time()
    for k, (_, img_path, gt_path) in enumerate(pairs, 1):
        img, gt = load_pair(img_path, gt_path)
        gt_t = torch.from_numpy(gt.astype(np.int64)).cuda()
        valid = gt_t != IGNORE
        for m, model in models.items():
            pred = predict_probs(model, img).argmax(0)
            conf[m] += torch.bincount(gt_t[valid] * NC + pred[valid], minlength=NC * NC)
        if k % 100 == 0:
            print(f"{k}/{len(pairs)}  {time.time() - t0:.0f} с")
    for m in models:
        c = conf[m].view(NC, NC).double().cpu()
        iou = c.diag() / (c.sum(0) + c.sum(1) - c.diag())
        print(f"{m}: mIoU {100 * iou.nanmean():.2f}  pixel acc {100 * c.diag().sum() / c.sum():.2f}")


if __name__ == "__main__":
    main()
