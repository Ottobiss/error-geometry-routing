"""Routing evaluation on Cityscapes val with the trained small and large models: small, large and routed mIoU
for each threshold, calibration of the hyperbolic uncertainties, how often the large model is launched and how
well the routed segments cover the pixels it corrects.

    python -m src.eval_router configs/eval_router.yaml root=/data/cityscapes
    tensorboard --logdir runs        # route/* and routed/* charts: x axis = 100 * threshold
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.data import CLASSES, IGNORE, Cityscapes, colorize, denormalize, heatmap, make_panel
from src.hyperbolic import hyperbolic_uncertainty
from src.model import load_model
from src.routing import Router, segment_gain
from src.train import SegMetrics


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. root=/data/cityscapes 'thresholds=[0.1,0.3]'")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router = Router(load_model(cfg.small, len(CLASSES)), load_model(cfg.large, len(CLASSES))).to(device).eval()
    loader = DataLoader(Cityscapes(cfg.root, "val"), 1, num_workers=cfg.workers)
    Path(cfg.out).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(cfg.out)

    thresholds = list(cfg.thresholds)
    metrics = {name: SegMetrics(len(CLASSES)) for name in ("small", "large")}
    routed = [SegMetrics(len(CLASSES)) for _ in thresholds]
    counts = torch.zeros(len(thresholds), 3)  # images with the large model launched, routed pixels, fixed pixels
    valid_px = fixable_px = unc_err = 0.0
    fixable_sample, score_sample = [], []

    for i, (image, label) in enumerate(tqdm(loader)):
        image, target = image.to(device), label[0].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            (logits_s, dist_s, large_pred), (logits_l, dist_l, _) = router(image)
        pred_s, pred_l = logits_s[0].argmax(0), logits_l[0].argmax(0)
        unc_s, unc_l = hyperbolic_uncertainty(dist_s)[0], hyperbolic_uncertainty(dist_l)[0]
        unc_l_pred = large_pred[0].sigmoid()
        score = segment_gain(pred_s, unc_s - unc_l_pred, cfg.stride)
        valid = target != IGNORE
        fixable = valid & (pred_s != target) & (pred_l == target)  # pixels the large model corrects

        metrics["small"].update(pred_s, target, hyperbolic=unc_s)
        metrics["large"].update(pred_l, target, hyperbolic=unc_l)
        for j, threshold in enumerate(thresholds):
            mask = score > threshold
            routed[j].update(torch.where(mask, pred_l, pred_s), target)
            counts[j] += torch.stack([mask.any(), (mask & valid).sum(), (mask & fixable).sum()]).cpu()
        valid_px += valid.sum().item()
        fixable_px += fixable.sum().item()
        unc_err += (unc_l_pred - unc_l)[valid].abs().sum().item()
        fixable_sample.append(fixable[valid][::64].cpu())
        score_sample.append(score[valid][::64].cpu())
        if i < cfg.images:
            tiles = [denormalize(image[0]), colorize(target), colorize(pred_s), colorize(pred_l),
                     heatmap(unc_s), heatmap(unc_l_pred), heatmap(unc_l), heatmap(score.clamp(0, 1))]
            writer.add_image(f"eval/sample_{i}", make_panel(tiles, cols=4), 0)

    results = metrics["small"].compute("small") | metrics["large"].compute("large") | {
        "small/large_unc_mae": unc_err / valid_px,
        "route/fixable_pixels": fixable_px / valid_px,
    }
    for tag, value in results.items():
        writer.add_scalar(tag, value, 0)
    writer.add_pr_curve("route/fixable", torch.cat(fixable_sample), (torch.cat(score_sample) + 1) / 2, 0)
    print(f"mIoU small {results['small/mIoU']:.4f}, large {results['large/mIoU']:.4f} | hyperbolic ECE small "
          f"{results['small/hyperbolic_ECE']:.4f}, large {results['large/hyperbolic_ECE']:.4f} | large uncertainty "
          f"MAE {results['small/large_unc_mae']:.4f} | fixable pixels {results['route/fixable_pixels']:.3f}")

    print("threshold  launch_rate  routed_px  fix_precision  fix_recall  routed_mIoU")
    rows = []
    for threshold, routed_metrics, (launched, routed_px, fixed) in zip(thresholds, routed, counts.tolist()):
        row = routed_metrics.compute("routed") | {
            "route/launch_rate": launched / len(loader),
            "route/routed_pixels": routed_px / valid_px,
            "route/fix_precision": fixed / max(routed_px, 1),
            "route/fix_recall": fixed / max(fixable_px, 1),
        }
        for tag, value in row.items():
            writer.add_scalar(tag, value, round(threshold * 100))
        print(f"{threshold:9.2f}  {row['route/launch_rate']:11.2f}  {row['route/routed_pixels']:9.3f}  "
              f"{row['route/fix_precision']:13.2f}  {row['route/fix_recall']:10.2f}  {row['routed/mIoU']:11.4f}")
        rows.append({"threshold": threshold} | row)
    writer.close()
    with open(Path(cfg.out, "results.json"), "w") as f:  # for scripts, e.g. src.sweep
        json.dump({"summary": results, "thresholds": rows}, f, indent=1)


if __name__ == "__main__":
    main()
