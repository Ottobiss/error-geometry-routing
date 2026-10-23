"""SegFormer-B0 + Hyperbolic Uncertainty Loss on Cityscapes.

    python train.py configs/train.yaml data.root=/data/cityscapes
    tensorboard --logdir runs        # parallel monitoring
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import load_config
from data import CLASSES, IGNORE, Cityscapes, colorize, denormalize, heatmap, make_panel
from hyperbolic import hyperbolic_uncertainty
from loss import HyperUL
from model import HyperSegFormer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. data.root=/data/cityscapes loss.t=2.718")
    p.add_argument("--resume")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_intermixed_args()


class SegMetrics:
    """mIoU, ECE of the hyperbolic confidence 1 - uncertainty, mean uncertainty on correct / wrong pixels."""

    def __init__(self, num_classes, bins=15):
        self.k, self.bins = num_classes, bins
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        self.calibration = torch.zeros(3, bins, dtype=torch.float64)  # pixels, confidence sum, correct sum
        self.uncertainty = torch.zeros(2, 2, dtype=torch.float64)  # (correct, wrong) x (sum, pixels)

    def update(self, pred, target, uncertainty):
        valid = target != IGNORE
        pred, target, uncertainty = pred[valid], target[valid], uncertainty[valid].double()
        self.confusion += torch.bincount(target * self.k + pred, minlength=self.k ** 2).view(self.k, self.k).cpu()

        correct = (pred == target).double()
        confidence = 1 - uncertainty
        bins = (confidence * self.bins).long().clamp(max=self.bins - 1)
        for row, weights in enumerate((None, confidence, correct)):
            self.calibration[row] += torch.bincount(bins, weights, minlength=self.bins).cpu()

        wrong = (pred != target).long()
        self.uncertainty[:, 0] += torch.bincount(wrong, uncertainty, minlength=2).cpu()
        self.uncertainty[:, 1] += torch.bincount(wrong, minlength=2).cpu()

    def compute(self):
        tp = self.confusion.diag()
        iou = tp / (self.confusion.sum(0) + self.confusion.sum(1) - tp)
        pixels, confidence, correct = self.calibration
        uncertainty = self.uncertainty[:, 0] / self.uncertainty[:, 1].clamp_min(1)
        return {
            "val/mIoU": iou.nanmean().item(),
            "val/ECE": ((correct - confidence).abs().sum() / pixels.sum()).item(),
            "val/uncertainty_correct": uncertainty[0].item(),
            "val/uncertainty_wrong": uncertainty[1].item(),
            **{f"val_iou/{name}": value for name, value in zip(CLASSES, iou.tolist())},
        }


@torch.no_grad()
def evaluate(model, loader, writer, step, amp=False, images=3):
    model.eval()
    device = next(model.parameters()).device
    metrics = SegMetrics(len(CLASSES))
    for i, (image, label) in enumerate(tqdm(loader, desc="val", leave=False)):
        image, label = image.to(device), label.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, dist = model(image)
        pred, uncertainty = logits.argmax(1), hyperbolic_uncertainty(dist)
        metrics.update(pred, label, uncertainty)
        if i < images:
            tiles = [denormalize(image[0]), colorize(label[0]), colorize(pred[0]), heatmap(uncertainty[0])]
            writer.add_image(f"val/sample_{i}", make_panel(tiles, cols=2), step)
    model.train()

    results = metrics.compute()
    for tag, value in results.items():
        writer.add_scalar(tag, value, step)
    writer.flush()
    tqdm.write(f"step {step}: mIoU {results['val/mIoU']:.4f}  ECE {results['val/ECE']:.4f}  uncertainty correct/wrong "
               f"{results['val/uncertainty_correct']:.3f}/{results['val/uncertainty_wrong']:.3f}")
    return results["val/mIoU"]


def endless(loader):
    while True:
        yield from loader


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = Cityscapes(cfg.data.root, "train", cfg.data.crop, cfg.data.base_size, cfg.data.scale_range)
    train_loader = DataLoader(train_set, cfg.data.batch_size, shuffle=True, num_workers=cfg.data.workers,
                              pin_memory=True, drop_last=True, persistent_workers=cfg.data.workers > 0)
    val_loader = DataLoader(Cityscapes(cfg.data.root, "val"), 1, num_workers=cfg.data.workers, pin_memory=True)

    model = HyperSegFormer(len(CLASSES), cfg.model.embed_dim, cfg.model.c, cfg.model.pretrained).to(device)
    criterion = HyperUL(cfg.loss.t, cfg.loss.hr, IGNORE, cfg.loss.class_weights).to(device)
    decoder = [*model.net.decode_head.parameters(), *model.mlr.parameters()]
    optimizer = torch.optim.AdamW([{"params": model.net.segformer.parameters(), "lr": cfg.train.lr},
                                   {"params": decoder, "lr": cfg.train.lr * cfg.train.head_lr_mult}],
                                  weight_decay=cfg.train.weight_decay)
    warmup, iters, power = cfg.train.warmup, cfg.train.iters, cfg.train.poly_power
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: min(1.0, (it + 1) / max(warmup, 1)) * (1 - it / iters) ** power)

    start, best = 0, 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start, best = ckpt["iter"], ckpt["best"]

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out, flush_secs=30)
    if args.eval_only:
        evaluate(model, val_loader, writer, start, cfg.train.amp, cfg.log.val_images)
        writer.close()
        return
    OmegaConf.save(cfg, out / "config.yaml")
    writer.add_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg)}```")

    batches = endless(train_loader)
    running, tic = defaultdict(float), time.perf_counter()
    for step in tqdm(range(start + 1, iters + 1), initial=start, total=iters, desc="train"):
        image, label = next(batches)
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.amp):
            logits, dist = model(image)
        loss, stats = criterion(logits, dist, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        for key, value in {"loss": loss, **stats}.items():
            running[key] += value.detach()
        if step % cfg.log.every == 0:
            elapsed, tic = time.perf_counter() - tic, time.perf_counter()
            for key, value in running.items():
                writer.add_scalar(f"train/{key}", value.item() / cfg.log.every, step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            writer.add_scalar("train/img_per_s", cfg.log.every * cfg.data.batch_size / elapsed, step)
            if device.type == "cuda":
                writer.add_scalar("train/mem_gb", torch.cuda.max_memory_allocated() / 2 ** 30, step)
            writer.add_histogram("train/dist", dist.detach()[:, ::4, ::4], step)
            running.clear()

        if step % cfg.log.val_every == 0 or step == iters:
            miou = evaluate(model, val_loader, writer, step, cfg.train.amp, cfg.log.val_images)
            ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "iter": step, "best": max(best, miou),
                    "config": OmegaConf.to_container(cfg)}
            torch.save(ckpt, out / "last.pt")
            if miou > best:
                best = miou
                torch.save(ckpt, out / "best.pt")
            tic = time.perf_counter()
    writer.close()


if __name__ == "__main__":
    main()
