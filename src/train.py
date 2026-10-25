"""HyperUL training of one SegFormer on Cityscapes; with data.large_uncertainty (stage 3) the model also learns
the large model's exported uncertainty.

    python -m src.train configs/train.yaml data.root=/data/cityscapes                    # HyperUL baseline
    python -m src.train configs/train.yaml configs/large.yaml data.root=/data/cityscapes  # stage 1: large model
    python -m src.train configs/train.yaml configs/small.yaml data.root=/data/cityscapes  # stage 3: small model
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

from src.config import load_config
from src.data import CLASSES, IGNORE, Cityscapes, colorize, denormalize, heatmap, make_panel
from src.hyperbolic import hyperbolic_uncertainty
from src.loss import HyperUL, RoutingLoss
from src.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="more YAML files and dotted overrides, e.g. data.root=/data/cityscapes")
    p.add_argument("--resume")
    p.add_argument("--eval-only", action="store_true")
    return p.parse_intermixed_args()


class SegMetrics:
    """mIoU and, for every given uncertainty map, the ECE of 1 - uncertainty and its mean on correct / wrong pixels."""

    def __init__(self, num_classes, bins=15):
        self.k, self.bins = num_classes, bins
        self.confusion = torch.zeros(num_classes, num_classes, dtype=torch.float64)
        self.calibration = defaultdict(lambda: torch.zeros(3, bins, dtype=torch.float64))  # pixels, conf, correct
        self.uncertainty = defaultdict(lambda: torch.zeros(2, 2, dtype=torch.float64))  # (correct, wrong) x (sum, n)

    def update(self, pred, target, **uncertainties):
        valid = target != IGNORE
        pred, target = pred[valid], target[valid]
        self.confusion += torch.bincount(target * self.k + pred, minlength=self.k ** 2).view(self.k, self.k).cpu()
        correct, wrong = (pred == target).double(), (pred != target).long()
        for name, uncertainty in uncertainties.items():
            uncertainty = uncertainty[valid].double()
            confidence = 1 - uncertainty
            bins = (confidence * self.bins).long().clamp(max=self.bins - 1)
            for row, weights in enumerate((None, confidence, correct)):
                self.calibration[name][row] += torch.bincount(bins, weights, minlength=self.bins).cpu()
            self.uncertainty[name][:, 0] += torch.bincount(wrong, uncertainty, minlength=2).cpu()
            self.uncertainty[name][:, 1] += torch.bincount(wrong, minlength=2).cpu()

    def compute(self, prefix="val"):
        tp = self.confusion.diag()
        iou = tp / (self.confusion.sum(0) + self.confusion.sum(1) - tp)
        results = {f"{prefix}/mIoU": iou.nanmean().item()}
        for name, (pixels, confidence, correct) in self.calibration.items():
            uncertainty = self.uncertainty[name][:, 0] / self.uncertainty[name][:, 1].clamp_min(1)
            results[f"{prefix}/{name}_ECE"] = ((correct - confidence).abs().sum() / pixels.sum()).item()
            results[f"{prefix}/{name}_uncertainty_correct"] = uncertainty[0].item()
            results[f"{prefix}/{name}_uncertainty_wrong"] = uncertainty[1].item()
        return results | {f"{prefix}_iou/{name}": value for name, value in zip(CLASSES, iou.tolist())}


@torch.no_grad()
def evaluate(model, loader, writer, step, amp=False, images=3):
    model.eval()
    device = next(model.parameters()).device
    metrics = SegMetrics(len(CLASSES))
    for i, (image, label) in enumerate(tqdm(loader, desc="val", leave=False)):
        image, label = image.to(device), label.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, dist, large = model(image)
        pred, uncertainty = logits.argmax(1), hyperbolic_uncertainty(dist)
        metrics.update(pred, label, hyperbolic=uncertainty)
        if i < images:
            tiles = [denormalize(image[0]), colorize(label[0]), colorize(pred[0]), heatmap(uncertainty[0])]
            if large is not None:
                tiles.append(heatmap(large[0].sigmoid()))  # predicted uncertainty of the large model
            writer.add_image(f"val/sample_{i}", make_panel(tiles, cols=2 if large is None else 3), step)
    model.train()

    results = metrics.compute()
    for tag, value in results.items():
        writer.add_scalar(tag, value, step)
    writer.flush()
    tqdm.write(f"step {step}: mIoU {results['val/mIoU']:.4f}  hyperbolic ECE {results['val/hyperbolic_ECE']:.4f}")
    return results["val/mIoU"]


def endless(loader):
    while True:
        yield from loader


def make_loaders(data):
    train_set = Cityscapes(data.root, "train", data.crop, data.base_size, data.scale_range, data.large_uncertainty)
    train_loader = DataLoader(train_set, data.batch_size, shuffle=True, num_workers=data.workers, pin_memory=True,
                              drop_last=True, persistent_workers=data.workers > 0)
    val_loader = DataLoader(Cityscapes(data.root, "val"), 1, num_workers=data.workers, pin_memory=True)
    return train_loader, val_loader


def param_groups(model, lr, head_lr_mult):
    """Encoder at lr; decoder and hyperbolic MLR at lr * head_lr_mult. Frozen parameters are left out."""
    head = [*model.net.decode_head.parameters(), *model.mlr.parameters()]
    groups = [{"params": [p for p in params if p.requires_grad], "lr": group_lr}
              for params, group_lr in ((model.net.segformer.parameters(), lr), (head, lr * head_lr_mult))]
    return [group for group in groups if group["params"]]


def poly_scheduler(optimizer, train):
    """Linear warmup times polynomial decay over train.iters iterations."""
    warmup, iters, power = train.warmup, train.iters, train.poly_power
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda it: min(1.0, (it + 1) / max(warmup, 1)) * (1 - it / iters) ** power)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(cfg.data)

    model = build_model(cfg, len(CLASSES)).to(device)
    criterion = RoutingLoss(HyperUL(cfg.loss.t, cfg.loss.hr, IGNORE, cfg.loss.class_weights),
                            cfg.loss.large_unc_weight, cfg.loss.large_unc_loss).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.train.lr, cfg.train.head_lr_mult),
                                  weight_decay=cfg.train.weight_decay)
    scheduler = poly_scheduler(optimizer, cfg.train)

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

    iters = cfg.train.iters
    batches = endless(train_loader)
    running, tic = defaultdict(float), time.perf_counter()
    for step in tqdm(range(start + 1, iters + 1), initial=start, total=iters, desc="train"):
        image, label, *large_unc = next(batches)
        image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
        large_unc = large_unc[0].to(device, non_blocking=True) if large_unc else None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.amp):
            logits, dist, large_pred = model(image)
        loss, stats = criterion(logits, dist, large_pred, label, large_unc)
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
