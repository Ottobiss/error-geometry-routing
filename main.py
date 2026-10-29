"""Command line; all settings come from config.yaml (defaults in pipeline.Config).

  python main.py fit                         fit features and the flag threshold on fit_set
  python main.py evaluate                    mIoU, spent budget and time on eval_sets
  python main.py run "frames/*.png" masks/   segment frames as one stream: masks + levels.csv
  ... --config other.yaml                    another config
"""
import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from loaders import IGNORE, get, unpack
from pipeline import Clock, Config, Pipeline
from router import LEVELS


def fit(cfg):
    pipe = Pipeline(cfg)
    pipe.fit(get(cfg.fit_set, cfg.data))
    pipe.save()
    print(f"saved {cfg.state}, flag threshold {pipe.router.threshold}")


def run(cfg, images, out):
    pipe = Pipeline.load(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "levels.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["name", "level", *pipe.features, "ms"])
        for img, _, name in DataLoader(get(images), collate_fn=unpack, num_workers=2):
            r = pipe(img)
            Image.fromarray(np.asarray(r.pred)).save(out / f"{name}.png")
            log.writerow([name, r.level, *(f"{s:.5f}" for s in r.scores.values()), f"{sum(r.times.values()):.1f}"])


def confusion(pred, gt, n):
    pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
    m = gt != IGNORE
    return np.bincount(n * gt[m].astype(np.int64) + pred[m], minlength=n * n).reshape(n, n)


def miou(conf):
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf) + 1e-9)
    return 100 * iou[conf.sum(1) > 0].mean()


def summary(frames):
    """mIoU of light / pipeline / heavy, % of the light -> heavy gap closed, cost, levels and ms."""
    light, pipe, heavy = [miou(sum(f["conf"][j] for f in frames)) for j in range(3)]
    levels = Counter(f["level"] for f in frames)
    return {"frames": len(frames), "miou_light": round(light, 2), "miou_pipeline": round(pipe, 2),
            "miou_heavy": round(heavy, 2), "gap_closed": round(100 * (pipe - light) / (heavy - light), 1),
            "cost": round(float(np.mean([LEVELS[f["level"]].cost for f in frames])), 3),
            "levels": {l: round(k / len(frames), 3) for l, k in levels.items()},
            "ms_pipeline": round(float(np.mean([f["ms"] for f in frames])), 1),
            "ms_heavy": round(float(np.mean([f["ms_heavy"] for f in frames])), 1)}


@torch.inference_mode()
def evaluate(cfg):
    """The labelled eval_sets as one stream; light and heavy alone are run after each timed pipeline call."""
    pipe = Pipeline.load(cfg)
    sets = [get(s, cfg.data) for s in cfg.eval_sets]
    if cfg.limit:
        sets = [Subset(s, range(min(cfg.limit, len(s)))) for s in sets]
    stream = ConcatDataset(sets)
    owner = np.repeat(cfg.eval_sets, [len(s) for s in sets])
    order = np.arange(len(stream)) if cfg.shuffle is None else np.random.default_rng(cfg.shuffle).permutation(len(stream))
    segment = lambda model, img: F.interpolate(model(img[None]).logits, img.shape[-2:], mode="bilinear")[0].argmax(0).cpu()

    for i in order[:cfg.warmup]:  # warm up every path, then start the stream from scratch
        img = stream[i][0]
        pipe(img), pipe.heavy(img[None], 0.5), pipe.heavy(img[None], 0.75), segment(pipe.heavy, img)
    pipe.router.reset()

    frames, n = [], pipe.light.num_classes
    for i, (img, gt, _) in zip(order, tqdm(DataLoader(Subset(stream, order), collate_fn=unpack, num_workers=2),
                                           total=len(order))):
        r = pipe(img)
        light = segment(pipe.light, img)
        clock = Clock()
        heavy = segment(pipe.heavy, img)
        clock("heavy")
        frames.append({"set": owner[i], "level": r.level, "ms": sum(r.times.values()), "ms_heavy": clock.times["heavy"],
                       "conf": [confusion(p, gt, n) for p in (light, r.pred, heavy)]})

    report = {s: summary([f for f in frames if f["set"] == s]) for s in cfg.eval_sets}
    report["pool"] = summary(frames)
    print(f"\n{'set':16} {'frames':>6} {'light':>6} {'pipe':>6} {'heavy':>6} {'closed':>7} {'cost':>5} "
          f"{'ms pipe':>8} {'ms heavy':>8}  levels")
    for s, x in report.items():
        print(f"{s:16} {x['frames']:6} {x['miou_light']:6.2f} {x['miou_pipeline']:6.2f} {x['miou_heavy']:6.2f} "
              f"{x['gap_closed']:7.1f} {x['cost']:5.2f} {x['ms_pipeline']:8.1f} {x['ms_heavy']:8.1f}  "
              + " ".join(f"{l} {v:.0%}" for l, v in x["levels"].items()))
    report["config"] = asdict(cfg)
    Path(cfg.results).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.results).write_text(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["fit", "evaluate", "run"])
    ap.add_argument("paths", nargs="*", help='run: glob of frames and the output folder, e.g. "frames/*.png" masks/')
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    if args.command == "run":
        if len(args.paths) != 2:
            ap.error('run needs two paths: "frames/*.png" masks/')
        run(cfg, *args.paths)
    else:
        {"fit": fit, "evaluate": evaluate}[args.command](cfg)


if __name__ == "__main__":
    main()
