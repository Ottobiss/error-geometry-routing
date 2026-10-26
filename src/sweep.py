"""Random search over the main hyperparameters of one stage, on all visible GPUs.

    python -m src.sweep configs/sweep.yaml stage=large root=/data/cityscapes  # stage 1 (B2), scored on val
    python -m src.sweep configs/sweep.yaml stage=small root=/data/cityscapes  # stage 3 (B0), scored by eval_router

Trial 0 keeps the defaults of train.yaml, the others sample configs/sweep.yaml:space. The per-GPU batch is the
largest that fits: training steps and a full-resolution validation forward are probed in separate processes.
Each trial trains in <out>/<stage>/<trial> (src.train, under torchrun on several GPUs) and is appended to
<out>/<stage>/trials.jsonl. A rerun skips recorded trials and resumes an interrupted one from its last.pt.
"""
import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from src.config import load_config
from src.data import CLASSES, IGNORE, Cityscapes
from src.loss import HyperUL, RoutingLoss
from src.model import build_model
from src.train import param_groups

STAGES = {"large": ["configs/train.yaml", "configs/large.yaml"], "small": ["configs/train.yaml", "configs/small.yaml"]}
SHOWN = {"val/mIoU": "mIoU", "val/hyperbolic_ECE": "ECE", "gap": "gap", "small/large_unc_mae": "unc_mae",
         "routed/mIoU": "routed", "route/launch_rate": "launch", "threshold": "thr"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. stage=small root=/data/cityscapes")
    p.add_argument("--probe", type=int, help=argparse.SUPPRESS)  # internal: config and overrides are src.train's
    return p.parse_intermixed_args()


def probe_step(cfg, batch):
    """Prints the peak reserved GPU memory and the GPU size (GiB) of two training steps at this per-GPU batch
    and a validation forward at full resolution; zeros on CPU."""
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, len(CLASSES), pretrained=False).to(device)
    criterion = RoutingLoss(HyperUL(cfg.loss.t, cfg.loss.hr, IGNORE, cfg.loss.class_weights),
                            cfg.loss.large_unc_weight, cfg.loss.large_unc_loss).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg.train.lr, cfg.train.head_lr_mult))
    h, w = cfg.data.crop
    image, label = torch.randn(batch, 3, h, w, device=device), torch.randint(len(CLASSES), (batch, h, w), device=device)
    large_unc = torch.rand(batch, h, w, device=device) if cfg.data.large_uncertainty else None
    for _ in range(2):  # the second step runs with cuDNN and the allocator warmed up
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.amp):
            logits, dist, large = model(image)
        loss, _ = criterion(logits, dist, large, label, large_unc)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.train.amp):
        model(Cityscapes(cfg.data.root, "val")[0][0][None].to(device))
    cuda, gib = device.type == "cuda", 2 ** 30
    print(json.dumps({"peak": torch.cuda.max_memory_reserved() / gib if cuda else 0.0,
                      "total": torch.cuda.get_device_properties(device).total_memory / gib if cuda else 0.0}))


def probe(train_args, batch):
    """Peak memory and GPU size (GiB) at this per-GPU batch in a fresh process; None if it runs out of memory."""
    run = subprocess.run([sys.executable, "-m", "src.sweep", *train_args, "--probe", str(batch)],
                         capture_output=True, text=True)
    if run.returncode == 0:
        return json.loads(run.stdout.splitlines()[-1])
    if any(s in run.stderr for s in ("out of memory", "ALLOC_FAILED")):
        return None
    raise RuntimeError(f"memory probe failed:\n{run.stderr[-3000:]}")


def largest_batch(train_args, headroom):
    """Largest per-GPU batch that keeps headroom GiB of the GPU free: linear fit of the probes at 2 and 4 images,
    then checked and reduced by 10% until it fits."""
    two, four = probe(train_args, 2), probe(train_args, 4)
    if two is None:
        sys.exit("a per-GPU batch of 2 does not fit")
    budget = two["total"] - headroom
    if four is None or four["peak"] > budget:
        return 2
    per_image = max((four["peak"] - two["peak"]) / 2, 0.05)
    batch = min(4 + int((budget - four["peak"]) / per_image), 256)
    while batch > 4:
        fit = probe(train_args, batch)
        if fit is not None and fit["peak"] <= budget:
            break
        batch = int(batch * 0.9)
    return batch


def sample(space, rng):
    """One value per entry of the space: log-uniform on log: [low, high] or one of choice: [...]."""
    params = {}
    for entry in space:
        if "choice" in entry:
            params[entry.key] = rng.choice(list(entry.choice))
        else:
            low, high = entry.log
            params[entry.key] = float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.3g}")
    return params


def dotted(params):
    return [f"{key}={value}" for key, value in params.items()]


def run(command, log):
    return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode == 0


def run_trial(cfg, gpus, train_args, out):
    """Trains (resuming from last.pt if present) and, for the small stage, evaluates routing -> metrics or None."""
    launcher = [sys.executable] if gpus == 1 else [
        sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}"]
    resume = ["--resume", str(out / "last.pt")] if (out / "last.pt").exists() else []
    with open(out / "log.txt", "a") as log:
        if not run([*launcher, "-m", "src.train", *train_args, *resume], log):
            return None
        metrics = json.loads((out / "metrics.jsonl").read_text().splitlines()[-1])  # the last validation
        metrics["gap"] = metrics["val/hyperbolic_uncertainty_wrong"] - metrics["val/hyperbolic_uncertainty_correct"]
        if cfg.stage == "small":
            if not run([sys.executable, "-m", "src.eval_router", "configs/eval_router.yaml", f"root={cfg.root}",
                        f"small={out / 'last.pt'}", f"large={cfg.large_checkpoint}", f"out={out / 'eval_router'}",
                        f"thresholds={list(cfg.thresholds)}", f"workers={cfg.workers}"], log):
                return None
            metrics |= routing(json.loads((out / "eval_router" / "results.json").read_text()), cfg.max_launch_rate)
    return metrics


def routing(results, max_launch):
    """Best routed mIoU among the thresholds that launch the large model on at most max_launch of the images;
    the small model alone if none does."""
    summary = results["summary"]
    never = {"threshold": None, "route/launch_rate": 0.0, "routed/mIoU": summary["small/mIoU"]}
    best = max((r for r in results["thresholds"] if r["route/launch_rate"] <= max_launch),
               key=lambda r: r["routed/mIoU"], default=never)
    return ({key: summary[key] for key in ("large/mIoU", "small/large_unc_mae", "route/fixable_pixels")}
            | {key: best[key] for key in never})


def score(cfg, metrics):
    """Higher is better: routed mIoU for the small stage, cfg.metric for the large one (ECE negated)."""
    if metrics is None:
        return None
    if cfg.stage == "small":
        return metrics["routed/mIoU"]
    return -metrics[cfg.metric] if cfg.metric.endswith("ECE") else metrics[cfg.metric]


def summary(trial):
    m = trial["metrics"] or {}
    shown = "  ".join(f"{name} {m[key]:.4f}" for key, name in SHOWN.items() if isinstance(m.get(key), float))
    status = "failed, see log.txt" if trial["score"] is None else f"score {trial['score']:.4f}"
    params = " ".join(dotted(trial["params"]))
    return f"trial {trial['trial']:3d}  {status}  {shown}  | {params}  ({trial['minutes']} min)"


def report(cfg, trials, top=10):
    ranked = sorted((t for t in trials.values() if t["score"] is not None), key=lambda t: -t["score"])
    failed = sorted(t["trial"] for t in trials.values() if t["score"] is None)
    criterion = "routed mIoU" if cfg.stage == "small" else cfg.metric
    print(f"\n{cfg.stage}: {len(ranked)} trials by {criterion}, best first" + (f"; failed: {failed}" if failed else ""))
    for trial in ranked[:top]:
        print(summary(trial))
    if ranked:
        print("best:", *dotted(ranked[0]["params"]), f"data.batch_size={ranked[0]['batch']}")


def main():
    args = parse_args()
    if args.probe:
        return probe_step(load_config(args.config, args.overrides), args.probe)
    cfg = load_config(args.config, args.overrides)
    gpus = cfg.gpus or max(torch.cuda.device_count(), 1)
    stage = Path(cfg.out, cfg.stage)
    stage.mkdir(parents=True, exist_ok=True)
    lines = (stage / "trials.jsonl").read_text().splitlines() if (stage / "trials.jsonl").exists() else []
    trials = {t["trial"]: t for t in map(json.loads, lines)}

    base = [*STAGES[cfg.stage], f"data.root={cfg.root}", f"data.workers={cfg.workers}"]
    after = list(cfg.overrides)  # the user's overrides win
    space = list(cfg.space)
    if cfg.stage == "small":  # the exported maps are per patch of the large model: use its patch, do not sample it
        patch = torch.load(cfg.large_checkpoint, map_location="cpu")["config"]["model"].get("patch", 1)
        base.append(f"data.large_uncertainty={cfg.large_uncertainty}")
        after.insert(0, f"model.patch={patch}")
        space = [entry for entry in space if entry.key != "model.patch"]
    defaults = load_config(base[0], base[1:] + after)

    if cfg.batch_per_gpu:
        batch = cfg.batch_per_gpu * gpus
    elif trials:  # keep the trials comparable
        batch = next(iter(trials.values()))["batch"]
    elif torch.cuda.is_available():  # probed with the largest numeric choices of the space, e.g. embed_dim
        heavy = {e.key: max(e.choice) for e in space if "choice" in e and not any(isinstance(v, str) for v in e.choice)}
        batch = largest_batch([*base, *dotted(heavy), *after], cfg.headroom_gb) * gpus
    else:
        batch = 2 * gpus
    iters = math.ceil(cfg.samples / batch)
    schedule = [f"data.batch_size={batch}", f"train.iters={iters}", f"train.warmup={iters // 20}",
                f"log.every={max(iters // 100, 1)}", f"log.val_every={max(iters // cfg.validations, 1)}"]
    print(f"{cfg.stage}: batch {batch} on {gpus} GPU(s), {iters} iterations per trial", flush=True)

    for trial in range(cfg.trials):
        if trial in trials:
            continue
        params = {} if trial == 0 else sample(space, random.Random(f"{cfg.seed}-{trial}"))
        lr = params.get("train.lr", defaults.train.lr)  # given for batch 8
        params["train.lr"] = float(f"{lr * math.sqrt(batch / 8):.3g}")
        out = stage / f"{trial:03d}"
        out.mkdir(exist_ok=True)
        tic = time.time()
        metrics = run_trial(cfg, gpus, [*base, *schedule, *dotted(params), *after, f"out={out}"], out)
        trials[trial] = {"trial": trial, "batch": batch, "params": params, "metrics": metrics,
                         "score": score(cfg, metrics), "minutes": round((time.time() - tic) / 60, 1)}
        with open(stage / "trials.jsonl", "a") as f:
            f.write(json.dumps(trials[trial]) + "\n")
        print(summary(trials[trial]), flush=True)
    report(cfg, trials)


if __name__ == "__main__":
    main()
