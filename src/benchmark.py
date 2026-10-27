"""Benchmark on a labelled set (all ACDC by default), for all images and per <city> folder (an ACDC condition).

- Models: B0 and B2 without fine-tuning (the pretrained SegFormer, rebuilt from each checkpoint's config) and after
  it; mIoU with bootstrap confidence intervals, IoU per class, paired tests (bootstrap, Wilcoxon on per-image pixel
  accuracy).
- Uncertainty of B0 and B2 after fine-tuning: ECE of the hyperbolic uncertainty and its mean on correct and wrong
  pixels; MAE between B0's prediction of B2's uncertainty and B2's own.
- Image score: mean over the `top` share of the most uncertain patches of a B0 signal: its own uncertainty, its gain
  (own - predicted large uncertainty) or, as a baseline, its softmax entropy.
- Routing: B2 after fine-tuning replaces B0 after fine-tuning on whole images whose gain score is above a threshold.
  Thresholds are absolute, so one chosen on Cityscapes val applies to ACDC as is.
- Adaptive inference: the same replacement for given shares of images (budgets) ranked by each score, compared with
  a random choice and an oracle (images where B2 is right on more pixels).
- Decisions per patch: a patch needs B2 if B2 is right on more of its labelled pixels than B0; precision, recall, F1,
  accuracy for the most uncertain shares of patches, AUROC and AP of the scores.
- B0's errors per patch: share of its wrong pixels in the most uncertain shares of patches; AUROC and AP for mostly
  wrong (>= 50% of labelled pixels) and noticeably wrong (>= 10%) patches; AUROC for mostly wrong patches averaged
  over entropy deciles, i.e. whether a signal finds errors that entropy cannot rank.

    python -m src.benchmark configs/benchmark.yaml root=/data/acdc_cityscapes                          # test
    python -m src.benchmark configs/benchmark.yaml root=/data/cityscapes split=val out=runs/benchmark_val  # threshold
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy import stats
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_jpeg
from tqdm import tqdm

from src.config import load_config
from src.data import CLASSES, IGNORE, Cityscapes, colorize, denormalize, heatmap, make_panel
from src.hyperbolic import hyperbolic_uncertainty
from src.logs import keep_log
from src.model import build_model, load_model
from src.train import SegMetrics

K = len(CLASSES)
MODELS = ("B0", "B2", "B0_ft", "B2_ft")
FINE_TUNED = ("B0_ft", "B2_ft")  # with a trained hyperbolic uncertainty
PAIRS = (("B0_ft", "B0"), ("B2_ft", "B2"), ("B2", "B0"), ("B2_ft", "B0_ft"))
# B0's own uncertainty; own - its prediction of the large model's uncertainty; its softmax entropy (baseline)
SIGNALS = ("own", "gain", "entropy")


def untrained(path):
    """The model a checkpoint was fine-tuned from: its config with the pretrained weights only."""
    return build_model(OmegaConf.create(torch.load(path, map_location="cpu")["config"]), K)


def original_pred(model, x):
    """Prediction of the pretrained SegFormer itself: its logits (the classifier is only scaled), upsampled."""
    logits = model.net(pixel_values=x).logits[:, :K].float()
    return F.interpolate(logits, x.shape[-2:], mode="bilinear", align_corners=False)[0].argmax(0)


def patch_sums(m, patch):
    """(H, W) -> sums over patch x patch squares (partial at the right and bottom edges)."""
    h, w = m.shape
    m = F.pad(m.float(), (0, -w % patch, 0, -h % patch))
    return m.view(m.shape[0] // patch, patch, m.shape[1] // patch, patch).sum((1, 3))


def image_score(u, patch, top):
    """Mean of an uncertainty-like map over the `top` share of its highest patches."""
    patches = (patch_sums(u, patch) / patch_sums(torch.ones_like(u), patch)).flatten()
    return patches.topk(max(1, round(top * patches.numel()))).values.mean().item()


def class_iou(conf):
    """(..., K * K) confusion counts -> (..., K) IoU per class, NaN for a class absent from labels and predictions."""
    conf = conf.reshape(*conf.shape[:-1], K, K)
    tp = conf.diagonal(dim1=-2, dim2=-1)
    return tp / (conf.sum(-1) + conf.sum(-2) - tp)


def miou(conf):
    """(..., K * K) confusion counts -> mIoU over the classes present."""
    return class_iou(conf).nanmean(-1)


def auroc(score, positive):
    pos, neg = score[positive], score[~positive]
    if not len(pos) or not len(neg):
        return float("nan")
    return stats.mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg))


def average_precision(score, positive):
    hits = positive[np.argsort(-score, kind="stable")]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / max(hits.sum(), 1))


def models_and_tests(conf, n_boot, rng):
    """mIoU with bootstrap CIs, IoU per class, paired differences with bootstrap p-values and Wilcoxon on pixel
    accuracy. conf: (images, models, K * K)."""
    n, q = conf.shape[0], torch.tensor([0.025, 0.975], dtype=torch.float64)
    weights = torch.zeros(n_boot, n, dtype=torch.float64).scatter_add_(
        1, torch.randint(n, (n_boot, n), generator=rng), torch.ones(n_boot, n, dtype=torch.float64))
    boot = miou((weights @ conf.flatten(1)).view(n_boot, len(MODELS), K * K))  # (n_boot, models)
    iou = class_iou(conf.sum(0))  # (models, K)
    point = iou.nanmean(-1)
    diag = conf.view(n, len(MODELS), K, K).diagonal(dim1=-2, dim2=-1).sum(-1)
    acc = diag / conf.sum(-1).clamp_min(1)  # per-image pixel accuracy
    out = {"models": {}, "differences": {}}
    for i, name in enumerate(MODELS):
        out["models"][name] = {"mIoU": point[i].item(), "ci": boot[:, i].nanquantile(q).tolist(),
                               "pixel_accuracy": (diag.sum(0)[i] / conf[:, i].sum().clamp_min(1)).item(),
                               "iou": dict(zip(CLASSES, iou[i].tolist()))}
    for a, b in PAIRS:
        i, j = MODELS.index(a), MODELS.index(b)
        delta = boot[:, i] - boot[:, j]
        diff = (acc[:, i] - acc[:, j]).numpy()
        out["differences"][f"{a}-{b}"] = {
            "mIoU": (point[i] - point[j]).item(),
            "ci": delta.nanquantile(q).tolist(),
            "p_bootstrap": min(1.0, 2 * min((delta <= 0).double().mean().item(), (delta >= 0).double().mean().item())),
            "p_wilcoxon": float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
        }
    return out


def uncertainty(calibration, mae, labelled):
    """ECE of 1 - hyperbolic uncertainty, its bins (pixels, mean confidence, accuracy) and its mean on correct / wrong
    pixels for each fine-tuned model (SegMetrics, as in validation); MAE of B0's prediction of B2's uncertainty from
    per-image sums over the labelled pixels."""
    out = {}
    for name, metrics in calibration.items():
        values = metrics.compute(name)
        out[name] = {key: values[f"{name}/hyperbolic_{key}"]
                     for key in ("ECE", "uncertainty_correct", "uncertainty_wrong")}
        pixels, confidence, correct = metrics.calibration["hyperbolic"]
        out[name]["reliability"] = {"pixels": pixels.tolist(), "confidence": (confidence / pixels).tolist(),
                                    "accuracy": (correct / pixels).tolist()}
    return out | {"large_unc_mae": (mae.sum() / labelled.sum().clamp_min(1)).item()}


def routing(conf, score, fixable, thresholds, cost_ratio, group):
    """B2 after fine-tuning on the whole images whose score is above each threshold, B0 after fine-tuning on the rest:
    mIoU, launch rate, cost, gap closed; fixable pixels (B0 wrong, B2 right) on the launched images per labelled pixel
    there (precision) and per fixable pixel of the group (recall)."""
    b0, b2 = conf[group, MODELS.index("B0_ft")], conf[group, MODELS.index("B2_ft")]
    base, top = miou(b0.sum(0)).item(), miou(b2.sum(0)).item()
    labelled, fixable, score = b0.sum(1), fixable[group], score[group]
    rows = []
    for threshold in thresholds:
        launched = score > threshold
        value, launch = miou(torch.where(launched[:, None], b2, b0).sum(0)).item(), launched.double().mean().item()
        fixed = fixable[launched].sum().item()
        rows.append({"threshold": threshold, "launch_rate": launch, "mIoU": value, "cost": 1 + cost_ratio * launch,
                     "gap_closed": (value - base) / (top - base) if top != base else float("nan"),
                     "fix_precision": fixed / max(labelled[launched].sum().item(), 1),
                     "fix_recall": fixed / max(fixable.sum().item(), 1)})
    return {"fixable_pixels": fixable.sum().item() / max(labelled.sum().item(), 1), "thresholds": rows}


def adaptive(conf, scores, benefit, launch_order, budgets, cost_ratio, group):
    """mIoU of adaptive inference per budget for each policy, random and oracle, on the images of `group`."""
    b0, b2 = conf[:, MODELS.index("B0_ft")], conf[:, MODELS.index("B2_ft")]
    base, top = miou(b0[group].sum(0)).item(), miou(b2[group].sum(0)).item()
    curves = defaultdict(list)
    for budget in budgets:
        random = miou((1 - budget) * b0[group].sum(0) + budget * b2[group].sum(0)).item()
        rows = {"random": (random, budget)}
        for name, order in launch_order.items():
            launched = torch.zeros(len(conf), dtype=torch.bool)
            launched[order[:round(budget * len(conf))]] = True
            mixed = torch.where(launched[:, None], b2, b0)[group].sum(0)
            rows[name] = (miou(mixed).item(), launched[group].double().mean().item())
        oracle = rows["oracle"][0]
        for name, (value, launch) in rows.items():
            curves[name].append({
                "budget": budget, "launch_rate": launch, "mIoU": value, "cost": 1 + cost_ratio * launch,
                "gap_closed": (value - base) / (top - base) if top != base else float("nan"),
                "oracle_efficiency": (value - random) / (oracle - random) if oracle != random else float("nan"),
            })
    positive = (benefit[group] > 0).numpy()
    frame_auroc = {name: float(auroc(scores[group][:, k].numpy(), positive)) for k, name in enumerate(SIGNALS)}
    return dict(curves), frame_auroc


def decisions(patches, thresholds, budgets, group):
    """Patch decisions of each signal against 'B2 is right on more pixels' (thresholds from all patches)."""
    need = patches["need"][group]
    out = {"patches": int(group.sum()), "base_rate": float(need.mean())}
    for signal in SIGNALS:
        score = patches[signal][group]
        rows = []
        for budget, threshold in zip(budgets, thresholds[signal]):
            flagged = score >= threshold if budget > 0 else np.zeros_like(need)
            tp = (flagged & need).sum()
            precision, recall = tp / max(flagged.sum(), 1), tp / max(need.sum(), 1)
            rows.append({"budget": budget, "flagged": float(flagged.mean()), "precision": float(precision),
                         "recall": float(recall), "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
                         "accuracy": float((flagged == need).mean())})
        out[signal] = {"auroc": float(auroc(score, need)), "ap": average_precision(score, need), "budgets": rows}
    return out


def errors(patches, thresholds, budgets, cuts, group):
    """B0's errors found by each signal (thresholds from all patches): recall = share of B0's wrong pixels in the
    flagged patches; AUROC and AP for mostly wrong (y50: >= 50% of labelled pixels) and noticeably wrong (y_any:
    >= 10%) patches; AUROC for y50 averaged over entropy deciles (cut points `cuts` from all patches)."""
    wrong = patches["wrong"][group]
    share = wrong / patches["valid"][group]
    labels = {"y50": share >= 0.5, "y_any": share >= 0.1}
    bins = np.digitize(patches["entropy"][group], cuts)
    out = {"rate": {k: float(y.mean()) for k, y in labels.items()}}
    for signal in SIGNALS:
        score = patches[signal][group]
        recall = [wrong[score >= threshold].sum() / max(wrong.sum(), 1) for threshold in thresholds[signal]]
        deciles = [auroc(score[bins == b], labels["y50"][bins == b]) for b in range(len(cuts) + 1)]
        deciles = [a for a in deciles if not np.isnan(a)]  # bins without positives or negatives
        out[signal] = {"auroc": {k: float(auroc(score, y)) for k, y in labels.items()},
                       "ap": {k: average_precision(score, y) for k, y in labels.items()},
                       "auroc_y50_entropy_deciles": float(np.mean(deciles)) if deciles else float("nan"),
                       "budgets": [{"budget": b, "recall": float(r)} for b, r in zip(budgets, recall)]}
    return out


def report(name, result, budgets):
    print(f"\n== {name}: {result['images']} images")
    print("model    mIoU    95% CI            pixel acc")
    for model, m in result["models"].items():
        print(f"{model:7}  {m['mIoU']:.4f}  [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]  {m['pixel_accuracy']:.4f}")
    print("IoU per class " + "  ".join(f"{m:>6}" for m in MODELS))
    for cls in CLASSES:
        print(f"{cls:13} " + "  ".join(f"{result['models'][m]['iou'][cls]:6.3f}" for m in MODELS))
    print("difference   dmIoU    95% CI              p boot   p Wilcoxon")
    for pair, d in result["differences"].items():
        print(f"{pair:11}  {d['mIoU']:+.4f}  [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  {d['p_bootstrap']:.4f}   "
              f"{d['p_wilcoxon']:.4f}")
    u = result["uncertainty"]
    print("hyperbolic uncertainty  ECE     on correct  on wrong")
    for m in FINE_TUNED:
        print(f"{m:22}  {u[m]['ECE']:.4f}  {u[m]['uncertainty_correct']:10.4f}  {u[m]['uncertainty_wrong']:8.4f}")
    print(f"MAE of B0_ft's prediction of B2_ft's uncertainty {u['large_unc_mae']:.4f}")
    r = result["routing"]
    print(f"routing: B2_ft on the images with gain score > threshold; fixable pixels {r['fixable_pixels']:.3f}")
    print("threshold  launch_rate  mIoU    gap_closed  fix_precision  fix_recall")
    for row in r["thresholds"]:
        print(f"{row['threshold']:9.2f}  {row['launch_rate']:11.2f}  {row['mIoU']:.4f}  {row['gap_closed']:10.2f}  "
              f"{row['fix_precision']:13.3f}  {row['fix_recall']:10.3f}")
    names = ["random", "oracle", *SIGNALS]
    print("adaptive mIoU (launch rate) by budget\nbudget  " + "  ".join(f"{n:>15}" for n in names))
    for k, budget in enumerate(budgets):
        cells = [f"{result['adaptive'][n][k]['mIoU']:.4f} ({result['adaptive'][n][k]['launch_rate']:.2f})"
                 for n in names]
        print(f"{budget:6.2f}  " + "  ".join(f"{c:>15}" for c in cells))
    print("frame AUROC (B2 better on the image): " + ", ".join(
        f"{n} {v:.3f}" for n, v in result["frame_auroc"].items()))
    p = result["patches"]
    print(f"patches: {p['patches']}, need B2 {p['base_rate']:.3f} | " + " | ".join(
        f"{s}: AUROC {p[s]['auroc']:.3f}, AP {p[s]['ap']:.3f}" for s in SIGNALS))
    print("budget  " + "  ".join(f"{s + ' precision/recall/F1':>28}" for s in SIGNALS))
    for k, budget in enumerate(budgets):
        cells = [f"{p[s]['budgets'][k]['precision']:.3f} / {p[s]['budgets'][k]['recall']:.3f} / "
                 f"{p[s]['budgets'][k]['f1']:.3f}" for s in SIGNALS]
        print(f"{budget:6.2f}  " + "  ".join(f"{c:>28}" for c in cells))
    e = result["errors"]
    print(f"B0_ft errors: patches mostly wrong (y50) {e['rate']['y50']:.3f}, noticeably wrong (y_any) "
          f"{e['rate']['y_any']:.3f}")
    print("signal   AUROC y50  AUROC y_any  AP y50  AP y_any  AUROC y50 in entropy deciles")
    for s in SIGNALS:
        print(f"{s:7}  {e[s]['auroc']['y50']:9.3f}  {e[s]['auroc']['y_any']:11.3f}  {e[s]['ap']['y50']:6.3f}  "
              f"{e[s]['ap']['y_any']:8.3f}  {e[s]['auroc_y50_entropy_deciles']:28.3f}")
    print("budget  " + "  ".join(f"{s + ' error recall':>20}" for s in SIGNALS))
    for k, budget in enumerate(budgets):
        print(f"{budget:6.2f}  " + "  ".join(f"{e[s]['budgets'][k]['recall']:20.3f}" for s in SIGNALS))


def log(writer, prefix, value, step=0):
    """Numbers of a nested result as TensorBoard scalars; lists of rows per budget or threshold become curves with
    step = 100 x budget or threshold."""
    if isinstance(value, dict):
        for key, item in value.items():
            log(writer, f"{prefix}/{key}", item, step)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        x = "budget" if "budget" in value[0] else "threshold"
        for row in value:
            for key, item in row.items():
                if key != x and np.isfinite(item):
                    writer.add_scalar(f"{prefix}/{key}", item, round(100 * row[x]))
    elif isinstance(value, (int, float)) and np.isfinite(value):
        writer.add_scalar(prefix, value, step)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. root=/data/acdc_cityscapes")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    out = Path(cfg.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)
    keep_log(out / "benchmark.log")
    OmegaConf.save(cfg, out / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = {"B0": untrained(cfg.small), "B2": untrained(cfg.large),
            "B0_ft": load_model(cfg.small, K), "B2_ft": load_model(cfg.large, K)}
    assert nets["B0_ft"].large_head, "the small model must be trained with data.large_uncertainty"
    nets = {name: net.to(device).eval() for name, net in nets.items()}
    dataset = Cityscapes(cfg.root, cfg.split)
    loader = DataLoader(dataset, 1, num_workers=cfg.workers)
    writer = SummaryWriter(out)

    calibration = defaultdict(lambda: {name: SegMetrics(K) for name in FINE_TUNED})  # "all" and one per folder
    conf, scores, benefit, fixable, mae, folders, patches = [], [], [], [], [], [], defaultdict(list)
    panels = defaultdict(int)
    for path, (image, label) in zip(dataset.images, tqdm(loader)):
        x, target, folder = image.to(device), label[0].to(device), path.parent.name
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            preds = [original_pred(nets["B0"], x), original_pred(nets["B2"], x)]
            logits0, dist0, large0 = nets["B0_ft"](x)
            logits2, dist2, _ = nets["B2_ft"](x)
        preds += [logits0[0].argmax(0), logits2[0].argmax(0)]
        valid = target != IGNORE
        conf.append(torch.stack([torch.bincount(target[valid] * K + pred[valid], minlength=K * K).cpu()
                                 for pred in preds]))
        # hyperbolic uncertainties of B0 and B2 after fine-tuning, B0's prediction of B2's
        own, own2, predicted = hyperbolic_uncertainty(dist0)[0], hyperbolic_uncertainty(dist2)[0], large0[0].sigmoid()
        for name in ("all", folder):
            calibration[name]["B0_ft"].update(preds[2], target, hyperbolic=own)
            calibration[name]["B2_ft"].update(preds[3], target, hyperbolic=own2)
        mae.append((predicted - own2)[valid].abs().sum().item())
        log_p = logits0[0].float().log_softmax(0)
        signals = {"own": own, "gain": own - predicted, "entropy": -(log_p.exp() * log_p).sum(0)}
        scores.append([image_score(signals[s], cfg.patch, cfg.top) for s in SIGNALS])
        right0, right2 = (preds[2] == target) & valid, (preds[3] == target) & valid
        benefit.append((right2.sum() - right0.sum()).item())
        fixable.append((right2 & ~right0).sum().item())
        folders.append(folder)
        if panels[folder] < cfg.images:
            tiles = [denormalize(x[0]), colorize(target), colorize(preds[2]), colorize(preds[3]),
                     heatmap(own), heatmap(predicted), heatmap(own2), heatmap(signals["gain"].clamp(0, 1))]
            panel = make_panel(tiles, cols=4)
            writer.add_image(f"{folder}/sample_{panels[folder]}", panel, 0)
            write_jpeg(panel, str(out / "panels" / f"{folder}_{panels[folder]}.jpg"), quality=90)
            panels[folder] += 1
        n_valid, n_right0 = patch_sums(valid, cfg.patch), patch_sums(right0, cfg.patch)
        labelled = n_valid > 0
        area = patch_sums(torch.ones_like(own), cfg.patch)
        patches["need"].append((patch_sums(right2, cfg.patch) > n_right0)[labelled].cpu())
        patches["valid"].append(n_valid[labelled].cpu())
        patches["wrong"].append((n_valid - n_right0)[labelled].cpu())
        for s in SIGNALS:
            patches[s].append((patch_sums(signals[s], cfg.patch) / area)[labelled].cpu())
        patches["folder"] += [folder] * int(labelled.sum())

    conf = torch.stack(conf).double()
    scores, benefit = torch.tensor(scores, dtype=torch.float64), torch.tensor(benefit, dtype=torch.float64)
    fixable, mae = torch.tensor(fixable, dtype=torch.float64), torch.tensor(mae, dtype=torch.float64)
    patches = {k: np.array(v) if k == "folder" else torch.cat(v).float().numpy() for k, v in patches.items()}
    patches["need"] = patches["need"].astype(bool)
    launch_order = {name: scores[:, k].argsort(descending=True) for k, name in enumerate(SIGNALS)}
    launch_order["oracle"] = benefit.argsort(descending=True)
    budgets = list(cfg.budgets)
    thresholds = {s: [np.quantile(patches[s], 1 - b) if b > 0 else np.inf for b in budgets] for s in SIGNALS}
    cuts = np.quantile(patches["entropy"], np.arange(1, 10) / 10)  # entropy deciles of all patches
    folders, rng = np.array(folders), torch.Generator().manual_seed(cfg.seed)

    results = {}
    for name in ["all", *dict.fromkeys(folders)]:
        group = np.ones(len(folders), bool) if name == "all" else folders == name
        group_t = torch.from_numpy(group)
        curves, frame_auroc = adaptive(conf, scores, benefit, launch_order, budgets, cfg.cost_ratio, group_t)
        in_group = np.ones(len(patches["need"]), bool) if name == "all" else patches["folder"] == name
        labelled = conf[group_t, MODELS.index("B0_ft")].sum(1)
        results[name] = {"images": int(group.sum()), **models_and_tests(conf[group_t], cfg.bootstrap, rng),
                         "uncertainty": uncertainty(calibration[name], mae[group_t], labelled),
                         "routing": routing(conf, scores[:, SIGNALS.index("gain")], fixable, list(cfg.thresholds),
                                            cfg.cost_ratio, group_t),
                         "adaptive": curves, "frame_auroc": frame_auroc,
                         "patches": decisions(patches, thresholds, budgets, in_group),
                         "errors": errors(patches, thresholds, budgets, cuts, in_group)}
        report(name, results[name], budgets)
        log(writer, name, {k: v for k, v in results[name].items() if k != "images"})
    writer.close()
    with open(out / "benchmark.json", "w") as f:  # for scripts, e.g. src.sweep and src.report
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
