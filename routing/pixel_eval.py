"""Попиксельная оценка Махаланобиса по выходу B0 (Henriksson et al., 2024): статистики на P0, оценка на тесте.

    python -m routing.pixel_eval
    python -m routing.pixel_eval --eps 1e-3 --variants logits --name md_pixel_logits

Метка: пиксель ошибочный (pred_b0 ≠ GT, без ignore). Для каждого варианта v × eps, MSP и энтропии:
  - AUROC «пиксель ошибочный»;
  - pooled: общий порог на все пиксели, оставляем долю с наименьшим score, риск = 1 − mIoU
    оставленных; AURC — среднее риска по 100 точкам покрытия; то же для доли ошибочных пикселей
    (aurc_err, err@…) — 1 − mIoU штрафует набор, где класс остался только ошибочными пикселями;
  - paper (Sec. III-C, IV-A): в каждом кадре 60 порогов от min до max score кадра, принят пиксель
    с score < ε; coverage = принятые / размеченные, risk = 1 − IoU принятых; risk и coverage
    усредняются по кадрам. cov@risk15 — наибольшее покрытие при risk ≤ 15% (рабочая точка статьи).
    ДОПУЩЕНИЕ: IoU кадра — mIoU по классам, присутствующим среди принятых пикселей и в GT.
Оценка — на 5% размеченных пикселей каждого кадра (те же, что у остальных попиксельных оценок).
Результаты: results/runs/<датасет>_<популяция>/<name>/ — pixel_auroc.csv, risk_coverage.csv, config.json.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch

from core.components import NC
from core.data import FIT_DATASET, FIT_POP
from core.evaluation import PixelSampler, entropy_score, msp_score, routing_cache
from core.paths import run_dir, write_json
from features.logits import LogitStore
from routing.methods.density import VARIANTS, OutputStats, fit_samples, transform

COVERAGES = np.linspace(0.01, 1.0, 100)
N_THR = 60


def msp(x, cls):
    return msp_score(x.softmax(1), dim=1)


def entropy(x, cls):
    return entropy_score(x.softmax(1), dim=1)


@torch.inference_mode()
def collect_pixels(dataset, pop, scorers, frac=0.05, seed=0):
    """Пиксели теста и их оценки. scorers {имя: f(x (n, 19) логиты, cls (n,)) → (n,)}.
    Возвращает (PixelSampler с оценками, dict gt / pred / image по выбранным пикселям)."""
    cache, logits = routing_cache(dataset, pop), LogitStore(dataset)
    sampler = PixelSampler(frac, seed)
    extra = {"gt": [], "pred": [], "image": []}
    for k, image_id in enumerate(cache["image_id"]):
        gt, pred = cache["gt"][k], cache["b0"][k]
        sel = sampler.frame(f"{dataset}_{pop}", gt, pred)
        x = logits[image_id].reshape(NC, -1)[:, torch.from_numpy(sel).cuda()].T
        cls = torch.from_numpy(pred.ravel()[sel].astype(np.int64)).cuda()
        for name, f in scorers.items():
            sampler.add(name, f(x, cls).float().cpu().numpy())
        extra["gt"].append(gt.ravel()[sel])
        extra["pred"].append(pred.ravel()[sel])
        extra["image"].append(np.full(len(sel), k))
    return sampler, {k: np.concatenate(v) for k, v in extra.items()}


def pixel_auroc(name, scorer, dataset="cityscapes", pop="p1"):
    """AUROC «пиксель B0 ошибочный» для scorer рядом с MSP и энтропией: {имя: AUROC}."""
    sampler, _ = collect_pixels(dataset, pop, {name: scorer, "MSP": msp, "entropy": entropy})
    t = sampler.auroc_table()
    return dict(zip(t.score, t.auroc))


def prefix_risk(g, p, ks):
    """1 − mIoU по первым k пикселям (g, p уже отсортированы) для каждого k из ks (по возрастанию);
    NaN при k = 0. Матрица ошибок наращивается по отрезкам между соседними k."""
    codes, conf, prev, out = g * NC + p, np.zeros(NC * NC, np.int64), 0, []
    for k in ks:
        assert k >= prev
        conf += np.bincount(codes[prev:k], minlength=NC * NC)
        prev = k
        c = conf.reshape(NC, NC)
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.diag(c) / (c.sum(0) + c.sum(1) - np.diag(c))
        out.append(1 - np.nanmean(iou) if k > 0 else np.nan)
    return np.array(out)


def risk_coverage(score, gt, pred):
    """pooled: риск 1 − mIoU оставленных пикселей при каждом покрытии из COVERAGES."""
    order = np.argsort(score, kind="stable")
    ks = np.round(COVERAGES * len(order)).astype(int)
    return prefix_risk(gt[order].astype(np.int64), pred[order].astype(np.int64), ks)


def paper_curve(score, gt, pred, image):
    """paper: пороги по кадру, средние по кадрам coverage и risk в каждой из N_THR точек."""
    covs, risks = [], []
    _, starts = np.unique(image, return_index=True)        # пиксели кадра идут подряд
    for a, b in zip(starts, list(starts[1:]) + [len(image)]):
        s = score[a:b]
        order = np.argsort(s, kind="stable")
        ks = np.searchsorted(s[order], np.linspace(s.min(), s.max(), N_THR), side="left")
        covs.append(ks / (b - a))
        risks.append(prefix_risk(gt[a:b][order].astype(np.int64), pred[a:b][order].astype(np.int64), ks))
    with warnings.catch_warnings():                 # при ε = min принятых нет → risk NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.mean(covs, 0), np.nanmean(risks, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--eps", default="0,1e-4,1e-3,1e-2", help="0 = псевдообратная")
    ap.add_argument("--pixel-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", default="cityscapes:p1")
    ap.add_argument("--name", default="md_output_pixel")
    args = ap.parse_args()

    dataset, pop = args.test.split(":")
    samples = fit_samples(routing_cache(FIT_DATASET, FIT_POP), LogitStore(FIT_DATASET), args.seed)
    counts = {c: len(samples[c]) for c in range(NC)}
    print("верных пикселей P0 в fit по классам:", counts)
    configs = {f"MD {v} eps={e}": OutputStats(v, float(e)).fit(samples)
               for v in args.variants.split(",") for e in args.eps.split(",")}
    scorers = {"MSP": msp, "entropy": entropy,
               **{name: (lambda x, cls, st=st: st.md(transform(x, st.variant), cls))
                  for name, st in configs.items()}}
    sampler, px = collect_pixels(dataset, pop, scorers, args.pixel_frac, args.seed)
    err = np.concatenate(sampler.cols["_err"])
    auroc = sampler.auroc_table().set_index("score").auroc

    rows, rc_rows = [], []
    for name in scorers:
        s = np.concatenate(sampler.cols[name])
        risk = risk_coverage(s, px["gt"], px["pred"])
        e = np.cumsum(err[np.argsort(s, kind="stable")])
        err_risk = e[np.round(COVERAGES * len(e)).astype(int) - 1] / np.round(COVERAGES * len(e))
        p_cov, p_risk = paper_curve(s, px["gt"], px["pred"], px["image"])
        ok = p_risk <= 0.15
        rows.append({"domain": args.test, "score": name, "auroc": auroc[name], "aurc": risk.mean(),
                     "risk@50%": risk[49], "risk@80%": risk[79], "aurc_err": err_risk.mean(),
                     "err@50%": err_risk[49], "err@80%": err_risk[79],
                     "paper cov@risk15": p_cov[ok].max() if ok.any() else 0.0})
        rc_rows += [{"score": name, "curve": "pooled", "coverage": c, "risk": r}
                    for c, r in zip(COVERAGES, risk)]
        rc_rows += [{"score": name, "curve": "paper", "coverage": c, "risk": r}
                    for c, r in zip(p_cov, p_risk)]
    table = pd.DataFrame(rows)

    out = run_dir(args.name, args.test)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "pixel_auroc.csv", index=False)
    pd.DataFrame(rc_rows).to_csv(out / "risk_coverage.csv", index=False)
    write_json(out / "config.json", {"script": "routing.pixel_eval", "args": vars(args),
                                     "fit_counts": counts, "n_pixels": int(len(err)),
                                     "error_rate": float(err.mean())})
    pd.set_option("display.width", 200)
    print(f"{args.test}: {len(err)} пикселей, доля ошибок {err.mean():.4f}")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
