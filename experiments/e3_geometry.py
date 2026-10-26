"""E3: geometric scores on B0 decoder features vs entropy.

Fitted on cs_fit pixels with GT (feature resolution 1/8):
  maha       - min Mahalanobis distance to class Gaussians (tied covariance)
  geo_margin - d1 - d2 between the two nearest classes (higher = closer to a boundary)
  norm       - feature norm, sign chosen on cs_fit
  knn        - mean cosine distance to K nearest features from a cs_fit bank
  lr_all     - logistic regression on entropy + all of the above, target "B5 helps", fit on cs_fit
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import CACHE, DOMAINS, IGNORE, NUM_CLASSES, RESULTS, TILE, load, route_curve
from e2_confidence import BUDGETS, uncertainty

STRIDE = 8  # features are at 1/8 resolution
FIT_PIXELS, PIXELS_PER_IMAGE = 500, 1000
BANK, K = 20000, 10
GEO = ["maha", "geo_margin", "norm", "knn"]

rng = np.random.default_rng(0)
torch.manual_seed(0)


TILE_STATS = {"mean": lambda v: v.mean(-1), "p90": lambda v: np.percentile(v, 90, axis=-1)}


def tile_values(maps):
    """(N, H, W) -> (N, tiles_h, tiles_w, pixels_per_tile)"""
    n, h, w = maps.shape
    t = TILE // STRIDE
    return maps.reshape(n, h // t, t, w // t, t).transpose(0, 1, 3, 2, 4).reshape(n, h // t, w // t, -1)


def at_feat_res(*arrays):
    return [a[STRIDE // 2::STRIDE, STRIDE // 2::STRIDE].ravel() for a in arrays]


def sample(gt, n):
    valid = np.flatnonzero(gt != IGNORE)
    return rng.choice(valid, min(n, valid.size), replace=False)


def fit_geometry():
    feats, labels = [], []
    for gt, feat in load("cs_fit", "gt", "feat_b0"):
        (gt,) = at_feat_res(gt)
        idx = sample(gt, FIT_PIXELS)
        feats.append(feat.reshape(feat.shape[0], -1).T[idx])
        labels.append(gt[idx])
    f = torch.from_numpy(np.concatenate(feats)).cuda().float()
    y = torch.from_numpy(np.concatenate(labels)).cuda().long()

    mu = torch.stack([f[y == c].mean(0) for c in range(NUM_CLASSES)])
    centered = f - mu[y]
    cov = centered.T @ centered / len(f) + 1e-4 * torch.eye(f.shape[1], device="cuda")
    whiten = torch.linalg.cholesky(torch.linalg.inv(cov))  # |x W|^2 = x^T cov^-1 x
    bank = F.normalize(f[torch.randperm(len(f), device="cuda")[:BANK]], dim=1)
    return mu @ whiten, whiten, bank


@torch.inference_mode()
def score_maps(feat, logits, geo):
    mu_w, whiten, bank = geo
    f = torch.from_numpy(feat).cuda().float().flatten(1).T
    d = torch.cdist(f @ whiten, mu_w).topk(2, dim=1, largest=False).values
    knn = 1 - (F.normalize(f, dim=1) @ bank.T).topk(K, dim=1).values.mean(1)
    scores = {"maha": d[:, 0], "geo_margin": d[:, 0] - d[:, 1], "norm": f.norm(dim=1), "knn": knn}
    maps = {k: v.view(feat.shape[1:]).cpu().numpy() for k, v in scores.items()}
    entropy = uncertainty(torch.from_numpy(logits).float())["entropy"]
    maps["entropy"] = F.avg_pool2d(entropy[None], 2)[0].numpy()  # 1/4 -> 1/8
    return maps


def collect(domain, geo):
    maps, pix, err, helps = {}, {}, [], []
    for gt, b0, b5, logits, feat in load(domain, "gt", "pred_b0", "pred_b5", "logits_b0", "feat_b0"):
        m = score_maps(feat, logits, geo)
        gt, b0, b5 = at_feat_res(gt, b0, b5)
        idx = sample(gt, PIXELS_PER_IMAGE)
        for k, v in m.items():
            maps.setdefault(k, []).append(v)
            pix.setdefault(k, []).append(v.ravel()[idx])
        wrong = b0[idx] != gt[idx]
        err.append(wrong)
        helps.append(wrong & (b5[idx] == gt[idx]))
    return ({k: np.stack(v) for k, v in maps.items()}, {k: np.concatenate(v) for k, v in pix.items()},
            np.concatenate(err), np.concatenate(helps))


def main():
    base = json.loads((RESULTS / "e0.json").read_text())
    geo = fit_geometry()
    data = {d: collect(d, geo) for d in DOMAINS}

    _, fit_pix, fit_err, fit_help = data["cs_fit"]
    if roc_auc_score(fit_err, fit_pix["norm"]) < 0.5:
        for maps, pix, *_ in data.values():
            maps["norm"], pix["norm"] = -maps["norm"], -pix["norm"]

    cols = ["entropy"] + GEO
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    lr.fit(np.stack([fit_pix[c] for c in cols], 1), fit_help)
    predict = lambda d: lr.predict_proba(np.stack([d[c].ravel() for c in cols], 1))[:, 1]
    for maps, pix, *_ in data.values():
        maps["lr_all"] = predict(maps).reshape(maps["entropy"].shape)
        pix["lr_all"] = predict(pix)
    print("lr_all coefs (standardized):", dict(zip(cols, lr[-1].coef_[0].round(2))))

    np.savez(CACHE / "e3_tiles.npz", **{f"{d}__{k}__{stat}": f(tile_values(maps[k]))
                                         for d, (maps, *_) in data.items() for k in ["entropy"] + GEO
                                         for stat, f in TILE_STATS.items()})

    report = {}
    print(f"{'domain':8} {'method':10} {'AUC err':>8} {'AUC help':>9}" + "".join(f"{b:>7.0%}" for b in BUDGETS))
    for domain, (maps, pix, err, helps) in data.items():
        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        report[domain] = {}
        for k in ["entropy"] + GEO + ["lr_all"]:
            tiles = tile_values(maps[k]).mean(-1)
            closed = [100 * (m - b0_miou) / (b5_miou - b0_miou) for m in route_curve(domain, tiles, BUDGETS)]
            a_err, a_help = roc_auc_score(err, pix[k]), roc_auc_score(helps, pix[k])
            report[domain][k] = {"auc_err": round(a_err, 3), "auc_help": round(a_help, 3),
                                 "closed": [round(c, 1) for c in closed]}
            print(f"{domain:8} {k:10} {a_err:8.3f} {a_help:9.3f}" + "".join(f"{c:7.1f}" for c in closed))

    (RESULTS / "e3.json").write_text(json.dumps({"budgets": BUDGETS, **report}, indent=2))


if __name__ == "__main__":
    main()
