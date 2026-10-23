"""kNN-признаки для замороженной B0: DNP / cDNP (Galesso et al., ICCVW 2023, arXiv 2211.06660,
github.com/silviogalesso/dense-ood-knns), косинусный kNN и CSLS (Conneau, Lample et al. 2018,
arXiv 1710.04087).

Признаки (банк = признак × вариант опорного набора):
  Q, K, V — выходы q/k/v_proj последнего блока стадии 4, 8 голов × 32 склеены (как в коде статьи),
            сетка 1/32; S4 — выход стадии 4; C — вход classifier, сетка 1/4.
  Для L2 — без нормировки (как в статье), для cos / CSLS — L2-нормированные.
Опорный набор — кадры P0 (Cityscapes train нет): n_hold кадров отложены, остальные — опорные.
  Метка токена — большинство GT-пикселей в ячейке без ignore (в коде статьи — nearest), токены без
  размеченных пикселей не берутся (как в коде).
  A — случайные n_ref токенов (как в коде); B — равная квота ⌊n_ref/19⌋ на класс, у редких —
  все их токены (ОТСТУПЛЕНИЕ: в статье per-class greedy coreset).
kNN — точный перебор на GPU пачками, k соседей:
  dnp   — среднее L2-расстояние до k соседей (ур. 1–2 статьи);
  cdnp  — dnp / max(dnp) + (S_P − min S_P) / (max S_P − min S_P), S_P = −logsumexp(логиты) (ур. 3–5);
          max dnp — по отложенным кадрам P0, min dnp = 0 (как в коде), min/max S_P — по пикселям P0;
          dnp поднимается до рабочего разрешения ближайшим соседом (как в коде);
  cos   — 1 − среднее косинусное сходство к k соседям;
  csls  — 1 − среднее CSLS(x, b) = 2 cos(x, b) − r_T(x) − r_S(b) к k лучшим; r_T(x) — среднее cos
          к K ближайшим опорным, r_S(b) — среднее cos b к её K ближайшим в опорном наборе (ТЗ);
  cslsb — то же, но r_S(b) по двудольной окрестности, как в статье: к K ближайшим признакам
          отложенных кадров P0.
Хабность — сколько раз опорная точка попадает в k ближайших признаков отложенных кадров.
Колонки: <банк>_<оценка>_<mean|max>, банк = <Q|K|V|S4|C>_<A|B>; плюс neglse_mean / neglse_max.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import skew

from core.components import IGNORE, NC, comp_max, comp_mean
from core.data import FIT_DATASET, FIT_POP
from core.evaluation import routing_cache

FEATS = ("Q", "K", "V", "S4", "C")
CELL = {"Q": 32, "K": 32, "V": 32, "S4": 32, "C": 4}    # размер ячейки токена в пикселях рабочего кадра
VARIANTS = ("A", "B")
SCORES = ("dnp", "cdnp", "cos", "csls", "cslsb")
CHUNK = 1024
dev = "cuda"


def token_labels(gt, cell):
    """Метка токена — большинство размеченных GT-пикселей в ячейке cell×cell; −1, если их нет."""
    h, w = gt.shape
    g = torch.from_numpy(gt.astype(np.int64)).to(dev)
    g = torch.where(g == IGNORE, torch.full_like(g, NC), g)
    oh = F.one_hot(g, NC + 1)[..., :NC]
    cnt = oh.reshape(h // cell, cell, w // cell, cell, NC).sum((1, 3)).reshape(-1, NC)
    lab = cnt.argmax(1)
    lab[cnt.sum(1) == 0] = -1
    return lab


def up(tok, cell, shape):
    """Карта токенов (n,) на сетке ячеек cell → (H·W,) ближайшим соседом; shape = (H, W)."""
    h, w = shape
    m = tok.reshape(1, 1, h // cell, w // cell)
    return F.interpolate(m, size=(h, w), mode="nearest").reshape(-1)


def topk_chunks(q, ref, k, metric):
    """Для каждого q — k лучших среди ref пачками: "l2" — наименьшие расстояния, "cos" (q, ref
    нормированы) — наибольшие сходства. Возвращает (значения (n, k), индексы (n, k))."""
    vals, idx = [], []
    for s in range(0, len(q), CHUNK):
        qc = q[s:s + CHUNK]
        if metric == "l2":
            v, i = torch.cdist(qc, ref).topk(k, dim=1, largest=False)
        else:
            v, i = (qc @ ref.T).topk(k, dim=1)
        vals.append(v)
        idx.append(i)
    return torch.cat(vals), torch.cat(idx)


def rs_within(refn, K):
    """r_S(b) (ТЗ): среднее cos опорной точки к её K ближайшим в опорном наборе без неё самой."""
    out = []
    for s in range(0, len(refn), CHUNK):
        sim = refn[s:s + CHUNK] @ refn.T
        sim[torch.arange(len(sim)), torch.arange(s, s + len(sim))] = -np.inf
        out.append(sim.topk(K, dim=1).values.mean(1))
    return torch.cat(out)


class RunningTopK:
    """Потоковый top-K наибольших cos для каждой опорной точки по поступающим запросам (CSLSb)."""

    def __init__(self, n, K):
        self.best = torch.full((n, K), -np.inf, device=dev)

    def update(self, refn, qn, step=4096):
        for s in range(0, len(refn), step):
            sim = refn[s:s + step] @ qn.T
            both = torch.cat([self.best[s:s + step], sim], dim=1)
            self.best[s:s + step] = both.topk(self.best.shape[1], dim=1).values


def bank_scores(q, qn, bank, k, K):
    """Оценки токенов по банку: dnp, cos, csls, cslsb (все (n,), больше = подозрительнее).
    Одна матрица cos на пачку: из неё top-K (r_T и cos) и обе версии CSLS."""
    out = {s: [] for s in ("dnp", "cos", "csls", "cslsb")}
    for s in range(0, len(q), CHUNK):
        d = torch.cdist(q[s:s + CHUNK], bank["ref"]).topk(k, dim=1, largest=False).values
        out["dnp"].append(d.mean(1))
        sim = qn[s:s + CHUNK] @ bank["refn"].T
        top = sim.topk(max(K, k), dim=1).values
        out["cos"].append(1 - top[:, :k].mean(1))
        r_t = top[:, :K].mean(1, keepdim=True)
        for name, r_s in (("csls", bank["rs"]), ("cslsb", bank["rsb"])):
            out[name].append(1 - (2 * sim - r_t - r_s[None, :]).topk(k, dim=1).values.mean(1))
    return {name: torch.cat(v) for name, v in out.items()}


def hubness_stats(hits, n_queries):
    h = hits.cpu().numpy()
    top = np.sort(h)[::-1][: max(1, len(h) // 100)]
    return {"N": len(h), "queries": n_queries, "mean_hits": h.mean(), "frac_zero": (h == 0).mean(),
            "max_hits": int(h.max()), "p99_hits": float(np.percentile(h, 99)), "skewness": skew(h),
            "top1pct_share": top.sum() / max(h.sum(), 1)}


class KNN:
    def __init__(self, n_ref=100_000, k=3, K=10, n_hold=50):
        self.n_ref, self.k, self.K, self.n_hold = n_ref, k, K, n_hold

    def fit(self, ctx):
        frames = ctx.fit_ids()
        rng = np.random.default_rng(ctx.seed)
        perm = rng.permutation(len(frames))
        n_hold = min(self.n_hold, len(frames) // 2)
        hold = [frames[i] for i in sorted(perm[:n_hold])]
        refs = [frames[i] for i in sorted(perm[n_hold:])]
        cache = routing_cache(FIT_DATASET, FIT_POP)

        # 0. Метки токенов опорных кадров (без B0) → доли сабсэмплинга (с запасом 1.3)
        grids = sorted(set(CELL.values()))
        labels = {c: {i: token_labels(cache["gt"][cache["pos"][i]], c) for i in refs} for c in grids}
        n_ref, quota, rate_a, rate_b = {}, {}, {}, {}
        for c in grids:
            lab = torch.cat(list(labels[c].values()))
            lab = lab[lab >= 0]
            cnt = torch.bincount(lab, minlength=NC).double()
            n_ref[c] = min(self.n_ref, len(lab))
            quota[c] = torch.minimum(cnt, torch.full_like(cnt, n_ref[c] // NC))   # B: равные квоты
            rate_a[c] = min(1.0, 1.3 * n_ref[c] / len(lab))
            rate_b[c] = (1.3 * quota[c] / cnt.clamp(min=1)).clamp(max=1)

        # 1. Кандидаты в опорный набор (опорные кадры) и экстремумы S_P (все кадры P0)
        gen = torch.Generator(device=dev).manual_seed(ctx.seed)
        cand = {(f, v): [] for f in FEATS for v in VARIANTS}
        cand_lab = {(c, v): [] for c in grids for v in VARIANTS}
        self.sp_min, self.sp_max = np.inf, -np.inf
        ref_ids = set(refs)
        for fd in ctx.iterate(frames):
            neglse = fd.model.neg_lse(fd.size)
            self.sp_min = min(self.sp_min, neglse.min().item())
            self.sp_max = max(self.sp_max, neglse.max().item())
            if fd.image_id not in ref_ids:
                continue
            feats = {f: fd.model.tokens(f) for f in FEATS}
            for c in grids:
                lab = labels[c][fd.image_id]
                u = torch.rand(len(lab), generator=gen, device=dev, dtype=torch.float64)
                for v in VARIANTS:
                    r = rate_b[c][lab.clamp(min=0)] if v == "B" else rate_a[c]
                    keep = (lab >= 0) & (u < r)
                    cand_lab[c, v].append(lab[keep])
                    for f in FEATS:
                        if CELL[f] == c:
                            cand[f, v].append(feats[f][keep].cpu())

        # Опорные наборы: A — случайные n, B — квоты по классам
        self.banks, self.ref_counts = {}, {}
        for c in grids:
            for v in VARIANTS:
                lab = torch.cat(cand_lab[c, v]).cpu()
                if v == "A":
                    sel = torch.from_numpy(rng.permutation(len(lab))[: n_ref[c]])
                else:
                    sel = torch.cat([torch.from_numpy(rng.permutation(np.flatnonzero(lab.numpy() == cl))
                                                      [: int(quota[c][cl])]) for cl in range(NC)])
                self.ref_counts[c, v] = np.bincount(lab[sel].numpy(), minlength=NC).tolist()
                for f in FEATS:
                    if CELL[f] == c:
                        ref = torch.cat(cand[f, v])[sel].to(dev)
                        self.banks[f, v] = {"ref": ref, "refn": F.normalize(ref, dim=1)}
        del cand
        for b in self.banks.values():
            b["rs"] = rs_within(b["refn"], self.K)

        # 2. Отложенные кадры P0: max dnp, хабность, r_S по двудольной окрестности (CSLSb)
        self.dnp_max = {key: 0.0 for key in self.banks}
        hits = {(key, m): torch.zeros(len(b["ref"]), device=dev)
                for key, b in self.banks.items() for m in ("l2", "cos")}
        rsb = {key: RunningTopK(len(b["ref"]), self.K) for key, b in self.banks.items()}
        n_queries = {c: 0 for c in grids}
        for fd in ctx.iterate(hold):
            feats = {f: fd.model.tokens(f) for f in FEATS}
            for c in grids:
                n_queries[c] += len(next(feats[f] for f in FEATS if CELL[f] == c))
            for (f, v), b in self.banks.items():
                q = feats[f]
                qn = F.normalize(q, dim=1)
                d, i = topk_chunks(q, b["ref"], self.k, "l2")
                self.dnp_max[f, v] = max(self.dnp_max[f, v], d.mean(1).max().item())
                hits[(f, v), "l2"] += torch.bincount(i.reshape(-1), minlength=len(b["ref"]))
                _, i = topk_chunks(qn, b["refn"], self.k, "cos")
                hits[(f, v), "cos"] += torch.bincount(i.reshape(-1), minlength=len(b["ref"]))
                rsb[f, v].update(b["refn"], qn)
        for key, b in self.banks.items():
            b["rsb"] = rsb[key].best.mean(1)
        self.hubness = pd.DataFrame([{"bank": f"{f}_{v}", "metric": m,
                                      **hubness_stats(hits[(f, v), m], n_queries[CELL[f]])}
                                     for (f, v) in self.banks for m in ("l2", "cos")])
        self.hold_ids, self.n_ref_frames = hold, len(refs)
        print(self.hubness.round(4).to_string(index=False))

    def apply(self, fd):
        shape = (fd.size[1], fd.size[0])
        neglse = fd.model.neg_lse(fd.size)
        sp =(neglse - self.sp_min) / (self.sp_max - self.sp_min)
        maps = {"neglse": neglse}
        for (f, v), b in self.banks.items():
            q = fd.model.tokens(f)
            for s, x in bank_scores(q, F.normalize(q, dim=1), b, self.k, self.K).items():
                maps[f"{f}_{v}_{s}"] = up(x, CELL[f], shape)
            maps[f"{f}_{v}_cdnp"] = maps[f"{f}_{v}_dnp"] / self.dnp_max[f, v] + sp
        cols = {}
        for name, x in maps.items():
            cols[f"{name}_mean"] = comp_mean(x, fd.comp_t, fd.n).cpu().numpy()
            cols[f"{name}_max"] = comp_max(x, fd.comp_t, fd.n).cpu().numpy()
        return cols, maps

    def summary(self):
        return {"hold_frames": self.hold_ids, "n_ref_frames": self.n_ref_frames,
                "ref_sizes": {f"{f}_{v}": len(b["ref"]) for (f, v), b in self.banks.items()},
                "ref_class_counts": {f"1/{c}_{v}": n for (c, v), n in self.ref_counts.items()},
                "dnp_max_hold": {f"{f}_{v}": m for (f, v), m in self.dnp_max.items()},
                "sp_min": self.sp_min, "sp_max": self.sp_max}

    def save(self, out):
        self.hubness.to_csv(out / "hubness.csv", index=False)
