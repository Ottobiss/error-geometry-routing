"""Близость к прототипам классов (Williams et al., 2024, arXiv 2402.17653); B0 заморожена, γ-SSL не обучается.

z — вход classifier, поднятый до рабочего разрешения и L2-нормированный в каждом пикселе.
Подгонка на P0:
  1. прототипы p_c = normalize(Σ z по пикселям GT-класса c), ignore не учитывается;
     A — все пиксели класса (как в статье), B — только где B0 права. Там же — статистики
     Махаланобиса по z для сравнения в попиксельной AUROC: по верным пикселям P0 класс-условные
     mu_c и S_c (нормировка 1/N, без сабсэмплинга), inv(S_c + md_eps·I);
  2. порог — q-квантиль s_pred по всем пикселям P0 (гистограмма, 20000 бинов на [−1, 1]).
Применение: s = z·P (19 косинусов), s_pred (к прототипу класса B0), s_max, margin = s_top1 − s_top2,
флаг argmax s ≠ класс B0; по компонентам для v ∈ {A, B}: v_spred_mean, v_spred_min, v_spred_p10,
v_margin_mean, v_flag_frac, v_low_frac (доля пикселей с s_pred ниже порога; порог — ДОПУЩЕНИЕ,
в статье его нет).
"""
import numpy as np
import torch
import torch.nn.functional as F

from core.components import IGNORE, NC, comp_mean, comp_min, comp_quantile
from core.stats import class_md, inverse, moments

VARIANTS = ("A", "B")
TAU = 0.07          # температура softmax по прототипам в статье
BINS = 20000


def similarities(P, z, pred):
    """P (19, 256), z (256, n), pred (n,) → s_pred, s_max, margin, flag (все (n,))."""
    s = P @ z
    top = s.topk(2, dim=0).values
    s_pred = s.gather(0, pred[None])[0]
    return s_pred, top[0], top[0] - top[1], s.argmax(0) != pred


def component_aggregates(s_pred, margin, flag, comp, n, thr):
    """Агрегаты по компонентам; на входе тензоры (H·W,), comp — тензор long."""
    return {"spred_mean": comp_mean(s_pred, comp, n), "spred_min": comp_min(s_pred, comp, n),
            "spred_p10": torch.from_numpy(comp_quantile(
                s_pred.cpu().numpy(), comp.cpu().numpy(), n, 0.10)),
            "margin_mean": comp_mean(margin, comp, n), "flag_frac": comp_mean(flag.float(), comp, n),
            "low_frac": comp_mean((s_pred < thr).float(), comp, n)}


class Prototypes:
    def __init__(self, q=0.10, md_eps=1e-4):
        self.q, self.md_eps = q, md_eps

    def fit(self, ctx):
        dev = "cuda"
        frames = ctx.fit_ids()
        # 1. Прототипы и статистики Махаланобиса
        sums = n = s = ss = None
        for fd in ctx.iterate(frames):
            z, gt, pred = fd.zn, fd.gt_t, fd.pred_t
            if sums is None:
                dim = z.shape[0]
                sums = {v: torch.zeros(NC, dim, dtype=torch.float64, device=dev) for v in VARIANTS}
                n = torch.zeros(NC, dtype=torch.float64, device=dev)
                s = torch.zeros(NC, dim, dtype=torch.float64, device=dev)
                ss = torch.zeros(NC, dim, dim, dtype=torch.float64, device=dev)
            valid = gt != IGNORE
            ok = valid & (pred == gt)
            # внутри кадра — fp32, накопление по кадрам — float64
            for v, m in (("A", valid), ("B", ok)):
                sums[v] += torch.zeros(NC, dim, device=dev).index_add_(0, gt[m], z[:, m].T).double()
            z_ok, g_ok = z[:, ok], gt[ok]
            for c in g_ok.unique().tolist():
                zc = z_ok[:, g_ok == c].T
                n[c] += len(zc)
                s[c] += zc.sum(0).double()
                ss[c] += (zc.T @ zc).double()
        # в смоук-тесте части классов в P0 нет — у них нулевой прототип и mu = 0
        assert ctx.limit is not None or (n > 1).all(), f"мало верных пикселей P0: {n.tolist()}"
        self.P = {v: F.normalize(sums[v], dim=1).float() for v in VARIANTS}
        mu, cov = moments(n.clamp(min=1), s, ss)
        self.md_inv = inverse(cov, self.md_eps).float()
        self.mu = mu.float()

        # 2. Порог: q-квантиль s_pred по всем пикселям P0
        hist = {v: torch.zeros(BINS, dtype=torch.float64, device=dev) for v in VARIANTS}
        for fd in ctx.iterate(frames):
            for v in VARIANTS:
                s_pred = similarities(self.P[v], fd.zn, fd.pred_t)[0]
                hist[v] += torch.histc(s_pred, BINS, -1, 1).double()
        edges = np.linspace(-1, 1, BINS + 1)
        self.thr = {v: float(edges[1:][np.searchsorted(
            np.cumsum(hist[v].cpu().numpy()) / hist[v].sum().item(), self.q)]) for v in VARIANTS}

    def apply(self, fd):
        z, pred = fd.zn, fd.pred_t
        cols, pixel = {}, {}
        for v in VARIANTS:
            s_pred, s_max, margin, flag = similarities(self.P[v], z, pred)
            for name, x in component_aggregates(s_pred, margin, flag, fd.comp_t, fd.n,
                                                self.thr[v]).items():
                cols[f"{v}_{name}"] = x.cpu().numpy()
            pixel[f"{v}: −s_pred"], pixel[f"{v}: −s_max"], pixel[f"{v}: −margin"] = -s_pred, -s_max, -margin
        if not fd.test:
            return cols, {}
        pixel["MD z (класс B0)"] = class_md(z.T, pred, self.mu, self.md_inv)
        pixel["proto-softmax τ=0.07 (статья)"] = 1 - ((self.P["A"] @ z) / TAU).softmax(0).max(0).values
        return cols, pixel

    def summary(self):
        return {"thr": self.thr}

    def save(self, out):
        np.savez(out / "prototypes.npz", P_A=self.P["A"].cpu().numpy(), P_B=self.P["B"].cpu().numpy(),
                 thr_A=self.thr["A"], thr_B=self.thr["B"], q=self.q)
