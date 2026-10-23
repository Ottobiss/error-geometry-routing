"""DDU-признаки (Mukhoti et al., 2021, arXiv 2111.00079) для замороженной B0.

z — вход classifier, поднятый до рабочего разрешения, без нормировки. GDA: p(z) = Σ_c N(z; mu_c,
Σ_c + eps·I) p(c). Ограничение: в DDU сеть обучается со спектральной нормировкой, B0 — без неё.
Подгонка на P0:
  0. prior p(c) и доли сабсэмплинга — по счётчикам GT-пикселей классов P0;
  1. пиксель класса c (по GT, без ignore) попадает в fit с вероятностью min(1, per_class / N_c) —
     в среднем до per_class на класс, равномерно по P0. Достаточные статистики копятся в float64
     отдельно по двум половинам P0 (чётные / нечётные кадры); отдельно — отложенные пиксели
     (до held_per_class на класс) для выбора eps;
  2. eps из EPS_GRID: GDA на одной половине, средний log p(z) отложенных пикселей другой и
     наоборот; берётся eps с наибольшим средним, финальная GDA — на всём P0.
Применение: log p(z), MD_pred (до класса B0), MD_min; по компонентам logp_mean, logp_p10, mdpred_p90.
"""
import numpy as np
import pandas as pd
import torch

from core.components import IGNORE, NC, comp_mean, comp_quantile
from core.data import FIT_DATASET, FIT_POP
from core.evaluation import routing_cache
from core.stats import moments

EPS_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]


class GDA:
    """Класс-условные гауссианы: p(z) = Σ_c N(z; mu_c, Σ_c) p(c).

    Строится из достаточных статистик в float64: n (C,), s (C, D) = Σ z, ss (C, D, D) = Σ z zᵀ;
    Σ_c = ss/n − mu muᵀ + eps·I (нормировка 1/N). Скоринг — в float32 через L_c⁻¹ (Холецкий).
    Если Σ_c + eps·I не положительно определена, бросает torch.linalg.LinAlgError.
    """

    def __init__(self, n, s, ss, prior, eps):
        d = s.shape[1]
        mu, cov = moments(n, s, ss)
        cov = cov + eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov)
        eye = torch.eye(d, dtype=cov.dtype, device=cov.device).expand_as(L)
        self.Linv = torch.linalg.solve_triangular(L, eye, upper=False).float()
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(1)
        self.const = (-0.5 * (d * np.log(2 * np.pi) + logdet) + torch.log(prior)).float()
        self.mu, self.cov = mu.float(), cov

    def md2(self, z):
        """z (n, D) float32 → квадраты расстояний Махаланобиса до всех классов (n, C)."""
        out = torch.empty(len(z), len(self.mu), device=z.device)
        for c in range(len(self.mu)):
            out[:, c] = (((z - self.mu[c]) @ self.Linv[c].T) ** 2).sum(1)
        return out

    def scores(self, z, pred):
        """z (n, D), pred (n,) → log p(z), MD до класса pred, MD до ближайшего класса."""
        md2 = self.md2(z)
        logp = torch.logsumexp(self.const - 0.5 * md2, dim=1)
        md_pred = md2.gather(1, pred[:, None])[:, 0].sqrt()
        return logp, md_pred, md2.min(1).values.sqrt()

    def mean_logp(self, z, chunk=200_000):
        return torch.cat([torch.logsumexp(self.const - 0.5 * self.md2(c), dim=1)
                          for c in z.split(chunk)]).mean().item()


def _empty_stats(dim, dev):
    return {"n": torch.zeros(NC, dtype=torch.float64, device=dev),
            "s": torch.zeros(NC, dim, dtype=torch.float64, device=dev),
            "ss": torch.zeros(NC, dim, dim, dtype=torch.float64, device=dev)}


class DDU:
    def __init__(self, per_class=200_000, held_per_class=10_000):
        self.per_class, self.held_per_class = per_class, held_per_class

    def fit(self, ctx):
        dev = "cuda"
        frames = ctx.fit_ids()
        # 0. Счётчики GT-классов на P0 → prior и доли сабсэмплинга
        cache = routing_cache(FIT_DATASET, FIT_POP)
        counts = np.zeros(NC)
        for image_id in frames:
            g = cache["gt"][cache["pos"][image_id]]
            counts += np.bincount(g[g != IGNORE], minlength=NC)[:NC]
        assert ctx.limit is not None or (counts > 0).all(), counts
        self.prior = torch.tensor(np.maximum(counts, 1) / counts.sum(), dtype=torch.float64, device=dev)
        rate = torch.tensor(np.minimum(1, self.per_class / np.maximum(counts, 1)), device=dev)
        held_rate = torch.tensor(np.minimum(1, self.held_per_class / np.maximum(counts, 1)), device=dev)

        # 1. Статистики по половинам P0 и отложенные пиксели
        gen = torch.Generator(device=dev).manual_seed(ctx.seed)
        stats = None
        held = [[[] for _ in range(NC)], [[] for _ in range(NC)]]
        for k, fd in enumerate(ctx.iterate(frames)):
            z, gt = fd.z, fd.gt_t
            if stats is None:
                stats = [_empty_stats(z.shape[0], dev), _empty_stats(z.shape[0], dev)]
            valid = gt != IGNORE
            u = torch.rand(gt.shape, generator=gen, device=dev)
            u2 = torch.rand(gt.shape, generator=gen, device=dev)
            g = gt.clamp(max=NC - 1)
            fit_m = valid & (u < rate[g])
            held_m = valid & (u2 < held_rate[g]) & ~fit_m
            z_fit, g_fit, z_held, g_held = z[:, fit_m], gt[fit_m], z[:, held_m], gt[held_m]
            st = stats[k % 2]
            for c in gt[valid].unique().tolist():
                zc = z_fit[:, g_fit == c].T.double()
                st["n"][c] += len(zc)
                st["s"][c] += zc.sum(0)
                st["ss"][c] += zc.T @ zc
                held[k % 2][c].append(z_held[:, g_held == c].T.cpu())
        held = [torch.cat([torch.cat(x) for x in hh if x]).cuda() for hh in held]
        self.n_fit = (stats[0]["n"] + stats[1]["n"]).cpu().numpy()

        # 2. Выбор eps по отложенным пикселям другой половины
        rows = []
        for eps in EPS_GRID:
            try:
                ll = [GDA(stats[a]["n"].clamp(min=1), stats[a]["s"], stats[a]["ss"], self.prior, eps)
                      .mean_logp(held[1 - a]) for a in (0, 1)]
                rows.append({"eps": eps, "heldout_logp": float(np.mean(ll)), "ok": True})
            except torch.linalg.LinAlgError:
                rows.append({"eps": eps, "heldout_logp": np.nan, "ok": False})
        self.selection = pd.DataFrame(rows)
        self.eps = float(self.selection.loc[self.selection.heldout_logp.idxmax(), "eps"])
        full = {k: stats[0][k] + stats[1][k] for k in stats[0]}
        self.gda = GDA(full["n"].clamp(min=1), full["s"], full["ss"], self.prior, self.eps)
        print(self.selection.to_string(index=False), f"\neps = {self.eps}")

    def apply(self, fd):
        logp, md_pred, md_min = self.gda.scores(fd.z.T.contiguous(), fd.pred_t)
        comp = fd.components[0].ravel().astype(np.int64)
        cols = {"logp_mean": comp_mean(logp, fd.comp_t, fd.n).cpu().numpy(),
                "logp_p10": comp_quantile(logp.cpu().numpy().astype(np.float64), comp, fd.n, 0.10),
                "mdpred_p90": comp_quantile(md_pred.cpu().numpy().astype(np.float64), comp, fd.n, 0.90)}
        return cols, {"−log p(z)": -logp, "MD_pred": md_pred, "MD_min": md_min}

    def summary(self):
        return {"eps": self.eps, "eps_grid": EPS_GRID, "n_fit": self.n_fit.tolist(),
                "note": "B0 без спектральной нормировки (в DDU — со SN)"}

    def save(self, out):
        np.savez(out / "gda.npz", mu=self.gda.mu.cpu().numpy(), cov=self.gda.cov.cpu().numpy(),
                 prior=self.prior.cpu().numpy(), eps=self.eps, n_fit=self.n_fit)
        self.selection.to_csv(out / "eps_selection.csv", index=False)
