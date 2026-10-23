"""Плотностные методы: Махаланобис по выходу B0, Махаланобис по векторам компонент, DDU.

md_output — Махаланобис по выходу модели (Henriksson et al., 2024, arXiv 2401.17013, Sec. III):
  o — softmax-выход в пикселе; по верно классифицированным пикселям train для каждого класса c
  mu_c = (1/N_c) Σ o, S_c = (1/N_c) Σ (o − mu_c)(o − mu_c)ᵀ; до 10⁶ пикселей на класс, не более
  10⁴ с кадра на класс; в пикселе с предсказанным классом c = argmax o: MD = sqrt((o − mu_c)ᵀ S_c⁻¹ (o − mu_c)).
  Отступления: fit на P0 (CS val), а не на CS train; ДОПУЩЕНИЕ — статья не говорит, как обращать S_c,
  а у softmax она вырождена (Σ o = 1): eps = 0 — псевдообратная, eps > 0 — inv(S_c + eps·I);
  кроме softmax — варианты log_softmax и logits (в статье их нет).
  Логиты — results/<dataset>/logits_b0.npy (features.logits), класс пикселя — маска pred_b0.
"""
import numpy as np
import torch

from core.components import IGNORE, NC, comp_quantile
from core.evaluation import routing_cache
from core.stats import class_md, inverse
from features.logits import LogitStore
from routing.methods import FeatureColumn, register

VARIANTS = ("softmax", "log_softmax", "logits")


def transform(logits, variant):
    """Вектор выхода из логитов; ось классов — 0 для карт (19, H, W) и −1 для строк (n, 19)."""
    dim = 0 if logits.ndim == 3 else -1
    if variant == "softmax":
        return logits.softmax(dim)
    if variant == "log_softmax":
        return logits.log_softmax(dim)
    assert variant == "logits", variant
    return logits


def sample_correct(logits, pred, gt, rng, per_image=10_000):
    """Логиты верно распознанных пикселей кадра (pred == gt, без ignore), до per_image на класс.
    Возвращает {c: (n_c, 19) float16 numpy}."""
    flat = logits.reshape(NC, -1)
    ok = (pred == gt) & (gt != IGNORE)
    out = {}
    for c in np.unique(pred[ok]):
        idx = np.flatnonzero(ok & (pred == c))
        if len(idx) > per_image:
            idx = rng.choice(idx, per_image, replace=False)
        out[int(c)] = flat[:, torch.from_numpy(idx).to(flat.device)].T.half().cpu().numpy()
    return out


def merge_samples(per_image, rng, per_class=1_000_000):
    """Склейка выборок по кадрам и сабсэмплинг до per_class на класс."""
    out = {}
    for c in range(NC):
        parts = [s[c] for s in per_image if c in s]
        if not parts:
            continue
        x = np.concatenate(parts)
        if len(x) > per_class:
            x = x[rng.choice(len(x), per_class, replace=False)]
        out[c] = x
    return out


def fit_samples(cache, logits, seed=0, per_image=10_000, per_class=1_000_000):
    """Выборка верных пикселей по кадрам кэша популяции (gt, b0) и логитам LogitStore."""
    rng = np.random.default_rng(seed)
    per = [sample_correct(logits[i], cache["b0"][k], cache["gt"][k], rng, per_image)
           for k, i in enumerate(cache["image_id"])]
    return merge_samples(per, rng, per_class)


class OutputStats:
    """Статистики MD по классам: fit по выборке логитов, md / md_map для оценки."""

    def __init__(self, variant="softmax", eps=0.0, device="cuda"):
        self.variant, self.eps, self.device = variant, eps, device

    def fit(self, samples, min_pixels=100):
        """samples {c: (n_c, 19) логиты} → mu (19, 19), inv (19, 19, 19); ковариация 1/N_c, как в статье."""
        missing = [c for c in range(NC) if len(samples.get(c, ())) < min_pixels]
        assert not missing, f"мало верных пикселей для классов {missing}"
        self.counts = {c: len(samples[c]) for c in range(NC)}
        mu, inv = [], []
        for c in range(NC):
            v = transform(torch.from_numpy(samples[c]).double(), self.variant)
            mu.append(v.mean(0))
            inv.append(inverse(torch.cov(v.T, correction=0), self.eps))
        self.mu = torch.stack(mu).float().to(self.device)
        self.inv = torch.stack(inv).float().to(self.device)
        return self

    def md(self, v, cls):
        """v (n, 19), cls (n,) → MD (n,), всё на device."""
        return class_md(v, cls, self.mu, self.inv)

    def md_map(self, logits, pred):
        """logits (19, H, W) на device, pred (H, W) → карта MD (H, W) float32 numpy."""
        v = transform(logits, self.variant).reshape(NC, -1).T
        cls = torch.from_numpy(np.asarray(pred, dtype=np.int64).ravel()).to(v.device)
        return self.md(v, cls).reshape(pred.shape).cpu().numpy()

    def quantiles(self, samples, q):
        """q-квантиль MD на выборке fit для каждого класса (порог для доли пикселей выше порога)."""
        thr = np.zeros(NC)
        for c in range(NC):
            v = transform(torch.from_numpy(samples[c]).float().to(self.device), self.variant)
            cls = torch.full((len(v),), c, device=self.device)
            thr[c] = np.quantile(self.md(v, cls).cpu().numpy(), q)
        return thr


def aggregate(md, comp, n, how, thr=None):
    """Агрегация карты MD по компонентам: mean, p90 (ближайший ранг снизу) или frac > thr."""
    c, x = comp.ravel(), md.ravel()
    k = np.bincount(c, minlength=n)
    if how == "mean":
        return np.bincount(c, weights=x, minlength=n) / k
    if how == "p90":
        return comp_quantile(x, c, n, 0.9)
    assert how == "frac", how
    return np.bincount(c, weights=(x > thr.ravel()), minlength=n) / k


@register("md_output")
class MahalanobisOutput:
    """MD по выходу B0, агрегированный по компоненте. Метки train не читает.

    variant — softmax (статья) / log_softmax / logits; eps — 0 = псевдообратная, > 0 = inv(S + eps·I);
    agg — mean / p90 / frac; q — квантиль MD на fit-пикселях класса — порог для frac
    (ДОПУЩЕНИЕ: в статье агрегации по регионам нет).
    """

    def __init__(self, variant="softmax", eps=0.0, agg="mean", q=0.95, seed=0):
        self.variant, self.agg, self.q, self.seed = variant, agg, q, seed
        self.stats = OutputStats(variant, eps)

    def fit(self, train):
        cache = routing_cache(train.dataset, train.population, train.root)
        samples = fit_samples(cache, LogitStore(train.dataset), self.seed)
        self.stats.fit(samples)
        self.thr = self.stats.quantiles(samples, self.q) if self.agg == "frac" else None
        return self

    def score(self, test):
        cache = routing_cache(test.dataset, test.population, test.root)
        logits = LogitStore(test.dataset)
        out = np.zeros(len(test.table))
        for image_id, idx in test.table.groupby("image_id").indices.items():
            k = cache["pos"][image_id]
            pred, comp = cache["b0"][k], cache["comp"][k].astype(np.int64)
            md = self.stats.md_map(logits[image_id], pred)
            thr = self.thr[pred] if self.thr is not None else None
            out[idx] = aggregate(md, comp, len(idx), self.agg, thr)
        return out


@register("mahalanobis")
class MahalanobisEmbedding:
    """Махаланобис по классу на векторах декодера B0 (256-d среднее по компоненте).

    fit: по верным компонентам P0 (y_fail = 0) центры классов mu_c; ковариация общая (shared: по
    отклонениям от центра своего класса) или поклассовая (class); нормировка 1/N; eps — как в md_output.
    score: MD до центра предсказанного класса компоненты. ДОПУЩЕНИЕ: компоненты не взвешиваются по площади.
    """

    def __init__(self, cov="shared", eps=1e-4, correct_only=True):
        assert cov in ("shared", "class"), cov
        self.cov, self.eps, self.correct_only = cov, eps, correct_only

    def fit(self, train):
        t = train.table
        keep = (t.y_fail.values == 0) if self.correct_only else np.ones(len(t), bool)
        x = torch.from_numpy(train.emb[keep]).double()
        cls = t.class_id.values[keep]
        counts = np.bincount(cls, minlength=NC)
        assert (counts > 1).all(), f"мало компонент для fit: {counts}"
        self.mu = torch.stack([x[cls == c].mean(0) for c in range(NC)])
        d = x - self.mu[cls]
        if self.cov == "shared":
            inv = inverse(d.T @ d / len(d), self.eps)
            self.inv = inv.expand(NC, *inv.shape)
        else:
            self.inv = torch.stack([inverse(d[cls == c].T @ d[cls == c] / (cls == c).sum(), self.eps)
                                    for c in range(NC)])
        return self

    def score(self, test):
        x = torch.from_numpy(test.emb).double()
        cls = torch.tensor(test.table.class_id.values)
        return class_md(x, cls, self.mu, self.inv).numpy()


@register("ddu")
class DDU(FeatureColumn):
    """DDU-признак компоненты (features.ddu: GDA на z, fit на P0). Ограничение: B0 без спектральной нормировки.
    feature — logp_mean / logp_p10 (минус: меньше плотность = больше выигрыш) или mdpred_p90 (плюс)."""
    table = "ddu"
    SIGN = {"logp_mean": -1, "logp_p10": -1, "mdpred_p90": 1}

    def __init__(self, feature="logp_mean"):
        super().__init__(feature, self.SIGN[feature])
