"""Оценка: метрики провала по Kang et al. (S1), роутинг по выигрышу Δ и попиксельные метрики."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from core.components import IGNORE, NC, extract_components
from core.data import get_dataset, population
from core.paths import RESULTS, DatasetResults


# --- Метрики провала (Kang et al., S1) ---

def image_macro_ce(df, y, prob, eps=1e-6):
    """Кросс-энтропия: среднее внутри кадра, затем среднее по кадрам (nats)."""
    q = np.clip(prob, eps, 1 - eps)
    ll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    return pd.Series(ll, index=df.index).groupby(df.image_id).mean().mean()


def retained_mean(df, risk, value, keep=0.9):
    """Отбор с бюджетом на изображение: в кадре с n_j компонентами оставляем k_j = ⌈keep·n_j⌉
    с наименьшим риском (по убыванию риска отбрасываются первые n_j − k_j, ничьи —
    по возрастанию comp_id). Возвращает среднее value по оставленным, image-macro.
    """
    d = pd.DataFrame({"image_id": df.image_id.values, "comp_id": df.comp_id.values,
                      "risk": np.asarray(risk), "value": np.asarray(value, dtype=float)})
    d = d.sort_values(["image_id", "risk", "comp_id"], ascending=[True, False, True])
    rank = d.groupby("image_id").cumcount()
    n = d.groupby("image_id").image_id.transform("size")
    keep_mask = rank >= n - np.ceil(keep * n)
    return d[keep_mask].groupby("image_id").value.mean().mean()


# --- Роутинг по выигрышу Δ ---
#
# Компоненты кадра сортируются по оценке метода и отдаются B2, пока их площадь укладывается
# в бюджет (доля пикселей кадра); итоговая маска — B2 в отданных компонентах, B0 в остальных.

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
BOOT_BUDGETS = [0.10, 0.20]


def routing_cache(dataset, pop, root=RESULTS):
    """Компоненты B0, GT и маски B0/B2 популяции; строится один раз и кладётся в cache_<pop>.npz,
    в процессе загружается один раз (не изменять возвращённые массивы).

    Возвращает dict: image_id (n,), comp int16, gt / b0 / b2 uint8 — все (n, H, W);
    pos — {image_id: номер кадра в массивах}.
    """
    return _routing_cache(dataset, pop, Path(root))


@lru_cache(maxsize=4)
def _routing_cache(dataset, pop, root):
    res = DatasetResults(dataset, root)
    path = res.cache(pop)
    image_ids = population(dataset, pop)
    if path.exists():
        z = dict(np.load(path))
        assert list(z["image_id"]) == image_ids, "кэш построен для других кадров"
        z["pos"] = {i: k for k, i in enumerate(image_ids)}
        return z
    ds = get_dataset(dataset, image_ids)
    out = {"comp": [], "gt": [], "b0": [], "b2": []}
    for i, image_id in enumerate(ds.image_ids):
        masks = np.load(res.mask(image_id))
        comp, _ = extract_components(masks["pred_b0"].astype(np.int64))
        out["comp"].append(comp.astype(np.int16))
        out["gt"].append(ds.label(i))
        out["b0"].append(masks["pred_b0"])
        out["b2"].append(masks["pred_b2"])
    z = {k: np.stack(v) for k, v in out.items()}
    z["image_id"] = np.asarray(image_ids, dtype=str)
    np.savez_compressed(path, **z)
    z["pos"] = {i: k for k, i in enumerate(image_ids)}
    return z


def route(area, score, budget, hw):
    """Номера компонент кадра (comp_id), отданных B2 при бюджете budget·hw пикселей; area, score — (n,).

    Жадно по убыванию score (ничьи — по номеру): компонента, которая не помещается в остаток
    бюджета, пропускается, отбор идёт дальше. score = −inf — компонента не отдаётся B2 никогда
    (так оракул отдаёт только компоненты с Δ > 0: бюджет — предел, а не обязательство).
    """
    left, chosen = budget * hw, []
    for i in np.argsort(-score, kind="stable"):
        if score[i] == -np.inf:
            break                       # дальше по порядку — только −inf
        if area[i] <= left:
            chosen.append(i)
            left -= area[i]
    return np.asarray(chosen, dtype=np.int64)


def oracle_score(table):
    """Оценка оракула: истинная Δ на пиксель, компоненты с Δ ≤ 0 — −inf. Верхняя граница, не метод."""
    d = table.density.values
    return np.where(d > 0, d, -np.inf)


def _conf(g, p):
    return np.bincount(g * NC + p, minlength=NC * NC).reshape(NC, NC)


def _miou(conf):
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf))
    return 100 * np.nanmean(iou)


def _miou_batch(conf):
    """mIoU (в %) для пачки матриц ошибок (B, NC, NC); классы без пикселей не учитываются."""
    diag = np.diagonal(conf, axis1=1, axis2=2).astype(float)
    union = conf.sum(1) + conf.sum(2) - diag
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * np.nanmean(np.where(union > 0, diag / union, np.nan), axis=1)


def evaluate_routing(table, scores, cache, references=("entropy",), budgets=BUDGETS,
                     boot_budgets=BOOT_BUDGETS, n_boot=2000, seed=0):
    """Роутинг всех компонент table по каждой оценке из scores {имя: (N,)}.

    Для каждого метода: Spearman оценки с Δ/пиксель (компоненты с меткой), mIoU итоговой маски
    и доля выигрыша оракула при каждом бюджете. Для каждой опорной строки из references, которая
    есть в scores, — парный bootstrap по кадрам (n_boot повторов, сид seed; ресэмплы одни и те же
    для всех опорных): разница mIoU «метод − опорная» при boot_budgets, точечная оценка и
    95%-интервал. Колонки: Δ<ref>@b и для entropy ci_lo@b / ci_hi@b, для остальных опорных —
    ci_lo[<ref>]@b / ci_hi[<ref>]@b.
    Возвращает (DataFrame строк-методов, {"B0": mIoU, "B2": mIoU}).
    """
    scores = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    names = list(scores) + ["_oracle"]
    all_scores = {**scores, "_oracle": oracle_score(table)}
    conf = {(n, b): np.zeros((NC, NC), np.int64) for n in names for b in budgets}
    gain = {(n, b): 0 for n in names for b in budgets}
    per_image = {(n, b): [] for n in scores for b in boot_budgets}   # матрицы ошибок по кадрам
    conf_b0, conf_b2 = np.zeros((NC, NC), np.int64), np.zeros((NC, NC), np.int64)
    comp_id, area, delta = table.comp_id.values, table.area.values, table.delta.values
    for image_id, idx in table.groupby("image_id").indices.items():
        k = cache["pos"][image_id]
        gt = cache["gt"][k]
        valid = gt != IGNORE
        g = gt[valid].astype(np.int64)
        cv = cache["comp"][k][valid].astype(np.int64)
        b0v, b2v = cache["b0"][k][valid].astype(np.int64), cache["b2"][k][valid].astype(np.int64)
        assert (comp_id[idx] == np.arange(len(idx))).all() and cache["comp"][k].max() + 1 == len(idx)
        # Вклад каждой компоненты в матрицу ошибок при замене B0 на B2: (n_comp, NC, NC).
        m = len(idx) * NC * NC
        d = (np.bincount(cv * NC * NC + g * NC + b2v, minlength=m)
             - np.bincount(cv * NC * NC + g * NC + b0v, minlength=m)).reshape(len(idx), NC, NC)
        c0 = _conf(g, b0v)
        conf_b0 += c0
        conf_b2 += c0 + d.sum(0)
        for n in names:
            for b in budgets:
                chosen = route(area[idx], all_scores[n][idx], b, gt.size)
                c = c0 + d[chosen].sum(0)
                conf[n, b] += c
                gain[n, b] += delta[idx][chosen].sum()
                if (n, b) in per_image:
                    per_image[n, b].append(c)

    lab = table.has_valid.values
    rows = []
    for n in scores:
        row = {"метод": n, "spearman": spearmanr(scores[n][lab], table.density.values[lab]).statistic}
        for b in budgets:
            row[f"mIoU@{round(b * 100)}%"] = _miou(conf[n, b])
        for b in budgets:
            row[f"выигр@{round(b * 100)}%"] = gain[n, b] / gain["_oracle", b]
        rows.append(row)

    for reference in [r for r in references if r in scores]:
        n_img = len(per_image[reference, boot_budgets[0]])
        rng = np.random.default_rng(seed)
        w = rng.multinomial(n_img, np.full(n_img, 1 / n_img), size=n_boot)   # (n_boot, n_img)
        tag = "" if reference == "entropy" else f"[{reference}]"
        for b in boot_budgets:
            ref = np.stack(per_image[reference, b]).reshape(n_img, -1)
            ref_boot = _miou_batch((w @ ref).reshape(n_boot, NC, NC))
            for row in rows:
                m = np.stack(per_image[row["метод"], b]).reshape(n_img, -1)
                diff = _miou_batch((w @ m).reshape(n_boot, NC, NC)) - ref_boot
                key = f"{round(b * 100)}%"
                row[f"Δ{reference}@{key}"] = row[f"mIoU@{key}"] - _miou(ref.sum(0).reshape(NC, NC))
                row[f"ci_lo{tag}@{key}"], row[f"ci_hi{tag}@{key}"] = np.percentile(diff, [2.5, 97.5])
    return pd.DataFrame(rows), {"B0": _miou(conf_b0), "B2": _miou(conf_b2)}


# --- Попиксельная оценка: AUROC «пиксель B0 ошибочный» ---

def msp_score(probs, dim=0):
    """1 − max softmax по оси классов dim; probs (19, n) при dim=0, (n, 19) при dim=1."""
    return 1 - probs.max(dim).values


def entropy_score(probs, dim=0):
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim)


class PixelSampler:
    """Одни и те же случайные пиксели для всех оценок: доля frac размеченных пикселей кадра.

    Кадры нужно подавать в порядке сортировки image_id — тогда выборка совпадает между
    скриптами (генератор один на весь проход).
        sel = sampler.frame(domain, gt, pred)   # индексы пикселей кадра (H·W), метка ошибки B0
        sampler.add(имя, оценки[sel])           # больше = подозрительнее
    """

    def __init__(self, frac=0.05, seed=0):
        self.frac, self.rng = frac, np.random.default_rng(seed)
        self.cols = {}

    def frame(self, domain, gt, pred):
        g, p = gt.ravel(), pred.ravel()
        valid = np.flatnonzero(g != IGNORE)
        sel = np.sort(self.rng.choice(valid, int(round(self.frac * len(valid))), replace=False))
        self.add("_err", p[sel] != g[sel])
        self.add("_domain", np.full(len(sel), domain))
        return sel

    def add(self, name, values):
        self.cols.setdefault(name, []).append(np.asarray(values))

    def auroc_table(self):
        """AUROC каждой оценки по каждому домену (и по всем вместе, если доменов несколько)."""
        cols = {k: np.concatenate(v) for k, v in self.cols.items()}
        err, domain = cols.pop("_err"), cols.pop("_domain")
        groups = {d: domain == d for d in np.unique(domain)}
        if len(groups) > 1:
            groups["all"] = np.ones(len(err), bool)
        rows = [{"domain": d, "score": k, "auroc": roc_auc_score(err[m], s[m]),
                 "n_pixels": int(m.sum()), "error_rate": float(err[m].mean())}
                for d, m in groups.items() for k, s in cols.items()]
        return pd.DataFrame(rows)
