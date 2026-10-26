"""Компоненты предсказанной карты, метки и признаки по Kang et al. (arXiv 2610.16058), S1–S2,
и агрегация попиксельных карт по компонентам.

Компонента — 8-связный регион одного предсказанного класса. Все функции работают
с одним изображением: pred (H, W) int, probs (C, H, W) float — softmax лёгкой модели.

Места, где статья неоднозначна, помечены «ДОПУЩЕНИЕ».
"""
import numpy as np
import torch
from scipy import ndimage

NC = 19                        # классы Cityscapes (trainIds)
IGNORE = 255
EIGHT = np.ones((3, 3), dtype=bool)

# Поля базового предиктора p (таблица S2): 18 числовых + one-hot класса
P_NUMERIC = [
    "area", "boundary_size", "interior_size", "mean_max_probability", "mean_entropy",
    "mean_probability_margin", "mean_predicted_class_probability", "mean_entropy_boundary",
    "mean_entropy_interior", "mean_max_probability_boundary", "mean_max_probability_interior",
    "mean_probability_margin_boundary", "mean_probability_margin_interior",
    "relative_component_area", "relative_boundary_size", "relative_interior_size",
    "touches_image_boundary", "has_interior",
]
P_FIELDS = P_NUMERIC + [f"class_{c}" for c in range(NC)]
# Поля адаптированного MetaSeg (S2.6): 15 дескрипторов + 19 средних вероятностей классов
M_FIELDS = ([f"ms_{s}_{t}" for s in ("E", "D") for t in ("mean", "in", "bd", "rel", "rel_in")]
            + ["ms_S", "ms_S_in", "ms_S_bd", "ms_S_rel", "ms_S_rel_in"]
            + [f"ms_prob_mean_{c}" for c in range(NC)])


def extract_components(pred):
    """Метки компонент (H, W) с id 0..n-1 и класс каждой компоненты (n,)."""
    comp = np.full(pred.shape, -1, dtype=np.int32)
    classes = []
    n = 0
    for c in np.unique(pred):
        lab, k = ndimage.label(pred == c, structure=EIGHT)
        mask = lab > 0
        comp[mask] = lab[mask] - 1 + n
        classes += [c] * k
        n += k
    return comp, np.asarray(classes, dtype=np.int64)


def _shifted_differs(comp, offsets, edge_value):
    """True, если хотя бы один сосед по offsets принадлежит другой компоненте.

    Сосед за краем изображения даёт edge_value.
    """
    h, w = comp.shape
    pad = np.pad(comp, 1, constant_values=-2)
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in offsets:
        nb = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        outside = nb == -2
        out |= np.where(outside, edge_value, nb != comp)
    return out


FOUR = [(-1, 0), (1, 0), (0, -1), (0, 1)]
EIGHT_OFFS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def boundary_p(comp):
    """S2.1: граница — пиксель на краю изображения или с 4-соседом вне компоненты."""
    return _shifted_differs(comp, FOUR, edge_value=True)


def boundary_metaseg(comp):
    """S2.6: граница — хотя бы один из 8 соседей внутри изображения из другой компоненты."""
    return _shifted_differs(comp, EIGHT_OFFS, edge_value=False)


def pixel_maps(probs):
    """Попиксельные карты из softmax: max-prob, энтропия, маржа (top1 − top2)."""
    top2 = np.partition(probs, -2, axis=0)[-2:]
    p1, p2 = top2[1], top2[0]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(0)
    return {"max_prob": p1, "entropy": entropy, "margin": p1 - p2}


def _means(comp, n, values, mask=None):
    """Среднее values по каждой компоненте (по пикселям mask); NaN для пустых."""
    idx = comp.ravel() if mask is None else comp[mask]
    val = values.ravel() if mask is None else values[mask]
    s = np.bincount(idx, weights=val, minlength=n)
    k = np.bincount(idx, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / k, k


def failure_labels(comp, classes, gt):
    """S1: Y = 1, если у компоненты нулевой IoU со всеми GT-компонентами того же класса.

    IoU > 0 тогда и только тогда, когда хотя бы один валидный пиксель компоненты
    имеет в GT тот же класс, поэтому GT-компоненты строить не нужно.
    Возвращает (y, valid): valid = False, если у компоненты нет валидных пикселей.
    """
    n = len(classes)
    valid_px = gt != IGNORE
    n_valid = np.bincount(comp[valid_px], minlength=n)
    hit = valid_px & (gt == classes[comp])
    n_hit = np.bincount(comp[hit], minlength=n)
    return (n_hit == 0).astype(np.int8), n_valid > 0


def delta_labels(comp, n, gt, pred_light, pred_heavy):
    """Наша метка: Δ = #(права тяжёлая, лёгкая нет) − #(права лёгкая, тяжёлая нет).

    Считается по валидным пикселям компоненты лёгкой модели. Возвращает Δ и число
    валидных пикселей (для нормированной версии Δ / n_valid).
    """
    valid_px = gt != IGNORE
    ok_l = pred_light == gt
    ok_h = pred_heavy == gt
    gain = (valid_px & ok_h & ~ok_l).astype(np.int64) - (valid_px & ok_l & ~ok_h)
    delta = np.bincount(comp.ravel(), weights=gain.ravel(), minlength=n)
    n_valid = np.bincount(comp[valid_px], minlength=n)
    return delta.astype(np.int64), n_valid


def base_fields(comp, classes, maps, num_classes=NC):
    """37 полей базового предиктора p (таблица S2): 18 числовых + one-hot класса."""
    n = len(classes)
    h, w = comp.shape
    bd = boundary_p(comp)
    inner = ~bd
    area = np.bincount(comp.ravel(), minlength=n)
    f = {}
    f["area"] = area
    f["boundary_size"] = np.bincount(comp[bd], minlength=n)
    f["interior_size"] = np.bincount(comp[inner], minlength=n)
    for key, name in [("max_prob", "max_probability"), ("entropy", "entropy"),
                      ("margin", "probability_margin")]:
        f[f"mean_{name}"], _ = _means(comp, n, maps[key])
        f[f"mean_{name}_boundary"], _ = _means(comp, n, maps[key], bd)
        # ДОПУЩЕНИЕ: среднее по пустому интерьеру = 0 (статья не уточняет;
        # отсутствие интерьера кодирует отдельное поле has_interior).
        f[f"mean_{name}_interior"] = np.nan_to_num(_means(comp, n, maps[key], inner)[0])
    # Предсказанный класс — argmax, поэтому его вероятность равна max-prob.
    f["mean_predicted_class_probability"] = f["mean_max_probability"].copy()
    # ДОПУЩЕНИЕ: относительная площадь — доля изображения, относительные размеры
    # границы и интерьера — доля площади компоненты.
    f["relative_component_area"] = area / (h * w)
    f["relative_boundary_size"] = f["boundary_size"] / area
    f["relative_interior_size"] = f["interior_size"] / area
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    f["touches_image_boundary"] = (np.bincount(comp[edge], minlength=n) > 0).astype(np.int8)
    f["has_interior"] = (f["interior_size"] > 0).astype(np.int8)
    for c in range(num_classes):
        f[f"class_{c}"] = (classes == c).astype(np.int8)
    return f


def metaseg_fields(comp, classes, maps, probs):
    """34 поля адаптированного MetaSeg (S2.6): 15 дескрипторов + 19 средних вероятностей классов.

    ДОПУЩЕНИЕ: стемы E и D — энтропия и D = 1 − (p1 − p2) (дисперсия по марже),
    как в MetaSeg Rottmann et al.
    """
    n = len(classes)
    bd = boundary_metaseg(comp)
    inner = ~bd
    S = np.bincount(comp.ravel(), minlength=n).astype(float)
    S_bd = np.bincount(comp[bd], minlength=n).astype(float)
    S_in = np.bincount(comp[inner], minlength=n).astype(float)
    # ДОПУЩЕНИЕ: компонента без границы (вся картинка) — делим на max(S_bd, 1).
    S_bd_safe = np.maximum(S_bd, 1.0)
    f = {}
    for stem, m in [("E", maps["entropy"]), ("D", 1.0 - maps["margin"])]:
        mean, _ = _means(comp, n, m)
        mean_in = np.nan_to_num(_means(comp, n, m, inner)[0])
        mean_bd = np.nan_to_num(_means(comp, n, m, bd)[0])
        f[f"{stem}_mean"] = mean
        f[f"{stem}_in"] = mean_in
        f[f"{stem}_bd"] = mean_bd
        f[f"{stem}_rel"] = mean * S / S_bd_safe
        f[f"{stem}_rel_in"] = mean_in * S_in / S_bd_safe
    f["S"], f["S_in"], f["S_bd"] = S, S_in, S_bd
    f["S_rel"] = S / S_bd_safe
    f["S_rel_in"] = S_in / S_bd_safe
    for c in range(probs.shape[0]):
        f[f"prob_mean_{c}"], _ = _means(comp, n, probs[c])
    return f


# --- Агрегация попиксельной карты x (H·W,) по компонентам comp (H·W,) с n компонентами ---

def comp_mean(x, comp, n):
    """Среднее по компоненте; x, comp — тензоры на одном устройстве, результат float64."""
    k = torch.bincount(comp, minlength=n).double()
    return torch.zeros(n, dtype=torch.float64, device=x.device).index_add_(0, comp, x.double()) / k


def comp_max(x, comp, n):
    return torch.full((n,), -np.inf, device=x.device).scatter_reduce(0, comp, x.float(), "amax")


def comp_min(x, comp, n):
    return torch.full((n,), np.inf, device=x.device).scatter_reduce(0, comp, x.float(), "amin")


def comp_quantile(x, comp, n, q):
    """q-квантиль по компоненте (ближайший ранг снизу); x, comp — numpy."""
    k = np.bincount(comp, minlength=n)
    order = np.lexsort((x, comp))
    start = np.cumsum(k) - k
    return x[order][start + np.floor(q * (k - 1)).astype(np.int64)]
