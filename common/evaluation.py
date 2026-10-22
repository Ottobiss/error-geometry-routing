"""Метрики Kang et al. (S1): AUROC по компонентам, image-macro CE, отбор с бюджетом на изображение."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auroc(y, score):
    """AUROC, пул по всем компонентам."""
    return roc_auc_score(y, score)


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
