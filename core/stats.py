"""Класс-условные статистики признаков: моменты, регуляризованная обратная ковариация, Махаланобис."""
import torch


def moments(n, s, ss):
    """Средние и ковариации (нормировка 1/N) из достаточных статистик:
    n (C,), s (C, D) = Σ x, ss (C, D, D) = Σ x xᵀ → mu (C, D), cov (C, D, D)."""
    mu = s / n[:, None]
    return mu, ss / n[:, None, None] - mu[:, :, None] * mu[:, None, :]


def inverse(S, eps):
    """Обратная к S (D, D) или пачке (C, D, D): eps = 0 — псевдообратная, иначе inv(S + eps·I)."""
    if eps > 0:
        return torch.linalg.inv(S + eps * torch.eye(S.shape[-1], dtype=S.dtype, device=S.device))
    return torch.linalg.pinv(S, hermitian=True)


def class_md(x, cls, mu, inv):
    """Расстояние Махаланобиса строк x (n, D) до центра своего класса: mu (C, D), inv (C, D, D),
    cls (n,) — тензор на устройстве x. → (n,)."""
    out = torch.empty(len(x), dtype=x.dtype, device=x.device)
    for c in cls.unique().tolist():
        m = cls == c
        d = x[m] - mu[c]
        out[m] = ((d @ inv[c]) * d).sum(1).clamp(min=0).sqrt()
    return out
