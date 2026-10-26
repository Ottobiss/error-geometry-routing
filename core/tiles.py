"""Патчи для каскада по патчам: сетка на рабочем разрешении, окна с контекстом, перенос оценок на патчи.

Патч — квадрат tile×tile сетки рабочего разрешения (крайние обрезаются по кадру). Тяжёлая модель
получает окно (tile + 2·margin)² вокруг патча, сдвинутое внутрь кадра (размер окна постоянный, чтобы
окна шли пачкой); из её ответа берётся только сам патч.
"""
import numpy as np

from core.components import IGNORE, NC


def grid(size, tile):
    """Патчи (x0, y0, x1, y1) сетки на кадре size = (W, H), построчно."""
    W, H = size
    return [(x, y, min(x + tile, W), min(y + tile, H))
            for y in range(0, H, tile) for x in range(0, W, tile)]


def window(box, tile, margin, size):
    """Левый верхний угол окна с контекстом вокруг патча box."""
    W, H = size
    w = tile + 2 * margin
    return min(max(box[0] - margin, 0), W - w), min(max(box[1] - margin, 0), H - w)


def tile_index(size, tile):
    """Карта (H, W): номер патча каждого пикселя (тот же порядок, что у grid)."""
    W, H = size
    nx = -(-W // tile)
    return (np.arange(H) // tile)[:, None] * nx + (np.arange(W) // tile)[None, :]


def comp_to_tiles(comp, n_comp, tidx, n_tiles):
    """Матрица пересечений (n_comp, n_tiles): сколько пикселей компоненты в каждом патче."""
    return np.bincount(comp.ravel().astype(np.int64) * n_tiles + tidx.ravel(),
                       minlength=n_comp * n_tiles).reshape(n_comp, n_tiles)


def tile_scores(score, overlap):
    """Оценка патча — среднее оценок компонент по его пикселям (взвешено по площади пересечения)."""
    return score @ overlap / overlap.sum(0)


def tile_effects(gt, b0, b2t, tidx, n_tiles):
    """Вклад каждого патча при замене B0 на B2 (по патчам): изменение матрицы ошибок (n_tiles, NC, NC),
    исходная матрица B0 (NC, NC) и Δ патча в пикселях (n_tiles,)."""
    v = gt != IGNORE
    g, p0, p2, t = (a[v].astype(np.int64) for a in (gt, b0, b2t, tidx))
    m = n_tiles * NC * NC
    d = (np.bincount(t * NC * NC + g * NC + p2, minlength=m)
         - np.bincount(t * NC * NC + g * NC + p0, minlength=m)).reshape(n_tiles, NC, NC)
    c0 = np.bincount(g * NC + p0, minlength=NC * NC).reshape(NC, NC)
    delta = np.bincount(t, weights=(p2 == g) & (p0 != g), minlength=n_tiles) \
        - np.bincount(t, weights=(p0 == g) & (p2 != g), minlength=n_tiles)
    return d, c0, delta


def batch_time(k, per_batch, max_batch):
    """Время B2 на k окнах пачками по max_batch; per_batch[b] — время пачки из b окон (per_batch[0] = 0)."""
    return (k // max_batch) * per_batch[max_batch] + per_batch[k % max_batch]
