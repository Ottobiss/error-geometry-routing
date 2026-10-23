import numpy as np

from core.components import (
    IGNORE, base_fields, boundary_metaseg, boundary_p, delta_labels,
    extract_components, failure_labels, metaseg_fields, pixel_maps,
)


def toy_pred():
    # Класс 1 — квадрат 4×4 в углу, класс 2 — две диагонально касающиеся точки
    # (по 8-связности одна компонента), остальное — класс 0.
    pred = np.zeros((8, 8), dtype=np.int64)
    pred[:4, :4] = 1
    pred[6, 5] = 2
    pred[7, 6] = 2
    return pred


def test_components_eight_connected():
    comp, classes = extract_components(toy_pred())
    assert sorted(classes.tolist()) == [0, 1, 2]
    assert len(np.unique(comp)) == 3 and comp.min() == 0


def test_two_separate_regions_same_class():
    pred = np.zeros((5, 5), dtype=np.int64)
    pred[0, 0] = 1
    pred[4, 4] = 1
    comp, classes = extract_components(pred)
    assert (classes == 1).sum() == 2


def test_boundaries():
    comp, classes = extract_components(toy_pred())
    sq = comp == comp[0, 0]
    bp = boundary_p(comp)
    # Для p граница — верх/лево (край изображения) и низ/право (сосед другой
    # компоненты), поэтому у квадрата 4×4 в углу интерьер 2×2.
    assert (sq & ~bp).sum() == 4
    bm = boundary_metaseg(comp)
    # Для MetaSeg край не граница: граница — только последняя строка и столбец квадрата.
    assert (sq & bm).sum() == 7


def test_failure_label_and_ignore():
    pred = toy_pred()
    comp, classes = extract_components(pred)
    gt = np.zeros_like(pred)
    gt[0, 0] = 1                       # один пиксель квадрата верно → IoU > 0
    gt[6, 5] = IGNORE
    gt[7, 6] = IGNORE                  # компонента класса 2 целиком ignore
    y, valid = failure_labels(comp, classes, gt)
    by_class = {c: (y[i], valid[i]) for i, c in enumerate(classes)}
    assert by_class[1] == (0, True)
    assert by_class[0] == (0, True)
    assert by_class[2][1] == False     # нет валидных пикселей → без метки
    gt[0, 0] = 0
    y, _ = failure_labels(comp, classes, gt)
    assert y[classes.tolist().index(1)] == 1


def test_delta():
    pred_l = toy_pred()
    comp, classes = extract_components(pred_l)
    gt = pred_l.copy()
    pred_h = pred_l.copy()
    gt[0, 0] = 0                       # лёгкая ошибается в 2 пикселях квадрата,
    gt[0, 1] = 0
    pred_h[0, 0] = 0                   # тяжёлая исправляет оба,
    pred_h[0, 1] = 0
    pred_h[3, 3] = 0                   # но портит один, где лёгкая была права
    gt[1, 1] = IGNORE                  # ignore не учитывается
    pred_h[1, 1] = 0
    delta, n_valid = delta_labels(comp, len(classes), gt, pred_l, pred_h)
    i = classes.tolist().index(1)
    assert delta[i] == 2 - 1
    assert n_valid[i] == 15


def test_field_counts_and_values():
    rng = np.random.default_rng(0)
    pred = toy_pred()
    logits = rng.normal(size=(19, 8, 8))
    logits[pred, np.arange(8)[:, None], np.arange(8)] += 10   # argmax = pred
    probs = np.exp(logits) / np.exp(logits).sum(0)
    comp, classes = extract_components(probs.argmax(0))
    maps = pixel_maps(probs)
    fb = base_fields(comp, classes, maps)
    fm = metaseg_fields(comp, classes, maps, probs)
    assert len(fb) == 37 and len(fm) == 34
    i = classes.tolist().index(1)
    assert fb["area"][i] == 16 and fb["interior_size"][i] == 4
    assert np.isclose(fb["mean_max_probability"][i], probs.max(0)[:4, :4].mean())
    assert fb["touches_image_boundary"][i] == 1
    assert np.allclose(sum(fm[f"prob_mean_{c}"] for c in range(19)), 1.0)
