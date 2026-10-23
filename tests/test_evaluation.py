import numpy as np

from core.evaluation import route


def test_route_skips_component_that_does_not_fit():
    area = np.array([50, 10, 30, 5])
    score = np.array([4.0, 3.0, 2.0, 1.0])
    # бюджет 40: 0 (50) не помещается и пропускается, 1 (10) и 2 (30) берутся, 3 — уже нет места
    assert route(area, score, 0.4, 100).tolist() == [1, 2]
    # бюджет 46: после 1 и 2 остаётся 6 — помещается и 3
    assert route(area, score, 0.46, 100).tolist() == [1, 2, 3]


def test_route_never_sends_minus_inf():
    score = np.array([2.0, -np.inf, 1.0])
    assert route(np.array([10, 10, 10]), score, 1.0, 100).tolist() == [0, 2]   # 1 не берётся


def test_route_empty_when_nothing_fits():
    assert route(np.array([50, 60]), np.array([1.0, 2.0]), 0.1, 100).tolist() == []
