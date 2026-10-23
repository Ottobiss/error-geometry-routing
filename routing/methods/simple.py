"""Опорные методы: random, entropy, oracle."""
import numpy as np

from core.evaluation import oracle_score
from routing.methods import register


@register("random")
class Random:
    def __init__(self, seed=0):
        self.seed = seed

    def fit(self, train):
        return self

    def score(self, test):
        return np.random.default_rng(self.seed).random(len(test.table))


@register("entropy")
class Entropy:
    """Средняя энтропия softmax B0 по компоненте."""

    def fit(self, train):
        return self

    def score(self, test):
        return test.table.mean_entropy.values


@register("oracle")
class Oracle:
    """ОРАКУЛ: читает метку test (Δ на размеченный пиксель). Верхняя граница, не метод.
    Компоненты с Δ ≤ 0 получают −inf — роутинг их не отдаёт B2 (бюджет — предел)."""

    def fit(self, train):
        return self

    def score(self, test):
        return oracle_score(test.table)
