"""Близость к прототипам классов (Williams et al., 2024, arXiv 2402.17653): колонки features.prototypes."""
from routing.methods import FeatureColumn, register

# Знак: сходство — минус (меньше сходство = больше ожидаемый выигрыш), доли — плюс
SIGN = {"spred_mean": -1, "spred_min": -1, "spred_p10": -1, "margin_mean": -1,
        "flag_frac": 1, "low_frac": 1}


@register("proto")
class Prototype(FeatureColumn):
    """variant — A (все GT-пиксели класса, как в статье) / B (только верные B0);
    feature — spred_mean / spred_min / spred_p10 / margin_mean / flag_frac / low_frac."""
    table = "prototypes"

    def __init__(self, variant="A", feature="spred_mean"):
        assert variant in ("A", "B") and feature in SIGN, (variant, feature)
        super().__init__(f"{variant}_{feature}", SIGN[feature])
