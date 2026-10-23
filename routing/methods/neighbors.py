"""kNN-методы: DNP / cDNP (Galesso et al., ICCVW 2023), косинусный kNN и CSLS (Conneau et al., 2018) —
колонки features.knn. Все оценки там уже «больше = подозрительнее», знак плюс."""
from features.knn import FEATS, SCORES
from routing.methods import FeatureColumn, register


@register("knn")
class KNN(FeatureColumn):
    """feat — Q / K / V (стадия 4, attention) / S4 (выход стадии 4) / C (вход classifier);
    ref — A (случайный опорный набор) / B (сбалансированно по классам, отступление от статьи);
    score — dnp / cdnp / cos / csls (r_S внутри опорного набора) / cslsb (r_S двудольный, статья);
    agg — mean / max по пикселям компоненты."""
    table = "knn"

    def __init__(self, feat="K", ref="A", score="dnp", agg="mean"):
        assert feat in FEATS and ref in ("A", "B") and score in SCORES and agg in ("mean", "max")
        super().__init__(f"{feat}_{ref}_{score}_{agg}")
