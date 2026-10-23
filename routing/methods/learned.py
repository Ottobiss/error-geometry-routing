"""Обучаемые методы: регрессия Δ на пиксель по полям компоненты и предиктор провала p как роутер."""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from baseline.failure import fit_p
from core.components import M_FIELDS, P_FIELDS
from routing.methods import register

# Префикс поля «префикс:колонка» → таблица признаков в results/<dataset>/features/
FEATURE_TABLES = {"ddu": "ddu", "proto": "prototypes", "knn": "knn"}


def fit_density(frame, fields):
    """HistGBR: Δ на пиксель; вес строки — число размеченных пикселей (шумные крошки весят мало)."""
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                          min_samples_leaf=50, random_state=0)
    model.fit(frame[fields], frame.density, sample_weight=frame.n_valid)
    return model


@register("regr")
class Regressor:
    """fit_density на произвольном наборе полей.

    fields — через запятую: колонки таблицы компонент (mean_entropy, …) или «префикс:колонка» из
    таблиц признаков FEATURE_TABLES (ddu:logp_mean, proto:B_spred_mean, knn:K_A_dnp_mean, …).
    Сочетание: fields=mean_entropy,ddu:logp_mean; его контроль — fields=mean_entropy.
    """

    def __init__(self, fields="mean_entropy"):
        self.fields = fields.split(",") if isinstance(fields, str) else list(fields)

    def _frame(self, units):
        cols = {}
        for f in self.fields:
            if ":" in f:
                prefix, col = f.split(":", 1)
                cols[f] = units.feature(FEATURE_TABLES[prefix], col)
            else:
                cols[f] = units.table[f].values
        return pd.DataFrame(cols)

    def fit(self, train):
        frame = pd.concat([self._frame(train), train.table[["density", "n_valid"]]], axis=1)
        self.model = fit_density(frame, self.fields)
        return self

    def score(self, test):
        return self.model.predict(self._frame(test))


@register("p_fields_delta")
class PFieldsDelta(Regressor):
    """«p-поля → Δ»: fit_density на 37 полях p."""

    def __init__(self):
        super().__init__(P_FIELDS)


@register("metaseg_fields_delta")
class MetaSegFieldsDelta(Regressor):
    """«MetaSeg-поля → Δ»: fit_density на 34 полях MetaSeg."""

    def __init__(self):
        super().__init__(M_FIELDS)


@register("p_failure")
class PFailure:
    """Предиктор провала p (Kang et al.) как роутер: вероятность провала компоненты."""

    def fit(self, train):
        self.model = fit_p(train.table)
        return self

    def score(self, test):
        return self.model.predict_proba(test.table[P_FIELDS])[:, 1]
