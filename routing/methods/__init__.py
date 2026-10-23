"""Реестр методов роутера.

Метод — класс с интерфейсом:
    __init__(**params)          гиперпараметры, только именованные
    fit(train: Units) -> self   обучение на компонентах Cityscapes P0 (с меткой)
    score(test: Units) -> (N,)  одно число на компоненту, больше = больше ожидаемый выигрыш от B2

Регистрация: @register("имя") над классом. Методы без учителя не читают delta / density у train.
"""
METHODS = {}


def register(name):
    def wrap(cls):
        assert name not in METHODS, f"метод {name} уже зарегистрирован"
        METHODS[name] = cls
        return cls
    return wrap


class FeatureColumn:
    """Метод без обучения: готовая колонка таблицы признаков results/<dataset>/features/<table>.parquet
    (её построил features.build на Cityscapes P0), со знаком sign."""
    table = None

    def __init__(self, col, sign=1):
        self.col, self.sign = col, sign

    def fit(self, train):
        return self

    def score(self, test):
        return self.sign * test.feature(self.table, self.col)


from routing.methods import density, learned, neighbors, prototypes, simple  # noqa: E402,F401  импорт = регистрация
