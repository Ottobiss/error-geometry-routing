"""Раскладка results/ и служебное: где лежит каждый артефакт.

results/
  <dataset>/                    всё, что посчитано для кадров датасета
    masks/<image_id>.npz          pred_b0, pred_b2 (uint8, рабочее разрешение)
    components.parquet            строка на компоненту B0: метки и поля p / MetaSeg
    features/<name>.parquet       признаки компонент (embeddings, prototypes, ddu, knn), тот же порядок строк
    logits_b0.npy, logits_b0_ids.txt   логиты B0 в разрешении выхода модели
    cache_<pop>.npz               компоненты, GT и маски популяции — для быстрой оценки роутинга
    tiles_<cfg>.npz, tiles_<cfg>.json   ответ B2 по патчам с контекстом и замеры времени (features.tiles)
  fits/<name>/                  то, что подогнано на Cityscapes P0 (прототипы, GDA, …), и попиксельные AUROC
  runs/<dataset>_<pop>/<name>/  запуски роутера на тестовой популяции: scores.parquet, metrics.json, config.json
"""
import json
import subprocess
from datetime import date
from pathlib import Path

RESULTS = Path("results")


class DatasetResults:
    """Пути к артефактам одного датасета."""

    def __init__(self, dataset, root=RESULTS):
        self.dataset = dataset
        self.dir = Path(root) / dataset

    def mask(self, image_id):
        return self.dir / "masks" / f"{image_id}.npz"

    @property
    def components(self):
        return self.dir / "components.parquet"

    def feature(self, name):
        return self.dir / "features" / f"{name}.parquet"

    @property
    def logits(self):
        return self.dir / "logits_b0.npy"

    @property
    def logits_ids(self):
        return self.dir / "logits_b0_ids.txt"

    def cache(self, pop):
        return self.dir / f"cache_{pop}.npz"

    def tiles(self, cfg, ext="npz"):
        return self.dir / f"tiles_{cfg}.{ext}"


def fit_dir(name, root=RESULTS):
    return Path(root) / "fits" / name


def run_dir(name, test="cityscapes:p1", root=RESULTS):
    """Папка запуска name на тестовой популяции test = <датасет>:<популяция>."""
    return Path(root) / "runs" / test.replace(":", "_") / name


def git_commit():
    """Короткий хэш HEAD с пометкой +dirty, если есть незакоммиченные изменения."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def write_json(path, obj, stamp=True):
    """JSON с кириллицей как есть; stamp — добавить дату и коммит (для config)."""
    if stamp:
        obj = {**obj, "date": date.today().isoformat(), "commit": git_commit()}
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def parse_params(items):
    """Параметры командной строки -p ключ=значение → dict; значение — int, float, true/false или строка."""
    def value(s):
        for cast in (int, float):
            try:
                return cast(s)
            except ValueError:
                pass
        return {"true": True, "false": False}.get(s.lower(), s)
    return {k: value(v) for k, v in (kv.split("=", 1) for kv in items)}
