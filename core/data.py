"""Датасеты (torch Dataset), популяции и компоненты популяции.

Элемент датасета — dict:
    image_id  str
    image     uint8 (3, H, W) — кадр в исходном разрешении (вход тяжёлой модели; нет при full=False)
    small     uint8 (3, h, w) — кадр, уменьшенный до рабочего разрешения (вход лёгкой модели)
    label     uint8 (h, w)    — GT в trainIds на рабочем разрешении, ignore = 255
Уменьшение — PIL bilinear (как при подготовке всех сохранённых результатов).
Популяция — список image_id в splits/<dataset>_<pop>.txt (Cityscapes: p0 — подгонка, p1 — тест;
ACDC: по условию — fog, night, rain, snow).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from core.components import IGNORE, NC
from core.paths import RESULTS, DatasetResults

SPLITS = Path("splits")


def _chw(img):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(img).transpose(2, 0, 1)))


class SegDataset(Dataset):
    """Общая часть: список (image_id, путь к кадру, путь к разметке) и чтение элемента."""
    size = None                    # рабочее разрешение (W, H)
    heavy_full = True              # тяжёлая модель получает кадр в исходном разрешении (иначе — small)

    def __init__(self, items, ids=None, full=True):
        if ids is not None:
            ids = set(ids)
            items = [it for it in items if it[0] in ids]
        self.items, self.full = items, full
        self.image_ids = [it[0] for it in items]

    def __len__(self):
        return len(self.items)

    def label(self, i):
        """GT элемента i без чтения кадра: uint8 (h, w) numpy."""
        raise NotImplementedError

    def __getitem__(self, i):
        image_id, image_path, _ = self.items[i]
        img = Image.open(image_path).convert("RGB")
        sample = {"image_id": image_id, "small": _chw(img.resize(self.size, Image.BILINEAR)),
                  "label": torch.from_numpy(self.label(i))}
        if self.full:
            sample["image"] = _chw(img)
        return sample


class Cityscapes(SegDataset):
    """Официальная структура: leftImg8bit/<split>/<город>/*, gtFine/<split>/<город>/*_labelIds.png."""
    root = Path("data/cityscapes")
    size = (1024, 512)             # вдвое меньше исходных 2048×1024

    def __init__(self, ids=None, full=True, split="val"):
        from cityscapesscripts.helpers.labels import labels
        self.to_train = np.full(256, IGNORE, dtype=np.uint8)     # labelId → trainId, прочее → ignore
        for lab in labels:
            if 0 <= lab.id < 256 and 0 <= lab.trainId < NC:
                self.to_train[lab.id] = lab.trainId
        items = []
        for img in sorted((self.root / "leftImg8bit" / split).rglob("*_leftImg8bit.png")):
            image_id = img.name[: -len("_leftImg8bit.png")]
            label = self.root / "gtFine" / split / img.parent.name / f"{image_id}_gtFine_labelIds.png"
            items.append((image_id, img, label))
        super().__init__(items, ids, full)

    def label(self, i):
        gt = Image.open(self.items[i][2]).resize(self.size, Image.NEAREST)
        return self.to_train[np.asarray(gt)]


class ACDC(SegDataset):
    """ACDC: rgb_anon/<условие>/<split>/<последовательность>/*_rgb_anon.png,
    gt/<условие>/<split>/<последовательность>/*_gt_labelTrainIds.png (уже trainIds, ignore = 255).
    image_id = <условие>_<кадр>: имена кадров в разных условиях могут совпадать."""
    root = Path("data/acdc")
    size = (1024, 576)             # 1920×1080 с тем же масштабом по ширине, что у Cityscapes
    # B2 на ACDC сильнее на уменьшенном кадре (проверка на 120 кадрах: 51.4 против 47.8 mIoU при 1920×1080).
    # ДОПУЩЕНИЕ: выбрано по разметке ACDC val, т. е. подогнано на тестовых данных (EXPERIMENTS.md).
    heavy_full = False
    CONDITIONS = ("fog", "night", "rain", "snow")

    def __init__(self, ids=None, full=True, split="val"):
        items = []
        for cond in self.CONDITIONS:
            for img in sorted((self.root / "rgb_anon" / cond / split).rglob("*_rgb_anon.png")):
                stem = img.name[: -len("_rgb_anon.png")]
                label = self.root / "gt" / cond / split / img.parent.name / f"{stem}_gt_labelTrainIds.png"
                items.append((f"{cond}_{stem}", img, label))
        super().__init__(items, ids, full)

    def label(self, i):
        return np.array(Image.open(self.items[i][2]).resize(self.size, Image.NEAREST))


DATASETS = {"cityscapes": Cityscapes, "acdc": ACDC}


def get_dataset(name, ids=None, full=True):
    """Датасет name; ids — только эти кадры (порядок — по image_id); full=False — без кадра
    в исходном разрешении (достаточно для лёгкой модели)."""
    return DATASETS[name](ids, full)


def loader(dataset, workers=2):
    """Поэлементный DataLoader: кадры читаются и уменьшаются в фоновых процессах."""
    return DataLoader(dataset, batch_size=None, num_workers=workers)


FIT_DATASET, FIT_POP = "cityscapes", "p0"       # на чём подгоняется всё: признаки и роутеры


def population(dataset, pop):
    """image_id популяции в порядке сортировки; all — объединение всех популяций датасета."""
    if pop == "all":
        return sorted({i for p in populations(dataset) for i in population(dataset, p)})
    return sorted((SPLITS / f"{dataset}_{pop}.txt").read_text().split())


def populations(dataset):
    """Все популяции датасета (по файлам splits/<dataset>_<pop>.txt)."""
    return sorted(p.stem[len(dataset) + 1:] for p in SPLITS.glob(f"{dataset}_*.txt"))


def test_populations(dataset):
    """Популяции для оценки: все, кроме популяции подгонки."""
    return [p for p in populations(dataset) if (dataset, p) != (FIT_DATASET, FIT_POP)]


@dataclass
class Units:
    """Компоненты B0 одной популяции: таблица (строка на компоненту) и векторы декодера B0."""
    dataset: str
    population: str
    table: pd.DataFrame
    emb: np.ndarray                # (N, 256) float32, строка i ↔ строка i таблицы
    root: Path = RESULTS

    def feature(self, name, col):
        """Колонка таблицы признаков results/<dataset>/features/<name>.parquet в порядке строк."""
        f = pd.read_parquet(DatasetResults(self.dataset, self.root).feature(name),
                            columns=["image_id", "comp_id", col])
        t = self.table[["image_id", "comp_id"]].merge(f, how="left", on=["image_id", "comp_id"])
        assert len(t) == len(self.table) and t[col].notna().all(), (name, col)
        return t[col].values


def load_units(dataset, pop, labeled_only=False, root=RESULTS):
    """Компоненты популяции с векторами B0 и колонкой density = Δ / n_valid (Δ на размеченный пиксель).

    labeled_only — только компоненты с размеченными пикселями (для обучения).
    """
    res = DatasetResults(dataset, root)
    table = pd.read_parquet(res.components)
    emb = pd.read_parquet(res.feature("embeddings"))
    assert table[["image_id", "comp_id"]].equals(emb[["image_id", "comp_id"]])
    keep = table.image_id.isin(set(population(dataset, pop))).to_numpy()
    if labeled_only:
        keep = keep & table.has_valid.to_numpy()
    table = table[keep].reset_index(drop=True)
    table["density"] = np.where(table.n_valid > 0, table.delta / table.n_valid.clip(lower=1), 0.0)
    vectors = emb.drop(columns=["image_id", "comp_id"]).to_numpy(np.float32)[keep]
    return Units(dataset, pop, table, vectors, Path(root))
