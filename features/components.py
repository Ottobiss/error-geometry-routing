"""Прогон B0 и B2 по кадрам датасета: маски и таблица компонент B0 с полями p / MetaSeg и метками.

    python -m features.components
    python -m features.components --limit 3           # смоук-тест: первые 3 кадра, ничего не пишется

Выход: results/<dataset>/masks/<image_id>.npz (pred_b0, pred_b2 — uint8 на рабочем разрешении)
и results/<dataset>/components.parquet — строка на компоненту B0:
image_id, comp_id, class_id, y_fail (провал по Kang et al.), has_valid, n_valid, delta (Δ — пиксели,
где права только B2, минус пиксели, где права только B0), 37 полей p и 34 поля MetaSeg (ms_*).
"""
import argparse
import time

import numpy as np
import pandas as pd

from core.components import (
    base_fields, delta_labels, extract_components, failure_labels, metaseg_fields, pixel_maps,
)
from core.data import DATASETS, get_dataset, loader
from core.models import load_model
from core.paths import DatasetResults


def image_table(image_id, probs_b0, pred_b2, gt):
    """Таблица компонент одного кадра."""
    pred_b0 = probs_b0.argmax(0)
    comp, classes = extract_components(pred_b0)
    n = len(classes)
    maps = pixel_maps(probs_b0)
    y, valid = failure_labels(comp, classes, gt)
    delta, n_valid = delta_labels(comp, n, gt, pred_b0, pred_b2)
    cols = {"image_id": [image_id] * n, "comp_id": np.arange(n), "class_id": classes,
            "y_fail": y, "has_valid": valid, "n_valid": n_valid, "delta": delta}
    cols.update(base_fields(comp, classes, maps))
    cols.update({f"ms_{k}": v for k, v in metaseg_fields(comp, classes, maps, probs_b0).items()})
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    full = DATASETS[args.dataset].heavy_full
    ds = get_dataset(args.dataset, full=full)
    if args.limit is not None:
        ds = get_dataset(args.dataset, ds.image_ids[: args.limit], full=full)
    res = DatasetResults(args.dataset)
    b0, b2 = load_model("b0"), load_model("b2")
    if args.limit is None:
        res.mask("x").parent.mkdir(parents=True, exist_ok=True)

    tables, t0 = [], time.time()
    for k, sample in enumerate(loader(ds), 1):
        image_id, gt = sample["image_id"], sample["label"].numpy()
        probs_b0 = b0.probs(sample, ds.size).cpu().numpy()
        pred_b2 = b2.probs(sample, ds.size, full).argmax(0).cpu().numpy().astype(np.uint8)
        if args.limit is None:
            np.savez_compressed(res.mask(image_id),
                                pred_b0=probs_b0.argmax(0).astype(np.uint8), pred_b2=pred_b2)
        tables.append(image_table(image_id, probs_b0, pred_b2, gt))
        if k % 50 == 0 or k == len(ds):
            print(f"{k}/{len(ds)}  {time.time() - t0:.0f} с")

    df = pd.concat(tables, ignore_index=True)
    if args.limit is None:
        df.to_parquet(res.components)
    lab = df[df.has_valid]
    print(f"{len(df)} компонент, {len(df) / len(ds):.1f} на кадр; с меткой {len(lab)}, "
          f"доля провалов {lab.y_fail.mean():.3f}; delta>0 {(lab.delta > 0).mean():.3f}, "
          f"delta<0 {(lab.delta < 0).mean():.3f}, delta=0 {(lab.delta == 0).mean():.3f}")


if __name__ == "__main__":
    main()
