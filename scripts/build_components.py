"""Прогон B0 и B2 по датасету: сохраняет маски и таблицу компонент B0 с признаками и метками.

    python scripts/build_components.py
    python scripts/build_components.py --limit 3 --out <папка>

На выходе в --out:
  masks/<image_id>.npz            — pred_b0, pred_b2 (uint8, 512×1024, сжатые)
  components_<dataset>_<split>.parquet — строка на компоненту B0
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.components import (  # noqa: E402
    base_fields, delta_labels, extract_components, failure_labels, metaseg_fields, pixel_maps,
)
from common.data import cityscapes_pairs, load_pair  # noqa: E402
from common.models import load_model, predict_probs  # noqa: E402


def image_table(image_id, probs_b0, pred_b2, gt):
    """Таблица компонент одного кадра (pandas.DataFrame)."""
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    pairs = cityscapes_pairs(split=args.split)[: args.limit]
    b0, b2 = load_model("b0"), load_model("b2")

    tables, t0 = [], time.time()
    for k, (image_id, img_path, gt_path) in enumerate(pairs, 1):
        img, gt = load_pair(img_path, gt_path)
        probs_b0 = predict_probs(b0, img).cpu().numpy()
        pred_b2 = predict_probs(b2, img).argmax(0).cpu().numpy().astype(np.uint8)
        pred_b0 = probs_b0.argmax(0).astype(np.uint8)
        np.savez_compressed(out / "masks" / f"{image_id}.npz", pred_b0=pred_b0, pred_b2=pred_b2)
        tables.append(image_table(image_id, probs_b0, pred_b2, gt))
        if k % 50 == 0 or k == len(pairs):
            print(f"{k}/{len(pairs)}  {time.time() - t0:.0f} с")

    df = pd.concat(tables, ignore_index=True)
    path = out / f"components_{args.dataset}_{args.split}.parquet"
    df.to_parquet(path)
    lab = df[df.has_valid]
    print(f"{path}: {len(df)} компонент, {len(df) / len(pairs):.1f} на кадр; "
          f"с меткой {len(lab)}, доля провалов {lab.y_fail.mean():.3f}; "
          f"delta>0 {(lab.delta > 0).mean():.3f}, delta<0 {(lab.delta < 0).mean():.3f}, delta=0 {(lab.delta == 0).mean():.3f}")


if __name__ == "__main__":
    main()
