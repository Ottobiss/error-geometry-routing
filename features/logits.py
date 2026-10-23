"""Логиты B0 в разрешении выхода модели (19 × h × w, fp16) для всех кадров датасета — для методов,
которым нужен выход модели целиком (md_output, routing.pixel_eval).

    python -m features.logits

До рабочего разрешения логиты поднимаются при использовании (core.models.upsample_logits), как в
SegModel.forward. Проверяется, что argmax поднятых fp16-логитов совпадает с сохранённой маской pred_b0.
Выход: results/<dataset>/logits_b0.npy (n, 19, h, w) и logits_b0_ids.txt (image_id по строкам).
"""
import argparse
import time

import numpy as np
import torch

from core.data import DATASETS, get_dataset, loader
from core.models import load_model, upsample_logits
from core.paths import DatasetResults


class LogitStore:
    """Логиты B0 по image_id (memmap, читаются по одному кадру)."""

    def __init__(self, dataset="cityscapes"):
        res = DatasetResults(dataset)
        self.size = DATASETS[dataset].size
        self.arr = np.load(res.logits, mmap_mode="r")
        self.pos = {i: k for k, i in enumerate(res.logits_ids.read_text().split())}

    def __getitem__(self, image_id):
        """Логиты на рабочем разрешении, (19, H, W) float32 на GPU."""
        return upsample_logits(np.array(self.arr[self.pos[image_id]]), self.size)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cityscapes")
    args = ap.parse_args()

    ds, res = get_dataset(args.dataset, full=False), DatasetResults(args.dataset)
    b0 = load_model("b0")
    arr, agree, t0 = None, [], time.time()
    for k, sample in enumerate(loader(ds)):
        b0.probs(sample, ds.size)
        lr = b0.features["logits"].half()
        if arr is None:
            arr = np.lib.format.open_memmap(res.logits, mode="w+", dtype=np.float16,
                                            shape=(len(ds), *lr.shape))
        arr[k] = lr.cpu().numpy()
        pred = np.load(res.mask(sample["image_id"]))["pred_b0"]
        agree.append((upsample_logits(lr, ds.size).argmax(0).cpu().numpy() == pred).mean())
        if (k + 1) % 50 == 0 or k + 1 == len(ds):
            print(f"{k + 1}/{len(ds)}  {time.time() - t0:.0f} с")
    arr.flush()
    res.logits_ids.write_text("\n".join(ds.image_ids))
    print(f"{res.logits}: {arr.shape}; совпадение argmax fp16-логитов с pred_b0: "
          f"мин {min(agree):.6f}, среднее {np.mean(agree):.6f}")


if __name__ == "__main__":
    main()
