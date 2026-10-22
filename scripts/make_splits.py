"""Разбиение Cityscapes val на P0 (300 кадров) и P1 (200 кадров), как в Kang et al.

Статья не говорит, как выбирались кадры, поэтому берём случайное разбиение
с фиксированным сидом. Списки сохраняются в splits/ и коммитятся.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import cityscapes_pairs  # noqa: E402

SEED = 0


def main():
    ids = [image_id for image_id, _, _ in cityscapes_pairs()]
    perm = np.random.default_rng(SEED).permutation(len(ids))
    out = Path("splits")
    out.mkdir(exist_ok=True)
    for name, idx in [("p0", perm[:300]), ("p1", perm[300:])]:
        (out / f"cityscapes_{name}.txt").write_text("\n".join(sorted(ids[i] for i in idx)) + "\n")
        print(name, len(idx))


if __name__ == "__main__":
    main()
