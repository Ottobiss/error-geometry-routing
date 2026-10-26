"""E5: real cascade - B5 runs only on selected tiles, each cropped with some context.

Tiles are selected as in E4 (domain-wide ranking) by entropy, rank (entropy+knn) and the oracle.
Each tile is seen by B5 inside a (64 + 2*ctx)^2 window, shifted to stay inside the image;
only the tile itself is pasted into B0's prediction. "full" = B5 on the full frame (cached).
Reported as % of the full-frame B5-B0 gap closed.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CACHE, DOMAINS, NUM_CLASSES, RESULTS, SIZE, TILE, SegSet, confusion, gain, load,
                    load_model, miou, tile_sum, top_tiles)
from e4_router import rank

CONTEXTS = [0, 32, 64, 128]
BUDGETS = [0.1, 0.2]
BATCH = 32


@torch.inference_mode()
def crop_preds(model, img, tiles, ctx, tile=TILE):
    """B5 predictions for (row, col) tiles, each seen with `ctx` pixels of context."""
    (h, w), size = SIZE, tile + 2 * ctx
    crops, offsets = [], []
    for r, c in tiles:
        y = min(max(r * tile - ctx, 0), h - size)
        x = min(max(c * tile - ctx, 0), w - size)
        crops.append(img[:, y:y + size, x:x + size])
        offsets.append((r * tile - y, c * tile - x))

    preds = []
    for i in range(0, len(crops), BATCH):
        logits = model(pixel_values=torch.stack(crops[i:i + BATCH]).cuda().half()).logits
        preds.append(F.interpolate(logits.float(), (size, size), mode="bilinear").argmax(1).byte().cpu())
    preds = torch.cat(preds).numpy() if preds else []
    return {t: p[oy:oy + tile, ox:ox + tile] for t, p, (oy, ox) in zip(tiles, preds, offsets)}


def paste(base, tiles, preds, tile=TILE):
    out = base.copy()
    for r, c in tiles:
        out[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = preds[(r, c)]
    return out


def main():
    model = load_model("b5")
    base = json.loads((RESULTS / "e0.json").read_text())
    feats = np.load(CACHE / "e3_tiles.npz")
    report = {}

    print(f"{'domain':8} {'method':8} {'budget':>6}" + "".join(f"{f'ctx{c}':>8}" for c in CONTEXTS) + f"{'full':>8}")
    for domain in DOMAINS:
        dataset = SegSet(domain)
        assert [p.stem for p, _ in dataset.items] == [p.stem for p in sorted((CACHE / domain).glob("*.npz"))]

        entropy, knn = feats[f"{domain}__entropy__mean"], feats[f"{domain}__knn__mean"]
        gains = np.stack([tile_sum(gain(*a)) for a in load(domain, "gt", "pred_b0", "pred_b5")])
        scores = {"entropy": entropy, "rank": rank(entropy) + rank(knn), "oracle": gains}
        masks = {(k, b): top_tiles(s, b) for k, s in scores.items() for b in BUDGETS}

        confs = np.zeros((len(masks), len(CONTEXTS) + 1, NUM_CLASSES, NUM_CLASSES), np.int64)
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        for i, ((img, _, _), (gt, b0, b5)) in enumerate(zip(loader, load(domain, "gt", "pred_b0", "pred_b5"))):
            needed = sorted({t for m in masks.values() for t in zip(*np.nonzero(m[i]))})
            preds = {ctx: crop_preds(model, img, needed, ctx) for ctx in CONTEXTS}
            for j, m in enumerate(masks.values()):
                tiles = list(zip(*np.nonzero(m[i])))
                for k, ctx in enumerate(CONTEXTS):
                    confs[j, k] += confusion(paste(b0, tiles, preds[ctx]), gt)
                confs[j, -1] += confusion(np.where(m[i].repeat(TILE, 0).repeat(TILE, 1), b5, b0), gt)

        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        report[domain] = {}
        for (k, b), c in zip(masks, confs):
            closed = [round(100 * (miou(x) - b0_miou) / (b5_miou - b0_miou), 1) for x in c]
            report[domain][f"{k}@{b}"] = closed
            print(f"{domain:8} {k:8} {b:6.0%}" + "".join(f"{v:8.1f}" for v in closed))

    (RESULTS / "e5.json").write_text(json.dumps({"contexts": CONTEXTS + ["full"], **report}, indent=2))


if __name__ == "__main__":
    main()
