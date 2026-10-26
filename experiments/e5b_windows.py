"""E5b: route large windows instead of 64px tiles, at matched B5 compute.

The frame is split into 2x4 windows of 256x256. A window's score is the mean (oracle: sum)
of its tile scores. B5 sees the window with `ctx` pixels of context; only the window is pasted.
Compute = B5 pixels processed / full-frame pixels, overlaps counted.
Baseline at the same compute: whole images routed to full-frame B5 (cached predictions).
Reported as % of the full-frame B5-B0 gap closed.
"""
import json

import numpy as np
from torch.utils.data import DataLoader

from common import (CACHE, DOMAINS, NUM_CLASSES, RESULTS, SIZE, TILE, SegSet, confusion, gain, load,
                    load_model, miou, route_curve, tile_sum, top_tiles)
from e4_router import rank
from e5_crops import crop_preds, paste

WIN = 256
CONTEXTS = [0, 32, 64]
COMPUTE = [0.25, 0.5]


def window_cost(ctx):
    return (WIN + 2 * ctx) ** 2 / (SIZE[0] * SIZE[1])


def to_windows(tiles, reduce):
    k = WIN // TILE
    n, h, w = tiles.shape
    return reduce(tiles.reshape(n, h // k, k, w // k, k), axis=(2, 4))


def main():
    model = load_model("b5")
    base = json.loads((RESULTS / "e0.json").read_text())
    feats = np.load(CACHE / "e3_tiles.npz")
    report = {}

    cols = [f"img@{c:.0%}" for c in COMPUTE] + [f"ctx{x}@{c:.0%}" for x in CONTEXTS for c in COMPUTE]
    print(f"{'domain':8} {'method':8}" + "".join(f"{c:>10}" for c in cols))
    for domain in DOMAINS:
        dataset = SegSet(domain)
        entropy, knn = feats[f"{domain}__entropy__mean"], feats[f"{domain}__knn__mean"]
        gains = np.stack([tile_sum(gain(*a)) for a in load(domain, "gt", "pred_b0", "pred_b5")])
        tiles = {"entropy": entropy, "rank": rank(entropy) + rank(knn), "oracle": gains}
        reduce = {"entropy": np.mean, "rank": np.mean, "oracle": np.sum}
        windows = {k: to_windows(s, reduce[k]) for k, s in tiles.items()}
        masks = {(k, ctx, c): top_tiles(w, c / (w[0].size * window_cost(ctx)))
                 for k, w in windows.items() for ctx in CONTEXTS for c in COMPUTE}

        confs = np.zeros((len(masks), NUM_CLASSES, NUM_CLASSES), np.int64)
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        for i, ((img, _, _), (gt, b0)) in enumerate(zip(loader, load(domain, "gt", "pred_b0"))):
            for ctx in CONTEXTS:
                selected = {key: list(zip(*np.nonzero(m[i]))) for key, m in masks.items() if key[1] == ctx}
                needed = sorted({w for ws in selected.values() for w in ws})
                preds = crop_preds(model, img, needed, ctx, tile=WIN)
                for key, ws in selected.items():
                    confs[list(masks).index(key)] += confusion(paste(b0, ws, preds, tile=WIN), gt)

        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        closed = lambda m: round(100 * (m - b0_miou) / (b5_miou - b0_miou), 1)
        report[domain] = {}
        for k, s in tiles.items():
            per_image = reduce[k](s, axis=(1, 2), keepdims=True)
            row = [closed(m) for m in route_curve(domain, np.broadcast_to(per_image, s.shape), COMPUTE)]
            row += [closed(miou(confs[list(masks).index((k, ctx, c))])) for ctx in CONTEXTS for c in COMPUTE]
            report[domain][k] = dict(zip(cols, row))
            print(f"{domain:8} {k:8}" + "".join(f"{v:10.1f}" for v in row))

    (RESULTS / "e5b.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
