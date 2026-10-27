"""E8: gate for the adapter idea - is a frozen stage-3 context good enough for B5?

B5 sees the whole frame, but in the chosen stages only the tokens of routed tiles are updated
by the transformer blocks. Every other token keeps its stage-entry value for all blocks, so it
only serves as keys/values ("frozen context"). Here the entry values come from the real B5
(an upper bound for any adapter that maps B0 tokens into B5 token space).

Quality only: the whole frame is still computed, only the write-back is masked. The saving is
counted analytically for now (blocks run on routed queries only; context costs K/V projections),
real FLOPs and latency are measured later.

Stages (0-based): 2 = stride 16, 40 blocks, most of B5's encoder compute; 3 = stride 32.
Variants: "s3" freezes stage 2 only, "s3+s4" freezes stages 2 and 3. Stages 0-1 are run on the
whole frame in this experiment (known simplification, about a fifth of the encoder cost).

Tiles are selected as in E5 (domain-wide ranking at 10% and 20%): `rank` (entropy + knn) and the oracle.
Reported as % of the B5-B0 gap closed, next to:
  full  - B5 on the full frame inside the routed tiles (the E5 "full" column, the target)
  agree - pixel agreement with that full-frame B5 prediction inside the routed tiles
Gate: s3 should stay close to `full` on ACDC. For reference E5 ctx64 (cost x0.9) closed 42 / 17 / 35 / 60
(fog / night / rain / snow) against 61 / 30 / 59 / 59 for full.
"""
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (CACHE, DOMAINS, NUM_CLASSES, RESULTS, SIZE, TILE, SegSet, confusion, gain, load,
                    load_model, miou, tile_sum, top_tiles)
def rank(scores):
    return scores.ravel().argsort().argsort().reshape(scores.shape)

DOMAINS = ["cs_fit", "cs_test"]

BUDGETS = [0.1, 0.2]
STAGES = {"s3": [2], "s3+s4": [2, 3]}


class Frozen(nn.Module):
    """Wraps a SegformerLayer: tokens outside state.mask are reset to the stage entry value."""

    def __init__(self, layer, state, first):
        super().__init__()
        self.layer, self.state, self.first = layer, state, first

    def forward(self, hidden_states, *args, **kwargs):
        s = self.state
        if self.first:
            s.entry = hidden_states
        out = self.layer(hidden_states, *args, **kwargs)
        if s.mask is None:
            return out
        return (torch.where(s.mask, out[0], s.entry),) + tuple(out[1:])


def install(model, stages=(2, 3)):
    """Wrap every block of the given stages, returns {stage: state}."""
    state = {st: SimpleNamespace(mask=None, entry=None) for st in stages}
    blocks = model.segformer.encoder.block
    for st in stages:
        for i, layer in enumerate(blocks[st]):
            blocks[st][i] = Frozen(layer, state[st], i == 0)
    return state


def token_mask(m, stage):
    """(8, 16) bool tile mask -> (1, N, 1) bool mask over the tokens of a stage (stride 4 * 2**stage)."""
    k = TILE // (4 * 2 ** stage)
    t = torch.from_numpy(m).cuda().repeat_interleave(k, 0).repeat_interleave(k, 1)
    return t.reshape(1, -1, 1)


@torch.inference_mode()
def predict(model, state, img, masks):
    """B5 prediction on the full frame where only `masks` {stage: token mask} tokens are updated."""
    for st, s in state.items():
        s.mask = masks.get(st)
    try:
        logits = model(pixel_values=img[None].cuda().half()).logits
        return F.interpolate(logits.float(), SIZE, mode="bilinear", align_corners=False).argmax(1)[0].byte().cpu().numpy()
    finally:
        for s in state.values():
            s.mask = None
            s.entry = None



def sanity(model, state):
    """An all-true mask must reproduce the cached full-frame B5 prediction."""
    img = SegSet("cs_fit")[0][0]
    (cached,) = next(load("cs_fit", "pred_b5"))
    full = {st: token_mask(np.ones((SIZE[0] // TILE, SIZE[1] // TILE), bool), st) for st in state}
    agree = (predict(model, state, img, full) == cached).mean()
    assert agree > 0.999, f"all-true mask does not reproduce B5, agreement {agree:.4f}"


def main():
    model = load_model("b5")
    state = install(model)
    sanity(model, state)

    base = json.loads((RESULTS / "e0_cityscapes.json").read_text())
    feats = np.load(CACHE / "e8_tiles_cityscapes.npz")
    report_path = RESULTS / "e8_cityscapes.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    print(f"{'domain':8} {'method':8} {'budget':>6} {'full':>7}"
          + "".join(f"{n:>9}" for n in STAGES) + "".join(f"{f'agree {n}':>14}" for n in STAGES))
    for domain in DOMAINS:
        if domain in report:
            print(domain, "уже завершён, пропускаем", flush=True)
            continue
        dataset = SegSet(domain)
        assert [p.stem for p, _ in dataset.items] == [p.stem for p in sorted((CACHE / domain).glob("*.npz"))]

        entropy, knn = feats[f"{domain}__entropy__mean"], feats[f"{domain}__knn__mean"]
        gains = np.stack([tile_sum(gain(*a)) for a in load(domain, "gt", "pred_b0", "pred_b5")])
        scores = {"rank": rank(entropy) + rank(knn), "oracle": gains}
        masks = {(k, b): top_tiles(s, b) for k, s in scores.items() for b in BUDGETS}

        zeros = lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        full = {key: zeros() for key in masks}
        conf = {(*key, n): zeros() for key in masks for n in STAGES}
        agree = {(*key, n): np.zeros(2, np.int64) for key in masks for n in STAGES}

        started = __import__("time").monotonic()
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        for i, ((img, _, _), (gt, b0, b5)) in enumerate(zip(loader, load(domain, "gt", "pred_b0", "pred_b5"))):
            for key, m in masks.items():
                px = m[i].repeat(TILE, 0).repeat(TILE, 1)
                full[key] += confusion(np.where(px, b5, b0), gt)
                for n, stages in STAGES.items():
                    pred = predict(model, state, img, {st: token_mask(m[i], st) for st in stages})
                    conf[(*key, n)] += confusion(np.where(px, pred, b0), gt)
                    agree[(*key, n)] += [(pred == b5)[px].sum(), px.sum()]

            if (i+1) % 5 == 0 or i+1 == len(dataset):
                elapsed = __import__("time").monotonic()-started
                eta = elapsed/(i+1)*(len(dataset)-i-1)/60
                print(f"E8 {domain}: {i+1}/{len(dataset)}, {elapsed/60:.1f} мин, ETA ~{eta:.1f} мин", flush=True)

        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        closed = lambda c: round(100 * (miou(c) - b0_miou) / (b5_miou - b0_miou), 1)
        report[domain] = {}
        for key in masks:
            row = {"full": closed(full[key])}
            row.update({n: closed(conf[(*key, n)]) for n in STAGES})
            row.update({f"agree_{n}": round(100 * agree[(*key, n)][0] / agree[(*key, n)][1], 1) for n in STAGES})
            report[domain][f"{key[0]}@{key[1]}"] = row
            print(f"{domain:8} {key[0]:8} {key[1]:6.0%} {row['full']:7.1f}"
                  + "".join(f"{row[n]:9.1f}" for n in STAGES)
                  + "".join(f"{row[f'agree_{n}']:14.1f}" for n in STAGES))

        temp = report_path.with_suffix(".json.part")
        temp.write_text(json.dumps(report, indent=2))
        temp.replace(report_path)
        print("Сохранён домен:", domain, flush=True)

    print("Результат:", report_path, flush=True)


if __name__ == "__main__":
    main()
