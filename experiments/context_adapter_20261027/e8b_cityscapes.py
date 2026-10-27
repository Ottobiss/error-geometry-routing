"""E8b: why did E8 fail? Separates two effects and tries other frozen contexts.

E8 had two confounds:
  1. the frozen context was the stage-entry state, far from what block l expects as keys/values;
  2. tokens outside the routed tiles stayed at entry values after stage 3, so stage 4 attention and the
     decode head (bilinear upsampling across tile borders) read them too.

Only stage 3 (index 2) is masked. B5 runs on the full frame; per image one unmasked pass records the
real states: entry (input of block 0), mid (input of block 20), final (output of block 39).
Routed tokens always start from their real entry state; all other tokens are frozen to one recorded
state in the keys/values of every block. Variants:
  entry_raw - E8's "s3": frozen entry state, output outside the routed tiles left as is
  entry     - same, output outside the routed tiles restored to the real final state (isolates effect 2)
  mid, final - frozen to a later real state, output restored
Reported as % of the B5-B0 gap closed, plus pixel agreement with full-frame B5 inside routed tiles.
agree_b0 is the reference: how often B0 itself agrees with B5 on those pixels.
"""
import json
import os
import time
from pathlib import Path
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
OUT = Path(os.environ["E8B_OUT"])


STAGE = 2
BUDGETS = [0.1, 0.2]
VARIANTS = {"entry_raw": ("entry", False), "entry": ("entry", True),
            "mid": ("mid", True), "final": ("final", True)}


class Frozen(nn.Module):
    """Wraps block i of n of the masked stage; see the module docstring."""

    def __init__(self, layer, state, i, n):
        super().__init__()
        self.layer, self.state, self.i, self.n = layer, state, i, n

    def forward(self, hidden_states, *args, **kwargs):
        s = self.state
        if s.recording:
            if self.i == 0:
                s.rec["entry"] = hidden_states
            if self.i == self.n // 2:
                s.rec["mid"] = hidden_states
            out = self.layer(hidden_states, *args, **kwargs)
            if self.i == self.n - 1:
                s.rec["final"] = out[0]
            return out
        if s.mask is None:
            return self.layer(hidden_states, *args, **kwargs)
        ctx = s.rec[s.freeze]
        out = self.layer(torch.where(s.mask, hidden_states, ctx), *args, **kwargs)
        y = torch.where(s.mask, out[0], ctx)
        if self.i == self.n - 1 and s.restore:
            y = torch.where(s.mask, y, s.rec["final"])
        return (y,) + tuple(out[1:])


def install(model):
    state = SimpleNamespace(recording=False, rec={}, mask=None, freeze="entry", restore=True)
    blocks = model.segformer.encoder.block[STAGE]
    for i, layer in enumerate(blocks):
        blocks[i] = Frozen(layer, state, i, len(blocks))
    return state


def token_mask(m):
    """(8, 16) bool tile mask -> (1, N, 1) bool mask over the tokens of the stage."""
    k = TILE // (4 * 2 ** STAGE)
    t = torch.from_numpy(m).cuda().repeat_interleave(k, 0).repeat_interleave(k, 1)
    return t.reshape(1, -1, 1)


@torch.inference_mode()
def run(model, state, img, **settings):
    """Full-frame B5 prediction with the given state settings; the state is reset afterwards."""
    for k, v in settings.items():
        setattr(state, k, v)
    try:
        logits = model(pixel_values=img).logits
        return F.interpolate(logits.float(), SIZE, mode="bilinear", align_corners=False).argmax(1)[0].byte().cpu().numpy()
    finally:
        state.recording, state.mask = False, None


def sanity(model, state):
    """The recording pass and an all-true mask (any frozen state) must reproduce the cached B5."""
    img = SegSet("cs_fit")[0][0][None].cuda().half()
    (cached,) = next(load("cs_fit", "pred_b5"))
    agree = (run(model, state, img, recording=True) == cached).mean()
    assert agree > 0.999, f"recording pass does not reproduce B5, agreement {agree:.4f}"
    ones = token_mask(np.ones((SIZE[0] // TILE, SIZE[1] // TILE), bool))
    for freeze in ("entry", "mid", "final"):
        agree = (run(model, state, img, mask=ones, freeze=freeze, restore=True) == cached).mean()
        assert agree > 0.999, f"all-true mask ({freeze}) does not reproduce B5, agreement {agree:.4f}"


def main():
    model = load_model("b5")
    state = install(model)
    sanity(model, state)

    base = json.loads((RESULTS / "e0_cityscapes.json").read_text())
    feats = np.load(CACHE / "e8_tiles_cityscapes.npz")
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "e8b_cityscapes.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}


    print(f"{'domain':8} {'method':8} {'budget':>6} {'full':>7}" + "".join(f"{n:>11}" for n in VARIANTS)
          + "".join(f"{f'agree {n}':>16}" for n in VARIANTS) + f"{'agree_b0':>10}")
    for domain in DOMAINS:
        if domain in report:
            print(domain, "уже завершён — пропускаем", flush=True)
            continue
        dataset = SegSet(domain)
        assert [p.stem for p, _ in dataset.items] == [p.stem for p in sorted((CACHE / domain).glob("*.npz"))]

        entropy, knn = feats[f"{domain}__entropy__mean"], feats[f"{domain}__knn__mean"]
        gains = np.stack([tile_sum(gain(*a)) for a in load(domain, "gt", "pred_b0", "pred_b5")])
        scores = {"rank": rank(entropy) + rank(knn), "oracle": gains}
        masks = {(k, b): top_tiles(s, b) for k, s in scores.items() for b in BUDGETS}

        zeros = lambda: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
        full = {key: zeros() for key in masks}
        conf = {(*key, n): zeros() for key in masks for n in VARIANTS}
        agree = {(*key, n): np.zeros(2, np.int64) for key in masks for n in VARIANTS}
        agree_b0 = {key: np.zeros(2, np.int64) for key in masks}

        per_frame = []
        started = time.monotonic()
        loader = DataLoader(dataset, batch_size=None, num_workers=2)
        for i, ((img, _, _), (gt, b0, b5)) in enumerate(zip(loader, load(domain, "gt", "pred_b0", "pred_b5"))):
            img = img[None].cuda().half()
            run(model, state, img, recording=True)
            for key, m in masks.items():
                px = m[i].repeat(TILE, 0).repeat(TILE, 1)
                frame_full = confusion(np.where(px, b5, b0), gt)
                full[key] += frame_full
                agree_b0[key] += [(b0 == b5)[px].sum(), px.sum()]
                tm = token_mask(m[i])
                for n, (freeze, restore) in VARIANTS.items():
                    pred = run(model, state, img, mask=tm, freeze=freeze, restore=restore)
                    frame_conf = confusion(np.where(px, pred, b0), gt)
                    conf[(*key, n)] += frame_conf
                    per_frame.append((i, key[0], key[1], n, frame_conf.copy(), frame_full.copy()))
                    agree[(*key, n)] += [(pred == b5)[px].sum(), px.sum()]

            if (i+1) % 5 == 0 or i+1 == len(dataset):
                elapsed = time.monotonic()-started
                eta = elapsed/(i+1)*(len(dataset)-i-1)/60
                print(f"E8b {domain}: {i+1}/{len(dataset)}, {elapsed/60:.1f} мин, ETA ~{eta:.1f} мин", flush=True)

        b0_miou, b5_miou = base[domain]["b0"], base[domain]["b5"]
        closed = lambda c: round(100 * (miou(c) - b0_miou) / (b5_miou - b0_miou), 1)
        pct = lambda a: round(100 * a[0] / a[1], 1)
        report[domain] = {}
        for key in masks:
            row = {"full": closed(full[key]), "miou_full": float(miou(full[key])),
                   "miou_b0": b0_miou, "miou_b5": b5_miou}
            row.update({n: closed(conf[(*key, n)]) for n in VARIANTS})
            row.update({f"miou_{n}": float(miou(conf[(*key, n)])) for n in VARIANTS})
            row.update({f"agree_{n}": pct(agree[(*key, n)]) for n in VARIANTS})
            row["agree_b0"] = pct(agree_b0[key])
            report[domain][f"{key[0]}@{key[1]}"] = row
            print(f"{domain:8} {key[0]:8} {key[1]:6.0%} {row['full']:7.1f}"
                  + "".join(f"{row[n]:11.1f}" for n in VARIANTS)
                  + "".join(f"{row[f'agree_{n}']:16.1f}" for n in VARIANTS) + f"{row['agree_b0']:10.1f}")

        # Save frame confusions for later paired comparisons, before marking domain complete.
        np.savez_compressed(OUT / f"{domain}_frame_confusions.npz",
            frame=np.array([r[0] for r in per_frame]),
            method=np.array([r[1] for r in per_frame]), budget=np.array([r[2] for r in per_frame]),
            variant=np.array([r[3] for r in per_frame]), conf=np.stack([r[4] for r in per_frame]),
            full=np.stack([r[5] for r in per_frame]),
            image_names=np.array([p.stem for p, _ in dataset.items]))
        tmp=report_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(report, indent=2));tmp.replace(report_path)
        print("Сохранён домен:", domain, flush=True)

    print("Результат:", report_path, flush=True)


if __name__ == "__main__":
    main()
