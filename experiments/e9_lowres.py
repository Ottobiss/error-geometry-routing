"""E9: B5 on the whole frame at reduced resolution as a compute knob.

B5 runs on every frame downscaled by `scale` (cost ~ scale^2 of full-frame B5); its logits are
cached at the B0 logit resolution (cache/e9/<domain>/<name>.npz). Predictions:
  b5   - low-res B5 alone
  ens  - mean of B0 and low-res B5 probabilities
  conf - per pixel, the model with the higher max probability
Reported as % of the full-frame B5-B0 gap closed, per domain and on the test pool (cs_test + ACDC).
Reference at matched compute (E7, pool, entropy): 25% of frames -> 59.9, 50% -> 77.3.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CACHE, DOMAINS, NUM_CLASSES, RESULTS, SIZE, SegSet, confusion, load, load_model, miou

SCALES = [0.5, 0.75]
LOGIT_SIZE = (SIZE[0] // 4, SIZE[1] // 4)
METHODS = ["b5", "ens", "conf"]
TEST = [d for d in DOMAINS if d != "cs_fit"]


@torch.inference_mode()
def cache_lowres(domains=DOMAINS):
    b5 = load_model("b5")
    for domain in domains:
        out = CACHE / "e9" / domain
        out.mkdir(parents=True, exist_ok=True)
        for img, _, names in tqdm(DataLoader(SegSet(domain), batch_size=2, num_workers=2), desc=domain):
            logits = {}
            for s in SCALES:
                x = F.interpolate(img.cuda(), scale_factor=s, mode="bilinear", antialias=True).half()
                logits[f"s{round(s * 100)}"] = F.interpolate(b5(pixel_values=x).logits.float(), LOGIT_SIZE,
                                                             mode="bilinear").half().cpu().numpy()
            for i, name in enumerate(names):
                np.savez_compressed(out / f"{name}.npz", **{k: v[i] for k, v in logits.items()})


def predictions(logits0, logits5):
    p0, p5 = logits0.softmax(0), logits5.softmax(0)
    up = lambda p: F.interpolate(p[None], SIZE, mode="bilinear")[0].argmax(0).byte().numpy()
    b0_wins = p0.max(0).values >= p5.max(0).values
    return {"b5": up(p5), "ens": up((p0 + p5) / 2), "conf": up(torch.where(b0_wins[None], p0, p5))}


def main():
    if not (CACHE / "e9").exists():
        cache_lowres()

    keys = [(m, s) for s in SCALES for m in METHODS]
    confs = {}
    for domain in DOMAINS:
        c = np.zeros((len(keys) + 2, NUM_CLASSES, NUM_CLASSES), np.int64)  # + B0, B5 full
        lowres = sorted((CACHE / "e9" / domain).glob("*.npz"))
        for (gt, b0, b5, logits0), path in zip(load(domain, "gt", "pred_b0", "pred_b5", "logits_b0"), lowres):
            with np.load(path) as z:
                low = {s: torch.from_numpy(z[f"s{round(s * 100)}"]).float() for s in SCALES}
            preds = {s: predictions(torch.from_numpy(logits0).float(), low[s]) for s in SCALES}
            for j, (m, s) in enumerate(keys):
                c[j] += confusion(preds[s][m], gt)
            c[-2] += confusion(b0, gt)
            c[-1] += confusion(b5, gt)
        confs[domain] = c
    confs["pool"] = sum(confs[d] for d in TEST)

    report = {}
    print(f"{'method':10} {'cost':>5}" + "".join(f"{d:>8}" for d in DOMAINS + ["pool"]))
    for j, (m, s) in enumerate(keys):
        row = {}
        for d, c in confs.items():
            m0, m5 = miou(c[-2]), miou(c[-1])
            row[d] = round(100 * (miou(c[j]) - m0) / (m5 - m0), 1)
        report[f"{m}@{s}"] = row
        print(f"{f'{m}@{s}':10} {s * s:5.2f}" + "".join(f"{row[d]:8.1f}" for d in DOMAINS + ["pool"]))

    (RESULTS / "e9.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
