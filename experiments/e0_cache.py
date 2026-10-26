"""E0: run B0 and B5 once, cache everything later experiments need, report the mIoU gap.

Per image (experiments/cache/<domain>/<name>.npz):
  gt, pred_b0, pred_b5  - uint8, 512x1024
  logits_b0             - fp16, 19 x 128 x 256 (1/4)
  feat_b0               - fp16, 256 x 64 x 128 (decoder features before classifier, 1/8)
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CACHE, DOMAINS, RESULTS, SIZE, NUM_CLASSES, SegSet, confusion, load_model, miou


@torch.inference_mode()
def main():
    b0, b5 = load_model("b0"), load_model("b5")
    feats = {}
    b0.decode_head.classifier.register_forward_pre_hook(lambda m, x: feats.update(f=x[0]))

    report = {}
    for domain in DOMAINS:
        out = CACHE / domain
        out.mkdir(parents=True, exist_ok=True)
        conf = {k: np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64) for k in ("b0", "b5")}

        for img, gt, names in tqdm(DataLoader(SegSet(domain), batch_size=2, num_workers=2), desc=domain):
            img = img.cuda().half()
            logits0 = b0(pixel_values=img).logits
            feat0 = F.avg_pool2d(feats["f"], 2)
            up = lambda x: F.interpolate(x.float(), SIZE, mode="bilinear").argmax(1).byte().cpu().numpy()
            pred0, pred5 = up(logits0), up(b5(pixel_values=img).logits)

            for i, name in enumerate(names):
                g = gt[i].numpy()
                conf["b0"] += confusion(pred0[i], g)
                conf["b5"] += confusion(pred5[i], g)
                np.savez_compressed(out / f"{name}.npz", gt=g, pred_b0=pred0[i], pred_b5=pred5[i],
                                    logits_b0=logits0[i].cpu().numpy(), feat_b0=feat0[i].cpu().numpy())

        report[domain] = {k: round(miou(c), 2) for k, c in conf.items()}

    print(f"{'domain':8} {'b0':>6} {'b5':>6} {'gap':>6}")
    for d, r in report.items():
        print(f"{d:8} {r['b0']:6.2f} {r['b5']:6.2f} {r['b5'] - r['b0']:6.2f}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e0.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
