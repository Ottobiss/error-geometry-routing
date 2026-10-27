import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import SegformerForSemanticSegmentation
from cityscapesscripts.helpers.labels import labels

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ["E8_DATA"])
CACHE = Path(os.environ["E8_CACHE"])
RESULTS = Path(os.environ["E8_RESULTS"])

SIZE = (512, 1024)
NUM_CLASSES = 19
IGNORE = 255

MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

# frankfurt is used to fit scores/routers, the rest is test only
DOMAINS = ["cs_fit", "cs_test", "fog", "night", "rain", "snow"]

_LUT = np.full(256, IGNORE, np.uint8)
for l in labels:
    if 0 <= l.id < 256 and l.trainId not in (-1, 255):
        _LUT[l.id] = l.trainId


def list_domain(domain):
    """Returns sorted (image, gt) path pairs for a domain."""
    if domain.startswith("cs_"):
        cities = ["frankfurt"] if domain == "cs_fit" else ["lindau", "munster"]
        imgs = [p for c in cities for p in (DATA / "cityscapes/leftImg8bit/val" / c).glob("*.png")]
        gt = lambda p: Path(str(p).replace("leftImg8bit", "gtFine", 1)
                            .replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
    else:
        imgs = list((DATA / "acdc/rgb_anon" / domain / "val").rglob("*_rgb_anon.png"))
        gt = lambda p: Path(str(p).replace("rgb_anon", "gt", 1)
                            .replace("_rgb_anon.png", "_gt_labelTrainIds.png"))
    return [(p, gt(p)) for p in sorted(imgs)]


def image_tensor(image):
    # Tensor bilinear antialiased resize, matching the original torchvision v2 intent.
    import torch.nn.functional as F
    a = np.array(image, dtype=np.uint8, copy=True)
    x = torch.from_numpy(a).permute(2,0,1).float().div(255)
    x = F.interpolate(x[None], size=SIZE, mode="bilinear", align_corners=False, antialias=True)[0]
    mean = torch.tensor([.485,.456,.406])[:,None,None]
    std = torch.tensor([.229,.224,.225])[:,None,None]
    return (x-mean)/std


class SegSet(Dataset):
    def __init__(self, domain):
        self.domain = domain
        self.items = list_domain(domain)
        self.img_tf = image_tensor


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, gt_path = self.items[i]
        gt = np.array(Image.open(gt_path))
        if self.domain.startswith("cs_"):
            gt = _LUT[gt]
        gt = Image.fromarray(gt).resize(SIZE[::-1], Image.NEAREST)
        return self.img_tf(Image.open(img_path).convert("RGB")), torch.from_numpy(np.array(gt)), img_path.stem


def load_model(name, device="cuda"):
    model = SegformerForSemanticSegmentation.from_pretrained(MODELS[name])
    return model.half().to(device).eval()


def confusion(pred, gt):
    m = gt != IGNORE
    return np.bincount(NUM_CLASSES * gt[m].astype(np.int64) + pred[m],
                       minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def miou(conf):
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf) + 1e-9)
    return 100 * iou[conf.sum(1) > 0].mean()


TILE = 64  # 512x1024 -> 8x16 tiles


def load(domain, *keys):
    """Yields cached arrays of every image in a domain, in a fixed order."""
    for p in sorted((CACHE / domain).glob("*.npz")):
        with np.load(p) as z:
            yield [z[k] for k in keys]


def tile_sum(x):
    h, w = x.shape
    return x.reshape(h // TILE, TILE, w // TILE, TILE).sum((1, 3))


def gain(gt, pred_b0, pred_b5):
    """+1 where only B5 is right, -1 where only B0 is right, 0 elsewhere or on ignore."""
    return ((pred_b5 == gt).astype(np.int8) - (pred_b0 == gt)) * (gt != IGNORE)


def top_tiles(scores, budget):
    """Boolean mask of the top `budget` share of tiles by score, over the whole domain."""
    order = np.argsort(-scores.ravel(), kind="stable")
    m = np.zeros(scores.size, bool)
    m[order[:round(budget * scores.size)]] = True
    return m.reshape(scores.shape)


def route_confs(domain, scores, budgets):
    """Per-image confusions (budgets x images x C x C) when top_tiles get B5."""
    masks = [top_tiles(scores, b) for b in budgets]
    confs = np.zeros((len(budgets), len(scores), NUM_CLASSES, NUM_CLASSES), np.int64)
    for i, (gt, b0, b5) in enumerate(load(domain, "gt", "pred_b0", "pred_b5")):
        for j, m in enumerate(masks):
            px = m[i].repeat(TILE, 0).repeat(TILE, 1)
            confs[j, i] = confusion(np.where(px, b5, b0), gt)
    return confs


def route_curve(domain, scores, budgets):
    """mIoU per budget, see route_confs."""
    return [miou(c.sum(0)) for c in route_confs(domain, scores, budgets)]
