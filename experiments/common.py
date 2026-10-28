from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from transformers import SegformerForSemanticSegmentation
from cityscapesscripts.helpers.labels import labels

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"

SIZE = (512, 1024)
NUM_CLASSES = 19
IGNORE = 255

MODELS = {
    "b0": "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "b5": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}

# frankfurt is used to fit scores/routers, the rest is test only
DOMAINS = ["cs_fit", "cs_test", "fog", "night", "rain", "snow"]
# synthetic shift on cs_test (Cityscapes-C style), severity 1-2, see corrupt()
CORRUPTIONS = [f"c_{k}{s}" for k in ("fog", "dark", "blur", "noise") for s in (1, 2)]

_LUT = np.full(256, IGNORE, np.uint8)
for l in labels:
    if 0 <= l.id < 256 and l.trainId not in (-1, 255):
        _LUT[l.id] = l.trainId


def list_domain(domain):
    """Returns sorted (image, gt) path pairs for a domain."""
    if domain in CORRUPTIONS:
        return list_domain("cs_test")
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


def corrupt(x, domain, seed):
    """Synthetic shift on a [0, 1] image: fog, dark (low light + sensor noise), blur, noise."""
    kind, s = domain[2:-1], int(domain[-1]) - 1
    g = torch.Generator().manual_seed(seed)
    if kind == "fog":  # denser towards the top of the frame
        a = (0.4, 0.6)[s] * torch.linspace(1, 0.5, x.shape[1])[None, :, None]
        return x * (1 - a) + 0.75 * a
    if kind == "dark":
        x = x ** 2 * (0.35, 0.2)[s]
        return (x + (0.02, 0.04)[s] * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind == "blur":
        sigma = (2.0, 4.0)[s]
        return T.functional.gaussian_blur(x, 2 * round(2 * sigma) + 1, sigma)
    return (x + (0.08, 0.16)[s] * torch.randn(x.shape, generator=g)).clamp(0, 1)


class SegSet(Dataset):
    def __init__(self, domain):
        self.domain = domain
        self.items = list_domain(domain)
        self.load_tf = T.Compose([T.ToImage(), T.Resize(SIZE), T.ToDtype(torch.float32, scale=True)])
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, gt_path = self.items[i]
        gt = np.array(Image.open(gt_path))
        if gt_path.name.endswith("_gtFine_labelIds.png"):
            gt = _LUT[gt]
        gt = Image.fromarray(gt).resize(SIZE[::-1], Image.NEAREST)
        img = self.load_tf(Image.open(img_path).convert("RGB"))
        if self.domain in CORRUPTIONS:
            img = corrupt(img, self.domain, i)
        return self.norm(img), torch.from_numpy(np.array(gt)), img_path.stem


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
