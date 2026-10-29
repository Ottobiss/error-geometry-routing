"""Image sets as torch Datasets of (image, gt, name).

image: float [0, 1] tensor (3, H, W) at the working size; gt: uint8 trainIds (H, W) with IGNORE for
unlabelled pixels, or None for sets without labels. Model-specific normalisation is done in models/.
get(name) gives a named set ("cs_fit", "acdc_night", "cs_test:fog1" with a corruption) or, for a glob such
as "frames/*.png", unlabelled images.
"""
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as TF

SIZE = (512, 1024)
IGNORE = 255

# Cityscapes labelIds -> the 19 trainIds
CITYSCAPES_LUT = np.full(256, IGNORE, np.uint8)
for label_id, train_id in {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                           25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}.items():
    CITYSCAPES_LUT[label_id] = train_id


class Images(Dataset):
    """Sorted (image path, label path or None) pairs; labels are mapped through `lut` if given."""

    def __init__(self, pairs, lut=None):
        self.pairs, self.lut = sorted(pairs), lut

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, gt_path = self.pairs[i]
        img = Image.open(img_path).convert("RGB").resize(SIZE[::-1], Image.BILINEAR)
        img = TF.to_dtype(TF.to_image(img), torch.float32, scale=True)
        gt = None
        if gt_path:
            gt = np.array(Image.open(gt_path).resize(SIZE[::-1], Image.NEAREST))
            gt = torch.from_numpy(gt if self.lut is None else self.lut[gt])
        return img, gt, Path(img_path).stem


def cityscapes(root, cities=None, split="val"):
    """<root>/cityscapes/leftImg8bit/<split>/<city>/*.png, labels from gtFine (labelIds)."""
    base = Path(root) / "cityscapes" / "leftImg8bit" / split
    imgs = [p for c in (cities or [d.name for d in base.iterdir()]) for p in (base / c).glob("*.png")]
    label = lambda p: str(p).replace("leftImg8bit", "gtFine", 1).replace("_leftImg8bit.png", "_gtFine_labelIds.png")
    return Images([(str(p), label(p)) for p in imgs], CITYSCAPES_LUT)


def acdc(root, condition, split="val"):
    """<root>/acdc/rgb_anon/<condition>/<split>/<sequence>/*_rgb_anon.png, labels from gt (trainIds)."""
    imgs = (Path(root) / "acdc" / "rgb_anon" / condition / split).rglob("*_rgb_anon.png")
    label = lambda p: str(p).replace("rgb_anon", "gt", 1).replace("_rgb_anon.png", "_gt_labelTrainIds.png")
    return Images([(str(p), label(p)) for p in imgs])


def corrupt(x, kind, severity, seed):
    """Synthetic shift on a [0, 1] image, severity 1-2: fog, dark (low light + sensor noise), blur, noise."""
    s = severity - 1
    g = torch.Generator().manual_seed(seed)
    if kind == "fog":  # denser towards the top of the frame
        a = (0.4, 0.6)[s] * torch.linspace(1, 0.5, x.shape[1])[None, :, None]
        return x * (1 - a) + 0.75 * a
    if kind == "dark":
        x = x ** 2 * (0.35, 0.2)[s]
        return (x + (0.02, 0.04)[s] * torch.randn(x.shape, generator=g)).clamp(0, 1)
    if kind == "blur":
        sigma = (2.0, 4.0)[s]
        return TF.gaussian_blur(x, 2 * round(2 * sigma) + 1, sigma)
    if kind == "noise":
        return (x + (0.08, 0.16)[s] * torch.randn(x.shape, generator=g)).clamp(0, 1)
    raise KeyError(f"unknown corruption {kind!r}")


class Corrupted(Dataset):
    """A set with a corruption such as "fog1" applied to every image (seeded by the index)."""

    def __init__(self, base, corruption):
        self.base, self.kind, self.severity = base, corruption[:-1], int(corruption[-1])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, gt, name = self.base[i]
        return corrupt(img, self.kind, self.severity, i), gt, name


def get(name, root="data"):
    if "*" in name or "/" in name:
        return Images([(p, None) for p in glob(name, recursive=True)])
    name, _, corruption = name.partition(":")
    if name == "cs_fit":
        ds = cityscapes(root, ["frankfurt"])
    elif name == "cs_test":
        ds = cityscapes(root, ["lindau", "munster"])
    elif name == "cs_val":
        ds = cityscapes(root)
    elif name.startswith("acdc_"):
        ds = acdc(root, name.removeprefix("acdc_"))
    else:
        raise KeyError(f"unknown set {name!r}: cs_fit, cs_test, cs_val, acdc_<condition> or a glob of images")
    return Corrupted(ds, corruption) if corruption else ds


def unpack(batch):
    """collate_fn for DataLoader(batch_size=1): returns the single sample as is (gt may be None)."""
    return batch[0]
