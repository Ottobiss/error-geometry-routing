from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2
from torchvision.utils import make_grid

IGNORE = 255
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLASSES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
           "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
LABEL_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)
COLORS = ((128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
          (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
          (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32))

ID_TO_TRAIN = torch.full((256,), IGNORE, dtype=torch.uint8)
ID_TO_TRAIN[list(LABEL_IDS)] = torch.arange(len(LABEL_IDS), dtype=torch.uint8)
PALETTE = torch.zeros(256, 3, dtype=torch.uint8)
PALETTE[:len(COLORS)] = torch.tensor(COLORS, dtype=torch.uint8)


class Cityscapes(Dataset):
    """leftImg8bit/<split>/<city>/*_leftImg8bit.png + gtFine/<split>/<city>/*_gtFine_labelIds.png; split may be a list,
    e.g. [train, val] for all labelled ACDC.

    crop (H, W) enables training augmentation: random resize of base_size (W, H) by scale_range, crop, flip.
    large_uncertainty: output dir of src.export_uncertainty for uncertainty_split. Items then also carry the large
    model's uncertainty, augmented together with the label, NaN where it is no target: images of uncertainty_split
    have it on their labelled pixels, and only images of the train split keep their labels (the others get 255).
    """

    def __init__(self, root, split, crop=None, base_size=(2048, 1024), scale_range=(0.5, 2.0), large_uncertainty=None,
                 uncertainty_split="val"):
        splits = dict.fromkeys([split] if isinstance(split, str) else split)
        self.images = sorted(p for s in splits for p in Path(root, "leftImg8bit", s).glob("*/*_leftImg8bit.png"))
        self.labels = [Path(root, "gtFine", *p.parts[-3:-1], p.name.replace("leftImg8bit", "gtFine_labelIds"))
                       for p in self.images]
        self.large_uncertainty, self.uncertainty_split = large_uncertainty, uncertainty_split
        augment = [
            v2.ScaleJitter(tuple(base_size), tuple(scale_range)),
            v2.RandomCrop(tuple(crop), pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE}),
            v2.RandomHorizontalFlip(),
        ] if crop else []
        self.transform = v2.Compose([*augment, v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = tv_tensors.Image(read_image(str(self.images[i]), ImageReadMode.RGB))
        label = tv_tensors.Mask(ID_TO_TRAIN[read_image(str(self.labels[i]))[0].long()])
        if self.large_uncertainty is None:
            image, label = self.transform(image, label)
            return image.as_subclass(torch.Tensor), label.as_subclass(torch.Tensor).long()

        split, city, name = self.images[i].parts[-3:]
        unc = torch.full(label.shape, torch.nan)
        if split == self.uncertainty_split:  # saved at reduced resolution
            saved = read_image(str(Path(self.large_uncertainty, split, city, name)))[None].float() / 255
            unc = F.interpolate(saved, label.shape, mode="bilinear", align_corners=False)[0, 0]
        image, label, unc = self.transform(image, label, tv_tensors.Mask(unc))
        label, unc = label.as_subclass(torch.Tensor).long(), unc.as_subclass(torch.Tensor)
        unc = torch.where(label == IGNORE, torch.nan, unc)  # a target on labelled pixels only (not on padding)
        if split != "train":  # segmentation is learned on train only
            label = torch.full_like(label, IGNORE)
        return image.as_subclass(torch.Tensor), label, unc


def colorize(label):
    """(H, W) class ids -> (3, H, W) uint8"""
    return PALETTE[label.cpu().long()].permute(2, 0, 1)


def heatmap(x):
    """(H, W) values in [0, 1] -> (3, H, W) uint8, black -> red -> yellow -> white"""
    x = x.cpu().float()
    return (torch.stack([3 * x, 3 * x - 1, 3 * x - 2]).clamp(0, 1) * 255).byte()


def denormalize(image):
    mean, std = torch.tensor(MEAN)[:, None, None], torch.tensor(STD)[:, None, None]
    return ((image.cpu().float() * std + mean).clamp(0, 1) * 255).byte()


def make_panel(tiles, cols, scale=0.5):
    grid = make_grid(tiles, nrow=cols, padding=8, pad_value=255)
    return F.interpolate(grid[None].float(), scale_factor=scale, mode="area")[0].byte()
