"""Light model on every frame -> frame features -> router -> heavy-model path -> segmentation."""
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from features import FEATURES
from loaders import unpack
from models import load_model
from router import Router

FIT_KEYS = ("light", "heavy", "flag", "priority", "quantile")  # the fitted state depends on these


@dataclass
class Config:
    """All settings; defaults here, overrides in config.yaml."""
    light: str = "segformer-b0"
    heavy: str = "segformer-b5"
    flag: str | None = "knn"    # feature that marks frames unlike the fit data -> ens@0.75; None = entropy-only routing
    priority: str = "entropy"   # feature that ranks frames for the full heavy model
    budget: float = 0.5         # mean heavy-model compute per frame, share of one full run
    quantile: float = 0.95      # flag threshold quantile on the fit frames
    window: int = 50            # recent frames for the priority quantile
    data: str = "data"          # data root
    state: str = "state.pt"     # fitted features and threshold
    fit_set: str = "cs_fit"     # clean frames: a named set or a glob of images
    eval_sets: list = field(default_factory=lambda: ["cs_test", "acdc_fog", "acdc_night", "acdc_rain", "acdc_snow"])
    results: str = "results/evaluate.json"
    shuffle: int | None = None  # seed to shuffle the evaluation stream, None = sets one after another
    warmup: int = 5             # frames run before timing
    limit: int | None = None    # frames per set, for a quick check
    device: str = "cuda"

    @classmethod
    def load(cls, path="config.yaml"):
        path = Path(path)
        values = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise KeyError(f"unknown keys in {path}: {sorted(unknown)}")
        return cls(**values)

    def fit_part(self):
        return {k: v for k, v in asdict(self).items() if k in FIT_KEYS}


@dataclass
class Result:
    pred: torch.Tensor  # (H, W) uint8 class ids at the input size
    level: str
    scores: dict        # feature name -> float
    times: dict         # stage -> ms


class Pipeline:
    """Orchestration of nn.Modules (models, features) and the stateful router of one stream."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.light, self.heavy = load_model(cfg.light, cfg.device), load_model(cfg.heavy, cfg.device)
        if self.light.num_classes != self.heavy.num_classes:
            raise ValueError("light and heavy models must predict the same classes")
        names = dict.fromkeys(n for n in (cfg.flag, cfg.priority) if n)
        self.features = nn.ModuleDict({n: FEATURES[n]() for n in names}).to(cfg.device)
        self.taps = tuple(dict.fromkeys(t for f in self.features.values() for t in f.taps))
        self.router = Router(cfg.budget, cfg.quantile, cfg.window)

    def outputs(self, dataset):
        for img, _, _ in DataLoader(dataset, collate_fn=unpack, num_workers=2):
            yield self.light(img[None], taps=self.taps)

    @torch.inference_mode()
    def fit(self, dataset):
        """Fits the features, then the flag threshold, on clean frames (two passes of the light model)."""
        for out in self.outputs(dataset):
            for f in self.features.values():
                f.update(out)
        for f in self.features.values():
            f.finish()
        if self.cfg.flag:
            self.router.fit([float(self.features[self.cfg.flag](out)) for out in self.outputs(dataset)])

    @torch.inference_mode()
    def __call__(self, image) -> Result:
        """Segments one [0, 1] image (3, H, W); the router keeps its state across calls (one stream)."""
        clock = Clock()
        out = self.light(image[None], taps=self.taps)
        clock("light")
        scores = {n: float(f(out)) for n, f in self.features.items()}
        level = self.router.decide(scores.get(self.cfg.flag), scores[self.cfg.priority])
        clock("route")
        probs = out.logits.softmax(1)
        if level.scale:
            heavy = F.interpolate(self.heavy(image[None], level.scale).logits, probs.shape[-2:], mode="bilinear")
            probs = (probs + heavy.softmax(1)) / 2 if level.ensemble else heavy.softmax(1)
        pred = F.interpolate(probs, image.shape[-2:], mode="bilinear")[0].argmax(0).byte().cpu()
        clock("heavy")
        return Result(pred, level.name, scores, clock.times)

    def save(self):
        torch.save({"fit": self.cfg.fit_part(), "threshold": self.router.threshold,
                    "features": self.features.state_dict()}, self.cfg.state)

    @classmethod
    def load(cls, cfg: Config):
        """Pipeline of cfg with the fitted state from cfg.state; the fit settings must match cfg."""
        state = torch.load(cfg.state, weights_only=False)
        if state["fit"] != cfg.fit_part():
            raise ValueError(f"{cfg.state} was fitted with {state['fit']}, the config has {cfg.fit_part()}: "
                             "run `python main.py fit`")
        pipe = cls(cfg)
        pipe.router.threshold = state["threshold"]
        pipe.features.load_state_dict(state["features"])
        return pipe


class Clock:
    """Wall-clock ms of consecutive stages, with CUDA sync."""

    def __init__(self):
        torch.cuda.synchronize()
        self.t, self.times = time.perf_counter(), {}

    def __call__(self, stage):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.times[stage] = 1000 * (now - self.t)
        self.t = now
