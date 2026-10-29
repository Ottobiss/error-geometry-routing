"""Per-frame choice of the compute level under an average budget, online.

Levels (cost = share of one full heavy-model run, on top of the light model):
  light     the light model alone                                         0
  ens@0.5   mean of light and heavy probabilities, heavy on a 0.5x frame  0.25
  ens@0.75  the same with a 0.75x frame                                   0.5625
  heavy     the heavy model on the full frame                             1
Rule (E10/E11): a frame whose `flag` score is above the clean-data quantile gets ens@0.75 (ens@0.5 or light
when the credit does not allow it); any frame whose `priority` score is in the top share of the recent window
is upgraded to heavy. The top share follows the budget left after the flagged frames, and a credit (each frame
adds `budget`, each level spends its cost) keeps the long-run average within the budget.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Level:
    name: str
    scale: float = 0.0     # heavy-model input scale, 0 = no heavy run
    ensemble: bool = False

    @property
    def cost(self):
        return self.scale ** 2


LEVELS = {l.name: l for l in [Level("light"), Level("ens@0.5", 0.5, True), Level("ens@0.75", 0.75, True),
                              Level("heavy", 1.0)]}


class Router:
    def __init__(self, budget=0.5, quantile=0.95, window=50, flagged="ens@0.75", fallback="ens@0.5",
                 max_debt=1.0, max_credit=2.0):
        self.budget, self.quantile, self.window = budget, quantile, window
        self.flagged, self.fallback = LEVELS[flagged], LEVELS[fallback]
        self.max_debt, self.max_credit = max_debt, max_credit
        self.threshold = None
        self.reset()

    def reset(self):
        """Starts a new stream: empty window and zero credit."""
        self.credit = 0.0
        self.priorities, self.flags = deque(maxlen=self.window), deque(maxlen=self.window)

    def fit(self, flag_scores):
        self.threshold = float(np.quantile(flag_scores, self.quantile))

    def affordable(self, level):
        return self.credit - level.cost >= -self.max_debt

    def decide(self, flag, priority) -> Level:
        self.credit = min(self.credit + self.budget, self.max_credit)
        is_flagged = flag is not None and flag > self.threshold
        self.flags.append(is_flagged)
        self.priorities.append(priority)

        base = LEVELS["light"]
        if is_flagged:
            base = next(l for l in (self.flagged, self.fallback, base) if self.affordable(l))
        # share of frames that can go to heavy with the budget left after the flagged ones
        flagged_cost = np.mean(self.flags) * self.flagged.cost
        share = np.clip((self.budget - flagged_cost) / (1 - flagged_cost), 0, 1)
        level = base
        if share > 0 and priority >= np.quantile(self.priorities, 1 - share) and self.affordable(LEVELS["heavy"]):
            level = LEVELS["heavy"]
        self.credit -= level.cost
        return level

