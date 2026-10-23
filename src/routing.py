import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from src.hyperbolic import hyperbolic_uncertainty


def segment_gain(pred, gain, stride=4):
    """Mean gain over each segment (connected region of one predicted class), broadcast to its pixels.

    Segments are found at 1/stride resolution. pred, gain: (H, W) -> (H, W).
    """
    pred_low = pred[::stride, ::stride].cpu().numpy()
    gain_low = gain[::stride, ::stride].float().cpu().numpy()
    segments, count = np.zeros(pred_low.shape, np.int64), 0
    for k in np.unique(pred_low):
        labels, n = ndimage.label(pred_low == k)
        segments[labels > 0] = labels[labels > 0] + count - 1
        count += n
    mean = np.bincount(segments.ravel(), gain_low.ravel(), count) / np.bincount(segments.ravel(), minlength=count)
    score = torch.from_numpy(mean[segments]).float().to(pred.device)
    return score.repeat_interleave(stride, 0).repeat_interleave(stride, 1)[:pred.shape[0], :pred.shape[1]]


class Router(nn.Module):
    """Cascade of a small model with the large-uncertainty head and a large model.

    Uncertainties are hyperbolic (Eq. 15): the small model's own, and its prediction of the large model's. The large
    model runs only if some segment of the small prediction has mean(own - predicted large uncertainty) above the
    threshold; its prediction then replaces the small one on those segments.
    """

    def __init__(self, small, large):
        super().__init__()
        assert small.large_head, "the small model must be trained with data.large_uncertainty"
        self.small, self.large = small, large

    def forward(self, x):
        return self.small(x), self.large(x)

    @torch.no_grad()
    def cascade(self, x, threshold, stride=4):
        """x: (1, 3, H, W) -> prediction, own uncertainty, predicted large uncertainty, routed mask; all (H, W)."""
        logits, dist, large = self.small(x)
        pred, own, large = logits[0].argmax(0), hyperbolic_uncertainty(dist)[0], large[0].sigmoid()
        routed = segment_gain(pred, own - large, stride) > threshold
        if routed.any():
            large_logits = self.large(x)[0]
            pred = torch.where(routed, large_logits[0].argmax(0), pred)
        return pred, own, large, routed
