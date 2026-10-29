from torch import nn


class Feature(nn.Module):
    """forward(out) scores one light-model output (models.common.Out, batch 1) as a 0-dim tensor,
    higher = the heavy model is more needed. `taps` names the light-model taps the feature reads.
    A feature with a fit accumulates clean frames in update(out) and completes the fit in finish();
    fitted tensors are buffers, so they are saved with state_dict()."""
    taps = ()

    def update(self, out):
        pass

    def finish(self):
        pass
