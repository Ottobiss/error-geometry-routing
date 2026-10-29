"""What every model shares: the output type and capturing the inputs of internal modules."""
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Out:
    logits: torch.Tensor                       # (B, C, h, w), float32
    taps: dict = field(default_factory=dict)   # tap name -> module input tensor


@contextmanager
def capture(model, taps, names):
    """Within the block, the inputs of the tapped modules are stored in the yielded dict.
    names maps a tap name to a module path; an unknown tap name is used as the path itself."""
    captured = {}
    hook = lambda name: lambda module, inputs: captured.__setitem__(name, inputs[0])
    handles = [model.get_submodule(names.get(t, t)).register_forward_pre_hook(hook(t)) for t in taps]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
