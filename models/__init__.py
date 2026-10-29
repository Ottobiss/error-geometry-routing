"""Segmentation models, one file per model. A model is an nn.Module whose forward(images, scale=1.0, taps=())
takes [0, 1] images (B, 3, H, W), optionally downscaled by `scale`, and returns models.common.Out: logits plus
the inputs of the tapped modules. Taps have model-level names (e.g. "decoder"), so features do not depend on
the architecture. To add a model: a file with its class and a branch in load_model."""
from models.segformer import SegFormer


def load_model(name, device="cuda"):
    """A model by its config name, e.g. "segformer-b0", in fp16 eval mode."""
    if name.startswith("segformer-"):
        model = SegFormer(name.removeprefix("segformer-"))
    else:
        raise KeyError(f"unknown model {name!r}")
    return model.to(device).eval().half()
