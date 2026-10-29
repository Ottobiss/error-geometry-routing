"""Patch error score of SegFormer-B0 from bottleneck-AE features (R, Z, ZR, F3F0NR)."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoost, Pool

from sae_lib import (BLOCKS, PATCH, MixFFNTap, frame_features, load_encoder, load_model, output_features,
                          pool_to_patches)

F0 = ["ent_mean", "ent_max", "ent_p90", "msp_mean", "msp_min", "margin_mean", "margin_min",
      "boundary_frac", "row", "col", "cls"]
PARTS = {"R": ("F0", "N", "R"), "Z": ("F0", "Z"), "ZR": ("F0", "N", "Z", "R"), "F3F0NR": ("raw", "F0", "N", "R")}


class ErrorScorer:
    """features: "R", "Z", "ZR" or "F3F0NR"; root: the results/ae folder of the SAE project, with
    ckpt/ae_b8/<block>.pt and runs/<features>_ae_b8_catboost/model.cbm."""

    def __init__(self, features, root="results/ae", device="cuda", b0=None):
        root = Path(root)
        self.parts = PARTS[features]
        self.b0 = b0 if b0 is not None else load_model("b0", device)
        self.enc = {b: load_encoder(root / "ckpt" / "ae_b8" / f"{b}.pt", device)[0] for b in BLOCKS}
        self.norms = {b: (e.mean, e.scale.item()) for b, e in self.enc.items()}
        self.model = CatBoost()
        self.model.load_model(str(root / "runs" / f"{features}_ae_b8_catboost" / "model.cbm"))

    @torch.inference_mode()
    def __call__(self, small):
        """small: (3, H, W) uint8 RGB frame at the working resolution, H and W multiples of 16.
        Returns score (Hp, Wp), B0 pred (H, W), probs (19, H, W) and the per-patch features N, Z, R
        (and raw for F3F0NR)."""
        H, W = small.shape[1:]
        hw = (H // PATCH, W // PATCH)
        with MixFFNTap(self.b0) as tap:
            probs = self.b0(small[None].to(self.b0.mean.device), (W, H)).softmax(1)[0]
            pred = probs.argmax(0)
            enc_out, N = frame_features(tap.acts, {"ae": self.enc}, self.norms, hw)
            if "raw" in self.parts:      # max over the patch positions, stored as fp16 (as in training)
                raw = np.concatenate([pool_to_patches(tap.acts[b].float(), b, hw).half().cpu().numpy()
                                      for b in BLOCKS], axis=1).astype(np.float32)
        Z, R = enc_out["ae"]
        f0 = output_features(probs, pred)
        feats = {"N": N, "Z": Z, "R": R, **({"raw": raw} if "raw" in self.parts else {})}
        cols = {"F0": pd.DataFrame({c: f0[c] for c in F0}), **{k: pd.DataFrame(v) for k, v in feats.items()}}
        X = pd.concat([cols[p] for p in self.parts], axis=1)
        X.columns = self.model.feature_names_
        X["cls"] = X["cls"].astype(np.int32)
        score = self.model.predict(Pool(X, cat_features=["cls"]), prediction_type="Probability")[:, 1]
        return {"score": score.reshape(hw).astype(np.float32), "pred": pred, "probs": probs, **feats}
