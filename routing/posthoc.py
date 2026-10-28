"""Post-hoc rejector for fixed light/heavy models.

Implementation of the second strategy in Narasimhan et al. (NeurIPS 2022),
"Post-hoc Estimators for Learning to Defer to an Expert".  The segmentation
models are frozen.  Only a small router is fitted from features available after
the B0 pass.

For one unit (a tile here), let ``e0`` and ``e2`` be the numbers of valid
pixels misclassified by B0 and B2.  Calling B2 has a configurable cost ``c``
measured in equivalent erroneous pixels.  The desired decision is therefore

    defer to B2 iff E[e0 | x] > E[e2 + c * n_valid | x].

The paper's partial rejector loss is

    e0 * softplus(-r(x)) + (e2 + c*n) * softplus(r(x)),

where positive ``r`` means defer.  A weighted ordinary logistic regression on
two copies of every tile gives exactly this objective, without retraining B0 or
B2.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PostHocRejector:
    """Cost-sensitive B0 -> B2 router trained with the post-hoc loss.

    ``defer_cost`` is a per-valid-pixel cost of a B2 tile.  Zero learns the
    quality-only ordering; a positive value makes the router more conservative.
    The selection threshold/budget remains a deployment decision and is not
    fitted on P1.
    """

    def __init__(self, defer_cost=0.0, c=1.0, max_iter=1000):
        if defer_cost < 0 or c <= 0:
            raise ValueError("defer_cost must be nonnegative and c positive")
        self.defer_cost = float(defer_cost)
        self.c = float(c)
        self.max_iter = int(max_iter)

    def fit(self, frame, fields):
        required = {"b0_error", "b2_error", "n_valid", *fields}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("Post-hoc rejector needs columns: " + ", ".join(sorted(missing)))

        d = frame.loc[frame.n_valid > 0]
        if d.empty:
            raise ValueError("No labelled tiles for post-hoc rejector")
        x = d.loc[:, fields].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite router features")

        # Costs in the notation of Eq. (14) in Narasimhan et al.
        c_model = d.b0_error.to_numpy(float)
        c_expert = d.b2_error.to_numpy(float) + self.defer_cost * d.n_valid.to_numpy(float)
        weights = np.concatenate([c_model, c_expert])
        labels = np.concatenate([np.ones(len(d), dtype=np.int8), np.zeros(len(d), dtype=np.int8)])
        design = np.concatenate([x, x])
        keep = weights > 0
        if not np.any(keep & (labels == 1)) or not np.any(keep & (labels == 0)):
            raise ValueError("Both B0 and B2 must incur positive training cost")

        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=self.c, max_iter=self.max_iter, random_state=0),
        )
        self.model.fit(design[keep], labels[keep], logisticregression__sample_weight=weights[keep])
        return self

    def score(self, frame, fields):
        if not hasattr(self, "model"):
            raise RuntimeError("Fit the post-hoc rejector before scoring")
        x = frame.loc[:, fields].to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite router features")
        # Higher score = lower expected cost for B2 relative to B0.
        return self.model.decision_function(x)
