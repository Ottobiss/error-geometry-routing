import numpy as np
import pandas as pd
import unittest

from routing.posthoc import PostHocRejector


class PostHocRejectorTests(unittest.TestCase):
    def test_ranks_tiles_where_b2_has_more_to_fix(self):
        # x=1: B0 makes many errors while B2 makes few.  x=0: the two models
        # are similarly costly. Positive scores must favour the former.
        n = 80
        x = np.repeat([0.0, 1.0], n)
        frame = pd.DataFrame({
            "x": x,
            "n_valid": np.full(2 * n, 100),
            "b0_error": np.where(x == 1, 35, 8),
            "b2_error": np.where(x == 1, 4, 7),
        })
        model = PostHocRejector().fit(frame, ["x"])
        score = model.score(frame, ["x"])
        self.assertGreater(score[x == 1].mean(), score[x == 0].mean())

    def test_validates_required_costs(self):
        frame = pd.DataFrame({"x": [0.0, 1.0], "n_valid": [1, 1]})
        with self.assertRaisesRegex(ValueError, "b0_error"):
            PostHocRejector().fit(frame, ["x"])
