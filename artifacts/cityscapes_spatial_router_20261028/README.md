# Cityscapes spatial-router results

`cityscapes_p0_oof.csv` is the five-fold out-of-fold P0 guard for the
Cityscapes-trained spatial B0 router.  Only budgets one and three passed the
predefined P0 approval guard.

`cityscapes_p1_bootstrap.csv` is the held-out P1 comparison against entropy.
The router was used only at approved budgets: it has a confirmed improvement
at three B5 tiles per frame (delta mIoU +0.379997, 95% CI
[+0.014202, +0.733425]).

The ACDC zero-shot result and the inference checkpoint are stored separately
in `artifacts/acdc_zero_shot_20261028/`.
