import numpy as np
import torch

from core.components import comp_quantile
from features.prototypes import component_aggregates, similarities


def test_similarities():
    P = torch.eye(3, 4)                                   # 3 прототипа в R^4
    z = torch.tensor([[0.8, 0.0], [0.6, 0.0], [0.0, 1.0], [0.0, 0.0]])   # 2 пикселя
    s_pred, s_max, margin, flag = similarities(P, z, torch.tensor([1, 2]))
    assert torch.allclose(s_pred, torch.tensor([0.6, 1.0]))
    assert torch.allclose(s_max, torch.tensor([0.8, 1.0]))
    assert torch.allclose(margin, torch.tensor([0.2, 1.0]))
    assert flag.tolist() == [True, False]


def test_component_aggregates():
    comp = torch.tensor([0, 0, 0, 1])
    s_pred = torch.tensor([0.1, 0.5, 0.3, 0.9])
    a = component_aggregates(s_pred, s_pred * 0 + 0.2, torch.tensor([1, 0, 0, 1]).bool(),
                             comp, 2, thr=0.4)
    assert np.allclose(a["spred_mean"].numpy(), [0.3, 0.9])
    assert np.allclose(a["spred_min"].numpy(), [0.1, 0.9])
    assert np.allclose(a["spred_p10"].numpy(), [0.1, 0.9])
    assert np.allclose(a["flag_frac"].numpy(), [1 / 3, 1.0])
    assert np.allclose(a["low_frac"].numpy(), [2 / 3, 0.0])


def test_comp_quantile():
    x = np.array([5.0, 1.0, 3.0, 2.0, 9.0])
    comp = np.array([0, 0, 0, 1, 1])
    assert np.allclose(comp_quantile(x, comp, 2, 0.5), [3.0, 2.0])
