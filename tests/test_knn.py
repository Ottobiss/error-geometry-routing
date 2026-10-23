import numpy as np
import pytest
import torch
import torch.nn.functional as F

from features import knn as bk

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="нужен GPU")
H, W = 512, 1024


@cuda
def test_token_labels_majority_without_ignore():
    gt = np.full((H, W), 255, dtype=np.uint8)
    gt[:4, :4] = 1
    gt[0, 0] = 2                      # 15 пикселей класса 1 против 1 пикселя класса 2
    gt[:4, 4:8] = 255                 # ячейка из одних ignore
    gt[:2, 8:12] = 3                  # 8 пикселей класса 3, остальные ignore — большинство 3
    lab = bk.token_labels(gt, 4).cpu().numpy()
    assert lab[0] == 1 and lab[1] == -1 and lab[2] == 3


@cuda
def test_bank_scores_match_bruteforce():
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(300, 8, generator=g).cuda()
    q = torch.randn(50, 8, generator=g).cuda()
    refn, qn = F.normalize(ref, dim=1), F.normalize(q, dim=1)
    k, K = 3, 10
    bank = {"ref": ref, "refn": refn, "rs": bk.rs_within(refn, K), "rsb": torch.rand(300).cuda()}
    sc = bk.bank_scores(q, qn, bank, k, K)
    R, Q, Rn, Qn = (x.cpu().numpy().astype(np.float64) for x in (ref, q, refn, qn))
    d = np.linalg.norm(Q[:, None] - R[None], axis=2)
    assert np.allclose(sc["dnp"].cpu(), np.sort(d, 1)[:, :k].mean(1), atol=1e-4)
    cos = Qn @ Rn.T
    assert np.allclose(sc["cos"].cpu(), 1 - np.sort(cos, 1)[:, ::-1][:, :k].mean(1), atol=1e-5)
    self_cos = Rn @ Rn.T
    np.fill_diagonal(self_cos, -np.inf)
    rs = np.sort(self_cos, 1)[:, ::-1][:, :K].mean(1)
    assert np.allclose(bank["rs"].cpu(), rs, atol=1e-5)
    r_t = np.sort(cos, 1)[:, ::-1][:, :K].mean(1)
    csls = 2 * cos - r_t[:, None] - rs[None]
    assert np.allclose(sc["csls"].cpu(), 1 - np.sort(csls, 1)[:, ::-1][:, :k].mean(1), atol=1e-5)


@cuda
def test_running_topk():
    g = torch.Generator().manual_seed(1)
    refn = F.normalize(torch.randn(20, 5, generator=g), dim=1).cuda()
    qs = [F.normalize(torch.randn(30, 5, generator=g), dim=1).cuda() for _ in range(3)]
    rt = bk.RunningTopK(20, 4)
    for qn in qs:
        rt.update(refn, qn, step=7)
    full = (refn @ torch.cat(qs).T).topk(4, dim=1).values
    assert torch.allclose(rt.best, full)


@cuda
def test_up_nearest():
    tok = torch.arange(H // 32 * (W // 32), dtype=torch.float32, device="cuda")
    m = bk.up(tok, 32, (H, W)).reshape(H, W)
    assert m[0, 0] == 0 and m[31, 31] == 0 and m[0, 32] == 1 and m[32, 0] == W // 32
