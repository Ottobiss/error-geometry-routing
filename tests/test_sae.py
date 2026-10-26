import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from sae.classify import auroc, recall_at, training_rows
from sae.hooks import BLOCKS, MixFFNTap, pool_to_patches, selection_index, stride
from sae.model import BatchTopKSAE
from sae.patch_features import boundary_map, labels, per_patch


def test_selection_index_cityscapes_and_acdc():
    for n_rows in (32, 36):                      # patch rows: CS 512 / 16, ACDC 576 / 16
        H = n_rows * 16
        for b in BLOCKS:
            r = stride(b)
            idx = selection_index(n_rows, H // r, b)
            for u in range(n_rows):
                centres = r * idx[u]
                if r < 16:                       # all positions with the centre inside the patch
                    assert list(centres) == list(range(16 * u, 16 * u + 16, r))
                else:                            # the one closest to the patch centre
                    c = 16 * u + 7.5
                    best = np.argmin(np.abs(r * np.arange(H // r) - c))
                    assert idx[u].tolist() == [best]
    assert selection_index(32, 32, "s3b1")[:, 0].tolist() == list(range(32))
    s4 = selection_index(32, 16, "s4b1")[:, 0]
    assert s4.tolist() == [min((u + 1) // 2, 15) for u in range(32)]


def test_pool_to_patches_matches_loops():
    torch.manual_seed(0)
    Hp, Wp = 4, 6
    for b in BLOCKS[::2]:
        r = stride(b)
        h, w = max(Hp * 16 // r, 1), max(Wp * 16 // r, 1)
        x = torch.randn(h, w, 5)
        got = pool_to_patches(x, b, (Hp, Wp))
        rows, cols = selection_index(Hp, h, b), selection_index(Wp, w, b)
        for u in range(Hp):
            for v in range(Wp):
                ref = x[rows[u]][:, cols[v]].reshape(-1, 5).amax(0)
                assert torch.equal(got[u * Wp + v], ref)


@pytest.fixture(scope="module")
def tiny_b0():
    """SegModel-like wrapper around a randomly initialised MiT-B0 (SegformerConfig defaults are B0)."""
    torch.manual_seed(0)
    model = torch.nn.Module()
    model.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=19)).eval()
    return model


@torch.inference_mode()
def test_tap_shapes_and_unchanged_logits(tiny_b0):
    x = torch.randn(1, 3, 128, 256)
    ref = tiny_b0.net(pixel_values=x).logits
    with MixFFNTap(tiny_b0) as tap:
        out = tiny_b0.net(pixel_values=x).logits
    assert torch.equal(ref, out)
    dims = {1: 128, 2: 256, 3: 640, 4: 1024}
    for b in BLOCKS:
        r = stride(b)
        assert tuple(tap.acts[b].shape) == (128 // r, 256 // r, dims[int(b[1])])
    # Splice: identity keeps logits, a different input changes them.
    with MixFFNTap(tiny_b0, blocks=[], replace={"s2b1": lambda h: h.clone()}):
        assert torch.equal(tiny_b0.net(pixel_values=x).logits, ref)
    with MixFFNTap(tiny_b0, blocks=["s2b1"], replace={"s2b1": torch.zeros_like}) as tap:
        assert not torch.equal(tiny_b0.net(pixel_values=x).logits, ref)
        assert tap.acts["s2b1"].abs().sum() > 0          # acts keep the original activations
    assert not tap.handles


def test_batch_topk_and_decoder_norm():
    torch.manual_seed(0)
    sae = BatchTopKSAE(16, 32, 4)
    x = torch.randn(64, 16)
    loss, mse, z = sae.loss(x)
    assert (z > 0).sum() == 4 * 64
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    loss.backward()
    sae.project_decoder_grad()
    opt.step()
    sae.normalize_decoder()
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(32), atol=1e-6)
    sae.threshold.fill_(sae.batch_cutoff(x).item())
    assert (sae.encode(x) > 0).sum(1).float().mean() > 0


def test_auroc_ties_and_weights():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 300).astype(float)    # many ties
    y = rng.random(300) < 0.3
    assert np.isclose(auroc(s, y), roc_auc_score(y, s))
    w = rng.integers(0, 4, 300)
    rep = np.repeat(np.arange(300), w)
    assert np.isclose(auroc(s, y, w.astype(float)), roc_auc_score(y[rep], s[rep]))


def test_recall_at_budget_and_weights():
    score = np.array([0.9, 0.8, 0.1, 0.5])
    n_err = np.array([10, 0, 5, 1])
    order = np.argsort(-score, kind="stable")
    assert recall_at(order, n_err, 0.5) == 10 / 16          # top 2 patches: 10 + 0
    w = np.array([2.0, 0.0, 1.0, 1.0])                       # frame resample as multiplicities
    rep = np.repeat(np.arange(4), w.astype(int))
    o2 = np.argsort(-score[rep], kind="stable")
    assert np.isclose(recall_at(order, n_err, 0.5, w), recall_at(o2, n_err[rep], 0.5))


def test_patch_labels_and_boundary():
    gt = np.zeros((32, 32), np.uint8)
    gt[:16, :16] = 255                                       # patch 0 has no valid pixels
    pred = np.zeros((32, 32), np.int64)
    pred[16:, 16:20] = 1                                     # 64 wrong pixels in patch 3
    n_valid, n_err = labels(gt, pred)
    assert n_valid.tolist() == [0, 256, 256, 256] and n_err.tolist() == [0, 0, 0, 64]
    b = boundary_map(pred)
    assert b[16:, [15, 16, 19, 20]].all() and b[15:17, 15:21].all()
    assert not b[17:, 17:19].any() and not b[:, :15].any() and not b[:, 21:].any() and not b[:15].any()
    assert per_patch(np.arange(32 * 32).reshape(32, 32))[1, 0] == 16


def test_sparse_codes_are_zeros_for_catboost():
    import pandas as pd
    import scipy.sparse as sp
    from catboost import CatBoost, Pool
    from sae.classify import sparse_frame

    m = sp.random(500, 10, density=0.2, format="csr", dtype=np.float32, random_state=0)
    names = [f"c{j}" for j in range(10)]
    sparse = sparse_frame(m, names)
    assert (sparse.sparse.to_dense().to_numpy() == m.toarray()).all()
    dense = pd.DataFrame(m.toarray(), columns=names)
    cls = np.arange(500) % 3
    sparse["cls"], dense["cls"] = cls, cls
    y = np.clip(m[:, 0].toarray().ravel() + 0.3 * (cls == 1), 0, 1)
    params = dict(loss_function="CrossEntropy", iterations=20, verbose=0, random_seed=0, allow_writing_files=False)
    a = CatBoost(params).fit(Pool(dense, y, cat_features=["cls"])).predict(Pool(dense, cat_features=["cls"]))
    b = CatBoost(params).fit(Pool(sparse, y, cat_features=["cls"])).predict(Pool(sparse, cat_features=["cls"]))
    assert np.allclose(a, b)


def test_training_rows():
    import pandas as pd
    meta = pd.DataFrame({"n_err": [0] * 8 + [3, 1], "n_valid": [100] * 10})
    keep, w = training_rows(meta)
    assert keep[8:].all() and keep[:8].sum() == 2
    assert (w[:8] == 400).all() and (w[8:] == 100).all()
