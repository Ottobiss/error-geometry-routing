import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from ae.classify import Logistic, linear_inputs
from ae.metrics import ROBUST_DECILES, deciles, holm, metrics, paired_bootstrap, robust_deciles
from ae.model import PCA, BottleneckAE
from ae.patch_features import frame_features, scalar_columns
from ae.pca import fit_pca
from sae.classify import auroc
from sae.hooks import BLOCKS, pool_to_patches, selection_index, stride


def test_pool_to_patches_mean_matches_loops():
    torch.manual_seed(0)
    Hp, Wp = 4, 6
    for b in BLOCKS[::2]:
        r = stride(b)
        h, w = max(Hp * 16 // r, 1), max(Wp * 16 // r, 1)
        x = torch.randn(h, w, 3)
        got = pool_to_patches(x, b, (Hp, Wp), reduce="mean")
        rows, cols = selection_index(Hp, h, b), selection_index(Wp, w, b)
        for u in range(Hp):
            for v in range(Wp):
                ref = x[rows[u]][:, cols[v]].reshape(-1, 3).mean(0)
                assert torch.allclose(got[u * Wp + v], ref, atol=1e-6)


def test_pca_matches_numpy_and_interface():
    rng = np.random.default_rng(0)
    d, b = 12, 4
    A = rng.normal(size=(d, d))
    h = (rng.normal(size=(5000, d)) @ A + 3.0).astype(np.float32)
    mean, scale = h.mean(0), 2.0
    pca, explained = fit_pca(torch.from_numpy(h).half(), mean, scale, b, batch=777)
    x = (torch.from_numpy(h).half().float().numpy() - mean) / scale
    evals, evecs = np.linalg.eigh(np.cov(x.T, bias=True))
    V = evecs[:, ::-1][:, :b].T
    C = pca.components.numpy()
    assert np.allclose(np.abs(C @ V.T), np.eye(b), atol=1e-4)       # same subspace, sign fixed per row
    assert np.allclose(C @ C.T, np.eye(b), atol=1e-5)
    assert np.isclose(explained.sum(), evals[::-1][:b].sum() / evals.sum(), atol=1e-6)
    xt = pca.normalize(torch.from_numpy(h).half())
    xh = pca.decode(pca.encode(xt))
    ref = x.mean(0) + (x - x.mean(0)) @ V.T @ V
    assert np.allclose(xh.numpy(), ref, atol=1e-3)
    assert torch.allclose(pca.denormalize(xt), torch.from_numpy(h).half().float(), atol=1e-4)


def test_bottleneck_ae_shapes():
    torch.manual_seed(0)
    ae = BottleneckAE(16, 2, mean=np.zeros(16, np.float32), scale=1.0)
    x = torch.randn(10, 16)
    z = ae.encode(x)
    assert z.shape == (10, 2) and ae.decode(z).shape == (10, 16)
    assert ae.config() == {"d": 16, "b": 2, "h": 16}


def test_frame_features_against_positions():
    torch.manual_seed(0)
    Hp, Wp = 2, 3
    dims = {1: 8, 2: 8, 3: 16, 4: 16}
    acts, encs, norms = {}, {"e": {}}, {}
    for b in BLOCKS:
        r, d = stride(b), dims[int(b[1])]
        acts[b] = torch.randn(max(Hp * 16 // r, 1), max(Wp * 16 // r, 1), d) * 2 + 1
        mean, scale = torch.randn(d) * 0.1, 1.5
        norms[b] = (mean, scale)
        p = PCA(d, d // 4, mean=mean, scale=scale)
        p.components.copy_(torch.linalg.qr(torch.randn(d, d))[0][: d // 4])
        encs["e"][b] = p
    out, N = frame_features(acts, encs, norms, (Hp, Wp))
    Z, R = out["e"]
    assert R.shape == N.shape == (Hp * Wp, len(scalar_columns("r"))) == (6, 12)
    assert Z.shape == (6, sum(dims[int(b[1])] // 4 for b in BLOCKS))
    col = 0
    for b in BLOCKS:                                  # check patch (1, 2) by hand
        a, (mean, scale), enc = acts[b], norms[b], encs["e"][b]
        rows, cols = selection_index(Hp, a.shape[0], b)[1], selection_index(Wp, a.shape[1], b)[2]
        x = ((a[rows][:, cols].reshape(-1, a.shape[2]) - mean) / scale)
        r = (enc.decode(enc.encode(x)) - x).pow(2).sum(1) / x.shape[1]
        n = x.pow(2).sum(1) / x.shape[1]
        if stride(b) < 16:
            assert np.allclose(R[5, col:col + 2], [r.mean(), r.max()], atol=1e-5)
            assert np.allclose(N[5, col:col + 2], [n.mean(), n.max()], atol=1e-5)
            col += 2
        else:
            assert np.isclose(R[5, col], r.item(), atol=1e-5) and np.isclose(N[5, col], n.item(), atol=1e-5)
            col += 1


def test_logistic_matches_sklearn_two_row_trick():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n).astype(np.float32) * 3 + 1,
                      "b": rng.normal(size=n).astype(np.float32),
                      "cls": rng.integers(0, 19, n), "row": rng.integers(0, 32, n), "col": rng.integers(0, 64, n)})
    frac = 1 / (1 + np.exp(-(0.8 * X.a - 1 + 0.5 * X.b))).to_numpy()
    frac = np.clip(frac + rng.normal(scale=0.1, size=n), 0, 1).astype(np.float32)
    w = rng.integers(50, 400, n).astype(np.float32)
    num, oh, names = linear_inputs(X, "cityscapes", "cpu")
    assert names == ["a", "b"] and oh.shape == (n, 19 + 8 + 8)
    rows = torch.arange(300)
    for C in (1e-2, 1.0):
        m = Logistic(C).fit(num, oh, torch.from_numpy(frac), torch.from_numpy(w), rows)
        # sklearn: standardised numeric + one-hot, two rows per patch, weights scaled to mean 1 over the patches
        Z = np.concatenate([((num - m.mu) / m.sd).numpy(), oh.numpy()], 1)[:300]
        ws = w[:300] / w[:300].mean()
        sk = LogisticRegression(C=C, tol=1e-10, max_iter=10000).fit(
            np.concatenate([Z, Z]), np.r_[np.ones(300), np.zeros(300)],
            sample_weight=np.r_[ws * frac[:300], ws * (1 - frac[:300])])
        ref = sk.predict_proba(np.concatenate([((num - m.mu) / m.sd).numpy(), oh.numpy()], 1))[:, 1]
        assert np.allclose(m.predict(num, oh), ref, atol=2e-4), C


def test_robust_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    n = 5000
    ent = rng.random(n)
    err = np.clip(ent + rng.normal(scale=0.3, size=n), 0, 1)
    df = pd.DataFrame({"image_id": np.repeat(np.arange(50), 100).astype(str), "patch": np.tile(np.arange(100), 50),
                       "n_valid": 256, "n_err": np.round(err * 256).astype(int), "ent_mean": ent})
    df["err_frac"] = df.n_err / df.n_valid
    df["score"] = err + rng.normal(scale=0.2, size=n)
    edges = np.quantile(ent, np.arange(1, 10) / 10)
    m = metrics(df, edges)
    dec = deciles(ent, edges)
    y50 = df.err_frac.to_numpy() >= 0.5
    ref = np.mean([auroc(df.score.to_numpy()[dec == j], y50[dec == j]) for j in ROBUST_DECILES])
    assert np.isclose(m["auroc_y50_robust"], ref)
    assert np.isclose(m["wrong_pixels_below_p0_entropy_median"],
                      df.n_err[ent < np.median(ent)].sum() / df.n_err.sum(), atol=1e-3)
    robust, pos = robust_deciles(df, edges)
    assert sum(pos) == y50.sum() and all(pos[j] >= 200 for j in robust)
    counts = rng.multinomial(50, np.full(50, 1 / 50), size=20)
    b = df.assign(score=df.ent_mean)
    point, lo, hi, p = paired_bootstrap(df, b, "auroc_y50_robust", edges, counts)
    assert np.isclose(point, ref - metrics(b, edges)["auroc_y50_robust"])
    point, lo, hi, p = paired_bootstrap(df, df, "recall@10%", edges, counts)
    assert point == lo == hi == 0


def test_holm():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.001]), [0.5, 0.002])
