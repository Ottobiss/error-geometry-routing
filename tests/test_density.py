import numpy as np
import torch

from routing.methods.density import NC, OutputStats, aggregate, merge_samples


def toy_samples(rng, n=500):
    return {c: (rng.normal(size=(n, NC)) + c).astype(np.float16) for c in range(NC)}


def test_md_matches_explicit_formula():
    rng = np.random.default_rng(0)
    samples = toy_samples(rng)
    st = OutputStats("logits", eps=1e-3, device="cpu").fit(samples)
    v = rng.normal(size=(7, NC))
    cls = np.array([0, 3, 3, 5, 18, 0, 7])
    md = st.md(torch.tensor(v, dtype=torch.float32), torch.tensor(cls)).numpy()
    for i, c in enumerate(cls):
        x = samples[c].astype(np.float64)
        S = np.cov(x.T, bias=True) + 1e-3 * np.eye(NC)        # 1/N, как в статье
        d = v[i] - x.mean(0)
        assert np.isclose(md[i], np.sqrt(d @ np.linalg.inv(S) @ d), rtol=1e-3)


def test_softmax_variant_is_finite():
    rng = np.random.default_rng(0)
    for eps in (0.0, 1e-4):
        st = OutputStats("softmax", eps=eps, device="cpu").fit(toy_samples(rng))
        assert torch.isfinite(st.inv).all()


def test_pinv_equals_inv_for_nondegenerate():
    rng = np.random.default_rng(0)
    samples = toy_samples(rng)
    a = OutputStats("logits", eps=0.0, device="cpu").fit(samples)
    x = samples[2].astype(np.float64)
    assert np.allclose(a.inv[2].numpy(), np.linalg.inv(np.cov(x.T, bias=True)), rtol=1e-3, atol=1e-4)


def test_aggregate():
    comp = np.array([[0, 0, 1], [1, 1, 1]])
    md = np.array([[1.0, 3.0, 0.0], [10.0, 20.0, 30.0]])
    assert np.allclose(aggregate(md, comp, 2, "mean"), [2.0, 15.0])
    assert np.allclose(aggregate(md, comp, 2, "p90"), [1.0, 20.0])   # ближайший ранг снизу
    assert np.allclose(aggregate(md, comp, 2, "frac", np.full(comp.shape, 15.0)), [0.0, 0.5])


def test_embedding_mahalanobis_matches_formula():
    import pandas as pd
    from core.data import Units
    from routing.methods.density import MahalanobisEmbedding

    rng = np.random.default_rng(0)
    n, dim = 40 * NC, 6
    cls = np.repeat(np.arange(NC), 40)
    emb = (rng.normal(size=(n, dim)) + cls[:, None]).astype(np.float32)
    train = Units("toy", "p0", pd.DataFrame({"class_id": cls, "y_fail": np.zeros(n, int)}), emb)
    test = Units("toy", "p1", pd.DataFrame({"class_id": [0, 5]}),
                 rng.normal(size=(2, dim)).astype(np.float32))
    for cov in ("shared", "class"):
        s = MahalanobisEmbedding(cov=cov, eps=1e-3).fit(train).score(test)
        x = emb.astype(np.float64)
        mu = np.stack([x[cls == c].mean(0) for c in range(NC)])
        for i, c in enumerate([0, 5]):
            d0 = x - mu[cls] if cov == "shared" else x[cls == c] - mu[c]
            S = d0.T @ d0 / len(d0) + 1e-3 * np.eye(dim)
            d = test.emb[i] - mu[c]
            assert np.isclose(s[i], np.sqrt(d @ np.linalg.inv(S) @ d), rtol=1e-6)


def test_gda_matches_scipy():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from features.ddu import GDA

    rng = np.random.default_rng(0)
    C, d = 3, 4
    xs = [rng.normal(size=(200, d)) * (c + 1) + c for c in range(C)]
    n = torch.tensor([len(x) for x in xs], dtype=torch.float64)
    s = torch.tensor(np.stack([x.sum(0) for x in xs]))
    ss = torch.tensor(np.stack([x.T @ x for x in xs]))
    prior = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    gda = GDA(n, s, ss, prior, eps=1e-3)
    z = rng.normal(size=(5, d))
    pred = np.array([0, 1, 2, 0, 1])
    logp, md_pred, md_min = gda.scores(torch.tensor(z, dtype=torch.float32), torch.tensor(pred))
    comps, mds = [], []
    for c, x in enumerate(xs):
        cov = np.cov(x.T, bias=True) + 1e-3 * np.eye(d)
        comps.append(multivariate_normal(x.mean(0), cov).logpdf(z) + np.log(prior[c].item()))
        dz = z - x.mean(0)
        mds.append(np.sqrt(np.einsum("ni,ij,nj->n", dz, np.linalg.inv(cov), dz)))
    mds = np.stack(mds, 1)
    assert np.allclose(logp.numpy(), logsumexp(np.stack(comps), axis=0), rtol=1e-4, atol=1e-3)
    assert np.allclose(md_pred.numpy(), mds[np.arange(5), pred], rtol=1e-4)
    assert np.allclose(md_min.numpy(), mds.min(1), rtol=1e-4)


def test_merge_samples_caps_per_class():
    rng = np.random.default_rng(0)
    per = [{0: np.zeros((30, NC), np.float16)}, {0: np.zeros((30, NC), np.float16)}]
    assert len(merge_samples(per, rng, per_class=40)[0]) == 40
