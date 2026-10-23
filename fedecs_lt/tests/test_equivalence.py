"""The new helpers reproduce experiments/core.py exactly: partitions, networks, statistics and formulas.
Run: python -m pytest fedecs_lt/tests -q   (CPU, a few seconds; needs no dataset download)."""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))

import core  # noqa: E402  (the old pipeline, reference only)
import data  # noqa: E402
import fedecs  # noqa: E402
import models  # noqa: E402

core.DEV = "cpu"


def fake_labels(C, n_per, seed=0):
    rng = np.random.RandomState(seed)
    y = np.repeat(np.arange(C), n_per)
    return torch.from_numpy(rng.permutation(y)).long()


@pytest.mark.parametrize("C,K,IF,alpha,seed", [(10, 10, 0.01, 0.5, 3), (10, 10, 0.01, 0.5, 4), (10, 50, 0.01, 0.5, 5),
                                               (10, 100, 0.01, 0.5, 3), (10, 10, 1.0, None, 0), (10, 10, 0.1, 0.1, 7),
                                               (100, 10, 0.01, 0.5, 3), (10, 20, 0.02, 1.0, 9)])
def test_partition_identical(C, K, IF, alpha, seed):
    y = fake_labels(C, 6000 if C == 10 else 500)
    core.NUM_CLASSES = C
    frac = 0.1 if C == 10 else 0.22
    a = core.make_partition(y, K, frac, IF, alpha, seed)
    b = data.make_partition(y, C, K, frac, IF, alpha, seed)
    assert all(torch.equal(p, q) for p, q in zip(a["parts"], b["parts"]))
    assert torch.equal(a["heldout"], b["heldout"])
    assert np.array_equal(a["counts"], b["counts"])
    core.NUM_CLASSES = 10


def test_assign_arch_and_active():
    for K in (10, 20, 50, 100):
        assert core.assign_arch(K) == data.assign_arch(K)
    ctx = type("C", (), {})()
    ctx.K, ctx.pfrac, ctx.seed = 100, 0.1, 4
    for r in range(5):
        for st in ("i", "ii", "iii"):
            assert list(core.Ctx.active(ctx, r, st)) == data.active_clients(100, 0.1, 4, st, r)


def test_networks_identical():
    for in_ch, var, bn in ((1, "mnist", False), (3, "cifar10", True)):
        torch.manual_seed(0)
        a = core.PaperFE(in_ch, var)
        torch.manual_seed(0)
        b = models.PaperFE(in_ch, var)
        x = torch.randn(4, in_ch, 32, 32)
        assert torch.equal(a(x), b(x))
        for g, w in enumerate(core.HEADER_WIDTHS):
            torch.manual_seed(g)
            ha = core.PaperHeader(w, a.out_ch, bn, C=10)
            torch.manual_seed(g)
            hb = models.PaperHeader(w, b.out_ch, bn, C=10)
            f = a(x)
            assert torch.equal(ha(f), hb(f))
    torch.manual_seed(1)
    ga = core.PaperCVAE(3, 16, C=10)
    torch.manual_seed(1)
    gb = models.PaperCVAE(3, 16, C=10)
    assert models.n_params(ga) == models.n_params(gb) == 5546659
    x, y = torch.rand(8, 3, 16, 16), torch.arange(8) % 10
    torch.manual_seed(2)
    la = ga.loss(x, y)
    torch.manual_seed(2)
    assert torch.equal(la, gb.loss(x, y))
    proto = torch.rand(10, 3, 16, 16)
    torch.manual_seed(3)
    pa = core.ProtoCVAE(3, 16, proto)
    torch.manual_seed(3)
    pb = models.ProtoCVAE(3, 16, proto)
    torch.manual_seed(4)
    sa = pa.sample(y)
    torch.manual_seed(4)
    assert torch.allclose(sa, pb.sample(y))


def test_statistics_gate_mc_readout():
    torch.manual_seed(0)
    C, K = 10, 4
    feats = [torch.rand(50 + 10 * k, 3, 16, 16) for k in range(K)]
    ys = [torch.randint(0, C, (len(f),)) for f in feats]
    core.NUM_CLASSES = C
    n0, S10, S20 = core.anchor_stats(feats, ys)
    parts = [fedecs.class_stats(f, y, C) for f, y in zip(feats, ys)]
    n, S1, S2 = (sum(p[i] for p in parts) for i in range(3))
    nn_, mu, V = fedecs.moments_from_sums(n, S1, S2)
    assert torch.allclose(nn_.float(), n0) and torch.allclose(S1.float(), S10, atol=1e-4)
    assert torch.allclose(S2.float(), S20, rtol=1e-5)
    # gate F: same formula as core.run_one
    mu_ = S10 / n0[:, None]
    Vw = (S20 / n0 - mu_.pow(2).sum(1)).clamp(min=1e-12)
    F_core = float((mu_ - mu_.mean(0)).pow(2).sum(1).mean() / Vw.mean())
    assert abs(fedecs.separation_F(n, S1, S2) - F_core) < 1e-5 * max(1, abs(F_core))
    # CCWA weights
    cnt = np.array([[0, 3, 100, 5000]])
    spec = dict(dp_eps=None, weight="en", beta=0.999)
    assert np.allclose(core.class_weights(cnt, spec, None), fedecs.ccwa_weights(cnt, 0.999))
    # CCWA rows = core.Averager with class weights
    rows = [torch.randn(C, 7) for _ in range(K)]
    W = [np.where(np.arange(C) < 5 + k, 1.0 + k, 0.0) for k in range(K)]
    prev = {"r": torch.randn(C, 7)}
    avg = core.Averager({"r": 0}, prev=prev)
    for r_, w_ in zip(rows, W):
        avg.add({"r": r_}, w_)
    rs = sum(torch.tensor(w_, dtype=torch.float32)[:, None] * r_ for r_, w_ in zip(rows, W))
    ws = sum(torch.tensor(w_, dtype=torch.float32) for w_ in W)
    assert torch.allclose(avg.result()["r"], fedecs.ccwa_rows(rs, ws, prev["r"]))
    # exact readout from sums = core.lda_readout
    head = core.PaperHeader(core.HEADER_WIDTHS[1], 3, False, C=C)
    Wc, bc = core.lda_readout(head, feats, ys, list(range(K)))
    head.eval()
    with torch.no_grad():
        sums = [fedecs.embed_sums(head.embed(f), y, C) for f, y in zip(feats, ys)]
    Wn, bn = fedecs.lda_from_sums(*(sum(s[i] for s in sums) for i in range(3)))
    assert torch.allclose(Wc, Wn, rtol=1e-4, atol=1e-4) and torch.allclose(bc, bn, rtol=1e-4, atol=1e-3)
    # moment correction: same map as core.MomentCalibratedGen given the same generator samples
    G = core.PaperCVAE(3, 16, C=C)
    torch.manual_seed(5)
    mc_core = core.MomentCalibratedGen(G, feats, ys)
    mc_new = fedecs.MomentCalibratedGen(G, mu.float(), V.float(), seed=5)
    assert torch.allclose(mc_core.m, mc_new.m, atol=1e-5) and torch.allclose(mc_core.scale, mc_new.scale, atol=1e-4)
    # logit adjustment prior and FedLC / FedRS adjustments
    ck = np.array([0, 4, 9, 300, 0, 1, 2, 3, 4, 5])
    assert torch.allclose(core.local_log_prior(ck), fedecs.local_log_prior(ck, "cpu"))
    assert torch.allclose(core.client_loss_adj(dict(loss="lc", tau=1.0), ck), fedecs.client_loss_adj("lc", ck, 1.0, "cpu"))
    # CSL pool layout
    sizes = [700, 64, 30, 1200]
    P, off, need = fedecs.csl_pool_size(sizes, 10, 64, 4, 4)
    assert P == max(need) and off == [0, 0, 0, 0]
    P, off, need = fedecs.csl_pool_size(sizes, 10, 64, 4, 2)
    assert P == sum(need) and off == [0, need[0], need[0] + need[1], need[0] + need[1] + need[2]]


def test_gaussian_generator_from_sums():
    torch.manual_seed(1)
    C = 10
    feats = [torch.rand(80, 3, 4, 4) for _ in range(3)]
    ys = [torch.randint(0, C, (80,)) for _ in range(3)]
    core.NUM_CLASSES = C
    G, _ = core.stage_ii_gauss(type("c", (), dict(seed=0))(), dict(dp_eps=None), feats, ys)
    n = sum(torch.bincount(y, minlength=C).double() for y in ys)
    S = sum(torch.zeros(C, 48, dtype=torch.float64).index_add_(0, y, f.flatten(1).double()) for f, y in zip(feats, ys))
    M = sum(f.flatten(1).double().T @ f.flatten(1).double() for f in feats)
    mu, L = fedecs.gaussian_from_sums(n, S, M, (3, 4, 4))
    assert torch.allclose(G.mu, mu, atol=1e-5) and torch.allclose(G.chol @ G.chol.T, L @ L.T, atol=1e-4)
