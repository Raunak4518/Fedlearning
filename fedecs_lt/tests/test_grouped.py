"""GeFL's grouped baselines (grouped_client.py / grouped_server.py): the FedETF head's classifier is a simplex ETF;
model_wise_FedAvg is the unweighted mean; CCVR's class statistics from per-class sums equal the covariance of the
pooled embeddings; AvgKD's targets are GeFL's softmax(one-hot + sum of the other models' softmax); FedProx's proximal
term is GeFL's (zero at the start of local training)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import ETFHead, GroupedHead  # noqa: E402
from settings import HEADER_WIDTHS  # noqa: E402


def test_etf_classifier_is_simplex():
    C = 10
    h = ETFHead(HEADER_WIDTHS[3], 3, True, C, seed=4)
    M = h.proto
    cols = M / M.norm(dim=0, keepdim=True)
    assert torch.allclose(M.norm(dim=0), torch.ones(C), atol=1e-5)
    G = cols.T @ cols
    off = G[~torch.eye(C, dtype=torch.bool)]
    assert torch.allclose(off, torch.full_like(off, -1 / (C - 1)), atol=1e-5)
    assert torch.equal(ETFHead(HEADER_WIDTHS[3], 3, True, C, seed=4).proto, M)   # the same for every client
    out = h(torch.randn(5, 3, 16, 16))
    assert out.shape == (5, C)
    assert "proto" not in dict(h.named_parameters())                            # fixed, not trained


def test_groupwise_average_is_unweighted():
    from grouped_server import average
    a, b = GroupedHead(HEADER_WIDTHS[0], 3, True, 10), GroupedHead(HEADER_WIDTHS[0], 3, True, 10)
    m = average([a.state_dict(), b.state_dict()])
    for n, v in m.items():
        if v.is_floating_point():
            assert torch.allclose(v, (a.state_dict()[n] + b.state_dict()[n]) / 2)


def test_ccvr_statistics_equal_pooled_covariance():
    torch.manual_seed(0)
    d = 6
    parts = [torch.randn(n, d, dtype=torch.float64) * 2 + 1 for n in (7, 12, 3)]
    n = sum(len(p) for p in parts)
    S1 = sum(p.sum(0) for p in parts)
    S2 = sum(p.T @ p for p in parts)
    mu = S1 / n
    cov = (S2 - n * torch.outer(mu, mu)) / (n - 1)
    pooled = torch.cat(parts)
    assert torch.allclose(mu, pooled.mean(0))
    assert torch.allclose(cov, torch.cov(pooled.T))


def test_avgkd_targets_as_gefl():
    torch.manual_seed(1)
    C, y = 5, torch.tensor([0, 3, 4])
    others = [torch.randn(3, C), torch.randn(3, C)]
    t = F.softmax(F.one_hot(y, C).float() + sum(F.softmax(o, 1) for o in others), 1)
    assert torch.allclose(t.sum(1), torch.ones(3))
    assert (t.argmax(1) == y).all()   # the true label's one-hot dominates two softmax terms here


def test_fedprox_term_zero_at_start():
    h = GroupedHead(HEADER_WIDTHS[1], 3, True, 10)
    w0 = [p.detach().clone() for p in h.parameters()]
    prox = sum((p - p0).norm(2) for p, p0 in zip(h.parameters(), w0))
    assert float(prox) == 0.0
