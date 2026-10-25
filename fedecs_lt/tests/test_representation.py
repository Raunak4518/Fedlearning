"""Representation and testing statistics against cases with known answers."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import representation as rp  # noqa: E402


def simplex(C, d):
    """C class means forming a simplex equiangular tight frame in R^d (d >= C)."""
    Q, _ = np.linalg.qr(np.random.RandomState(0).randn(d, C))
    return (np.eye(C) - 1.0 / C) @ Q.T


def test_nc2_nc3_zero_at_collapse():
    C, d = 5, 8
    M = simplex(C, d)
    y = np.repeat(np.arange(C), 20)
    H = M[y] + 3.0                    # every embedding at its class mean (collapse), shifted by a global mean
    out = rp.nc2(H, y, C)
    assert out["nc2_equinorm"] < 1e-9 and out["nc2_equiangle"] < 1e-9 and out["nc2_maxangle"] < 1e-9
    assert rp.nc3(H, y, C, 2.5 * M) < 1e-9


def test_nc3_positive_when_misaligned():
    C, d = 4, 6
    rs = np.random.RandomState(1)
    y = np.repeat(np.arange(C), 30)
    H = rs.randn(C, d)[y] + 0.1 * rs.randn(len(y), d)
    assert rp.nc3(H, y, C, rs.randn(C, d)) > 0.1


def test_minority_angle():
    W = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    assert abs(rp.minority_angle(W, [0, 1]) - 90.0) < 1e-9
    assert abs(rp.minority_angle(W, [0, 2]) - 45.0) < 1e-9


def test_linear_cka_invariances():
    rs = np.random.RandomState(2)
    X = rs.randn(200, 10)
    Q, _ = np.linalg.qr(rs.randn(10, 10))
    assert abs(rp.linear_cka(X, X) - 1.0) < 1e-9
    assert abs(rp.linear_cka(X, 3.0 * X @ Q + 5.0) - 1.0) < 1e-9     # orthogonal map, scale, shift
    assert rp.linear_cka(X, rs.randn(200, 10)) < 0.2


def test_mmd_and_mismatch():
    rs = np.random.RandomState(3)
    A, B = rs.randn(300, 5), rs.randn(300, 5)
    assert abs(rp.mmd_rbf(A, B)) < 0.02
    assert rp.mmd_rbf(A, B + 2.0) > 0.2
    assert rp.mean_mismatch(A, A) < 1e-12
    assert abs(rp.mean_mismatch(B + 1.0, B) - 1.0) < 0.05             # one within-class SD in every dimension


def test_holm():
    # Holm (1979): sorted p (0.01, 0.02, 0.04) -> 3 x 0.01, max(0.03, 2 x 0.02), max(0.04, 0.04)
    assert np.allclose(rp.holm([0.04, 0.01, 0.02]), [0.04, 0.03, 0.04])
    assert rp.holm([0.6, 0.9]) == [1.0, 1.0]


def test_nc1_nonnegative_and_zero_at_collapse():
    C, d = 5, 64
    M = simplex(C, d)
    y = np.repeat(np.arange(C), 40)
    assert rp.nc1(M[y] + 1.0, y, C) < 1e-12
    rs = np.random.RandomState(4)
    H = M[y] + 0.3 * rs.randn(len(y), d)
    v = rp.nc1(H, y, C)
    assert v > 0
    # against the direct formula with an eigen-truncated pseudo-inverse
    mus = np.stack([H[y == c].mean(0) for c in range(C)])
    Sb = np.cov((mus - H.mean(0)).T, bias=True)
    Sw = sum(np.cov((H[y == c] - mus[c]).T, bias=True) * (y == c).mean() for c in range(C))
    w, V = np.linalg.eigh(Sb)
    k = w > 1e-8 * w.max()
    ref = np.trace(Sw @ (V[:, k] / w[k]) @ V[:, k].T) / C
    assert abs(v - ref) < 1e-8 * max(1.0, ref)
