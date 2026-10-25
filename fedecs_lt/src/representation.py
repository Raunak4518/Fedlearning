"""Representation, generator-quality and testing statistics computed from a run's saved outputs (test embeddings,
last layers, real and generated validation features); used by plots.py and report.py.

Neural collapse (Papyan, Han and Donoho, PNAS 2020): with class means m_c of the embeddings, their mean m_G and the
centred means M_c = m_c - m_G,
  NC2 equinorm      = std_c ||M_c|| / mean_c ||M_c||                         (0 at collapse)
  NC2 equiangularity = std over class pairs of cos(M_a, M_b)                 (0 at collapse)
  NC2 maximal angle  = mean over pairs of |cos(M_a, M_b) + 1 / (C - 1)|      (0 for a simplex ETF)
  NC3 self-duality   = || W / ||W||_F - M / ||M||_F ||_F^2                   (0 when the classifier aligns with M)
Minority collapse (Fang et al., PNAS 2021): the mean pairwise angle between the classifier vectors of the tail
classes (small = the rare classes' classifiers collapse onto each other).
Linear CKA (Kornblith et al., ICML 2019) between two architectures' embeddings of the same images.
MMD: unbiased squared maximum mean discrepancy with a Gaussian kernel, bandwidth = median pairwise distance.
Holm (1979) step-down adjustment of p-values.
"""
import numpy as np


def class_means(H, y, C):
    present = [c for c in range(C) if (y == c).any()]
    return np.stack([H[y == c].mean(0) for c in present]), present


def nc1(H, y, C, tol=1e-8):
    """NC1 = tr(S_W S_B^+) / C, with S_B^+ the pseudo-inverse over the between-class directions whose eigenvalue is
    above tol x the largest (S_B has rank at most C - 1; inverting its numerically zero eigenvalues, as a plain
    pinv of a covariance in many dimensions does, gives meaningless or even negative values). Always >= 0."""
    H = np.asarray(H, dtype=np.float64)
    mus, present = class_means(H, y, C)
    mg = H.mean(0)
    B = (mus - mg) / np.sqrt(len(mus))             # S_B = B^T B
    U, sv, _ = np.linalg.svd(B.T, full_matrices=False)
    lam = sv ** 2
    keep = lam > tol * lam.max() if len(lam) and lam.max() > 0 else np.zeros(len(lam), bool)
    if not keep.any():
        return None
    U, lam = U[:, keep], lam[keep]
    tot = 0.0
    for c in present:
        X = H[y == c] - mus[present.index(c)]
        if len(X) > 1:
            P = X @ U                                 # within-class scatter along the between-class directions
            tot += ((P ** 2).sum(0) / len(X) / lam).sum() * (len(X) / len(H))
    return float(tot / len(present))


def nc2(H, y, C):
    mus, _ = class_means(H.astype(np.float64), y, C)
    M = mus - mus.mean(0)
    norms = np.linalg.norm(M, axis=1)
    U = M / np.maximum(norms[:, None], 1e-12)
    cos = U @ U.T
    iu = np.triu_indices(len(M), 1)
    k = len(M)
    return dict(nc2_equinorm=float(norms.std() / max(norms.mean(), 1e-12)),
                nc2_equiangle=float(cos[iu].std()),
                nc2_maxangle=float(np.abs(cos[iu] + 1.0 / (k - 1)).mean()) if k > 1 else 0.0)


def nc3(H, y, C, W):
    mus, present = class_means(H.astype(np.float64), y, C)
    M = mus - mus.mean(0)
    Wp = np.asarray(W, dtype=np.float64)[present]
    if Wp.shape[1] != M.shape[1]:
        return None
    return float(np.linalg.norm(Wp / np.linalg.norm(Wp) - M / np.linalg.norm(M)) ** 2)


def minority_angle(W, classes):
    """Mean pairwise angle (degrees) between the classifier vectors of the given classes."""
    V = np.asarray(W, dtype=np.float64)[list(classes)]
    if len(V) < 2:
        return None
    U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    cos = np.clip(U @ U.T, -1.0, 1.0)
    iu = np.triu_indices(len(V), 1)
    return float(np.degrees(np.arccos(cos[iu])).mean())


def linear_cka(X, Y):
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    xy = np.linalg.norm(X.T @ Y) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    return float(xy / max(xx * yy, 1e-300))


def mmd_rbf(X, Y, bandwidth=None):
    """Unbiased MMD^2 between samples X and Y (rows), Gaussian kernel exp(-d^2 / (2 b^2))."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.concatenate([X, Y])
    sq = (Z * Z).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    if bandwidth is None:
        iu = np.triu_indices(len(Z), 1)
        bandwidth = np.sqrt(np.median(D2[iu])) if len(iu[0]) else 1.0
    K = np.exp(-D2 / (2 * max(bandwidth, 1e-12) ** 2))
    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        return None
    kxx = (K[:n, :n].sum() - np.trace(K[:n, :n])) / (n * (n - 1))
    kyy = (K[n:, n:].sum() - np.trace(K[n:, n:])) / (m * (m - 1))
    kxy = K[:n, n:].mean()
    return float(kxx + kyy - 2 * kxy)


def mean_mismatch(Xg, Xr):
    """Distance between the generated and the real class mean, in units of the real within-class spread
    (square root of the mean per-dimension variance)."""
    Xg = np.asarray(Xg, dtype=np.float64)
    Xr = np.asarray(Xr, dtype=np.float64)
    spread = np.sqrt(Xr.var(0).mean()) if len(Xr) > 1 else np.nan
    return float(np.linalg.norm(Xg.mean(0) - Xr.mean(0)) / np.sqrt(Xr.shape[1]) / max(spread, 1e-12))


def holm(p):
    """Holm-adjusted p-values (same order as p)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(1.0, run)
    return adj.tolist()
