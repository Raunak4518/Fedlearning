"""FedECS-LT components: everything is computed from exact per-class sums over all clients (secure aggregation)
instead of from averaged models. Formulas are those of experiments/core.py (tests/test_equivalence.py checks them):

  statistics release   n_c, S1_c = sum h, S2_c = sum ||h||^2 per class           (anchor_stats)
  separation gate      F = mean_c ||mu_c - mean mu||^2 / mean_c V_c               (run_one, 'sep')
  mean anchor          generator conditioned on mu_c = S1_c / n_c                  (ProtoCVAE)
  CCWA                 class rows averaged with weights E(n_kc) over holders        (class_weights, Averager)
  moment correction    x' = ReLU(mu_c + s_c (x - m_c)), s_c = sqrt(V_c / V~_c)     (MomentCalibratedGen)
  logit adjustment     logits + tau log p_k (add-one smoothed client prior)         (local_log_prior)
  consensus labels     (1 - beta) onehot(y) + beta mean_g softmax(h_g(x))          (stage_iii, CSL)
  exact readout        class-balanced LDA from per-class sums of embeddings         (lda_readout)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- statistics release (client side)
@torch.no_grad()
def class_stats(X, Y, C):
    """One client's contribution: count, sum of h and sum of ||h||^2 per class (float64)."""
    Xf = X.flatten(1).double()
    n = torch.bincount(Y, minlength=C).double()
    S1 = torch.zeros(C, Xf.shape[1], dtype=torch.float64, device=X.device).index_add_(0, Y, Xf)
    S2 = torch.zeros(C, dtype=torch.float64, device=X.device).index_add_(0, Y, Xf.pow(2).sum(1))
    return n, S1, S2


def moments_from_sums(n, S1, S2):
    """Exact class means and spreads V_c = E||h - mu_c||^2 from the summed statistics (counts floored at 1, as
    core.anchor_stats)."""
    n = n.clamp(min=1.0)
    mu = S1 / n[:, None]
    V = S2 / n - mu.pow(2).sum(1)
    return n, mu, V


def separation_F(n, S1, S2):
    """Class separation: between-class variance of the exact means over the mean within-class spread."""
    _, mu, V = moments_from_sums(n, S1, S2)
    Vw = V.clamp(min=1e-12)
    return float((mu - mu.mean(0)).pow(2).sum(1).mean() / Vw.mean())


# ---------------------------------------------------------------- class-count weighted averaging (CCWA)
def ccwa_weights(counts_k, beta):
    """E(n) = (1 - beta^n) / (1 - beta): zero for a client without the class, about n for few samples, saturating
    at 1 / (1 - beta) so that one large client cannot dominate a class."""
    c = np.asarray(counts_k, dtype=float)
    return (1 - beta ** c) / (1 - beta)


def ccwa_rows(row_sum, w_sum, prev_rows):
    """Average of class rows from secure sums: sum_k E(n_kc) row_kc / sum_k E(n_kc); a class no participating client
    holds keeps its previous rows."""
    shape = (-1,) + (1,) * (row_sum.dim() - 1)
    rows = row_sum / w_sum.clamp(min=1e-12).view(shape)
    return torch.where((w_sum > 0).view(shape), rows, prev_rows)


# ---------------------------------------------------------------- moment correction (server side)
class MomentCalibratedGen(nn.Module):
    """Sampling-time correction of each class's generator distribution to the exact class mean and spread.
    m_c and V~_c are measured on n samples per class drawn with a fixed generator seed (reproducible)."""

    def __init__(self, G, mu, V, n=500, clip=(0.1, 10.0), seed=0):
        super().__init__()
        self.G, self.kind = G, G.kind
        C = mu.shape[0]
        dev = mu.device
        G.eval()
        gen_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        m, Vg = [], []
        with torch.no_grad():
            for c in range(C):
                g = G.sample(torch.full((n,), c, device=dev, dtype=torch.long)).flatten(1)
                m.append(g.mean(0))
                Vg.append((g - g.mean(0)).pow(2).sum(1).mean())
        torch.random.set_rng_state(gen_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        m, Vg = torch.stack(m).float(), torch.stack(Vg).float()
        scale = (V.float().clamp(min=0) / Vg.clamp(min=1e-8)).sqrt().clamp(*clip)
        self.register_buffer("mu", mu.float())
        self.register_buffer("m", m)
        self.register_buffer("scale", scale)

    @torch.no_grad()
    def sample(self, y, z=None):
        x = self.G.sample(y, z)
        shp = x.shape[1:]
        out = self.mu[y] + self.scale[y, None] * (x.flatten(1) - self.m[y])
        return F.relu(out).view(-1, *shp)


class MCParams:
    """The moment-correction parameters sent to clients (they hold the generator) so that they can draw the same
    corrected samples as the server: per-class target mean, generator mean and scale."""

    def __init__(self, mu, m, scale):
        self.mu, self.m, self.scale = mu, m, scale

    @torch.no_grad()
    def apply(self, x, y):
        shp = x.shape[1:]
        return F.relu(self.mu[y] + self.scale[y, None] * (x.flatten(1) - self.m[y])).view(-1, *shp)


# ---------------------------------------------------------------- head losses
def local_log_prior(counts_k, device):
    p = np.asarray(counts_k, dtype=float) + 1.0
    p = p / p.sum()
    return torch.tensor(np.log(p), dtype=torch.float32, device=device)


def client_loss_adj(loss, counts_k, tau, device):
    """FedLC: z_c - tau (n_kc + 1)^(-1/4); FedRS: mask of classes the client lacks."""
    if loss == "lc":
        return torch.tensor(tau * (np.asarray(counts_k, dtype=float) + 1.0) ** -0.25, dtype=torch.float32, device=device)
    if loss == "rs":
        return torch.tensor(np.asarray(counts_k) == 0, device=device)
    return None


def head_loss(logits, y, loss, log_prior=None, tau=1.0, adj=None, rs_alpha=0.5):
    if loss == "lc":
        return F.cross_entropy(logits - adj, y)
    if loss == "rs":
        return F.cross_entropy(torch.where(adj[None, :], logits * rs_alpha, logits), y)
    if loss == "la" and log_prior is not None and tau:
        logits = logits + tau * log_prior
    return F.cross_entropy(logits, y)


# ---------------------------------------------------------------- consensus soft labels
def csl_pool_size(sizes, ts, B, K, n_groups):
    """Synthetic samples each client trains on per round, and the pool layout (as core.stage_iii): one shared pool
    when each architecture has one client, otherwise disjoint slices."""
    need = [ts * (sz // B if sz >= B else 1) * (B if sz >= B else sz) for sz in sizes]
    if K == n_groups:
        return max(need), [0] * K, need
    off = list(np.cumsum([0] + need[:-1]))
    return int(sum(need)), [int(o) for o in off], need


def csl_targets(py, cons, beta, C):
    return (1 - beta) * F.one_hot(py, C).float() + beta * cons


# ---------------------------------------------------------------- exact readout
@torch.no_grad()
def embed_sums(E, Y, C):
    """One client's contribution to the readout of one architecture: counts, sums of e and sum of e e^T."""
    Ed = E.double()
    n = torch.bincount(Y, minlength=C).double()
    S = torch.zeros(C, Ed.shape[1], dtype=torch.float64, device=E.device).index_add_(0, Y, Ed)
    return n, S, Ed.T @ Ed


def lda_from_sums(n, S, M, gamma=0.1):
    """Class-balanced LDA last layer (W, b) from summed statistics, as core.lda_readout; absent classes b = -1e4."""
    C, d = S.shape
    mu = S / n[:, None].clamp(min=1)
    absent = n == 0
    Sw = (M - (mu.T * n) @ mu) / max(float(n.sum()) - int((~absent).sum()), 1.0)
    Sw = (Sw + Sw.T) / 2
    Ss = (1 - gamma) * Sw + (gamma * torch.trace(Sw) / d + 1e-6) * torch.eye(d, dtype=torch.float64, device=S.device)
    # the pseudo-inverse on the CPU (milliseconds for d <= a few hundred): on a nearly full GPU cuSOLVER's syevd failed
    # (CUSOLVER_STATUS_EXECUTION_FAILED, one shared model, Kaggle, 27 Oct 2026); only rounding can differ
    W = mu @ torch.linalg.pinv(Ss.cpu(), hermitian=True).to(Ss.device)
    b = -0.5 * (W * mu).sum(1)
    b[absent] = -1e4
    return W.float(), b.float()


# ---------------------------------------------------------------- Gaussian feature generator (control)
def gaussian_from_sums(n, S, M, shape, shrink=0.1):
    """Shared-covariance class Gaussian from exact sums (as core.stage_ii_gauss)."""
    C, D = S.shape
    mu = S / n.clamp(min=1.0)[:, None]
    W = M - (mu.T * n) @ mu
    Sig = W / max(float(n.sum()) - C, 1.0)
    Sig = (Sig + Sig.T) / 2
    Sig = (1 - shrink) * Sig + shrink * (torch.trace(Sig) / D) * torch.eye(D, device=S.device, dtype=torch.float64)
    evals, evecs = torch.linalg.eigh(Sig)
    evals = evals.clamp(min=1e-6 * float(torch.trace(Sig)) / D)
    return mu.float(), (evecs * evals.sqrt()).float()
