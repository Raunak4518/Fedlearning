"""N2: Stage I in miniature. Shared bottleneck FE (Linear d->p, ReLU) + one personalised head per client (K = 10,
one client per architecture as in GeFL-F), trained by FedAvg exactly like core.stage_i (FE averaged uniformly over
clients, 5 local epochs, SGD lr 0.1, B = 64, heads never averaged). Client composition = the actual MNIST seed-0
K = 10 counts (reconstructed by n6). Data: C = 10 Gaussian classes in d = 30 dims.

Compared FEs (same init, same number of SGD steps on the FE):
  fed        : FedAvg + personalised heads (GeFL-F stage I)
  fed_shared : FedAvg, but each client also trains a SHARED linear classifier on the FE output, averaged over all
               clients (proposal P2: an auxiliary global softmax) -- heads still personalised
  fed_scaf   : fed with SCAFFOLD control variates on the FE parameters (proposal P3)
  cen_erm    : centralised, pooled instance sampling, one head per 'architecture' all trained on the pool
  cen_bal    : same with class-balanced sampling
Metrics on the frozen FE: balanced accuracy of the class-balanced LDA on the FE output (exact sums: what ER sees),
pairwise Fisher separation J_cc' of the class means in FE space, and its ratio fed/cen against the
count-only pair-pressure ratio rho_cc' = sum_k n_kc pi_kc' / (n_c pi_c').
Seeds: 5 data draws.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = _HERE
torch.set_num_threads(1)
cnt = np.load(f"{OUT}/mnist_s0_K10_counts.npy").astype(int)
K, C = cnt.shape
d, p = 30, int(sys.argv[1]) if len(sys.argv) > 1 else 4
nc = cnt.sum(0); N = nc.sum()
pik = cnt / cnt.sum(1, keepdims=True); pig = nc / N
rho = np.array([[(cnt[:, a] * pik[:, b]).sum() / (nc[a] * pig[b]) if a != b else np.nan for b in range(C)] for a in range(C)])


def make_data(seed):
    g = np.random.default_rng(seed)
    M = g.normal(0, 1.0, (C, d))
    def draw(c, n):
        return M[c] + g.normal(0, 1.0, (n, d))
    Xk, Yk = [], []
    for k in range(K):
        X = np.concatenate([draw(c, cnt[k, c]) for c in range(C)])
        Y = np.concatenate([np.full(cnt[k, c], c) for c in range(C)])
        perm = g.permutation(len(Y))
        Xk.append(torch.tensor(X[perm], dtype=torch.float32)); Yk.append(torch.tensor(Y[perm]))
    Xte = np.concatenate([draw(c, 500) for c in range(C)]); Yte = np.repeat(np.arange(C), 500)
    return Xk, Yk, torch.tensor(Xte, dtype=torch.float32), torch.tensor(Yte)


class FE(nn.Module):
    def __init__(self):
        super().__init__(); self.l = nn.Linear(d, p)
    def forward(self, x):
        return F.relu(self.l(x))


def head():
    return nn.Sequential(nn.Linear(p, 16), nn.ReLU(), nn.Linear(16, C))


def lda_eval(fe, Xk, Yk, Xte, Yte, gamma=0.1):
    with torch.no_grad():
        Z = torch.cat([fe(x) for x in Xk]).double(); Y = torch.cat(Yk)
        n = torch.bincount(Y, minlength=C).double()
        mu = torch.zeros(C, p, dtype=torch.float64).index_add_(0, Y, Z) / n[:, None]
        Sw = (Z.T @ Z - (mu.T * n) @ mu) / (n.sum() - C)
        Ss = (1 - gamma) * Sw + (gamma * torch.trace(Sw) / p + 1e-9) * torch.eye(p, dtype=torch.float64)
        Si = torch.linalg.inv(Ss)
        W = mu @ Si; b = -0.5 * (W * mu).sum(1)
        pred = (fe(Xte).double() @ W.T + b).argmax(1)
        bal = float(np.mean([(pred[Yte == c] == c).double().mean().item() for c in range(C)]))
        J = np.array([[float((mu[a] - mu[b2]) @ Si @ (mu[a] - mu[b2])) for b2 in range(C)] for a in range(C)])
    return bal, J


def train(mode, Xk, Yk, seed, rounds=20):
    torch.manual_seed(seed)
    fe = FE(); heads = [head() for _ in range(K)]
    shared = nn.Linear(p, C)
    fe_sd = {k: v.clone() for k, v in fe.state_dict().items()}
    sh_sd = {k: v.clone() for k, v in shared.state_dict().items()}
    if mode.startswith("cen"):
        X, Y = torch.cat(Xk), torch.cat(Yk)
        w = 1.0 / torch.bincount(Y, minlength=C).float()
        opt = torch.optim.SGD(list(fe.parameters()) + [q for h in heads for q in h.parameters()], lr=0.1)
        steps = sum(5 * int(np.ceil(len(y) / 64)) for y in Yk) * rounds // K  # = FedAvg's FE steps per client-avg
        for t in range(steps):
            b = torch.multinomial(w[Y], 64, replacement=True) if mode == "cen_bal" else torch.randint(0, len(Y), (64,))
            z = fe(X[b])
            loss = sum(F.cross_entropy(h(z), Y[b]) for h in heads) / K
            opt.zero_grad(); loss.backward(); opt.step()
        return fe
    c_glob = {k: torch.zeros_like(v) for k, v in fe_sd.items()}
    c_loc = [{k: torch.zeros_like(v) for k, v in fe_sd.items()} for _ in range(K)]
    for r in range(rounds):
        acc = {k: torch.zeros_like(v) for k, v in fe_sd.items()}
        acc_sh = {k: torch.zeros_like(v) for k, v in sh_sd.items()}
        new_c = []
        for k in range(K):
            fe.load_state_dict(fe_sd); shared.load_state_dict(sh_sd)
            params = list(fe.parameters()) + list(heads[k].parameters()) + (list(shared.parameters()) if mode == "fed_shared" else [])
            opt = torch.optim.SGD(params, lr=0.1)
            X, Y = Xk[k], Yk[k]; steps = 0
            for _ in range(5):
                perm = torch.randperm(len(Y))
                for s in range(0, len(Y), 64):
                    b = perm[s:s + 64]
                    if b.numel() < 2:
                        continue
                    z = fe(X[b])
                    loss = F.cross_entropy(heads[k](z), Y[b])
                    if mode == "fed_shared":
                        loss = loss + F.cross_entropy(shared(z), Y[b])
                    opt.zero_grad(); loss.backward()
                    if mode == "fed_scaf":
                        for (nm, q) in fe.named_parameters():
                            q.grad += c_glob[nm] - c_loc[k][nm]
                    opt.step(); steps += 1
            sd = fe.state_dict()
            for nm in acc:
                acc[nm] += sd[nm] / K
            if mode == "fed_shared":
                for nm, v in shared.state_dict().items():
                    acc_sh[nm] += v / K
            if mode == "fed_scaf":  # option II: c_k+ = c_k - c + (x - y_k) / (steps * lr)
                new_c.append({nm: c_loc[k][nm] - c_glob[nm] + (fe_sd[nm] - sd[nm]) / (steps * 0.1) for nm in fe_sd})
        if mode == "fed_scaf":
            c_loc = new_c
            c_glob = {nm: sum(c[nm] for c in c_loc) / K for nm in fe_sd}
        fe_sd = acc
        if mode == "fed_shared":
            sh_sd = acc_sh
    fe.load_state_dict(fe_sd)
    return fe


res = {m: [] for m in ("fed", "fed_shared", "fed_scaf", "cen_erm", "cen_bal")}
Js = {m: [] for m in res}
for seed in range(5):
    Xk, Yk, Xte, Yte = make_data(seed)
    for m in res:
        fe = train(m, Xk, Yk, seed)
        bal, J = lda_eval(fe, Xk, Yk, Xte, Yte)
        res[m].append(bal); Js[m].append(J)
    print(seed, {m: round(100 * res[m][-1], 1) for m in res}, flush=True)
print("p =", p, "balanced LDA accuracy on the frozen FE, mean over 5 seeds:",
      {m: round(100 * float(np.mean(v)), 2) for m, v in res.items()})
off = ~np.eye(C, dtype=bool)
ratio = np.log(np.mean(Js["fed"], 0)[off] + 1e-9) - np.log(np.mean(Js["cen_erm"], 0)[off] + 1e-9)
from scipy.stats import spearmanr
print("Spearman(log J_fed/J_cen, log rho) over ordered pairs:", spearmanr(ratio, np.log(rho[off])))
json.dump({m: [float(x) for x in v] for m, v in res.items()}, open(f"{OUT}/n2_out_p{p}.json", "w"))
