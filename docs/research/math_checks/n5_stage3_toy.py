"""N5: Stage III in miniature (Sec. 2c, 4).  K = 10 clients with the actual MNIST seed-0 composition, one head per
client (one client per architecture), frozen 'features' = C = 10 Gaussian classes in d = 30 dims.
Per round: synthetic phase (T_s epochs of n_k samples, uniform labels, plain CE or CSL soft labels), then the real
phase (T_r = 5 epochs, LA with the client's smoothed prior), SGD lr 0.1, B = 64, 50 rounds. Head = MLP 30-32-16-C.
Generators (all class-conditional samplers of the feature space):
  faithful   : x ~ N(M_y, I)                                    (signal fraction s = 1)
  s=0.5      : w.p. 0.5 from class y, else from the class-mixture marginal (label noise)
  agnostic   : x ~ marginal, label y uniform and independent of x (s = 0; GeFL-F on CIFAR-100: fidelity ~ 1/C)
  agnostic+CSL: as agnostic, target 0.5 onehot(y) + 0.5 * mean softmax of all 10 heads (consensus)
  real-only  : T_s = 0
Metrics: heads as trained and ER (class-balanced LDA on each body's embeddings of ALL clients' real data),
balanced test accuracy averaged over the 10 heads; and the retention of the synthetic phase's change of the
absent-class last-layer rows through the real phase (with and without LA).
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
d = 30
sep = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8


def make(seed):
    g = np.random.default_rng(seed)
    M = g.normal(0, sep, (C, d))
    Xk, Yk = [], []
    for k in range(K):
        Y = np.concatenate([np.full(cnt[k, c], c) for c in range(C)])
        X = M[Y] + g.normal(size=(len(Y), d))
        Xk.append(torch.tensor(X, dtype=torch.float32)); Yk.append(torch.tensor(Y))
    Yte = np.repeat(np.arange(C), 400)
    Xte = M[Yte] + g.normal(size=(len(Yte), d))
    return torch.tensor(M, dtype=torch.float32), Xk, Yk, torch.tensor(Xte, dtype=torch.float32), torch.tensor(Yte)


def head():
    return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, C))


def body(h, x):
    return h[:4](x)


def sample(M, y, mode, gen):
    if mode == "faithful":
        return M[y] + torch.randn(len(y), d, generator=gen)
    if mode == "agnostic" or mode == "agnostic_csl":
        yy = torch.randint(0, C, (len(y),), generator=gen)  # marginal over classes (uniform mixture)
        return M[yy] + torch.randn(len(y), d, generator=gen)
    s = float(mode.split("=")[1])
    keep = torch.rand(len(y), generator=gen) < s
    yy = torch.where(keep, y, torch.randint(0, C, (len(y),), generator=gen))
    return M[yy] + torch.randn(len(y), d, generator=gen)


def bal_acc(logits, y):
    pred = logits.argmax(1)
    return float(np.mean([(pred[y == c] == c).float().mean().item() for c in range(C)]))


def er(h, Xk, Yk, Xte, Yte, gamma=0.1):
    with torch.no_grad():
        E = torch.cat([body(h, x) for x in Xk]).double(); Y = torch.cat(Yk)
        q = E.shape[1]
        n = torch.bincount(Y, minlength=C).double()
        mu = torch.zeros(C, q, dtype=torch.float64).index_add_(0, Y, E) / n[:, None]
        Sw = (E.T @ E - (mu.T * n) @ mu) / (n.sum() - C)
        Ss = (1 - gamma) * Sw + (gamma * torch.trace(Sw) / q + 1e-6) * torch.eye(q, dtype=torch.float64)
        W = mu @ torch.linalg.pinv(Ss); b = -0.5 * (W * mu).sum(1)
        return bal_acc(body(h, Xte).double() @ W.T + b, Yte)


def run(mode, Ts, seed, la=True, rounds=50, track=False):
    M, Xk, Yk, Xte, Yte = make(seed)
    torch.manual_seed(seed + 1)
    heads = [head() for _ in range(K)]
    gen = torch.Generator().manual_seed(seed + 2)
    ret_abs, ret_held = [], []
    for r in range(rounds):
        pool = None
        if Ts > 0 and mode == "agnostic_csl":
            with torch.no_grad():
                P = Ts * max(len(y) for y in Yk)
                py = torch.randint(0, C, (P,), generator=gen)
                px = sample(M, py, mode, gen)
                cons = torch.stack([h(px).softmax(1) for h in heads]).mean(0)
                pool = (px, 0.5 * F.one_hot(py, C).float() + 0.5 * cons)
        for k in range(K):
            h = heads[k]; X, Y = Xk[k], Yk[k]; n = len(Y)
            prior = torch.log(torch.tensor((cnt[k] + 1.0) / (cnt[k].sum() + C), dtype=torch.float32))
            absent = torch.tensor(cnt[k] == 0)
            w0 = h[4].weight.detach().clone()
            if Ts > 0:
                opt = torch.optim.SGD(h.parameters(), lr=0.1)
                ns = Ts * (n // 64) * 64 if n >= 64 else Ts * n
                if pool is not None:
                    xs, ts = pool[0][:ns], pool[1][:ns]
                else:
                    ys = torch.randint(0, C, (ns,), generator=gen); xs = sample(M, ys, mode, gen); ts = F.one_hot(ys, C).float()
                for s in range(0, ns, 64):
                    loss = -(ts[s:s + 64] * h(xs[s:s + 64]).log_softmax(1)).sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            w1 = h[4].weight.detach().clone()
            opt = torch.optim.SGD(h.parameters(), lr=0.1)
            for _ in range(5):
                perm = torch.randperm(n)
                for s in range(0, n, 64):
                    b = perm[s:s + 64]
                    if b.numel() < 2:
                        continue
                    z = h(X[b])
                    loss = F.cross_entropy(z + prior if la else z, Y[b])
                    opt.zero_grad(); loss.backward(); opt.step()
            w2 = h[4].weight.detach()
            if track and Ts > 0 and r >= rounds // 2:
                dsyn = (w1 - w0)
                # retention: projection of the post-real change onto the synthetic change, per row group
                for grp, store in ((absent, ret_abs), (~absent, ret_held)):
                    if grp.any():
                        a_ = dsyn[grp].flatten(); b_ = (w2 - w0)[grp].flatten()
                        store.append(float((a_ @ b_) / (a_ @ a_ + 1e-12)))
    with torch.no_grad():
        heads_acc = float(np.mean([bal_acc(h(Xte), Yte) for h in heads]))
    er_acc = float(np.mean([er(h, Xk, Yk, Xte, Yte) for h in heads]))
    out = dict(mode=mode, Ts=Ts, la=la, seed=seed, heads=round(100 * heads_acc, 2), er=round(100 * er_acc, 2))
    if track and ret_abs:
        out.update(retention_absent=round(float(np.median(ret_abs)), 3), retention_held=round(float(np.median(ret_held)), 3))
    return out


if __name__ == "__main__":
    rows = []
    for seed in range(3):
        for mode, Ts in (("real", 0), ("faithful", 1), ("faithful", 10), ("s=0.5", 1), ("s=0.5", 10),
                         ("agnostic", 1), ("agnostic", 10), ("agnostic_csl", 1), ("agnostic_csl", 10)):
            o = run(mode if mode != "real" else "faithful", Ts, seed, track=(mode == "faithful" and Ts == 10))
            o["mode"] = mode
            rows.append(o); print(json.dumps(o), flush=True)
        o = run("faithful", 10, seed, la=False, track=True); o["mode"] = "faithful_noLA"; rows.append(o); print(json.dumps(o), flush=True)
    json.dump(rows, open(f"{OUT}/n5_out_sep{sep}.json", "w"))
    import collections
    agg = collections.defaultdict(list)
    for o in rows:
        agg[(o["mode"], o["Ts"])].append((o["heads"], o["er"]))
    for k, v in agg.items():
        v = np.array(v)
        print(k, "heads %.2f  ER %.2f" % tuple(v.mean(0)))
