"""N3: class-conditioning rows under FedAvg + Adam (Sec. 2b), on the ACTUAL MNIST seed-0 K=10 counts.

Toy generator: rows only, x | y=c ~ N(m_c, sigma^2 I_d), loss per batch = sum_i ||x_i - r_{y_i}||^2 (the CVAE's
summed-MSE form), so the centralised MLE of r_c is the pooled class mean. Local training exactly as
core.gen_local_update: batches of 64 (drop size<2), 5 epochs per round, Adam lr 1e-3, betas (0.9, 0.999), eps 1e-8,
per-client optimiser state kept across rounds while the parameters are reset to the global ones.
Arms: coupled decay 1e-3 + flat FedAvg (GeFL-F) | coupled + HWA (E(n), beta .999) | AdamW 1e-3 + flat | no decay + flat
Prediction (sign-Adam model): under coupled decay + flat averaging, a row collapses iff
  kappa_c = sum_{k in H_c} tau_k f_kc / sum_{k not in H_c} tau_k  (< 1 : collapse)   [f_kc = P(class c in a batch)]
and otherwise shrinks to the equilibrium |m_j - r_j| ~ sigma_j / sqrt(b (kappa^2 - 1)) (b = class-c samples per batch).
Also reports, for no decay / HWA, the distance of the final row to the pooled class mean and to the unweighted
mean of holders' local means (the saturated flat-FedAvg fixed point).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import json, math, sys
import numpy as np
import torch

OUT = _HERE
torch.set_num_threads(1)
cnt = np.load(f"{OUT}/mnist_s0_K10_counts.npy").astype(int)  # K x C (reconstructed, validated)
K, C = cnt.shape
d = int(sys.argv[1]) if len(sys.argv) > 1 else 32
sigma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(1)
m = rng.normal(0, 0.15, size=(C, d))
data = []
for k in range(K):
    ys = np.concatenate([np.full(cnt[k, c], c) for c in range(C)])
    xs = m[ys] + sigma * rng.normal(size=(len(ys), d))
    data.append((torch.tensor(xs, dtype=torch.float32), torch.tensor(ys)))
pooled = np.stack([np.concatenate([data[k][0].numpy()[data[k][1].numpy() == c] for k in range(K)]).mean(0) for c in range(C)])
H = cnt > 0
loc_means = {(k, c): data[k][0].numpy()[data[k][1].numpy() == c].mean(0) for k in range(K) for c in range(C) if H[k, c]}
unif_holders = np.stack([np.mean([loc_means[(k, c)] for k in range(K) if H[k, c]], 0) for c in range(C)])
b0 = 1.0 / math.sqrt(16 + C)
init = torch.empty(C, d).uniform_(-b0, b0, generator=torch.Generator().manual_seed(0))


def run(arm):
    glob = init.clone()
    params = [torch.nn.Parameter(init.clone()) for _ in range(K)]
    wd = 0.0 if arm == "nodecay" else 1e-3
    opts = [(torch.optim.AdamW if arm == "adamw" else torch.optim.Adam)([p], lr=1e-3, weight_decay=wd) for p in params]
    E = (1 - 0.999 ** cnt) / (1 - 0.999)
    g = torch.Generator().manual_seed(2)
    hist = []
    for r in range(rounds):
        outs = []
        for k in range(K):
            p, opt = params[k], opts[k]
            with torch.no_grad():
                p.copy_(glob)
            X, Y = data[k]
            n = len(Y)
            for _ in range(5):
                perm = torch.randperm(n, generator=g)
                for s in range(0, n, 64):
                    bi = perm[s:s + 64]
                    if bi.numel() < 2:
                        continue
                    loss = (X[bi] - p[Y[bi]]).pow(2).sum()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            outs.append(p.detach().clone())
        S = torch.stack(outs)  # K x C x d
        if arm.endswith("hwa"):
            w = torch.tensor(E, dtype=torch.float32)  # K x C
            new = (w[:, :, None] * S).sum(0) / w.sum(0).clamp(min=1e-12)[:, None]
            glob = torch.where((w.sum(0) > 0)[:, None], new, glob)
        else:
            glob = S.mean(0)
        hist.append(glob.norm(dim=1).numpy().copy())
    return glob.numpy(), np.array(hist)


res = {}
for arm in ("coupled_flat", "coupled_hwa", "adamw", "nodecay"):
    G, hist = run(arm)
    res[arm] = dict(norm_ratio_to_target=(np.linalg.norm(G, axis=1) / np.linalg.norm(pooled, axis=1)).round(3).tolist(),
                    cos_to_target=[round(float(G[c] @ pooled[c] / (np.linalg.norm(G[c]) * np.linalg.norm(pooled[c]) + 1e-12)), 3) for c in range(C)],
                    dist_pooled=(np.linalg.norm(G - pooled, axis=1) / np.linalg.norm(pooled, axis=1)).round(3).tolist(),
                    dist_unif_holders=(np.linalg.norm(G - unif_holders, axis=1) / np.linalg.norm(pooled, axis=1)).round(3).tolist(),
                    tail_over_head=float(np.linalg.norm(G[7:], axis=1).mean() / np.linalg.norm(G[:3], axis=1).mean()))
    print(arm, json.dumps(res[arm]), flush=True)
print("pooled-mean vs unif-over-holders mean, relative distance per class:",
      (np.linalg.norm(pooled - unif_holders, axis=1) / np.linalg.norm(pooled, axis=1)).round(3).tolist())
json.dump(res, open(f"{OUT}/n3_out_d{d}_s{sigma}.json", "w"))
