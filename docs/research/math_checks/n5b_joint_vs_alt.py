"""N5b: alternation (GeFL-F's synthetic-then-real phases) vs a joint objective (synthetic samples interleaved into
every real batch, CSLM-style), at the SAME synthetic budget, for generators of different fidelity.
Theory (Prop. 2.9 / Sec. 4.4): alternation keeps synthetic information only in directions the client's real data do
not constrain (synthetic weight <= rho_r); the joint objective also lets it reduce variance in real-supported
directions -> joint should win for faithful generators and lose for biased / label-independent ones.
Also the counterfactual retention of the absent-class last-layer rows: R = <w_after - w_cf, w_syn - w_before> /
||w_syn - w_before||^2, where w_cf is the same real phase run from w_before (no synthetic phase).
Same toy as N5 (actual MNIST seed-0 composition, d = 30, C = 10).
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import json, sys, copy
import numpy as np
import torch
import torch.nn.functional as F
sys.argv = sys.argv[:1] + ["0.8"]
import importlib.util
spec = importlib.util.spec_from_file_location("n5", "" + _HERE + "/n5_stage3_toy.py")
n5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(n5)
C, K, cnt, d = n5.C, n5.K, n5.cnt, n5.d
OUT = n5.OUT


def run(mode, scheme, seed, Ts=10, rounds=50, la=True, track=False):
    M, Xk, Yk, Xte, Yte = n5.make(seed)
    torch.manual_seed(seed + 1)
    heads = [n5.head() for _ in range(K)]
    gen = torch.Generator().manual_seed(seed + 2)
    rets = []
    for r in range(rounds):
        for k in range(K):
            h = heads[k]; X, Y = Xk[k], Yk[k]; n = len(Y)
            prior = torch.log(torch.tensor((cnt[k] + 1.0) / (cnt[k].sum() + C), dtype=torch.float32))
            absent = torch.tensor(cnt[k] == 0)
            ns = Ts * (n // 64) * 64 if n >= 64 else Ts * n
            ys = torch.randint(0, C, (ns,), generator=gen); xs = n5.sample(M, ys, mode, gen)
            perms = [torch.randperm(n) for _ in range(5)]

            def real_phase(hh, mix=False):
                opt = torch.optim.SGD(hh.parameters(), lr=0.1)
                nb = sum((n + 63) // 64 for _ in range(5)); j = 0
                per = max(1, ns // nb) if mix else 0
                for perm in perms:
                    for s in range(0, n, 64):
                        b = perm[s:s + 64]
                        if b.numel() < 2:
                            continue
                        z = hh(X[b])
                        loss = F.cross_entropy(z + prior if la else z, Y[b])
                        if mix:  # joint objective: synthetic share of the batch, plain CE, same total budget
                            sl = slice(j * per, (j + 1) * per); j += 1
                            if xs[sl].shape[0] > 0:
                                loss = (len(b) * loss + per * F.cross_entropy(hh(xs[sl]), ys[sl])) / (len(b) + per)
                        opt.zero_grad(); loss.backward(); opt.step()
            if scheme == "alt":
                w0 = h[4].weight.detach().clone()
                hc = copy.deepcopy(h) if track else None
                opt = torch.optim.SGD(h.parameters(), lr=0.1)
                for s in range(0, ns, 64):
                    loss = F.cross_entropy(h(xs[s:s + 64]), ys[s:s + 64])
                    opt.zero_grad(); loss.backward(); opt.step()
                w1 = h[4].weight.detach().clone()
                real_phase(h)
                if track and r >= rounds // 2 and absent.any():
                    real_phase(hc)
                    a_ = (w1 - w0)[absent].flatten(); b_ = (h[4].weight.detach() - hc[4].weight.detach())[absent].flatten()
                    hel = (w1 - w0)[~absent].flatten(); hb = (h[4].weight.detach() - hc[4].weight.detach())[~absent].flatten()
                    rets.append((float(a_ @ b_ / (a_ @ a_ + 1e-12)), float(hel @ hb / (hel @ hel + 1e-12))))
            else:
                real_phase(h, mix=True)
    with torch.no_grad():
        heads_acc = float(np.mean([n5.bal_acc(h(Xte), Yte) for h in heads]))
    er_acc = float(np.mean([n5.er(h, Xk, Yk, Xte, Yte) for h in heads]))
    out = dict(mode=mode, scheme=scheme, seed=seed, la=la, heads=round(100 * heads_acc, 2), er=round(100 * er_acc, 2))
    if rets:
        rr = np.array(rets)
        out.update(ret_absent=round(float(np.median(rr[:, 0])), 3), ret_held=round(float(np.median(rr[:, 1])), 3))
    return out


rows = []
for seed in range(3):
    for mode in ("faithful", "s=0.5", "agnostic"):
        for scheme in ("alt", "joint"):
            o = run(mode, scheme, seed, track=(scheme == "alt" and mode == "faithful"))
            rows.append(o); print(json.dumps(o), flush=True)
    o = run("faithful", "alt", seed, la=False, track=True); rows.append(o); print(json.dumps(o), flush=True)
json.dump(rows, open(f"{OUT}/n5b_out.json", "w"))
import collections
agg = collections.defaultdict(list)
for o in rows:
    agg[(o["mode"], o["scheme"], o["la"])].append((o["heads"], o["er"]))
for k, v in agg.items():
    print(k, "heads %.2f  ER %.2f" % tuple(np.array(v).mean(0)))
