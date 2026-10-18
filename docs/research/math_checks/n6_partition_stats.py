"""N6: statistics of the ACTUAL client partitions (core.make_partition), no training.

For each (dataset, K, seed): reconstruct counts n_kc exactly (validated against the n_train / min_client /
tail_holders fields recorded in runs.jsonl), then compute the count-only quantities used in the report:
  * FedNova surrogate weights of Stage I (uniform 1/K averaging x tau_k local steps) vs pooled-ERM weights n_k/N
  * pair-pressure ratio rho_cc' = sum_k n_kc pi_kc' / (n_c pi_c')   (Sec. 2a heuristic)
  * variance-inflation factor of the saturated flat-FedAvg row fixed point  V_c = (n_c / m_c^2) sum_{k in H_c} 1/n_kc
  * generator step-count ratios kappa_c = sum_{H} tau_k f_kc / sum_{not H} tau_k  and the sparse-Adam variant with sqrt(f)
  * per-class dilution rate of flat FedAvg without decay  (1/K) sum_k tau_k f_kc  relative to HWA
Run:  python n6_partition_stats.py
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, math
import numpy as np
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import torch
import core

ROOT = _os.path.join(_REPO, "experiments", "data")


def labels(ds):
    if ds in ("mnist", "fmnist", "cifar100"):
        d = torch.load(f"{ROOT}/_gefl_tensors_{ds}.pt")
        return d["ytr"]
    if ds == "cifar10":
        return torch.arange(10).repeat_interleave(5000)
    raise ValueError(ds)


def hyper_present(n, nc, b):
    """P(at least one class-c sample in a batch of b drawn without replacement from n with nc of class c)."""
    if nc == 0:
        return 0.0
    if n - nc < b:
        return 1.0
    # log C(n-nc, b) - log C(n, b)
    lp = sum(math.log((n - nc - i) / (n - i)) for i in range(b))
    return 1.0 - math.exp(lp)


def gen_steps(n, B=64, epochs=5):
    """local Adam steps in gen_local_update: full batches + last batch if size >= 2."""
    nb = n // B + (1 if n % B >= 2 else 0)
    return epochs * nb


def fe_steps(n, B, epochs=5):
    nb = n // B + (1 if n % B >= 2 else 0)
    return epochs * nb


def stats(ds, K, seed, IF=0.01, alpha=0.5):
    core.set_num_classes(ds)
    y = labels(ds)
    part = core.make_partition(y, K, core.DATASETS[ds]["frac"], IF, alpha, seed)
    cnt = part["counts"].astype(float)  # K x C
    C = cnt.shape[1]
    nk = cnt.sum(1)
    nc = cnt.sum(0)
    N = nc.sum()
    H = cnt > 0
    m = H.sum(0)
    out = dict(ds=ds, K=K, seed=seed, n_train=int(N), min_client=int(nk.min()),
               tail_holders=int(H[:, core.TAIL()].sum(0).min()))
    # Stage I FedNova surrogate weights
    B = core.DATASETS[ds]["B"]
    tau = np.array([fe_steps(int(n), B) for n in nk], float)
    w_fednova = tau / tau.sum()
    out["stageI_TV_fednova_vs_pooled"] = float(0.5 * np.abs(w_fednova - nk / N).sum())
    out["stageI_tau_range"] = [int(tau.min()), int(tau.max())]
    # pair pressure
    pik = cnt / nk[:, None]
    pig = nc / N
    rho = np.zeros((C, C))
    for c in range(C):
        for c2 in range(C):
            if c != c2:
                rho[c, c2] = (cnt[:, c] * pik[:, c2]).sum() / (nc[c] * pig[c2])
    off = ~np.eye(C, dtype=bool)
    co_held = (H.T.astype(int) @ H.astype(int))  # number of clients holding both
    out["rho_mean"] = float(rho[off].mean())
    out["rho_min"] = float(rho[off].min())
    tail = list(range(C))[core.TAIL()]
    head = list(range(C))[core.HEAD()]
    out["rho_tail_tail"] = float(np.mean([rho[a, b] for a in tail for b in tail if a != b]))
    out["rho_tail_head"] = float(np.mean([rho[a, b] for a in tail for b in head]))
    out["pairs_never_coheld"] = int(((co_held == 0) & off).sum() // 2)
    out["n_pairs"] = int(C * (C - 1) // 2)
    # variance inflation of saturated flat FedAvg (uniform over holders) vs pooled mean
    V = np.array([(nc[c] / m[c] ** 2) * (1.0 / cnt[H[:, c], c]).sum() for c in range(C)])
    out["Vinfl_per_class"] = V.round(2).tolist()
    out["Vinfl_tail_mean"] = float(V[tail].mean())
    out["Vinfl_head_mean"] = float(V[head].mean())
    # generator step ratios
    taug = np.array([gen_steps(int(n)) for n in nk], float)
    f = np.array([[hyper_present(int(nk[k]), int(cnt[k, c]), min(64, int(nk[k]))) for c in range(C)] for k in range(K)])
    kappa, kappa_s, dil = [], [], []
    for c in range(C):
        hold = H[:, c]
        nh = taug[~hold].sum()
        num = (taug[hold] * f[hold, c]).sum()
        nums = (taug[hold] * np.sqrt(f[hold, c])).sum()
        kappa.append(num / nh if nh > 0 else float("inf"))
        kappa_s.append(nums / nh if nh > 0 else float("inf"))
        dil.append((taug * f[:, c]).sum() / K)  # flat, no decay: rate in units of eta per round
    out["m_c"] = m.astype(int).tolist()
    out["n_c"] = nc.astype(int).tolist()
    out["kappa"] = [round(x, 2) if np.isfinite(x) else None for x in kappa]
    out["kappa_sqrtf"] = [round(x, 2) if np.isfinite(x) else None for x in kappa_s]
    out["flat_rate_steps"] = np.round(dil, 1).tolist()
    # HWA (E(n) weights, beta=.999) rate: sum_k w_k tau_k f_kc / sum_k w_k
    E = (1 - 0.999 ** cnt) / (1 - 0.999)
    hwa = [(E[:, c] * taug * f[:, c]).sum() / E[:, c].sum() for c in range(C)]
    out["hwa_rate_steps"] = np.round(hwa, 1).tolist()
    out["hwa_over_flat_tail"] = float(np.mean([hwa[c] / dil[c] for c in tail]))
    return out, cnt


if __name__ == "__main__":
    res = []
    for ds in ("mnist", "fmnist", "cifar10", "cifar100"):
        for K in ((10, 50, 100) if ds in ("mnist", "fmnist") else (10,)):
            for seed in (0, 1, 2):
                o, cnt = stats(ds, K, seed)
                res.append(o)
                print(json.dumps({k: v for k, v in o.items() if k not in ("Vinfl_per_class", "kappa", "kappa_sqrtf",
                                                                          "flat_rate_steps", "hwa_rate_steps", "m_c", "n_c")}))
                if ds == "mnist" and K == 10 and seed == 0:
                    for k in ("m_c", "n_c", "kappa", "kappa_sqrtf", "Vinfl_per_class", "flat_rate_steps", "hwa_rate_steps"):
                        print("   ", k, o[k])
                    np.save("" + _HERE + "/mnist_s0_K10_counts.npy", cnt)
    json.dump(res, open("" + _HERE + "/n6_out.json", "w"), indent=1)
