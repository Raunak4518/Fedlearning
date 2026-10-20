"""E40 follow-up: one predictor for centralised and federated collapse. For each item row, fbar = share of all
optimiser steps (over all clients) whose batch contains the item: centrally p_c = 1 - (1 - n_c / N)^B, federated
fbar_c = sum_k tau_k f_kc / sum_k tau_k. Adam's second moment forgets with time constant 1 / (1 - beta2) = 1000 steps,
so rows seen less often than that get decay steps of near-full size between their data steps (centrally too);
federated non-holders add pure-decay steps on top. Seed 0. Output: x5_step_frequency_mf.json"""
import os as _os, sys, json, math
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import numpy as np
from scipy.stats import spearmanr
import fedembed as fe

R = [json.loads(l) for l in open(_os.path.join(_REPO, "experiments", "results", "E40_fedembed_mf", "runs.jsonl"))]
seed, B = 0, 256
u, i, r, test = fe.load_ml1m(seed); tr = ~test
U, V = u.max() + 1, i.max() + 1
N = tr.sum(); n_c = np.bincount(i[tr], minlength=V)
p_cen = 1 - (1 - n_c / N) ** B
norm_c = np.array([x for x in R if x["arm"] == "central_l2" and x["seed"] == seed][0]["norm_ratio"])
bins = [(0, 1 / 3000), (1 / 3000, 1 / 1000), (1 / 1000, 1 / 300), (1 / 300, 1 / 100), (1 / 100, 1)]
out = dict(central=dict(spearman=float(spearmanr(np.log(p_cen + 1e-12), norm_c)[0]),
                        bins=[dict(lo=lo, hi=hi, n=int(((p_cen >= lo) & (p_cen < hi)).sum()),
                                   median_norm=float(np.median(norm_c[(p_cen >= lo) & (p_cen < hi)]))) for lo, hi in bins]))
for K in [10, 50]:
    rng = np.random.RandomState(seed + 1); cl = rng.randint(0, K, size=U)[u]
    counts = np.array([np.bincount(i[tr & (cl == k)], minlength=V) for k in range(K)]); nk = counts.sum(1)
    tau = np.array([math.ceil(n / B) for n in nk], float)
    f = np.array([[fe.hyper_present(int(nk[k]), int(counts[k, c]), min(B, int(nk[k]))) for c in range(V)] for k in range(K)])
    fbar = (tau[:, None] * f).sum(0) / tau.sum()
    nf = np.array([x for x in R if x["arm"] == "adam_l2" and x["seed"] == seed and x["K"] == K][0]["norm_ratio"])
    out[f"federated_K{K}"] = dict(spearman=float(spearmanr(np.log(fbar + 1e-12), nf)[0]),
                                  bins=[dict(lo=lo, hi=hi, n=int(((fbar >= lo) & (fbar < hi)).sum()),
                                             median_norm=float(np.median(nf[(fbar >= lo) & (fbar < hi)])),
                                             median_norm_same_rows_central=float(np.median(norm_c[(fbar >= lo) & (fbar < hi)])))
                                        for lo, hi in bins])
print(json.dumps(out, indent=1))
json.dump(out, open(_os.path.join(_HERE, "x5_step_frequency_mf.json"), "w"), indent=1)
