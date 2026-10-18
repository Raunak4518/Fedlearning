"""N6b: first-order pair weights of the Stage-I FE objective (Prop. 2.1(b)):
   fed  w_cc' = sum_k (n_k/N) pi_kc pi_kc'      cen  w_cc' = pi_c pi_c'      balanced  w = 1/C^2
reported: spread of log10 weights over unordered pairs, tail-tail / head-head weight ratio, and the
distance (in log10) of each weighting from the balanced one."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, numpy as np
sys.path.insert(0, _HERE)
from n6_partition_stats import labels
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import core
for ds in ("mnist", "fmnist", "cifar10", "cifar100"):
    for seed in (0, 1, 2):
        core.set_num_classes(ds)
        part = core.make_partition(labels(ds), 10, core.DATASETS[ds]["frac"], 0.01, 0.5, seed)
        cnt = part["counts"].astype(float); C = cnt.shape[1]
        nk = cnt.sum(1); N = nk.sum(); pik = cnt / nk[:, None]; pig = cnt.sum(0) / N
        wf = np.einsum("k,kc,kd->cd", nk / N, pik, pik); wc = np.outer(pig, pig)
        iu = np.triu_indices(C, 1)
        lf, lc = np.log10(wf[iu]), np.log10(wc[iu]); lb = np.log10(1 / C ** 2)
        T = list(range(C))[core.TAIL()]; Hh = list(range(C))[core.HEAD()]
        tt = lambda w: np.mean([w[a, b] for a in T for b in T if a < b]) / np.mean([w[a, b] for a in Hh for b in Hh if a < b])
        print(f"{ds:8s} s{seed}: sd log10 w  fed {lf.std():.2f}  cen {lc.std():.2f} | mean |log10 w - log10 1/C^2| fed {np.abs(lf-lb).mean():.2f} cen {np.abs(lc-lb).mean():.2f} | tail-tail/head-head fed {tt(wf):.2e} cen {tt(wc):.2e}")
