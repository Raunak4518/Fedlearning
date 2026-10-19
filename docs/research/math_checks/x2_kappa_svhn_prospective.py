"""Prospective test of the count-only collapse predictor (paper Approximation 5, §3.5) on SVHN with K = 50 / 100
(K18), a dataset the predictor was not fitted or checked on. Prediction fixed in the paper before the runs returned:
every class row with kappa_c < 1 ends at <= 0.15 of its initial norm under GeFL-F (coupled Adam decay, flat
averaging), and none does under HWA. kappa_c is computed from the per-client class counts the runs log
(client_counts), exactly as in n6c_kappa_vs_logs.py. Output: x2_kappa_svhn.json
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, math
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, _HERE)
from n6_partition_stats import hyper_present, gen_steps
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import core

core.set_num_classes("svhn")
C = core.NUM_CLASSES
init = (1 / math.sqrt(core.DATASETS["svhn"]["latent"] + C)) * math.sqrt(512 / 3)
out = {}
for l in open(_os.path.join(_REPO, "experiments", "results", "K18_kaggle_more_clients_svhn", "runs.jsonl")):
    r = json.loads(l)
    if r["label"] not in ("GeFL-F", "GeFL-F + HWA + LA") or not r.get("cond_norms_end"):
        continue
    cnt = np.array(r["client_counts"]); nk = cnt.sum(1); K = len(nk)
    tau = np.array([gen_steps(int(n)) for n in nk], float)
    nr = np.array(r["cond_norms_end"]) / init
    for c in range(C):
        H = cnt[:, c] > 0
        if H.all():
            kap = float("inf")
        else:
            f = np.array([hyper_present(int(nk[k]), int(cnt[k, c]), min(64, int(nk[k]))) for k in range(K)])
            kap = float((tau[H] * f[H]).sum() / tau[~H].sum())
        out.setdefault(r["label"], []).append(dict(K=K, seed=r["seed"], c=c, kappa=kap, norm=float(nr[c])))
res = {}
for lab, rows in out.items():
    k = np.array([x["kappa"] for x in rows]); n = np.array([x["norm"] for x in rows])
    lo = k < 1
    fin = np.isfinite(k)
    res[lab] = dict(rows=len(rows), kappa_lt1=int(lo.sum()), collapsed_of_kappa_lt1=int((n[lo] <= 0.15).sum()),
                    max_norm_kappa_lt1=float(n[lo].max()) if lo.any() else None,
                    min_norm_kappa_ge1=float(n[~lo].min()) if (~lo).any() else None,
                    spearman_logkappa_norm=float(spearmanr(np.log(k[fin]), n[fin])[0]) if fin.sum() > 2 else None,
                    median_norm=float(np.median(n)))
print(json.dumps(res, indent=1))
json.dump(dict(summary=res, rows=out), open(_os.path.join(_HERE, "x2_kappa_svhn.json"), "w"), indent=1)
