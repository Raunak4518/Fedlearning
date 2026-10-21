"""N6c: out-of-sample test of Approximation 2.5 (count-only collapse threshold kappa_c < 1) against the per-class
final row norms logged by every GeFL-F run (cond_norms_end), coupled Adam decay 1e-3 + flat FedAvg.
kappa_c = sum_{k in H_c} tau_k f_kc / sum_{k not in H_c} tau_k (tau_k: local Adam steps, f_kc: P(class c in a batch)).
Initial row norm: uniform(-b, b) rows, b = 1/sqrt(16 + C) (latent 16), d = 512 -> E||r|| ~ b sqrt(d/3)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, glob, math, collections
import numpy as np
sys.path.insert(0, _HERE)
from n6_partition_stats import labels, hyper_present, gen_steps
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import core
runs = collections.defaultdict(list)
for f in glob.glob(_os.path.join(_REPO, "experiments", "results", "*", "runs.jsonl")):
    for l in open(f):
        r = json.loads(l)
        m = r.get("method") or {}
        g = m.get("gen") or {}
        if r.get("quick") or "IF" not in r or r["IF"] != 0.01 or not r.get("cond_norms_end") \
                or "[p=" in r.get("label", ""):  # partial participation: kappa assumes every client takes part
            continue
        if False:
            continue
        if g.get("type") == "cvae" and g.get("agg") == "flat" and g.get("wd") == 1e-3 and g.get("opt", "adam") == "adam" \
                and not g.get("lazy") and not g.get("rowwd0") and r["dataset"] in ("mnist", "fmnist", "cifar10", "cifar100"):
            runs[(r["dataset"], r["K"], r["seed"])].append(np.array(r["cond_norms_end"]))
rows = []
for (ds, K, seed), norms in sorted(runs.items()):
    core.set_num_classes(ds)
    C = core.NUM_CLASSES
    part = core.make_partition(labels(ds), K, core.DATASETS[ds]["frac"], 0.01, 0.5, seed)
    cnt = part["counts"]; nk = cnt.sum(1)
    tau = np.array([gen_steps(int(n)) for n in nk], float)
    lat = core.DATASETS[ds]["latent"]
    init = (1 / math.sqrt(lat + C)) * math.sqrt(512 / 3)
    nr = np.mean(norms, 0) / init
    for c in range(C):
        H = cnt[:, c] > 0
        if H.all():
            kap = np.inf
        else:
            f = np.array([hyper_present(int(nk[k]), int(cnt[k, c]), min(64, int(nk[k]))) for k in range(K)])
            kap = (tau[H] * f[H]).sum() / tau[~H].sum()
        rows.append((ds, K, seed, c, kap, nr[c], len(norms)))
rows = np.array(rows, dtype=object)
for ds in ("mnist", "fmnist", "cifar10", "cifar100"):
    sel = rows[rows[:, 0] == ds]
    if len(sel) == 0:
        continue
    kap = sel[:, 4].astype(float); nr = sel[:, 5].astype(float)
    lo, hi = kap < 1, kap >= 1
    print(f"{ds}: settings {sorted(set((int(a), int(b)) for a, b in zip(sel[:, 1], sel[:, 2])))}")
    print(f"   kappa<1 : n={lo.sum():3d} rows, end/init norm median {np.median(nr[lo]) if lo.any() else float('nan'):.3f}, max {nr[lo].max() if lo.any() else float('nan'):.3f}")
    print(f"   kappa>=1: n={hi.sum():3d} rows, end/init norm median {np.median(nr[hi]):.3f}, min {nr[hi].min():.3f}")
    for thr in (0.3, 0.5):
        pred = kap < 1; act = nr < thr
        print(f"   threshold end/init<{thr}: accuracy of 'kappa<1' as a predictor = {(pred == act).mean():.3f}  (TP {int((pred&act).sum())}, FP {int((pred&~act).sum())}, FN {int((~pred&act).sum())})")
from scipy.stats import spearmanr
fin = np.isfinite(rows[:, 4].astype(float))
print("Spearman(log kappa, end/init norm) over rows with a non-holder:", spearmanr(np.log(rows[fin, 4].astype(float)), rows[fin, 5].astype(float)))
print("end/init row norm by kappa bin (all datasets):")
kap = rows[:, 4].astype(float); nr = rows[:, 5].astype(float)
for lo_, hi_ in ((0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 1e9), (1e9, np.inf)):
    s = (kap >= lo_) & (kap < hi_) if np.isfinite(hi_) else ~np.isfinite(kap)
    if s.any():
        print(f"   kappa in [{lo_}, {hi_}): n={s.sum():3d}  median {np.median(nr[s]):.3f}  IQR [{np.quantile(nr[s], .25):.3f}, {np.quantile(nr[s], .75):.3f}]")
json.dump([dict(ds=r[0], K=int(r[1]), seed=int(r[2]), c=int(r[3]), kappa=(None if not np.isfinite(float(r[4])) else float(r[4])),
                norm=float(r[5])) for r in rows], open(_os.path.join(_HERE, "n6c_rows.json"), "w"))
