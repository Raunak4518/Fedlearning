"""X14 (action plan §10, E-1): membership inference against the exact class statistics our method releases.

Release: per-class sums of the frozen features (secure-aggregated; the anchor m_c = S1_c / n_c). Attack (a standard
likelihood-ratio-style test for a released mean): the attacker knows a target's label c and holds public reference
data (half of the 2,000 held-out images) to estimate the population class mean mu_c; score(x) = <x - mu_c, m_c - mu_c>.
A member pulls m_c toward itself by (x - m_c) / n_c, so members score higher, most for rare classes (small n_c).
Evaluated: members (training samples of every client) vs non-members (the other half of the held-out images), AUC per
class group, for the exact release and for the Gaussian mechanism of E26 (clip at the 90% quantile of public feature
norms, zCDP calibration core.zcdp_rho, delta = 1e-5) at eps = 8 and 2. The FE itself has 36 parameters (MNIST /
FMNIST) and is not attacked. Output: x14_<ds>_s<seed>.json
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, math, os
import numpy as np
import torch
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import core


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort() + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(ds, seed):
    torch.set_num_threads(int(os.environ.get("NT", "1")))
    core.set_num_classes(ds)
    C = core.NUM_CLASSES
    data0 = core.load_dataset(ds, _os.path.join(_REPO, "experiments", "data"))
    part = core.make_partition(data0["ytr"], 10, core.DATASETS[ds]["frac"], 0.01, 0.5, seed)
    data, part = core.localize(data0, part)
    ctx = core.Ctx(core.default_cfg(), data, ds, part, 10, seed)
    core.set_seed(seed)
    fe_sd, _, fe = core.stage_i(ctx, core.DATASETS[ds]["T_FE"], "all")
    fe.load_state_dict(fe_sd)
    X = torch.cat([core.compute_features(fe, data["xtr"][p]) for p in part["parts"]]).flatten(1).double()
    Y = torch.cat([data["ytr"][p] for p in part["parts"]])
    H = core.compute_features(fe, data["xtr"][part["heldout"]]).flatten(1).double()
    Yh = data["ytr"][part["heldout"]]
    rng = np.random.RandomState(seed)
    ref = torch.from_numpy(rng.rand(len(Yh)) < 0.5)
    n = torch.bincount(Y, minlength=C).double()
    mu_ref = torch.stack([H[ref & (Yh == c)].mean(0) for c in range(C)])
    clip = float(torch.quantile(H[ref].norm(dim=1), 0.9))
    out = dict(ds=ds, seed=seed, n_c=n.int().tolist(), clip=clip)
    for name, eps in (("exact", None), ("eps8", 8.0), ("eps2", 2.0)):
        Xc = X if eps is None else X * (clip / X.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0)
        S1 = torch.zeros(C, X.shape[1], dtype=torch.float64).index_add_(0, Y, Xc)
        if eps is not None:
            sig = math.sqrt(2.0 / core.zcdp_rho(eps, 1e-5))
            g = torch.Generator().manual_seed(9001 + seed)
            S1 = S1 + sig * clip * torch.randn(S1.shape, generator=g, dtype=torch.float64)
        m = S1 / n[:, None]
        sc = lambda Z, L: ((Z - mu_ref[L]) * (m[L] - mu_ref[L])).sum(1).numpy()
        mem, non = sc(X, Y), sc(H[~ref], Yh[~ref])
        res = {}
        for grp, cls in (("all", range(C)), ("head", core.HEAD()), ("tail", core.TAIL())):
            cls = list(range(C))[cls] if isinstance(cls, slice) else list(cls)
            pm = np.isin(Y.numpy(), cls)
            pn = np.isin(Yh[~ref].numpy(), cls)
            # balance members to the non-member count per class group (AUC is insensitive to the ratio anyway)
            res[grp] = round(auc(mem[pm], non[pn]), 4)
        res["rarest_class"] = round(auc(mem[(Y == C - 1).numpy()], non[(Yh[~ref] == C - 1).numpy()]), 4)
        out[name] = res
    print(json.dumps(out), flush=True)
    json.dump(out, open(f"{_HERE}/x14_{ds}_s{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
