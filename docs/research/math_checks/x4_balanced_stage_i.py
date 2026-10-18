"""X4 (action plan §10): pilot of P2, a class-balanced Stage-I loss with weights N / (C n_y) from the exact global
counts, clipped at 20 (core.stage_i(..., bal=20), component BALFE).

For one (dataset, seed), long tail IF = 100, Dirichlet 0.5, K = 10, it trains GeFL-F's federated FE twice — plain
('fed') and class-balanced ('fed_bal') — and scores each on the 2,000 held-out training images that no client holds
(the validation split the action plan fixes for selection):
  lda_raw   balanced LDA on the raw FE features (exact sums)
  one_client / pooled / pooled_bal bodies + ER, as in N7 (n7b_fe_body_decomp_sched.py)
Test-set raw LDA is also logged. Pre-registered prediction (math report §6, P2): ER of the bodies +1 +- 1 on
FashionMNIST, 0 +- 0.5 on MNIST, <= 0 on CIFAR-100.
Output: x4_<ds>_s<seed>.jsonl
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, time, os
import torch
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
sys.path.insert(0, _HERE)
import core
from n7b_fe_body_decomp_sched import lda_raw, bodies

ROOT = _os.path.join(_REPO, "experiments", "data")


def run(ds, seed):
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    core.set_num_classes(ds)
    data0 = core.load_dataset(ds, ROOT)
    cfg = core.default_cfg()
    fout = open(f"{_HERE}/x4_{ds}_s{seed}.jsonl", "a")
    part = core.make_partition(data0["ytr"], 10, core.DATASETS[ds]["frac"], 0.01, 0.5, seed)
    data, part = core.localize(data0, part)
    ctx = core.Ctx(cfg, data, ds, part, 10, seed)
    xs = [data["xtr"][p] for p in part["parts"]]
    ys = [data["ytr"][p] for p in part["parts"]]
    hold = part["heldout"]
    xv, yv = data["xtr"][hold], data["ytr"][hold]
    for name, bal in (("fed", 0.0), ("fed_bal", 20.0)):
        t = time.time()
        core.set_seed(seed)
        fe_sd, _, fe = core.stage_i(ctx, core.DATASETS[ds]["T_FE"], "all", bal=bal)
        fe.load_state_dict(fe_sd)
        feats = [core.compute_features(fe, x) for x in xs]
        fv = core.compute_features(fe, xv)
        fte = core.compute_features(fe, data["xte"])
        r = dict(fe=name, ds=ds, seed=seed,
                 lda_raw_val=lda_raw(feats, ys, fv, yv, core.NUM_CLASSES),
                 lda_raw_test=lda_raw(feats, ys, fte, data["yte"], core.NUM_CLASSES))
        r.update({k + "_val": v for k, v in bodies(ctx, feats, ys, fv, yv, name, seed=seed).items()})
        r["t"] = round(time.time() - t)
        print(json.dumps(r), flush=True)
        fout.write(json.dumps(r) + "\n")
        fout.flush()


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]))
