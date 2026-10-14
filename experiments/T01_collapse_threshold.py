"""T01 - Verify the conditioning-row contraction theorem (paper section 3.1) on the real CVAE-F.

The theorem: under flat FedAvg, for class c's conditioning row r_c,
    ||r_c^+|| <= ||r_c|| + (1/K) (sum_{holders} G_k - sum_{non-holders} D_k),
with G_k = ||r_kc|| - ||r_c|| (holder's norm change over its local round) and
D_k = ||r_c|| - ||r_kc|| (non-holder's norm loss). The row contracts whenever
m_c / K < theta_c = Dbar / (Gbar + Dbar). The old statement (m_c < K/2) is the
special case Gbar = Dbar.

This script runs GeFL-F's stage (i) and an instrumented stage (ii) - the paper's
CVAE-F, Adam (lr 1e-3) with coupled weight decay lambda, persistent per-client
optimizer state, flat FedAvg or HWA - and records, every round, every client's
post-local norm of every class row. Arms: lambda in {0, 1e-4, 1e-3} with flat
FedAvg, and lambda = 1e-3 with HWA.

    python T01_collapse_threshold.py --dataset mnist --seed 0 --out_dir results/T01_collapse_threshold
then  python T01_analyze.py results/T01_collapse_threshold
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import K09_kaggle_ablation_final as E

ARMS = [("GeFL-F wd=1e-3", 1e-3, "flat"), ("GeFL-F wd=1e-4", 1e-4, "flat"),
        ("GeFL-F wd=0", 0.0, "flat"), ("HWA wd=1e-3", 1e-3, "A")]


def instrumented_stage_ii(ctx, gspec, feats, ys_list, rounds):
    """E.stage_ii for the CVAE-F, plus per-client row norms after every local round."""
    G = E.make_gen(ctx, gspec, feats[0].shape[1], feats[0].shape[1:])
    g_sd = E.sd_clone(G)
    clients = [E.GenClient(G, gspec, False) for _ in range(ctx.K)]
    W = E.class_weights(ctx.part["counts"], gspec, np.random.RandomState(ctx.seed + 7))
    glob = [E.cond_row_norms(G)]
    local = []                                   # rounds x K x C
    for _ in range(rounds):
        avg = E.Averager(G.cond if gspec["agg"] == "A" else None, prev=g_sd)
        loc = []
        for k in range(ctx.K):
            G.load_state_dict(g_sd)
            E.gen_local_update(G, clients[k], gspec, feats[k], ys_list[k])
            loc.append(E.cond_row_norms(G))
            avg.add(G.state_dict(), W[k] if gspec["agg"] == "A" else None)
        g_sd = avg.result()
        G.load_state_dict(g_sd)
        glob.append(E.cond_row_norms(G))
        local.append(np.stack(loc))
    return np.array(glob), np.array(local)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--IF", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=None, help="generator rounds (default: T_KA)")
    ap.add_argument("--fe_rounds", type=int, default=None, help="stage (i) rounds (default: T_FE)")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./results/T01_collapse_threshold")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    spec = E.DATASETS[a.dataset]
    data = E.load_dataset(a.dataset, a.data_root)
    part = E.make_partition(data["ytr"], a.K, spec["frac"], a.IF, a.alpha, a.seed)
    data, part = E.localize(data, part)
    ctx = E.Ctx(dict(quick=False), data, a.dataset, part, a.K, a.seed)
    E.set_seed(a.seed)
    t0 = time.time()
    fe_sd, _, fe = E.stage_i(ctx, a.fe_rounds or spec["T_FE"], "all")
    fe.load_state_dict(fe_sd)
    feats = [E.compute_features(fe, data["xtr"][p]) for p in part["parts"]]
    ys_list = [data["ytr"][p] for p in part["parts"]]
    E.log(f"stage (i) done in {time.time() - t0:.0f}s")
    rounds = a.rounds or spec["T_KA"]
    counts = np.asarray(part["counts"])
    sizes = np.array([len(y) for y in ys_list])
    for label, wd, agg in ARMS:
        t1 = time.time()
        E.set_seed(a.seed + 1000)
        gspec = dict(E.GEN_DEFAULTS, type="cvae", wd=wd, agg=agg)
        glob, local = instrumented_stage_ii(ctx, gspec, feats, ys_list, rounds)
        tag = f"{a.dataset}_s{a.seed}_{label.replace(' ', '_').replace('=', '')}"
        np.savez_compressed(os.path.join(a.out_dir, tag + ".npz"), glob=glob, local=local,
                            counts=counts, sizes=sizes)
        with open(os.path.join(a.out_dir, tag + ".json"), "w") as f:
            json.dump(dict(label=label, wd=wd, agg=agg, dataset=a.dataset, seed=a.seed, K=a.K, IF=a.IF,
                           alpha=a.alpha, rounds=rounds, batch=64, local_epochs=5, lr=1e-3,
                           time_s=time.time() - t1), f)
        E.log(f"{label}: tail/head row norm end = "
              f"{glob[-1, 6:].mean() / glob[-1, :3].mean():.3f}  ({time.time() - t1:.0f}s)")


if __name__ == "__main__":
    main()
