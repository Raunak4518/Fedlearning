#!/usr/bin/env python3
"""
scripts/sweep.py

Runs a grid of experiments over imbalance_factor x dirichlet_alpha x seed,
for a chosen set of mechanism combinations -- this is the proposal's own
Week 6-8 milestone ("sweep imbalance factor x Dirichlet alpha on
CIFAR-10-LT and CIFAR-100-LT"). Results from every run are collected into
one CSV for easy comparison / plotting afterward.

    python scripts/sweep.py --config configs/cifar10_lt.yaml \\
        --imbalance_factors 0.1 0.05 0.01 --dir_params 1.0 0.3 0.1 --seeds 0 1 2
"""
import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from args import parse_args  # noqa: E402
from engine import run_gefl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--imbalance_factors", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    p.add_argument("--dir_params", type=float, nargs="+", default=[1.0, 0.3])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--mechanisms", type=str, nargs="+", default=["baseline", "a_only", "b_only", "proposed"],
                    choices=["baseline", "a_only", "b_only", "proposed"])
    p.add_argument("--out_csv", type=str, default="./logs/sweep_results.csv")
    sweep_args, remaining = p.parse_known_args()

    mech_flags = {
        "baseline": (0, 0), "a_only": (1, 0), "b_only": (0, 1), "proposed": (1, 1),
    }

    rows = []
    combos = list(itertools.product(sweep_args.imbalance_factors, sweep_args.dir_params,
                                     sweep_args.seeds, sweep_args.mechanisms))
    for i, (imb, dir_p, seed, mech) in enumerate(combos):
        mech_a, mech_b = mech_flags[mech]
        run_name = f"sweep_imb{imb}_dir{dir_p}_seed{seed}_{mech}"
        argv = ["--config", sweep_args.config, "--imbalance_factor", str(imb), "--dir_param", str(dir_p),
                "--seed", str(seed), "--mechanism_a", str(mech_a), "--mechanism_b", str(mech_b),
                "--name", run_name] + remaining
        args = parse_args(argv)
        print(f"[{i + 1}/{len(combos)}] {run_name}")
        results = run_gefl(args)
        final = results.get("final_scores", {})
        row = {"run": run_name, "imbalance_factor": imb, "dir_param": dir_p, "seed": seed, "mechanism": mech,
               **{k: v for k, v in final.items() if k.startswith("acc_")}}
        if "gap" in results:
            row.update({f"gap_{k}": v for k, v in results["gap"].items()})
        if "mnd_ratio" in results:
            row["mnd_ratio"] = results["mnd_ratio"]
        rows.append(row)

    os.makedirs(os.path.dirname(sweep_args.out_csv), exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r})
    with open(sweep_args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {sweep_args.out_csv}")


if __name__ == "__main__":
    main()
