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

After all runs complete, auto-generates sweep heatmaps and comparison charts.
"""
import argparse
import csv
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from args import parse_args  # noqa: E402
from engine import run_gefl  # noqa: E402


def _generate_sweep_plots(rows, out_dir):
    """Auto-generate visualization from sweep results."""
    try:
        from utils.visualize import plot_sweep_heatmap, plot_bucket_comparison
    except ImportError:
        print("matplotlib not available, skipping sweep plots")
        return

    plots_dir = os.path.join(out_dir, "plots", "sweep")
    os.makedirs(plots_dir, exist_ok=True)

    # Heatmaps for key metrics
    for metric in ("acc_overall", "acc_head", "acc_medium", "acc_tail"):
        if any(metric in r for r in rows):
            try:
                plot_sweep_heatmap(
                    rows, metric,
                    os.path.join(plots_dir, f"sweep_{metric}.png"),
                    title=f"Sweep: {metric.replace('acc_', '').title()} Accuracy"
                )
            except Exception as e:
                print(f"Warning: could not generate sweep heatmap for {metric}: {e}")

    # Mechanism comparison at each (imb, dir) setting
    try:
        imb_dir_pairs = {(r["imbalance_factor"], r["dir_param"]) for r in rows}
        for imb, dir_p in sorted(imb_dir_pairs):
            subset = [r for r in rows
                      if r["imbalance_factor"] == imb and r["dir_param"] == dir_p]
            method_scores = {}
            for r in subset:
                mech = r["mechanism"]
                if mech not in method_scores:
                    method_scores[mech] = {}
                for k in ("acc_overall", "acc_head", "acc_medium", "acc_tail"):
                    bucket = k.replace("acc_", "")
                    if k in r:
                        # Average across seeds
                        if bucket in method_scores[mech]:
                            method_scores[mech][bucket] = (method_scores[mech][bucket] + r[k]) / 2
                        else:
                            method_scores[mech][bucket] = r[k]

            if method_scores:
                safe_name = f"imb{imb}_dir{dir_p}".replace(".", "p")
                plot_bucket_comparison(
                    method_scores,
                    os.path.join(plots_dir, f"comparison_{safe_name}.png"),
                    title=f"Mechanism Comparison (IF={imb}, α={dir_p})"
                )
    except Exception as e:
        print(f"Warning: could not generate comparison charts: {e}")

    # Aggregate statistics table
    try:
        _write_aggregate_stats(rows, os.path.join(plots_dir, "aggregate_stats.csv"))
    except Exception as e:
        print(f"Warning: could not write aggregate stats: {e}")

    print(f"Sweep plots saved to {plots_dir}")


def _write_aggregate_stats(rows, out_path):
    """Write mean ± std across seeds for each (imb, dir, mechanism) combo."""
    import numpy as np

    groups = {}
    for r in rows:
        key = (r["imbalance_factor"], r["dir_param"], r["mechanism"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    agg_rows = []
    for (imb, dir_p, mech), group in sorted(groups.items()):
        agg = {
            "imbalance_factor": imb,
            "dir_param": dir_p,
            "mechanism": mech,
            "n_seeds": len(group),
        }
        for metric in ("acc_overall", "acc_head", "acc_medium", "acc_tail"):
            vals = [r[metric] for r in group if metric in r]
            if vals:
                agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
        agg_rows.append(agg)

    if agg_rows:
        fieldnames = list(agg_rows[0].keys())
        for r in agg_rows[1:]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(agg_rows)
        print(f"Aggregate stats: {out_path}")


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
    total = len(combos)

    print(f"Starting sweep: {total} experiments")
    print(f"  imbalance_factors: {sweep_args.imbalance_factors}")
    print(f"  dir_params: {sweep_args.dir_params}")
    print(f"  seeds: {sweep_args.seeds}")
    print(f"  mechanisms: {sweep_args.mechanisms}")
    print("=" * 60)

    for i, (imb, dir_p, seed, mech) in enumerate(combos):
        mech_a, mech_b = mech_flags[mech]
        run_name = f"sweep_imb{imb}_dir{dir_p}_seed{seed}_{mech}"
        argv = ["--config", sweep_args.config, "--imbalance_factor", str(imb), "--dir_param", str(dir_p),
                "--seed", str(seed), "--mechanism_a", str(mech_a), "--mechanism_b", str(mech_b),
                "--name", run_name] + remaining
        args = parse_args(argv)
        print(f"\n[{i + 1}/{total}] {run_name}")
        results = run_gefl(args)
        final = results.get("final_scores", {})
        row = {"run": run_name, "imbalance_factor": imb, "dir_param": dir_p, "seed": seed, "mechanism": mech,
               **{k: v for k, v in final.items() if k.startswith("acc_")}}
        # Include new metrics if available
        for extra_key in ("macro_f1", "weighted_f1", "class_balanced_accuracy",
                          "gen_mean_confidence", "gen_label_accuracy"):
            if extra_key in final:
                row[extra_key] = final[extra_key]
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
    print(f"\nWrote {len(rows)} rows to {sweep_args.out_csv}")

    # Auto-generate sweep visualizations
    out_dir = os.path.dirname(sweep_args.out_csv) or "./logs"
    _generate_sweep_plots(rows, out_dir)


if __name__ == "__main__":
    main()
