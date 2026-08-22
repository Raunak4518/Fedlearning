#!/usr/bin/env python3
"""
GeFL_CVAE.py

GeFL (optionally + Mechanism A / Mechanism B) using a Conditional VAE as
the shared generator. This is the primary, most-tested entry point --
mirrors the original repo's `GeFL_CVAE.py` invocation style:

    # GeFL baseline (flat aggregation, uniform sampling) on synthetic data
    python GeFL_CVAE.py --dataset synthetic --num_users 5 --epochs 15 \\
        --imbalance_factor 0.02 --dir_param 0.4

    # proposed method on CIFAR-10-LT
    python GeFL_CVAE.py --config configs/cifar10_lt.yaml \\
        --mechanism_a 1 --mechanism_b 1 --name proposed_cifar10lt

    # plain FedAvg, no generator at all (aid_by_gen=0 baseline)
    python GeFL_CVAE.py --dataset cifar10 --aid_by_gen 0 --name fedavg_baseline

    # repeat 3 times with different seeds
    python GeFL_CVAE.py --config configs/cifar10_lt.yaml --num_experiment 3
"""
from args import parse_args, save_args
from engine import run_gefl
from utils.seed import set_seed


def main():
    args = parse_args()
    args.gen_model = "vae"
    base_name = args.name if args.name != "run" else "gefl_cvae"

    all_results = []
    base_seed = args.seed
    for exp_idx in range(args.num_experiment):
        args.seed = base_seed + exp_idx
        args.name = f"{base_name}_seed{args.seed}" if args.num_experiment > 1 else base_name
        set_seed(args.seed)
        save_args(args, f"{args.out_dir}/{args.name}_args.json")
        results = run_gefl(args)
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    main()
