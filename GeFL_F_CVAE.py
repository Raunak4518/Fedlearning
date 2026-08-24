#!/usr/bin/env python3
"""
GeFL_F_CVAE.py

GeFL-F (feature extractor variant) with a Conditional VAE as the
shared feature-space generator. Implements Algorithm 3 from Kang et al.
2025 — three-stage training with a common feature extractor.

    # GeFL-F on CIFAR-10-LT with CVAE feature generator
    python GeFL_F_CVAE.py --config configs/cifar10_gefl_f.yaml

    # With mechanisms A+B
    python GeFL_F_CVAE.py --config configs/cifar10_gefl_f.yaml \\
        --mechanism_a 1 --mechanism_b 1 --name gefl_f_proposed
"""
from args import parse_args, save_args
from gefl_f.engine_f import run_gefl_f
from utils.seed import set_seed


def main():
    args = parse_args()
    args.gen_model = "vae"
    args.gefl_f = 1
    base_name = args.name if args.name != "run" else "gefl_f_cvae"

    all_results = []
    base_seed = args.seed
    for exp_idx in range(args.num_experiment):
        args.seed = base_seed + exp_idx
        args.name = f"{base_name}_seed{args.seed}" if args.num_experiment > 1 else base_name
        set_seed(args.seed)
        save_args(args, f"{args.out_dir}/{args.name}_args.json")
        results = run_gefl_f(args)
        all_results.append(results)
    return all_results


if __name__ == "__main__":
    main()
