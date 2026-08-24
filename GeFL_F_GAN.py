#!/usr/bin/env python3
"""
GeFL_F_GAN.py — GeFL-F with DCGAN feature-space generator.
"""
from args import parse_args, save_args
from gefl_f.engine_f import run_gefl_f
from utils.seed import set_seed


def main():
    args = parse_args()
    args.gen_model = "gan"
    args.gefl_f = 1
    base_name = args.name if args.name != "run" else "gefl_f_gan"

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
