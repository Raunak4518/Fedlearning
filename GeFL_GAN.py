#!/usr/bin/env python3
"""
GeFL_GAN.py

GeFL (optionally + Mechanism A / Mechanism B) using a Conditional DCGAN as
the shared generator. Same CLI conventions as GeFL_CVAE.py -- see its
docstring for examples; just swap the script name. GAN-specific
hyperparameters (--b1, --b2 for the Adam betas) are exposed in args.py.

    python GeFL_GAN.py --config configs/cifar10_lt.yaml --mechanism_a 1 --mechanism_b 1
"""
from args import parse_args, save_args
from engine import run_gefl
from utils.seed import set_seed


def main():
    args = parse_args()
    args.gen_model = "gan"
    base_name = args.name if args.name != "run" else "gefl_gan"

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
