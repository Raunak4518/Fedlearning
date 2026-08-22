#!/usr/bin/env python3
"""
GeFL_DDPM.py

GeFL (optionally + Mechanism A / Mechanism B) using a Conditional DDPM as
the shared generator. Same CLI conventions as GeFL_CVAE.py. This is the
expensive variant the proposal itself flags as a stretch goal (diffusion
sampling cost is roughly three orders of magnitude higher than the
VAE/GAN variants per GeFL's own reported MACs) -- reduce --n_T and
--epochs for a quick sanity run:

    python GeFL_DDPM.py --dataset synthetic --n_T 20 --epochs 3 --num_users 3
"""
from args import parse_args, save_args
from engine import run_gefl
from utils.seed import set_seed


def main():
    args = parse_args()
    args.gen_model = "ddpm"
    base_name = args.name if args.name != "run" else "gefl_ddpm"

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
