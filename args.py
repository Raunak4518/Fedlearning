"""
args.py

Single source of truth for every experiment parameter. Follows the same
convention as the original GeFL reference implementation
(honggkang/hetero-model-fl-gen): one flat argparse namespace, grouped by
comment headers, everything overridable from the command line so nothing
in the rest of the codebase is hardcoded.

A YAML file can additionally be passed via --config; its keys are used as
new argparse *defaults* before CLI parsing, so CLI flags always win. This
gives you both a reproducible, diffable experiment record (the YAML) and
the original repo's quick command-line override style at the same time.

    python GeFL_CVAE.py --config configs/cifar10_lt.yaml --seed 1
    python GeFL_CVAE.py --dataset cifar100 --num_users 20 --imbalance_factor 0.01
"""
import argparse
import json
import os

import yaml


def _add_all_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # ---------------------------------------------------------------- config
    parser.add_argument('--config', type=str, default=None,
                         help='optional YAML file; its keys override the defaults below '
                              '(CLI flags still override the YAML)')

    # ---------------------------------------------------------------- clients
    parser.add_argument('--num_users', type=int, default=10, help='number of federated clients')
    parser.add_argument('--frac', type=float, default=1.0, help='fraction of clients sampled each round')
    parser.add_argument('--num_models', type=int, default=3,
                         help='number of distinct target-network architectures in the heterogeneous pool; '
                              'clients are assigned round-robin, dev_spec_idx = idx %% num_models')

    # ---------------------------------------------------------------- dataset (dataset-agnostic)
    parser.add_argument('--dataset', type=str, default='synthetic',
                         choices=['cifar10', 'cifar100', 'mnist', 'fmnist', 'svhn', 'stl10', 'synthetic'],
                         help='any registered dataset name; new datasets are added by registering them in '
                              'datasets/ -- nothing elsewhere references a dataset name directly')
    parser.add_argument('--data_root', type=str, default='./data', help='torchvision download / cache root')
    parser.add_argument('--img_size', type=int, default=None,
                         help='override the dataset\'s native resolution (images are resized); '
                              'default: use the dataset\'s native size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker processes')

    # ---------------------------------------------------------------- long-tail + non-IID partitioning
    parser.add_argument('--noniid', action='store_true', default=True,
                         help='partition training data across clients as non-IID (Dirichlet); '
                              'if false, falls back to an IID shard split')
    parser.add_argument('--iid', dest='noniid', action='store_false',
                         help='force an IID partition (overrides --noniid)')
    parser.add_argument('--dir_param', type=float, default=0.3,
                         help='Dirichlet concentration alpha for cross-client heterogeneity '
                              '(low = more heterogeneous)')
    parser.add_argument('--imbalance_factor', type=float, default=1.0,
                         help='global long-tail severity as min/max per-class count ratio; '
                              '1.0 = perfectly balanced dataset (no long tail applied), '
                              '0.01 = severe long tail (matches the CIFAR-10-LT/100-LT convention used by CReFF)')
    parser.add_argument('--max_per_class', type=int, default=None,
                         help='cap samples for the majority class before applying imbalance_factor; '
                              'default: use every available sample of the majority class')
    parser.add_argument('--partition_seed', type=int, default=0, help='seed for the client partition only')

    # ---------------------------------------------------------------- optimizer / training
    parser.add_argument('--bs', type=int, default=128, help='evaluation batch size')
    parser.add_argument('--local_bs', type=int, default=64, help='local training batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='target-network SGD learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])

    # ---------------------------------------------------------------- reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_experiment', type=int, default=1, help='number of repeated runs (seed, seed+1, ...)')
    parser.add_argument('--device', type=str, default='cuda' if _cuda_available() else 'cpu')

    # ---------------------------------------------------------------- GeFL / GeFL-F federated rounds
    parser.add_argument('--gen_wu_epochs', type=int, default=5, help='generator warm-up communication rounds')
    parser.add_argument('--epochs', type=int, default=15, help='main communication rounds')
    parser.add_argument('--local_ep', type=int, default=1, help='local epochs for target-network training')
    parser.add_argument('--gen_local_ep', type=int, default=1, help='local epochs for generator training')
    parser.add_argument('--aid_by_gen', type=int, default=1,
                         help='1: augment local target-net training with synthetic samples from the shared '
                              'generator (GeFL). 0: plain FedAvg baseline with no generator at all')
    parser.add_argument('--freeze_gen', type=int, default=0,
                         help='1: stop updating the generator after warm-up; 0: keep training it every round')
    parser.add_argument('--synth_batch', type=int, default=64,
                         help='number of synthetic samples drawn per client per round')

    # ---------------------------------------------------------------- generator choice + shared params
    parser.add_argument('--gen_model', type=str, default='vae', choices=['vae', 'gan', 'ddpm'],
                         help='which registered conditional generator architecture to use')
    parser.add_argument('--latent_size', type=int, default=32, help='VAE / GAN latent dimension')
    parser.add_argument('--gen_lr', type=float, default=1e-3)
    parser.add_argument('--gen_channels', type=int, default=64, help='base channel width of the generator')

    # ---- GAN-specific
    parser.add_argument('--b1', type=float, default=0.5, help='Adam beta1 (GAN)')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam beta2 (GAN)')

    # ---- DDPM-specific
    parser.add_argument('--n_feat', type=int, default=64, help='DDPM UNet base feature width')
    parser.add_argument('--n_T', type=int, default=200, help='DDPM diffusion steps')
    parser.add_argument('--guide_w', type=float, default=0.3, help='classifier-free guidance weight at sampling')

    # ---------------------------------------------------------------- target networks
    parser.add_argument('--target_models', type=str, default='cnn_small,cnn_deep,mobilenet_lite',
                         help='comma-separated list of registered target-net architectures, length == num_models '
                              '(or a single name repeated for all, for a homogeneous ablation)')

    # ---------------------------------------------------------------- Mechanism A (frequency-weighted aggregation)
    parser.add_argument('--mechanism_a', type=int, default=0,
                         help='1: aggregate the generator\'s conditioning pathway with per-class '
                              'effective-number-of-samples weighting instead of a flat average; '
                              '0: GeFL baseline (flat average over the whole generator)')
    parser.add_argument('--mech_a_beta', type=float, default=0.999,
                         help='beta in the effective-number weight E_n = (1-beta^n)/(1-beta)')

    # ---------------------------------------------------------------- Mechanism B (fidelity-gated sampling)
    parser.add_argument('--mechanism_b', type=int, default=0,
                         help='1: draw synthetic conditioning labels from a fidelity-gated distribution that '
                              'drifts from the natural class frequency toward inverse-frequency rebalancing as '
                              'the generator becomes trustworthy on each class; 0: GeFL baseline (uniform p(y))')
    parser.add_argument('--mech_b_ema_decay', type=float, default=0.6,
                         help='EMA decay for the per-class fidelity signal (higher = slower to update)')
    parser.add_argument('--mech_b_init_fidelity', type=float, default=0.15,
                         help='initial (cautious) fidelity value before any measurement')

    # ---------------------------------------------------------------- CReFF baseline
    parser.add_argument('--creff_feat_dim', type=int, default=128, help='shared backbone feature dimension')
    parser.add_argument('--creff_rounds', type=int, default=15, help='Phase-1 FedAvg rounds')
    parser.add_argument('--creff_synth_per_class', type=int, default=100,
                         help='synthetic feature vectors per class in Phase 2 (class-balanced)')
    parser.add_argument('--creff_head_retrain_steps', type=int, default=300)

    # ---------------------------------------------------------------- evaluation
    parser.add_argument('--sample_test', type=int, default=1, help='evaluate every N rounds')
    parser.add_argument('--eval_centralized_upper_bound', type=int, default=1)
    parser.add_argument('--centralized_epochs', type=int, default=10)

    # ---------------------------------------------------------------- logging / checkpointing
    parser.add_argument('--name', type=str, default='run', help='experiment name (used in output paths)')
    parser.add_argument('--out_dir', type=str, default='./logs')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint')
    parser.add_argument('--save_ckpt', type=int, default=1)
    parser.add_argument('--wandb', type=int, default=0, help='1: also log to Weights & Biases if installed')
    parser.add_argument('--wandb_proj_name', type=str, default='gefl-classbalanced')
    parser.add_argument('--verbose', type=int, default=1)

    return parser


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def parse_args(argv=None) -> argparse.Namespace:
    """
    Two-pass parse so a --config YAML can supply new *defaults* while CLI
    flags still take final precedence, e.g.:

        python GeFL_CVAE.py --config configs/cifar10_lt.yaml --seed 3

    uses every value in cifar10_lt.yaml except seed, which the explicit
    CLI flag overrides.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)

    parser = argparse.ArgumentParser(description='Class-Balanced GeFL')
    parser = _add_all_arguments(parser)

    if pre_args.config is not None:
        with open(pre_args.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in parser._actions}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown key(s) in {pre_args.config}: {sorted(unknown)}")
        parser.set_defaults(**cfg)

    args = parser.parse_args(argv)

    # ---- derived / validated fields -------------------------------------
    target_list = [m.strip() for m in args.target_models.split(',') if m.strip()]
    if len(target_list) == 1:
        target_list = target_list * args.num_models
    if len(target_list) != args.num_models:
        raise ValueError(
            f"--target_models has {len(target_list)} entries but --num_models={args.num_models}; "
            f"pass exactly num_models comma-separated names, or a single name to repeat."
        )
    args.target_models_list = target_list

    if args.imbalance_factor <= 0 or args.imbalance_factor > 1:
        raise ValueError("--imbalance_factor must be in (0, 1] (1.0 = balanced)")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    return args


def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)


if __name__ == '__main__':
    a = parse_args()
    print(json.dumps(vars(a), indent=2, default=str))
