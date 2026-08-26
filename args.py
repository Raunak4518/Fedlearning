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
    parser.add_argument('--data_fraction', type=float, default=1.0,
                         help='fraction of the full dataset to retain BEFORE applying imbalance_factor; '
                              '(paper Table XIV: 0.5 for CIFAR10). 1.0 = use all data')
    parser.add_argument('--partition_seed', type=int, default=0, help='seed for the client partition only')

    # ---------------------------------------------------------------- optimizer / training
    parser.add_argument('--bs', type=int, default=128, help='evaluation batch size')
    parser.add_argument('--local_bs', type=int, default=64,
                         help='local training batch size (paper Table XIV: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                         help='target-network SGD learning rate (paper Table XIV: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])

    # ---------------------------------------------------------------- reproducibility
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_experiment', type=int, default=1, help='number of repeated runs (seed, seed+1, ...)')
    parser.add_argument('--device', type=str, default='cuda' if _cuda_available() else 'cpu')

    # ---------------------------------------------------------------- GeFL / GeFL-F federated rounds
    parser.add_argument('--gen_wu_epochs', type=int, default=5,
                         help='generator warm-up rounds (paper: T_KA/2 = 100 for CIFAR10)')
    parser.add_argument('--epochs', type=int, default=100,
                         help='main (target-net + interleaved generator) communication rounds '
                              '(paper: T_TN = 100 for CIFAR10)')
    parser.add_argument('--local_ep', type=int, default=1,
                         help='(DEPRECATED — use target_ts/target_tr) legacy local epochs for '
                              'target-network training; only used if target_tr is not set')
    parser.add_argument('--gen_local_ep', type=int, default=5,
                         help='local epochs for generator training (paper Table XIV: T_g = 5)')
    parser.add_argument('--aid_by_gen', type=int, default=1,
                         help='1: augment local target-net training with synthetic samples from the shared '
                              'generator (GeFL). 0: plain FedAvg baseline with no generator at all')
    parser.add_argument('--freeze_gen', type=int, default=0,
                         help='1: stop updating the generator after warm-up; 0: keep training it every round')
    parser.add_argument('--synth_batch', type=int, default=64,
                         help='(DEPRECATED — sequential Ts/Tr uses local_bs) number of synthetic '
                              'samples drawn per client per round')

    # ---- sequential Ts/Tr target training (paper Algorithm 1, Table XIV)
    parser.add_argument('--target_ts', type=int, default=1,
                         help='synthetic-only local epochs during target-net training '
                              '(paper Table XIV: T_s = 1)')
    parser.add_argument('--target_tr', type=int, default=5,
                         help='real-only local epochs during target-net training '
                              '(paper Table XIV: T_r = 5)')

    # ---------------------------------------------------------------- generator choice + shared params
    parser.add_argument('--gen_model', type=str, default='vae', choices=['vae', 'gan', 'ddpm'],
                         help='which registered conditional generator architecture to use')
    parser.add_argument('--latent_size', type=int, default=32,
                         help='VAE / GAN latent dimension (paper: CVAE l = 50)')
    parser.add_argument('--gen_lr', type=float, default=1e-3,
                         help='generator learning rate for CVAE (paper Table XV: 1e-3)')
    parser.add_argument('--gen_lr_gan', type=float, default=2e-4,
                         help='DCGAN generator/discriminator LR (paper Table XV: 2e-4)')
    parser.add_argument('--gen_lr_ddpm', type=float, default=1e-4,
                         help='DDPM learning rate (paper Table XV: 1e-4)')
    parser.add_argument('--weight_decay_ddpm', type=float, default=1e-3,
                         help='DDPM weight decay (paper Table XV: 1e-3)')
    parser.add_argument('--gen_channels', type=int, default=64,
                         help='base channel width of the generator (CVAE); '
                              'for DCGAN use dcgan_g_channels/dcgan_d_channels instead')
    parser.add_argument('--dcgan_g_channels', type=int, default=128,
                         help='DCGAN generator base channel width (paper Table XV: d_g = 256)')
    parser.add_argument('--dcgan_d_channels', type=int, default=128,
                         help='DCGAN discriminator base channel width (paper Table XV: d_d = 64)')

    # ---- GAN-specific
    parser.add_argument('--b1', type=float, default=0.5, help='Adam beta1 (GAN)')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam beta2 (GAN)')

    # ---- DDPM-specific
    parser.add_argument('--n_feat', type=int, default=64,
                         help='DDPM UNet base feature width (paper Table XV: 128)')
    parser.add_argument('--n_T', type=int, default=200,
                         help='DDPM diffusion timesteps (paper Table XV: 400)')
    parser.add_argument('--guide_w', type=float, default=0.3,
                         help='classifier-free guidance weight at sampling '
                              '(paper: 0 or 2 tested, not 0.3)')

    # ---------------------------------------------------------------- target networks
    parser.add_argument('--target_models', type=str, default='cnn_small,cnn_deep,mobilenet_lite',
                         help='comma-separated list of registered target-net architectures, length == num_models '
                              '(or a single name repeated for all, for a homogeneous ablation)')

    # ---------------------------------------------------------------- GeFL-F (feature extractor variant)
    parser.add_argument('--gefl_f', type=int, default=0,
                         help='1: run GeFL-F (feature extractor + heterogeneous headers) '
                              'instead of plain GeFL')
    parser.add_argument('--fe_channels', type=int, default=32,
                         help='feature extractor output channels (paper unspecified, our choice: 32)')
    parser.add_argument('--fe_rounds', type=int, default=50,
                         help='stage (i) FE warm-up rounds (paper Table XIV: T_FE = 50)')
    parser.add_argument('--header_models', type=str, default='header_small,header_deep,header_wide',
                         help='comma-separated list of GeFL-F header architectures')

    # ---------------------------------------------------------------- Mechanism A (frequency-weighted aggregation)
    parser.add_argument('--mechanism_a', type=int, default=0,
                         help='1: aggregate the generator\'s conditioning pathway with per-class '
                              'effective-number-of-samples weighting instead of a flat average; '
                              '0: GeFL baseline (flat average over the whole generator)')
    parser.add_argument('--mech_a_beta', type=float, default=0.999,
                         help='beta in the effective-number weight E_n = (1-beta^n)/(1-beta)')
    parser.add_argument('--mech_a_support_floor', type=int, default=0,
                         help='min total count of a class across selected clients before Mech A '
                              'trusts frequency-weighted aggregation for that class conditioning row; '
                              'below this, fall back to flat mean (paper baseline). 0 disables. '
                              'Recommended: 20 for CIFAR-10-LT to protect tail-class rows from noise')

    # ---------------------------------------------------------------- Mechanism B (fidelity-gated sampling)
    parser.add_argument('--mechanism_b', type=int, default=0,
                         help='1: draw synthetic conditioning labels from a fidelity-gated distribution that '
                              'drifts from the natural class frequency toward inverse-frequency rebalancing as '
                              'the generator becomes trustworthy on each class; 0: GeFL baseline (uniform p(y))')
    parser.add_argument('--mech_b_ema_decay', type=float, default=0.6,
                         help='EMA decay for the per-class fidelity signal (higher = slower to update)')
    parser.add_argument('--mech_b_init_fidelity', type=float, default=0.15,
                        help='initial (cautious) fidelity value before any measurement')
    parser.add_argument('--mech_b_inv_freq_power', type=float, default=1.0,
                        help='exponent for the inverse frequency weighting (e.g. 0.5 for softened inverse)')

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
    parser.add_argument('--tensorboard', type=int, default=0,
                         help='1: log scalars and images to TensorBoard')
    parser.add_argument('--log_per_class', type=int, default=1,
                         help='1: log per-class accuracy each eval step (not just buckets)')
    parser.add_argument('--log_per_client', type=int, default=0,
                         help='1: evaluate and log each client model individually')
    parser.add_argument('--plot_every', type=int, default=0,
                         help='generate intermediate plots every N rounds; 0 = final only')
    parser.add_argument('--save_synthetic_samples', type=int, default=1,
                         help='1: save a grid of generator output at the end of training')
    parser.add_argument('--show_plots', type=int, default=0,
                         help='1: auto-display plots at end of training (blocks on headless servers)')

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
