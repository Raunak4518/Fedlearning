"""Fixed settings of the FedECS-LT experiment suite: datasets, rounds, seeds and the method registry.

Values for the five tier-1 datasets are copied from experiments/core.py (DATASETS, HEADER_WIDTHS), which produced every
earlier result, so new runs pair with old ones (same partitions for the same seeds). Every method carries its citation
and the source of its code ("ours", "paste": licence verified, code included with credit, "fetch": no licence, the
authors' repository is downloaded at run time at a pinned commit).
"""

SEEDS = [3, 4, 5]
EVAL_EVERY = 5            # rounds between evaluations (pre- and post-aggregation)
DEADLINE_H = 11.0         # a notebook starts no new run after this many hours (Kaggle limit 12 h)

# ---------------------------------------------------------------- datasets
# in_ch, normalisation, BN in the heads, FE variant, fraction of the training set used, batch size, rounds per stage
# (T_FE extractor, T_KA generator, T_TN heads), generator latent size, augmentation, number of classes, imbalance:
#   "IF": long tail made by subsampling (imbalance factor from the run), "natural": the dataset's own class sizes.
DATASETS = {
    "mnist": dict(name="MNIST", in_ch=1, norm="none", header_bn=False, fe="mnist", frac=0.1, B=64,
                  T_FE=20, T_KA=100, T_TN=50, latent=16, augment=False, C=10, imbalance="IF", tier=1),
    "fmnist": dict(name="Fashion-MNIST", in_ch=1, norm="half", header_bn=True, fe="mnist", frac=0.1, B=64,
                   T_FE=20, T_KA=100, T_TN=50, latent=16, augment=False, C=10, imbalance="IF", tier=1),
    "svhn": dict(name="SVHN", in_ch=3, norm=((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614)),
                 header_bn=True, fe="mnist", frac=0.1, B=64, T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False,
                 C=10, imbalance="IF", tier=1),
    "cifar10": dict(name="CIFAR-10", in_ch=3, norm=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    header_bn=False, fe="cifar10", frac=0.5, B=128, T_FE=50, T_KA=200, T_TN=100, latent=50,
                    augment=True, C=10, imbalance="IF", tier=1),
    "cifar100": dict(name="CIFAR-100", in_ch=3, norm=((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
                     header_bn=False, fe="cifar10", frac=0.22, B=128, T_FE=50, T_KA=200, T_TN=100, latent=50,
                     augment=True, C=100, imbalance="IF", tier=1),
    # Tier 2: natural imbalance, training sets capped at 20,000 images (frac = min(1, 20000 / N)); 28x28 images are
    # resized to 32x32 as MNIST is; grey images use MNIST's extractor and rounds, colour images CIFAR-10's.
    "dermamnist": dict(name="DermaMNIST", in_ch=3, norm="half", header_bn=True, fe="cifar10", frac=None, B=64,
                       T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False, C=7, imbalance="natural", tier=2),
    "organamnist": dict(name="OrganAMNIST", in_ch=1, norm="half", header_bn=True, fe="mnist", frac=None, B=64,
                        T_FE=20, T_KA=100, T_TN=50, latent=16, augment=False, C=11, imbalance="natural", tier=2),
    "bloodmnist": dict(name="BloodMNIST", in_ch=3, norm="half", header_bn=True, fe="cifar10", frac=None, B=64,
                       T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False, C=8, imbalance="natural", tier=2),
    "pathmnist": dict(name="PathMNIST", in_ch=3, norm="half", header_bn=True, fe="cifar10", frac=None, B=64,
                      T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False, C=9, imbalance="natural", tier=2),
    "gtsrb": dict(name="GTSRB", in_ch=3, norm="half", header_bn=True, fe="cifar10", frac=None, B=64,
                  T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False, C=43, imbalance="natural", tier=2),
    "emnist": dict(name="EMNIST (ByClass)", in_ch=1, norm="half", header_bn=True, fe="mnist", frac=None, B=64,
                   T_FE=20, T_KA=100, T_TN=50, latent=16, augment=False, C=62, imbalance="natural", tier=2),
    "fer2013": dict(name="FER-2013", in_ch=1, norm="half", header_bn=True, fe="mnist", frac=None, B=64,
                    T_FE=50, T_KA=100, T_TN=100, latent=16, augment=False, C=7, imbalance="natural", tier=2),
    # Track 2 (one shared model, FedLTLib / CReFF protocol): standard CIFAR-LT profile, 20 clients, 40% per round,
    # ResNet-8 split after its second stage (frozen extractor) for FedECS-LT; SGD 0.1, momentum 0.9, learning-rate
    # decay 0.99 per round, 10 local epochs, batch 32, as the baselines in LongTailLib. T_FE + T_TN = 200 rounds, the
    # baselines' total; the generator's rounds (T_KA) are extra and reported.
    "cifar10_t2": dict(name="CIFAR-10", raw="cifar10", in_ch=3, norm=((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                       header_bn=True, fe="resnet8", heads="resnet8", frac=None, B=32, T_FE=100, T_KA=100, T_TN=100,
                       latent=50, augment=True, C=10, imbalance="IF", partition="lt_standard", single_arch=True,
                       track=2, local_epochs=10, opt=dict(lr=0.1, momentum=0.9, decay=0.99), tier=1),
    "cifar100_t2": dict(name="CIFAR-100", raw="cifar100", in_ch=3, norm=((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                        header_bn=True, fe="resnet8", heads="resnet8", frac=None, B=32, T_FE=100, T_KA=100, T_TN=100,
                        latent=50, augment=True, C=100, imbalance="IF", partition="lt_standard", single_arch=True,
                        track=2, local_epochs=10, opt=dict(lr=0.1, momentum=0.9, decay=0.99), tier=1),
}
CAP_IMAGES = 20000
HEADER_WIDTHS = [(16,), (16, 32), (20, 40), (10, 20), (16, 32, 64), (20, 40, 80),
                 (10, 20, 40), (16, 32, 64, 128), (20, 40, 80, 100), (10, 20, 40, 80)]
N_ARCH = 10


def class_groups(C):
    """Head / medium / tail classes by frequency rank (first 30%, next 30%, last 40%), as in all earlier results."""
    a, b = int(round(0.3 * C)), int(round(0.6 * C))
    return slice(0, a), slice(a, b), slice(b, C)


# ---------------------------------------------------------------- our method's settings (action plan section 12)
FEDECS = dict(
    gate_F=0.05,          # F below this: class-row generator with class-count weighted averaging (CCWA)
    ccwa_beta=0.999,      # E(n) = (1 - beta^n) / (1 - beta)
    mc_samples=500,       # generator samples per class to measure its moments (moment correction)
    mc_clip=(0.1, 10.0),  # bounds of the spread ratio
    csl_beta=0.5,         # consensus soft label = (1 - beta) one-hot + beta consensus
    la_tau=1.0,           # logit adjustment strength
    ts=10,                # synthetic epochs per round (T_s), all datasets
    tr=5,                 # real epochs per round
    er_gamma=0.1,         # shrinkage of the exact readout's within-class covariance
)

# Secure aggregation (Flower SecAgg+): 30-bit quantisation (Flower's default 22 bits loses precision on sums), modulus
# 2^40, every vector divided by a public bound before quantisation so that it lies inside the clipping range +-8.
SECAGG = dict(quantization_range=2 ** 30, modulus_range=2 ** 40, clipping_range=8.0, max_weight=1.0)

# ---------------------------------------------------------------- methods
# kind: "pipeline" (runs the GeFL-F three-stage pipeline with the given options) or "baseline" (an adapter in
# baselines/). Every method of the main comparison; "rank" A = must have, B = second priority, C = control.
CITE = {
    "GeFL-F": "Kang et al., IEEE TMC 2025",
    "FedProto": "Tan et al., AAAI 2022", "FedTGP": "Zhang et al., AAAI 2024", "FedGen": "Zhu et al., ICML 2021",
    "FedDF": "Lin et al., NeurIPS 2020", "CReFF": "Shang et al., IJCAI 2022", "CCVR": "Luo et al., NeurIPS 2021",
    "FedETF": "Li et al., ICCV 2023", "FedLoGe": "Xiao et al., ICLR 2024", "Fed-GraB": "Xiao et al., NeurIPS 2023",
    "FedYoYo": "Yan et al., ICCV 2025", "FedLC": "Zhang et al., ICML 2022", "FedRS": "Li and Zhan, KDD 2021",
    "RUCR": "Huang et al., IEEE TIFS 2024", "FedGH": "Yi et al., ACM MM 2023", "LG-FedAvg": "Liang et al., 2020",
    "FedNH": "Dai et al., AAAI 2023", "FedVLS": "Guo et al., AAAI 2025", "FedMRL": "Yi et al., NeurIPS 2024",
    "FedAvg": "McMahan et al., AISTATS 2017", "FedProx": "Li et al., MLSys 2020", "MOON": "Li et al., CVPR 2021",
    "CLIP2FL": "Shi et al., AAAI 2024", "Balanced Softmax": "Ren et al., NeurIPS 2020",
    "LA": "Menon et al., ICLR 2021", "FedALA": "Zhang et al., AAAI 2023", "AvgKD": "Afonin and Karimireddy, ICLR 2022",
}
