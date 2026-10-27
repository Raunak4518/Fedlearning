"""Run-time estimates of the notebooks and their split into parts that fit one Kaggle session (repository tool, used
by build_notebooks.py; nothing here runs on Kaggle).

Every cost is calibrated on NB00_acceptance (Kaggle T4 x2, two runs per GPU, 24 Oct 2026; results/NB00_acceptance):
a stage's time is a + b x (training images x rounds of the stage), fitted through the MNIST and CIFAR-10 runs. A
notebook's time is simulated: 4 workers take runs from the queue in order; a run first needs its seed's Stage I and
its generator, which the first worker to need them trains while later ones wait (lock files), then its own head stage.
"""
import math

# (intercept s, slope s per image-round) from NB00: CIFAR-10 (18,485 images) and MNIST (5,996 images), seed 3
STAGE1 = (49.0, 4.15e-4)          # 433 s at 18,485 x 50; 99 s at 5,996 x 20
GEN_CVAE = (320.0, 1.579e-3)      # GeFL-F: 6,161 s at 18,485 x 200; 1,267 s at 5,996 x 100
GEN_FEDECS = (556.0, 1.009e-3)    # FedECS-LT: 4,288 s; 1,161 s
HEAD = (147.0, 1.70e-4)           # GeFL-F Stage III: 461 s at 18,485 x 100; 198 s at 5,996 x 50
HEAD_FEDECS = (531.0, 8.88e-4)    # FedECS-LT Stage III (consensus, readout, secure sums): 2,174 s; 797 s
HTFL = (60.0, 3.81e-3)            # FedTGP: 7,040 s at 18,485 x 100 (HtFLlib batch 10)
# relative cost of each method's own stage (measured: GeFL-F, FedECS-LT, FedTGP; the others are estimates from what
# each adds: FedDF a server distillation, FedGen a generator, FedMRL a second model)
HEAD_FACTOR = {"FedDF": 2.0, "FedDF + LA": 2.0, "Local": 1.5, "GeFL-F w/ Gaussian generator + LA": 1.2}
HTFL_FACTOR = {"FedProto": 1.0, "FedTGP": 1.0, "FedGen": 1.2, "FedGH": 0.9, "LG-FedAvg": 0.8, "FedMRL": 1.6}
# Track 2 (one shared ResNet-8): FedECS-LT 7,409 s and LongTailLib FedAvg 12,018 s measured on CIFAR-10-LT IF 100;
# the other LongTailLib methods relative to FedAvg (estimates from what each adds per round)
T2_FEDECS = 7409.0
T2_FEDAVG = 12018.0
LT_FACTOR = {"FedAvg": 1.0, "FedProx": 1.1, "MOON": 1.6, "FedLC": 1.0, "CCVR": 1.1, "CReFF": 1.4, "FedETF": 1.0,
             "Fed-GraB": 1.3, "FedLoGe": 1.3, "FedYoYo": 2.0, "RUCR": 1.3, "CLIP2FL": 2.5}
WORKERS = 4
BUDGET_H = 9.5                    # a part's estimate must stay below this (no new run starts after 11 h)
SETUP_S = 25 * 60                 # packages, data, report and packing

PIPE_GEN = {"FedECS-LT": "fedecs", "GeFL-F": "cvae", "FedDF": "cvae", "FedDF + LA": "cvae",
            "GeFL-F + LA": "cvae_adamw",
            # component ablation: four variants share FedECS-LT's generator, two train class-row generators
            "FedECS-LT w/o moment correction": "fedecs", "FedECS-LT w/o consensus soft labels": "fedecs",
            "FedECS-LT w/o logit adjustment": "fedecs", "FedECS-LT w/ one synthetic epoch": "fedecs",
            "FedECS-LT w/ class rows (CCWA)": "rows_ccwa", "FedECS-LT w/ class rows (flat averaging)": "rows_flat"}
HTFL_METHODS = tuple(HTFL_FACTOR)


def _lin(c, x):
    return c[0] + c[1] * x


def n_train(ds):
    """Training images of the main setting (as data.make_partition: frac of the training set, tier 2 capped)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
    from settings import DATASETS
    known = {"mnist": 5996, "fmnist": 6000, "svhn": 7300, "cifar10": 18485, "cifar100": 11000, "dermamnist": 6231,
             "organamnist": 19994, "bloodmnist": 10359, "pathmnist": 19997, "gtsrb": 19984, "emnist": 19970,
             "fer2013": 19997}
    return known.get(ds, 20000), DATASETS[ds]


def n_track2(ds, IF):
    """Training images of a Track-2 split (track2.lt_partition: 5,000 images per class in CIFAR-10, 500 in CIFAR-100,
    validation held out, then n_c = n_max IF^(c / (C - 1)))."""
    C = 100 if ds.startswith("cifar100") else 10
    n_max = 50000 // C - min(200, 2000 // C)
    return sum(int(n_max * IF ** (c / (C - 1))) for c in range(C))


def run_parts(r):
    """(shared stages as (key, seconds), own seconds) of one run."""
    m, ds, seed = r["label"], r["ds"], r["seed"]
    if ds.endswith("_t2"):   # measured on CIFAR-10-LT IF 100; scaled by the split's training images
        scale = n_track2(ds, r["IF"]) / n_track2("cifar10_t2", 0.01)
        if m.endswith("(LongTailLib)"):
            return [], scale * T2_FEDAVG * LT_FACTOR[m[:-len(" (LongTailLib)")]]
        extra = 3101.0 if (r.get("rounds") or {}).get("T_FE") == 200 else 0.0   # Stage I: 3,101 s per 100 rounds
        return [], scale * (T2_FEDECS + extra)
    N, spec = n_train(ds)
    shared = [(("s1", ds, seed), _lin(STAGE1, N * spec["T_FE"]))]
    if m in HTFL_FACTOR:
        return shared, HTFL_FACTOR[m] * _lin(HTFL, N * spec["T_TN"])
    g = PIPE_GEN.get(m)
    if g == "fedecs":
        shared.append((("gen", g, ds, seed), _lin(GEN_FEDECS, N * spec["T_KA"])))
        return shared, _lin(HEAD_FEDECS, N * spec["T_TN"])
    if g:
        shared.append((("gen", g, ds, seed), _lin(GEN_CVAE, N * spec["T_KA"])))
    if m.startswith("FedECS-LT"):   # variants keep FedECS-LT's head stage (consensus, readout, secure sums)
        return shared, _lin(HEAD_FEDECS, N * spec["T_TN"])
    return shared, HEAD_FACTOR.get(m, 1.0) * _lin(HEAD, N * spec["T_TN"])


ALONE_SPEEDUP = 0.75   # assumed (not measured): a run alone on its GPU takes 75% of its time when sharing it


def simulate(runs, workers=WORKERS, alone=False):
    """Wall-clock seconds of a run list on `workers` workers sharing cached stages (list scheduling in queue order)."""
    free = [0.0] * workers
    done_at = {}           # shared stage -> time it is available
    for r in runs:
        w = min(range(workers), key=lambda i: free[i])
        t = free[w]
        shared, own = run_parts(r)
        if alone:
            shared, own = [(k, c * ALONE_SPEEDUP) for k, c in shared], own * ALONE_SPEEDUP
        for key, cost in shared:
            if key in done_at:
                t = max(t, done_at[key])     # trained (or being trained) by another worker: wait for it
            else:
                t += cost
                done_at[key] = t
        free[w] = t + own
    return max(free) + SETUP_S


def hours(runs, per_gpu=2):
    return simulate(runs, workers=2 * per_gpu, alone=per_gpu == 1) / 3600.0


def split_track1(runs, methods_pipe, methods_htfl, seeds):
    """Parts of a main-comparison notebook: all methods if they fit; else the pipeline methods and the HtFLlib
    methods apart; a part that still does not fit is split by method halves, then by seeds."""
    def sel(ms, ss):
        return [r for r in runs if r["label"] in ms and r["seed"] in ss]

    def fit(ms, ss):
        part = sel(ms, ss)
        if not part:
            return []
        if hours(part) <= BUDGET_H:
            return [part]
        if len(ms) > 1:
            h = math.ceil(len(ms) / 2)
            return fit(ms[:h], ss) + fit(ms[h:], ss)
        if len(ss) > 1:
            h = math.ceil(len(ss) / 2)
            return fit(ms, ss[:h]) + fit(ms, ss[h:])
        return [part]      # one method and one seed: cannot be split further
    whole = sel(methods_pipe + methods_htfl, seeds)
    if hours(whole) <= BUDGET_H:
        return [whole]
    return fit(list(methods_pipe), seeds) + fit(list(methods_htfl), seeds)


def split_track2(runs):
    """Parts of a Track-2 notebook: runs dealt to the fewest parts that fit, longest first (each part then has a
    balanced load). Returns (parts, long runs): a run that alone exceeds the budget is returned apart, for a notebook
    that gives it a GPU of its own."""
    long_ = [r for r in runs if hours([r]) > BUDGET_H]
    runs = [r for r in runs if hours([r]) <= BUDGET_H]
    for k in range(1, len(runs) + 1):
        parts = [[] for _ in range(k)]
        for r in sorted(runs, key=lambda r: -run_parts(r)[1]):
            min(parts, key=lambda p: sum(run_parts(x)[1] for x in p)).append(r)
        if all(hours(p) <= BUDGET_H for p in parts):
            return parts, long_   # longest run first in each part's queue (best for the workers)
    return [[r] for r in runs], long_
