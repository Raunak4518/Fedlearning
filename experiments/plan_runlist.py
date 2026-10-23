"""Run list for the collection phase (action plan sections 12-18), its time estimate and the split across helpers.

    python experiments/plan_runlist.py   -> docs/review/run_list.csv, docs/review/run_list_summary.md

A planning tool only: nothing here trains anything. Every run is one row; runs of Track 1 that share a feature
extractor and generators (same dataset, seed and setting) form a block and are assigned to the same session, so the
shared stages are trained once. Times are estimates (minutes on one T4) derived from the logged K17 stage times and
scaled for new datasets; see ASSUMPTIONS in the summary.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "docs", "review"))

SEEDS = [3, 4, 5]
HELPERS = 15
SESSION_H = 12          # Kaggle session limit
GPUS = 2                # T4 x 2
PACK = 2                # runs packed per GPU (small models)
MARGIN = 0.85           # keep 15% of every session free for overruns
OVERHEAD = 1.7          # Flower / Ray (+30%) and the full evaluation and logging of sections 14-17 (+30%)

# ---------------------------------------------------------------- datasets (section 18)
# stage minutes on one T4 for the main setting: extractor, one generator, one baseline head stage.
# Tier-1 values: logged K17 stage times (fe_time_s, gen_time_s, head_time_s of GeFL-F), rounded up.
# New datasets: training sets capped at 20,000 images by uniform subsampling (natural imbalance kept), images at
# 28x28 (grey) or 32x32 (colour); times scaled from the closest tier-1 dataset.
DATASETS = {
    # name: (tier, classes, fe_min, gen_min, head_min, imbalance)
    "mnist": (1, 10, 0.5, 8, 0.7, "IF"),
    "fmnist": (1, 10, 0.6, 8, 0.7, "IF"),
    "svhn": (1, 10, 1.6, 10, 1.6, "IF"),
    "cifar10": (1, 10, 2.2, 50, 2.6, "IF"),
    "cifar100": (1, 100, 1.4, 31, 1.5, "IF"),
    "dermamnist": (2, 7, 0.6, 10, 0.8, "natural"),
    "organamnist": (2, 11, 1.0, 16, 1.0, "natural"),
    "bloodmnist": (2, 8, 0.8, 14, 0.9, "natural"),
    "pathmnist": (2, 9, 1.2, 30, 1.5, "natural"),
    "gtsrb": (2, 43, 1.5, 30, 1.8, "natural"),
    "emnist": (2, 62, 1.0, 16, 1.0, "natural"),
    "fer2013": (2, 7, 1.2, 25, 1.4, "natural"),
    "fedisic2019": (3, 8, 2.5, 50, 3.0, "natural, 6 real hospitals"),
    "femnist": (3, 62, 2.0, 40, 2.0, "natural, writers as clients"),
    "tinyimagenet": (3, 200, 3.0, 70, 3.5, "IF"),
    "cinic10": (3, 10, 2.5, 50, 3.0, "IF"),
    "flowers102": (3, 102, 1.0, 15, 1.2, "natural"),
    "ham10000": (3, 7, 1.5, 30, 1.8, "natural"),
    "aptos2019": (3, 5, 1.0, 15, 1.2, "natural"),
    "fedheart": (4, 2, 0.2, 2, 0.2, "natural, 4 real hospitals (tabular)"),
    "officecaltech10": (4, 10, 1.0, 15, 1.2, "domain shift"),
    "agnews": (4, 4, 2.0, 30, 2.0, "IF (text)"),
    "speechcommands": (4, 35, 3.0, 40, 3.0, "natural (audio)"),
}

# ---------------------------------------------------------------- methods (sections 13, 16)
# name: (priority group, head-stage cost relative to GeFL-F's, needs which generator: None / "gefl" / "decay" / "ours")
TRACK1 = {
    "Ours": (0, 2.5, "ours"),
    "GeFL-F": (0, 1.0, "gefl"),
    "GeFL-F + AdamW + LA": (0, 1.0, "decay"),
    "Local only": (0, 0.6, None),
    "Real only": (0, 0.6, None),
    "Real only + LA (balanced softmax)": (0, 0.6, None),
    "FedLC": (0, 0.6, None),
    "FedRS": (0, 0.6, None),
    "Gaussian generator + LA": (0, 1.2, None),
    "FedDF": (0, 2.0, "gefl"),
    "FedDF + LA": (0, 2.0, "gefl"),
    "FedProto": (1, 1.2, None),
    "FedTGP": (1, 1.5, None),
    "FedGen": (1, 2.0, None),
    "FedGH": (2, 1.2, None),
    "LG-FedAvg": (2, 0.8, None),
    "FedMRL": (2, 2.0, None),
}
# Reduced on 23 Oct 2026 (action plan section 22): long-tail methods built for one shared model run in Track 2 only;
# FEDIC, FedGELA, FedLF, FML, FD, FedKD, FedKTL, FedMD, SCAFFOLD and CAPT are dropped. The notebooks actually built
# are listed by fedecs_lt/build_notebooks.py (fedecs_lt/notebooks/START_HERE.md); this file only estimates time.
# Track 2: one shared ResNet-8 (LongTailLib / FedLTLib protocol): 20 clients, 8 per round, 200 rounds, 10 local epochs,
# batch 32, SGD 0.1. Minutes per run on one T4 before OVERHEAD.
TRACK2_BASE = {"cifar10": 18, "cifar100": 18, "tinyimagenet": 55}
TRACK2 = {
    "Ours (one architecture)": (0, 2.0), "FedAvg": (0, 1.0), "FedProx": (0, 1.1), "MOON": (0, 1.6), "FedLC": (0, 1.0),
    "CCVR": (0, 1.1), "CReFF": (1, 1.4), "FedETF": (1, 1.0), "Fed-GraB": (1, 1.3), "FedLoGe": (1, 1.3),
    "FedYoYo": (1, 2.0), "RUCR": (2, 1.3), "CLIP2FL (extra resources)": (4, 2.5),
}
SWEEP_METHODS = ["Ours", "GeFL-F", "GeFL-F + AdamW + LA", "FedDF + LA", "Gaussian generator + LA", "FedTGP",
                 "FedGen", "FedProto"]   # top baselines: replaced by the five strongest once the main results exist


def main_setting(ds):
    return dict(IF=0.01 if DATASETS[ds][5] == "IF" else None, alpha=0.5, K=10, part=1.0)


def gens_needed(methods):
    return {TRACK1[m][2] for m in methods if TRACK1[m][2]}


def block_runs(phase, priority, ds, seed, setting, methods, factor=1.0):
    """One Track-1 block: shared extractor + the generators its methods need + one head stage per method."""
    tier, C, fe, gen, head, _ = DATASETS[ds]
    kfac = {10: 1.0, 50: 1.3, 100: 1.6}[setting["K"]]
    rows = []
    gens = gens_needed(methods)
    shared = (fe + gen * len(gens)) * kfac * factor * OVERHEAD
    key = f"T1|{ds}|s{seed}|IF{setting['IF']}|a{setting['alpha']}|K{setting['K']}|p{setting['part']}|{setting.get('extra', '')}"
    for i, m in enumerate(methods):
        t = head * TRACK1[m][1] * kfac * factor * OVERHEAD + (shared if i == 0 else 0.0)
        rows.append(dict(phase=phase, priority=priority, track="1", tier=tier, dataset=ds, method=m, seed=seed,
                         IF=setting["IF"], alpha=setting["alpha"], K=setting["K"], participation=setting["part"],
                         extra=setting.get("extra", ""), block=key, est_min=round(t, 1)))
    return rows


def build():
    runs = []
    t1_by_group = defaultdict(list)
    for m, (g, _, _) in TRACK1.items():
        t1_by_group[g].append(m)
    # Phase A: main setting, every Track-1 method, by dataset tier and method priority group
    for tier in (1, 2, 3, 4):
        for g in sorted(t1_by_group):
            for ds in [d for d, v in DATASETS.items() if v[0] == tier]:
                for s in SEEDS:
                    methods = t1_by_group[g] if g > 0 else t1_by_group[0]
                    if g > 0:
                        methods = methods  # these reuse the block of group 0 (extractor already cached)
                    runs += block_runs("A main setting", 10 * tier + g, ds, s, main_setting(ds), methods,
                                       factor=1.0 if g == 0 else 0.35)
    # Phase B: Track 2
    for ds, base in TRACK2_BASE.items():
        ifs = [0.1, 0.02, 0.01] if ds != "tinyimagenet" else [0.02]
        for g in sorted({v[0] for v in TRACK2.values()}):
            for IF in ifs:
                for s in SEEDS:
                    for m, (gg, c) in TRACK2.items():
                        if gg != g:
                            continue
                        runs.append(dict(phase="B track 2", priority=15 + g, track="2", tier=1 if ds != "tinyimagenet" else 3,
                                         dataset=ds, method=m, seed=s, IF=IF, alpha=0.5, K=20, participation=0.4,
                                         extra="ResNet-8", block=f"T2|{ds}|{m}|IF{IF}|s{s}",
                                         est_min=round(base * c * OVERHEAD, 1)))
    # Phase C: sweeps (one factor at a time from the main setting), tier-1 datasets, sweep methods
    variants = [dict(IF=0.1), dict(IF=0.02), dict(alpha=0.1), dict(alpha=1.0), dict(K=50), dict(K=100),
                dict(K=100, part=0.1), dict(K=100, part=0.2), dict(extra="random rare classes")]
    for v in variants:
        for ds in [d for d, x in DATASETS.items() if x[0] == 1]:
            for s in SEEDS:
                st = dict(main_setting(ds), **v)
                runs += block_runs("C sweeps", 30, ds, s, st, SWEEP_METHODS)
    # Phase D: checks
    for ds in ["mnist", "fmnist"]:
        for eps in (8, 2):
            for s in SEEDS:
                runs += block_runs("D checks", 40, ds, s, dict(main_setting(ds), extra=f"DP eps={eps}"), ["Ours"])
    for ds in ["cifar10", "cifar100"]:
        for s in SEEDS:
            runs += block_runs("D checks", 40, ds, s, dict(main_setting(ds), extra="ResNet-18 frozen extractor"),
                               SWEEP_METHODS, factor=1.5)
    for ds in ["mnist", "cifar10"]:
        for s in SEEDS:
            runs += block_runs("D checks", 40, ds, s, dict(main_setting(ds), extra="full training set"),
                               SWEEP_METHODS, factor=3.0)
    return runs


def assign(runs):
    """Blocks in priority order, filled session by session across helpers (day 1 = everyone's first session)."""
    blocks = defaultdict(list)
    order = []
    for r in runs:
        if r["block"] not in blocks:
            order.append(r["block"])
        blocks[r["block"]].append(r)
    order.sort(key=lambda b: (min(x["priority"] for x in blocks[b]), -sum(x["est_min"] for x in blocks[b])))
    cap = SESSION_H * 60 * GPUS * PACK * MARGIN
    load = defaultdict(float)
    session, helper = 1, 1
    for b in order:
        t = sum(x["est_min"] for x in blocks[b])
        placed = False
        while not placed:
            for h in range(1, HELPERS + 1):
                if load[(session, h)] + t <= cap or load[(session, h)] == 0:
                    load[(session, h)] += t
                    for x in blocks[b]:
                        x["session"], x["helper"] = session, h
                    placed = True
                    break
            if not placed:
                session += 1
    return load


def main():
    runs = build()
    load = assign(runs)
    for i, r in enumerate(runs):
        r["run_no"] = i + 1
    cols = ["run_no", "phase", "priority", "track", "tier", "dataset", "method", "seed", "IF", "alpha", "K",
            "participation", "extra", "block", "est_min", "session", "helper"]
    with open(os.path.join(OUT_DIR, "run_list.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(runs, key=lambda r: (r["session"], r["helper"], r["priority"])))
    tot_h = sum(r["est_min"] for r in runs) / 60
    sessions = max(r["session"] for r in runs)
    L = ["# Collection-phase run list: summary", "",
         "Generated by `experiments/plan_runlist.py` (planning only; times are estimates). Full list: `run_list.csv`.", "",
         f"- Runs: **{len(runs)}**; estimated single-GPU time **{tot_h:.0f} h**.",
         f"- Helpers: {HELPERS}; each session {SESSION_H} h on T4 x {GPUS}, {PACK} runs per GPU, {int((1 - MARGIN) * 100)}% "
         f"margin -> {SESSION_H * GPUS * PACK * MARGIN:.0f} run-hours per session.",
         f"- Sessions needed per helper (one per day): **{sessions}**.", "",
         "| phase | runs | est. GPU-h | sessions (first-last) |", "|---|---|---|---|"]
    by_phase = defaultdict(list)
    for r in runs:
        by_phase[r["phase"]].append(r)
    for p in sorted(by_phase):
        rs = by_phase[p]
        L.append(f"| {p} | {len(rs)} | {sum(x['est_min'] for x in rs) / 60:.0f} | "
                 f"{min(x['session'] for x in rs)}-{max(x['session'] for x in rs)} |")
    L += ["", "| tier (phase A) | datasets | runs | est. GPU-h |", "|---|---|---|---|"]
    for t in (1, 2, 3, 4):
        rs = [r for r in by_phase["A main setting"] if r["tier"] == t]
        L.append(f"| {t} | {len({r['dataset'] for r in rs})} | {len(rs)} | {sum(x['est_min'] for x in rs) / 60:.0f} |")
    L += ["", "| session (day) | helpers busy | run-hours used |", "|---|---|---|"]
    for s in range(1, sessions + 1):
        used = [v for (ss, h), v in load.items() if ss == s]
        L.append(f"| {s} | {len(used)} | {sum(used) / 60:.0f} |")
    L += ["", "## ASSUMPTIONS", "",
          "- Track 1 main setting: IF = 100 where the imbalance is made, natural imbalance otherwise; Dirichlet 0.5; "
          "K = 10; full participation; seeds 3-5 (pair with the old runs).",
          "- New datasets: training sets capped at 20,000 images by uniform subsampling; 28x28 grey or 32x32 colour.",
          "- Track-1 methods of priority groups 1-4 reuse the block's cached extractor (cost factor 0.35 of a fresh block).",
          "- Track 2: ResNet-8, 20 clients, 8 per round, 200 rounds, 10 local epochs (CReFF protocol; check against "
          "the LongTailLib code), IF 10 / 50 / 100 on CIFAR-10/100-LT, IF 50 on Tiny-ImageNet-LT.",
          f"- Overhead factor {OVERHEAD} for Flower / Ray and the full evaluation (pre/post aggregation, all metrics, logs).",
          f"- {PACK} runs per GPU after a benchmark; if packing fails, multiply the sessions by {PACK}.",
          "- Sweep methods are placeholders until the main results name the five strongest baselines."]
    open(os.path.join(OUT_DIR, "run_list_summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
