"""Pre-registered analysis of K17 (written 16 Oct 2026, before any K17 result existed).

    python k17_analysis.py      -> docs/paper_assets/tables/T22_fair_retest.md

Selection uses validation scores only (val_*): Ours' T_s per dataset by mean val_er_bal; every method's readout by
mean val_<readout>_bal over the federated readouts. The selected configuration is then reported on the TEST set,
paired over seeds 3-5, with Holm's correction across the baselines within each dataset. Also checks P2-P4.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "paper_assets", "tables", "T22_fair_retest.md"))
READOUTS = ["final", "bbc", "er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"]
BASELINES = ["GeFL-F", "GeFL-F + AdamW + LA", "Real only", "Real only + LA", "Real only + FedLC", "Real only + FedRS"]
DATASETS = ["mnist", "fmnist", "svhn", "cifar10", "cifar100"]
RULE_TS = {"mnist": 10, "fmnist": 10, "svhn": 1, "cifar10": 1, "cifar100": 1}


def load():
    D = defaultdict(dict)  # (dataset, label) -> seed -> record
    for f in glob.glob(os.path.join(HERE, "results", "K17_kaggle_fair_retest*", "runs.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if not r.get("quick"):
                D[(r["dataset"], r["label"])][r["seed"]] = r
    return D


def mean_of(runs, key):
    xs = [r[key] for r in runs.values() if isinstance(r.get(key), (int, float))]
    return float(np.mean(xs)) if xs else float("nan")


def pick_readout(runs):
    scores = {ro: mean_of(runs, f"val_{ro}_bal") for ro in READOUTS}
    scores = {k: v for k, v in scores.items() if v == v}
    return max(scores, key=scores.get)


def holm(ps):
    order = np.argsort(ps)
    adj, m, run = [0.0] * len(ps), len(ps), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[j]))
        adj[j] = run
    return adj


def main():
    D = load()
    lines = ["### T22_fair_retest: validation-selected comparison on new seeds 3-5 (K17, long tail)", "",
             "Selection on validation only (pre-registered). Test-set balanced accuracy of each method's selected "
             "readout; difference = Ours - baseline, paired over seeds, 95% CI, Holm-adjusted p.", ""]
    p2, p3 = [], None
    for ds in DATASETS:
        o1, o10 = D.get((ds, "Ours Ts=1")), D.get((ds, "Ours Ts=10"))
        if not o1 or not o10:
            lines.append(f"**{ds}**: not run yet.\n")
            continue
        ts = 1 if mean_of(o1, "val_er_bal") >= mean_of(o10, "val_er_bal") else 10
        p2.append(ts == RULE_TS[ds])
        ours = o1 if ts == 1 else o10
        ro_o = pick_readout(ours)
        lines += [f"**{ds}**: selected T_s = {ts} (rule: {RULE_TS[ds]}); Ours readout = {ro_o}; "
                  f"Ours test = {100 * mean_of(ours, ro_o + '_bal'):.2f}", "",
                  "| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |", "|---|---|---|---|---|"]
        rows, ps = [], []
        for b in BASELINES:
            runs = D.get((ds, b))
            if not runs:
                continue
            ro = pick_readout(runs)
            seeds = sorted(set(runs) & set(ours))
            d = np.array([100 * (ours[s][ro_o + "_bal"] - runs[s][ro + "_bal"]) for s in seeds])
            if len(d) >= 2:
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = stats.t.ppf(0.975, len(d) - 1)
                ci = (d.mean() - t * se, d.mean() + t * se)
                p = float(stats.ttest_1samp(d, 0).pvalue) if d.std(ddof=1) > 0 else 0.0
            else:
                ci, p = (float("nan"), float("nan")), float("nan")
            rows.append((b, ro, 100 * mean_of(runs, ro + "_bal"), d.mean(), ci))
            ps.append(p)
        adj = holm(ps) if ps else []
        for (b, ro, v, dm, ci), pa in zip(rows, adj):
            lines.append(f"| {b} | {ro} | {v:.2f} | {dm:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] | {pa:.3g} |")
        lines.append("")
        if ds == "cifar10" and D.get((ds, "Ours rows Ts=1")):
            rw = D[(ds, "Ours rows Ts=1")]
            ro_r = pick_readout(rw)
            p3 = 100 * (mean_of(o1, ro_o + "_bal") - mean_of(rw, ro_r + "_bal"))
            lines.append(f"CIFAR-10 gate check (P3): anchor branch minus forced row branch = {p3:+.2f} points.\n")
    # P4: label-free readout-gap estimate
    xs, ys = [], []
    for runs in D.values():
        for r in runs.values():
            if isinstance(r.get("ro_disagree"), (int, float)):
                xs.append(r["ro_disagree"])
                ys.append(r["er_bal"] - r["final_bal"])
    if len(xs) >= 5:
        rho, pv = stats.spearmanr(xs, ys)
        lines.append(f"P4 (label-free gap): Spearman(ro_disagree, er_bal - final_bal) = {rho:.2f} (p = {pv:.2g}, "
                     f"n = {len(xs)} runs); prediction >= 0.7.")
    if p2:
        lines.append(f"P2 (validation picks the same T_s as the old rule): {sum(p2)} of {len(p2)} datasets; prediction >= 4 of 5.")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
