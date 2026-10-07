"""
experiments/report.py

Collects every experiment's runs into results/REPORT.md: a headline table
(GeFL-F vs +LA vs Ours on every setting where all three ran), then each
experiment's own summary.md, with its purpose, hypothesis and verdict data.

    python experiments/report.py [--quick]
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OURS = "Ours (HWA+LCD+PCM+LA)"


def load(root, quick):
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "*", "runs.jsonl"))):
        exp = os.path.basename(os.path.dirname(f))
        if exp.endswith("_quick") != quick:
            continue
        for line in open(f, encoding="utf-8"):
            runs[exp].append(json.loads(line))
    return runs


def ms(xs):
    xs = np.array(xs) * 100
    if len(xs) == 0:
        return "-"
    return f"{xs.mean():.2f} ± {xs.std(ddof=1):.2f}" if len(xs) > 1 else f"{xs.mean():.2f}"


def headline(runs):
    rows = []
    for exp, rs in runs.items():
        by = defaultdict(lambda: defaultdict(list))
        for r in rs:
            by[(r["dataset"], r["IF"], r["alpha"], r["K"])][r["label"]].append(r)
        for (ds, IF, a, K), labs in by.items():
            if "GeFL-F" not in labs or OURS not in labs:
                continue
            base = {r["seed"]: r for r in labs["GeFL-F"]}
            ours = {r["seed"]: r for r in labs[OURS]}
            la = {r["seed"]: r for r in labs.get("+LA", [])}
            common = sorted(set(base) & set(ours))
            d_bal = [ours[s]["final_bal"] - base[s]["final_bal"] for s in common]
            d_tail = [ours[s]["final_tail"] - base[s]["final_tail"] for s in common]
            d_la = [ours[s]["final_bal"] - la[s]["final_bal"] for s in common if s in la]
            rows.append([exp.replace("_quick", ""), ds, IF, a if a is not None else "IID", K, len(common),
                         ms([base[s]["final_bal"] for s in common]),
                         ms([la[s]["final_bal"] for s in common if s in la]),
                         ms([ours[s]["final_bal"] for s in common]),
                         f"{np.mean(d_bal) * 100:+.2f}", f"{np.mean(d_tail) * 100:+.2f}",
                         f"{np.mean(d_la) * 100:+.2f}" if d_la else "-",
                         ms([base[s]["best_mean_acc"] for s in common]), ms([ours[s]["best_mean_acc"] for s in common])])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--root", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    runs = load(a.root, a.quick)
    out = ["# GeFL-F class-imbalance study: results", "",
           f"Mode: {'QUICK viability pass (reduced rounds, 1 seed) - trends only' if a.quick else 'full runs'}",
           f"Experiments with results: {len(runs)}; runs: {sum(len(v) for v in runs.values())}", "",
           "## Headline: Ours vs GeFL-F (and vs logit adjustment alone)", "",
           "final_bal = final-round class-balanced accuracy (%), mean ± std over seeds; deltas are paired by seed.", "",
           "| experiment | dataset | IF | alpha | K | seeds | GeFL-F bal | +LA bal | Ours bal | Ours−GeFL-F bal | "
           "Ours−GeFL-F tail | Ours−LA bal | GeFL-F best_mean_acc | Ours best_mean_acc |",
           "|" + "---|" * 14]
    for r in headline(runs):
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")
    for exp in sorted(runs):
        p = os.path.join(a.root, exp, "summary.md")
        if os.path.exists(p):
            out.append(open(p, encoding="utf-8").read().replace("# ", "## ", 1))
    path = os.path.join(a.root, "REPORT_quick.md" if a.quick else "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(path)


if __name__ == "__main__":
    main()
