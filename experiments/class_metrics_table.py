"""Class-imbalance metrics for every method of the fair re-test (K17 + K19, seeds 3-5, long tail), each with the readout
selected on validation data exactly as in k17_analysis.py.

    python experiments/class_metrics_table.py   -> docs/paper_assets/tables/T23_class_metrics.md

Columns (test set, mean over seeds): balanced accuracy (= macro recall, test sets are balanced except SVHN's), macro-F1,
G-mean of the class recalls, worst-class recall, recall on the tail classes (the rarest 40%), and per-class recall.
Precision is logged only by runs made after 22 Oct 2026 (core._class_metrics), so it is not in this table.
"""
import os

import numpy as np

import k17_analysis as k17

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "paper_assets", "tables", "T23_class_metrics.md"))
METHODS = [("GeFL-F", "GeFL-F"), ("GeFL-F + AdamW + LA", "GeFL-F, decoupled decay + LA"),
           ("Real only", "Real data only (no generator)"), ("Real only + LA", "Real data only + LA / balanced softmax"),
           ("Real only + FedLC", "FedLC"), ("Real only + FedRS", "FedRS"), ("FSG + LA", "Gaussian generator + LA"),
           ("FedDF", "FedDF"), ("FedDF + LA", "FedDF + LA")]
NAMES = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100-LT"}


def mean_metric(runs, key):
    xs = [r[key] for r in runs.values() if isinstance(r.get(key), (int, float))]
    return 100 * float(np.mean(xs)) if xs else float("nan")


def mean_recall(runs, key):
    xs = [r[key] for r in runs.values() if isinstance(r.get(key), list)]
    return 100 * np.mean(xs, 0) if xs else None


def main():
    D = k17.load()
    L = ["### T23_class_metrics: class-imbalance metrics, fair re-test (seeds 3-5, long tail, IF = 100)", "",
         "Each method with its validation-selected readout (as in T22); test set, mean over three seeds, in %. "
         "Balanced accuracy = macro recall. G-mean = geometric mean of class recalls (each floored at 1%). "
         "Tail = mean recall of the rarest 40% of classes.", ""]
    for ds in k17.DATASETS:
        o1, o10 = D.get((ds, "Ours Ts=1")), D.get((ds, "Ours Ts=10"))
        if not o1 or not o10:
            continue
        ours = o1 if k17.mean_of(o1, "val_er_bal") >= k17.mean_of(o10, "val_er_bal") else o10
        rows = [("**Ours**", ours)] + [(name, D[(ds, lab)]) for lab, name in METHODS if D.get((ds, lab))]
        L += [f"**{NAMES[ds]}**", "",
              "| method | readout | balanced acc. | macro-F1 | G-mean | worst class | tail classes |",
              "|---|---|---|---|---|---|---|"]
        per_class = []
        for name, runs in rows:
            ro = k17.pick_readout(runs)
            cells = [mean_metric(runs, f"{ro}_{k}") for k in ("bal", "f1", "gmean", "worst", "tail")]
            L.append(f"| {name} | {ro} | " + " | ".join(f"{c:.1f}" for c in cells) + " |")
            rec = mean_recall(runs, f"{ro}_recall")
            if rec is not None and len(rec) <= 10:
                per_class.append((name, rec))
        L += ["", f"Same metrics with every head as trained (no readout), {NAMES[ds]}:", "",
              "| method | balanced acc. | macro-F1 | G-mean | worst class | tail classes |", "|---|---|---|---|---|---|"]
        for name, runs in rows:
            cells = [mean_metric(runs, f"final_{k}") for k in ("bal", "f1", "gmean", "worst", "tail")]
            L.append(f"| {name} | " + " | ".join(f"{c:.1f}" for c in cells) + " |")
        L.append("")
        if per_class:
            C = len(per_class[0][1])
            L += [f"Per-class recall, {NAMES[ds]} (class 0 = most common, class {C - 1} = rarest):", "",
                  "| method | " + " | ".join(str(c) for c in range(C)) + " |", "|---|" + "---|" * C]
            L += [f"| {name} | " + " | ".join(f"{v:.0f}" for v in rec) + " |" for name, rec in per_class]
            L.append("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
