"""Every table and figure of the paper, computed from experiments/results only.

    python experiments/paper_assets.py

Writes docs/paper_assets/:
    tables/<id>.md, tables/<id>.tex   one per paper table (mean ± sd over seeds, n, p-values)
    figures/<id>.png, figures/<id>.pdf
    numbers.csv                        every cell: value, sd, n, seeds, source folder + label
    CHECK.md                           every cell whose value differs from the number the draft quotes
    INDEX.md                           which asset goes in which paper section, and its sources

Nothing here is typed in by hand except numbers published by others (marked `paper:`) and the
draft's quoted values (used only to flag mismatches). Re-run after adding results; never edit outputs.
"""
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.normpath(os.path.join(HERE, "..", "docs", "paper_assets"))
PCT = ("final", "best", "oracle", "fidelity", "referee", "bbc", "ncm")


def pct(metric):
    return 100.0 if metric.startswith(PCT) else 1.0


# ------------------------------------------------------------------ loading
class Store:
    """runs[(folder, label, ds, IF, K)] -> {seed: row} (jsonl), or agg[...] -> {metric: (mean, sd, n)} (summary only)."""

    def __init__(self):
        self.runs, self.agg = defaultdict(dict), {}
        for d in sorted(glob.glob(os.path.join(RES, "*/"))):
            folder = os.path.basename(d.rstrip("/"))
            jl, sm = os.path.join(d, "runs.jsonl"), os.path.join(d, "summary.md")
            if os.path.exists(jl):
                for line in open(jl, encoding="utf-8"):
                    r = json.loads(line)
                    self.runs[(folder, r["label"], r["dataset"], float(r["IF"]), int(r["K"]))][int(r["seed"])] = r
            elif os.path.exists(sm):
                self._parse_summary(folder, sm)

    def _parse_summary(self, folder, path):
        cur, cols = None, None
        for line in open(path, encoding="utf-8"):
            h = re.match(r"## (\w+)\s+IF=([\d.]+)\s+alpha=(\S+)\s+K=(\d+)", line)
            if h:
                cur, cols = (h.group(1), float(h.group(2)), int(h.group(4))), None
                continue
            if cur and line.startswith("| label | n |"):
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                continue
            if cur and cols and line.startswith("| ") and "±" in line:
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                lab, n = cells[0], int(cells[1])
                vals = {}
                for c, v in zip(cols[2:], cells[2:]):
                    m = re.match(r"([-\d.]+) ± ([\d.]+)", v)
                    if m:
                        vals[c] = (float(m.group(1)), float(m.group(2)), n)
                self.agg[(folder, lab) + cur] = vals

    def cell(self, folder, label, ds, IF, K, metric):
        """-> dict(mean, sd, n, seeds{seed: value} or None, src)"""
        key = (folder, label, ds, float(IF), int(K))
        src = f"{folder} :: {label} :: {ds} IF={IF} K={K}"
        if key in self.runs:
            xs = {s: r[metric] * pct(metric) for s, r in self.runs[key].items()
                  if r.get(metric) is not None and r.get(metric) == r.get(metric)}
            if not xs:
                return None
            v = np.array(list(xs.values()))
            return dict(mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, n=len(v), seeds=xs, src=src)
        if key in self.agg and metric in self.agg[key]:
            m, s, n = self.agg[key][metric]
            return dict(mean=m, sd=s, n=n, seeds=None, src=src + " (summary.md only)")
        return None


def ptest(a, b):
    """Paired t-test when both have per-seed values on >= 2 shared seeds, else Welch from mean/sd."""
    if a is None or b is None:
        return None, ""
    if a["seeds"] and b["seeds"]:
        sh = sorted(set(a["seeds"]) & set(b["seeds"]))
        if len(sh) >= 2:
            x, y = [a["seeds"][s] for s in sh], [b["seeds"][s] for s in sh]
            if np.allclose(np.array(x) - np.array(y), 0):
                return 1.0, "paired"
            return float(stats.ttest_rel(x, y).pvalue), "paired"
    if a["n"] >= 2 and b["n"] >= 2 and (a["sd"] > 0 or b["sd"] > 0):
        r = stats.ttest_ind_from_stats(a["mean"], a["sd"], a["n"], b["mean"], b["sd"], b["n"], equal_var=False)
        return float(r.pvalue), "Welch"
    return None, ""


# ------------------------------------------------------------------ table registry
# A cell is ("folder", "label", ds, IF, K, metric[, draft value]) or ("paper", value) or None.
LT, IID = 0.01, 1.0


def C(folder, label, ds, IF, K=10, metric="final_bal", draft=None):
    return ("run", folder, label, ds, IF, K, metric, draft)


def P(value):
    return ("paper", value)


TABLES = []


def table(tid, section, title, cols, rows, ref=None, note=""):
    TABLES.append(dict(id=tid, section=section, title=title, cols=cols, rows=rows, ref=ref, note=note))


# §5.2 parity
table("T01_parity", "5.2", "Our GeFL-F against the paper and the authors' code (best_mean_acc; LT: final balanced acc.)",
      ["Paper", "Authors' code", "Ours"],
      [("MNIST IID, K=10", [P(95.47), P(96.07), C("F10_combined_method", "GeFL-F", "mnist", IID, metric="best_mean_acc", draft=96.02)]),
       ("MNIST IID, K=50", [P(95.04), None, C("F09_consensus_paper_setting", "GeFL-F", "mnist", IID, 50, "best_mean_acc", 95.16)]),
       ("MNIST IID, K=100", [P(94.63), None, C("F09_consensus_paper_setting", "GeFL-F", "mnist", IID, 100, "best_mean_acc", 94.61)]),
       ("FMNIST IID, K=10", [P(83.14), None, C("F09_consensus_paper_setting", "GeFL-F", "fmnist", IID, metric="best_mean_acc", draft=82.67)])],
      note="Authors' code value from experiments/reference_check (one seed).")

# §5.3 long tail, minimal variant
_lt = [("FedAvg (per architecture)", "FedAvg (grouped)", 62.9, 52.8), ("LG-FedAvg", "LG-FedAvg", 63.2, 52.5),
       ("LG-FedAvg + LA", "LG-FedAvg+LA", 67.8, 57.6), ("GeFL-F", "GeFL-F", 74.6, 58.6),
       ("GeFL-F + LA", "+LA", 78.9, 64.8), ("GeFL-F + HWA + LA", "+HWA+LA", 87.9, 76.0)]
table("T02_longtail_minimal", "5.3", "Long-tailed clients (IF = 100, Dir 0.5, K = 10; F01)",
      ["MNIST bal.", "MNIST tail", "FMNIST bal.", "FMNIST tail"],
      [(name, [C("F01_main_longtail", lab, "mnist", LT, draft=dm), C("F01_main_longtail", lab, "mnist", LT, metric="final_tail"),
               C("F01_main_longtail", lab, "fmnist", LT, draft=df), C("F01_main_longtail", lab, "fmnist", LT, metric="final_tail")])
       for name, lab, dm, df in _lt], ref="GeFL-F")

# §5.3 more clients (MNIST LT)
table("T03_clients_minimal", "5.3", "More clients, MNIST long tail (balanced accuracy; F01/F10 at K = 10, F04 at K = 50/100)",
      ["K = 10", "K = 50", "K = 100"],
      [("GeFL-F", [C("F10_combined_method", "GeFL-F", "mnist", LT, draft=74.2), C("F04_clients", "GeFL-F", "mnist", LT, 50, draft=78.1),
                   C("F04_clients", "GeFL-F", "mnist", LT, 100, draft=76.0)]),
       ("+ LA", [C("F01_main_longtail", "+LA", "mnist", LT, draft=78.9), C("F04_clients", "+LA", "mnist", LT, 50, draft=81.4),
                 C("F04_clients", "+LA", "mnist", LT, 100, draft=79.5)]),
       ("+ HWA + LA", [C("F10_combined_method", "+HWA+LA", "mnist", LT, draft=87.8), C("F04_clients", "+HWA+LA", "mnist", LT, 50, draft=86.5),
                       C("F04_clients", "+HWA+LA", "mnist", LT, 100, draft=83.1)]),
       ("+ HWA + LA + CSL", [C("F10_combined_method", "+HWA+LA+CSL", "mnist", LT, draft=89.9), C("F04_clients", "+HWA+LA+CSL", "mnist", LT, 50, draft=86.8),
                             C("F04_clients", "+HWA+LA+CSL", "mnist", LT, 100, draft=82.7)]),
       ("FSG + LA", [C("F01_main_longtail", "FSG+LA", "mnist", LT, draft=87.0), C("F04_clients", "FSG+LA", "mnist", LT, 50, draft=87.7),
                     C("F04_clients", "FSG+LA", "mnist", LT, 100, draft=86.1)]),
       ("FSG + LA + CSL", [None, C("F04_clients", "FSG+LA+CSL", "mnist", LT, 50, draft=87.2), C("F04_clients", "FSG+LA+CSL", "mnist", LT, 100, draft=85.5)]),
       ("MIX + HWA + LA + CSL", [None, C("F04_clients", "MIX+HWA+LA+CSL", "mnist", LT, 50, draft=87.9),
                                 C("F04_clients", "MIX+HWA+LA+CSL", "mnist", LT, 100, draft=85.2)])], ref="GeFL-F")

# §5.4 IID + CSL
table("T04_iid_csl", "5.4", "GeFL-F's own IID setting: consensus soft labels (best_mean_acc; F09, K01)",
      ["Paper GeFL-F", "Our GeFL-F", "+ CSL"],
      [("MNIST, K=10", [P(95.47), C("F09_consensus_paper_setting", "GeFL-F", "mnist", IID, metric="best_mean_acc", draft=96.02),
                        C("F09_consensus_paper_setting", "+CSL (beta=0.5)", "mnist", IID, metric="best_mean_acc", draft=96.71)]),
       ("FMNIST, K=10", [P(83.14), C("F09_consensus_paper_setting", "GeFL-F", "fmnist", IID, metric="best_mean_acc", draft=82.67),
                         C("F09_consensus_paper_setting", "+CSL (beta=0.5)", "fmnist", IID, metric="best_mean_acc", draft=83.16)]),
       ("MNIST, K=50", [P(95.04), C("F09_consensus_paper_setting", "GeFL-F", "mnist", IID, 50, "best_mean_acc", 95.16),
                        C("F09_consensus_paper_setting", "+CSL (beta=0.5)", "mnist", IID, 50, "best_mean_acc", 95.41)]),
       ("MNIST, K=100", [P(94.63), C("F09_consensus_paper_setting", "GeFL-F", "mnist", IID, 100, "best_mean_acc", 94.61),
                         C("F09_consensus_paper_setting", "+CSL (beta=0.5)", "mnist", IID, 100, "best_mean_acc", 94.72)]),
       ("SVHN, K=10", [P(76.26), C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", IID, metric="best_mean_acc", draft=75.82),
                       C("K01_kaggle_cifar10_svhn__svhn", "+CSL (beta=0.5)", "svhn", IID, metric="best_mean_acc", draft=76.30)]),
       ("CIFAR-10, K=10", [P(55.86), C("K01_kaggle_cifar10_svhn__cifar10", "GeFL-F", "cifar10", IID, metric="best_mean_acc", draft=59.10),
                           C("K01_kaggle_cifar10_svhn__cifar10", "+CSL (beta=0.5)", "cifar10", IID, metric="best_mean_acc")])],
      ref=None, note="p for + CSL vs our GeFL-F is computed per row in numbers.csv.")

# §5.5 combined (F10)
_f10 = [("GeFL-F", "GeFL-F", (74.24, 58.74, 96.01, 82.80)), ("+ CSL", "+CSL", (76.35, 58.59, 96.82, 83.25)),
        ("+ HWA + LA", "+HWA+LA", (87.78, 75.99, 96.01, 82.74)), ("+ HWA + LA + CSL (ours)", "+HWA+LA+CSL", (89.87, 76.37, 96.77, 83.21))]
table("T05_combined", "5.5", "One method for both regimes (F10; LT: final balanced acc., IID: best_mean_acc)",
      ["MNIST LT", "FMNIST LT", "MNIST IID", "FMNIST IID"],
      [(name, [C("F10_combined_method", lab, "mnist", LT, draft=d[0]), C("F10_combined_method", lab, "fmnist", LT, draft=d[1]),
               C("F10_combined_method", lab, "mnist", IID, metric="best_mean_acc", draft=d[2]),
               C("F10_combined_method", lab, "fmnist", IID, metric="best_mean_acc", draft=d[3])]) for name, lab, d in _f10],
      ref="GeFL-F")

# §5.6 regimes (K02)
_k02 = "K02_kaggle_regimes_privacy_ablations"
table("T06_regimes", "5.6", "The gain grows with imbalance (balanced accuracy, K = 10; K02, minimal variant)",
      ["MNIST GeFL-F", "MNIST ours", "FMNIST GeFL-F", "FMNIST ours"],
      [(name, [C(_k02, "GeFL-F", "mnist", IF, draft=d[0]), C(_k02, "Ours", "mnist", IF, draft=d[1]),
               C(_k02, "GeFL-F", "fmnist", IF, draft=d[2]), C(_k02, "Ours", "fmnist", IF, draft=d[3])])
       for name, IF, d in [("label skew only (IF = 1, Dir 0.5)", 1.0, (89.8, 95.6, 74.1, 79.2)),
                           ("mild tail (IF = 10)", 0.1, (86.9, 94.3, 72.6, 79.4)),
                           ("heavy tail (IF = 100)", 0.01, (75.3, 90.2, 58.9, 76.5))]])
table("T07_k02_ablations", "5.6", "Ablations and privacy of the minimal variant (IF = 100; K02; balanced accuracy)",
      ["MNIST", "FMNIST"],
      [(lab, [C(_k02, lab, "mnist", LT), C(_k02, lab, "fmnist", LT)])
       for lab in ["GeFL-F", "Ours", "no HWA (flat averaging)", "no LA (plain CE)", "no CSL", "HWA weights = counts n",
                   "HWA weights = uniform over holders", "CSL beta=0.25", "CSL beta=0.75", "LA tau=1.5", "LA tau=2.0",
                   "Ours, DP eps=10", "Ours, DP eps=1", "Ours, DP eps=0.1"]], ref="Ours")

# E26 formal DP
table("T08_dp", "5.6", "Formal DP on every statistic the anchored method adds (E26; LT, K = 10)",
      ["MNIST", "FMNIST"],
      [("GeFL-F (no formal privacy)", [C("E20_proto_generator", "GeFL-F", "mnist", LT, draft=75.25), C("E20_proto_generator", "GeFL-F", "fmnist", LT, draft=58.47)]),
       ("final method, exact statistics", [C("E25_anchored_stack", "Ours-A Ts=10", "mnist", LT, draft=94.82), C("E25_anchored_stack", "Ours-A Ts=10", "fmnist", LT, draft=79.67)]),
       ("(8, 1e-5)-DP statistics", [C("E26_private_anchoring", "Ours-A Ts=10, DP eps=8", "mnist", LT, draft=90.36),
                                    C("E26_private_anchoring", "Ours-A Ts=10, DP eps=8", "fmnist", LT, draft=72.77)]),
       ("(2, 1e-5)-DP statistics", [C("E26_private_anchoring", "Ours-A Ts=10, DP eps=2", "mnist", LT, draft=85.06),
                                    C("E26_private_anchoring", "Ours-A Ts=10, DP eps=2", "fmnist", LT, draft=65.59)])],
      ref="final method, exact statistics")

# §5.9 synthetic budget (E16)
_e16 = "E16_csl_synthetic_budget"
table("T09_budget_iid", "5.9", "Synthetic budget T_s, FMNIST IID (best_mean_acc; E16)",
      ["T_s=1", "T_s=3", "T_s=5", "T_s=10"],
      [("GeFL-F (hard labels)", [C(_e16, "GeFL-F", "fmnist", IID, metric="best_mean_acc", draft=82.96), C(_e16, "GeFL-F Ts=3", "fmnist", IID, metric="best_mean_acc", draft=83.20),
                                 C(_e16, "GeFL-F Ts=5", "fmnist", IID, metric="best_mean_acc", draft=83.18), None]),
       ("+ CSL", [C(_e16, "+CSL", "fmnist", IID, metric="best_mean_acc", draft=83.31), C(_e16, "+CSL Ts=3", "fmnist", IID, metric="best_mean_acc", draft=83.90),
                  C(_e16, "+CSL Ts=5", "fmnist", IID, metric="best_mean_acc", draft=84.14), C(_e16, "+CSL Ts=10", "fmnist", IID, metric="best_mean_acc", draft=84.37)]),
       ("Ours (HWA + LA + CSL)", [None, None, C(_e16, "Ours (HWA+LA+CSL) Ts=5", "fmnist", IID, metric="best_mean_acc", draft=84.12),
                                 C(_e16, "Ours (HWA+LA+CSL) Ts=10", "fmnist", IID, metric="best_mean_acc", draft=84.34)])])
table("T10_budget_lt", "5.9", "Synthetic budget under the long tail (FMNIST, IF = 100; E16; balanced accuracy)",
      ["balanced", "tail"],
      [(n, [C(_e16, lab, "fmnist", LT, draft=d), C(_e16, lab, "fmnist", LT, metric="final_tail", draft=t)])
       for n, lab, d, t in [("GeFL-F", "GeFL-F", 61.33, 38.86), ("Ours, T_s=1", "Ours (HWA+LA+CSL)", 76.98, 70.17),
                            ("Ours, T_s=5", "Ours (HWA+LA+CSL) Ts=5", 78.76, 72.37), ("Ours, T_s=10", "Ours (HWA+LA+CSL) Ts=10", 79.33, 72.44)]],
      ref="GeFL-F")

# §5.11 anchored method
table("T11_anchored", "5.11", "The anchored method (K = 10; LT: final balanced acc., IID: best_mean_acc)",
      ["MNIST LT", "FMNIST LT", "MNIST IID", "FMNIST IID"],
      [("GeFL-F", [C("E20_proto_generator", "GeFL-F", "mnist", LT, draft=75.25), C("E20_proto_generator", "GeFL-F", "fmnist", LT, draft=58.47),
                   C("F10_combined_method", "GeFL-F", "mnist", IID, metric="best_mean_acc", draft=96.01),
                   C("E20_proto_generator", "GeFL-F", "fmnist", IID, metric="best_mean_acc", draft=82.76)]),
       ("minimal variant (CVAE + HWA + LA + CSL)", [C("E20_proto_generator", "Ours (HWA+LA+CSL)", "mnist", LT, draft=90.40),
                                                    C("E20_proto_generator", "Ours (HWA+LA+CSL)", "fmnist", LT, draft=76.60),
                                                    C("F10_combined_method", "+HWA+LA+CSL", "mnist", IID, metric="best_mean_acc", draft=96.77),
                                                    C("E20_proto_generator", "Ours (HWA+LA+CSL)", "fmnist", IID, metric="best_mean_acc", draft=83.19)]),
       ("PC-VAE + MC", [C("E20_proto_generator", "Ours-PC+MC", "mnist", LT, draft=92.31), C("E20_proto_generator", "Ours-PC+MC", "fmnist", LT, draft=77.56),
                        None, C("E20_proto_generator", "Ours-PC+MC", "fmnist", IID, metric="best_mean_acc", draft=82.92)]),
       ("KME-Gen (1 seed)", [C("E24_kme_generator", "Ours-KME (KME+LA+CSL)", "mnist", LT, draft=89.88),
                             C("E24_kme_generator", "Ours-KME (KME+LA+CSL)", "fmnist", LT, draft=75.11), None,
                             C("E24_kme_generator", "Ours-KME (KME+LA+CSL)", "fmnist", IID, metric="best_mean_acc", draft=81.82)]),
       ("Ours-A, T_s = 1", [C("E25_anchored_stack", "Ours-A (PC+MC+KH+LA+CSL)", "mnist", LT, draft=93.11),
                            C("E25_anchored_stack", "Ours-A (PC+MC+KH+LA+CSL)", "fmnist", LT, draft=78.22),
                            C("E25_anchored_stack", "Ours-A (PC+MC+KH+LA+CSL)", "mnist", IID, metric="best_mean_acc", draft=96.67),
                            C("E25_anchored_stack", "Ours-A (PC+MC+KH+LA+CSL)", "fmnist", IID, metric="best_mean_acc", draft=83.35)]),
       ("Ours-A, T_s = 10", [C("E25_anchored_stack", "Ours-A Ts=10", "mnist", LT, draft=94.82), C("E25_anchored_stack", "Ours-A Ts=10", "fmnist", LT, draft=79.67),
                             C("E25_anchored_stack", "Ours-A Ts=10", "mnist", IID, metric="best_mean_acc", draft=97.62),
                             C("E25_anchored_stack", "Ours-A Ts=10", "fmnist", IID, metric="best_mean_acc", draft=83.96)]),
       ("Ours-A, T_s = 10, + gated BBC", [C("E25_anchored_stack", "Ours-A Ts=10", "mnist", LT, metric="bbc_bal", draft=95.17),
                                         C("E25_anchored_stack", "Ours-A Ts=10", "fmnist", LT, metric="bbc_bal", draft=80.55), None, None]),
       ("Ours-A-B (MC-B), T_s = 10", [C("E32_moment_anchor_bayes", "Ours-A-B Ts=10", "mnist", LT), C("E32_moment_anchor_bayes", "Ours-A-B Ts=10", "fmnist", LT),
                                      None, C("E32_moment_anchor_bayes", "Ours-A-B Ts=10", "fmnist", IID, metric="best_mean_acc")]),
       ("oracle: heads re-fit on pooled real data", [C("E25_anchored_stack", "Ours-A Ts=10", "mnist", LT, metric="oracle_bal", draft=94.93),
                                                     C("E25_anchored_stack", "Ours-A Ts=10", "fmnist", LT, metric="oracle_bal", draft=81.55),
                                                     None, None])],
      ref="GeFL-F")

# §5.11 many clients (E22 / K08 FMNIST / K03)
table("T12_clients_anchored", "5.11", "More clients under the long tail (balanced accuracy; E22 MNIST, K03/K08 FMNIST)",
      ["MNIST K=100", "FMNIST K=50", "FMNIST K=100"],
      [("GeFL-F", [C("F04_clients", "GeFL-F", "mnist", LT, 100, draft=75.98), C("K03_kaggle_client_scaling__K50", "GeFL-F", "fmnist", LT, 50),
                   C("K03_kaggle_client_scaling__K100", "GeFL-F", "fmnist", LT, 100)]),
       ("CVAE + HWA + LA", [C("F04_clients", "+HWA+LA", "mnist", LT, 100, draft=83.07), C("K03_kaggle_client_scaling__K50", "+HWA+LA", "fmnist", LT, 50),
                            C("K03_kaggle_client_scaling__K100", "+HWA+LA", "fmnist", LT, 100)]),
       ("FSG + LA", [C("F04_clients", "FSG+LA", "mnist", LT, 100, draft=86.05), C("K03_kaggle_client_scaling__K50", "FSG+LA", "fmnist", LT, 50),
                     C("K03_kaggle_client_scaling__K100", "FSG+LA", "fmnist", LT, 100)]),
       ("Ours-A, T_s = 1", [C("E22_pc_many_clients", "Ours-A (PC+MC+KH+LA+CSL)", "mnist", LT, 100, draft=90.41),
                            C("K08_kaggle_anchored_stack__fmnist", "Ours-A (PC+MC+KH+LA+CSL)", "fmnist", LT, 50),
                            C("K08_kaggle_anchored_stack__fmnist", "Ours-A (PC+MC+KH+LA+CSL)", "fmnist", LT, 100)]),
       ("Ours-A, T_s = 10", [C("E22_pc_many_clients", "Ours-A Ts=10", "mnist", LT, 100, draft=94.30),
                             C("K08_kaggle_anchored_stack__fmnist", "Ours-A Ts=10", "fmnist", LT, 50),
                             C("K08_kaggle_anchored_stack__fmnist", "Ours-A Ts=10", "fmnist", LT, 100)])],
      ref="GeFL-F", note="MNIST GeFL-F/HWA/FSG at K=100 come from F04 (same splits as E22).")

# CIFAR-10 / SVHN
table("T13_cifar_svhn", "5.11", "CIFAR-10 and SVHN (LT: final balanced acc., IID: best_mean_acc; K01, K08, K11)",
      ["CIFAR-10 LT", "SVHN LT", "CIFAR-10 IID", "SVHN IID"],
      [("GeFL-F", [C("K01_kaggle_cifar10_svhn__cifar10", "GeFL-F", "cifar10", LT, draft=33.74), C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", LT, draft=50.49),
                   C("K01_kaggle_cifar10_svhn__cifar10", "GeFL-F", "cifar10", IID, metric="best_mean_acc", draft=59.10),
                   C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", IID, metric="best_mean_acc", draft=75.82)]),
       ("+ HWA + LA", [C("K01_kaggle_cifar10_svhn__cifar10", "+HWA+LA", "cifar10", LT, draft=40.73),
                       C("K01_kaggle_cifar10_svhn__svhn", "+HWA+LA (ours)", "svhn", LT, draft=62.45), None,
                       C("K01_kaggle_cifar10_svhn__svhn", "+HWA+LA (ours)", "svhn", IID, metric="best_mean_acc", draft=75.64)]),
       ("minimal variant / + CSL", [C("K01_kaggle_cifar10_svhn__cifar10", "Ours (HWA+LA+CSL)", "cifar10", LT, draft=42.10), None,
                                    C("K01_kaggle_cifar10_svhn__cifar10", "+CSL (beta=0.5)", "cifar10", IID, metric="best_mean_acc", draft=60.02),
                                    C("K01_kaggle_cifar10_svhn__svhn", "+CSL (beta=0.5)", "svhn", IID, metric="best_mean_acc", draft=76.30)]),
       ("FSG + LA", [C("K01_kaggle_cifar10_svhn__cifar10", "FSG+LA", "cifar10", LT, draft=44.17), C("K01_kaggle_cifar10_svhn__svhn", "FSG+LA", "svhn", LT, draft=44.23),
                     None, None]),
       ("Ours-A, T_s = 1", [C("K08_kaggle_anchored_stack__cifar10_lt", "Ours-A (PC+MC+KH+LA+CSL)", "cifar10", LT, draft=46.06),
                            C("K08_kaggle_anchored_stack__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", LT, draft=55.89),
                            C("K08_kaggle_anchored_stack__cifar10_iid", "Ours-A (PC+MC+KH+LA+CSL)", "cifar10", IID, metric="best_mean_acc", draft=60.39),
                            C("K08_kaggle_anchored_stack__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc", draft=73.86)]),
       ("Ours-A, T_s = 1, + gated BBC", [C("K08_kaggle_anchored_stack__cifar10_lt", "Ours-A (PC+MC+KH+LA+CSL)", "cifar10", LT, metric="bbc_bal", draft=49.0),
                                         C("K08_kaggle_anchored_stack__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", LT, metric="bbc_bal", draft=57.0), None, None]),
       ("Ours-A, T_s = 10", [C("K08_kaggle_anchored_stack__cifar10_lt", "Ours-A Ts=10", "cifar10", LT, draft=44.38),
                             C("K08_kaggle_anchored_stack__svhn", "Ours-A Ts=10", "svhn", LT, draft=54.41),
                             C("K08_kaggle_anchored_stack__cifar10_iid", "Ours-A Ts=10", "cifar10", IID, metric="best_mean_acc", draft=60.93),
                             C("K08_kaggle_anchored_stack__svhn", "Ours-A Ts=10", "svhn", IID, metric="best_mean_acc", draft=74.38)]),
       ("Ours-A, T_s = 10 + VCT", [C("K13_kaggle_vct_vs_mixup__cifar10_lt", "Ours-A + VCT", "cifar10", LT), None,
                                   C("K13_kaggle_vct_vs_mixup__cifar10_iid", "Ours-A Ts=10 + VCT", "cifar10", IID, metric="best_mean_acc"), None]),
       ("Ours-A, T_s = 10 + MixUp", [C("K13_kaggle_vct_vs_mixup__cifar10_lt", "Ours-A + MixUp", "cifar10", LT), None,
                                     C("K13_kaggle_vct_vs_mixup__cifar10_iid", "Ours-A Ts=10 + MixUp", "cifar10", IID, metric="best_mean_acc"), None]),
       ("paper: GeFL-F / best of ten", [None, None, P(55.86), P(76.26)])],
      ref="GeFL-F", note="K13 long-tail arms run at T_s = 1. paper best on CIFAR-10 IID is 59.36 (62.67 with MixUp).")

table("T14_svhn_gate", "5.11", "SVHN: MC-S, the hybrid and the rows (K = 10; K01, K08, K10, K11)",
      ["long tail", "+ gated BBC", "IID best_mean_acc"],
      [("GeFL-F", [C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", LT, draft=50.49), None,
                   C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", IID, metric="best_mean_acc", draft=75.82)]),
       ("+ HWA + LA", [C("K01_kaggle_cifar10_svhn__svhn", "+HWA+LA (ours)", "svhn", LT, draft=62.45), None,
                       C("K01_kaggle_cifar10_svhn__svhn", "+HWA+LA (ours)", "svhn", IID, metric="best_mean_acc", draft=75.64)]),
       ("Ours-A (PC + MC + KH)", [C("K11_kaggle_covariance_anchoring__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", LT, draft=55.76),
                                  C("K11_kaggle_covariance_anchoring__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", LT, metric="bbc_bal", draft=57.27),
                                  C("K11_kaggle_covariance_anchoring__svhn", "Ours-A (PC+MC+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc", draft=74.38)]),
       ("Ours-A-S (PC + MC-S + KH)", [C("K11_kaggle_covariance_anchoring__svhn", "Ours-A-S (PC+MCS+KH+LA+CSL)", "svhn", LT, draft=61.80),
                                      C("K11_kaggle_covariance_anchoring__svhn", "Ours-A-S (PC+MCS+KH+LA+CSL)", "svhn", LT, metric="bbc_bal", draft=63.30),
                                      C("K11_kaggle_covariance_anchoring__svhn", "Ours-A-S (PC+MCS+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc", draft=77.05)]),
       ("hybrid PCR + MC + KH", [C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours-R", "svhn", LT, draft=60.86),
                                 C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours-R", "svhn", LT, metric="bbc_bal", draft=62.0),
                                 C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours-R", "svhn", IID, metric="best_mean_acc", draft=76.06)]),
       ("hybrid PCR + MC-S + KH", [C("K11_kaggle_covariance_anchoring__svhn", "Ours-R-S (PCR+MCS+KH+LA+CSL)", "svhn", LT, draft=62.89),
                                   C("K11_kaggle_covariance_anchoring__svhn", "Ours-R-S (PCR+MCS+KH+LA+CSL)", "svhn", LT, metric="bbc_bal", draft=64.57),
                                   C("K11_kaggle_covariance_anchoring__svhn", "Ours-R-S (PCR+MCS+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc", draft=77.39)]),
       ("rows (CVAE + HWA) + MC + KH", [C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn", LT, draft=65.60),
                                        C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn", LT, metric="bbc_bal", draft=67.0),
                                        C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn", IID, metric="best_mean_acc", draft=77.07)]),
       ("paper: GeFL-F / best of ten", [None, None, P(76.26)])],
      ref="Ours-A (PC + MC + KH)")

# K09 ablation (+ K14 seeds 3-4 when present)
_abl = ["Ours-A Ts=10", "-LA", "-CSL", "-MC", "-KH", "-PC (CVAE+HWA)"]
table("T15_ablation", "5.11", "Ablation of the final method (LT, K = 10; K09 seeds 0-2 + K14 seeds 3-4 when present)",
      ["MNIST bal.", "MNIST BBC", "FMNIST bal.", "FMNIST BBC"],
      [(lab, [C("K09+K14:mnist", lab, "mnist", LT), C("K09+K14:mnist", lab, "mnist", LT, metric="bbc_bal"),
              C("K09+K14:fmnist", lab, "fmnist", LT), C("K09+K14:fmnist", lab, "fmnist", LT, metric="bbc_bal")]) for lab in _abl]
      + [("Ours-A-B (MC-B; E32 + K14)", [C("E32+K14:mnist", "Ours-A-B Ts=10", "mnist", LT), C("E32+K14:mnist", "Ours-A-B Ts=10", "mnist", LT, metric="bbc_bal"),
                                         C("E32+K14:fmnist", "Ours-A-B Ts=10", "fmnist", LT), C("E32+K14:fmnist", "Ours-A-B Ts=10", "fmnist", LT, metric="bbc_bal")])],
      ref="Ours-A Ts=10")

# DDPM-F
table("T16_ddpm", "5.8", "Generator-agnostic: on the paper's DDPM-F (IID, best_mean_acc; F11, K04)",
      ["FMNIST", "SVHN"],
      [("GeFL-F (DDPM-F)", [C("F11_ddpm_generator", "GeFL-F (DDPM-F)", "fmnist", IID, metric="best_mean_acc", draft=83.16),
                            C("K04_kaggle_ddpm_cifar10_svhn__svhn", "GeFL-F (DDPM-F)", "svhn", IID, metric="best_mean_acc")]),
       ("+ CSL (DDPM-F)", [C("F11_ddpm_generator", "+CSL (DDPM-F)", "fmnist", IID, metric="best_mean_acc", draft=83.04),
                           C("K04_kaggle_ddpm_cifar10_svhn__svhn", "+CSL (DDPM-F)", "svhn", IID, metric="best_mean_acc")]),
       ("Ours (DDPM-F)", [C("F11_ddpm_generator", "Ours (DDPM-F)", "fmnist", IID, metric="best_mean_acc", draft=83.10),
                          C("K04_kaggle_ddpm_cifar10_svhn__svhn", "Ours (DDPM-F)", "svhn", IID, metric="best_mean_acc")]),
       ("paper: GeFL-F DDPM-F", [P(84.28), None])], ref="GeFL-F (DDPM-F)")

MERGED = {"K09+K14:mnist": ["K09_kaggle_ablation_final__mnist", "K14_kaggle_ablation_seeds_mcb__mnist"],
          "K09+K14:fmnist": ["K09_kaggle_ablation_final__fmnist", "K14_kaggle_ablation_seeds_mcb__fmnist"],
          "E32+K14:mnist": ["E32_moment_anchor_bayes", "K14_kaggle_ablation_seeds_mcb__mnist"],
          "E32+K14:fmnist": ["E32_moment_anchor_bayes", "K14_kaggle_ablation_seeds_mcb__fmnist"]}


def get(store, spec):
    if spec is None:
        return None
    if spec[0] == "paper":
        return dict(mean=spec[1], sd=None, n=None, seeds=None, src="published (paper)", paper=True)
    _, folder, label, ds, IF, K, metric, draft = spec
    if folder in MERGED:
        parts = [store.cell(f, label, ds, IF, K, metric) for f in MERGED[folder]]
        parts = [p for p in parts if p]
        if not parts:
            return None
        seeds = {}
        for p in parts:
            seeds.update(p["seeds"] or {})
        v = np.array(list(seeds.values()))
        c = dict(mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0, n=len(v), seeds=seeds,
                 src=" + ".join(p["src"] for p in parts))
    else:
        c = store.cell(folder, label, ds, IF, K, metric)
    if c is not None:
        c["draft"] = draft
    return c


# ------------------------------------------------------------------ writers
def fmt(c, sd=True):
    if c is None:
        return "–"
    if c.get("paper"):
        return f"{c['mean']:.2f}"
    s = f"{c['mean']:.2f}"
    if sd and c["n"] and c["n"] > 1:
        s += f" ± {c['sd']:.2f}"
    if c["n"] == 1:
        s += " (1 seed)"
    elif c["n"] and c["n"] != 3:
        s += f" (n={c['n']})"
    return s


def build_tables(store):
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    allcells, mism = [], []
    for t in TABLES:
        grid = [(name, [get(store, s) for s in specs]) for name, specs in t["rows"]]
        ref = dict(grid).get(t["ref"]) if t["ref"] else None
        md = [f"### {t['id']}: {t['title']}", "", f"*Paper section {t['section']}.* Mean ± sd over seeds; "
              f"p = two-sided t-test against **{t['ref']}** (paired by seed when per-seed data exist, else Welch)."
              if t["ref"] else f"*Paper section {t['section']}.* Mean ± sd over seeds.", "",
              "| | " + " | ".join(t["cols"]) + " |", "|---" * (len(t["cols"]) + 1) + "|"]
        tex = [r"\begin{tabular}{l" + "c" * len(t["cols"]) + "}", r"\toprule",
               " & " + " & ".join(t["cols"]) + r" \\", r"\midrule"]
        for name, cells in grid:
            row_md, row_tex = [], []
            for j, c in enumerate(cells):
                txt = fmt(c)
                if ref and c and not c.get("paper") and name != t["ref"]:
                    p, kind = ptest(c, ref[j])
                    if p is not None:
                        txt += f" (p={p:.3g})"
                    c["p"], c["test"] = p, kind
                row_md.append(txt)
                row_tex.append(txt.replace("±", r"$\pm$").replace("–", "--"))
                if c is not None:
                    allcells.append(dict(table=t["id"], row=name, col=t["cols"][j], mean=round(c["mean"], 4),
                                         sd=None if c["sd"] is None else round(c["sd"], 4), n=c["n"],
                                         seeds=json.dumps({k: round(v, 4) for k, v in c["seeds"].items()}) if c.get("seeds") else "",
                                         p_vs_ref=c.get("p"), test=c.get("test", ""), draft=c.get("draft"), source=c["src"]))
                    d = c.get("draft")
                    if d is not None and abs(round(c["mean"], 2) - d) > (0.051 if d == round(d, 1) else 0.006) + (0.5 if d in (49.0, 57.0, 62.0, 67.0) else 0):
                        mism.append((t["id"], name, t["cols"][j], d, c["mean"], c["src"]))
            md.append(f"| {name} | " + " | ".join(row_md) + " |")
            tex.append(name.replace("_", r"\_") + " & " + " & ".join(row_tex) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}"]
        if t["note"]:
            md += ["", t["note"]]
        missing = [f"{name} / {t['cols'][j]}" for (name, specs), (_, cells) in zip(t["rows"], grid)
                   for j, (s, c) in enumerate(zip(specs, cells)) if s is not None and c is None]
        if missing:
            md += ["", "**Missing (no run found):** " + "; ".join(missing)]
        open(os.path.join(OUT, "tables", t["id"] + ".md"), "w").write("\n".join(md) + "\n")
        open(os.path.join(OUT, "tables", t["id"] + ".tex"), "w").write("\n".join(tex) + "\n")
        t["missing"] = missing
    with open(os.path.join(OUT, "numbers.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allcells[0].keys()))
        w.writeheader()
        w.writerows(allcells)
    return allcells, mism


# ------------------------------------------------------------------ figures
INK, MUTED, OURS, NEUT, GOOD, BAD = "#1f2a37", "#5b6676", "#1c7ed6", "#868e96", "#2f9e44", "#c92a2a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
FIGS = []


def save(fig, fid, section, caption, sources):
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, "figures", f"{fid}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    FIGS.append(dict(id=fid, section=section, caption=caption, sources=sources))


def grouped_bars(store, fid, section, caption, groups, series, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(groups) + 1.5), 3.4))
    w = 0.8 / len(series)
    srcs = set()
    for i, (name, specs, color) in enumerate(series):
        cells = [get(store, s) for s in specs]
        xs = np.arange(len(groups)) + (i - (len(series) - 1) / 2) * w
        for x, c in zip(xs, cells):
            if c is None:
                continue
            srcs.add(c["src"].split(" :: ")[0])
            ax.bar(x, c["mean"], w * 0.92, color=color, label=name if x == xs[[k for k, cc in enumerate(cells) if cc][0]] else None,
                   yerr=c["sd"] if c.get("sd") else None, capsize=2, error_kw=dict(lw=0.8, color=INK))
            if c.get("seeds"):
                ax.scatter([x] * len(c["seeds"]), list(c["seeds"].values()), s=6, color=INK, zorder=3, lw=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=7.5, ncol=min(3, len(series)), loc="upper left", bbox_to_anchor=(0, 1.18))
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    save(fig, fid, section, caption, sorted(srcs))


def figures(store):
    lt = lambda f, l, d, m="final_bal", K=10: C(f, l, d, LT, K, m)  # noqa: E731
    grouped_bars(store, "F02_longtail", "5.3 / 5.11",
                 "Long-tailed clients (IF = 100, Dir 0.5, K = 10): balanced accuracy, mean ± sd, dots = seeds.",
                 ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"],
                 [("GeFL-F", [lt("F01_main_longtail", "GeFL-F", "mnist"), lt("F01_main_longtail", "GeFL-F", "fmnist"),
                              lt("K01_kaggle_cifar10_svhn__cifar10", "GeFL-F", "cifar10"), lt("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn")], NEUT),
                  ("+ LA", [lt("F01_main_longtail", "+LA", "mnist"), lt("F01_main_longtail", "+LA", "fmnist"),
                            lt("K01_kaggle_cifar10_svhn__cifar10", "+LA", "cifar10"), lt("K01_kaggle_cifar10_svhn__svhn", "+LA", "svhn")], "#adb5bd"),
                  ("+ HWA + LA", [lt("F01_main_longtail", "+HWA+LA", "mnist"), lt("F01_main_longtail", "+HWA+LA", "fmnist"),
                                  lt("K01_kaggle_cifar10_svhn__cifar10", "+HWA+LA", "cifar10"), lt("K01_kaggle_cifar10_svhn__svhn", "+HWA+LA (ours)", "svhn")], "#74c0fc"),
                  ("FSG + LA", [lt("F01_main_longtail", "FSG+LA", "mnist"), lt("F01_main_longtail", "FSG+LA", "fmnist"),
                                lt("K01_kaggle_cifar10_svhn__cifar10", "FSG+LA", "cifar10"), lt("K01_kaggle_cifar10_svhn__svhn", "FSG+LA", "svhn")], "#ffc078"),
                  ("Ours-A (T_s=10; CIFAR-10 T_s=1; SVHN: rows+MC+KH)",
                   [lt("E25_anchored_stack", "Ours-A Ts=10", "mnist"), lt("E25_anchored_stack", "Ours-A Ts=10", "fmnist"),
                    lt("K08_kaggle_anchored_stack__cifar10_lt", "Ours-A (PC+MC+KH+LA+CSL)", "cifar10"),
                    lt("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn")], OURS)],
                 "balanced accuracy (%)", (0, 102))

    # ablation: paired per-seed differences
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    srcs = set()
    for ax, ds in zip(axes, ["mnist", "fmnist"]):
        ref = get(store, C(f"K09+K14:{ds}", "Ours-A Ts=10", ds, LT))
        for i, lab in enumerate(_abl[1:] + ["Ours-A-B Ts=10"]):
            fold = f"E32+K14:{ds}" if lab == "Ours-A-B Ts=10" else f"K09+K14:{ds}"
            c = get(store, C(fold, lab, ds, LT))
            if c is None or ref is None:
                continue
            srcs.update(x.split(" :: ")[0] for x in c["src"].split(" + "))
            sh = sorted(set(c["seeds"]) & set(ref["seeds"]))
            d = [c["seeds"][s] - ref["seeds"][s] for s in sh]
            color = GOOD if lab == "Ours-A-B Ts=10" else BAD
            ax.scatter([i] * len(d), d, color=color, s=18, zorder=3)
            ax.hlines(np.mean(d), i - 0.25, i + 0.25, color=INK, lw=1.5)
        ax.axhline(0, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(6))
        ax.set_xticklabels(["−LA", "−CSL", "−MC", "−KH", "−PC", "MC→MC-B"], fontsize=8)
        ax.set_title({"mnist": "MNIST", "fmnist": "FashionMNIST"}[ds] + " long tail")
    axes[0].set_ylabel("change in balanced accuracy vs Ours-A (pts)\n(dots = seeds, bar = mean)")
    save(fig, "F04_ablation", "5.11", "Each component removed (red) and MC replaced by MC-B (green), paired by seed.",
         sorted(srcs))

    # synthetic budget (E16)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for lab, ts_lab, color in [("GeFL-F", {1: "GeFL-F", 3: "GeFL-F Ts=3", 5: "GeFL-F Ts=5"}, NEUT),
                               ("+ CSL", {1: "+CSL", 3: "+CSL Ts=3", 5: "+CSL Ts=5", 10: "+CSL Ts=10"}, "#74c0fc"),
                               ("Ours (HWA+LA+CSL)", {5: "Ours (HWA+LA+CSL) Ts=5", 10: "Ours (HWA+LA+CSL) Ts=10"}, OURS)]:
        cs = {t: get(store, C(_e16, l, "fmnist", IID, metric="best_mean_acc")) for t, l in ts_lab.items()}
        cs = {t: c for t, c in cs.items() if c}
        axes[0].errorbar(list(cs), [c["mean"] for c in cs.values()], [c["sd"] for c in cs.values()], marker="o", color=color,
                         label=lab, capsize=2)
    axes[0].axhline(84.28, color=BAD, lw=0.8, ls="--")
    axes[0].text(10, 84.33, "paper best (DDPM-F) 84.28", ha="right", fontsize=7, color=BAD)
    axes[0].set(xlabel="synthetic epochs T_s", ylabel="best_mean_acc (%)", title="FMNIST IID", xticks=[1, 3, 5, 10])
    axes[0].legend(frameon=False, fontsize=7.5)
    cs = {t: get(store, C(_e16, l, "fmnist", LT)) for t, l in {1: "Ours (HWA+LA+CSL)", 5: "Ours (HWA+LA+CSL) Ts=5", 10: "Ours (HWA+LA+CSL) Ts=10"}.items()}
    cs = {t: c for t, c in cs.items() if c}
    axes[1].errorbar(list(cs), [c["mean"] for c in cs.values()], [c["sd"] for c in cs.values()], marker="o", color=OURS, capsize=2, label="Ours")
    g = get(store, C(_e16, "GeFL-F", "fmnist", LT))
    if g:
        axes[1].axhline(g["mean"], color=NEUT, ls="--", lw=0.8)
        axes[1].text(10, g["mean"] + 0.4, f"GeFL-F {g['mean']:.1f}", ha="right", fontsize=7, color=MUTED)
    axes[1].set(xlabel="synthetic epochs T_s", ylabel="balanced accuracy (%)", title="FMNIST long tail", xticks=[1, 5, 10])
    save(fig, "F05_budget", "5.9", "Consensus labels make a larger synthetic budget useful (Proposition 3).", [_e16])

    # regimes (K02)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for ds, color in [("mnist", OURS), ("fmnist", "#e8590c")]:
        xs, gain, err = [], [], []
        for IF in [1.0, 0.1, 0.01]:
            a, b = get(store, C(_k02, "Ours", ds, IF)), get(store, C(_k02, "GeFL-F", ds, IF))
            if a and b:
                sh = sorted(set(a["seeds"]) & set(b["seeds"]))
                d = np.array([a["seeds"][s] - b["seeds"][s] for s in sh])
                xs.append(1 / IF)
                gain.append(d.mean())
                err.append(d.std(ddof=1) if len(d) > 1 else 0)
        ax.errorbar(xs, gain, err, marker="o", color=color, capsize=2, label={"mnist": "MNIST", "fmnist": "FashionMNIST"}[ds])
    ax.set(xscale="log", xlabel="imbalance factor (head/tail)", ylabel="gain over GeFL-F (pts, paired)",
           title="The gain grows with imbalance (K02, Dir 0.5)")
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["1", "10", "100"])
    ax.legend(frameon=False)
    save(fig, "F06_regimes", "5.6", "Gain of the minimal variant over GeFL-F against the imbalance factor.", [_k02])

    # clients
    fig, ax = plt.subplots(figsize=(5.8, 3.3))
    for name, specs, color in [
        ("GeFL-F", {10: C("F10_combined_method", "GeFL-F", "mnist", LT), 50: C("F04_clients", "GeFL-F", "mnist", LT, 50),
                    100: C("F04_clients", "GeFL-F", "mnist", LT, 100)}, NEUT),
        ("CVAE + HWA + LA", {10: C("F10_combined_method", "+HWA+LA", "mnist", LT), 50: C("F04_clients", "+HWA+LA", "mnist", LT, 50),
                             100: C("F04_clients", "+HWA+LA", "mnist", LT, 100)}, "#74c0fc"),
        ("FSG + LA", {10: C("F01_main_longtail", "FSG+LA", "mnist", LT), 50: C("F04_clients", "FSG+LA", "mnist", LT, 50),
                      100: C("F04_clients", "FSG+LA", "mnist", LT, 100)}, "#ffc078"),
        ("Ours-A, T_s=10", {10: C("E25_anchored_stack", "Ours-A Ts=10", "mnist", LT), 100: C("E22_pc_many_clients", "Ours-A Ts=10", "mnist", LT, 100)}, OURS)]:
        cs = {k: get(store, s) for k, s in specs.items()}
        cs = {k: c for k, c in cs.items() if c}
        ax.errorbar(list(cs), [c["mean"] for c in cs.values()], [c["sd"] for c in cs.values()], marker="o", color=color, capsize=2, label=name)
    ax.set(xscale="log", xlabel="clients K", ylabel="balanced accuracy (%)", title="MNIST long tail, more clients")
    ax.set_xticks([10, 50, 100])
    ax.set_xticklabels(["10", "50", "100"])
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F07_clients", "5.3 / 5.11", "Robustness to the number of clients (MNIST long tail).",
         ["F10_combined_method", "F04_clients", "F01_main_longtail", "E25_anchored_stack", "E22_pc_many_clients"])

    # privacy
    grouped_bars(store, "F08_privacy", "5.6", "Formal (eps, 1e-5)-DP on every anchoring statistic (E26), long tail, K = 10.",
                 ["MNIST", "FashionMNIST"],
                 [("GeFL-F (no formal privacy)", [lt("E20_proto_generator", "GeFL-F", "mnist"), lt("E20_proto_generator", "GeFL-F", "fmnist")], NEUT),
                  ("exact statistics", [lt("E25_anchored_stack", "Ours-A Ts=10", "mnist"), lt("E25_anchored_stack", "Ours-A Ts=10", "fmnist")], OURS),
                  ("eps = 8", [lt("E26_private_anchoring", "Ours-A Ts=10, DP eps=8", "mnist"), lt("E26_private_anchoring", "Ours-A Ts=10, DP eps=8", "fmnist")], "#74c0fc"),
                  ("eps = 2", [lt("E26_private_anchoring", "Ours-A Ts=10, DP eps=2", "mnist"), lt("E26_private_anchoring", "Ours-A Ts=10, DP eps=2", "fmnist")], "#a5d8ff")],
                 "balanced accuracy (%)", (40, 102))

    # SVHN gate
    k = "K11_kaggle_covariance_anchoring__svhn"
    grouped_bars(store, "F09_svhn", "5.11", "SVHN (class means carry no class information, F = 0.01): what restores class identity.",
                 ["long tail (balanced)", "IID (best_mean_acc)"],
                 [("GeFL-F", [lt("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn"), C("K01_kaggle_cifar10_svhn__svhn", "GeFL-F", "svhn", IID, metric="best_mean_acc")], NEUT),
                  ("Ours-A (mean anchor + MC)", [lt(k, "Ours-A (PC+MC+KH+LA+CSL)", "svhn"), C(k, "Ours-A (PC+MC+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc")], "#ffa8a8"),
                  ("mean anchor + MC-S", [lt(k, "Ours-A-S (PC+MCS+KH+LA+CSL)", "svhn"), C(k, "Ours-A-S (PC+MCS+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc")], "#ffc078"),
                  ("hybrid + MC-S", [lt(k, "Ours-R-S (PCR+MCS+KH+LA+CSL)", "svhn"), C(k, "Ours-R-S (PCR+MCS+KH+LA+CSL)", "svhn", IID, metric="best_mean_acc")], "#a5d8ff"),
                  ("rows (CVAE+HWA) + MC + KH", [lt("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn"),
                                                 C("K10_kaggle_hybrid_svhn_cifar10__svhn", "Ours+MC+KH (CVAE+HWA)", "svhn", IID, metric="best_mean_acc")], OURS)],
                 "accuracy (%)", (40, 85))

    collapse_figure()


def collapse_figure():
    """T01: the contraction theorem on the real CVAE-F (row norms, bound check)."""
    d = os.path.join(RES, "T01_collapse_threshold")
    runs = {}
    for js in sorted(glob.glob(os.path.join(d, "*.json"))):
        meta = json.load(open(js))
        runs[meta["label"]] = (meta, np.load(js[:-5] + ".npz"))
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    order = [l for l in ["GeFL-F wd=1e-3", "GeFL-F wd=1e-4", "GeFL-F wd=0", "HWA wd=1e-3"] if l in runs]
    colors = {"GeFL-F wd=1e-3": BAD, "GeFL-F wd=1e-4": "#f08c00", "GeFL-F wd=0": NEUT, "HWA wd=1e-3": OURS}
    for lab in order:
        meta, z = runs[lab]
        g = z["glob"]
        ratio = g[:, 6:].mean(1) / g[:, :3].mean(1)
        axes[0].plot(ratio, color=colors[lab], label=lab)
    axes[0].axhline(1, color=MUTED, lw=0.6, ls=":")
    axes[0].set(xlabel="generator round", ylabel="tail / head row norm", title="(a) Tail rows collapse; HWA prevents it")
    axes[0].legend(frameon=False, fontsize=7.5)
    if "GeFL-F wd=1e-3" in runs:
        meta, z = runs["GeFL-F wd=1e-3"]
        g, loc, counts = z["glob"], z["local"], z["counts"]
        bound = loc.mean(1)                                   # R x C
        actual = g[1:]
        axes[1].scatter(((bound - g[:-1]) / g[:-1]).ravel() * 100, ((actual - g[:-1]) / g[:-1]).ravel() * 100, s=5,
                        color=BAD, alpha=0.5)
        lim = np.abs(np.concatenate([((bound - g[:-1]) / g[:-1]).ravel(), ((actual - g[:-1]) / g[:-1]).ravel()])).max() * 100
        axes[1].plot([-lim, lim], [-lim, lim], color=INK, lw=0.8)
        axes[1].set(xlabel="bound: change of ||r_c|| allowed (%)", ylabel="actual change of ||r_c|| (%)",
                    title="(b) Theorem 1 holds in every round")
        m = (counts > 0).sum(0)
        rel = g[-1] / g[0]
        axes[2].scatter(m, rel, color=BAD, zorder=3, label="flat FedAvg")
        if "HWA wd=1e-3" in runs:
            gh = runs["HWA wd=1e-3"][1]["glob"]
            axes[2].scatter(m + 0.12, gh[-1] / gh[0], color=OURS, zorder=3, marker="s", s=22, label="HWA")
        axes[2].legend(frameon=False, fontsize=7.5, loc="center left")
        for c in range(len(m)):
            axes[2].annotate(str(c), (m[c], rel[c]), textcoords="offset points", xytext=(4, 2), fontsize=7)
        axes[2].axvline(meta["K"] / 2, color=MUTED, ls="--", lw=0.8)
        axes[2].text(meta["K"] / 2, rel.max(), " old rule: K/2", fontsize=7, color=MUTED, va="top")
        axes[2].axhline(1, color=MUTED, lw=0.6, ls=":")
        axes[2].set(xlabel="holders m_c of class c", ylabel="row norm end / start",
                    title="(c) Tail rows shrink although m_c > K/2")
    save(fig, "F01_collapse", "3", "The conditioning-row contraction theorem checked on the paper's CVAE-F (T01, MNIST LT, K = 10).",
         ["T01_collapse_threshold"])


# ------------------------------------------------------------------ index and check
def write_index(allcells, mism):
    lines = ["# Paper assets (generated by `python experiments/paper_assets.py`; do not edit)", "",
             "Every table and figure of the paper, computed from `experiments/results` only. "
             "`numbers.csv` gives each cell's mean, sd, seeds, p-value and source. `CHECK.md` lists draft numbers "
             "that do not match the data.", "", "## Tables", "", "| id | paper § | title | missing cells |", "|---|---|---|---|"]
    for t in TABLES:
        lines.append(f"| [{t['id']}](tables/{t['id']}.md) | {t['section']} | {t['title']} | {len(t['missing']) or ''} |")
    lines += ["", "## Figures", "", "| id | paper § | caption | sources |", "|---|---|---|---|"]
    for f in FIGS:
        lines.append(f"| [{f['id']}](figures/{f['id']}.png) | {f['section']} | {f['caption']} | {', '.join(f['sources'])} |")
    open(os.path.join(OUT, "INDEX.md"), "w").write("\n".join(lines) + "\n")
    chk = ["# Draft numbers that differ from the data", "",
           "Generated by `paper_assets.py`. Each row: the value quoted in docs/paper_draft.md vs the value in the results. "
           "Fix the draft (or explain the difference) before submission.", "",
           "| table | row | column | draft | data | source |", "|---|---|---|---|---|---|"]
    chk += [f"| {a} | {b} | {c} | {d} | {e:.2f} | {s} |" for a, b, c, d, e, s in mism]
    if not mism:
        chk.append("| – | none | | | | |")
    open(os.path.join(OUT, "CHECK.md"), "w").write("\n".join(chk) + "\n")


def main():
    store = Store()
    allcells, mism = build_tables(store)
    figures(store)
    write_index(allcells, mism)
    print(f"{len(TABLES)} tables, {len(FIGS)} figures, {len(allcells)} cells, {len(mism)} draft mismatches -> {OUT}")


if __name__ == "__main__":
    main()
