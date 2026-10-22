"""Figures of the LaTeX paper (docs/latex/paper.tex). Every number is read from the result files.
Run: python docs/latex/figures.py  -> docs/latex/fig_kappa.pdf, docs/latex/fig_signal.pdf"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "experiments", "results")
MC = os.path.join(REPO, "docs", "research", "math_checks")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42})
DS = ["mnist", "fmnist", "svhn", "cifar10", "cifar100"]
DSN = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}
NOHOLD = 300.0  # plotted position of rows that no client lacks (kappa = infinity)


def kappa_figure():
    R = json.load(open(os.path.join(MC, "n6c_rows.json")))
    S = json.load(open(os.path.join(MC, "x2_kappa_svhn.json")))["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    cols = {"mnist": "#1c7ed6", "fmnist": "#7048e8", "cifar10": "#e8590c", "cifar100": "#495057"}
    for ds, c in cols.items():
        k = np.array([r["kappa"] if r["kappa"] is not None else NOHOLD for r in R if r["ds"] == ds])
        n = np.array([r["norm"] for r in R if r["ds"] == ds])
        ax[0].scatter(k, n, s=4, alpha=0.45, color=c, lw=0, label=DSN[ds])
    for lab, c, mk, name in [("GeFL-F", "#c92a2a", "o", "flat FedAvg (GeFL-F)"),
                             ("GeFL-F + HWA + LA", "#2f9e44", "s", "holder-weighted (HWA)")]:
        k = np.array([r["kappa"] if np.isfinite(r["kappa"]) else NOHOLD for r in S[lab]])
        n = np.array([r["norm"] for r in S[lab]])
        ax[1].scatter(k, n, s=9, alpha=0.75, color=c, marker=mk, lw=0, label=name)
    for a in ax:
        a.axvline(1, color="#c92a2a", lw=0.7, ls="--")
        a.axhline(0.15, color="#868e96", lw=0.6, ls=":")
        a.set_xscale("log")
        a.set_yscale("log")
        a.legend(fontsize=6.5, loc="lower right", markerscale=2, handletextpad=0.2)
    ax[0].set_ylabel("final / initial row norm")
    lo = sum(1 for r in R if r["kappa"] is not None and r["kappa"] < 1)
    ax[0].set_title("(a) retrospective: GeFL-F runs, flat FedAvg", fontsize=7.5)
    k = np.array([r["kappa"] for r in S["GeFL-F"]])
    n = np.array([r["norm"] for r in S["GeFL-F"]])
    print(f"(a) {len(R)} rows, {lo} with kappa < 1, max final norm among them "
          f"{max(r['norm'] for r in R if r['kappa'] is not None and r['kappa'] < 1):.3f}; "
          f"(b) {int(((k < 1) & (n <= 0.15)).sum())} of {int((k < 1).sum())} predicted collapses")
    ax[1].set_title("(b) prospective: SVHN, $K=50/100$", fontsize=7.5)
    fig.supxlabel(r"$\kappa_c$ computed from client class counts before training (right edge: no client lacks $c$)",
                  fontsize=8, y=0.06)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(HERE, "fig_kappa.pdf"), bbox_inches="tight")
    plt.close(fig)


def signal_figure():
    R = []
    for f in sorted(glob.glob(os.path.join(RES, "K17_kaggle_fair_retest*", "runs.jsonl"))):
        R += [json.loads(l) for l in open(f)]
    by = {(r["label"], r["dataset"], r["seed"]): r for r in R}
    arms = [("GeFL-F", "#c92a2a", "GeFL-F"), ("GeFL-F + AdamW + LA", "#e8590c", "GeFL-F, decoupled decay"),
            ("Ours Ts=10", "#1c7ed6", "exact-statistics generator")]
    mk = {"mnist": "o", "fmnist": "s", "svhn": "^", "cifar10": "D", "cifar100": "v"}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for lab, c, _ in arms:
        for ds in DS:
            ss, gs = [], []
            for sd in (3, 4, 5):
                r, z = by.get((lab, ds, sd)), by.get(("Real only", ds, sd))
                if not r or not z or r.get("er_bal") is None:
                    continue
                f = r.get("fidelity_mc") or r.get("fidelity")
                if not f:
                    continue
                C = len(f)
                ss.append((np.mean(f) - 1 / C) / (1 - 1 / C))
                gs.append(100 * (r["er_bal"] - z["er_bal"]))
            if ss:
                ax.scatter(np.mean(ss), np.mean(gs), color=c, marker=mk[ds], s=22, edgecolor="white", lw=0.4)
    for lab, c, name in arms:
        ax.scatter([], [], color=c, s=14, label=name)
    for ds in DS:
        ax.scatter([], [], color="#adb5bd", marker=mk[ds], s=14, label=DSN[ds])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1.5)
    ax.set_ylim(-3.5, 14)
    ax.set_xlabel("signal fraction $s$ of the generated features")
    ax.set_ylabel("synthetic gain (ER points)")
    ax.legend(fontsize=5.6, ncol=2, loc="upper left", handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig_signal.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    kappa_figure()
    signal_figure()
    print("wrote fig_kappa.pdf, fig_signal.pdf")
