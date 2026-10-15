"""Figures for Part 7 of the handout ("Why it fails, and why each fix works"), w01-w07.
Numbers: experiments/results/T01_collapse_threshold, D01 evidence (docs/professor_report/fig_evidence.py),
E16/E22 (synthetic budget), K09 (ablation), T03 (last-layer gap). Typed here from those summaries."""
import matplotlib.pyplot as plt
import numpy as np
from make_figures import save, INK, MUTED, BAD, GOOD, NEUT, OURS

ORANGE = "#e8590c"


def _clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def _bars(ax, labels, series, colors, names=None, fmt="{:.0f}", ylim=None, title=None, ylabel=None):
    x = np.arange(len(labels))
    k = len(series)
    w = 0.8 / k
    for i, (vals, c) in enumerate(zip(series, colors)):
        pos = x - 0.4 + w / 2 + i * w
        ax.bar(pos, vals, w * 0.92, color=c, label=names[i] if names else None)
        for p, v in zip(pos, vals):
            ax.text(p, v + (ylim[1] - ylim[0]) * 0.015 if ylim else v, fmt.format(v), ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, labels, fontsize=7.5)
    if ylim:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    _clean(ax)


def w01():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    arms = ["normal\n(decay 0.001)", "decay 10×\nsmaller", "decay\nswitched off", "only holders\naveraged"]
    vals = {"class 7 (67 samples)": [10, 10, 120, 125], "class 8 (40)": [4, 4, 114, 119], "class 9 (24)": [9, 9, 112, 117]}
    _bars(a, arms, list(vals.values()), [BAD, "#e03131aa", "#ffa8a8"], list(vals.keys()), ylim=(0, 145),
          title="(a) Rare-class lists after 40 rounds, % of start (MNIST)", ylabel="% of starting length")
    a.axhline(100, color=MUTED, lw=0.8, ls=":")
    a.legend(fontsize=6.5, frameon=False, loc="upper left")
    meth = ["original\nmethod", "+ holder\naveraging", "mean-\nanchored\ngenerator", "Ours-A", "real data\n(ceiling)"]
    _bars(b, meth, [[10, 31, 93, 90, 97], [20, 71, 88, 79, 84]], [OURS, ORANGE], ["MNIST", "FashionMNIST"], ylim=(0, 112),
          title="(b) Rare-class fakes the judge labels correctly (%)")
    b.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "w01_erasing.png")


def w02():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.8))
    _bars(a, ["mean-anchored\ngenerator", "+ moment\ncorrection", "+ sample\nselection"], [[0.47, 0.85, 0.95], [0.45, 0.90, 0.98]],
          [OURS, ORANGE], ["MNIST", "FashionMNIST"], fmt="{:.2f}", ylim=(0, 1.25), title="(a) Spread of fakes ÷ spread of real data")
    a.axhline(1, color=GOOD, lw=1, ls="--"); a.text(-0.4, 1.03, "real data", color=GOOD, fontsize=7)
    a.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2)
    _bars(b, ["original\nmethod", "mean-anchored\ngenerator", "Ours-A", "real data"], [[0.09, 0.78, 0.32, 0.30], [0.32, 1.35, 0.61, 0.60]],
          [OURS, ORANGE], ["MNIST", "FashionMNIST"], fmt="{:.2f}", ylim=(0, 1.55), title="(b) How cleanly classes are separated")
    b.legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "w02_narrow.png")


def w03():
    fig, ax = plt.subplots(figsize=(6, 2.6))
    _bars(ax, ["original\nmethod", "mean-anchored\ngenerator", "+ moment\ncorrection", "+ sample\nselection"],
          [[0.22, 0.20, 0.089, 0.071], [0.25, 0.19, 0.093, 0.078]], [OURS, ORANGE], ["MNIST", "FashionMNIST"],
          fmt="{:.3f}", ylim=(0, 0.3), title="Distance between fake and real features (lower = fakes closer to real)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "w03_shape.png")


def w04():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.6))
    _bars(a, ["MNIST", "FashionMNIST"], [[0.64, 0.67], [0.44, 0.52]], [NEUT, OURS], ["without logit adjustment", "with"],
          fmt="{:.2f}", ylim=(0, 0.85), title="(a) Client's class shares vs its predictions (correlation)")
    a.legend(fontsize=6.5, frameon=False)
    _bars(b, ["MNIST", "FashionMNIST"], [[92.0, 75.5], [95.2, 79.9]], [NEUT, OURS], ["without", "with"],
          fmt="{:.1f}", ylim=(60, 100), title="(b) Balanced accuracy (%), seed 0")
    b.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "w04_skew.png")


def w06():
    fig, ax = plt.subplots(figsize=(6, 2.5))
    labs = ["original generator\n(MNIST, 10 clients)", "our generator\n(MNIST, 10 clients)", "our generator\n(MNIST, 100 clients)"]
    vals = [-0.05, 1.71, 3.9]
    ax.bar(range(3), vals, 0.55, color=[BAD, OURS, OURS])
    for i, v in enumerate(vals):
        ax.text(i, v + (0.12 if v >= 0 else -0.35), f"{v:+.2f}", ha="center", fontsize=7.5)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(3), labs, fontsize=7.5)
    ax.set_ylim(-0.8, 4.6)
    ax.set_title("Gain from 10 fake epochs per round instead of 1 (points)", fontsize=8.5, loc="left")
    _clean(ax)
    fig.tight_layout()
    save(fig, "w06_budget.png")


def w07():
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 2.7))
    ds = ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"]
    v = [82, 69, 41, 12]
    a.bar(range(4), v, 0.55, color=[OURS, OURS, OURS, BAD])
    for i, x in enumerate(v):
        a.text(i, x + 2, f"{x}", ha="center", fontsize=7.5)
    a.axhline(10, color=MUTED, ls="--", lw=0.8); a.text(2.45, 13, "chance (10)", fontsize=7, color=MUTED, ha="right")
    a.set_xticks(range(4), ds, fontsize=7.5); a.set_ylim(0, 100)
    a.set_title("(a) Guessing the class from the nearest class average (%)", fontsize=8.5, loc="left"); _clean(a)
    m = ["original\nmethod", "Ours-A\n(mean anchor)", "Ours-A\nwith switch"]
    w = [50.5, 54.4, 65.6]
    b.bar(range(3), w, 0.55, color=[NEUT, "#ffa8a8", OURS])
    for i, x in enumerate(w):
        b.text(i, x + 1, f"{x}", ha="center", fontsize=7.5)
    b.set_xticks(range(3), m, fontsize=7.5); b.set_ylim(40, 72)
    b.set_title("(b) SVHN long tail, balanced accuracy (%)", fontsize=8.5, loc="left"); _clean(b)
    fig.tight_layout()
    save(fig, "w07_svhn.png")


def w08():
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    groups = ["class missing\nat the client", "held, 1–20\nsamples", "held, more than\n20 samples"]
    _bars(ax, groups, [[74.4, 19.8, -2.2], [28.2, 7.7, -1.4], [17.4, 7.2, -1.4]], [NEUT, "#74c0fc", OURS],
          ["original method", "+ holder averaging + logit adjustment", "Ours-A"], fmt="{:.0f}", ylim=(-10, 85),
          title="Accuracy points lost in the last layer, by how much of the class the client has (FashionMNIST)")
    ax.axhline(0, color=INK, lw=0.8)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, "w08_lastlayer.png")


if __name__ == "__main__":
    w01(); w02(); w03(); w04(); w06(); w07(); w08()
