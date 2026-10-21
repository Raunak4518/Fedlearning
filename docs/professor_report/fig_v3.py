"""Figures added in handout version 3 (21 Oct 2026). Every number is read from the result files:
experiments/results/*/runs.jsonl, docs/paper_assets/tables/T22_fair_retest.md and docs/research/math_checks/*.json(l).
Run: python docs/professor_report/fig_v3.py"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "experiments", "results")
MC = os.path.join(REPO, "docs", "research", "math_checks")
OUT = os.path.join(HERE, "fig")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED = "#1f2a37", "#5b6676"
GOOD, BAD, NEUT, OURS, WARN = "#2f9e44", "#c92a2a", "#868e96", "#1c7ed6", "#e8590c"
DS = ["mnist", "fmnist", "svhn", "cifar10", "cifar100"]
DSN = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}


def runs(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern, "runs.jsonl"))):
        out += [json.loads(l) for l in open(f)]
    return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# v01: collapse predicted from counts alone (kappa), GeFL runs + the prospective SVHN test
def v01():
    R = json.load(open(os.path.join(MC, "n6c_rows.json")))
    S = json.load(open(os.path.join(MC, "x2_kappa_svhn.json")))["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    cols = {"mnist": "#1c7ed6", "fmnist": "#7048e8", "cifar10": "#e8590c", "cifar100": "#868e96"}
    for ds, c in cols.items():
        k = np.array([r["kappa"] if r["kappa"] is not None else 300 for r in R if r["ds"] == ds])
        n = np.array([r["norm"] for r in R if r["ds"] == ds])
        ax[0].scatter(k, n, s=6, alpha=0.5, color=c, label=DSN[ds])
    ax[0].axvline(1, color=BAD, lw=1, ls="--")
    ax[0].axhline(0.15, color=MUTED, lw=0.8, ls=":")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("κ (computed from client class counts before training; 300 = no client lacks the class)")
    ax[0].set_ylabel("final / initial row length")
    lo = sum(1 for r in R if r["kappa"] is not None and r["kappa"] < 1)
    ax[0].set_title(f"(a) 900 class rows, GeFL-F runs: all {lo} rows with κ < 1 collapsed", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower right", markerscale=2)
    for lab, c, mk in [("GeFL-F", BAD, "o"), ("GeFL-F + HWA + LA", GOOD, "s")]:
        k = np.array([r["kappa"] if np.isfinite(r["kappa"]) else 300 for r in S[lab]])
        n = np.array([r["norm"] for r in S[lab]])
        ax[1].scatter(k, n, s=14, alpha=0.7, color=c, marker=mk, label=lab)
    ax[1].axvline(1, color=BAD, lw=1, ls="--"); ax[1].axhline(0.15, color=MUTED, lw=0.8, ls=":")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("κ"); ax[1].set_ylabel("final / initial row length")
    k = np.array([r["kappa"] for r in S["GeFL-F"]]); n = np.array([r["norm"] for r in S["GeFL-F"]])
    ax[1].set_title(f"(b) prediction fixed before the runs: SVHN, 50 / 100 clients\n"
                    f"{int(((k < 1) & (n <= 0.15)).sum())} of {int((k < 1).sum())} predicted rows collapsed; none under HWA",
                    fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")
    save(fig, "v01_kappa.png")


# v02: fair re-test, Ours minus every baseline (validation-selected readouts, new seeds)
def v02():
    txt = open(os.path.join(REPO, "docs", "paper_assets", "tables", "T22_fair_retest.md")).read()
    blocks = re.split(r"\*\*(\w+)\*\*: selected", txt)[1:]
    data = {}
    for ds, body in zip(blocks[0::2], blocks[1::2]):
        rows = re.findall(r"\| ([^|]+?) \| \w+ \| [\d.]+ \| ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\] \| ([\d.e-]+) \|", body)
        data[ds] = [(b.strip(), float(d), float(l), float(h), float(p)) for b, d, l, h, p in rows]
    keep = ["GeFL-F", "GeFL-F + AdamW + LA", "Real only + LA", "FSG + LA", "FedDF", "FedDF + LA"]
    colors = dict(zip(keep, ["#adb5bd", "#e8590c", "#7048e8", "#0c8599", "#c92a2a", "#f08c00"]))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.13
    for j, b in enumerate(keep):
        for i, ds in enumerate(DS):
            row = [r for r in data.get(ds, []) if r[0] == b]
            if not row:
                continue
            _, d, l, h, p = row[0]
            x = i + (j - (len(keep) - 1) / 2) * w
            ax.bar(x, d, w * 0.9, color=colors[b], label=b if i == 0 else None, edgecolor="black" if p < 0.05 else "none",
                   lw=0.8)
            ax.plot([x, x], [l, h], color=INK, lw=0.8)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([DSN[d] for d in DS])
    ax.set_ylabel("Ours − baseline (points, test)")
    ax.set_ylim(-3, 6)
    ax.set_title("Fair re-test: new seeds 3–5, every method's readout chosen on validation data; "
                 "bars with black edge: Holm-adjusted p < 0.05; lines: 95% CI (SVHN real-only bars ≈ +11, cut)", fontsize=8.5)
    ax.legend(fontsize=7, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save(fig, "v02_fair_retest.png")


# v03: the synthetic phase helps only when the generated features carry their label (signal fraction)
def v03():
    R = runs("K17_kaggle_fair_retest*")
    by = {(r["label"], r["dataset"], r["seed"]): r for r in R}
    arms = [("GeFL-F", BAD), ("GeFL-F + AdamW + LA", WARN), ("Ours Ts=10", OURS)]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    mk = {"mnist": "o", "fmnist": "s", "svhn": "^", "cifar10": "D", "cifar100": "v"}
    for lab, c in arms:
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
                ss.append((np.mean(f) - 1 / C) / (1 - 1 / C)); gs.append(100 * (r["er_bal"] - z["er_bal"]))
            if ss:
                ax.scatter(np.mean(ss), np.mean(gs), color=c, marker=mk[ds], s=40, edgecolor="white", lw=0.5)
    for lab, c in arms:
        ax.scatter([], [], color=c, label=lab)
    for ds in DS:
        ax.scatter([], [], color=NEUT, marker=mk[ds], label=DSN[ds])
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1.5)
    ax.set_xlabel("signal fraction s of the generated features (referee, 0 = carry no label)")
    ax.set_ylabel("gain of the synthetic phase\n(exact-readout accuracy minus real-data-only heads)")
    ax.set_title("Synthetic data hurt when they carry no label (Proposition 9; K17, seeds 3–5)", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)
    save(fig, "v03_signal_fraction.png")


# v04: federated vs centralised (pooled-body comparator), long tail and balanced
def v04():
    def n7(f, fe, key):
        for l in open(os.path.join(MC, f)):
            d = json.loads(l)
            if d.get("fe") == fe:
                v = d[key]
                return v["bal"] if isinstance(v, dict) else v
    k16 = {}
    for r in runs("K16_kaggle_recalibration__*"):
        if r["label"].startswith("Ours") and r["seed"] == 0:
            k16[(r["dataset"], "lt" if r["IF"] == 0.01 else "iid")] = 100 * r["er_bal"]
    rows = {
        "MNIST": dict(one=n7("n7_mnist_s0.jsonl", "fed", "one_client_ER"), ours=k16[("mnist", "lt")],
                      pooled=n7("n7_mnist_s0.jsonl", "fed", "pooled_ER"), cen=n7("n7_mnist_s0.jsonl", "cen_bal", "pooled_ER"),
                      iid=n7("n7_mnist_s0.jsonl", "fed_iid", "pooled_ER"), ours_iid=k16[("mnist", "iid")]),
        "FashionMNIST": dict(one=n7("n7_fmnist_s0.jsonl", "fed", "one_client_ER"), ours=k16[("fmnist", "lt")],
                             pooled=n7("n7_fmnist_s0.jsonl", "fed", "pooled_ER"), cen=n7("n7_fmnist_s0.jsonl", "cen_bal", "pooled_ER"),
                             iid=n7("n7_fmnist_s0.jsonl", "fed_iid", "pooled_ER"), ours_iid=k16[("fmnist", "iid")]),
        "CIFAR-100*": dict(one=n7("n7c_cifar100_s0.jsonl", "fed", "one_client_ER"), ours=k16[("cifar100", "lt")],
                           pooled=n7("n7c_cifar100_s0.jsonl", "fed", "pooled_ER"), cen=n7("n7c_cifar100_s0.jsonl", "cen_bal", "pooled_ER"),
                           iid=n7("n7c_cifar100_s0.jsonl", "fed_iid", "pooled_ER"), ours_iid=k16[("cifar100", "iid")]),
    }
    keys = [("one", "federated: body trained on its own client only", "#adb5bd"),
            ("ours", "federated: Ours (synthetic data from the other clients)", OURS),
            ("pooled", "centralised: body trained on all clients' data", GOOD),
            ("cen", "centralised: + class-balanced extractor", "#2b8a3e"),
            ("ours_iid", "Ours on balanced data", "#74c0fc"),
            ("iid", "centralised on balanced data", "#b2f2bb")]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for a, (name, v) in zip(ax, rows.items()):
        vals = [v[k] for k, _, _ in keys]
        a.bar(range(len(keys)), vals, color=[c for _, _, c in keys])
        for i, x in enumerate(vals):
            a.text(i, x + 0.2, f"{x:.1f}", ha="center", fontsize=7)
        lo = min(vals) - 3
        a.set_ylim(lo, max(vals) + 2)
        a.set_xticks([]); a.set_title(name, fontsize=9)
    ax[0].set_ylabel("balanced accuracy, exact readout (%)")
    ax[1].legend(handles=[Patch(color=c, label=lab) for _, lab, c in keys], fontsize=7, ncol=3, loc="upper center",
                 bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Where federated training loses accuracy (long tail, seed 0; *CIFAR-100 centralised arms on 3 of 10 architectures)",
                 fontsize=9, y=1.02)
    save(fig, "v04_centralised.png")


# v05: what happens at each aggregation (instrumented runs, K25)
def v05():
    R = {(r["label"].split(" [")[0], r["dataset"]): r for r in runs("K25_instrumented")}
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, ds in zip(ax[:2], ["mnist", "cifar10"]):
        for j, (lab, c) in enumerate([("GeFL-F", BAD), ("GeFL-F + AdamW + LA", GOOD)]):
            r = R.get((lab, ds))
            if not r or not r.get("inst_rows"):
                continue
            rows = r["inst_rows"][5:]
            G = np.mean([x["holder_gain"] for x in rows], 0); D = np.mean([x["nonholder_loss"] for x in rows], 0)
            x = np.arange(len(G))
            a.bar(x + (j - 0.5) * 0.4, D, 0.38, color=c, alpha=0.9)
            a.plot(x + (j - 0.5) * 0.4, G, "_", color="black", ms=8, mew=1.5)
        a.set_xticks(range(10)); a.set_xlabel("class (0 = most common)")
        a.set_title(f"{DSN[ds]}: change of class-row length per round", fontsize=9)
    ax[0].set_ylabel("mean change per round")
    ax[0].set_ylim(0, 0.68)
    ax[0].legend(handles=[Patch(color=BAD, label="GeFL-F: shrink by clients without the class"),
                          Patch(color=GOOD, label="decay fixed (AdamW): shrink ≈ 0"),
                          Line2D([], [], color="black", marker="_", ls="none", ms=8, mew=1.5,
                                 label="growth by clients with the class")], fontsize=7, loc="upper left")
    for lab, c in [("GeFL-F", BAD), ("GeFL-F + AdamW + LA", WARN), ("Ours Ts=1", OURS)]:
        r = R.get((lab, "cifar10"))
        if not r:
            continue
        ph = r["inst_phases"]
        ax[2].plot([p["round"] for p in ph], [p["spread_after_syn"] / p["spread_before"] for p in ph], "-o", ms=3, color=c,
                   label=lab.replace("Ours Ts=1", "Ours"))
    ax[2].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[2].set_xlabel("Stage-III round"); ax[2].set_ylabel("logit spread after / before")
    ax[2].set_title("CIFAR-10: synthetic phase flattens the logits", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "v05_instrumentation.png")


# v06: beyond GeFL, federated matrix factorisation on MovieLens-1M (E40)
def v06():
    X = json.load(open(os.path.join(MC, "x5_step_frequency_mf.json")))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    centres = [np.sqrt(max(b["lo"], 1e-4) * b["hi"]) for b in X["central"]["bins"]]
    ax.plot(centres, [b["median_norm"] for b in X["central"]["bins"]], "-o", color=INK, label="centralised Adam + L2")
    for K, c in [(10, WARN), (50, BAD)]:
        bs = X[f"federated_K{K}"]["bins"]
        ax.plot(centres, [b["median_norm"] for b in bs], "-s", color=c, label=f"federated, {K} clients")
    ax.axvline(1 / 1000, color=MUTED, ls=":", lw=0.8)
    ax.text(1 / 1000, 0.75, " Adam's memory\n (1 / 1,000 steps)", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("share of optimiser steps whose batch contains the item")
    ax.set_ylabel("final / initial item-embedding length (median)")
    ax.set_title("MovieLens-1M, local Adam + coupled L2 (seed 0)", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "v06_movielens.png")


if __name__ == "__main__":
    for f in (v01, v02, v03, v04, v05, v06):
        f()
