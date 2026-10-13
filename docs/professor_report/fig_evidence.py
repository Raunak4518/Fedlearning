"""Evidence figures from real runs (experiments/results/D01_component_evidence, K09 ablation).

    python fig_evidence.py [D01 results dir] [K09 results dir ...]

Every figure is one claim of the handout, measured; mean ± sd over the seeds found.
"""
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig")
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "experiments", "results"))
D01 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "D01_component_evidence")
K09 = sys.argv[2:] or sorted(glob.glob(os.path.join(ROOT, "K09_kaggle_ablation_final*")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
INK, MUTED, GOOD, BAD, OURS, NEUT = "#1f2a37", "#5b6676", "#2f9e44", "#c92a2a", "#1c7ed6", "#868e96"
COL = {"CVAE-F": NEUT, "CVAE-F+HWA": "#74c0fc", "PC-VAE": "#f59f00", "PC-VAE+MC": "#e8590c",
       "PC-VAE+MC+KH": OURS, "real (test)": GOOD}
LAB = {"CVAE-F": "CVAE-F (GeFL-F)", "CVAE-F+HWA": "CVAE-F + HWA", "PC-VAE": "PC-VAE", "PC-VAE+MC": "PC-VAE + MC",
       "PC-VAE+MC+KH": "PC-VAE + MC + KH (Ours-A)", "real (test)": "real test features"}
DSN = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}
CLS = np.arange(10)


def load():
    runs = {}
    for f in sorted(glob.glob(os.path.join(D01, "*_s[0-9].json"))):
        r = json.load(open(f, encoding="utf-8"))
        runs.setdefault(r["dataset"], []).append(r)
    return runs


def stat(rs, name, key):
    a = np.array([r["metrics"][name][key] for r in rs if name in r.get("metrics", {})], dtype=float)
    return a.mean(0), (a.std(0) if len(a) > 1 else np.zeros_like(a.mean(0))), len(a)


def style(ax, ylabel=None, title=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", lw=0.6)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")


def tailband(ax):
    ax.axvspan(5.5, 9.5, color="#fff5f5", zorder=0)
    ax.text(7.5, ax.get_ylim()[1], "rare classes", ha="center", va="bottom", fontsize=7.5, color=BAD)


def grouped(ax, rs, names, key, ylabel, title, ref=None, ylim=None):
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        m, s, n = stat(rs, nm, key)
        if n == 0:
            continue
        ax.bar(CLS - 0.4 + w * (i + 0.5), m, w * 0.92, yerr=s if n > 1 else None, color=COL[nm], label=LAB[nm],
               error_kw=dict(lw=0.6, capsize=1.5))
    if ref is not None:
        ax.axhline(ref, color=GOOD, ls="--", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(CLS, [f"{c}" for c in CLS])
    ax.set_xlabel("class (0 = most samples, 9 = fewest)")
    style(ax, ylabel, title)
    tailband(ax)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def seeds_note(rs):
    return f"{len(rs)} seed{'s' if len(rs) > 1 else ''}"


# ------------------------------------------------ E1: class-row collapse (Proposition 1)
def fig_collapse(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs and "norms_CVAE-F" in runs[d][0]]
    if not dss:
        return
    fig, axs = plt.subplots(len(dss), 3, figsize=(10, 3.0 * len(dss)), squeeze=False)
    for row, ds in enumerate(dss):
        r0 = runs[ds][0]
        hold = np.array(r0["holders"])
        for col, nm in enumerate(("CVAE-F", "CVAE-F+HWA")):
            ax = axs[row, col]
            N = np.array(r0[f"norms_{nm}"])
            N = N / N[0:1]
            cm = plt.get_cmap("coolwarm_r")
            for c in range(10):
                ax.plot(N[:, c], color=cm(hold[c] / 10), lw=1.6 if c >= 6 else 1.0,
                        label=f"class {c}: {hold[c]}/10 clients")
            ax.axhline(1, color=MUTED, lw=0.6, ls=":")
            style(ax, "row norm / its start value", f"{DSN[ds]}: {LAB[nm]} class rows")
            ax.set_xlabel("Stage II round")
            if col == 0:
                ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper left")
        ax = axs[row, 2]
        for nm, mk in (("CVAE-F", "o"), ("CVAE-F+HWA", "s")):
            xs, ys = [], []
            for r in runs[ds]:
                if f"norms_{nm}" not in r:
                    continue
                N = np.array(r[f"norms_{nm}"])
                xs += list(np.array(r["holders"]) / 10)
                ys += list(N[-1] / N[0])
            ax.scatter(xs, ys, marker=mk, s=22, color=COL[nm], label=LAB[nm], edgecolor="white", lw=0.4)
        ax.axvline(0.5, color=BAD, ls="--", lw=1)
        ax.text(0.27, ax.get_ylim()[1] * 0.92, "Prop. 1: shrinks\nwhen m_c/K < ½", color=BAD, fontsize=7, ha="center", va="top")
        ax.axhline(1, color=MUTED, lw=0.6, ls=":")
        style(ax, "final / initial row norm", f"{DSN[ds]}: end of Stage II vs share of holders ({seeds_note(runs[ds])})")
        ax.set_xlabel("share of clients holding the class, m_c / K")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Claim: under FedAvg the learned class rows of rare classes shrink every round; HWA stops it",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    save(fig, "ev01_collapse.png")


# ------------------------------------------------ E2: class fidelity of fake features
def fig_fidelity(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs]
    if not dss:
        return
    fig, axs = plt.subplots(len(dss), 1, figsize=(10, 2.9 * len(dss)), squeeze=False)
    names = ["CVAE-F", "CVAE-F+HWA", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"]
    for i, ds in enumerate(dss):
        ax = axs[i, 0]
        ref = np.mean([np.mean(r["metrics"]["real (test)"]["fid"]) for r in runs[ds]])
        grouped(ax, runs[ds], names, "fid", "referee says 'class c'", None, ref=ref, ylim=(0, 1.08))
        style(ax, "fake samples recognised\nas their class",
              f"{DSN[ds]} ({seeds_note(runs[ds])}); dashed: real test features")
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.13))
    fig.suptitle("Claim: GeFL-F's fake features of rare classes do not look like their class; the exact-mean anchor fixes it\n"
                 "(referee = a head trained on held-out REAL features; it labels 1,000 fake samples per class)",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5, y=1.07)
    fig.tight_layout()
    save(fig, "ev02_fidelity.png")


# ------------------------------------------------ E3: moments (MC)
def fig_moments(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs]
    if not dss:
        return
    fig, axs = plt.subplots(len(dss), 2, figsize=(10, 2.8 * len(dss)), squeeze=False)
    names = ["CVAE-F", "PC-VAE", "PC-VAE+MC", "real (test)"]
    for i, ds in enumerate(dss):
        grouped(axs[i, 0], runs[ds], names, "spread", "spread / real spread", f"{DSN[ds]}: spread (1 = real)")
        axs[i, 0].axhline(1, color=INK, lw=0.8, ls=":")
        grouped(axs[i, 1], runs[ds], names, "mean_err", "‖fake mean − m_c‖ / class std",
                f"{DSN[ds]}: error of the class mean (0 = exact)")
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.13))
    fig.suptitle("Claim: the VAE's samples are too narrow and their means are off; MC gives the exact mean and spread",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5, y=1.05)
    fig.tight_layout()
    save(fig, "ev03_moments.png")


# ------------------------------------------------ E4: distribution match (KH) and Lemma 1
def fig_mmd(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs]
    if not dss:
        return
    fig, axs = plt.subplots(len(dss), 2, figsize=(10, 2.9 * len(dss)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.6, 1]))
    names = ["CVAE-F", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"]
    for i, ds in enumerate(dss):
        ax = axs[i, 0]
        grouped(ax, runs[ds], names, "mmd", "MMD to real test features", f"{DSN[ds]}: distance fake ↔ real, per class")
        floor = np.mean([np.mean(r["metrics"]["real (test)"]["mmd"]) for r in runs[ds]])
        ax.axhline(floor, color=GOOD, ls="--", lw=1)
        ax.text(9.6, floor, " real vs real\n (noise floor)", color=GOOD, fontsize=6.5, va="center")
        ax = axs[i, 1]
        for nm in ["CVAE-F", "CVAE-F+HWA", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"]:
            xs, ys = [], []
            for r in runs[ds]:
                if nm in r["metrics"] and "tstr_recall" in r["metrics"][nm]:
                    xs += r["metrics"][nm]["mmd"]
                    ys += r["metrics"][nm]["tstr_recall"]
            if xs:
                ax.scatter(xs, ys, s=12, color=COL[nm], label=LAB[nm], alpha=0.85, edgecolor="white", lw=0.3)
        allx = np.concatenate([np.array(r["metrics"][nm]["mmd"]) for r in runs[ds] for nm in r["metrics"]
                               if nm != "real (test)" and "tstr_recall" in r["metrics"][nm]])
        ally = np.concatenate([np.array(r["metrics"][nm]["tstr_recall"]) for r in runs[ds] for nm in r["metrics"]
                               if nm != "real (test)" and "tstr_recall" in r["metrics"][nm]])
        rho = np.corrcoef(allx, ally)[0, 1]
        style(ax, "real-test recall of a head\ntrained only on fake", f"{DSN[ds]}: Lemma 1 (r = {rho:.2f})")
        ax.set_xlabel("MMD of that class (fake ↔ real)")
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.13))
    fig.suptitle("Claim: KH brings fake features closest to real ones (lowest MMD), and lower MMD means better real accuracy",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5, y=1.05)
    fig.tight_layout()
    save(fig, "ev04_mmd.png")


# ------------------------------------------------ E5: train on fake, test on real
def fig_tstr(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs]
    if not dss:
        return
    names = ["CVAE-F", "CVAE-F+HWA", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"]
    fig, axs = plt.subplots(1, len(dss), figsize=(10, 3.1), squeeze=False)
    for j, ds in enumerate(dss):
        ax = axs[0, j]
        x = np.arange(2)
        w = 0.8 / len(names)
        for i, nm in enumerate(names):
            a = np.array([r["metrics"][nm]["tstr_recall"] for r in runs[ds] if "tstr_recall" in r["metrics"].get(nm, {})])
            if not len(a):
                continue
            v = np.stack([a.mean(1), a[:, 6:].mean(1)]) * 100
            m, s = v.mean(1), v.std(1)
            ax.bar(x - 0.4 + w * (i + 0.5), m, w * 0.92, yerr=s if len(a) > 1 else None, color=COL[nm], label=LAB[nm],
                   error_kw=dict(lw=0.6, capsize=1.5))
            for xi, mi in zip(x - 0.4 + w * (i + 0.5), m):
                ax.text(xi, mi + 1, f"{mi:.0f}", ha="center", fontsize=6.5)
        ref = np.mean([r["referee_test_acc"] for r in runs[ds]]) * 100
        ax.axhline(ref, color=GOOD, ls="--", lw=1)
        ax.text(1.42, ref, " trained on\n real data", color=GOOD, fontsize=6.5, va="center")
        ax.set_xticks(x, ["all classes (balanced acc.)", "rare classes 6–9"])
        ax.set_ylim(0, 105)
        style(ax, "real test accuracy (%)", f"{DSN[ds]} ({seeds_note(runs[ds])})")
    axs[0, 0].legend(fontsize=7, frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.1))
    fig.suptitle("Claim: fake features of Ours-A teach real classes. A head trained ONLY on 1,000 fake features per class, tested on real data",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5, y=1.1)
    fig.tight_layout()
    save(fig, "ev05_tstr.png")


# ------------------------------------------------ E6: class separability of the shared feature space
def fig_separability(runs):
    dss = [d for d in ("mnist", "fmnist", "cifar10", "svhn") if d in runs]
    if not dss:
        return
    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[0, :2])
    F = [np.array([r["real_F"] for r in runs[d]]) for d in dss]
    ax.bar(range(len(dss)), [f.mean() for f in F], yerr=[f.std() for f in F], color=[OURS if d != "svhn" else BAD for d in dss])
    for i, f in enumerate(F):
        ax.text(i, f.mean() + 0.01, f"{f.mean():.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(dss)), [DSN[d] for d in dss])
    style(ax, "F = spread of class means\n/ spread within a class", "How far apart the class means are")
    ax = fig.add_subplot(gs[0, 2:])
    A = [np.array([r["real_ncm_acc"] for r in runs[d]]) * 100 for d in dss]
    ax.bar(range(len(dss)), [a.mean() for a in A], yerr=[a.std() for a in A], color=[OURS if d != "svhn" else BAD for d in dss])
    for i, a in enumerate(A):
        ax.text(i, a.mean() + 1, f"{a.mean():.0f}%", ha="center", fontsize=7.5)
    ax.axhline(10, color=MUTED, ls=":", lw=1)
    ax.text(len(dss) - 0.5, 11, "chance", color=MUTED, fontsize=7, ha="right")
    ax.set_xticks(range(len(dss)), [DSN[d] for d in dss])
    style(ax, "test accuracy (%)", "Nearest-class-mean classifier on real test features")
    cm = plt.get_cmap("tab10")
    for j, d in enumerate(dss):
        f = os.path.join(D01, f"{d}_s0_lda.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f)
        ax = fig.add_subplot(gs[1, j])
        ax.scatter(z["bg"][:, 0], z["bg"][:, 1], c=[cm(int(c)) for c in z["bg_y"]], s=1.5, alpha=0.35)
        ax.scatter(z["means"][:, 0], z["means"][:, 1], c=[cm(c) for c in range(10)], s=70, marker="X",
                   edgecolor="black", lw=0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{DSN[d]}: real test features\n(X = exact class means)", fontsize=8, loc="left")
    fig.suptitle("Claim: the exact class mean identifies the class on MNIST, FMNIST and CIFAR-10, but not on SVHN "
                 f"(why Ours-A is weak there). 2-D LDA view, {seeds_note(runs[dss[0]])}",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout()
    save(fig, "ev06_separability.png")


# ------------------------------------------------ E7: one rare class, real vs fake, at each step
def fig_lda_steps(runs):
    for ds in ("mnist", "fmnist"):
        f = os.path.join(D01, f"{ds}_s0_lda.npz")
        if not os.path.exists(f) or ds not in runs:
            continue
        z = np.load(f)
        r0 = [r for r in runs[ds] if r["seed"] == 0][0]
        steps = ["CVAE-F", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"]
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.9), sharex=True, sharey=True)
        c = 9
        for ax, nm in zip(axs, steps):
            ax.scatter(z["bg"][:, 0], z["bg"][:, 1], s=1.2, color="#ced4da", alpha=0.5)
            ax.scatter(z[f"real_{c}"][:, 0], z[f"real_{c}"][:, 1], s=4, color=GOOD, alpha=0.5, label=f"real class {c}")
            ax.scatter(z[f"{nm}_{c}"][:, 0], z[f"{nm}_{c}"][:, 1], s=4, color=COL[nm], alpha=0.6, label=f"fake class {c}")
            met = r0["metrics"][nm]
            ax.set_title(f"{LAB[nm]}\nrecognised {met['fid'][c] * 100:.0f}% · MMD {met['mmd'][c]:.3f}", fontsize=7.5, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        axs[0].legend(fontsize=6.5, frameon=False, loc="upper left", markerscale=3)
        fig.suptitle(f"{DSN[ds]}, rarest class ({r0['class_counts'][9]} real samples in the whole federation, held by "
                     f"{r0['holders'][9]}/10 clients): real vs fake, 2-D LDA of real features (grey: all classes)",
                     x=0.01, ha="left", fontweight="bold", fontsize=9)
        fig.tight_layout()
        save(fig, f"ev07_rare_class_{ds}.png")


# ------------------------------------------------ E8: LA and CSL (Stage III)
def fig_heads(runs):
    dss = [d for d in ("mnist", "fmnist") if d in runs and any("stage3::Ours-A" in r for r in runs[d])]
    if not dss:
        return
    fig, axs = plt.subplots(len(dss), 3, figsize=(10, 2.9 * len(dss)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.3, 0.9, 1.3]))
    for i, ds in enumerate(dss):
        r = [x for x in runs[ds] if "stage3::Ours-A" in x][0]
        la, ce = r["stage3::Ours-A"], r["stage3::Ours-A with CE (no LA)"]
        ax = axs[i, 0]
        ax.bar(CLS - 0.2, np.array(ce["pred_share"]) * 100, 0.4, color=NEUT, label=f"plain CE (bal. acc. {ce['final_bal'] * 100:.1f})")
        ax.bar(CLS + 0.2, np.array(la["pred_share"]) * 100, 0.4, color=OURS, label=f"with LA (bal. acc. {la['final_bal'] * 100:.1f})")
        ax.axhline(10, color=GOOD, ls="--", lw=1)
        ax.set_xticks(CLS)
        ax.set_xlabel("predicted class")
        style(ax, "share of test predictions (%)", f"{DSN[ds]}: what the heads predict (balanced test: 10% each)")
        tailband(ax)
        ax.legend(fontsize=6.8, frameon=False, loc="upper right")
        ax = axs[i, 1]
        for st, colr, nm in ((ce, NEUT, "CE"), (la, OURS, "LA")):
            pr = np.array([p["prior"] for p in st["per_client"]]).ravel() * 100
            sh = np.array([p["pred_share"] for p in st["per_client"]]).ravel() * 100
            rho = np.corrcoef(pr, sh)[0, 1]
            ax.scatter(pr, sh, s=9, color=colr, alpha=0.75, label=f"{nm}: r = {rho:.2f}", edgecolor="white", lw=0.3)
        style(ax, "head's prediction share (%)", "each client: own data vs predictions")
        ax.set_xlabel("class share in the client's own data (%)")
        ax.legend(fontsize=6.8, frameon=False)
        ax = axs[i, 2]
        th, tc = np.array(la["csl"]["tv_onehot"]), np.array(la["csl"]["tv_csl"])
        ax.bar(CLS - 0.2, th, 0.4, color=NEUT, label=f"one-hot label (mean {th.mean():.3f})")
        ax.bar(CLS + 0.2, tc, 0.4, color=OURS, label=f"consensus soft label (mean {tc.mean():.3f})")
        ax.set_xticks(CLS)
        ax.set_xlabel("class of the fake sample")
        style(ax, "distance to the real-data\nreferee's belief (TV)", "CSL: training target vs what real data says")
        ax.legend(fontsize=6.8, frameon=False, loc="upper left")
        tailband(ax)
    fig.suptitle("Claims: LA removes each client's own class skew; consensus labels are closer to the truth than the generator's one-hot label "
                 "(seed 0, final heads of Ours-A)", x=0.01, ha="left", fontweight="bold", fontsize=9.3, y=1.02)
    fig.tight_layout()
    save(fig, "ev08_la_csl.png")


# ------------------------------------------------ E9: T_s interaction (existing 3-seed runs)
def fig_ts():
    # final balanced accuracy, long tail, 3 seeds: E16 (CVAE-F + HWA generator) and E25 (Ours-A generator)
    data = {"MNIST": {"CVAE-F + HWA generator": (90.20, 90.15), "Ours-A generator (PC + MC + KH)": (93.11, 94.82)},
            "FashionMNIST": {"CVAE-F + HWA generator": (76.94, 79.23), "Ours-A generator (PC + MC + KH)": (78.22, 79.67)}}
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    for ax, (ds, d) in zip(axs, data.items()):
        for k, (nm, (a, b)) in enumerate(d.items()):
            col = NEUT if k == 0 else OURS
            ax.plot([1, 10], [a, b], "-o", color=col, label=f"{nm}: {b - a:+.2f}")
            ax.text(10.3, b, f"{b:.2f}", va="center", fontsize=7, color=col)
            ax.text(0.7, a, f"{a:.2f}", va="center", ha="right", fontsize=7, color=col)
        ax.set_xticks([1, 10], ["T_s = 1\n(GeFL-F)", "T_s = 10\n(Ours-A)"])
        ax.set_xlim(-1, 12.5)
        style(ax, "balanced accuracy (%)", f"{ds} long tail (3 seeds)")
        ax.legend(fontsize=6.8, frameon=False, loc="lower right")
    fig.suptitle("Claim (Prop. 5): more fake training pays more when the generator is more faithful",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout()
    save(fig, "ev09_ts.png")


# ------------------------------------------------ E10: leave-one-out ablation (K09)
def fig_ablation():
    rows = []
    for d in K09:
        f = os.path.join(d, "runs.jsonl")
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f, encoding="utf-8")]
    if not rows:
        return None
    arms = ["Ours-A Ts=10", "-PC (CVAE+HWA)", "-MC", "-KH", "-LA", "-CSL"]
    nice = {"Ours-A Ts=10": "Ours-A (all parts)", "-PC (CVAE+HWA)": "− PC-VAE\n(CVAE-F + HWA)", "-MC": "− MC",
            "-KH": "− KH", "-LA": "− LA", "-CSL": "− CSL"}
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, ds in zip(axs, ("mnist", "fmnist")):
        full = {r["seed"]: r for r in rows if r["dataset"] == ds and r["label"] == arms[0]}
        ms, ss, tl, pv = [], [], [], []
        for a in arms:
            rs = {r["seed"]: r for r in rows if r["dataset"] == ds and r["label"] == a}
            v = np.array([rs[s]["final_bal"] for s in sorted(rs)]) * 100
            t = np.array([rs[s]["final_tail"] for s in sorted(rs)]) * 100
            ms.append(v.mean() if len(v) else np.nan); ss.append(v.std() if len(v) > 1 else 0)
            tl.append(t.mean() if len(t) else np.nan)
            common = sorted(set(rs) & set(full))
            if a != arms[0] and len(common) > 1:
                from math import sqrt
                dlt = np.array([rs[s]["final_bal"] - full[s]["final_bal"] for s in common]) * 100
                tstat = dlt.mean() / (dlt.std(ddof=1) / sqrt(len(dlt)) + 1e-12)
                pv.append((dlt.mean(), tstat, len(common)))
            else:
                pv.append(None)
        out[ds] = dict(zip(arms, zip(ms, ss, tl, pv)))
        if not np.isfinite(ms[0]):
            continue
        x = np.arange(len(arms))
        ax.bar(x, ms, yerr=ss, color=[OURS] + [BAD] * (len(arms) - 1), error_kw=dict(lw=0.7, capsize=2))
        for xi, m in zip(x, ms):
            if np.isfinite(m):
                ax.text(xi, m + 0.4, f"{m:.1f}", ha="center", fontsize=7)
        lo = np.nanmin(ms) - 6
        ax.set_ylim(lo, np.nanmax(ms) + 3)
        ax.set_xticks(x, [nice[a] for a in arms], fontsize=7)
        n = len(full)
        style(ax, "balanced accuracy (%)", f"{DSN[ds]} long tail ({n} seed{'s' if n > 1 else ''})")
    fig.suptitle("Ablation: remove one part of Ours-A at a time (everything else, seeds and splits unchanged)",
                 x=0.01, ha="left", fontweight="bold", fontsize=9.5)
    fig.tight_layout()
    save(fig, "ev10_leave_one_out.png")
    json.dump(out, open(os.path.join(OUT, "ev10_leave_one_out.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    runs = load()
    print("D01 runs:", {k: len(v) for k, v in runs.items()})
    fig_collapse(runs); fig_fidelity(runs); fig_moments(runs); fig_mmd(runs); fig_tstr(runs)
    fig_separability(runs); fig_lda_steps(runs); fig_heads(runs); fig_ts(); fig_ablation()
