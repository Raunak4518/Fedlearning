"""Figures and rendered equations of the handout (docs/professor_report). Every result figure reads the result files:
experiments/results/*/runs.jsonl, docs/paper_assets/tables/T22_fair_retest.md, docs/research/math_checks/*.json(l).
Run: python docs/professor_report/figures.py   (then: node build_handout.js)"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(REPO, "experiments", "results")
MC = os.path.join(REPO, "docs", "research", "math_checks")
OUT = os.path.join(HERE, "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})
INK, MUTED = "#1f2a37", "#5b6676"
C_SERVER, C_CLIENT, C_GEN, C_FE, C_HEAD = "#e8eefc", "#eef7ee", "#fdf1e3", "#f3eafc", "#e6f6f8"
E_SERVER, E_CLIENT, E_GEN, E_FE, E_HEAD = "#3b5bdb", "#2f9e44", "#e8590c", "#7048e8", "#0c8599"
GOOD, BAD, NEUT, OURS, WARN = "#2f9e44", "#c92a2a", "#868e96", "#1c7ed6", "#e8590c"
DS = ["mnist", "fmnist", "svhn", "cifar10", "cifar100"]
DSN = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}


def runs(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern, "runs.jsonl"))):
        out += [json.loads(l) for l in open(f)]
    return out


def save(fig, name, tight=True):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white", **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK, fs=8.5, bold=False, lw=1.2, r=0.02, color=INK, va="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2 if va == "center" else y + h - 0.012, text, ha="center", va=va, fontsize=fs,
            color=color, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK, lw=1.3, style="-|>", ls="-", rad=0.0, text=None, tx=None, ty=None, fs=7.5, tc=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2, text, fontsize=fs,
                color=tc or color, ha="center", va="center", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.95))


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def fig_overview():
    fig, ax = canvas(10, 6.4)
    ax.text(0.5, 0.975, "GeFL-F: system and message flow (K clients, G = 10 head architectures)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=INK)
    # server
    box(ax, 0.22, 0.70, 0.56, 0.22, "", fc=C_SERVER, ec=E_SERVER, lw=1.6)
    ax.text(0.5, 0.905, "Server (no data; aggregation only)", ha="center", fontsize=10, fontweight="bold", color=E_SERVER)
    box(ax, 0.245, 0.725, 0.16, 0.15, "Stage I\nFedAvg of the\nfeature extractor φ\n(+ heads within each\narchitecture group)", fc="white", ec=E_FE, fs=7.5)
    box(ax, 0.42, 0.725, 0.16, 0.15, "Stage II\nFedAvg of the\nfeature generator G\n(CVAE-F, ≈5.5 M params)", fc="white", ec=E_GEN, fs=7.5)
    box(ax, 0.595, 0.725, 0.16, 0.15, "Stage III\nheads averaged within\neach architecture\ngroup (none at K = 10:\none client per group)", fc="white", ec=E_HEAD, fs=7.5)
    # clients
    xs = [0.03, 0.27, 0.51, 0.75]
    labels = ["Client 1\nhead CNN-1", "Client 2\nhead CNN-2", "Client k\nhead CNN-a(k)", "Client K\nhead CNN-10"]
    for x, lab in zip(xs, labels):
        box(ax, x, 0.05, 0.22, 0.36, "", fc=C_CLIENT, ec=E_CLIENT, lw=1.4)
        ax.text(x + 0.11, 0.385, lab, ha="center", va="top", fontsize=8.5, fontweight="bold", color=E_CLIENT)
        box(ax, x + 0.015, 0.215, 0.19, 0.075, "private data D_k\n(non-IID / long-tailed)", fc="white", ec=MUTED, fs=7)
        box(ax, x + 0.015, 0.13, 0.09, 0.07, "FE φ\n(shared)", fc=C_FE, ec=E_FE, fs=7)
        box(ax, x + 0.115, 0.13, 0.09, 0.07, "head h_k\n(private)", fc=C_HEAD, ec=E_HEAD, fs=7)
        box(ax, x + 0.015, 0.06, 0.19, 0.055, "generator G (shared copy)", fc=C_GEN, ec=E_GEN, fs=7)
    
    # up / down arrows
    arrow(ax, 0.14, 0.41, 0.30, 0.72, color=E_FE, rad=-0.1)
    arrow(ax, 0.38, 0.41, 0.48, 0.72, color=E_GEN, rad=-0.05)
    arrow(ax, 0.86, 0.41, 0.68, 0.72, color=E_HEAD, rad=0.1)
    arrow(ax, 0.70, 0.70, 0.62, 0.41, color=E_SERVER, ls="--", rad=0.1)
    ax.text(0.015, 0.58, "UPLINK (each round, per client)\n• Stage I: φ weights + head weights\n   (36–390 + 5.3k–110k numbers)\n"
            "• Stage II: all CVAE-F weights\n   ≈ 5.5 M numbers ≈ 22 MB (fp32)\n• Stage III: head weights (only if\n   several clients share an architecture)",
            fontsize=7.6, color=INK, va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#ced4da"))
    ax.text(0.80, 0.58, "DOWNLINK (each round)\n• averaged φ (Stage I)\n• averaged G (Stage II)\n• after Stage II: G frozen and\n   kept by every client; each client\n   samples synthetic features\n   locally to train its head",
            fontsize=7.6, color=INK, va="center", bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#ced4da"))
    ax.text(0.5, 0.012, "Raw images, real features and labels never leave a client. Only model parameters move.",
            ha="center", fontsize=8, color=MUTED, style="italic")
    save(fig, "f01_overview.png", tight=False)



def fig_method():
    """The method inside GeFL-F: what each stage changes (kernel herding is optional)."""
    fig, ax = canvas(10, 5.6)
    ax.text(0.5, 0.975, "The method inside GeFL-F (changes outlined in blue; dashed = optional)", ha="center", va="top",
            fontsize=12, fontweight="bold", color=INK)
    box(ax, 0.02, 0.70, 0.40, 0.20, "Once, after Stage I, each client uploads per-class sums\n"
        "n_kc,  \u03a3 h,  \u03a3 \u2016h\u2016\u00b2   (secure aggregation:\nthe server sees only federation totals)",
        fc=C_CLIENT, ec=E_CLIENT, fs=8.5)
    box(ax, 0.50, 0.70, 0.48, 0.20, "Server: exact class mean m_c and spread V_c\n"
        "separation F = mean \u2016m_c \u2212 m\u2016\u00b2 / mean V_c  (decides the generator)", fc=C_SERVER, ec=OURS, lw=1.8, fs=8.5)
    arrow(ax, 0.42, 0.80, 0.50, 0.80, color=OURS)
    box(ax, 0.02, 0.36, 0.30, 0.26, "1) Stage II generator\nF \u2265 0.05: mean anchor\n  h\u0303 = ReLU(m_y + dec(z, A m_y))\n"
        "  (no class row that can collapse)\nF < 0.05 (SVHN): class rows +\n  holder-weighted averaging", fc=C_GEN, ec=OURS, lw=1.8, fs=8)
    box(ax, 0.36, 0.36, 0.28, 0.26, "2) Moment correction (MC)\nx\u2032 = m_c + \u221a(V_c / V\u0303_c)(x \u2212 m\u0303_c)\n"
        "exact class mean and spread\nwith the smallest change", fc="white", ec=OURS, lw=1.8, fs=8)
    ax.add_patch(FancyBboxPatch((0.68, 0.36), 0.30, 0.26, boxstyle="round,pad=0.004,rounding_size=0.02", fc="#f8f9fa",
                                ec=MUTED, lw=1.2, ls="--"))
    ax.text(0.83, 0.49, "Optional: kernel herding (KH)\nkeep n of 4n samples closest to the\nexact class fingerprint\n"
            "+0.1 to +0.4 points, ~6\u00d7 head-stage time", ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.35)
    arrow(ax, 0.74, 0.70, 0.17, 0.62, color=OURS, rad=0.0)
    arrow(ax, 0.32, 0.49, 0.36, 0.49, color=OURS)
    arrow(ax, 0.64, 0.49, 0.68, 0.49, color=MUTED, ls="--")
    box(ax, 0.02, 0.05, 0.62, 0.24, "3) Stage III on client k (private head of any architecture)\n"
        "a) T_s epochs on corrected synthetic features with consensus labels (CSL):\n"
        "    target \u00bd e_y + \u00bd mean_g softmax h_g(h\u0303)\n"
        "b) T_r = 5 epochs on own real features with logit adjustment\n    by the client's own label prior (LA)",
        fc=C_HEAD, ec=OURS, lw=1.8, fs=8)
    arrow(ax, 0.50, 0.36, 0.40, 0.29, color=OURS)
    box(ax, 0.68, 0.05, 0.30, 0.24, "4) Exact readout (ER), once after training\nlast layer = class-balanced LDA from\n"
        "\u03a3 e,  \u03a3 ee\u1d40,  n_c  of each architecture's\nembeddings (one extra round)\n= the centralised readout",
        fc=C_SERVER, ec=OURS, lw=1.8, fs=8)
    arrow(ax, 0.64, 0.17, 0.68, 0.17, color=OURS)
    save(fig, "m01_method.png", tight=False)


EQ = {
    "eq_collapse": r"$\|r_c^{+}\|\ \leq\ \|r_c\| + p_c\,\bar G_c - (1-p_c)\,\bar D_c,\qquad \mathrm{shrinks\ whenever}\ \ p_c=\frac{m_c}{K} < \theta_c=\frac{\bar D_c}{\bar G_c+\bar D_c}$",
    "eq_kappa": r"$\kappa_c=\frac{\sum_{k\in H_c}\tau_k f_{kc}}{\sum_{k\notin H_c}\tau_k}\qquad(\kappa_c<1\ \Rightarrow\ \mathrm{the\ class\ row\ collapses})$",
    "eq_hwa": r"$w_c \leftarrow \frac{\sum_k E(n_{kc})\, w_c^{(k)}}{\sum_k E(n_{kc})},\qquad E(n)=\frac{1-\beta^{\,n}}{1-\beta},\ \ \beta = 0.999$",
    "eq_er": r"$w_c=S_\gamma^{-1}\mu_c,\quad b_c=-\frac{1}{2}\mu_c^{\top}S_\gamma^{-1}\mu_c,\qquad S_\gamma=(1-\gamma)S+\gamma\,\frac{\mathrm{tr}\,S}{d}\,I$",
    "eq_logitvar": r"$\mathbb{E}_{y\sim\mathrm{Unif}}\,\ell(f,y)=\log C+\mathrm{KL}(u\,\|\,\mathrm{softmax} f)\approx\log C+\frac{1}{2C}\sum_c (f_c-\bar f)^2$",
    "eq_pc": r"$\tilde h = \mathrm{ReLU}\left(m_y + \mathrm{dec}(z,\ A\,m_y)\right),\qquad z\sim\mathcal{N}(0,I)$",
    "eq_mc": r"$T^\star(x) = m_c + \sqrt{V_c/\tilde V_c}\,(x-\tilde m_c)\ \ =\ \ \arg\min_{T}\ \mathbb{E}\|T(X)-X\|^2\ \ \mathrm{s.t.}\ \ \mathbb{E}T=m_c,\ \mathbb{E}\|T-m_c\|^2=V_c$",
    "eq_la": r"$\ell_{LA}(h,y) = -\log \frac{\exp(f_y(h)+\tau\log\pi_{k,y})}{\sum_{j}\exp(f_j(h)+\tau\log\pi_{k,j})}$",
    "eq_csl": r"$t(\tilde h) = (1-\beta)\,e_y + \beta\,\frac{1}{G}\sum_{g=1}^{G}\mathrm{softmax}\left(h_g(\tilde h)\right),\qquad \beta = 0.5$",
}


def equations():
    for name, tex in EQ.items():
        fig = plt.figure(figsize=(0.1, 0.1), dpi=220)
        fig.text(0, 0, tex, fontsize=13, color=INK)
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        plt.close(fig)


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
    for f in (fig_overview, fig_method, equations, v01, v02, v03, v04, v06):
        f()
