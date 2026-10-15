"""Figures and rendered equations for the evaluation report (docs/professor_report)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})

INK, MUTED = "#1f2a37", "#5b6676"
C_SERVER, C_CLIENT, C_GEN, C_FE, C_HEAD = "#e8eefc", "#eef7ee", "#fdf1e3", "#f3eafc", "#e6f6f8"
E_SERVER, E_CLIENT, E_GEN, E_FE, E_HEAD = "#3b5bdb", "#2f9e44", "#e8590c", "#7048e8", "#0c8599"
GOOD, BAD, NEUT, OURS = "#2f9e44", "#c92a2a", "#868e96", "#1c7ed6"


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


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: system overview with messages
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
    save(fig, "f01_overview.png")


# ---------------------------------------------------------------- Fig 2: lifecycle timeline
def fig_timeline():
    fig, ax = canvas(10, 4.9)
    ax.text(0.5, 0.97, "Training lifecycle: three sequential stages", ha="center", va="top", fontsize=12, fontweight="bold")
    rows = [
        ("Stage I  ·  feature-extractor warm-up", E_FE, C_FE,
         "Rounds T_FE = 20 (MNIST, FMNIST) / 50 (SVHN, CIFAR-10)\nPer round, each client: 5 local epochs on its REAL images x\n"
         "Trains φ and its own head jointly · SGD lr 0.1 · batch 64 (128 CIFAR)\nLoss: cross-entropy  CE(h_k(φ(x)), y)",
         "Up: φ, head  →  Server: FedAvg(φ) over all clients;\nheads averaged within each architecture group\nDown: averaged φ, heads.  End: φ is FROZEN"),
        ("Stage II  ·  federated feature generator", E_GEN, C_GEN,
         "Rounds T_KA = 100 (200 CIFAR-10)\nPer round, each client: 5 local epochs on its REAL features h = φ(x)\n"
         "Trains the conditional VAE G(h | y) · Adam lr 1e-3, weight decay 1e-3\nLoss: ‖ĥ − h‖² (sum) + KL(q(z|h,y) ‖ N(0, I))",
         "Up: all G weights (≈5.5 M)  →  Server: FedAvg(G)\nDown: averaged G.  End: G is FROZEN and\nevery client holds a copy"),
        ("Stage III  ·  heterogeneous head training", E_HEAD, C_HEAD,
         "Rounds T_TN = 50 (MNIST, FMNIST) / 100 (SVHN, CIFAR-10)\nPer round, each client: T_s = 1 epoch on SYNTHETIC features h̃ ~ G(·|y),\n"
         "y uniform, then T_r = 5 epochs on its REAL features · SGD lr 0.1\nLoss: cross-entropy on both",
         "Up/down: head weights only when several clients\nshare an architecture (K = 50, 100)\nEvaluation: each head on the balanced test set"),
    ]
    y = 0.70
    for title, ec, fc, mid, right in rows:
        box(ax, 0.02, y, 0.20, 0.22, title, fc=fc, ec=ec, fs=8.5, bold=True, lw=1.5)
        box(ax, 0.235, y, 0.47, 0.22, mid, fc="white", ec=ec, fs=7.6)
        box(ax, 0.715, y, 0.265, 0.22, right, fc="#fafafa", ec="#ced4da", fs=7.4)
        if y > 0.1:
            arrow(ax, 0.12, y, 0.12, y - 0.06, color=MUTED)
        y -= 0.30
    ax.text(0.5, 0.02, "Metric of the paper: best_mean_acc = test accuracy of each architecture at its best round, averaged over the 10 architectures.",
            ha="center", fontsize=7.8, color=MUTED, style="italic")
    save(fig, "f02_timeline.png")


# ---------------------------------------------------------------- Fig 3: architectures
def fig_arch():
    fig, ax = canvas(10, 5.4)
    ax.text(0.5, 0.975, "Model architectures and tensor shapes (MNIST shown; CIFAR-10 in brackets)", ha="center",
            va="top", fontsize=11.5, fontweight="bold")
    # FE
    box(ax, 0.02, 0.72, 0.13, 0.16, "image x\n1×32×32 (resized)\n[3×32×32]", fc="white", ec=MUTED, fs=8)
    box(ax, 0.18, 0.70, 0.17, 0.20, "Feature extractor φ\nConv3×3 → BN → ReLU\n(+ Conv3×3 → BN → ReLU, CIFAR)\n→ MaxPool 2\n36 params [90; 390]", fc=C_FE, ec=E_FE, fs=7.6)
    box(ax, 0.38, 0.72, 0.13, 0.16, "feature h\n3×16×16 = 768\n[CIFAR-10:\n10×16×16 = 2560]", fc="white", ec=E_FE, fs=7.6)
    arrow(ax, 0.15, 0.80, 0.18, 0.80)
    arrow(ax, 0.35, 0.80, 0.38, 0.80)
    # CVAE
    box(ax, 0.55, 0.53, 0.43, 0.39, "", fc=C_GEN, ec=E_GEN, lw=1.5)
    ax.text(0.765, 0.905, "Conditional VAE feature generator G (CVAE-F, Table XX)", ha="center", fontsize=8.8, fontweight="bold", color=E_GEN)
    box(ax, 0.57, 0.71, 0.18, 0.15, "Encoder q(z | h, y)\n[h ; y-map] → 4 × (Conv4×4 s2,\nBN, ReLU): 64-128-256-512\n→ μ, log σ²  (l-dim)", fc="white", ec=E_GEN, fs=7.2)
    box(ax, 0.78, 0.71, 0.18, 0.15, "Decoder p(h | z, y)\nLinear(l + C → 512) →\n4 × ConvT4×4 s2 →  ĥ\nl = 16 (50 CIFAR)", fc="white", ec=E_GEN, fs=7.2)
    box(ax, 0.60, 0.55, 0.33, 0.12, "Label enters through one learned matrix W (C × 512):\nrow w_c is the ONLY class-specific parameter.\nIt is FedAvg-averaged like every other weight.",
        fc="#fff4e6", ec=BAD, fs=7.4, color="#862e2e")
    arrow(ax, 0.75, 0.785, 0.78, 0.785, color=E_GEN)
    arrow(ax, 0.51, 0.80, 0.57, 0.79, color=E_GEN)
    # heads
    box(ax, 0.02, 0.06, 0.96, 0.40, "", fc=C_HEAD, ec=E_HEAD, lw=1.5)
    ax.text(0.5, 0.44, "Heterogeneous heads CNN-1 … CNN-10 (Tables XXI–XXIII): input = feature h; each block = Conv3×3 → (BN) → ReLU → MaxPool 2; then Linear → 10 classes",
            ha="center", fontsize=8, fontweight="bold", color=E_HEAD)
    widths = [(16,), (16, 32), (20, 40), (10, 20), (16, 32, 64), (20, 40, 80), (10, 20, 40), (16, 32, 64, 128), (20, 40, 80, 100), (10, 20, 40, 80)]
    params = [10698, 10218, 14210, 5310, 26154, 39890, 10950, 98730, 109790, 39030]
    for i, (w, p) in enumerate(zip(widths, params)):
        x = 0.035 + i * 0.094
        hgt = 0.06 + 0.06 * len(w)
        box(ax, x, 0.13, 0.082, hgt, f"CNN-{i+1}\n{'-'.join(map(str, w))}", fc="white", ec=E_HEAD, fs=7)
        ax.text(x + 0.041, 0.095, f"{p/1000:.1f} k params", ha="center", fontsize=6.6, color=MUTED)
    save(fig, "f03_architectures.png")


# ---------------------------------------------------------------- Fig 4: stage III round in detail
def fig_round():
    fig, ax = canvas(10, 3.6)
    ax.text(0.5, 0.96, "One Stage-III round on client k (GeFL-F)", ha="center", va="top", fontsize=11.5, fontweight="bold")
    box(ax, 0.02, 0.35, 0.15, 0.36, "labels y ~ Uniform{0..9}\nz ~ N(0, I)", fc="white", ec=MUTED, fs=8)
    box(ax, 0.21, 0.35, 0.15, 0.36, "frozen G\nh̃ = G(z, y)\n(local copy)", fc=C_GEN, ec=E_GEN, fs=8)
    box(ax, 0.40, 0.35, 0.22, 0.36, "Synthetic phase\nT_s = 1 epoch\nloss CE(h_k(h̃), y)\nSGD lr 0.1", fc="white", ec=E_GEN, fs=8)
    box(ax, 0.66, 0.35, 0.22, 0.36, "Real phase\nT_r = 5 epochs on φ(x) of D_k\nloss CE(h_k(φ(x)), y)\nSGD lr 0.1", fc="white", ec=E_CLIENT, fs=8)
    box(ax, 0.905, 0.35, 0.085, 0.36, "updated\nhead h_k", fc=C_HEAD, ec=E_HEAD, fs=8)
    for a, b in [(0.17, 0.21), (0.36, 0.40), (0.62, 0.66), (0.88, 0.905)]:
        arrow(ax, a, 0.53, b, 0.53)
    ax.text(0.5, 0.15, "Knowledge transfer happens ONLY through the synthetic phase: features of classes a client does not hold come from G.\n"
            "If G cannot produce a class, no head can learn it from anyone else.", ha="center", fontsize=8.2, color="#862e2e")
    save(fig, "f04_round.png")


# ---------------------------------------------------------------- Fig 5: the problem
def fig_problem():
    fig = plt.figure(figsize=(10, 3.9), dpi=200)
    ax1 = fig.add_axes([0.06, 0.16, 0.27, 0.68])
    ds = ["MNIST", "FMNIST", "CIFAR-10", "SVHN"]
    iid = [96.0, 82.8, 59.1, 75.8]
    lt = [75.3, 58.5, 33.7, 50.5]
    import numpy as np
    x = np.arange(4)
    ax1.bar(x - 0.2, iid, 0.4, color=NEUT, label="balanced (IID)")
    ax1.bar(x + 0.2, lt, 0.4, color=BAD, label="long tail (IF = 100)")
    for i in range(4):
        ax1.text(x[i] + 0.2, lt[i] + 1, f"{lt[i]:.0f}", ha="center", fontsize=7.5, color=BAD)
        ax1.text(x[i] - 0.2, iid[i] + 1, f"{iid[i]:.0f}", ha="center", fontsize=7.5, color=MUTED)
    ax1.set_xticks(x, ds, fontsize=8)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel("accuracy (%)")
    ax1.set_title("(a) GeFL-F collapses under the long tail", fontsize=9, loc="left")
    ax1.legend(fontsize=7, frameon=False, loc="upper right")
    for s in ["top", "right"]:
        ax1.spines[s].set_visible(False)
    ax2 = fig.add_axes([0.39, 0.16, 0.23, 0.68])
    m = [("row-norm ratio\n(tail / head)", 0.33, 0.80), ("tail fidelity\n(referee acc.)", 0.088, 0.272)]
    for i, (lab, a, b) in enumerate(m):
        ax2.bar(i - 0.2, a, 0.4, color=BAD)
        ax2.bar(i + 0.2, b, 0.4, color=GOOD)
        ax2.text(i - 0.2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7.5)
        ax2.text(i + 0.2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7.5)
    ax2.set_xticks([0, 1], [m[0][0], m[1][0]], fontsize=7.6)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(b) MNIST: GeFL-F (red) vs rows fixed (green)", fontsize=9, loc="left")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    ax3 = fig.add_axes([0.73, 0.16, 0.25, 0.68])
    import numpy as np
    ratio = np.linspace(0.02, 1.5, 200)                      # Gbar / Dbar
    theta = 1 / (1 + ratio)
    ax3.plot(ratio, theta, color=INK)
    ax3.fill_between(ratio, 0, theta, color=BAD, alpha=0.10)
    ax3.text(1.0, 0.75, "row can grow", color=GOOD, fontsize=7.5)
    ax3.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax3.text(1.02, 0.52, "old rule: K/2", color=MUTED, fontsize=7)
    ax3.axvspan(0.07, 0.64, ymax=0.06, color=BAD, alpha=0.6)
    ax3.text(0.09, 0.08, "measured range", color=BAD, fontsize=7)
    ax3.text(0.12, 0.35, "row shrinks:\nholder share < θ", color=BAD, fontsize=7.5)
    ax3.set_xlabel("holder gain / non-holder loss,  Ḡ / D̄")
    ax3.set_ylabel("threshold θ on m_c / K")
    ax3.set_ylim(0, 1.0)
    ax3.set_title("(c) Theorem 1: the collapse threshold", fontsize=9, loc="left")
    for s_ in ["top", "right"]:
        ax3.spines[s_].set_visible(False)
    save(fig, "f05_problem.png")


# ---------------------------------------------------------------- Fig 6: research map
def fig_journey():
    fig, ax = canvas(10, 6.6)
    ax.text(0.5, 0.98, "Research map: every approach tested, where it acts, and the outcome", ha="center", va="top",
            fontsize=12, fontweight="bold")
    cols = [("Generator aggregation\n(server, Stage II)", 0.02), ("Generator design\n(Stage II)", 0.215), ("Sampling\n(server, before Stage III)", 0.41),
            ("Head training\n(client, Stage III)", 0.605), ("Calibration / server\n(after Stage III)", 0.80)]
    for t, x in cols:
        box(ax, x, 0.86, 0.18, 0.07, t, fc="#f1f3f5", ec="#adb5bd", fs=7.8, bold=True)
    items = {
        0.02: [("HWA: holder-weighted rows", 3), ("LCD: lazy row decay", 0), ("DP-noised counts for HWA", 3)],
        0.215: [("PC-VAE: class mean anchor", 1), ("F-gate: rows when means fail", 2), ("FSG: Gaussian from statistics", 0),
                ("Hybrid CVAE / Gaussian", 0), ("PCR: rows + anchor", 0), ("KME-Gen: server-only MMD", 0), ("DDPM-F generator", 0), ("ZP: latent prior", 0)],
        0.41: [("MC: exact mean + spread", 1), ("MC-S: + covariance (pooled)", 0), ("MC-B: + covariance (Bayes)", 2), ("KH: kernel herding", 1),
               ("DP on all statistics", 1)],
        0.605: [("LA: client-prior logit adj.", 1), ("CSL: consensus soft labels", 1), ("T_s = 10 synthetic budget", 1), ("Bayesian CSL", 0),
                ("CSLM: interleaved batches", 0), ("PCM: prior-completing", 0), ("RSA: alignment loss", 0), ("VCT / MixUp", 0)],
        0.80: [("BBC: bias calibration", 2), ("SED: server distillation", 0), ("BCR: classifier re-train", 0), ("GF: gradient filtering", 0)],
    }
    for x, lst in items.items():
        y = 0.79
        for name, ok in lst:
            mark, fc, ec = {0: ("✘  ", "#fff5f5", BAD), 1: ("✔  ", "#ebfbee", GOOD),
                            2: ("◷  ", "#fff9db", "#e0a800"), 3: ("➜  ", "#e7f5ff", "#4dabf7")}[ok]
            box(ax, x, y, 0.18, 0.06, mark + name, fc=fc, ec=ec, fs=7.2, color=INK)
            y -= 0.075
    ax.text(0.5, 0.215, "✔ in Ours-A     ➜ worked, then replaced by the mean anchor     ◷ future scope (not yet verified everywhere)     ✘ dropped",
            ha="center", fontsize=8, color=MUTED)
    fig.savefig(os.path.join(OUT, "f06_journey.png"), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 7: ESA architecture
def fig_esa():
    fig, ax = canvas(10, 6.6)
    ax.text(0.5, 0.985, "Ours-A inside GeFL-F (what changes is outlined in blue)", ha="center",
            va="top", fontsize=12, fontweight="bold")
    box(ax, 0.02, 0.74, 0.40, 0.19, "Each client k, ONCE after Stage I, uploads per-class sums:\n\n"
        "n_kc,   Σ h,   Σ ‖h‖²,   Σ φ_RFF(h)  (D = 6144)\nh = 3×16×16 = 768 numbers (CIFAR-10: 2560)\n≈ 0.3 MB per client",
        fc=C_CLIENT, ec=E_CLIENT, fs=7.8)
    box(ax, 0.50, 0.74, 0.48, 0.19, "Secure aggregation → server sees only totals\n\n"
        "m_c = S_c / n_c   (class mean)          V_c = Q_c / n_c − ‖m_c‖²   (spread)\n"
        "μ_c = Φ_c / n_c   (kernel embedding: class fingerprint)", fc=C_SERVER, ec=OURS, fs=7.8, lw=1.8)
    arrow(ax, 0.42, 0.835, 0.50, 0.835, color=OURS)
    box(ax, 0.02, 0.40, 0.30, 0.26, "1) Stage II generator PC-VAE (FedAvg)\n\n"
        "h̃ = ReLU( m_y + dec(z, A m_y) )\nz ~ N(0, I), 16-d\n\nclass enters only through exact m_y\n→ no class row to shrink", fc=C_GEN, ec=OURS, fs=7.8, lw=1.8)
    box(ax, 0.36, 0.40, 0.29, 0.26, "2) Moment fix MC (Proposition 3)\n\n"
        "x' = m_c + √(V_c / Ṽ_c) · (x − m̃_c)\n\nexact class mean and spread\nwith the smallest change",
        fc="white", ec=OURS, fs=7.8, lw=1.8)
    box(ax, 0.69, 0.40, 0.29, 0.26, "3) Fingerprint selection KH (Lemma 1)\n\ndraw 4n per class, keep n by\nkernel herding on\n"
        "‖ (1/n) Σ_t φ(x_t) − μ_c ‖\n→ fake ≈ real (low MMD)", fc="white", ec=OURS, fs=7.8, lw=1.8)
    arrow(ax, 0.62, 0.74, 0.20, 0.66, color=OURS)
    arrow(ax, 0.32, 0.53, 0.36, 0.53, color=OURS)
    arrow(ax, 0.65, 0.53, 0.69, 0.53, color=OURS)
    box(ax, 0.02, 0.09, 0.96, 0.23, "4) Stage III on client k  (T_TN rounds, private head of any architecture)\n\n"
        "a) T_s = 10 epochs on the corrected fake pool, target  t = ½ e_y + ½ · mean_g softmax h_g(h̃)   (consensus soft label, CSL)\n"
        "b) T_r = 5 epochs on own real features with logit adjustment:  CE( h_k(h) + τ log π_k , y ),  π_k = client's private label prior (LA)",
        fc=C_HEAD, ec=E_HEAD, fs=7.8, lw=1.5)
    arrow(ax, 0.83, 0.40, 0.60, 0.32, color=OURS, text="corrected fake pool", tx=0.80, ty=0.355)
    ax.text(0.5, 0.025, "Only sums leave the clients; no raw data, features or single-client statistics. Optional (ε, δ)-DP on every release.",
            ha="center", fontsize=7.8, color=MUTED, style="italic")
    save(fig, "f07_esa.png")


# ---------------------------------------------------------------- result charts
def bars(name, title, groups, series, ylim, ylabel="balanced accuracy (%)", w=10, h=3.6, annotate=True, refline=None):
    import numpy as np
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0.07, 0.17, 0.90, 0.70])
    n = len(series)
    x = np.arange(len(groups))
    bw = 0.8 / n
    for i, (lab, vals, col) in enumerate(series):
        pos = x - 0.4 + bw * (i + 0.5)
        ax.bar(pos, [v if v is not None else 0 for v in vals], bw * 0.92, color=col, label=lab)
        if annotate:
            for p, v in zip(pos, vals):
                if v is not None:
                    ax.text(p, v + (ylim[1] - ylim[0]) * 0.01, f"{v:.1f}", ha="center", fontsize=6.4, color=INK)
    if refline:
        for xi, yv, txt in refline:
            ax.plot([xi - 0.42, xi + 0.42], [yv, yv], color=BAD, ls="--", lw=1)
            ax.text(xi + 0.43, yv, txt, fontsize=6.5, color=BAD, va="center")
    ax.set_xticks(x, groups, fontsize=8.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.legend(fontsize=7, frameon=False, ncol=min(n, 6), loc="upper center", bbox_to_anchor=(0.5, -0.17))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e9ecef", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(OUT, name), dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def fig_results():
    bars("f08_longtail.png", "Long-tailed clients (IF = 100, Dirichlet 0.5, K = 10, 3 seeds)",
         ["MNIST", "FashionMNIST", "CIFAR-10", "SVHN"],
         [("GeFL-F", [75.25, 58.47, 33.74, 50.49], NEUT),
          ("GeFL-F + LA", [78.92, 64.78, 37.30, 56.44], "#adb5bd"),
          ("+ HWA + LA", [87.89, 76.02, 40.73, 62.45], "#74c0fc"),
          ("Gaussian from statistics (FSG) + LA", [87.04, 73.60, 44.17, 44.23], "#ffc078"),
          ("Ours-A", [94.82, 79.67, 44.38, 54.41], OURS)], (0, 105))
    bars("f09_iid.png", "Paper's own balanced setting: best_mean_acc vs the best of the paper's ten methods",
         ["MNIST K=10", "MNIST K=50", "MNIST K=100", "FMNIST K=10", "FMNIST K=50", "CIFAR-10 K=10", "SVHN K=10"],
         [("GeFL-F (paper)", [95.47, 95.04, 94.63, 83.14, 82.21, 55.86, 76.26], NEUT),
          ("best of 10 methods in paper", [96.44, 95.04, 94.63, 84.28, 82.96, 59.36, 76.26], "#ffa8a8"),
          ("Ours-A", [97.62, 96.93, 96.77, 83.96, 82.96, 60.93, 74.38], OURS)], (50, 102), ylabel="best_mean_acc (%)")
    bars("f10_ablation.png", "Building the method on the same seeds (MNIST / FashionMNIST, long tail)",
         ["GeFL-F", "+LA", "+HWA", "+CSL", "mean anchor\n(replaces HWA)", "+MC", "+KH", "+T_s=10\n= Ours-A"],
         [("MNIST", [75.25, 78.92, 87.89, 90.40, 90.75, 92.31, 93.11, 94.82], OURS),
          ("FashionMNIST", [58.47, 64.78, 76.02, 76.60, 75.96, 77.56, 78.22, 79.67], "#e8590c")], (50, 100), h=3.4)
    bars("f11_scaling_privacy.png", "(a) More clients: MNIST long tail, K = 100       (b) Formal privacy on every Ours-A statistic, long tail, K = 10",
         ["K=100\nMNIST", "exact\nMNIST", "ε = 8\nMNIST", "ε = 2\nMNIST", "exact\nFMNIST", "ε = 8\nFMNIST", "ε = 2\nFMNIST"],
         [("GeFL-F (no formal privacy)", [75.98, 75.25, 75.25, 75.25, 58.47, 58.47, 58.47], NEUT),
          ("Ours-A", [94.30, 94.82, 90.36, 85.06, 79.67, 72.77, 65.59], OURS)], (40, 102), h=3.3)
    bars("f12_svhn.png", "SVHN: where class means carry no class information (F = 0.01)",
         ["long tail (balanced acc.)", "IID (best_mean_acc)"],
         [("GeFL-F", [50.49, 75.82], NEUT), ("mean anchor + MC (Ours-A)", [55.76, 74.38], "#ffa8a8"),
          ("mean anchor + MC-S", [61.80, 77.05], "#ffc078"), ("hybrid + MC-S", [62.89, 77.39], "#a5d8ff"),
          ("HWA rows + MC + KH (ESA's gate choice)", [65.60, 77.07], OURS)], (40, 85), w=8, h=3.3)


# ---------------------------------------------------------------- equations
EQ = {
    "eq_collapse": r"$\|r_c^{+}\|\ \leq\ \|r_c\| + p_c\,\bar G_c - (1-p_c)\,\bar D_c,\qquad \mathrm{shrinks\ whenever}\ \ p_c=\frac{m_c}{K} < \theta_c=\frac{\bar D_c}{\bar G_c+\bar D_c}$",
    "eq_cvae": r"$\mathcal{L}_G = \|\hat h - h\|_2^2 + \mathrm{KL}\left(q(z\,|\,h,y)\ \|\ \mathcal{N}(0,I)\right),\qquad \hat h = \mathrm{dec}(z,\ W^\top e_y)$",
    "eq_fedavg": r"$\theta^{(r+1)} = \sum_{k=1}^{K} \frac{n_k}{n}\,\theta_k^{(r)},\qquad W_c^{(r+1)} = \frac{1}{K}\sum_{k} W_{c,k}^{(r)}\ \ \mathrm{(flat)}$",
    "eq_hwa": r"$w_c \leftarrow \frac{\sum_k E(n_{kc})\, w_c^{(k)}}{\sum_k E(n_{kc})},\qquad E(n)=\frac{1-\beta^{\,n}}{1-\beta},\ \ \beta = 0.999$",
    "eq_la": r"$\ell_{LA}(h,y) = -\log \frac{\exp(f_y(h)+\tau\log\pi_{k,y})}{\sum_{j}\exp(f_j(h)+\tau\log\pi_{k,j})}$",
    "eq_csl": r"$t(\tilde h) = (1-\beta)\,e_y + \beta\,\frac{1}{G}\sum_{g=1}^{G}\mathrm{softmax}\left(h_g(\tilde h)\right),\qquad \beta = 0.5$",
    "eq_stats": r"$m_c=\frac{S_c}{n_c},\quad V_c=\frac{Q_c}{n_c}-\|m_c\|^2,\quad \mu_c=\frac{\Phi_c}{n_c}$",
    "eq_pc": r"$\tilde h = \mathrm{ReLU}\left(m_y + \mathrm{dec}(z,\ A\,m_y)\right),\qquad z\sim\mathcal{N}(0,I)$",
    "eq_sep": r"$W_1(G_c, G_{c'})\ \leq\ (1+L\,\|A\|_2)\ \|m_c-m_{c'}\|$",
    "eq_mc": r"$T^\star(x) = m_c + \sqrt{V_c/\tilde V_c}\,(x-\tilde m_c)\ \ =\ \ \arg\min_{T}\ \mathbb{E}\|T(X)-X\|^2\ \ \mathrm{s.t.}\ \ \mathbb{E}T=m_c,\ \mathbb{E}\|T-m_c\|^2=V_c$",
    "eq_gelbrich": r"$A_c = \tilde\Sigma_c^{-1/2}\left(\tilde\Sigma_c^{1/2}\,\Sigma_c\,\tilde\Sigma_c^{1/2}\right)^{1/2}\tilde\Sigma_c^{-1/2},\qquad \min\ \mathbb{E}\|T(X)-X\|^2 = \|m_c-\tilde m_c\|^2 + \mathcal{B}^2(\Sigma_c,\tilde\Sigma_c)$",
    "eq_mcb": r"$\hat\Sigma_c = \frac{n_c\,\Sigma_c + k\,\tau_c\,\tilde\Sigma_c}{n_c + k},\qquad \tau_c = \frac{\mathrm{tr}\,\Sigma_c}{\mathrm{tr}\,\tilde\Sigma_c}\qquad (n_c\to 0:\ \mathrm{MC};\ \ n_c\to\infty:\ \mathrm{exact\ covariance})$",
    "eq_kh": r"$|\mathbb{E}_{G_c}\ell - \mathbb{E}_{P_c}\ell|\ \leq\ B\cdot\mathrm{MMD}(G_c,P_c),\qquad x_{t+1}=\arg\max_{x\in\mathcal{X}_c}\ \langle\phi(x),\ \mu_c\rangle - \frac{1}{t+1}\sum_{s\leq t}\langle\phi(x),\phi(x_s)\rangle$",
    "eq_budget": r"$\mathcal{E}(w) = \frac{\sigma^2(1-w)}{n_r} + w^2 b^2,\qquad w^\star = \frac{\sigma^2}{2\,n_r\,b^2}\quad (\mathrm{optimal\ synthetic\ share\ falls\ with\ label\ bias}\ b)$",
    "eq_bbc": r"$b \leftarrow b - \log\left(C\ \bar q(b)\right),\qquad \bar q_c(b) = \mathbb{E}_{\tilde h\sim \mathrm{balanced}}\ \mathrm{softmax}(f(\tilde h)+b)_c$",
    "eq_mixup": r"$\mathbb{E}_{Q}\,\ell_{mix}\ \approx\ \Psi\left(\{\pi_c,\ \mu_c,\ \Sigma_c\}_{c=1}^{C}\right)\ +\ O\left(\mathbb{E}(1-\lambda)^3\|x'-x\|^3\right)$",
    "eq_dp": r"$\sigma_{rel} = \sqrt{2/\rho},\qquad \rho = \left(\sqrt{\log(1/\delta)+\varepsilon}-\sqrt{\log(1/\delta)}\right)^2\quad \mathrm{(zCDP,\ split\ over\ 4\ releases)}$",
    "eq_lda": r"$\mathrm{FSG}:\ \tilde h\sim\mathcal{N}(m_c,\Sigma_{pooled})\ \Rightarrow\ \mathrm{Bayes\ head} = \arg\max_c\ m_c^\top\Sigma^{-1}h-\frac{1}{2}m_c^\top\Sigma^{-1}m_c\ \ (\mathrm{LDA})$",
    "eq_tv": r"$R_{real}(h)\ \leq\ R_{syn}(h) + \ell_{\max}\cdot\mathrm{TV}(p_{syn},\ p_{real})$",
}


def equations():
    for name, tex in EQ.items():
        fig = plt.figure(figsize=(0.1, 0.1), dpi=220)
        fig.text(0, 0, tex, fontsize=13, color=INK)
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, bbox_inches="tight", pad_inches=0.06, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    fig_overview(); fig_timeline(); fig_arch(); fig_round(); fig_problem(); fig_journey(); fig_esa(); fig_results(); equations()
    print(sorted(os.listdir(OUT)))
