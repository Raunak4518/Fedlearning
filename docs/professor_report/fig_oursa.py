"""Detailed Ours-A figures for the handout (f13 PC-VAE, f14 Stage III round, f15 MC + KH illustration)."""
import numpy as np
import matplotlib.pyplot as plt
from make_figures import (box, arrow, canvas, save, OUT, INK, MUTED, C_SERVER, C_CLIENT, C_GEN, C_FE, C_HEAD,
                          E_SERVER, E_CLIENT, E_GEN, E_FE, E_HEAD, GOOD, BAD, NEUT, OURS)
import os


def t(ax, x, y, s, **k):
    k.setdefault("fontsize", 8)
    k.setdefault("ha", "center")
    k.setdefault("va", "center")
    k.setdefault("color", INK)
    ax.text(x, y, s, **k)


# ------------------------------------------------ f13: PC-VAE with shapes, against CVAE-F
def fig_pcvae():
    fig, ax = canvas(10, 6.4)
    t(ax, 0.5, 0.975, "PC-VAE: the Stage II generator of Ours-A (MNIST / FMNIST / SVHN shapes; CIFAR-10 in brackets)",
      fontsize=11.5, fontweight="bold", va="top")

    # ---- training path (encoder)
    t(ax, 0.02, 0.905, "TRAINING  (each client, Stage II, 5 local epochs per round)", ha="left", fontsize=8.5,
      fontweight="bold", color=E_FE)
    box(ax, 0.02, 0.73, 0.11, 0.14, "real feature h\n3×16×16\n(10×16×16)", fc=C_CLIENT, ec=E_CLIENT, fs=7.6)
    box(ax, 0.02, 0.57, 0.11, 0.12, "exact class\nmean m_y\n3×16×16", fc=C_SERVER, ec=OURS, fs=7.6, lw=1.8)
    box(ax, 0.16, 0.65, 0.10, 0.14, "join\n[h ; m_y]\n6×16×16\n(20×16×16)", fc="white", ec=INK, fs=7.4)
    arrow(ax, 0.13, 0.80, 0.16, 0.74)
    arrow(ax, 0.13, 0.63, 0.16, 0.70)
    enc = ["64×8×8", "128×4×4", "256×2×2", "512×1×1"]
    x = 0.29
    arrow(ax, 0.26, 0.72, x, 0.72, color=E_FE)
    for i, s in enumerate(enc):
        box(ax, x, 0.66, 0.085, 0.12, s, fc=C_FE, ec=E_FE, fs=7.6)
        if i < 3:
            arrow(ax, x + 0.085, 0.72, x + 0.105, 0.72, color=E_FE)
        x += 0.105
    t(ax, 0.50, 0.625, "Encoder: 4 × (Conv 4×4, stride 2 → BN → ReLU); each step halves the size",
      fontsize=7.4, color=E_FE)
    box(ax, 0.73, 0.66, 0.10, 0.12, "Linear\n→ μ, log σ²\n16 + 16\n(50 + 50)", fc=C_FE, ec=E_FE, fs=7.3)
    arrow(ax, 0.705, 0.72, 0.73, 0.72, color=E_FE)
    box(ax, 0.86, 0.66, 0.12, 0.12, "z = μ + σ·ε\n16 numbers\n(50)", fc="#fff4e6", ec=E_GEN, fs=7.6, lw=1.6)
    arrow(ax, 0.83, 0.72, 0.86, 0.72, color=E_GEN)

    # ---- decoder
    t(ax, 0.02, 0.535, "DECODER  (training and generation)", ha="left", fontsize=8.5, fontweight="bold", color=E_GEN)
    box(ax, 0.02, 0.38, 0.12, 0.11, "Linear 16→512\nfrom z", fc=C_GEN, ec=E_GEN, fs=7.6)
    box(ax, 0.02, 0.245, 0.12, 0.11, "Linear 768→512\n(2560→512)\nfrom m_y:  A m_y", fc=C_SERVER, ec=OURS, fs=7.4, lw=1.8)
    t(ax, 0.165, 0.37, "+", fontsize=16, fontweight="bold", color=OURS)
    arrow(ax, 0.92, 0.66, 0.08, 0.49, color=E_GEN, rad=0.0)
    box(ax, 0.19, 0.31, 0.10, 0.12, "ReLU\n512×1×1", fc=C_GEN, ec=E_GEN, fs=7.6)
    arrow(ax, 0.14, 0.43, 0.19, 0.39, color=E_GEN)
    arrow(ax, 0.14, 0.30, 0.19, 0.34, color=OURS)
    dec = ["256×2×2", "128×4×4", "64×8×8", "3×16×16"]
    x = 0.32
    arrow(ax, 0.29, 0.37, x, 0.37, color=E_GEN)
    for i, s in enumerate(dec):
        box(ax, x, 0.31, 0.085, 0.12, s, fc=C_GEN, ec=E_GEN, fs=7.6)
        if i < 3:
            arrow(ax, x + 0.085, 0.37, x + 0.105, 0.37, color=E_GEN)
        x += 0.105
    t(ax, 0.53, 0.275, "4 × Transposed Conv 4×4, stride 2; each step doubles the size", fontsize=7.4, color=E_GEN)
    box(ax, 0.76, 0.29, 0.22, 0.16, "output\nh̃ = ReLU( m_y + decoder )\n3×16×16  (10×16×16)\n= one fake feature of class y",
        fc="white", ec=OURS, fs=7.6, lw=2.0)
    arrow(ax, 0.725, 0.37, 0.76, 0.37, color=OURS)

    # ---- bottom: loss and comparison
    box(ax, 0.02, 0.03, 0.46, 0.20, "Loss  =  ‖h̃ − h‖²  +  KL( N(μ, σ²) ‖ N(0, I) )\n"
        "Adam · lr 0.001 · weight decay 0.001 · batch 64\n100 rounds (CIFAR-10: 200) · FedAvg of ALL weights\n"
        "5.94 M parameters ≈ 23.7 MB per upload", fc="#f8f9fa", ec="#adb5bd", fs=7.8)
    box(ax, 0.52, 0.03, 0.46, 0.20, "CVAE-F (GeFL-F):  class y enters as a learned row w_y (10 × 512)\n"
        "→ clients without class c shrink w_c → rare classes collapse\n\n"
        "PC-VAE (Ours-A):  class y enters as the exact mean m_y (not learned)\n"
        "→ no class-specific weight → nothing for FedAvg to shrink",
        fc="#e7f5ff", ec=OURS, fs=7.8, lw=1.6)
    save(fig, "f13_pcvae.png")


# ------------------------------------------------ f14: one Stage III round
def fig_round3():
    fig, ax = canvas(10, 4.6)
    t(ax, 0.5, 0.96, "One Stage III round of Ours-A (50 rounds; SVHN / CIFAR-10: 100)", fontsize=11.5,
      fontweight="bold", va="top")
    # server lane
    box(ax, 0.01, 0.60, 0.08, 0.24, "SERVER", fc=C_SERVER, ec=E_SERVER, fs=8.5, bold=True)
    box(ax, 0.01, 0.12, 0.08, 0.40, "CLIENT k\n(own head\narch g)", fc=C_CLIENT, ec=E_CLIENT, fs=8, bold=True)
    box(ax, 0.11, 0.60, 0.25, 0.24, "① make this round's fake pool\nPC-VAE → MC → KH\nlabels uniform over 10 classes\n"
        "size ≈ 10 × client data", fc="white", ec=OURS, fs=7.6, lw=1.8)
    box(ax, 0.38, 0.60, 0.27, 0.24, "② consensus label per fake sample\naverage of softmax over the\n10 architectures' current heads\n"
        "(from summed predictions only)", fc="white", ec=OURS, fs=7.6, lw=1.8)
    arrow(ax, 0.36, 0.72, 0.38, 0.72, color=OURS)
    box(ax, 0.11, 0.12, 0.30, 0.40, "③ 10 epochs on the fake pool\n\ninput h̃: 3×16×16\ntarget t = ½·one-hot(y)\n        + ½·consensus\n"
        "loss: cross-entropy with soft target\nSGD lr 0.1", fc=C_HEAD, ec=E_HEAD, fs=7.6)
    box(ax, 0.44, 0.12, 0.27, 0.40, "④ 5 epochs on own real features\n\ninput h: 3×16×16\nloss: CE( head(h) + log π_k , y )\n"
        "π_k = own label shares\n(never leaves the client)\nSGD lr 0.1", fc=C_HEAD, ec=E_HEAD, fs=7.6)
    arrow(ax, 0.51, 0.60, 0.30, 0.52, color=OURS, text="pool + soft labels", tx=0.40, ty=0.555)
    arrow(ax, 0.41, 0.32, 0.44, 0.32, color=E_HEAD)
    box(ax, 0.74, 0.12, 0.24, 0.40, "⑤ upload\n\nhead weights → averaged with\nheads of the SAME architecture\n\n"
        "prediction sums on the pool\n(≈ 24 KB) → next consensus", fc="white", ec=E_CLIENT, fs=7.6)
    arrow(ax, 0.71, 0.32, 0.74, 0.32, color=E_CLIENT)
    box(ax, 0.68, 0.60, 0.30, 0.24, "⑥ server averages heads per\narchitecture group and\nsends them back", fc=C_SERVER,
        ec=E_SERVER, fs=7.6)
    arrow(ax, 0.86, 0.52, 0.86, 0.60, color=E_SERVER)
    t(ax, 0.5, 0.045, "Test time: image → frozen FE → 3×16×16 → own head → 10 logits (no generator, no log π_k offset)",
      fontsize=7.8, color=MUTED, style="italic")
    save(fig, "f14_round3.png")


# ------------------------------------------------ f15: MC and KH on a toy 2-D class
def fig_mc_kh():
    rng = np.random.default_rng(3)
    Lr = np.array([[1.0, 0.0], [0.55, 0.45]])
    real = rng.standard_normal((400, 2)) @ Lr.T + np.array([2.0, 1.0])
    fake = rng.standard_normal((600, 2)) @ (0.55 * Lr).T + np.array([1.2, 0.4])
    fake[:120] = rng.standard_normal((120, 2)) * 0.25 + np.array([0.6, 1.4])   # an off-class mode
    m, V = real.mean(0), ((real - real.mean(0)) ** 2).sum(1).mean()
    mt, Vt = fake.mean(0), ((fake - fake.mean(0)) ** 2).sum(1).mean()
    mc = m + np.sqrt(V / Vt) * (fake - mt)
    # kernel herding with an RBF kernel toward the real embedding, n = 150 of 600
    def k(a, b, s=0.6):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-d / (2 * s * s))
    Kcr = k(mc, real).mean(1)
    Kcc = k(mc, mc)
    pick, acc = [], np.zeros(len(mc))
    taken = np.zeros(len(mc), bool)
    for i in range(150):
        sc = Kcr - acc / (i + 1)
        sc[taken] = -np.inf
        j = int(sc.argmax()); pick.append(j); taken[j] = True
        acc += Kcc[:, j]
    kh = mc[pick]

    def mmd(a):
        return float(np.sqrt(max(k(a, a).mean() + k(real, real).mean() - 2 * k(a, real).mean(), 0)))
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), dpi=200, sharex=True, sharey=True)
    panels = [(fake, "(a) generator output", NEUT), (mc, "(b) after MC: exact mean and spread", "#f08c00"),
              (kh, "(c) after KH: keep n of 4n by fingerprint", OURS)]
    for ax, (pts, title, col) in zip(axs, panels):
        ax.scatter(real[:, 0], real[:, 1], s=5, c="#2f9e44", alpha=0.25, label="real class c (unseen by server)")
        ax.scatter(pts[:, 0], pts[:, 1], s=6, c=col, alpha=0.7, label="fake class c")
        ax.scatter(*m, marker="x", s=70, c="black", lw=2, label="exact mean m_c")
        ax.scatter(*pts.mean(0), marker="+", s=90, c=col, lw=2)
        ax.set_title(title, fontsize=9, loc="left")
        ax.text(0.03, 0.04, f"MMD to real = {mmd(pts):.2f}", transform=ax.transAxes, fontsize=8, color=INK,
                bbox=dict(fc="white", ec="#dee2e6"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axs[0].legend(fontsize=6.8, frameon=False, loc="upper left")
    fig.suptitle("What MC and KH do to one class (2-D illustration of the 768-D feature space)", fontsize=10.5,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "f15_mc_kh.png"), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_pcvae(); fig_round3(); fig_mc_kh()
    print("ok")
