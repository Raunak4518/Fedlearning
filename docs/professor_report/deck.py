"""Visual walkthrough deck (landscape PDF) for the professor meeting. python deck.py [page ...]"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "deck")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})

INK, MUTED, LINE = "#1d2633", "#5d6878", "#c9d1dc"
BLUE, BLUE_L = "#1f5fbf", "#e7effb"
GREEN, GREEN_L = "#2b8a3e", "#e9f6ec"
ORANGE, ORANGE_L = "#d9480f", "#fdf0e6"
PURPLE, PURPLE_L = "#6741d9", "#f1ecfd"
TEAL, TEAL_L = "#0b7285", "#e3f4f7"
RED, RED_L = "#c92a2a", "#fdecec"
GREY_L = "#f3f5f8"
W, H = 13.333, 7.5


def page(title, subtitle=None, num=None):
    fig = plt.figure(figsize=(W, H), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56.25)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 52.2), 100, 4.05, fc=BLUE, ec="none"))
    ax.text(2, 54.2, title, fontsize=19, color="white", fontweight="bold", va="center")
    if subtitle:
        ax.text(2, 50.2, subtitle, fontsize=12, color=MUTED, va="center")
    if num is not None:
        ax.text(98, 1.2, str(num), fontsize=9, color=MUTED, ha="right")
    return fig, ax


def box(ax, x, y, w, h, title=None, lines=(), fc="white", ec=INK, tsize=11, lsize=9.5, tcolor=None, lw=1.4, align="center", r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    n = (1 if title else 0) + len(lines)
    if n == 0:
        return
    xt = x + w / 2 if align == "center" else x + 1.0
    ha = "center" if align == "center" else "left"
    gap = min(2.6, (h - 1.2) / max(n, 1)) if n > 1 else 0
    y0 = y + h / 2 + gap * (n - 1) / 2
    i = 0
    if title:
        ax.text(xt, y0, title, fontsize=tsize, fontweight="bold", color=tcolor or ec, ha=ha, va="center")
        i = 1
    for j, l in enumerate(lines):
        ax.text(xt, y0 - gap * (i + j), l, fontsize=lsize, color=INK, ha=ha, va="center")


def tensor(ax, x, y, w, h, shape, note=None, fc=GREY_L, ec=MUTED):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=1.1, ls="-"))
    ax.text(x + w / 2, y + h / 2 + (0.8 if note else 0), shape, fontsize=10, fontweight="bold", color=INK, ha="center", va="center")
    if note:
        ax.text(x + w / 2, y + h / 2 - 1.1, note, fontsize=8, color=MUTED, ha="center", va="center")


def arr(ax, x1, y1, x2, y2, color=INK, lw=1.6, text=None, tx=None, ty=None, ts=8.5, ls="-", rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text(tx if tx is not None else (x1 + x2) / 2, ty if ty is not None else (y1 + y2) / 2 + 0.9, text,
                fontsize=ts, color=color, ha="center", va="center")


def chip(ax, x, y, text, fc, ec, size=9):
    ax.text(x, y, text, fontsize=size, color=ec, ha="left", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1))


# ======================================================================== pages
def p_generator(num):
    fig, ax = page("Shared model 2: the feature generator (CVAE-F)",
                   "Learns to make fake features of any class. All clients train it together in Stage II (FedAvg), then it is frozen.", num)
    # training path label
    chip(ax, 2, 46.5, "TRAINING PATH (Stage II, on each client)", BLUE_L, BLUE)
    tensor(ax, 2, 36.5, 9.5, 6.5, "3×16×16", "real feature h")
    tensor(ax, 2, 28.5, 9.5, 6.5, "1×16×16", "label map (= y)")
    arr(ax, 11.5, 39.7, 14.2, 36.5)
    arr(ax, 11.5, 31.7, 14.2, 34.5)
    tensor(ax, 14.2, 32, 8, 6.5, "4×16×16", "joined input")
    # encoder
    box(ax, 24.5, 26.5, 33, 17.5, fc=PURPLE_L, ec=PURPLE)
    ax.text(41, 42.4, "ENCODER  q(z | h, y)", fontsize=11, fontweight="bold", color=PURPLE, ha="center")
    xs = [25.6, 33.6, 41.6, 49.6]
    shapes = ["64×8×8", "128×4×4", "256×2×2", "512×1×1"]
    for i, (x, s) in enumerate(zip(xs, shapes)):
        tensor(ax, x, 33, 6.8, 5.5, s, fc="white", ec=PURPLE)
        if i:
            arr(ax, xs[i - 1] + 6.8, 35.75, x, 35.75, color=PURPLE)
    ax.text(41, 30.6, "each step = Conv 4×4, stride 2 → BatchNorm → ReLU   (size halves, channels double)", fontsize=8.5, color=PURPLE, ha="center")
    ax.text(41, 28.3, "flatten 512 → Linear → mean μ (16)   and   Linear → log-variance (16)", fontsize=8.5, color=PURPLE, ha="center")
    arr(ax, 22.2, 35.25, 25.6, 35.75, color=PURPLE)
    # latent
    tensor(ax, 59.5, 32, 9, 6.5, "z : 16 numbers", "z = μ + σ·noise", fc=ORANGE_L, ec=ORANGE)
    arr(ax, 57.5, 35.25, 59.5, 35.25, color=ORANGE)
    # decoder
    box(ax, 70.5, 14.5, 27.5, 29.5, fc=TEAL_L, ec=TEAL)
    ax.text(84.25, 42.4, "DECODER  p(h | z, y)", fontsize=11, fontweight="bold", color=TEAL, ha="center")
    tensor(ax, 71.5, 35.6, 12, 4.8, "Linear 16 → 512", fc="white", ec=TEAL)
    box(ax, 84.8, 34.6, 12.4, 6.8, fc=RED_L, ec=RED, lw=2)
    ax.text(91, 39.6, "+ label row  w_y", fontsize=10, fontweight="bold", color=RED, ha="center")
    ax.text(91, 37.5, "table W: 10 rows × 512", fontsize=8.5, color=RED, ha="center")
    ax.text(91, 35.7, "(1 row per class)", fontsize=8.5, color=RED, ha="center")
    ax.text(84.25, 33.0, "add → ReLU → reshape to 512×1×1", fontsize=8.5, color=TEAL, ha="center")
    dsh = ["256×2×2", "128×4×4", "64×8×8"]
    for i, s in enumerate(dsh):
        tensor(ax, 72.5 + i * 8.4, 25.2, 7.4, 5, s, fc="white", ec=TEAL)
        if i:
            arr(ax, 72.5 + (i - 1) * 8.4 + 7.4, 27.7, 72.5 + i * 8.4, 27.7, color=TEAL)
    ax.text(84.25, 22.8, "each step = Transposed Conv 4×4, stride 2 (size doubles)", fontsize=8.5, color=TEAL, ha="center")
    tensor(ax, 81.0, 15.6, 15.5, 5.6, "3×16×16", "fake feature ĥ (shape of h)", fc="white", ec=TEAL)
    arr(ax, 93.6, 25.2, 91.0, 21.2, color=TEAL)
    arr(ax, 68.5, 35.25, 71.5, 38.0, color=ORANGE)
    # generation path
    chip(ax, 2, 22.5, "GENERATION PATH (Stage III, on each client, no real data)", ORANGE_L, ORANGE)
    tensor(ax, 2, 13.5, 12, 6, "z ~ N(0, 1)", "16 random numbers", fc=ORANGE_L, ec=ORANGE)
    tensor(ax, 16, 13.5, 13, 6, "class y", "chosen: 0 … 9", fc=ORANGE_L, ec=ORANGE)
    ax.text(31, 16.5, "encoder is NOT used  →  only the decoder runs", fontsize=10, color=ORANGE, va="center")
    arr(ax, 57, 16.5, 70.5, 18.4, color=ORANGE, lw=1.6, ls="--")
    # facts strip
    facts = [("Loss (per client)", "‖ĥ − h‖²  +  KL( N(μ, σ) ‖ N(0, 1) )"),
             ("Optimiser", "Adam, lr 0.001, weight decay 0.001, 5 local epochs, batch 64"),
             ("Size", "5.55 M numbers ≈ 22 MB   (CIFAR-10: z has 50 numbers, 5.61 M)"),
             ("Sent to server", "ALL weights, every round, 100 rounds (200 CIFAR-10)")]
    for i, (k, v) in enumerate(facts):
        y = 9.3 - i * 2.35
        ax.text(2, y, k, fontsize=10, fontweight="bold", color=INK, va="center")
        ax.text(19, y, v, fontsize=10, color=INK, va="center")
    box(ax, 62, 1.6, 36, 8.6, fc=RED_L, ec=RED, lw=1.6)
    ax.text(80, 8.4, "Why this matters for our problem", fontsize=10.5, fontweight="bold", color=RED, ha="center")
    ax.text(80, 5.9, "Row w_y is the ONLY part that knows the class.", fontsize=9.5, color=INK, ha="center")
    ax.text(80, 3.6, "It is averaged by FedAvg like everything else.", fontsize=9.5, color=INK, ha="center")
    return fig


def p_hwa_card(num):
    fig, ax = page("Method card: HWA (holder-weighted aggregation)", "Kept in the final method", num)
    # mini pipeline
    stages = [("Stage I", "feature extractor", PURPLE, PURPLE_L, False), ("Stage II", "generator", ORANGE, ORANGE_L, True),
              ("Stage III", "client heads", TEAL, TEAL_L, False)]
    for i, (t, s, ec, fc, hi) in enumerate(stages):
        x = 2 + i * 16
        box(ax, x, 42, 13.5, 6, t, [s], fc=fc, ec=RED if hi else ec, lw=3 if hi else 1.2, tsize=10, lsize=9, tcolor=ec)
        if i:
            arr(ax, x - 2.5, 45, x, 45, color=MUTED)
    ax.text(18.75, 40.2, "▲ acts here: server, when averaging the label rows", fontsize=9, color=RED, ha="center", fontweight="bold")
    rows = [("PROBLEM", "A client that does not have class c still sends its row w_c. Its copy only shrinks (weight decay), and the average drags the true row toward zero."),
            ("WHERE", "Server, Stage II, every round. Only the 10 label rows change; every other weight is averaged as before."),
            ("CHANGE", "Average row c only over clients that HAVE class c, weighted by how many samples they hold."),
            ("WHY IT HELPS", "No client that never saw class c can pull its row down any more."),
            ("COST", "Server needs per-class sample counts (sums only, works with secure aggregation).")]
    import textwrap
    y = 36
    for k, v in rows:
        ax.text(2, y, k, fontsize=10.5, fontweight="bold", color=BLUE, va="top")
        wrapped = textwrap.wrap(v, 62)
        for i, l in enumerate(wrapped):
            ax.text(15, y - i * 2.1, l, fontsize=10.5, color=INK, va="top")
        y -= 2.1 * len(wrapped) + 1.4
    ax.text(15, y - 1.6, r"$w_c \leftarrow \frac{\sum_k E(n_{kc})\, w_c^{(k)}}{\sum_k E(n_{kc})}$" + "      (E(n) = effective number of samples; E(0) = 0)",
            fontsize=13, color=INK, va="center")
    # result table
    ax.text(62, 36.5, "RESULT  (long tail, 3 seeds, balanced accuracy %)", fontsize=10.5, fontweight="bold", color=BLUE)
    hdr = ["", "MNIST", "FMNIST", "CIFAR-10", "SVHN"]
    data = [["before (GeFL-F + LA)", "78.9", "64.8", "37.3", "56.4"], ["after (+ HWA)", "87.9", "76.0", "40.7", "62.5"], ["gain", "+9.0", "+11.2", "+3.4", "+6.1"]]
    cw = [13, 6, 6, 7, 6]
    xs = [62]
    for c in cw[:-1]:
        xs.append(xs[-1] + c)
    for j, t in enumerate(hdr):
        ax.text(xs[j] + 0.5, 33.5, t, fontsize=9.5, fontweight="bold", color=MUTED)
    for i, r in enumerate(data):
        yy = 30.5 - i * 3
        for j, t in enumerate(r):
            ax.text(xs[j] + 0.5, yy, t, fontsize=10, color=GREEN if i == 2 and j else INK, fontweight="bold" if i == 2 else "normal")
    ax.text(62, 20.5, "Tail/head row-size ratio:   0.33 → 0.80 (MNIST),   0.22 → 1.01 (FMNIST)", fontsize=9.5, color=INK)
    box(ax, 61, 11, 38, 6.5, fc=GREEN_L, ec=GREEN)
    ax.text(80, 14.25, "VERDICT: KEPT   (used when class means are weak, e.g. SVHN)", fontsize=10.5, color=GREEN, ha="center", fontweight="bold")
    return fig


PAGES = {"generator": p_generator, "hwa": p_hwa_card}

if __name__ == "__main__":
    names = sys.argv[1:] or list(PAGES)
    for i, n in enumerate(names, 1):
        fig = PAGES[n](i)
        fig.savefig(os.path.join(OUT, f"sample_{n}.png"), dpi=110, facecolor="white")
        plt.close(fig)
    print("ok", names)
