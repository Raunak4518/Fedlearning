"""Draws the overview of FedECS-LT (paper Fig. overview, slot fig_overview) as a page-wide IEEE figure: the stages of
the pipeline as implemented in fedecs_lt/src (server.py run_pipeline; client.py), what the clients compute, what the
server computes, and which quantities pass through secure aggregation. Not a result: a diagram of the procedure.

    python docs/papers/overview_figure.py      # writes generated/figures/fig_overview.pdf (+ .png)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "fedecs_lt", "src"))
import plot_style as ps  # noqa: E402

ps.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

CLIENT, SERVER, SECURE = "#0072B2", "#D55E00", "#009E73"
# (title, [(who, text)]): who = c (client), s (server), x (through secure aggregation)
STAGES = [
    ("Stage I", [("c", "train extractor $\\phi$\nand own head"), ("s", "FedAvg"), ("c", "freeze $\\phi$;\n$h=\\phi(x)$ once")]),
    ("Class statistics", [("x", "$n_{kc}$, $\\sum h$,\n$\\sum\\|h\\|^2$"), ("s", "$\\mu_c$, $V_c$,\ngate $F$")]),
    ("Stage II: generator", [("s", "$F\\geq F_0$: anchored\nto $\\mu_c$"), ("x", "$F<F_0$: class rows\nby CCWA"),
                             ("c", "train (Adam)"), ("s", "FedAvg; moment\ncorrection")]),
    ("Stage III: heads", [("x", "consensus soft labels\non a shared pool"),
                          ("c", "$T_s$ synthetic epochs"), ("c", "$T_r$ real epochs,\nlogit adjustment"),
                          ("s", "average heads\nper architecture")]),
    ("Exact readout", [("x", "$n_c$, $\\sum e$, $\\sum ee^{\\top}$\nper architecture"),
                    ("s", "exact readout\n(class-balanced LDA)")]),
]
WIDTHS = [1.0, 1.0, 1.15, 1.2, 1.15]


def main():
    fig, ax = plt.subplots(figsize=(ps.PAGE_IN, 2.15))
    ax.set_xlim(0, sum(WIDTHS) + 0.08 * (len(WIDTHS) - 1))
    ax.set_ylim(0, 1)
    ax.axis("off")
    x, boxes = 0.0, []
    for (title, items), w in zip(STAGES, WIDTHS):
        ax.add_patch(FancyBboxPatch((x, 0.06), w, 0.84, boxstyle="round,pad=0.0,rounding_size=0.03",
                                    fc="#F4F4F4", ec="#555555", lw=0.6))
        ax.text(x + w / 2, 0.84, title, ha="center", va="center", fontsize=8, fontweight="bold")
        y = 0.70
        step = 0.155
        for who, text in items:
            col = {"c": CLIENT, "s": SERVER, "x": SECURE}[who]
            ax.text(x + 0.04, y, {"c": "C", "s": "S", "x": "SA"}[who], ha="left", va="center", fontsize=6,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none"))
            ax.text(x + 0.25, y, text, ha="left", va="center", fontsize=6.5, linespacing=1.05)
            y -= step
        boxes.append((x, w))
        x += w + 0.08
    for (x0, w0), (x1, _) in zip(boxes, boxes[1:]):
        ax.add_patch(FancyArrowPatch((x0 + w0 - 0.01, 0.84), (x1 + 0.01, 0.84), arrowstyle="-|>", mutation_scale=7,
                                     lw=0.6, color="#333333"))
    ax.text(0.0, 0.0, "C: every client (its own head architecture)   S: server   SA: summed over clients by secure "
            "aggregation (SecAgg+)", ha="left", va="bottom", fontsize=6.5, color="#333333")
    out = os.path.join(HERE, "generated", "figures")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, "fig_overview.pdf"), bbox_inches="tight", pad_inches=0.01)
    fig.savefig(os.path.join(out, "fig_overview.png"), dpi=200, bbox_inches="tight", pad_inches=0.01)
    print("wrote", os.path.join(out, "fig_overview.pdf"))


if __name__ == "__main__":
    main()
