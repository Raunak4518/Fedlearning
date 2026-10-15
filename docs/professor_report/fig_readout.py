"""Readout-gap figure for the handout (f16). Numbers: docs/research/readout_report.md §2 (3 seeds, full schedule)."""
import matplotlib.pyplot as plt
from make_figures import save, INK, MUTED, BAD, GOOD, NEUT, OURS

ROWS = {  # dataset: [(method, head, oracle)]
    "MNIST LT": [("GeFL-F", 74.2, 90.7), ("+ HWA + LA", 87.8, 92.4), ("Ours-A", 94.8, 94.9)],
    "FashionMNIST LT": [("GeFL-F", 58.7, 79.2), ("+ HWA + LA", 76.0, 80.1), ("Ours-A", 79.7, 81.6)],
}


def fig_readout():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    for ax, (ds, rows) in zip(axes, ROWS.items()):
        for i, (m, h, o) in enumerate(rows):
            ax.bar(i - 0.2, h, 0.4, color=OURS if m == "Ours-A" else NEUT, label="head (as trained)" if i == 0 else None)
            ax.bar(i + 0.2, o, 0.4, color=GOOD, alpha=0.75, label="last layer refit on real data" if i == 0 else None)
            ax.text(i - 0.2, h + 0.8, f"{h:.1f}", ha="center", fontsize=7.5, color=INK)
            ax.text(i + 0.2, o + 0.8, f"{o:.1f}", ha="center", fontsize=7.5, color=INK)
            ax.text(i, 52, f"gap {o - h:.1f}", ha="center", fontsize=7.5, color="white", fontweight="bold")
        ax.set_xticks(range(3), [r[0] for r in rows], fontsize=8)
        ax.set_ylim(50, 100)
        ax.set_title(ds, fontsize=9, loc="left")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("balanced accuracy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7.5, frameon=False, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "f16_readout.png")


if __name__ == "__main__":
    fig_readout()
