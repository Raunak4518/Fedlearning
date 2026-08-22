"""
utils/visualize.py

Publication-quality visualization suite for federated learning experiments.
All functions save to ``{out_dir}/plots/`` and optionally return the Figure.
Uses matplotlib + seaborn for polished, research-paper-ready charts.

Every function is standalone -- call any subset from the engine or from a
post-hoc analysis script. All accept pre-computed data (dicts/arrays), so
they never import torch or touch a model directly.
"""
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server compatibility
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import seaborn as sns
    _HAS_SEABORN = True
except ImportError:
    _HAS_SEABORN = False

# ---- Shared style --------------------------------------------------------

_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def _apply_style():
    if _HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette=_PALETTE, font_scale=1.05)
        sns.set_context("paper", rc={
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "#FAFAFA",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 11,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        })


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _save(fig, path: str):
    _ensure_dir(path)
    fig.savefig(path)
    plt.close(fig)


# ---- 1. Training curves --------------------------------------------------

def plot_training_curves(history: List[dict], out_path: str,
                         title: str = "Training Progress") -> str:
    """Plot loss and accuracy (overall + per-bucket) vs. round.

    Args:
        history: list of row-dicts from engine, each with 'round',
                 'train_loss', 'acc_overall', 'acc_head', 'acc_medium',
                 'acc_tail', etc.
        out_path: full path to save the figure.
        title: plot title.
    Returns:
        The path the figure was saved to.
    """
    _apply_style()
    rounds = [r["round"] for r in history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    # Loss
    losses = [r.get("train_loss", float("nan")) for r in history]
    ax_loss.plot(rounds, losses, color=_PALETTE[0], linewidth=2, marker="o",
                 markersize=4, label="Train Loss")
    if any("gen_loss" in r for r in history):
        gen_losses = [r.get("gen_loss", float("nan")) for r in history]
        ax_loss.plot(rounds, gen_losses, color=_PALETTE[1], linewidth=2,
                     marker="s", markersize=4, label="Gen Loss", linestyle="--")
    ax_loss.set_xlabel("Communication Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend()

    # Accuracy
    acc_keys = [k for k in history[0].keys() if k.startswith("acc_")]
    bucket_order = ["acc_overall", "acc_head", "acc_medium", "acc_tail"]
    ordered_keys = [k for k in bucket_order if k in acc_keys]
    ordered_keys += [k for k in acc_keys if k not in ordered_keys]

    for i, key in enumerate(ordered_keys):
        label = key.replace("acc_", "").replace("_", " ").title()
        vals = [r.get(key, float("nan")) for r in history]
        style = "-" if "overall" in key else "--"
        lw = 2.5 if "overall" in key else 1.8
        ax_acc.plot(rounds, vals, color=_PALETTE[i % len(_PALETTE)],
                    linewidth=lw, marker="o", markersize=3, label=label,
                    linestyle=style)

    ax_acc.set_xlabel("Communication Round")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Test Accuracy by Bucket")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(loc="lower right")

    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 2. Per-class accuracy heatmap ---------------------------------------

def plot_per_class_accuracy_heatmap(per_class_history: List[Dict[int, float]],
                                    rounds: List[int], num_classes: int,
                                    out_path: str,
                                    title: str = "Per-Class Accuracy Over Rounds") -> str:
    """Heatmap: x = class, y = round, color = accuracy."""
    _apply_style()
    matrix = np.full((len(rounds), num_classes), np.nan)
    for i, pca in enumerate(per_class_history):
        for c, acc in pca.items():
            matrix[i, int(c)] = acc

    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.5), max(4, len(rounds) * 0.3)))

    if _HAS_SEABORN:
        sns.heatmap(matrix, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=(num_classes <= 20 and len(rounds) <= 30),
                    fmt=".2f", linewidths=0.5, linecolor="white",
                    xticklabels=range(num_classes),
                    yticklabels=rounds,
                    cbar_kws={"label": "Accuracy", "shrink": 0.8})
    else:
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(len(rounds)))
        ax.set_yticklabels(rounds)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
        if num_classes <= 20 and len(rounds) <= 30:
            for i in range(len(rounds)):
                for j in range(num_classes):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if val < 0.4 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color=color)

    ax.set_xlabel("Class")
    ax.set_ylabel("Round")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 3. Confusion matrix heatmap -----------------------------------------

def plot_confusion_matrix(cm: np.ndarray, out_path: str,
                          class_names: Optional[List[str]] = None,
                          title: str = "Confusion Matrix",
                          normalize: bool = True) -> str:
    """Standard confusion matrix heatmap with seaborn styling."""
    _apply_style()
    num_classes = cm.shape[0]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm.astype(float), row_sums,
                            out=np.zeros_like(cm, dtype=float),
                            where=row_sums != 0)
    else:
        cm_plot = cm.astype(float)

    labels = class_names if class_names else [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.5),
                                    max(5, num_classes * 0.45)))

    if _HAS_SEABORN:
        fmt_str = ".2f" if normalize else "d"
        annot_data = cm_plot if normalize else cm
        sns.heatmap(cm_plot, ax=ax, cmap="Blues", vmin=0,
                    vmax=1.0 if normalize else cm_plot.max(),
                    annot=annot_data if num_classes <= 20 else False,
                    fmt=fmt_str, linewidths=0.5, linecolor="white",
                    xticklabels=labels, yticklabels=labels, square=True,
                    cbar_kws={"label": "Recall" if normalize else "Count",
                              "shrink": 0.8})
    else:
        im = ax.imshow(cm_plot, cmap="Blues", vmin=0,
                       vmax=1.0 if normalize else cm_plot.max())
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, shrink=0.8,
                     label="Recall" if normalize else "Count")
        if num_classes <= 20:
            for i in range(num_classes):
                for j in range(num_classes):
                    val = cm_plot[i, j]
                    text = f"{val:.2f}" if normalize else f"{int(cm[i, j])}"
                    color = "white" if val > 0.5 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=7, color=color)

    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("True", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 4. Class distribution -----------------------------------------------

def plot_class_distribution(class_counts: Dict[int, int],
                            client_class_counts: Dict[int, Dict[int, int]],
                            num_classes: int, out_path: str,
                            title: str = "Data Distribution") -> str:
    """Global long-tail bar chart + per-client stacked bars."""
    _apply_style()
    fig, (ax_global, ax_clients) = plt.subplots(1, 2, figsize=(14, 5))

    # Global distribution
    classes = list(range(num_classes))
    counts = [class_counts.get(c, 0) for c in classes]

    if _HAS_SEABORN:
        palette = sns.color_palette("coolwarm_r", n_colors=num_classes)
        sns.barplot(x=classes, y=counts, ax=ax_global, hue=classes,
                    palette=palette, legend=False, edgecolor="white",
                    linewidth=0.5)
    else:
        colors_global = [_PALETTE[0] if counts[c] > np.median(counts) else
                         _PALETTE[1] if counts[c] > np.percentile(counts, 33) else
                         _PALETTE[2] for c in classes]
        ax_global.bar(classes, counts, color=colors_global, edgecolor="white",
                      linewidth=0.5)

    ax_global.set_xlabel("Class")
    ax_global.set_ylabel("Sample Count")
    ax_global.set_title("Global Class Distribution (Long-Tail)", fontweight="bold")

    # Per-client stacked
    client_ids = sorted(client_class_counts.keys())
    num_clients = len(client_ids)
    bottoms = np.zeros(num_clients)
    palette = sns.color_palette("husl", num_classes) if _HAS_SEABORN else _PALETTE

    for c in classes:
        heights = [client_class_counts.get(cid, {}).get(c, 0) for cid in client_ids]
        ax_clients.bar(range(num_clients), heights, bottom=bottoms,
                       color=palette[c % len(palette)], edgecolor="white",
                       linewidth=0.3, label=f"Class {c}" if c < 10 else None)
        bottoms += heights

    ax_clients.set_xlabel("Client ID")
    ax_clients.set_ylabel("Sample Count")
    ax_clients.set_title("Per-Client Class Distribution", fontweight="bold")
    ax_clients.set_xticks(range(num_clients))
    ax_clients.set_xticklabels([str(cid) for cid in client_ids])
    if num_classes <= 10:
        ax_clients.legend(loc="upper right", ncol=2, fontsize=7)

    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 5. Fidelity evolution (Mechanism B) ----------------------------------

def plot_fidelity_evolution(fidelity_history: List[Dict[str, object]],
                            rounds: List[int], num_classes: int,
                            out_path: str,
                            title: str = "Mechanism B: Fidelity Evolution") -> str:
    """Per-class fidelity score over rounds."""
    _apply_style()
    fig, (ax_mean, ax_per_class) = plt.subplots(1, 2, figsize=(14, 5))

    # Mean fidelity
    mean_fids = [h.get("mean_fidelity", float("nan")) for h in fidelity_history]
    ax_mean.plot(rounds, mean_fids, color=_PALETTE[0], linewidth=2.5,
                 marker="o", markersize=5)
    ax_mean.fill_between(rounds, 0, mean_fids, alpha=0.15, color=_PALETTE[0])
    ax_mean.set_xlabel("Communication Round")
    ax_mean.set_ylabel("Mean Fidelity")
    ax_mean.set_title("Mean Generator Fidelity", fontweight="bold")
    ax_mean.set_ylim(-0.05, 1.05)

    # Per-class fidelity heatmap
    matrix = np.full((len(rounds), num_classes), np.nan)
    for i, h in enumerate(fidelity_history):
        per_class = h.get("fidelity_per_class", [])
        for c, val in enumerate(per_class):
            if c < num_classes:
                matrix[i, c] = val

    if _HAS_SEABORN:
        sns.heatmap(matrix, ax=ax_per_class, cmap="YlOrRd", vmin=0, vmax=1,
                    xticklabels=range(num_classes), yticklabels=rounds,
                    linewidths=0.3, linecolor="white",
                    cbar_kws={"label": "Fidelity", "shrink": 0.8})
    else:
        im = ax_per_class.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0,
                                 vmax=1, interpolation="nearest")
        ax_per_class.set_xticks(range(num_classes))
        ax_per_class.set_yticks(range(len(rounds)))
        ax_per_class.set_yticklabels(rounds)
        fig.colorbar(im, ax=ax_per_class, shrink=0.8, label="Fidelity")

    ax_per_class.set_xlabel("Class")
    ax_per_class.set_ylabel("Round")
    ax_per_class.set_title("Per-Class Fidelity", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 6. Bucket accuracy comparison (multi-method) ------------------------

def plot_bucket_comparison(method_scores: Dict[str, Dict[str, float]],
                           out_path: str,
                           title: str = "Accuracy Comparison by Bucket") -> str:
    """Grouped bar chart comparing head/medium/tail/overall across methods.

    Args:
        method_scores: {"baseline": {"head": 0.8, "medium": 0.6, "tail": 0.3, "overall": 0.55}, ...}
    """
    _apply_style()
    methods = list(method_scores.keys())
    buckets = ["head", "medium", "tail", "overall"]
    buckets = [b for b in buckets if any(b in method_scores[m] for m in methods)]

    x = np.arange(len(buckets))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        vals = [method_scores[method].get(b, 0) for b in buckets]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=method,
                      color=_PALETTE[i % len(_PALETTE)], edgecolor="white",
                      linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    fontweight="bold")

    ax.set_xlabel("Bucket")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([b.title() for b in buckets])
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 7. Convergence plot -------------------------------------------------

def plot_convergence(convergence_history: List[dict], rounds: List[int],
                     out_path: str,
                     title: str = "Convergence Analysis") -> str:
    """Accuracy delta per round with plateau/divergence markers."""
    _apply_style()
    fig, (ax_delta, ax_best) = plt.subplots(1, 2, figsize=(14, 5))

    deltas = [c.get("acc_delta", 0) for c in convergence_history]
    best_accs = [c.get("best_acc", 0) for c in convergence_history]
    plateaus = [i for i, c in enumerate(convergence_history) if c.get("is_plateau")]
    diverging = [i for i, c in enumerate(convergence_history) if c.get("is_diverging")]

    # Delta plot
    colors_delta = [_PALETTE[4] if d >= 0 else _PALETTE[2] for d in deltas]
    ax_delta.bar(rounds, deltas, color=colors_delta, edgecolor="white", linewidth=0.5)
    ax_delta.axhline(y=0, color="#333333", linewidth=0.8)
    ax_delta.set_xlabel("Communication Round")
    ax_delta.set_ylabel("Accuracy Δ")
    ax_delta.set_title("Per-Round Accuracy Change", fontweight="bold")

    if plateaus:
        plateau_rounds = [rounds[i] for i in plateaus if i < len(rounds)]
        plateau_deltas = [deltas[i] for i in plateaus if i < len(rounds)]
        ax_delta.scatter(plateau_rounds, plateau_deltas, color=_PALETTE[1],
                         marker="^", s=80, zorder=5, label="Plateau detected")
    if diverging:
        div_rounds = [rounds[i] for i in diverging if i < len(rounds)]
        div_deltas = [deltas[i] for i in diverging if i < len(rounds)]
        ax_delta.scatter(div_rounds, div_deltas, color=_PALETTE[2],
                         marker="X", s=80, zorder=5, label="Divergence detected")
    if plateaus or diverging:
        ax_delta.legend()

    # Best accuracy tracking
    ax_best.plot(rounds, best_accs, color=_PALETTE[0], linewidth=2,
                 marker="o", markersize=4, label="Best Accuracy So Far")
    ax_best.fill_between(rounds, 0, best_accs, alpha=0.1, color=_PALETTE[0])
    ax_best.set_xlabel("Communication Round")
    ax_best.set_ylabel("Accuracy")
    ax_best.set_title("Best Accuracy Tracking", fontweight="bold")
    ax_best.set_ylim(-0.05, 1.05)
    ax_best.legend()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 8. Synthetic sample grid --------------------------------------------

def plot_synthetic_samples(images: np.ndarray, labels: np.ndarray,
                           out_path: str, num_classes: int,
                           samples_per_class: int = 5,
                           title: str = "Generated Samples") -> str:
    """Grid of generated images grouped by class.

    Args:
        images: (N, C, H, W) numpy array in [-1, 1] or [0, 1] range.
        labels: (N,) integer labels.
    """
    _apply_style()
    n_rows = min(num_classes, 10)
    fig, axes = plt.subplots(n_rows, samples_per_class,
                             figsize=(samples_per_class * 1.5, n_rows * 1.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for c in range(n_rows):
        mask = labels == c
        class_imgs = images[mask][:samples_per_class]
        for j in range(samples_per_class):
            ax = axes[c, j]
            ax.axis("off")
            if j < len(class_imgs):
                img = class_imgs[j]
                if img.shape[0] == 1:
                    ax.imshow(img[0], cmap="gray", vmin=-1, vmax=1)
                else:
                    # CHW -> HWC, rescale from [-1,1] to [0,1]
                    img_display = np.clip((img.transpose(1, 2, 0) + 1) / 2, 0, 1)
                    ax.imshow(img_display)
            if j == 0:
                ax.set_ylabel(f"Class {c}", fontsize=8, rotation=0,
                              labelpad=30, va="center")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 9. Per-client accuracy bar / violin ----------------------------------

def plot_per_client_accuracy(client_accuracies: Dict[int, float],
                             out_path: str,
                             title: str = "Per-Client Test Accuracy") -> str:
    """Bar chart showing each client's individual accuracy, with a
    seaborn-enhanced violin/strip overlay when available."""
    _apply_style()
    client_ids = sorted(client_accuracies.keys())
    accs = [client_accuracies[cid] for cid in client_ids]
    mean_acc = np.mean(accs)

    fig, ax = plt.subplots(figsize=(max(6, len(client_ids) * 0.6), 5))

    if _HAS_SEABORN and len(client_ids) > 3:
        # Seaborn strip + box for richer view
        sns.barplot(x=list(range(len(client_ids))), y=accs, ax=ax,
                    hue=[cid for cid in client_ids],
                    palette="viridis", edgecolor="white", linewidth=0.5,
                    legend=False)
        sns.stripplot(x=list(range(len(client_ids))), y=accs, ax=ax,
                      color="black", size=6, alpha=0.7, jitter=False)
    else:
        colors = [_PALETTE[4] if a >= mean_acc else _PALETTE[2] for a in accs]
        bars = ax.bar(range(len(client_ids)), accs, color=colors,
                      edgecolor="white", linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(y=mean_acc, color=_PALETTE[2], linewidth=2, linestyle="--",
               label=f"Mean: {mean_acc:.3f}")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(range(len(client_ids)))
    ax.set_xticklabels([str(cid) for cid in client_ids])
    ax.set_ylim(0, 1.15)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 10. Sweep heatmap ---------------------------------------------------

def plot_sweep_heatmap(sweep_rows: List[dict], metric_key: str,
                       out_path: str,
                       title: Optional[str] = None) -> str:
    """Heatmap of a sweep metric across imbalance_factor × dir_param.

    Args:
        sweep_rows: list of dicts from sweep CSV, each with 'imbalance_factor',
                    'dir_param', 'mechanism', and the metric column.
        metric_key: which column to plot (e.g. 'acc_overall', 'acc_tail').
    """
    _apply_style()
    mechanisms = sorted({r["mechanism"] for r in sweep_rows})
    n_mechs = len(mechanisms)

    fig, axes = plt.subplots(1, n_mechs, figsize=(6 * n_mechs, 5),
                             squeeze=False)

    for idx, mech in enumerate(mechanisms):
        ax = axes[0, idx]
        mech_rows = [r for r in sweep_rows if r["mechanism"] == mech]

        imbs = sorted({r["imbalance_factor"] for r in mech_rows})
        dirs = sorted({r["dir_param"] for r in mech_rows})

        matrix = np.full((len(imbs), len(dirs)), np.nan)
        for r in mech_rows:
            i = imbs.index(r["imbalance_factor"])
            j = dirs.index(r["dir_param"])
            val = r.get(metric_key, float("nan"))
            if not np.isnan(matrix[i, j]):
                matrix[i, j] = (matrix[i, j] + val) / 2  # average over seeds
            else:
                matrix[i, j] = val

        if _HAS_SEABORN:
            sns.heatmap(matrix, ax=ax, cmap="viridis",
                        annot=(len(imbs) <= 10 and len(dirs) <= 10),
                        fmt=".3f", linewidths=0.5, linecolor="white",
                        xticklabels=[f"{d:.2f}" for d in dirs],
                        yticklabels=[f"{i:.3f}" for i in imbs],
                        cbar_kws={"shrink": 0.8})
        else:
            im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                           interpolation="nearest")
            ax.set_xticks(range(len(dirs)))
            ax.set_yticks(range(len(imbs)))
            ax.set_xticklabels([f"{d:.2f}" for d in dirs])
            ax.set_yticklabels([f"{i:.3f}" for i in imbs])
            fig.colorbar(im, ax=ax, shrink=0.8)
            if len(imbs) <= 10 and len(dirs) <= 10:
                for i in range(len(imbs)):
                    for j in range(len(dirs)):
                        val = matrix[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.3f}", ha="center",
                                    va="center", fontsize=8,
                                    color="white" if val < np.nanmedian(matrix) else "black")

        ax.set_xlabel("Dir. α")
        ax.set_ylabel("Imbalance Factor")
        ax.set_title(f"{mech}", fontweight="bold")

    fig.suptitle(title or f"Sweep: {metric_key}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 11. F1 / Precision / Recall bar chart --------------------------------

def plot_classification_report(report: Dict, out_path: str,
                               title: str = "Classification Report") -> str:
    """Grouped bar chart of precision, recall, F1 per class."""
    _apply_style()
    class_keys = sorted(k for k in report if isinstance(k, int))

    if not class_keys:
        return out_path

    precisions = [report[c]["precision"] for c in class_keys]
    recalls = [report[c]["recall"] for c in class_keys]
    f1s = [report[c]["f1"] for c in class_keys]

    x = np.arange(len(class_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(class_keys) * 0.8), 5))
    ax.bar(x - width, precisions, width, label="Precision", color=_PALETTE[0],
           edgecolor="white", linewidth=0.5)
    ax.bar(x, recalls, width, label="Recall", color=_PALETTE[1],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + width, f1s, width, label="F1", color=_PALETTE[4],
           edgecolor="white", linewidth=0.5)

    # Add macro avg line
    macro = report.get("macro_avg", {})
    if macro:
        ax.axhline(y=macro.get("f1", 0), color=_PALETTE[2], linewidth=1.5,
                   linestyle="--", label=f"Macro F1: {macro.get('f1', 0):.3f}")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in class_keys])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- 12. Round timing breakdown ------------------------------------------

def plot_timing_breakdown(timing_history: List[dict], rounds: List[int],
                          out_path: str,
                          title: str = "Per-Round Timing Breakdown") -> str:
    """Stacked bar chart of gen/target/eval time per round."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, len(rounds) * 0.4), 5))

    gen_times = [t.get("gen_time", 0) for t in timing_history]
    target_times = [t.get("target_time", 0) for t in timing_history]
    eval_times = [t.get("eval_time", 0) for t in timing_history]

    ax.bar(rounds, gen_times, label="Generator", color=_PALETTE[0],
           edgecolor="white", linewidth=0.3)
    ax.bar(rounds, target_times, bottom=gen_times, label="Target Net",
           color=_PALETTE[1], edgecolor="white", linewidth=0.3)
    bottoms_eval = [g + t for g, t in zip(gen_times, target_times)]
    ax.bar(rounds, eval_times, bottom=bottoms_eval, label="Evaluation",
           color=_PALETTE[4], edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
    return out_path


# ---- Auto-display utility -------------------------------------------------

def show_all_plots(plots_dir: str):
    """Re-open and display all saved PNG plots. Call at end of training
    when --show_plots 1 is set. Switches to an interactive backend."""
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            return  # no interactive backend available

    import glob
    pngs = sorted(glob.glob(os.path.join(plots_dir, "*.png")))
    if not pngs:
        return

    for png_path in pngs:
        img = plt.imread(png_path)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(os.path.basename(png_path).replace(".png", "").replace("_", " ").title(),
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

    plt.show()
