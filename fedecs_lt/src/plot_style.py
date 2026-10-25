"""IEEE figure rules (action plan section 21), applied to every plot; nothing else sets fonts or sizes.

Sizes are the printed sizes: one column 3.5 in (88.9 mm), two columns 7.16 in (181.9 mm). Text is Times (or a
Times-compatible font: Liberation Serif, Nimbus Roman, TeX Gyre Termes, or matplotlib's own STIXGeneral) at 8 pt; lines at
least 0.5 pt; fonts embedded as TrueType (no Type 3). Every method has one fixed colour, marker and line style, chosen
to be distinguishable by colour-blind readers and in greyscale print. Figures are saved as vector PDF (dense bitmaps
at 600 dpi) together with a PNG preview and a CSV of the plotted data.
"""
import csv
import os
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

COLUMN_IN, PAGE_IN = 3.5, 7.16
FONT_PT = 8
MIN_LINE_PT = 0.5

# Times, or a metric-compatible clone; STIXGeneral (a Times design shipped with matplotlib) is always available.
_SERIFS = ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "Nimbus Roman No9 L", "TeX Gyre Termes",
           "STIXGeneral"]


def _serif():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in _SERIFS:
        if name in have:
            return name
    return "STIXGeneral"


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": [_serif()] + _SERIFS, "mathtext.fontset": "stix",
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "xtick.labelsize": FONT_PT,
        "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT, "figure.titlesize": FONT_PT,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5, "lines.linewidth": 1.0, "lines.markersize": 3.5, "patch.linewidth": 0.5,
        "grid.linewidth": 0.5, "grid.alpha": 0.35, "axes.grid": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "image.cmap": "viridis",
    })


# Colour-blind-safe palette (Okabe and Ito, 2008) plus greys; markers and dashes make lines distinct in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#882255", "#000000", "#F0E442", "#7F7F7F",
           "#56B4E9", "#44AA99", "#117733", "#332288", "#AA4499", "#999933", "#DDCC77", "#661100", "#6699CC"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p", "d", "8", "H", "1", "2", "3"]
DASHES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]
OURS = "FedECS-LT"
# Order of the methods in tables and legends: the main comparison, the ablations, then the one-shared-model baselines
ORDER = ["FedECS-LT", "FedECS-LT (200 extractor rounds)", "GeFL-F", "Balanced Softmax", "FedLC", "FedRS", "FedDF", "Local",
         "FedProto", "FedTGP", "FedGen", "FedGH", "LG-FedAvg", "FedMRL",
         "GeFL-F w/o generator", "GeFL-F + LA", "GeFL-F w/ Gaussian generator + LA", "FedDF + LA",
         "FedAvg", "FedProx", "MOON", "CCVR", "CReFF", "FedETF", "Fed-GraB", "FedLoGe", "FedYoYo", "RUCR", "CLIP2FL",
         "FedNH", "FedVLS"]
# Colour and marker index of every method (18 of each): unique among the methods one report can show together
# (main comparison 0-12, ablations 13-16 with 0-1, the variant 17; the one-shared-model baselines avoid 0, 1, 3, 8
# and 17, the indices of FedECS-LT, GeFL-F, FedLC, FedTGP and the variant, which share reports with them)
STYLE_INDEX = {m: i for i, m in enumerate(["FedECS-LT", "GeFL-F", "Balanced Softmax", "FedLC", "FedRS", "FedDF",
                                           "Local", "FedProto", "FedTGP", "FedGen", "FedGH", "LG-FedAvg", "FedMRL",
                                           "GeFL-F w/o generator", "GeFL-F + LA",
                                           "GeFL-F w/ Gaussian generator + LA", "FedDF + LA",
                                           "FedECS-LT (200 extractor rounds)"])}
STYLE_INDEX.update({"FedAvg": 2, "FedProx": 4, "MOON": 5, "CCVR": 6, "CReFF": 7, "FedETF": 9, "Fed-GraB": 10,
                    "FedLoGe": 11, "FedYoYo": 12, "RUCR": 13, "CLIP2FL": 14, "FedNH": 15, "FedVLS": 16})


def style(method):
    """Fixed (colour, marker, line style) for a method; ours is always the first colour, solid, circle."""
    variant = 0
    for v, suffix in enumerate((" (LongTailLib)",), 1):   # same colour as the method, other dashes
        if method.endswith(suffix) and method[:-len(suffix)] in STYLE_INDEX:
            method, variant = method[:-len(suffix)], v
    # unknown names: a stable index (str hash() changes between processes)
    i = STYLE_INDEX[method] if method in STYLE_INDEX else zlib.crc32(method.encode()) % len(PALETTE)
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)],
                linestyle=DASHES[(i + 3 * variant) % len(DASHES)] if method != OURS or variant else "-",
                linewidth=1.4 if method == OURS else 1.0, zorder=5 if method == OURS else 2)


def figure(width="column", height=2.4, ncols=1, nrows=1, **kw):
    apply()
    w = COLUMN_IN if width == "column" else PAGE_IN if width == "page" else float(width) / 25.4
    return plt.subplots(nrows, ncols, figsize=(w, height), **kw)


def panel_labels(axes, y=-0.32):
    """(a), (b), ... below each panel (IEEE subfigure convention)."""
    for i, ax in enumerate(list(axes)):
        ax.text(0.5, y, f"({chr(97 + i)})", transform=ax.transAxes, ha="center", va="top", fontsize=FONT_PT)


def enforce(fig):
    """IEEE minimums on every figure, whatever the plotting code set: text at least FONT_PT (8 pt), drawn lines at
    least MIN_LINE_PT (0.5 pt). Returns the number of artists changed."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text
    n = 0
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip() and t.get_fontsize() < FONT_PT - 1e-6:
            t.set_fontsize(FONT_PT)
            n += 1
    for ln in fig.findobj(Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "none", "") and 0 < ln.get_linewidth() < MIN_LINE_PT:
            ln.set_linewidth(MIN_LINE_PT)
            n += 1
    from matplotlib.ticker import AutoLocator, MaxNLocator
    for ax in fig.axes:   # rounds are whole numbers
        if ax.get_xlabel() == "Round" and isinstance(ax.xaxis.get_major_locator(), AutoLocator):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))   # also with one round
    place_legends(fig)
    fit_width(fig)
    return n


def fit_width(fig, tol=0.01):
    """Keep the saved figure at its nominal width (3.5 or 7.16 in): when tick labels, legends or colour bars reach
    past the figure's edges, the axes are narrowed (text keeps its 8 pt size) until everything fits."""
    W = fig.get_figwidth()
    r = fig.canvas.get_renderer()
    for _ in range(8):
        tb = fig.get_tightbbox(r)
        if tb.width <= W + tol:
            return
        a, b = -tb.x0 / W + tol / W, 1 - (tb.x1 - W) / W - tol / W   # where [0, 1] of the axes' span must map to
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position([a + pos.x0 * (b - a), pos.y0, pos.width * (b - a), pos.height])


WIDTH_PROBLEMS = []   # (figure name, saved width in inches, nominal width) of every figure wider than it should be


def points_under(ax, leg, r):
    """Plotted points (line vertices, scatter points) of an axes that its legend covers."""
    from matplotlib.collections import PathCollection
    lb = leg.get_window_extent(r)
    n = 0
    xys = [ln.get_transform().transform(ln.get_xydata()) for ln in ax.get_lines()
           if ln.get_visible() and len(ln.get_xydata())]
    xys += [c.get_offset_transform().transform(c.get_offsets()) for c in ax.collections
            if isinstance(c, PathCollection) and c.get_visible() and len(c.get_offsets())]
    for xy in xys:
        n += int(((xy[:, 0] > lb.x0) & (xy[:, 0] < lb.x1) & (xy[:, 1] > lb.y0) & (xy[:, 1] < lb.y1)).sum())
    return n


def place_legends(fig):
    """Legends never cover data: one with more than 6 entries goes below the axes (under the tick and axis labels)
    with as many columns as fit the figure's width; a bar chart's short legend goes in one row above the axes."""
    from matplotlib.container import BarContainer
    r = fig.canvas.get_renderer()
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None or leg.get_gid() == "keep":   # placed on purpose by the figure (e.g. inside a scatter plot)
            continue
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.texts]
        bars = any(isinstance(c, BarContainer) for c in ax.containers)
        too_wide = leg.get_window_extent(r).width > 0.98 * fig.get_figwidth() * fig.dpi
        if len(labels) <= 6 and not bars and not too_wide and not points_under(ax, leg, r):
            continue
        kw = dict(frameon=False, borderaxespad=0.0, handlelength=2.2, columnspacing=1.2,
                  markerscale=leg.markerscale)
        leg.remove()
        box = ax.get_window_extent(r)
        if len(labels) <= 6 and bars and not too_wide:
            ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(labels), **kw)
            continue
        tight = ax.get_tightbbox(r)
        fig_w, fig_h = fig.get_figwidth() * fig.dpi, fig.get_figheight() * fig.dpi
        y = tight.y0 / fig_h - 0.02            # just under the axes' tick and axis labels, in figure coordinates
        x = 0.5 if len(fig.axes) == 1 else (box.x0 + box.x1) / 2 / fig_w
        # widest column count whose legend still fits inside the figure's width (centred on the figure)
        for ncol in range(min(6, len(labels)), 0, -1):
            new = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(x, y), bbox_transform=fig.transFigure,
                            ncol=ncol, **kw)
            if new.get_window_extent(r).width <= 0.98 * fig_w or ncol == 1:
                break
            new.remove()


LAYOUT_PROBLEMS = []   # (figure name, problem) found by layout_problems on every saved figure


def _inter(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if w > 0 and h > 0 else 0.0


def layout_problems(fig):
    """What a reader would see as broken, measured on the drawn figure: (1) two pieces of text that overlap (more than
    10% of the smaller one's box); (2) text that runs into another panel's plotting area or a colour bar; (3) a legend
    placed inside the axes that covers plotted points. Returns a list of problem strings (empty = clean)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    # tick labels exist for ticks outside the view limits too; only the drawn ones count
    all_ticklabels, drawn = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                all_ticklabels.update((id(tk.label1), id(tk.label2)))
            for tk in axis._update_ticks():
                drawn.update(id(lb) for lb in (tk.label1, tk.label2) if lb.get_visible())
    texts = []
    for t in fig.findobj(Text):
        if id(t) in all_ticklabels and id(t) not in drawn:
            continue
        if t.get_visible() and t.get_text().strip() and t.get_alpha() != 0:
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if bb.width > 0 and bb.height > 0:
                texts.append((t, bb))
    out = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            (ta, a), (tb, b) = texts[i], texts[j]
            if _inter(a, b) > 0.10 * min(a.width * a.height, b.width * b.height):
                out.append(f"text '{ta.get_text()[:30]}' overlaps '{tb.get_text()[:30]}'")
    axes_boxes = [(ax, ax.patch.get_window_extent(r)) for ax in fig.axes if ax.get_visible()]
    for t, bb in texts:        # text of one panel inside another panel's plotting area (or a colour bar)
        own = t.axes
        if own is None:
            continue
        if any(ax is not own and _inter(bb, box) > 0.10 * bb.width * bb.height for ax, box in axes_boxes):
            out.append(f"text '{t.get_text()[:30]}' runs into another panel")
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(r)
        if _inter(lb, ax.patch.get_window_extent(r)) < 0.5 * lb.width * lb.height:
            continue          # legend outside the axes
        inside = points_under(ax, leg, r)
        if inside:
            out.append(f"legend covers {inside} plotted points")
    return sorted(set(out))


def save(fig, out_dir, name, data_rows=None, header=None, raster=False):
    """Save PDF (vector; 600 dpi when raster), PNG preview and the plotted data as CSV. Returns the PDF path."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name)
    for ax in fig.axes:
        ax.set_title("")  # no titles inside images: the caption says what the figure shows
    enforce(fig)
    nominal = fig.get_figwidth()
    width = fig.get_tightbbox(fig.canvas.get_renderer()).width   # inches, as saved with bbox_inches="tight"
    if width > nominal + 0.02:
        WIDTH_PROBLEMS.append((name, round(width, 2), nominal))
    LAYOUT_PROBLEMS.extend((name, p) for p in layout_problems(fig))
    fig.savefig(base + ".pdf", dpi=600 if raster else None, metadata={"Creator": None, "Producer": None,
                                                                      "CreationDate": None})
    fig.savefig(base + ".png", dpi=200)
    plt.close(fig)
    if data_rows is not None:
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(data_rows)
    return base + ".pdf"


def check_pdf(path):
    """IEEE check of a saved figure: width within the column / page width, embedded TrueType fonts only (no Type 3).
    Returns a list of problems (empty = passes)."""
    problems = []
    raw = open(path, "rb").read()
    if b"/Type3" in raw:
        problems.append("Type 3 font")
    import re
    m = re.search(rb"/MediaBox\s*\[\s*([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s+([\d.\-]+)\s*\]", raw)
    if m:
        w_in = (float(m.group(3)) - float(m.group(1))) / 72.0
        if w_in > PAGE_IN + 0.05:
            problems.append(f"width {w_in:.2f} in exceeds the page width")
    return problems
