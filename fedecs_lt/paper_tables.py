"""The paper's numbers, written from a results folder as LaTeX rows and CSV, so that nothing is typed by hand
(repository tool; no GPU needed). The values are the report's (same functions: plots.metric, report.fmt and the
paired test of report.paired_table), one block per setting.

    python fedecs_lt/paper_tables.py RESULTS_DIR OUT_DIR

RESULTS_DIR is a `results/` folder: from a notebook's `NAME__results.zip` or `NAME__summary.zip` (unpacked), or
the one `merge_results.py` writes. OUT_DIR receives, per setting (e.g. `cifar10_IF100_a0.5_K10`):
  <setting>_main.tex / .csv      every method x the main metrics (mean ± SD over seeds, %; best in bold)
  <setting>_readout.tex / .csv   the same with the exact readout (methods that have one)
  <setting>_more.tex / .csv      accuracy, kappa, top-5, AUC-ROC, head / medium recall, recall SD, calibration, cost
  <setting>_paired.tex / .csv    FedECS-LT minus each method, balanced accuracy, paired over seeds: mean, 95% CI,
                                 p and Holm-adjusted p (bold where Holm p < 0.05)
and `ALL.md`, every table in one readable file.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import captions as cap  # noqa: E402
import plots  # noqa: E402
import report  # noqa: E402
import representation as rp  # noqa: E402

LOWER = {"final_ece", "final_cw_ece", "final_nll", "final_brier", "final_rec_std", "comm_mb_per_client", "time_s",
         "fe_time_s", "gen_time_s", "gpu_peak_mb", "fwd_mflops", "train_gflops_round", "final_own_mix_sd"}
MORE = report.SECOND + report.CALIB + [k for k in report.SYSTEM if k[0] != "gpu_peak_mb"]


def tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("±", r"$\pm$")


def write(out, stem, header, rows, bold=()):
    """rows: list of lists of strings; bold: set of (row, col) cells to set in bold."""
    with open(os.path.join(out, stem + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    lines = [r"\begin{tabular}{l" + "c" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r"\\", r"\midrule"]
    for i, r in enumerate(rows):
        cells = [(r"\textbf{%s}" % tex_escape(c)) if (i, j) in bold else tex_escape(c) for j, c in enumerate(r)]
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(out, stem + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(("**%s**" % c) if (i, j) in bold else c for j, c in enumerate(r)) + " |"
           for i, r in enumerate(rows)]
    return md


def metric_rows(E, S, keys):
    rows, vals = [], []
    for m in E.methods:
        cells = [plots.metric(E, m, k, S) for k, _ in keys]
        if all(np.isnan(mu) for mu, _, _ in cells):
            continue
        rows.append([m] + [report.fmt(mu, sd, k, n) for (mu, sd, n), (k, _) in zip(cells, keys)])
        vals.append([mu for mu, _, _ in cells])
    bold = set()
    if vals:
        A = np.array(vals, dtype=float)
        for j, (k, _) in enumerate(keys):
            col = A[:, j]
            if np.all(np.isnan(col)):
                continue
            b = np.nanmin(col) if k in LOWER else np.nanmax(col)
            bold |= {(int(i), j + 1) for i in np.where(np.isclose(col, b, rtol=0, atol=1e-9))[0]}
    return rows, bold


def paired_rows(E, S):
    from scipy import stats
    ours = {r["seed"]: r for r in E.of(plots.OURS, S)}
    rows = []
    for m in E.methods:
        if m == plots.OURS:
            continue
        other = {r["seed"]: r for r in E.of(m, S)}
        d = [100 * (ours[x]["final_bal"] - other[x]["final_bal"]) for x in sorted(set(ours) & set(other))
             if ours[x].get("final_bal") is not None and other[x].get("final_bal") is not None]
        if len(d) >= 2:
            sd = float(np.std(d, ddof=1))
            half = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
            pv = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (0.0 if np.mean(d) != 0 else 1.0)
            rows.append([m, len(d), float(np.mean(d)), half, pv])
    if not rows:
        return [], set()
    adj = rp.holm([r[4] for r in rows])
    body = [[r[0], str(r[1]), f"{r[2]:+.1f}", f"[{r[2] - r[3]:+.1f}, {r[2] + r[3]:+.1f}]", f"{r[4]:.3g}", f"{a:.3g}"]
            for r, a in zip(rows, adj)]
    bold = {(i, 2) for i, a in enumerate(adj) if a < 0.05}
    return body, bold


def column_name(S, settings):
    """Short column header for a setting: the dataset's name, with the imbalance ratio when one dataset appears
    with several ratios and the number of clients when it appears with several client counts."""
    spec = cap.DATASETS.get(S[0], dict(name=S[0]))
    same = [T for T in settings if T[0] == S[0]]
    name = spec["name"]
    if len({T[1] for T in same}) > 1:
        name += f" (ratio {int(round(1 / S[1]))})" if S[1] else " (natural)"
    if len({T[3] for T in same}) > 1:
        name += f", K = {S[3]}"
    return name


def across(E, out):
    """Methods x settings, one table per quantity (the layout of the paper's main tables): mean ± SD of balanced
    accuracy as trained and with the exact readout, macro-F1, tail and worst-class recall, and the paired
    difference of FedECS-LT to each method (bold: Holm p < 0.05)."""
    if len(E.settings) < 2:
        return []
    md = ["## Across settings (one column per setting)", ""]
    order = list(cap.DATASETS)          # columns in the order of the datasets table (MNIST, Fashion-MNIST, ...)
    settings = sorted(E.settings, key=lambda S: (order.index(S[0]) if S[0] in order else len(order), S[0],
                                                 S[1] or 1.0, S[3]))   # ratio 100, 50, 10
    groups = [("track1", [S for S in settings if cap.DATASETS.get(S[0], {}).get("track") != 2]),
              ("track2", [S for S in settings if cap.DATASETS.get(S[0], {}).get("track") == 2])]
    for tag, sets in groups:
        if not sets:
            continue
        header = ["Method"] + [column_name(S, sets) for S in sets]
        for key, title in (("final_bal", "Balanced accuracy (%), heads as trained"),
                           ("er_bal", "Balanced accuracy (%), exact readout"),
                           ("final_f1", "Macro-F1 (%)"), ("final_tail", "Tail-class recall (%)"),
                           ("final_worst", "Worst-class recall (%)")):
            rows, vals = [], []
            for m in E.methods:
                cells = [plots.metric(E, m, key, S) for S in sets]
                if all(np.isnan(mu) for mu, _, _ in cells):
                    continue
                rows.append([m] + [report.fmt(mu, sd, key, n) for mu, sd, n in cells])
                vals.append([mu for mu, _, _ in cells])
            if not rows:
                continue
            A = np.array(vals, dtype=float)
            bold = set()
            for j in range(A.shape[1]):
                if not np.all(np.isnan(A[:, j])):
                    bold |= {(int(i), j + 1) for i in np.where(np.isclose(A[:, j], np.nanmax(A[:, j]), rtol=0,
                                                                          atol=1e-9))[0]}
            md += [f"### {title} ({tag})", ""] + write(out, f"across_{tag}_{key}", header, rows, bold) + [""]
        diffs = {S: {r[0]: r for r in paired_rows(E, S)[0]} for S in sets}
        holm = {S: {r[0]: float(r[5]) for r in paired_rows(E, S)[0]} for S in sets}
        rows, bold = [], set()
        for m in E.methods:
            if m == plots.OURS or not any(m in diffs[S] for S in sets):
                continue
            row = [m]
            for j, S in enumerate(sets):
                d = diffs[S].get(m)
                row.append(f"{d[2]} {d[3]}" if d else "-")
                if d and holm[S][m] < 0.05:
                    bold.add((len(rows), j + 1))
            rows.append(row)
        if rows:
            md += [f"### FedECS-LT minus each method, balanced accuracy (points, 95% CI; bold: Holm p < 0.05) "
                   f"({tag})", ""] + write(out, f"across_{tag}_paired", header, rows, bold) + [""]
    return md


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    res, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    E = plots.Exp(res)
    if not E.runs:
        raise SystemExit(f"no runs in {res}")
    md = ["# Tables for the paper", "", f"From `{os.path.abspath(res)}` ({len(E.runs)} runs). Values in % unless "
          "stated; mean ± sample SD over seeds (no SD with one seed); bold: best in the column (paired table: Holm "
          "p < 0.05).", ""]
    for S in E.settings:
        slug = cap.setting_slug(S)
        seeds = sorted({r["seed"] for r in E.runs if plots.setting_key(r) == S})
        md += [f"## {cap.setting_text(S).rstrip('.')}; seeds {', '.join(map(str, seeds))}", "",
               f"Files: `{slug}_*.tex / .csv`", ""]
        for stem, keys, title in (("main", report.MAIN, "Main metrics, heads as trained"),
                                  ("readout", report.READOUT, "With the exact readout"),
                                  ("more", MORE, "Further metrics, calibration (ECE in %, NLL, Brier) and cost")):
            rows, bold = metric_rows(E, S, keys)
            if rows:
                md += [f"### {title}", ""] + write(out, f"{slug}_{stem}", ["Method"] + [n for _, n in keys], rows,
                                                   bold) + [""]
        rows, bold = paired_rows(E, S)
        if rows:
            md += ["### FedECS-LT minus each method (balanced accuracy, points; paired over seeds)", ""]
            md += write(out, f"{slug}_paired", ["Method", "Seeds", "Difference", "95% CI", "p", "p (Holm)"], rows,
                        bold) + [""]
    md += across(E, out)
    with open(os.path.join(out, "ALL.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("wrote", out, "(ALL.md and one .tex / .csv per table)")


if __name__ == "__main__":
    main()
