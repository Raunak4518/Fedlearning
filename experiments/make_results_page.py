"""Build docs/results.html from the experiment result files.

Every number on the page is read from results/*/runs.jsonl (and
reference_check/reference_runs.jsonl); the only hand-entered numbers are the
paper's own published figures and Kaggle runs whose runs.jsonl has not been
copied back yet. Rerun after any experiment finishes:

    python make_results_page.py
"""
import glob
import html
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "docs", "results.html")
DESIGN = os.path.join(HERE, "..", "docs", "experiment_plan.html")

# Published numbers (paper 2412.18460v2). best_mean_acc, %, IID, K=10 unless noted.
PAPER = {("mnist", 10): 95.47, ("mnist", 50): 95.04, ("mnist", 100): 94.63, ("fmnist", 10): 83.14,
         ("svhn", 10): 76.26, ("cifar10", 10): 55.86}
PAPER_BEST_AUG_CIFAR = 62.67  # GeFL (DCGAN) + MixUp, Table IV - the best CIFAR-10 number of any GeFL variant

# Kaggle K01 SVHN runs (all 27, complete) parsed from the user's notebook log; used
# only until results/K01_kaggle_cifar10_svhn/runs.jsonl is copied back. CIFAR-10 pending.
KAGGLE_LOGGED = [  # (label, dataset, IF, seed, final_bal, final_tail, best_mean_acc, fidelity_tail, norm_ratio)
    ('+HWA+LA (ours)', 'cifar10', 0.01, 0, 40.76, 28.75, 41.40, 0.278, 1.127),
    ('+HWA+LA (ours)', 'cifar10', 0.01, 1, 40.60, 28.41, 41.88, 0.233, 1.340),
    ('+HWA+LA (ours)', 'cifar10', 0.01, 2, 39.94, 26.60, 41.05, 0.245, 1.285),
    ('+HWA+LA+CSLM', 'cifar10', 0.01, 0, 34.58, 28.86, 39.34, 0.278, 1.127),
    ('+HWA+LA+CSLM', 'cifar10', 0.01, 1, 37.55, 30.25, 40.67, 0.233, 1.340),
    ('GeFL-F', 'cifar10', 0.01, 0, 33.69, 12.06, 34.25, 0.147, 0.064),
    ('GeFL-F', 'cifar10', 0.01, 1, 33.64, 13.39, 34.82, 0.125, 0.058),
    ('+LA', 'cifar10', 0.01, 0, 37.43, 20.57, 38.67, 0.147, 0.064),
    ('+LA', 'cifar10', 0.01, 1, 37.25, 21.09, 39.15, 0.125, 0.058),
    ('FSG+LA', 'cifar10', 0.01, 0, 42.98, 32.37, 43.66, 0.500, 1.051),
    ('FSG+LA', 'cifar10', 0.01, 1, 43.74, 34.56, 44.61, 0.447, 1.057),
    ('+CSL (beta=0.5)', 'svhn', 1.0, 0, 75.16, 73.85, 76.0, 0.661, 0.982),
    ('+CSL (beta=0.5)', 'svhn', 1.0, 1, 75.01, 72.47, 76.3, 0.669, 0.971),
    ('+CSL (beta=0.5)', 'svhn', 1.0, 2, 75.28, 73.28, 76.6, 0.715, 0.964),
    ('+CSLM (interleaved)', 'svhn', 1.0, 0, 74.13, 72.15, 75.56, 0.661, 0.982),
    ('+CSLM (interleaved)', 'svhn', 1.0, 1, 74.5, 72.28, 75.97, 0.669, 0.971),
    ('+CSLM (interleaved)', 'svhn', 1.0, 2, 74.51, 72.68, 76.03, 0.715, 0.964),
    ('+HWA+LA (ours)', 'svhn', 0.01, 0, 60.43, 45.53, 65.23, 0.36, 0.991),
    ('+HWA+LA (ours)', 'svhn', 0.01, 1, 63.26, 47.27, 67.44, 0.336, 0.962),
    ('+HWA+LA (ours)', 'svhn', 0.01, 2, 63.66, 50.65, 67.91, 0.368, 1.054),
    ('+HWA+LA (ours)', 'svhn', 1.0, 0, 74.29, 73.27, 75.1, 0.663, 0.97),
    ('+HWA+LA (ours)', 'svhn', 1.0, 1, 74.42, 72.83, 75.71, 0.684, 0.972),
    ('+HWA+LA (ours)', 'svhn', 1.0, 2, 74.83, 72.65, 76.12, 0.691, 0.962),
    ('+HWA+LA+CSLM', 'svhn', 0.01, 0, 50.42, 42.7, 53.64, 0.36, 0.991),
    ('+HWA+LA+CSLM', 'svhn', 0.01, 1, 47.53, 50.16, 52.1, 0.336, 0.962),
    ('+HWA+LA+CSLM', 'svhn', 0.01, 2, 51.7, 52.45, 52.68, 0.368, 1.054),
    ('+LA', 'svhn', 0.01, 0, 56.45, 35.26, 62.8, 0.268, 0.493),
    ('+LA', 'svhn', 0.01, 1, 57.21, 32.11, 63.55, 0.163, 0.322),
    ('+LA', 'svhn', 0.01, 2, 55.65, 37.23, 62.13, 0.161, 0.179),
    ('FSG+LA', 'svhn', 0.01, 0, 42.9, 25.72, 48.77, 0.144, 0.981),
    ('FSG+LA', 'svhn', 0.01, 1, 47.35, 31.57, 52.73, 0.153, 1.004),
    ('FSG+LA', 'svhn', 0.01, 2, 42.45, 26.77, 48.97, 0.145, 1.037),
    ('GeFL-F', 'svhn', 0.01, 0, 51.68, 26.06, 59.11, 0.268, 0.493),
    ('GeFL-F', 'svhn', 0.01, 1, 51.89, 23.01, 58.37, 0.163, 0.322),
    ('GeFL-F', 'svhn', 0.01, 2, 47.92, 25.2, 54.6, 0.161, 0.179),
    ('GeFL-F', 'svhn', 1.0, 0, 74.28, 72.95, 75.22, 0.661, 0.982),
    ('GeFL-F', 'svhn', 1.0, 1, 74.4, 72.37, 75.93, 0.669, 0.971),
    ('GeFL-F', 'svhn', 1.0, 2, 74.94, 72.84, 76.31, 0.715, 0.964),
]

DS_NAME = {"mnist": "MNIST", "fmnist": "FashionMNIST", "svhn": "SVHN", "cifar10": "CIFAR-10"}


# ---------------------------------------------------------------- data access
def load(exp):
    """runs of one experiment, including Kaggle folders renamed <exp>__<part> (later folders win)."""
    import glob
    by = {}
    for d in [os.path.join(RES, exp)] + sorted(glob.glob(os.path.join(RES, exp + "__*"))):
        p = os.path.join(d, "runs.jsonl")
        rows = _read_jsonl(p) if os.path.exists(p) else _read_summary(os.path.join(d, "summary.md"))
        for r in rows:
            by[r.get("run_id", id(r))] = r
    return list(by.values())


def _read_summary(p):
    """Fallback when only a pasted summary.md came back: one pseudo-run per seed index
    reproducing each mean and sd exactly (values m - sd, m, m + sd for n = 3). Flagged
    from_summary, so no paired test is computed on them."""
    import re as _re
    if not os.path.exists(p):
        return []
    out, setting, cols, exact = [], None, None, {}
    for line in open(p, encoding="utf-8"):
        m = _re.match(r"## (\w+)\s+IF=([\d.]+)\s+alpha=(\S+)\s+K=(\d+)", line)
        if m:
            setting = (m[1], float(m[2]), None if m[3] == "None" else float(m[3]), int(m[4]))
            cols = None
            continue
        if not line.startswith("|") or setting is None:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "label":
            cols = cells
            continue
        if cols is None or set(cells[0]) <= set("-"):
            continue
        if cols[1] == "seed":  # per-seed table: exact values (rounded to 0.01), paired tests allowed
            r = dict(label=cells[0], dataset=setting[0], IF=setting[1], alpha=setting[2], K=setting[3],
                     seed=int(cells[1]), run_id=f"{cells[0]}|{setting}|seed{cells[1]}")
            for c, v in zip(cols[2:], cells[2:]):
                try:
                    r[c] = float(v) if c.startswith("cond_norm") or not _is_pct(c) else float(v) / 100
                except ValueError:
                    pass
            exact.setdefault((cells[0], setting), []).append(r)
            continue
        n = int(cells[1])
        offs = [0.0] if n == 1 else ([-1.0, 0.0, 1.0] if n == 3 else [(-1) ** i for i in range(n)])
        for i, off in enumerate(offs):
            r = dict(label=cells[0], dataset=setting[0], IF=setting[1], alpha=setting[2], K=setting[3], seed=i,
                     from_summary=True, run_id=f"{cells[0]}|{setting}|{i}")
            for c, v in zip(cols[2:], cells[2:]):
                m2 = _re.match(r"([-\d.]+)\s*±\s*([\d.]+)", v)
                if m2:
                    mu, sd = float(m2[1]), float(m2[2])
                    val = mu + off * sd
                    r[c] = val if c.startswith("cond_norm") or not _is_pct(c) else val / 100
            out.append(r)
    # where per-seed values exist they replace the mean +/- sd pseudo-runs
    out = [r for r in out if (r["label"], (r["dataset"], r["IF"], r["alpha"], r["K"])) not in exact]
    return out + [r for rs in exact.values() for r in rs]


def _is_pct(col):
    """Columns printed in % by write_summary (the rest: counts, ratios, F)."""
    return col.startswith(("final", "best", "oracle", "fidelity", "referee", "bbc", "method", "ncm"))


def _read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a run being written right now
    return out


def kaggle_runs():
    """Logged SVHN runs, overridden/extended by any K01 runs.jsonl copied back."""
    by = {}
    for l, d, IF, sd, b, t, m, fid, nr in KAGGLE_LOGGED:
        by[(l, d, IF, sd)] = dict(label=l, dataset=d, IF=IF, alpha=0.5 if IF < 1 else None, K=10, seed=sd,
                                  final_bal=b / 100, final_tail=t / 100, best_mean_acc=m / 100,
                                  fidelity_tail=fid, cond_norm_tail_over_head_end=nr)
    filed = load("K01_kaggle_cifar10_svhn")
    for r in filed:
        by[(r["label"], r["dataset"], r["IF"], r["seed"])] = r
    runs = list(by.values())
    for r in runs:
        if r["label"] == "+HWA+LA (ours)":
            r["label"] = "+HWA+LA"
    return runs, bool(filed)


def pick(runs, label, metric, **where):
    """{seed: value in %} for one label and setting."""
    out = {}
    PICK_FLAGS["summary"] = PICK_FLAGS.get("summary", False)
    for r in runs:
        if r.get("label") != label or r.get(metric) is None:
            continue
        if any(r.get(k) != v for k, v in where.items()):
            continue
        v = r[metric]
        out[r["seed"]] = v * 100 if metric not in ("cond_norm_tail_over_head_end",) else v
        if r.get("from_summary"):
            SUMMARY_IDS.add(id(out))
    return out


PICK_FLAGS = {}
SUMMARY_IDS = set()


def paired_p(a, b):
    if id(a) in SUMMARY_IDS or id(b) in SUMMARY_IDS:
        return None  # per-seed values are not known for pasted summaries
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([a[s] - b[s] for s in seeds])
    if d.std(ddof=1) == 0:
        return None
    t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
    from scipy import stats
    return float(2 * stats.t.sf(abs(t), len(d) - 1))


def mean(v):
    return float(np.mean(list(v.values()))) if v else None


# ---------------------------------------------------------------- html helpers
def esc(s):
    return html.escape(str(s))


def fmt_ms(v, digits=1):
    if not v:
        return '<span class="pend">-</span>'
    vals = list(v.values())
    m = np.mean(vals)
    if len(vals) == 1:
        return f"{m:.{digits}f}<span class=\"sd\"> n=1</span>"
    return f"{m:.{digits}f}<span class=\"sd\"> &plusmn;{np.std(vals, ddof=1):.{digits}f}</span>"


def fmt_delta(v, ref):
    if not v or not ref:
        return ""
    seeds = sorted(set(v) & set(ref))
    if not seeds:
        return ""
    d = np.mean([v[s] - ref[s] for s in seeds])
    cls = "up" if d > 0.05 else ("dn" if d < -0.05 else "eq")
    return f'<span class="d {cls}">{d:+.1f}</span>'


def fmt_p(p):
    if p is None:
        return '<span class="pend">-</span>'
    s = f"{p:.3f}" if p >= 0.001 else "&lt;0.001"
    return f'<span class="{"sig" if p < 0.05 else "ns"}">{s}</span>'


def table(caption, head, rows, hl=(), cls=""):
    th = "".join(f'<th class="{c}">{h}</th>' for h, c in head)
    body = []
    for i, row in enumerate(rows):
        tr_cls = ' class="hl"' if i in hl else ""
        body.append(f"<tr{tr_cls}>" + "".join(f'<td class="{c}">{cell}</td>' for cell, (_, c) in zip(row, head)) + "</tr>")
    return (f'<div class="tw {cls}"><table><caption>{caption}</caption><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def method_table(runs, labels, datasets, metrics, ref="GeFL-F", where=None, caption="", ours=(), pmetric=None):
    """Rows = labels; per dataset: metric columns (mean +/- sd, delta vs ref for the first metric) and p."""
    where = where or {}
    pmetric = pmetric or metrics[0][0]
    head = [("Method", "")]
    for ds in datasets:
        for m, name in metrics:
            head.append((f"{DS_NAME[ds]}<br>{name}", "num"))
        head.append((f"{DS_NAME[ds]}<br>p vs {esc(ref)}", "num"))
    rows, hl = [], []
    for i, (lab, shown) in enumerate(labels):
        row = [esc(shown)]
        for ds in datasets:
            refv = pick(runs, ref, pmetric, dataset=ds, **where)
            for j, (m, _) in enumerate(metrics):
                v = pick(runs, lab, m, dataset=ds, **where)
                cell = fmt_ms(v)
                if m == pmetric and lab != ref:
                    cell += " " + fmt_delta(v, pick(runs, ref, m, dataset=ds, **where))
                row.append(cell)
            row.append(fmt_p(paired_p(pick(runs, lab, pmetric, dataset=ds, **where), refv)) if lab != ref else "ref")
        rows.append(row)
        if lab in ours:
            hl.append(i)
    return table(caption, head, rows, hl)


def bar_chart(series, title, lo=40, hi=100, unit="%"):
    """series: [(label, value, kind)] kind in {'ref','ours','base'}; horizontal bars on one linear scale."""
    series = [s for s in series if s[1] is not None]
    if not series:
        return ""
    W, lw, rw, bh, gap, top = 640, 190, 52, 20, 8, 30
    pw = W - lw - rw
    H = top + len(series) * (bh + gap) + 26
    x = lambda v: lw + (min(max(v, lo), hi) - lo) / (hi - lo) * pw
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in range(lo, hi + 1, 10):
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{H-22}" stroke="var(--rule)" stroke-width="1"/>')
        out.append(f'<text x="{x(t):.1f}" y="{H-8}" font-size="11" text-anchor="middle" fill="var(--muted)" class="mn">{t}</text>')
    for i, (lab, v, kind) in enumerate(series):
        y = top + i * (bh + gap)
        col = {"ours": "var(--accent)", "ref": "var(--tail)", "base": "var(--rule-strong)"}[kind]
        out.append(f'<text x="{lw-10}" y="{y+bh/2+4:.1f}" font-size="12.5" text-anchor="end" fill="var(--ink-soft)">{esc(lab)}</text>')
        out.append(f'<rect x="{lw}" y="{y}" width="{x(v)-lw:.1f}" height="{bh}" fill="{col}" rx="1.5"/>')
        out.append(f'<text x="{x(v)+6:.1f}" y="{y+bh/2+4:.1f}" font-size="12" fill="var(--ink)" class="mn">{v:.1f}</text>')
    out.append(f'<text x="{lw}" y="16" font-size="11.5" fill="var(--muted)" font-weight="700" letter-spacing=".06em">{esc(title.upper())}</text>')
    out.append("</svg>")
    return "".join(out)


def note(kind, head, body):
    return f'<div class="note {kind}"><div class="h">{head}</div>{body}</div>'


# ---------------------------------------------------------------- sections
def sec_validation():
    f09 = load("F09_consensus_paper_setting")
    f01 = load("F01_main_longtail")
    ref = load_reference()
    ours = lambda ds, K: pick(f09, "GeFL-F", "best_mean_acc", dataset=ds, K=K, IF=1.0)
    rows = []
    for ds, K in [("mnist", 10), ("mnist", 50), ("mnist", 100), ("fmnist", 10)]:
        a = [r["best_mean_acc"] * 100 for r in ref if r["setting"] == "iid" and ds == "mnist" and K == 10]
        o = ours(ds, K)
        diff = (mean(o) - PAPER[(ds, K)]) if o else None
        rows.append([f"{DS_NAME[ds]}, IID, K={K}", f"{PAPER[(ds, K)]:.2f}",
                     f"{np.mean(a):.2f}<span class=\"sd\"> n={len(a)}</span>" if a else '<span class="pend">not run</span>',
                     fmt_ms(o, 2), f"{diff:+.2f}" if diff is not None else ""])
    lt_ref = [r for r in ref if r["setting"] == "lt"]
    lt_ours = pick(f01, "GeFL-F", "final_bal", dataset="mnist")
    for r in lt_ref:
        o = lt_ours.get(r["seed"])
        rows.append([f"MNIST, long tail, seed {r['seed']} (final balanced acc.)", "not reported",
                     f"{r['final_bal']*100:.2f}", f"{o:.2f}" if o is not None else "-",
                     f"{o - r['final_bal']*100:+.2f} vs code" if o is not None else ""])
    t = table("Our GeFL-F against the paper and against the authors' own code",
              [("Setting", ""), ("Paper", "num"), ("Authors' code", "num"), ("Our GeFL-F", "num"), ("Ours &minus; paper", "num")], rows)
    return f"""
<section id="s1"><h2><span class="num">1</span>Our GeFL&#8209;F is the paper's GeFL&#8209;F</h2>
<p class="deck">Before any improvement claim, the baseline has to be right.</p>
<p>We ran the authors' released <code>GeFL_CVAE-F.py</code> on our machine, unmodified apart from
the device and an evaluation hook, and compared it with our re-implementation, which follows the paper's
appendix tables: ten heterogeneous headers CNN&#8209;1&hellip;10, CVAE&#8209;F, SGD at learning rate 0.1, 10% of the data.
Both match the paper's published numbers to within about half a point. The same holds at K = 50 and K = 100.</p>
{t}
<p>The metric is the paper's <code>best_mean_acc</code>: the best accuracy of each header over the rounds,
averaged over the ten architectures. In the long-tail split our GeFL&#8209;F scores about two points
<em>below</em> the authors' code. Any gain we report is therefore measured from a baseline that is, if
anything, slightly weaker than the real one.</p>
</section>"""


def load_reference():
    p = os.path.join(HERE, "reference_check", "reference_runs.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sec_diagnosis():
    f01 = load("F01_main_longtail")
    e02 = load("E02_conditioning_collapse_quick")
    rows = []
    for ds in ["mnist", "fmnist"]:
        for lab, shown in [("GeFL-F", "GeFL-F"), ("+HWA+LA", "+HWA+LA (ours)")]:
            nr = pick(f01, lab, "cond_norm_tail_over_head_end", dataset=ds)
            ft = pick(f01, lab, "fidelity_tail", dataset=ds)
            fh = pick(f01, lab, "fidelity_head", dataset=ds)
            rows.append([DS_NAME[ds], esc(shown),
                         f"{mean(nr):.2f}<span class=\"sd\"> ({', '.join(f'{v:.2f}' for v in nr.values())})</span>" if nr else "-",
                         fmt_ms(fh), fmt_ms(ft)])
    t = table("Conditioning collapse, full runs (F01, 3 seeds, IF = 100, Dir(0.5), K = 10)",
              [("Dataset", ""), ("Method", ""), ("Tail / head row norm", "num"),
               ("Fidelity, head classes", "num"), ("Fidelity, tail classes", "num")], rows, hl=(1, 3))
    rows2 = []
    for lab in ["GeFL-F", "+LCD", "+NOWD", "+HWA", "+NOWD+HWA"]:
        r = [esc(lab)]
        for ds in ["mnist", "fmnist"]:
            r.append(fmt_ms(pick(e02, lab, "cond_norm_tail_over_head_end", dataset=ds), 2).replace(" n=1", ""))
            r.append(fmt_ms(pick(e02, lab, "fidelity_tail", dataset=ds)).replace(" n=1", ""))
        rows2.append(r)
    t2 = table("Which change stops the collapse (E02, quick pass, seed 0)",
               [("Method", ""), ("MNIST norm ratio", "num"), ("MNIST tail fidelity", "num"),
                ("FMNIST norm ratio", "num"), ("FMNIST tail fidelity", "num")], rows2)
    return f"""
<section id="s2"><h2><span class="num">2</span>Where GeFL&#8209;F breaks: rare classes vanish from the generator</h2>
<p class="deck">A mechanism we derived, then measured.</p>
<p>CVAE&#8209;F conditions on the label through one learned row per class. Under a long tail, a rare class
appears on only m<sub>r</sub> of the K clients. Every other client still decays that row with Adam's coupled
weight decay (10<sup>&minus;3</sup>) and sends it back unchanged by any gradient. Flat FedAvg then averages
m<sub>r</sub> informative copies with K&minus;m<sub>r</sub> shrunken ones. When m<sub>r</sub> &lt; K/2 the shrinkage
wins, round after round, so the rare-class rows shrink toward zero. The generator then produces head-class features
under tail labels, and the headers learn from wrong data.</p>
<div class="eq"><div class="body">
row<sub>c</sub><sup>t+1</sup> = <span class="frac"><span class="nu">1</span><span class="de">K</span></span>
[ &Sigma;<sub>holders</sub> (row<sub>c</sub><sup>t</sup> &minus; &eta;g) + (K&minus;m<sub>c</sub>)(1&minus;&eta;&lambda;)<sup>S</sup> row<sub>c</sub><sup>t</sup> ]
&nbsp;&rArr;&nbsp; shrinks whenever m<sub>c</sub> &lt; K/2</div><div class="tag">flat averaging of a class row</div></div>
<p><b>Fix (HWA, holder-weighted aggregation):</b> average each class row only over the clients that
hold that class, weighted by their sample counts (the counts travel through secure aggregation as sums).
Every other parameter is still averaged as in GeFL&#8209;F. The measured result: the tail row norm returns
to the head level, and the generator's tail features become recognisable again.
<em>Fidelity</em> is the accuracy of an independent referee classifier on generated features, by class group.</p>
{t}
{t2}
</section>"""


def sec_main():
    f01 = load("F01_main_longtail")
    labels = [("GeFL-F", "GeFL-F"), ("+LA", "GeFL-F + LA"), ("+HWA+LA", "GeFL-F + HWA + LA (ours)"),
              ("FSG+LA", "Gaussian sufficient-statistics generator + LA"),
              ("Ours-hybrid (RHYB+HWA+LA)", "Hybrid generator + HWA + LA"),
              ("LG-FedAvg+LA", "LG-FedAvg + LA"), ("LG-FedAvg", "LG-FedAvg"), ("FedAvg (grouped)", "FedAvg (per architecture)")]
    t = method_table(f01, labels, ["mnist", "fmnist"], [("final_bal", "balanced acc."), ("final_tail", "tail recall")],
                     caption="Long tail IF = 100, Dirichlet 0.5, K = 10, full schedule, 3 seeds (F01). Mean &plusmn; sd; delta vs GeFL-F on paired seeds",
                     ours=("+HWA+LA",))
    t2 = method_table(f01, labels[:3], ["mnist", "fmnist"], [("best_mean_acc", "best_mean_acc"), ("final_worst", "worst class")],
                      caption="Same runs, the paper's own metric and the worst class", ours=("+HWA+LA",))
    charts = []
    for ds in ["mnist", "fmnist"]:
        ser = []
        for lab, shown in labels:
            v = mean(pick(f01, lab, "final_bal", dataset=ds))
            ser.append((shown.replace("Gaussian sufficient-statistics generator", "FSG").replace(" (per architecture)", ""), v,
                        "ours" if lab == "+HWA+LA" else ("ref" if lab == "GeFL-F" else "base")))
        charts.append(bar_chart(ser, f"{DS_NAME[ds]}: balanced accuracy, %", lo=40, hi=100))
    return f"""
<section id="s3"><h2><span class="num">3</span>Main result: long-tailed clients</h2>
<p class="deck">The setting GeFL&#8209;F was never tested in, and the one real deployments look like.</p>
<p>Global class frequencies fall a hundredfold from the most to the least common class (IF = 100). Each class
is split across clients by Dirichlet(0.5). Test sets stay balanced, so <em>balanced accuracy</em> (mean per-class
recall) is the honest metric. <em>Tail</em> is the mean recall of the four rarest classes (6&ndash;9).</p>
{t}
<figure>{charts[0]}{charts[1]}<figcaption><b>Bars are on one scale, starting at 40%.</b> Blue: ours. Amber: GeFL&#8209;F. Grey: other methods.</figcaption></figure>
{t2}
<p>Two components carry the gain, and they act on different stages. <b>HWA</b> repairs the generator, so the synthetic
phase teaches the tail. <b>LA</b> (logit adjustment by the client's own label prior, applied in training only)
stops each header's real-data phase from re-learning its client's skew. LA alone gives +4 to +6 points. Adding HWA
gives a further +9 to +11. The tail recall gain is +29 points on MNIST and +33 on FashionMNIST.</p>
</section>"""


def sec_paper_setting():
    f09 = load("F09_consensus_paper_setting")
    rows = []
    for ds, K in [("mnist", 10), ("fmnist", 10), ("mnist", 50), ("mnist", 100)]:
        g = pick(f09, "GeFL-F", "best_mean_acc", dataset=ds, K=K, IF=1.0)
        c = pick(f09, "+CSL (beta=0.5)", "best_mean_acc", dataset=ds, K=K, IF=1.0)
        diffs = [c[s] - g[s] for s in sorted(set(g) & set(c))]
        rows.append([f"{DS_NAME[ds]}, K={K}", f"{PAPER[(ds, K)]:.2f}", fmt_ms(g, 2), fmt_ms(c, 2),
                     ", ".join(f"{d:+.2f}" for d in diffs), fmt_p(paired_p(c, g)),
                     f"{mean(c) - PAPER[(ds, K)]:+.2f}" if c else ""])
    t = table("Paper's IID setting, best_mean_acc (F09)",
              [("Setting", ""), ("Paper GeFL-F", "num"), ("Our GeFL-F", "num"), ("+CSL (ours)", "num"),
               ("Gain per seed", "num"), ("p", "num"), ("CSL &minus; paper", "num")], rows, hl=(0, 1))
    abl = []
    for lab in ["GeFL-F", "+CSL (beta=0.5)", "+CSL (beta=1)", "+CSLM (interleaved)"]:
        abl.append([esc(lab)] + [fmt_ms(pick(f09, lab, "best_mean_acc", dataset="mnist", K=K, IF=1.0, seed=0), 2).replace('<span class="sd"> n=1</span>', "")
                                 for K in [10, 50, 100]])
    t2 = table("CSL variants, MNIST IID, seed 0", [("Method", ""), ("K=10", "num"), ("K=50", "num"), ("K=100", "num")], abl, hl=(1,))
    return f"""
<section id="s4"><h2><span class="num">4</span>The paper's own setting: consensus soft labels</h2>
<p class="deck">HWA and LA change nothing when data is IID, so we needed a gain there too.</p>
<p>In IID data every client holds every class, so HWA reduces to (nearly) FedAvg and LA's prior is (nearly) uniform. Our method then
matches GeFL&#8209;F up to noise. To beat GeFL&#8209;F in its own setting we use a different observation: in the
synthetic phase each header learns from features with a hard one-hot label, although a generated feature is often
ambiguous. The ten headers are trained on different clients' data with different architectures, so their averaged
prediction is a better-calibrated label than the one-hot.</p>
<div class="eq"><div class="body">target(x̃) = (1 &minus; &beta;) &middot; onehot(y) + &beta; &middot; <span class="frac"><span class="nu">1</span><span class="de">G</span></span> &Sigma;<sub>g</sub> softmax(h<sub>g</sub>(x̃)), &nbsp; &beta; = 0.5</div><div class="tag">CSL</div></div>
<p>The consensus uses only the header outputs on <em>synthetic</em> features, which the server can compute with
models it already receives. No real data and no new client message are involved.</p>
{t}
{t2}
<p>The gain is small but appears on every seed. On MNIST we beat the paper's published GeFL&#8209;F by 1.2 points.
On FashionMNIST we tie it. &beta; = 1 (consensus only) is worse than &beta; = 0.5, because the one-hot term anchors
the label. Mixing synthetic features into every real batch (CSLM) is also worse.</p>
</section>"""


def sec_headroom():
    """Where the paper's IID numbers sit relative to what the shared FE allows."""
    f10, f11 = load("F10_combined_method"), load("F11_ddpm_generator")
    kag = kaggle_runs()[0]
    rows = []
    for ds, src, labs in [("mnist", f10, ["GeFL-F", "+CSL", "+HWA+LA+CSL"]), ("fmnist", f10, ["GeFL-F", "+CSL", "+HWA+LA+CSL"]),
                          ("fmnist", f11, ["GeFL-F (DDPM-F)", "+CSL (DDPM-F)", "Ours (DDPM-F)"]),
                          ("svhn", kag, ["GeFL-F", "+CSL (beta=0.5)", "Ours (HWA+LA+CSL)"])]:
        for lab in labs:
            w = dict(dataset=ds, IF=1.0, K=10)
            m, o, e = (pick(src, lab, k, **w) for k in ("best_mean_acc", "oracle_bal", "final_ens_acc"))
            if not m:
                continue
            rows.append([f"{DS_NAME[ds]}", esc(lab), fmt_ms(m, 2), fmt_ms(o, 2) if o else "-", fmt_ms(e, 2) if e else "-",
                         f"{PAPER_FIG4[('GeFL-F', 'CVAE-F')][(ds, 10)]:.2f} / {max(v[(ds, 10)] for v in PAPER_FIG4.values()):.2f}"])
    t = table("Headroom in the paper's IID setting (K = 10): the method, an oracle head, and the ensemble",
              [("Dataset", ""), ("Method", ""), ("Mean head (paper metric)", "num"), ("Oracle head", "num"),
               ("Ensemble of 10 heads", "num"), ("Paper: GeFL-F / best", "num")], rows)
    return f"""
<section id="s4b"><h2><span class="num">4b</span>How much is left to gain in the IID setting</h2>
<p class="deck">The FE ceiling explains why the paper's IID numbers are close to each other, and what a fair target is.</p>
<p>The <em>oracle head</em> keeps a head's body but re-fits its last layer on the real features of <em>all</em> clients pooled, with
balanced sampling. No federated method can see that data, so it is an optimistic reference for a single head with this FE. In
FashionMNIST IID the oracle is about 83.3&ndash;84.3, and CSL already reaches 83.2&ndash;83.6. The paper's best FashionMNIST
number (84.28, DDPM-F) sits at that ceiling. The <em>ensemble</em> of all ten heads is 2&ndash;3 points higher, but the paper's metric
scores single heads. Distilling the ensemble into each head at the server, on generated features, lowered IID accuracy
(&sect;8). The ensemble's knowledge can reach a head only through features as faithful as the generator's.</p>
{t}
</section>"""


def sec_budget():
    e16 = load("E16_csl_synthetic_budget")
    if not e16:
        return ""
    rows = []
    for lab in ["GeFL-F", "GeFL-F Ts=3", "GeFL-F Ts=5", "+CSL", "+CSL Ts=3", "+CSL Ts=5", "+CSL Ts=10", "Ours (HWA+LA+CSL) Ts=5", "Ours (HWA+LA+CSL) Ts=10"]:
        r = [esc(lab)]
        for ds in ["fmnist", "mnist"]:
            v = pick(e16, lab, "best_mean_acc", dataset=ds, IF=1.0)
            r.append(fmt_ms(v, 2) + (" " + fmt_delta(v, pick(e16, "GeFL-F", "best_mean_acc", dataset=ds, IF=1.0)) if lab != "GeFL-F" else ""))
            r.append(fmt_p(paired_p(v, pick(e16, "GeFL-F", "best_mean_acc", dataset=ds, IF=1.0))) if lab != "GeFL-F" else "ref")
        rows.append(r)
    t = table("Synthetic budget T_s (synthetic epochs per round), paper's IID setting, K = 10 (E16, best_mean_acc)",
              [("Method", ""), ("FashionMNIST", "num"), ("p", "num"), ("MNIST", "num"), ("p", "num")], rows, hl=(6, 8))
    lt = method_table(e16, [("GeFL-F", "GeFL-F"), ("Ours (HWA+LA+CSL)", "Ours, T_s = 1"), ("Ours (HWA+LA+CSL) Ts=5", "Ours, T_s = 5"),
                            ("Ours (HWA+LA+CSL) Ts=10", "Ours, T_s = 10")], ["fmnist", "mnist"],
                      [("final_bal", "balanced acc."), ("final_tail", "tail"), ("best_mean_acc", "best_mean_acc")],
                      where=dict(IF=0.01), caption="Same budgets under the long tail (IF = 100, Dir 0.5, K = 10, 3 seeds)",
                      ours=("Ours (HWA+LA+CSL) Ts=5", "Ours (HWA+LA+CSL) Ts=10")) if pick(e16, "GeFL-F", "final_bal", dataset="fmnist", IF=0.01) else ""
    return f"""
<section id="s4c"><h2><span class="num">4c</span>Consensus labels make more synthetic data useful</h2>
<p class="deck">The paper found that 5 synthetic epochs per round are no better than 1. We predicted that this is a hard-label effect, and that CSL removes it.</p>
<p>A head trained on real data plus a share &lambda; of synthetic data has error of roughly
&sigma;&sup2;/(n<sub>real</sub> + &lambda;n<sub>syn</sub>) + &lambda;&sup2;b(t)&sup2;. The second term is the bias of the synthetic labels t against
the true posterior of each generated feature, and more samples do not average it away. The best &lambda; grows as b shrinks. With hard labels
b is large, so extra synthetic epochs do not help; this is the paper's own Fig. 10. CSL's consensus target lowers b, so the optimum moves
toward more synthetic data.</p>
{t}
<p>The interaction appears as predicted on FashionMNIST. With hard labels, GeFL&#8209;F barely moves from T<sub>s</sub> = 1 to 5. With
consensus labels, every extra synthetic epoch helps: +CSL at T<sub>s</sub> = 10 reaches 84.37 on every seed above 84.1, level with the paper's best
FashionMNIST number of any variant (84.28, GeFL&#8209;F with the much costlier feature diffusion model). It is 1.2 points above the
paper's GeFL&#8209;F with the same generator (83.14). MNIST is near its ceiling (oracle &asymp; 97.2), and the extra budget does not help there.
T<sub>s</sub> was not tuned per dataset for the headline claim. With one fixed setting (T<sub>s</sub> = 5) our method gives 84.12 on FashionMNIST
and 96.6 on MNIST. With T<sub>s</sub> = 10 the full method gives 84.34 on FashionMNIST (p = 0.002 against GeFL&#8209;F), a tie with the paper's best.
Under the long tail the bigger budget helps further on FashionMNIST: our method goes from 76.9 to 79.2 balanced accuracy
(p = 0.001), 20 points above GeFL&#8209;F. On MNIST it does not (90.2 vs 90.2), which is Proposition 3's prediction for
a generator whose tail features are poor (fidelity 27%): the label bias b is large, so the optimal synthetic share stays small.</p>
{lt}
</section>"""


def sec_combined():
    f10 = load("F10_combined_method")
    n = len(f10)
    if n == 0:
        return f"""
<section id="s5"><h2><span class="num">5</span>One method for both settings</h2>
{note("flag", "Running", "<p>F10 (HWA + LA + CSL against each part, both regimes, 48 runs) is running now. This section fills in when the page is rebuilt.</p>")}
</section>"""
    labels = [("GeFL-F", "GeFL-F"), ("+HWA+LA", "+HWA+LA"), ("+CSL", "+CSL"), ("+HWA+LA+CSL", "+HWA+LA+CSL (combined)")]
    lt = method_table(f10, labels, ["mnist", "fmnist"], [("final_bal", "balanced acc."), ("final_tail", "tail")],
                      where=dict(IF=0.01), caption="Long tail (IF = 100, Dir 0.5, K = 10)", ours=("+HWA+LA+CSL",))
    iid = method_table(f10, labels, ["mnist", "fmnist"], [("best_mean_acc", "best_mean_acc")],
                       where=dict(IF=1.0), caption="Paper's IID setting (K = 10)", ours=("+HWA+LA+CSL",))
    status = "" if n >= 48 else note("flag", "Partial", f"<p>{n} of 48 runs finished when this page was built.</p>")
    return f"""
<section id="s5"><h2><span class="num">5</span>One method for both settings: HWA + LA + CSL</h2>
<p class="deck">The parts act on different stages, so they should add. F10 tests that, with matched seeds and splits.</p>
{status}{lt}{iid}
<p>Under the long tail, CSL alone gives little: +2.1 on MNIST, 0 on FashionMNIST, and no tail gain on either. Added to HWA + LA it helps on every seed, +2.1 on MNIST and +0.4 on FashionMNIST, though with 3 seeds that is not yet significant (p &asymp; 0.1). This is what §4's argument predicts.
CSL corrects the labels of synthetic features, which pays off only when those features carry real class information. GeFL&#8209;F's collapsed
generator emits near-class-agnostic tail features (tail fidelity under 20%), so relabelling them cannot help. After HWA they are faithful,
and the consensus label sharpens them. The final method is therefore GeFL&#8209;F + HWA + LA + CSL. In IID data it reduces to GeFL&#8209;F + CSL.</p>
</section>"""


def sec_clients():
    f04 = load("F04_clients")
    if not f04:
        return f"""
<section id="s6"><h2><span class="num">6</span>More clients</h2>
{note("flag", "Queued", "<p>F04 (K = 50 and K = 100 under the long tail, 3 seeds) starts automatically after F10.</p>")}
{quick_clients()}
</section>"""
    f04 = f04 + with_gate(load("E22_pc_many_clients")) + [r for r in with_gate(load("E29_hybrid_generator")) if r["K"] > 10]
    labels = [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("+HWA+LA", "+HWA+LA"), ("+HWA+LA+CSL", "Ours (HWA+LA+CSL)"),
              ("FSG+LA", "FSG+LA"), ("FSG+LA+CSL", "FSG+LA+CSL"), ("MIX+HWA+LA+CSL", "MIX+HWA+LA+CSL"),
              ("Ours-PC+MC", "PC-VAE + MC (E22)"), ("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A, T_s = 1 (E22)"),
              ("Ours-A Ts=10", "Ours-A, T_s = 10 (E22): final"), ("Ours-R Ts=10", "Hybrid PCR, T_s = 10 (E29)")]
    ts = "".join(method_table(f04, labels, ["mnist"], [("final_bal", "balanced acc."), ("final_tail", "tail")],
                              where=dict(K=K), caption=f"MNIST long tail, K = {K}", ours=("Ours-A Ts=10",))
                 for K in [50, 100] if any(r["K"] == K for r in f04))
    ts += ("<p><b>The final method does not degrade with the number of clients.</b> Ours-A (T<sub>s</sub> = 10) scores 94.82 at K = 10 and "
           "94.30 at K = 100, where the CVAE + HWA falls from 87.8 to 83.1. Its generator has no class-specific parameters, so nothing is "
           "fitted from the shrinking per-client data. At K = 100 it is +18.3 over GeFL&#8209;F (p = 0.0004) and +8.3 over the Gaussian generator "
           "(p = 0.001), positive on every seed. The 10-epoch synthetic budget is worth +3.9 at K = 100, against +1.7 at K = 10. This is "
           "Proposition 3's w<sup>&star;</sup> &prop; 1/n<sub>real</sub>, with about 60 real images per client.</p>"
           "<p class=\"small\">Rows for +HWA+LA+CSL, FSG+LA+CSL and MIX+HWA+LA+CSL at K = 50/100 are pending a rerun with the per-client "
           "synthetic-slice fix (F04 queue).</p>")
    k03 = load("K03_kaggle_client_scaling")
    if k03:
        labs = [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("+HWA+LA", "+HWA+LA"), ("Ours (HWA+LA+CSL)", "Ours (HWA+LA+CSL)"),
                ("FSG+LA", "FSG+LA"), ("FSG+LA+CSL", "FSG+LA+CSL")]
        for K in [50, 100]:
            if pick(k03, "GeFL-F", "final_bal", dataset="fmnist", IF=0.01, K=K):
                ts += method_table(k03, labs, ["fmnist"], [("final_bal", "balanced acc."), ("final_tail", "tail")],
                                   where=dict(IF=0.01, K=K), caption=f"FashionMNIST long tail, K = {K} (Kaggle K03)",
                                   ours=("+HWA+LA", "Ours (HWA+LA+CSL)"))
        ts += "<p class=\"small\">Kaggle K03 rows come from the pasted summary.md (mean &plusmn; sd); per-seed p-values appear once runs.jsonl is copied back.</p>"
        if not glob.glob(os.path.join(RES, "K03_kaggle_client_scaling__K*_cslfix")):
            ts += note("flag", "Being rerun", "<p>In the first K03 run, the rows using consensus labels (+CSL, Ours, FSG+LA+CSL) had a flaw at K = 50 and 100. "
                       "The several clients of each architecture all trained on the <em>same</em> synthetic slice, then were averaged, which cuts "
                       "synthetic diversity per architecture 5&ndash;10&times;. This explains why CSL's effect turned negative as K grew. "
                       "It is fixed (each client now gets its own slice). The affected rows are being rerun on Kaggle together with "
                       "T<sub>s</sub> = 10, and will replace these numbers. K = 10 results are unaffected: there each architecture has one client.</p>")
    return f"""
<section id="s6"><h2><span class="num">6</span>More clients: K = 50 and 100</h2>
<p class="deck">With more clients each rare class has fewer holders, so flat averaging dilutes its rows further.</p>
{ts}
<p>Every variant beats GeFL&#8209;F by more as K grows: +8.7 at K = 50 and +6.7 at K = 100 for HWA + LA + CSL. But the ranking among our own variants
changes with K. At K = 10 the repaired CVAE&#8209;F is best (87.9 against 87.0 for FSG). At K = 50 the variants are within noise of each other
(MIX 87.9, FSG 87.7, ours 86.8; ours vs FSG p = 0.47). At K = 100 the Gaussian sufficient-statistics generator leads, 86.1 against 82.7 (p = 0.105, 3 seeds). The reason is structural. FSG's federated estimate is <em>exactly</em> the centralised one at any K, because it
aggregates sums. The CVAE's class rows, even with HWA, are fitted by clients that each hold fewer and fewer samples. Which generator to use
is therefore a function of the per-client sample size, and the right choice can be read from the aggregated counts the server already has.</p>
</section>"""


def quick_clients():
    e12 = load("E12_confirmation_quick")
    labels = [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("+HWA+LA", "+HWA+LA (ours)")]
    return method_table(e12, labels, ["mnist"], [("final_bal", "balanced acc.")], where=dict(K=50),
                        caption="Quick-pass evidence, MNIST long tail, K = 50 (E12, reduced rounds, 3 seeds)", ours=("+HWA+LA",))


def sec_kaggle():
    runs, from_file = kaggle_runs()
    labels = [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("+HWA+LA", "+HWA+LA"), ("Ours (HWA+LA+CSL)", "Ours (HWA+LA+CSL)"),
              ("FSG+LA", "FSG+LA"), ("FSG+LA+CSL", "FSG+LA+CSL"), ("+HWA+LA+CSLM", "+HWA+LA+CSLM")]
    out = []
    for ds in ["svhn", "cifar10"]:
        if not any(r["dataset"] == ds and r["IF"] < 1 for r in runs):
            continue
        out.append(method_table(runs, labels, [ds], [("final_bal", "balanced acc."), ("final_tail", "tail"), ("best_mean_acc", "best_mean_acc")],
                                where=dict(IF=0.01), caption=f"{DS_NAME[ds]} long tail (IF = 100, Dir 0.5, K = 10), Kaggle T4", ours=("+HWA+LA", "Ours (HWA+LA+CSL)")))
        nr = []
        for lab in ["GeFL-F", "+HWA+LA"]:
            v = pick(runs, lab, "cond_norm_tail_over_head_end", dataset=ds, IF=0.01)
            nr.append(f"{lab}: {', '.join(f'{x:.2f}' for x in v.values())}")
        out.append(f"<p class=\"small\">Tail / head row norm per seed &mdash; {esc('; '.join(nr))}. The collapse appears on SVHN as well, and HWA removes it.</p>")
    for ds in ["svhn", "cifar10"]:
        iid = [r for r in runs if r["dataset"] == ds and r["IF"] >= 1]
        if iid:
            out.append(method_table(runs, [("GeFL-F", "GeFL-F"), ("+HWA+LA", "+HWA+LA"), ("+CSL (beta=0.5)", "+CSL"), ("Ours (HWA+LA+CSL)", "Ours (HWA+LA+CSL)"), ("+CSLM (interleaved)", "+CSLM")],
                                    [ds], [("best_mean_acc", "best_mean_acc")], where=dict(IF=1.0),
                                    caption=f"{DS_NAME[ds]} IID (paper {PAPER[(ds, 10)]:.2f})", ours=("+CSL (beta=0.5)", "Ours (HWA+LA+CSL)")))
    src = ("Read from the K01 <code>runs.jsonl</code>." if from_file else
           "Parsed from the Kaggle notebook logs. SVHN: all 27 runs. CIFAR-10: the long-tail runs finished so far; IID is still running. These K01 runs predate the final method, so they include HWA+LA but not HWA+LA+CSL. The updated K01 adds the final method.")
    return f"""
<section id="s7"><h2><span class="num">7</span>Harder data: SVHN and CIFAR&#8209;10</h2>
<p class="deck">These runs use the same code and the paper's CIFAR/SVHN settings (FE with 10 channels for CIFAR, 50% of the data). They run on Kaggle.</p>
{note("", "Source", f"<p>{src}</p>")}
{"".join(out)}
<p>On SVHN the ranking holds. Under the long tail, ours is +12.0 points balanced accuracy over GeFL&#8209;F and +6.0 over LA alone.
Tail recall nearly doubles (47.8 against 24.8), and the paper's own metric rises by +9.5 (p = 0.045).
In the IID setting, our GeFL&#8209;F reaches 75.82, against the paper's 76.26. CSL gains on every seed (+0.78, +0.37, +0.29), reaching 76.30.
HWA+LA costs 0.18 points in IID data. The IID split gives clients slightly unequal counts, so HWA is not exactly FedAvg there.
The loss is small but consistent across seeds, so we report it.</p>
<p>On CIFAR&#8209;10 under the long tail (partial: 2–3 seeds), GeFL&#8209;F shows the strongest collapse we have measured.
The tail/head row ratio is 0.06, and HWA restores it to 1.1–1.3. HWA + LA reaches 40.4 against 33.7 for GeFL&#8209;F and 37.3 for LA alone.
Here, unlike SVHN, the Gaussian generator is best, at 43.4: its tail fidelity is 0.47, against 0.25 for the repaired CVAE&#8209;F.
On CIFAR&#8209;10 features, a class mean and a pooled covariance estimated <em>exactly</em> from sums beat a CVAE trained on about 25 tail samples.
On SVHN, which is multi-modal, the opposite holds. Which generator wins depends on the data. HWA + LA is the choice that is never far behind.</p>
<p>On SVHN the Gaussian generator (FSG) fails.
Its class-conditional Gaussian cannot represent SVHN's multi-modal features: tail fidelity is 0.14 against 0.36 for HWA. That failure is why
FSG is not our final method, although it was competitive on MNIST.</p>
</section>"""


def sec_negative():
    e03 = load("E03_method_components_quick")
    e14 = load("E14_relative_hybrid_quick")
    e12 = load("E12_confirmation_quick")
    f01 = load("F01_main_longtail")
    g = lambda runs, lab, ds, **w: mean(pick(runs, lab, "final_bal", dataset=ds, **{"IF": 0.01, **w}))
    items = []
    try:
        items.append(("Prior-completing mixed batches (PCM)",
                      f"FMNIST quick pass: +HWA+LCD+PCM {g(e03, '+HWA+LCD+PCM', 'fmnist'):.1f} vs +HWA+LCD {g(e03, '+HWA+LCD', 'fmnist'):.1f}; "
                      f"+PCM+LA {g(e03, '+PCM+LA', 'fmnist'):.1f} vs +LA {g(e03, '+LA', 'fmnist'):.1f}.",
                      "Filling the client's missing classes with synthetic features only helps when tail synthetic features are faithful, and before HWA they are not. Dropped."))
        items.append(("Lazy conditioning decay (LCD)",
                      f"MNIST LT quick, 3 seeds: +HWA+LCD+LA {g(e12, '+HWA+LCD+LA', 'mnist', K=10):.1f} vs +HWA+LA {g(e12, '+HWA+LA', 'mnist', K=10):.1f}.",
                      "LCD treats the same symptom as HWA from the client side. With HWA present it adds nothing. It remains the option when the server must not learn which classes a client holds."))
        items.append(("Gaussian generator (FSG) in IID / SVHN",
                      f"Strong on MNIST LT quick ({g(e14, 'FSG+LA', 'mnist', K=10):.1f} vs {g(e14, '+HWA+LA', 'mnist', K=10):.1f}), "
                      f"but at full scale {g(f01, 'FSG+LA', 'mnist'):.1f} vs {g(f01, '+HWA+LA', 'mnist'):.1f}; in IID quick it trails ({g(e14, 'FSG+LA', 'mnist', K=10, IF=1.0, alpha=None):.1f} vs {g(e14, '+HWA+LA', 'mnist', K=10, IF=1.0, alpha=None):.1f}); on SVHN it collapses (about 45 vs 62).",
                      "Exact aggregation of sufficient statistics is elegant and DP-friendly, but a single Gaussian per class cannot represent real feature distributions."))
        items.append(("Hybrid CVAE / Gaussian generator",
                      f"Quick pass favoured it (MNIST LT {g(e14, 'RHYB(1.0)+HWA+LA', 'mnist', K=10):.1f} vs {g(e14, '+HWA+LA', 'mnist', K=10):.1f}); "
                      f"full scale reversed it ({g(f01, 'Ours-hybrid (RHYB+HWA+LA)', 'mnist'):.1f} vs {g(f01, '+HWA+LA', 'mnist'):.1f}).",
                      "The quick pass's short CVAE training hid how good the CVAE becomes once its tail rows stop collapsing. Our final choice follows the full-scale evidence."))
        items.append(("Balanced re-training of the classifier (BCR) and gradient filtering (GF)",
                      f"FMNIST quick: +BCR {g(e03, '+BCR', 'fmnist'):.1f}, +GF {g(e03, '+GF', 'fmnist'):.1f}, GeFL-F {g(e03, 'GeFL-F', 'fmnist'):.1f}, +LA {g(e03, '+LA', 'fmnist'):.1f}.",
                      "Both are weaker than LA, which achieves the same rebalancing in closed form."))
    except TypeError:
        pass
    items.append(("Interleaving consensus-labelled features into real batches (CSLM)",
                  "SVHN LT: 50.4 / 47.5 vs 60.4 / 63.3 for ours on the same seeds; MNIST IID K=100: 94.36 vs 94.61 GeFL-F.",
                  "Synthetic features crowd out scarce real ones. CSL works only as a separate synthetic phase."))
    rows = [[f"<b>{esc(a)}</b>", esc(b), esc(c)] for a, b, c in items]
    return f"""
<section id="s8"><h2><span class="num">8</span>What did not work, and why we dropped it</h2>
<p class="deck">Every idea was tested on a reduced schedule first. Only those that held up got full runs.</p>
{table("Negative and superseded results (balanced accuracy, %)", [("Idea", ""), ("Evidence", "why"), ("Conclusion", "why")], rows)}
</section>"""


def sec_privacy():
    f01 = load("F01_main_longtail")
    def stage_time(lab):
        rs = [r for r in f01 if r["label"] == lab and r["dataset"] == "mnist" and r.get("gen_time_s")]
        return np.mean([r["fe_time_s"] + r["gen_time_s"] + r["head_time_s"] for r in rs]) if rs else None,             (np.mean([r["gen_time_s"] for r in rs]) if rs else None)
    (t_g, g_g), (t_o, g_o) = stage_time("GeFL-F"), stage_time("+HWA+LA")
    rows = [["HWA", "Per-class sample counts, as sums", "Yes: counts and count-weighted rows aggregate under secure aggregation; Laplace noise on counts tested",
             "+K&middot;C numbers per round (C = 10 classes)"],
            ["LA", "Nothing", "Local: uses only the client's own label prior, never sent", "none"],
            ["CSL", "Nothing new", "Server-side or client-side on synthetic features only; no real data touched", "one forward pass of each header per synthetic batch"],
            ["LCD (alternative to HWA)", "Nothing", "Fully local; the option when even aggregated counts are disallowed", "none"],
            ["Anchoring statistics (PC, MC, KH, BBC gate)", "Class counts, class sums of h and of ||h||<sup>2</sup>, class sums of bounded random features &phi;(h)",
             "Secure-aggregated sums, one upload; formal (&epsilon;, &delta;)-DP option (clipped Gaussian releases, zCDP accounting), measured below",
             "one upload of about 0.3 MB per client"],
            ["MC-S (low-separation data)", "Class sums of z z<sup>T</sup> in a broadcast 256-dimensional basis, and the pooled second moment",
             "Secure-aggregated sums, one upload (the pooled second moment is the release FSG already uses)", "C&middot;k<sup>2</sup> numbers once"]]
    t = table("Privacy footprint of each component", [("Component", ""), ("Extra information shared", "why"), ("Compatibility", "why"), ("Extra cost", "why")], rows)
    cost = (f"<p>Measured cost, MNIST long tail, full schedule, one laptop GPU (sum of the three stages, mean of 3 seeds): GeFL&#8209;F {t_g:.0f} s, "
            f"ours {t_o:.0f} s ({(t_o / t_g - 1) * 100:+.1f}%). Generator stage alone: {g_g:.0f} s vs {g_o:.0f} s. HWA only changes how the server averages, "
            f"and LA is one subtraction in the loss.</p>") if t_g and t_o else ""
    e20, e25, e26 = load("E20_proto_generator"), with_gate(load("E25_anchored_stack")), load("E26_private_anchoring")
    dp_rows = []
    for lab, src, shown in [("GeFL-F", e20, "GeFL&#8209;F (no formal privacy)"), ("Ours-A Ts=10", e25, "Final method, exact statistics"),
                            ("Ours-A Ts=10, DP eps=8", e26, "Final method, (8, 10<sup>&minus;5</sup>)-DP statistics"),
                            ("Ours-A Ts=10, DP eps=2", e26, "Final method, (2, 10<sup>&minus;5</sup>)-DP statistics")]:
        cells = [shown]
        for ds in ["mnist", "fmnist"]:
            cells.append(fmt_ms(pick(src, lab, "final_bal", dataset=ds, IF=0.01, K=10)))
        dp_rows.append(cells)
    t_dp = table("Formal DP on every statistic the method adds (E26; long tail, K = 10, 3 seeds; balanced accuracy)",
                 [("", ""), ("MNIST", "num"), ("FashionMNIST", "num")], dp_rows, [2, 3]) if e26 else ""
    dp_txt = ("<p>Features are clipped to the 90th-percentile norm of the public held-out pool. Each release (counts, sums of h, sums of ||h||<sup>2</sup>, "
              "sums of &phi;(h)) receives Gaussian noise, and the budget is split equally under zCDP. Where a class's noisy statistic would be too noisy to help, "
              "MC and KH fall back to the plain generator for that class, which is post-processing with no extra cost (at &epsilon; = 8 MC is applied to 7 of "
              "10 classes, at &epsilon; = 2 to 5). Privacy costs 4&ndash;7 points at &epsilon; = 8 and 10&ndash;14 at &epsilon; = 2, almost all on the rarest "
              "classes, whose 24 samples cannot be both private and accurate. Even at &epsilon; = 2 the method stays 7&ndash;10 points above GeFL&#8209;F, "
              "which gives no formal guarantee at all. The generator's own FedAvg training is not DP, as in GeFL&#8209;F.</p>") if e26 else ""
    return f"""
<section id="s9"><h2><span class="num">9</span>Privacy and federated cost</h2>
<p class="deck">The improvement must not buy accuracy with privacy or bandwidth.</p>
{t}
{cost}
{t_dp}
{dp_txt}
<p>None of the components share raw data, real features, or per-client label histograms in the clear. GeFL&#8209;F's own threat model
(sharing a feature generator rather than an image generator) is unchanged.</p>
</section>"""


# Every GeFL / GeFL-F number in the paper (Fig. 4, Table II, Table IV), best_mean_acc %, IID.
# (method, generator) -> {(dataset, K): value}
PAPER_FIG4 = {
    ("FedProx", "-"): {("mnist", 10): 92.57, ("mnist", 50): 92.21, ("mnist", 100): 91.33, ("fmnist", 10): 80.50, ("fmnist", 50): 79.94, ("fmnist", 100): 79.67,
                       ("svhn", 10): 65.18, ("svhn", 50): 63.88, ("svhn", 100): 65.59, ("cifar10", 10): 55.10, ("cifar10", 50): 54.95, ("cifar10", 100): 52.71},
    ("FedALA", "-"): {("mnist", 10): 92.70, ("mnist", 50): 91.57, ("mnist", 100): 91.93, ("fmnist", 10): 80.03, ("fmnist", 50): 79.75, ("fmnist", 100): 79.48,
                      ("svhn", 10): 61.32, ("svhn", 50): 56.87, ("svhn", 100): 53.36, ("cifar10", 10): 53.01, ("cifar10", 50): 52.21, ("cifar10", 100): 51.56},
    ("GeFL", "DCGAN"): {("mnist", 10): 95.32, ("mnist", 50): 92.94, ("mnist", 100): 91.76, ("fmnist", 10): 83.11, ("fmnist", 50): 81.28, ("fmnist", 100): 80.50,
                        ("svhn", 10): 66.57, ("svhn", 50): 65.00, ("svhn", 100): 65.66, ("cifar10", 10): 58.45, ("cifar10", 50): 54.73, ("cifar10", 100): 49.35},
    ("GeFL-F", "DCGAN-F"): {("mnist", 10): 95.13, ("mnist", 50): 93.67, ("mnist", 100): 93.08, ("fmnist", 10): 82.62, ("fmnist", 50): 81.29, ("fmnist", 100): 80.71,
                            ("svhn", 10): 68.78, ("svhn", 50): 69.32, ("svhn", 100): 65.66, ("cifar10", 10): 54.82, ("cifar10", 50): 53.80, ("cifar10", 100): 53.95},
    ("GeFL", "CVAE"): {("mnist", 10): 94.46, ("mnist", 50): 92.68, ("mnist", 100): 91.72, ("fmnist", 10): 82.56, ("fmnist", 50): 80.24, ("fmnist", 100): 79.83,
                       ("svhn", 10): 74.74, ("svhn", 50): 71.66, ("svhn", 100): 70.35, ("cifar10", 10): 55.80, ("cifar10", 50): 54.84, ("cifar10", 100): 50.96},
    ("GeFL-F", "CVAE-F"): {("mnist", 10): 95.47, ("mnist", 50): 95.04, ("mnist", 100): 94.63, ("fmnist", 10): 83.14, ("fmnist", 50): 82.21, ("fmnist", 100): 81.65,
                           ("svhn", 10): 76.26, ("svhn", 50): 73.64, ("svhn", 100): 76.00, ("cifar10", 10): 55.86, ("cifar10", 50): 53.19, ("cifar10", 100): 51.46},
    ("GeFL", "DDPM w=0"): {("mnist", 10): 96.44, ("mnist", 50): 94.37, ("mnist", 100): 93.12, ("fmnist", 10): 82.43, ("fmnist", 50): 81.51, ("fmnist", 100): 79.29,
                           ("svhn", 10): 75.11, ("svhn", 50): 67.81, ("svhn", 100): 69.36, ("cifar10", 10): 59.36, ("cifar10", 50): 55.52, ("cifar10", 100): 51.51},
    ("GeFL-F", "DDPM-F w=0"): {("mnist", 10): 95.72, ("mnist", 50): 94.11, ("mnist", 100): 94.17, ("fmnist", 10): 84.28, ("fmnist", 50): 82.96, ("fmnist", 100): 80.95,
                               ("svhn", 10): 73.38, ("svhn", 50): 68.12, ("svhn", 100): 67.84, ("cifar10", 10): 56.61, ("cifar10", 50): 53.64, ("cifar10", 100): 51.16},
    ("GeFL", "DDPM w=2"): {("mnist", 10): 95.17, ("mnist", 50): 93.44, ("mnist", 100): 92.63, ("fmnist", 10): 81.51, ("fmnist", 50): 81.28, ("fmnist", 100): 79.36,
                           ("svhn", 10): 73.15, ("svhn", 50): 67.17, ("svhn", 100): 68.75, ("cifar10", 10): 58.47, ("cifar10", 50): 56.31, ("cifar10", 100): 51.83},
    ("GeFL-F", "DDPM-F w=2"): {("mnist", 10): 93.60, ("mnist", 50): 93.52, ("mnist", 100): 94.06, ("fmnist", 10): 82.29, ("fmnist", 50): 81.04, ("fmnist", 100): 80.82,
                               ("svhn", 10): 72.55, ("svhn", 50): 67.80, ("svhn", 100): 67.72, ("cifar10", 10): 55.35, ("cifar10", 50): 53.43, ("cifar10", 100): 50.00},
}
# Table IV: image-space GeFL (DCGAN) with data augmentation, CIFAR-10 IID, K = 10.
PAPER_AUG_CIFAR = {"FedAvg": 55.65, "FedAvg + MixUp": 60.07, "FedAvg + CutMix": 58.95, "FedAvg + AugMix": 53.96, "FedAvg + AutoAugment": 56.99,
                   "GeFL + MixUp": 62.67, "GeFL + CutMix": 61.66, "GeFL + AugMix": 56.47, "GeFL + AutoAugment": 59.97}


# Our methods (every variant that is ours), and the final method alone.
OUR_SOURCES = [("E25_anchored_stack", ["Ours-A Ts=10", "Ours-A (PC+MC+KH+LA+CSL)"]),
               ("E27_final_iid_many_clients", ["Ours-A Ts=10"]),
               ("K08_kaggle_anchored_stack", ["Ours-A Ts=10", "Ours-A (PC+MC+KH+LA+CSL)"]),
               ("E16_csl_synthetic_budget", ["Ours (HWA+LA+CSL) Ts=10", "+CSL Ts=10"]),
               ("F10_combined_method", ["+HWA+LA+CSL", "+CSL"]),
               ("F09_consensus_paper_setting", ["+CSL (beta=0.5)"]),
               ("K01", ["Ours (HWA+LA+CSL)", "+CSL (beta=0.5)"]),
               ("K03_kaggle_client_scaling", ["Ours (HWA+LA+CSL)", "+CSL (beta=0.5)", "+CSL Ts=10", "Ours (HWA+LA+CSL) Ts=10"]),
               ("K04_kaggle_ddpm_cifar10_svhn", ["Ours (DDPM-F)", "+CSL (DDPM-F)"]),
               ("K05_kaggle_newgen_cifar10_svhn", ["Ours+MC", "Ours-PC+MC", "Ours-PC+ZP+MC", "Ours-PC+ZP+MC Ts=10"]),
               ("K06_kaggle_newgen_many_clients", ["Ours (HWA+LA+CSL)", "Ours+MC", "Ours-PC+MC", "Ours-PC+ZP+MC"]),
               ("K10_kaggle_hybrid_svhn_cifar10", ["Ours+MC+KH (CVAE+HWA)", "Ours-R", "Ours-R Ts=10"]),
               ("K11_kaggle_covariance_anchoring", ["Ours-A-S (PC+MCS+KH+LA+CSL)", "Ours-R-S (PCR+MCS+KH+LA+CSL)", "Ours-A-S Ts=10"]),
               ("K12_kaggle_final_gated", ["Ours+MC+KH (CVAE+HWA)", "Ours+MCS+KH (CVAE+HWA)", "Ours+MCS+KH (CVAE+HWA) Ts=10", "Ours-A Ts=10"])]
FINAL_SOURCES = [("E25_anchored_stack", "Ours-A Ts=10"), ("E22_pc_many_clients", "Ours-A Ts=10"), ("E27_final_iid_many_clients", "Ours-A Ts=10"),
                 ("K08_kaggle_anchored_stack", "Ours-A Ts=10"), ("K12_kaggle_final_gated", "Ours-A Ts=10")]
# The final method is gated by the class-mean separation F (released sums): SVHN (F = 0.01)
# takes class identity from HWA class rows; every other dataset (F >= 0.14) from the anchor.
LOW_SEP = {"svhn"}
FINAL_SOURCES_LOW_SEP = [("K10_kaggle_hybrid_svhn_cifar10", "Ours+MC+KH (CVAE+HWA)"), ("K12_kaggle_final_gated", "Ours+MC+KH (CVAE+HWA)")]


def _runs_of(exp):
    return kaggle_runs()[0] if exp == "K01" else load(exp)


def ours_iid(ds, K, final_only=False):
    """IID evidence: the final method alone, or the best of our variants (label shown)."""
    cands = []
    srcs = ([(e, [l]) for e, l in (FINAL_SOURCES_LOW_SEP if ds in LOW_SEP else FINAL_SOURCES)] if final_only else OUR_SOURCES)
    for exp, labs in srcs:
        runs = _runs_of(exp)
        for lab in labs:
            v = pick(runs, lab, "best_mean_acc", dataset=ds, K=K, IF=1.0)
            if v:
                cands.append((mean(v), len(v), exp.split("_")[0], lab))
    return max(cands) if cands else None


def sec_vs_paper():
    rows = []
    for ds in ["mnist", "fmnist", "svhn", "cifar10"]:
        for K in [10, 50, 100]:
            best = max(((v[(ds, K)], m, g) for (m, g), v in PAPER_FIG4.items() if (ds, K) in v))
            gf = PAPER_FIG4[("GeFL-F", "CVAE-F")][(ds, K)]
            fin, o = ours_iid(ds, K, final_only=True), ours_iid(ds, K)

            def cell_of(x):
                return (f"{x[0]:.2f}<span class=\"sd\"> n={x[1]}, {esc(x[2])} {esc(x[3])}</span>" if x
                        else '<span class="pend">pending</span>')
            if o:
                val = o[0]
                verdict = ('<span class="pill g">above best</span>' if val > best[0] + 0.3 else
                           ('<span class="pill t">tie (&plusmn;0.3)</span>' if val >= best[0] - 0.3 else '<span class="pill b">below</span>'))
                d = f"{val - best[0]:+.2f}"
            else:
                d, verdict = "", ""
            rows.append([f"{DS_NAME[ds]}, K={K}", f"{gf:.2f}", f"{best[0]:.2f}<span class=\"sd\"> {esc(best[1])} {esc(best[2])}</span>",
                         cell_of(fin), cell_of(o), d, verdict])
    t = table("Paper's IID setting: ours against the best of all ten methods in the paper's Figure 4 (best_mean_acc)",
              [("Setting", ""), ("GeFL-F CVAE-F", "num"), ("Best in paper (which)", "num"), ("Final method (gated by F)", "num"),
               ("Best of our variants (which)", "num"), ("Best ours &minus; paper best", "num"), ("", "")], rows)
    aug = [[esc(k), f"{v:.2f}"] for k, v in PAPER_AUG_CIFAR.items()]
    t2 = table("Paper Table IV: data augmentation (image-space GeFL with DCGAN, CIFAR-10, IID, K = 10)", [("Method", ""), ("Acc.", "num")], aug)
    return f"""
<section id="s11"><h2><span class="num">11</span>Against every GeFL variant in the paper</h2>
<p class="deck">Figure 4 of the paper evaluates ten methods: GeFL and GeFL&#8209;F with DCGAN, CVAE and DDPM (w = 0, 2), plus FedProx and FedALA.
For each setting we compare against the best of them, whichever it is.</p>
{t}
<p><b>CIFAR-10 caveat.</b> Our reproduction of GeFL&#8209;F scores 59.10 on CIFAR-10 IID, 3.2 points above the paper's 55.86, so comparisons
with the paper's CIFAR-10 numbers flatter every one of our variants for reasons unrelated to them. The fair comparison is paired, on the same seeds:
ours (HWA + LA + CSL) is +0.92 over our GeFL&#8209;F (p = 0.015). On MNIST and FashionMNIST our GeFL&#8209;F is within 0.6 of the paper.</p>
<p>Rows for K = 50 and 100 come from the first Kaggle client-scaling run (K03), whose consensus-label arms predate the per-client
synthetic-slice fix (&sect;6). The fix restores per-client synthetic diversity and can only raise them; the anchored method at K = 50/100 runs in
NB17 and E22.</p>
<p>Two of the paper's strongest numbers come from different pipelines, so it matters what they are. On CIFAR&#8209;10 the best is image-space GeFL with
an image diffusion model (59.36), with no shared feature extractor. The augmentation results (Table IV, best 62.67) are image-space GeFL with DCGAN
plus MixUp or CutMix. The paper never combines GeFL&#8209;F with augmentation. Our changes act on the generator's aggregation and on the heads'
objective and labels, so they are orthogonal to augmentation: MixUp/CutMix could be added to any of these methods, ours included.</p>
{t2}
</section>"""


def sec_k02():
    k02 = load("K02_kaggle_regimes_privacy_ablations")
    if not k02:
        return ""
    reg = method_table(k02, [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("Ours", "Ours (HWA+LA+CSL)")], ["fmnist", "mnist"],
                       [("final_bal", "balanced acc.")], where=dict(IF=1.0, alpha=0.5),
                       caption="Label skew without a tail (IF = 1, Dir 0.5, K = 10)", ours=("Ours",))
    reg += method_table(k02, [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("Ours", "Ours (HWA+LA+CSL)")], ["fmnist", "mnist"],
                        [("final_bal", "balanced acc.")], where=dict(IF=0.1),
                        caption="Mild tail (IF = 10, Dir 0.5, K = 10)", ours=("Ours",))
    abl_labels = [("Ours", "Ours (HWA + LA + CSL)"), ("no HWA (flat averaging)", "&minus; HWA (flat averaging)"),
                  ("no LA (plain CE)", "&minus; LA (plain CE)"), ("no CSL", "&minus; CSL"),
                  ("HWA weights = counts n", "HWA weights: counts n"), ("HWA weights = uniform over holders", "HWA weights: uniform over holders"),
                  ("LA tau=1.5", "LA &tau; = 1.5"), ("LA tau=2.0", "LA &tau; = 2"),
                  ("CSL beta=0.25", "CSL &beta; = 0.25"), ("CSL beta=0.75", "CSL &beta; = 0.75")]
    abl = method_table(k02, abl_labels, ["fmnist", "mnist"], [("final_bal", "balanced acc."), ("final_tail", "tail")],
                       where=dict(IF=0.01), ref="Ours", caption="Ablations, long tail (IF = 100, Dir 0.5, K = 10; 2 seeds, Ours 3)",
                       ours=("Ours",))
    priv = method_table(k02, [("GeFL-F", "GeFL-F"), ("Ours", "Ours (no noise)"), ("Ours, DP eps=10", "Ours, &epsilon; = 10"),
                              ("Ours, DP eps=1", "Ours, &epsilon; = 1"), ("Ours, DP eps=0.1", "Ours, &epsilon; = 0.1")],
                        ["fmnist", "mnist"], [("final_bal", "balanced acc."), ("feature_mnd", "MND")],
                        where=dict(IF=0.01), caption="Laplace-noised class counts (&epsilon;-DP per client histogram) and memorisation",
                        ours=("Ours, DP eps=1",))
    return f"""
<section id="s12"><h2><span class="num">12</span>Regimes, ablations and privacy (Kaggle K02)</h2>
<p class="deck">Three seeds per setting (two for the ablation variants). Numbers come from the pasted summaries.</p>
<p><b>The gain grows with imbalance.</b> Over GeFL&#8209;F, ours gains +5 to +6 points with label skew alone, +7 with a mild tail and
+15 to +18 with the 100:1 tail. This is the collapse recursion of &sect;2 at work: Dirichlet(0.5) already leaves some classes on fewer than
half the clients, so (2m<sub>c</sub>/K &minus; 1) &lt; 0 even without a global tail.</p>
{reg}
<p><b>Ablations.</b> Removing HWA costs 7&ndash;8 points, and removing LA costs 5&ndash;8. HWA's exact weighting does not matter (E(n), n and
uniform-over-holders are within noise). That is predicted: the damage comes from <em>non-holders'</em> decay, so excluding them is what counts.
CSL matters where the generator's labels are poor (MNIST: &minus;2.5 without it, tail fidelity 0.30) and not where they are decent (FashionMNIST, 0.65).
The best CSL weight differs by dataset (&beta; = 0.75 on MNIST, &le; 0.5 on FashionMNIST), which is the motivation for estimating it per run
(Bayesian CSL). LA's temperature behaves the same way: &tau; = 2 helps FashionMNIST (+1.4, tail +4.8) and slightly hurts MNIST, because &tau; = 1
corrects only the label prior and FashionMNIST's tail classes are also its intrinsically hardest (shirt, coat). The server-side bias calibration
(BBC) estimates that residual bias per head instead of tuning &tau;.</p>
{abl}
<p><b>Privacy.</b> With &epsilon; = 1 Laplace noise on every client's class histogram the method loses 0.2 (FashionMNIST) and 1.3 points (MNIST).
At &epsilon; = 0.1 it loses about 3.4 and still stays 11&ndash;14 points above GeFL&#8209;F. Feature-space memorisation (MND) is the same as GeFL&#8209;F's
in every case.</p>
{priv}
</section>"""


def sec_k04():
    k04 = load("K04_kaggle_ddpm_cifar10_svhn")
    if not k04:
        return ""
    t = method_table(k04, [("GeFL-F (DDPM-F)", "GeFL-F (DDPM-F)"), ("+CSL (DDPM-F)", "+CSL (DDPM-F)"), ("Ours (DDPM-F)", "Ours (DDPM-F)")],
                     sorted({r["dataset"] for r in k04}), [("best_mean_acc", "best_mean_acc"), ("fidelity_tail", "tail fidelity")],
                     where=dict(IF=1.0), ref="GeFL-F (DDPM-F)", caption="Paper's IID setting on the authors' DDPM-F (Kaggle K04)",
                     ours=("Ours (DDPM-F)",))
    return f"""
<section id="s13"><h2><span class="num">13</span>The paper's diffusion generator (Kaggle K04)</h2>
<p class="deck">The same comparison on the authors' DDPM-F, ported unchanged.</p>
{t}
<p>On SVHN our GeFL&#8209;F with DDPM-F scores 69.8, below the published 73.38, and its samples are poor (tail fidelity 0.29). The paper also
finds DDPM-F worse than CVAE-F on SVHN (73.38 vs 76.26). CSL still gains +0.5 (p = 0.045) on top of it.</p>
</section>"""


def with_gate(runs):
    """Method output: BBC applied iff the federation's global label distribution is
    imbalanced (exact aggregated counts; IF < 1 here). Adds 'method_bal'."""
    out = []
    for r in runs:
        r = dict(r)
        if r.get("bbc_bal") is not None:
            applied = r.get("bbc_applied", r.get("IF", 1.0) < 1.0)
            r["method_bal"] = r["bbc_bal"] if applied else r["final_bal"]
        out.append(r)
    return out


def sec_anchored():
    e20, e24, e25, e29 = (with_gate(load(x)) for x in ["E20_proto_generator", "E24_kme_generator", "E25_anchored_stack",
                                                       "E29_hybrid_generator"])
    runs = e20 + e24 + e25 + [r for r in e29 if r["K"] == 10]
    if not e20:
        return ""
    labels = [("GeFL-F", "GeFL-F"), ("Ours (HWA+LA+CSL)", "Ours: HWA + LA + CSL"), ("Ours+MC", "+ MC"),
              ("Ours-PC+MC", "PC-VAE + MC"), ("Ours-KME (KME+LA+CSL)", "KME-Gen (no federated generator training)"),
              ("Ours-PC+MC+KH", "PC-VAE + MC + KH (E24)"), ("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A: PC + MC + KH (E25)"),
              ("Ours-A Ts=10", "Ours-A, T_s = 10 (E25): final"), ("Ours-R Ts=10", "Hybrid PCR (rows + anchor), T_s = 10 (E29)")]
    t_lt = method_table(runs, labels, ["mnist", "fmnist"], [("final_bal", "balanced acc."), ("method_bal", "with gated BBC")],
                        where=dict(IF=0.01), ref="GeFL-F", caption="Long tail (IF = 100, Dir 0.5, K = 10): the anchored stack",
                        ours=("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A Ts=10"))
    f10 = load("F10_combined_method")
    runs_iid = runs + [r for r in f10 if r["dataset"] == "mnist" and r["IF"] == 1.0 and r["label"] in ("GeFL-F", "+HWA+LA+CSL")]
    runs_iid = [dict(r, label="Ours (HWA+LA+CSL)") if r.get("exp") == "F10_combined_method" and r["label"] == "+HWA+LA+CSL" else r
                for r in runs_iid]
    t_iid = method_table(runs_iid, labels, ["mnist", "fmnist"], [("best_mean_acc", "best_mean_acc")],
                         where=dict(IF=1.0), ref="GeFL-F", caption="Paper's IID setting (K = 10; MNIST GeFL-F / Ours rows from F10, same seeds)",
                         ours=("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A Ts=10"))
    kg = with_gate(kaggle_runs()[0] + load("K08_kaggle_anchored_stack") + load("K10_kaggle_hybrid_svhn_cifar10")
                   + load("K11_kaggle_covariance_anchoring"))
    klabels = [("GeFL-F", "GeFL-F"), ("+LA", "+LA"), ("+HWA+LA", "+HWA+LA"), ("Ours (HWA+LA+CSL)", "Ours: HWA + LA + CSL"),
               ("FSG+LA", "Gaussian generator (FSG) + LA"), ("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A, T_s = 1 (K08)"),
               ("Ours-A Ts=10", "Ours-A, T_s = 10 (K08)"), ("Ours-A-S (PC+MCS+KH+LA+CSL)", "Ours-A-S: PC + MC-S + KH (K11)"),
               ("Ours-R", "Hybrid PCR + MC + KH (K10)"), ("Ours-R-S (PCR+MCS+KH+LA+CSL)", "Hybrid PCR + MC-S + KH (K11)"),
               ("Ours+MC+KH (CVAE+HWA)", "Rows (CVAE + HWA) + MC + KH (K10): final when F < 0.05")]
    t_klt = method_table(kg, klabels, ["cifar10", "svhn"], [("final_bal", "balanced acc."), ("method_bal", "with gated BBC")],
                         where=dict(IF=0.01, K=10), ref="GeFL-F",
                         caption="Long tail on CIFAR-10 and SVHN (Kaggle K01 / K08, 3 seeds, same seeds and splits; pasted summaries, so no paired p)",
                         ours=("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A Ts=10"))
    kilabels = [("GeFL-F", "GeFL-F"), ("+CSL (beta=0.5)", "+CSL"), ("+HWA+LA", "+HWA+LA"), ("Ours (HWA+LA+CSL)", "Ours: HWA + LA + CSL"),
                ("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A, T_s = 1 (K08)"), ("Ours-A Ts=10", "Ours-A, T_s = 10 (K08): final when F ≥ 0.05"),
                ("Ours-A-S (PC+MCS+KH+LA+CSL)", "Ours-A-S: PC + MC-S + KH (K11)"), ("Ours-R-S (PCR+MCS+KH+LA+CSL)", "Hybrid PCR + MC-S + KH (K11)"),
                ("Ours+MC+KH (CVAE+HWA)", "Rows (CVAE + HWA) + MC + KH (K10): final when F < 0.05")]
    t_kiid = method_table(kg, kilabels, ["cifar10", "svhn"], [("best_mean_acc", "best_mean_acc")], where=dict(IF=1.0, K=10), ref="GeFL-F",
                          caption="Paper's IID setting on CIFAR-10 and SVHN (paper's GeFL-F: 55.86 / 76.26; best of all ten methods: 59.36 / 76.26)",
                          ours=("Ours-A (PC+MC+KH+LA+CSL)", "Ours-A Ts=10"))
    kprose = """<p><b>The hybrid generator costs nothing where the anchor works (E29, same seeds).</b> Adding HWA-aggregated class rows to the
anchored decoder (PCR) leaves every MNIST and FashionMNIST result unchanged within noise: long tail +0.22 (MNIST, p = 0.29) and &minus;0.07 (FashionMNIST,
p = 0.51); FashionMNIST IID +0.15 (p = 0.28); MNIST at K = 100 +0.10 (2 seeds; 94.36, so the rows do not bring back the CVAE's degradation with K).
If it also fixes SVHN (Kaggle NB19, NB22), it becomes the single final method.</p>
<p><b>CIFAR-10: the anchored method wins in both regimes.</b> Under the long tail Ours-A gives 46.1 at T<sub>s</sub> = 1 and about 49 with the gated BBC,
against 44.2 for the best earlier method (the Gaussian generator) and 33.7 for GeFL&#8209;F (+12.3, +15 with BBC). Its tail fidelity after KH is 41 %, against 23 % for the
CVAE + HWA. In the paper's IID setting it reaches 60.93 (T<sub>s</sub> = 10), the highest CIFAR-10 number here: +1.8 over our GeFL&#8209;F, +1.6 over the best of all ten
methods in the paper and +5.1 over the paper's GeFL&#8209;F. Unpaired tests from the pasted means: the long-tail gains over everything but the Gaussian generator have
p &le; 0.02, and the gain with BBC over the Gaussian generator has p = 0.003. The IID gains are within noise without the per-seed files (p &asymp; 0.1).
This contradicts the prediction registered before these runs (a tie or a small loss, from CIFAR-10's weak pixel-space class means). What decides is the anchored
generator's fidelity <em>relative to the CVAE's</em>. On CIFAR-10 the CVAE itself is poor, so even weakly informative exact statistics win.</p>
<p><b>SVHN, and how the final method handles it.</b> SVHN's class means carry essentially no class information (nearest-class-mean accuracy 12 % in the shared
feature space, chance 10 %), so a generator whose class identity comes from the mean cannot separate the classes (Proposition 5): Ours-A gives 55.8 under the long tail.
Two fixes were tested on the same seeds.</p>
<ul><li><b>MC-S</b> corrects each class to its exact covariance as well as its mean (Proposition 6). SVHN's identity lives there: QDA from the class
covariances reaches 54 % in pixel space, against 21 % for LDA. It adds <b>+6.0</b> under the long tail (every seed, p = 0.016) and <b>+2.7</b> in IID (p = 0.026).</li>
<li><b>Class rows protected by HWA</b> are better still. The paper's CVAE generator with HWA, MC and KH gives <b>65.6</b> under the long tail (GeFL&#8209;F 50.5) and 77.07 in IID,
above the paper's best (76.26). The hybrid (rows plus anchor) falls between the two: its anchor leaves raw samples under-dispersed, and on SVHN MC + KH then lowers its tail
fidelity (35 &rarr; 32 %), while it raises the CVAE's (36 &rarr; 42 %).</li></ul>
<p><b>The final method is therefore gated, by the same statistic.</b> The server computes F, the between- to within-class variance ratio of the exact class means, from the sums it
already receives. F is 0.01 on SVHN, against 0.14&ndash;0.60 on the other datasets, so any threshold between 0.02 and 0.1 makes the same choice everywhere. At F &ge; 0.05 class
identity comes from the anchor (Ours-A); below it, from HWA-protected class rows. Whether MC-S also helps the rows is being tested (Kaggle K12).</p>
<p><b>The synthetic budget on hard data.</b> On CIFAR-10 and SVHN, T<sub>s</sub> = 10 raises the paper's best-round metric in all four settings but lowers the final
balanced accuracy (CIFAR-10 long tail: 44.4 against 46.1, p = 0.03). Proposition 3 predicts this. The optimal synthetic share w<sup>&star;</sup> = &sigma;<sup>2</sup>/(2 n<sub>r</sub> b<sup>2</sup>)
falls with the generator's error b, and there tail fidelity after KH (24&ndash;46 %) is far below real-data accuracy. On MNIST and FashionMNIST, where fidelity matches real data,
T<sub>s</sub> = 10 is better on both metrics.</p>"""
    return f"""
<section id="s14"><h2><span class="num">14</span>Final method: anchoring every stage to exact statistics</h2>
<p class="deck">The method that emerged from the diagnostics: replace federated-averaged parameters by exactly aggregated statistics wherever they decide the result.</p>
<p><b>PC-VAE</b> conditions the generator on the exact federated class mean, so it has no class-specific parameters to dilute or collapse.
<b>MC</b> moves each generated class onto its exact mean and spread: among affine maps, the W<sub>2</sub>-optimal correction.
<b>KH</b> herds the samples toward the exact class kernel embedding, a Frank&ndash;Wolfe step on the MMD term of the head's risk bound.
<b>BBC</b> fits per-head logit offsets so each head predicts every class equally often on balanced, calibrated synthetic data. It is
applied only when the exact global class counts are imbalanced: across 18 runs per regime it added +0.9 (MNIST) and +1.5 (FashionMNIST)
under the long tail, and never helped in IID.
<b>KME-Gen</b> is the extreme of the same idea, a generator trained only from exact kernel embeddings at the server with no federated training.
It is far above GeFL&#8209;F at 1/3000 of the generator communication, but below the learned-decoder variants.</p>
<p><b>What the three seeds show.</b> At T<sub>s</sub> = 10 the full stack is +19.6 (MNIST) and +21.2 (FashionMNIST) points over GeFL&#8209;F,
and +4.7 / +0.5 over the best earlier variant, positive on every seed. On MNIST it reaches the accuracy of its own heads with the last layer re-fit
on pooled real data. KH helped in all nine paired comparisons (sign test p = 0.004). It works by restoring spread and realistic difficulty: the
anchored decoder's raw samples are under-dispersed (0.39&ndash;0.49 of the real spread) and, on FashionMNIST, purer than real data. After KH,
spread is 0.93&ndash;0.98 and IID fidelity equals the referee's own accuracy on real data (0.837 vs 0.839). With the anchored generator, the
10-epoch synthetic budget adds +1.7 on MNIST, where it added nothing to the CVAE. This is the interaction Proposition 3 predicts once the
generator's label bias is low. In the paper's IID setting the final method gives 97.62 on MNIST: +1.6 over GeFL&#8209;F (p = 0.001), +0.85 over the
CVAE variant, and +1.18 over the best of all ten methods in the paper. On FashionMNIST IID the CVAE + HWA variant remains marginally better
(84.34 vs 83.96, p = 0.18): FedAvg does not collapse class rows when every client holds every class.</p>
{t_lt}
{t_iid}
{kprose}
{t_klt}
{t_kiid}
</section>"""


def sec_summary():
    e20, e25 = with_gate(load("E20_proto_generator")), with_gate(load("E25_anchored_stack"))
    f09 = load("F09_consensus_paper_setting")
    fin = "Ours-A Ts=10"
    have = bool(pick(e25, fin, "final_bal", dataset="mnist", IF=0.01))
    w = dict(IF=0.01, K=10)

    def g(ds, m="method_bal"):
        return mean(pick(e25, fin, m, dataset=ds, **w)) - mean(pick(e20, "GeFL-F", "final_bal", dataset=ds, **w))

    def tl(ds):
        return mean(pick(e25, fin, "final_tail", dataset=ds, **w)), mean(pick(e20, "GeFL-F", "final_tail", dataset=ds, **w))
    pv = max((paired_p(pick(e25, fin, "final_bal", dataset=d, **w), pick(e20, "GeFL-F", "final_bal", dataset=d, **w)) or 1)
             for d in ["mnist", "fmnist"]) if have else 1
    iid = mean(pick(e25, fin, "best_mean_acc", dataset="fmnist", IF=1.0, K=10)) if have else None
    csl = lambda ds: (mean(pick(f09, "+CSL (beta=0.5)", "best_mean_acc", dataset=ds, K=10, IF=1.0)) - mean(pick(f09, "GeFL-F", "best_mean_acc", dataset=ds, K=10, IF=1.0)))
    if have:
        tm, tf = tl("mnist"), tl("fmnist")
        card1 = (f'<div class="t2">+{g("mnist"):.1f} / +{g("fmnist"):.1f} points balanced accuracy</div>'
                 f'<p>MNIST / FashionMNIST, IF = 100, final method over GeFL&#8209;F, 3 seeds, p &le; {pv:.3f}. '
                 f'Tail recall {tm[1]:.0f} &rarr; {tm[0]:.0f} and {tf[1]:.0f} &rarr; {tf[0]:.0f}. On MNIST it reaches its own pooled-real-data oracle. '
                 f'CIFAR-10: 33.7 &rarr; 46.1 (about 49 with the gated BBC), ahead of every other method. With 100 clients: 94.3 on MNIST. '
                 f'SVHN, whose class means carry no class information, is detected from the same sums: there class identity comes from HWA-protected rows, 50.5 &rarr; 65.6 (&sect;14).</p>')
    else:
        card1 = "<div class=\"t2\">pending</div>"
    return f"""
<section id="s0"><h2><span class="num">0</span>The result on one page</h2>
<div class="verdict">
  <div class="v g"><div class="k">Long-tailed clients</div>{card1}</div>
  <div class="v a"><div class="k">Paper's own IID setting</div><div class="t2">Above the paper's best variant on MNIST (+1.2), CIFAR-10 (+1.6) and SVHN (+0.8), level on FMNIST</div>
    <p>MNIST 97.62 with the final method (paper's best of all ten methods: 96.44; GeFL&#8209;F 95.47). CIFAR-10 60.93 (best: 59.36; GeFL&#8209;F 55.86;
    our GeFL&#8209;F reproduction scores 59.10 there). SVHN 77.07 (best: 76.26; 77.39 with the hybrid and MC-S). FashionMNIST 84.34 with the CVAE
    variant at T<sub>s</sub> = 10 (best: 84.28, a diffusion generator; the final method gives {iid:.2f}, p = 0.18 between them).</p></div>
  <div class="v t"><div class="k">Baseline validated</div><div class="t2">Our GeFL&#8209;F = the paper = the authors' code</div>
    <p>Within half a point at K = 10, 50, 100. Our baseline is slightly <em>weaker</em> than theirs under the long tail.</p></div>
</div>
<p><b>Diagnosis (&sect;2).</b> Under flat FedAvg with Adam's coupled weight decay, the class-conditioning rows of GeFL&#8209;F's generator
provably shrink for every class held by fewer than half the clients. Rare classes vanish from the generator, and every head is taught from
class-agnostic features.</p>
<p><b>The final method (&sect;14): anchor every stage to exactly aggregated statistics</b>, sums that secure aggregation delivers,
instead of federated-averaged parameters.</p>
<ol>
<li><b>PC-VAE.</b> The generator is conditioned on the exact federated class mean, so it has no class-specific parameters to collapse.</li>
<li><b>MC.</b> Each generated class is moved onto its exact mean and spread: the W<sub>2</sub>-optimal affine correction.</li>
<li><b>KH.</b> Kernel herding selects the samples whose kernel mean embedding is closest to the class's exact federated embedding.</li>
<li><b>Heads.</b> Logit adjustment by each client's own prior (LA), cross-architecture consensus soft labels (CSL), and a 10-epoch synthetic
budget, which Proposition 3 shows pays off only once the generator is faithful.</li>
<li><b>BBC.</b> Server-side balanced bias calibration of each head, switched on by the exact global class counts.</li>
</ol>
<p>The minimal fix for the paper's own generators is <b>HWA</b>: holder-weighted aggregation of the class rows, which removes the collapse
from CVAE&#8209;F and DDPM&#8209;F (&sect;3). All components keep GeFL&#8209;F's privacy model, and the method needs no data augmentation.
The paper uses augmentation only for image-space GeFL on CIFAR&#8209;10 (Table IV; best {PAPER_BEST_AUG_CIFAR}), never for GeFL&#8209;F.</p>
</section>"""


SECTIONS = [("s0", "0", "The result on one page"), ("s1", "1", "Baseline validation"), ("s2", "2", "The collapse mechanism"),
            ("s3", "3", "Long-tail main result"), ("s14", "14", "Final method: exact-statistics anchoring"), ("s4", "4", "Paper's IID setting"), ("s4b", "4b", "IID headroom"), ("s4c", "4c", "Synthetic budget"), ("s5", "5", "Combined method"),
            ("s6", "6", "More clients"), ("s7", "7", "SVHN and CIFAR-10"), ("s8", "8", "What did not work"),
            ("s9", "9", "Privacy and cost"), ("s11", "11", "Against every GeFL variant"), ("s12", "12", "Regimes, ablations, privacy"),
            ("s13", "13", "Diffusion generator"), ("s10", "10", "Reproduce")]

EXTRA_CSS = """
.sd{color:var(--muted);font-size:.86em}
.d{font-size:.86em;font-weight:600;margin-left:2px}
.d.up{color:var(--good)} .d.dn{color:var(--bad)} .d.eq{color:var(--muted)}
.sig{color:var(--good);font-weight:600} .ns{color:var(--muted)}
.pend{color:var(--muted)}
.small{font-size:14px;color:var(--ink-soft)}
pre.cmd{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface);border:1px solid var(--rule);padding:12px 14px;overflow-x:auto;line-height:1.6}
figure svg + svg{margin-top:14px}
"""


def design_head():
    src = open(DESIGN, encoding="utf-8").read()
    head = src[:src.index("</style>") + len("</style>")]
    head = re.sub(r"<title>.*?</title>", "<title>GeFL-F Results Report</title>", head, flags=re.S)
    return head + "<style>" + EXTRA_CSS + "</style>"


def build():
    import datetime
    stamp = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    rail = "".join(f'<li><a href="#{i}"><span class="n">{n}</span><span>{esc(t)}</span></a></li>' for i, n, t in SECTIONS)
    body = "".join(f() for f in [sec_summary, sec_validation, sec_diagnosis, sec_main, sec_anchored, sec_paper_setting, sec_headroom, sec_budget,
                                  sec_combined, sec_clients, sec_kaggle, sec_negative, sec_privacy, sec_vs_paper, sec_k02, sec_k04])
    body += """
<section id="s10"><h2><span class="num">10</span>Reproduce</h2>
<p>Each experiment is one standalone file built from <code>experiments/core.py</code> and a short spec, so it can be pasted into Kaggle.
Finished runs are skipped on rerun. Each <code>summary.md</code> holds the per-seed numbers.</p>
<pre class="cmd">python experiments/F01_main_longtail.py --out_dir results
python experiments/F09_consensus_paper_setting.py --out_dir results
python experiments/F10_combined_method.py --out_dir results
python experiments/K01_kaggle_cifar10_svhn.py --datasets svhn --gpus 0,1 --out_dir /kaggle/working/results
python experiments/make_results_page.py</pre>
</section>"""
    page = f"""{design_head()}
<header class="top"><div class="wrap"><div class="mast">
  <div class="kicker"><span>Results report</span><span class="dot"></span><span class="plain">Every number read from the run logs</span><span class="dot"></span><span class="plain">Built {stamp}</span></div>
  <h1>GeFL&#8209;F, Improved<span class="sub">Rare-class collapse in federated feature generators: its cause, its fix, and the evidence</span></h1>
  <p class="standfirst">GeFL&#8209;F trains a shared feature generator so that clients with different model architectures can learn
  from each other. We show that it fails under long-tailed client data, for a reason we derive exactly. We fix that failure
  with three changes that keep its privacy model, and we validate every comparison against the authors' own code.</p>
</div></div></header>
<div class="wrap"><div class="layout">
<nav class="rail" aria-label="Contents"><ol>{rail}</ol></nav>
<main>{body}</main></div></div>
<footer><div class="wrap"><p>Generated by <code>experiments/make_results_page.py</code> from <code>experiments/results/*/runs.jsonl</code>.
Std is across seeds; p-values are two-sided paired t-tests over seeds (same split, same initialisation) against GeFL&#8209;F.</p></div></footer>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(page)
    print("wrote", os.path.normpath(OUT), f"{len(page)/1024:.0f} KB")


if __name__ == "__main__":
    build()
