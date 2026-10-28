"""Fills the paper's result slots from the returned notebook results (docs/papers/GUIDE.md).

    python docs/papers/fill.py              # tables and numbers from fedecs_lt/results/*/runs.jsonl (seconds)
    python docs/papers/fill.py --figures    # also the figures, from the NAME__results.zip files in
                                            # fedecs_lt/results/_downloads (fetch_kaggle.py --results); minutes
    python docs/papers/fill.py --check-highlights

It reads every run that came back (every folder of fedecs_lt/results, except NB00, whose runs are an earlier code
version), sorts the runs into the paper's experiment groups by method and dataset (so parts, reruns and the order of
the notebooks do not matter), and writes
  generated/numbers.tex      \\defnum{key}{value} for every number slot (\\R{key} in the text)
  generated/tables/*.tex     every table slot (\\slottab{name}{...})
  generated/figures/*.pdf    every figure slot (\\slotfig{name}{...}; with --figures), and its caption as .txt
  generated/STATUS.md        what is filled, what is partial (fewer than three seeds, methods missing) and which
                             notebooks have not come back yet
Values are computed by the same functions as the notebooks' reports (fedecs_lt/paper_tables.py, report.fmt).
"""
import glob
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FL = os.path.join(ROOT, "fedecs_lt")
RESULTS = os.path.join(FL, "results")
GEN = os.path.join(HERE, "generated")
sys.path.insert(0, FL)
sys.path.insert(0, os.path.join(FL, "src"))

import build_notebooks as bn  # noqa: E402
import captions as cap  # noqa: E402
import paper_tables as pt  # noqa: E402
import plots  # noqa: E402
import report  # noqa: E402
from settings import DATASETS  # noqa: E402

OURS = "FedECS-LT"
DIAG = "FedECS-LT (200 extractor rounds)"
# number slots per method: (short name in the key, runs.jsonl key)
METRICS = [("bal", "final_bal"), ("er_bal", "er_bal"), ("f1", "final_f1"), ("tail", "final_tail"),
           ("worst", "final_worst"), ("mcc", "final_mcc"), ("gmean", "final_gmean"), ("ece", "final_ece"),
           ("worst_arch", "final_worst_arch"), ("er_tail", "er_tail")]
# tables across settings: (slot suffix, runs.jsonl key)
ACROSS = [("bal", "final_bal"), ("er", "er_bal"), ("f1", "final_f1"), ("tail", "final_tail"),
          ("worst", "final_worst"), ("ece", "final_ece"), ("worst_arch", "final_worst_arch")]
LOWER = {"final_ece"}


# ---------------------------------------------------------------- experiment groups
def main_setting(r):
    """10 clients, Dirichlet 0.5, imbalance ratio 100 (or the dataset's own imbalance), all clients every round."""
    natural = DATASETS.get(r["dataset"], {}).get("imbalance") == "natural"
    return (r["K"] == 10 and r["alpha"] == 0.5 and (r["IF"] is None if natural else r["IF"] == 0.01)
            and r.get("pfrac", 1.0) == 1.0 and not r.get("tail_perm"))


def group_of(r):
    """The paper's experiment a run belongs to: main (long-tailed benchmarks, different models), nat (natural
    imbalance), t2 (one shared model), abl (variants of GeFL-F), comp (components of FedECS-LT), sweep (robustness:
    one setting changed), or None."""
    lab, ds = r["label"], r["dataset"]
    if DATASETS.get(ds, {}).get("track") == 2:
        return "t2"
    if not main_setting(r):
        return "sweep" if lab in bn.SWEEP_METHODS else None
    if lab in bn.ABLATIONS or lab == "FedDF + LA":
        return "abl"
    if lab.startswith("FedECS-LT w"):
        return "comp"
    if lab in bn.TRACK1_MAIN:
        return "main" if ds in bn.TIER1 else "nat" if ds in bn.TIER2 else None
    return None


def correct_fedgrab(r):
    """Fed-GraB runs recorded before the evaluation fix of 28 Oct 2026 scored the global model's untrained head
    instead of Fed-GraB's trained classifier (fedecs_lt/src/track2.py). The library's own per-round accuracy of the
    trained classifier is in history_library.rs_global_acc; its last value is the final accuracy, which equals
    balanced accuracy on the balanced test set. The other metrics of such a run are invalid and dropped."""
    h = (r.get("history_library") or {}).get("rs_global_acc")
    if not r["label"].startswith("Fed-GraB") or not h or r.get("fedgrab_eval_fixed"):
        return r
    keep = {k: v for k, v in r.items() if not k.startswith(("final_", "val_final_", "er_", "val_er_"))}
    keep.update(final_bal=float(h[-1]), final_acc=float(h[-1]), corrected="Fed-GraB: library's last-round accuracy")
    return keep


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*", "runs.jsonl"))):
        if os.path.basename(os.path.dirname(f)).startswith(("NB00", "_")):
            continue
        for line in open(f):
            if line.strip():
                r = correct_fedgrab(json.loads(line))
                r["_folder"] = os.path.basename(os.path.dirname(f))
                runs[r["run_id"]] = r
    return list(runs.values())


def exp_of(runs, tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_folder"}) + "\n")
    return plots.Exp(d)


def sid(S):
    """Setting id used in the number keys: the dataset key, or e.g. cifar10-r100 with one shared model."""
    ds, IF = S[0], S[1]
    if DATASETS.get(ds, {}).get("track") == 2:
        return f"{ds[:-3]}-r{int(round(1 / IF))}"
    return ds


def ordered(E):
    order = list(DATASETS)
    return sorted(E.settings, key=lambda S: (order.index(S[0]) if S[0] in order else 99, S[1] or 1.0, S[3]))


def tex(s):
    return pt.tex_escape(s)


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "twenty-one",
         "twenty-two", "twenty-three", "twenty-four"]
DATA_CITE = {"mnist": "lecun1998mnist", "fmnist": "xiao2017fmnist", "svhn": "netzer2011svhn",
             "cifar10": "krizhevsky2009cifar", "cifar100": "krizhevsky2009cifar", "dermamnist": "yang2023medmnist",
             "organamnist": "yang2023medmnist", "bloodmnist": "yang2023medmnist", "pathmnist": "yang2023medmnist",
             "gtsrb": "stallkamp2012gtsrb", "emnist": "cohen2017emnist", "fer2013": "goodfellow2013fer"}
LT_METHODS = {"FedLC", "CCVR", "CReFF", "FedETF", "Fed-GraB", "FedLoGe", "FedYoYo", "RUCR"}


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def and_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def signed(x):
    x = 0.0 if abs(x) < 0.05 else x   # no "-0.0"
    return f"{x:+.1f}"


# ---------------------------------------------------------------- statistics
def paired(E, S, key):
    """FedECS-LT minus each method on `key`, paired over common seeds: {method: (n, mean, half-width, p, p_Holm)}."""
    from scipy import stats
    import representation as rp
    ours = {r["seed"]: r for r in E.of(OURS, S)}
    rows = []
    for m in E.methods:
        if m == OURS:
            continue
        other = {r["seed"]: r for r in E.of(m, S)}
        d = [100 * (ours[x][key] - other[x][key]) for x in sorted(set(ours) & set(other))
             if ours[x].get(key) is not None and other[x].get(key) is not None]
        if len(d) >= 2:
            sd = float(np.std(d, ddof=1))
            half = float(stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d)))
            p = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (0.0 if np.mean(d) != 0 else 1.0)
            rows.append((m, len(d), float(np.mean(d)), half, p))
    adj = rp.holm([r[4] for r in rows]) if rows else []
    return {r[0]: (r[1], r[2], r[3], r[4], a) for r, a in zip(rows, adj)}


def matched(E, S, key, exclude=()):
    """FedECS-LT minus each method on `key` (points), averaged over the seeds both have: {method: (difference,
    seeds)}. Means over different seed sets are not comparable (some methods came back with fewer seeds), so every
    best / strongest / margin statement uses this."""
    ours = {r["seed"]: r.get(key) for r in E.of(OURS, S)}
    out = {}
    for m in E.methods:
        if m in (OURS, DIAG) or m in exclude:
            continue
        other = {r["seed"]: r.get(key) for r in E.of(m, S)}
        common = [x for x in sorted(set(ours) & set(other)) if ours[x] is not None and other[x] is not None]
        if common:
            out[m] = (100 * float(np.mean([ours[x] - other[x] for x in common])), common)
    return out


def ours_best(E, S):
    d = matched(E, S, "final_bal")
    return bool(d) and min(v for v, _ in d.values()) >= -1e-9


def mean_of(E, m, key, S):
    return plots.metric(E, m, key, S)[0]


# ---------------------------------------------------------------- tables
def write_table(name, header, rows, bold):
    os.makedirs(os.path.join(GEN, "tables"), exist_ok=True)
    out = os.path.join(GEN, "tables")
    pt.write(out, name, header, rows, bold)
    return name


def across_table(E, settings, key, methods=None, lower=False):
    header = ["Method"] + [pt.column_name(S, settings) for S in settings]
    rows, vals = [], []
    for m in (methods or E.methods):
        cells = [plots.metric(E, m, key, S) for S in settings]
        if all(np.isnan(mu) for mu, _, _ in cells):
            continue
        rows.append([m] + [report.fmt(mu, sd, key, n) for mu, sd, n in cells])
        vals.append([mu for mu, _, _ in cells])
    bold = set()
    if vals:
        A = np.array(vals, dtype=float)
        for j in range(A.shape[1]):
            if not np.all(np.isnan(A[:, j])):
                b = np.nanmin(A[:, j]) if lower else np.nanmax(A[:, j])
                bold |= {(int(i), j + 1) for i in np.where(np.isclose(A[:, j], b, rtol=0, atol=1e-9))[0]}
    return header, rows, bold


def paired_table(E, settings, key="final_bal"):
    header = ["Method"] + [pt.column_name(S, settings) for S in settings]
    P = {S: paired(E, S, key) for S in settings}
    rows, bold = [], set()
    for m in E.methods:
        if m == OURS or not any(m in P[S] for S in settings):
            continue
        row = [m]
        for j, S in enumerate(settings):
            d = P[S].get(m)
            row.append(f"{d[1]:+.1f} [{d[1] - d[2]:+.1f}, {d[1] + d[2]:+.1f}]" if d else "-")
            if d and d[4] < 0.05:
                bold.add((len(rows), j + 1))
        rows.append(row)
    return header, rows, bold


# ---------------------------------------------------------------- the fill
def fill_group(g, E, nums, made, status):
    settings = ordered(E)
    if g == "t2":
        settings = [S for S in settings if OURS in {r["label"] for r in E.runs if plots.setting_key(r) == S}] or settings
    status.append(f"## Group `{g}`: {len(E.runs)} runs, {len(settings)} settings\n")
    for S in settings:
        rs = [r for r in E.runs if plots.setting_key(r) == S]
        seeds = sorted({r["seed"] for r in rs})
        ms = sorted({r["label"] for r in rs})
        status.append(f"- {cap.setting_text(S).rstrip('.')}: {len(ms)} methods, seeds {seeds}"
                      + ("" if len(seeds) >= 3 else "  **(fewer than 3 seeds)**"))
        per = {m: sorted({r["seed"] for r in rs if r["label"] == m}) for m in ms}
        short = {m: x for m, x in per.items() if x != seeds}
        if short:   # comparisons use each method's seeds in common with FedECS-LT (matched)
            status.append("  - methods with fewer seeds: " + "; ".join(f"{m} {x}" for m, x in sorted(short.items())))
    status.append("")
    for S in settings:
        s = sid(S)
        nums[f"{g}/{s}/seeds"] = str(len({r["seed"] for r in E.runs if plots.setting_key(r) == S}))
        for m in E.methods:
            for short, key in METRICS:
                mu, sd, n = plots.metric(E, m, key, S)
                if not np.isnan(mu):
                    nums[f"{g}/{s}/{m}/{short}"] = tex(report.fmt(mu, sd, key, n))
            b, e = mean_of(E, m, "final_bal", S), mean_of(E, m, "er_bal", S)
            if not (np.isnan(b) or np.isnan(e)):
                nums[f"{g}/{s}/{m}/gain_er"] = signed(100 * (e - b))
        for key, pre in (("final_bal", ""), ("er_bal", "er_")):
            for m, (n, d, half, p, ph) in paired(E, S, key).items():
                nums[f"{g}/{s}/{pre}diff/{m}"] = signed(d)
                nums[f"{g}/{s}/{pre}ci/{m}"] = f"[{d - half:+.1f}, {d + half:+.1f}]"
                nums[f"{g}/{s}/{pre}p/{m}"] = f"{ph:.3g}"
        d = matched(E, S, "final_bal")   # FedECS-LT minus each method over common seeds
        means = {m: mean_of(E, m, "final_bal", S) for m in E.methods if m not in (DIAG,)}
        means = {m: v for m, v in means.items() if not np.isnan(v)}
        if d:
            nums[f"{g}/{s}/best"] = tex(OURS if ours_best(E, S) else min(d, key=lambda m: d[m][0]))
        elif means:
            nums[f"{g}/{s}/best"] = tex(max(means, key=means.get))
        others = {m: v for m, v in d.items() if m != "Local"}
        if others:
            st = min(others, key=lambda m: others[m][0])
            nums[f"{g}/{s}/strongest"] = tex(st)
            nums[f"{g}/{s}/margin"] = signed(others[st][0])
            nums[f"{g}/{s}/margin_seeds"] = and_list([str(x) for x in others[st][1]])
            pt_ = paired(E, S, "final_bal").get(st)   # paired over seeds, Holm-corrected over all methods
            if pt_:
                nums[f"{g}/{s}/strongest_p"] = f"{pt_[4]:.2g}"
        ers = matched(E, S, "er_bal", exclude=("Local",))
        if ers:   # the same comparison with every last layer replaced by the exact readout
            st = min(ers, key=lambda m: ers[m][0])
            nums[f"{g}/{s}/er_strongest"] = tex(st)
            nums[f"{g}/{s}/er_margin"] = signed(ers[st][0])
            pt_ = paired(E, S, "er_bal").get(st)
            if pt_:
                nums[f"{g}/{s}/er_strongest_p"] = f"{pt_[4]:.2g}"
                nums[f"{g}/{s}/er_strongest_ci"] = f"[{pt_[1] - pt_[2]:+.1f}, {pt_[1] + pt_[2]:+.1f}]"
    sr = [x for r in E.runs if r["label"] == OURS for x in (r.get("spread_ratio") or [])]
    if sr:   # generated over exact within-class spread after the moment correction, every class and run
        nums[f"{g}/mc_spread_lo"], nums[f"{g}/mc_spread_med"], nums[f"{g}/mc_spread_hi"] = (
            f"{np.min(sr):.2f}", f"{np.median(sr):.2f}", f"{np.max(sr):.2f}")
    with_ours = [S for S in settings if not np.isnan(mean_of(E, OURS, "final_bal", S))]
    nums[f"{g}/nsettings"] = str(len(with_ours))
    nums[f"{g}/nsettings_w"] = word(len(with_ours))
    nums[f"{g}/datasets"] = tex(and_list([pt.column_name(S, with_ours) for S in with_ours])) if with_ours else "none"
    if with_ours and g != "t2":   # the same list with each dataset's citation
        nums[f"{g}/datasets_cite"] = and_list([tex(pt.column_name(S, with_ours)) + "~\\cite{" + DATA_CITE[S[0]] + "}"
                                               for S in with_ours if S[0] in DATA_CITE])
    base = sorted({r["label"] for r in E.runs} - {OURS, DIAG, "Local"})
    nums[f"{g}/nbaselines_w"] = word(len(base))
    nums[f"{g}/nlt_w"] = word(len(set(base) & LT_METHODS))
    if g == "t2":
        ratios = sorted({int(round(1 / S[1])) for S in settings}, reverse=True)
        nums["t2/ratios"] = and_list([str(x) for x in ratios])
    nums[f"{g}/nbest"] = str(sum(1 for S in with_ours if ours_best(E, S)))
    nb, ns = int(nums[f"{g}/nbest"]), len(with_ours)   # "all six", "both", "5 of the six", "none of the six"
    nums[f"{g}/nbest_of"] = ("both" if ns == 2 else f"all {word(ns)}") if nb == ns and ns > 1 else \
        (("neither" if ns == 2 else f"none of the {word(ns)}") if nb == 0 else f"{nb} of the {word(ns)}")
    # tables
    for suffix, key in ACROSS:
        h, rows, bold = across_table(E, settings, key, [m for m in E.methods if m != DIAG], key in LOWER)
        if rows:
            made.append(write_table(f"{g}_{suffix}", h, rows, bold))
    h, rows, bold = paired_table(E, settings)
    if rows:
        made.append(write_table(f"{g}_paired", h, rows, bold))
    for S in settings:   # every metric of every setting (appendix)
        rows, bold = pt.metric_rows(E, S, report.MAIN)
        if rows:
            made.append(write_table(f"{g}_{sid(S)}_main", ["Method"] + [n for _, n in report.MAIN], rows, bold))


def fill_extras(E_by, nums, made):
    """Numbers and tables that combine groups: gate statistic, counts, ablation gains, diagnosis, cost, datasets."""
    allE = [E for E in E_by.values() if E is not None]
    ours_runs = [r for E in allE for r in E.runs if r["label"] == OURS]
    T1 = [r for r in ours_runs if DATASETS.get(r["dataset"], {}).get("track") != 2 and r.get("class_sep_F") is not None]
    if T1:
        F = [r["class_sep_F"] for r in T1]
        nums["gate/F/min"], nums["gate/F/max"] = f"{min(F):.3f}", f"{max(F):.3f}"
        rows_ds = sorted({DATASETS[r["dataset"]]["name"] for r in T1 if r.get("gate_branch") == "rows"})
        nums["gate/rows"] = tex(", ".join(rows_ds)) if rows_ds else "none of them"
    labels = {r["label"] for E in allE for r in E.runs}
    published = labels - {OURS, DIAG, "Local"} - set(bn.ABLATIONS) - {"FedDF + LA"}
    nums["count/methods"] = str(len(published))
    nums["count/datasets"] = str(len({r["dataset"].replace("_t2", "") for r in ours_runs}))
    nums["count/datasets_w"] = word(len({r["dataset"].replace("_t2", "") for r in ours_runs}))
    nums["count/methods_w"] = word(len(published))
    E = E_by.get("abl")
    if E is not None:
        for S in ordered(E):
            s = sid(S)
            g, la, us = (mean_of(E, m, "final_bal", S) for m in ("GeFL-F", "GeFL-F + LA", OURS))
            if not (np.isnan(g) or np.isnan(la)):
                nums[f"abl/{s}/GeFL-F + LA/gain"] = signed(100 * (la - g))
            if not (np.isnan(us) or np.isnan(la)):
                nums[f"abl/{s}/rest"] = signed(100 * (us - la))
        h, rows, bold = across_table(E, ordered(E), "final_bal")
        if rows:
            made.append(write_table("abl_bal", h, rows, bold))
        h, rows, bold = across_table(E, ordered(E), "er_bal")
        if rows:
            made.append(write_table("abl_er", h, rows, bold))
    E = E_by.get("t2")
    if E is not None:
        for S in ordered(E):
            d, o = mean_of(E, DIAG, "final_bal", S), mean_of(E, OURS, "final_bal", S)
            if not (np.isnan(d) or np.isnan(o)):
                nums[f"t2/{sid(S)}/fe200_diff"] = signed(100 * (d - o))
    E = E_by.get("main")
    if E is not None:
        for S in ordered(E):
            s = sid(S)
            for key, short in (("time_s", "time_ratio"), ("comm_mb_per_client", "comm_ratio")):
                a, b = mean_of(E, OURS, key, S), mean_of(E, "GeFL-F", key, S)
                if not (np.isnan(a) or np.isnan(b)) and b > 0:
                    nums[f"cost/{s}/{short}"] = f"{a / b:.2f}"
            if s == "cifar10":
                keys = [k for k in report.SYSTEM if k[0] in ("comm_mb_per_client", "time_s", "fwd_mflops",
                                                           "train_gflops_round")]
                rows, bold = pt.metric_rows(E, S, keys)
                if rows:
                    made.append(write_table("cost", ["Method"] + [n for _, n in keys], rows, bold))
    E = E_by.get("comp")
    if E is not None:   # components of FedECS-LT: rows = variants, columns = (dataset, as trained / ER)
        sets = ordered(E)
        header = ["Variant"] + [f"{pt.column_name(S, sets)} {w}" for S in sets for w in ("as trained", "ER")]
        rows = []
        for m in [OURS] + [m for m in E.methods if m.startswith("FedECS-LT w")]:
            cells = []
            for S in sets:
                for key in ("final_bal", "er_bal"):
                    mu, sd, n = plots.metric(E, m, key, S)
                    cells.append(report.fmt(mu, sd, key, n))
            if any(c != "-" for c in cells):
                rows.append([("FedECS-LT (full)" if m == OURS else m.replace("FedECS-LT ", ""))] + cells)
        if rows:
            made.append(write_table("abl_components", header, rows, set()))
    E = E_by.get("sweep")
    if E is not None:   # robustness: rows = methods, columns = the changed setting
        sets = ordered(E)

        def label(S):
            IF, alpha, K = S[1], S[2], S[3]
            return (f"Dirichlet {alpha}" if alpha != 0.5 else f"K = {K}" if K != 10 else
                    f"ratio {int(round(1 / IF))}" if IF else "natural")
        header = ["Method"] + [label(S) for S in sets]
        rows = []
        for m in bn.SWEEP_METHODS:
            cells = [report.fmt(*plots.metric(E, m, "final_bal", S)[:2], "final_bal", plots.metric(E, m, "final_bal",
                                                                                                    S)[2])
                     for S in sets]
            if any(c != "-" for c in cells):
                rows.append([m] + cells)
        if rows:
            made.append(write_table("robust", header, rows, set()))
            for S in sets:
                for m in bn.SWEEP_METHODS:
                    mu, sd, n = plots.metric(E, m, "final_bal", S)
                    if not np.isnan(mu):
                        nums[f"sweep/{label(S).replace(' ', '')}/{m}/bal"] = tex(report.fmt(mu, sd, "final_bal", n))
    # datasets table: from one run of each dataset
    rows = []
    for g in ("main", "nat"):
        E = E_by.get(g)
        if E is None:
            continue
        for S in ordered(E):
            r = next(r for r in E.runs if plots.setting_key(r) == S)
            spec = DATASETS[S[0]]
            tot = np.array(r["client_counts"]).sum(0)
            name = spec["name"] + ("-LT" if spec["imbalance"] == "IF" else "")
            rows.append([name, str(len(r["test_class_counts"])), f"{int(r['n_train']):,}",
                         f"{int(sum(r['test_class_counts'])):,}", f"{tot.max() / max(tot.min(), 1):.0f}"])
    if rows:
        made.append(write_table("datasets", ["Dataset", "Classes", "Training images", "Test images",
                                             "Largest / smallest class"], rows, set()))


# figure slots: name -> (figure number in the notebooks' reports, dataset key, imbalance factor)
FIGS = {"fig_rows": (30, "cifar10", 0.01), "fig_spread": (75, "cifar10", 0.01), "fig_forest": (39, "cifar10", 0.01),
        "fig_recall_size": (10, "cifar100", 0.01), "fig_hmt": (14, "cifar10", 0.01), "fig_tsne": (83, "cifar10", 0.01),
        "fig_realgen": (22, "cifar10", 0.01), "fig_shared_rounds": (71, "cifar100_t2", 0.01)}


def fill_figures(made, status, nums):
    sys.path.insert(0, FL)
    import merge_results
    zips = sorted(glob.glob(os.path.join(RESULTS, "_downloads", "*__results.zip")))
    fns = {num: fn for num, _, fn in plots.FIGS}
    os.makedirs(os.path.join(GEN, "figures"), exist_ok=True)
    cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        for slot, (num, ds, IF) in FIGS.items():
            key = (ds, IF)
            if key not in cache:
                if DATASETS[ds].get("track") == 2:
                    pre = [f"NB20_track2_{ds[:-3]}_IF{int(round(1 / IF)):03d}_", "NB20_track2_rerun_"]
                else:
                    pre = [f"NB10_main_{ds}_", f"NB10_main_{ds}__"]
                mine = [z for z in zips if any(os.path.basename(z).startswith(p) for p in pre)]
                if not mine:
                    cache[key] = None
                else:
                    d = os.path.join(tmp, f"{ds}_{IF}")
                    merge_results.merge(mine, d)
                    cache[key] = plots.Exp(d)
            E = cache[key]
            if E is None:
                status.append(f"- `{slot}`: no NAME__results.zip of {ds} in fedecs_lt/results/_downloads "
                              "(python fedecs_lt/fetch_kaggle.py --results LINK)")
                continue
            S = next((S for S in E.settings if S[0] == ds and abs((S[1] or 0) - IF) < 1e-9), None)
            try:
                got = fns[num](E, S, os.path.join(tmp, "figs", slot)) if S else []
            except Exception as e:
                got = []
                status.append(f"- `{slot}`: Fig. {num} failed: {e!r}"[:300])
            if got:
                pdf, caption = got[0]
                shutil.copy(pdf, os.path.join(GEN, "figures", slot + ".pdf"))
                open(os.path.join(GEN, "figures", slot + ".caption.txt"), "w").write(caption + "\n")
                made.append(slot)
            elif S:
                status.append(f"- `{slot}`: Fig. {num} not applicable here ({cap.not_applicable(num)})")
        # 10-NN accuracy of the embeddings (representation section), CIFAR-10-LT
        E = cache.get(("cifar10", 0.01))
        if E is not None:
            import representation as rp
            for m in (OURS, "GeFL-F"):
                accs = []
                for r in E.of(m):
                    d = E.pred(r)
                    if d is None:
                        continue
                    keys = sorted(k for k in d.files if k.startswith("embed_"))
                    if keys and "y" in d.files:
                        accs.append(rp.knn_correct(d[keys[0]].astype(np.float64), d["y"]).mean())
                if accs:
                    nums[f"repr/cifar10/{m}/knn"] = tex(f"{100 * np.mean(accs):.1f}%")


def content_files():
    return [p for p in glob.glob(os.path.join(HERE, "content", "*.tex")) if not p.endswith("macros.tex")]


def missing_notebooks():
    done = {os.path.basename(d) for d in glob.glob(os.path.join(RESULTS, "NB*"))}
    out = {}
    for n in bn.all_notebooks():
        if n not in done and not n.startswith("NB00"):
            out.setdefault(bn.FOLDERS[bn.priority(n)], []).append(n)
    return out


def check_highlights():
    t = open(os.path.join(HERE, "elsevier", "highlights.tex")).read()
    items = re.findall(r"\\item (.+)", t)
    for it in items:
        print(f"{len(it):3d} chars {'OK ' if len(it) <= 85 else 'TOO LONG'}  {it}")
    print(f"{len(items)} highlights ({'OK' if 3 <= len(items) <= 5 else 'need 3-5'})")


def main():
    if "--check-highlights" in sys.argv:
        return check_highlights()
    runs = load_runs()
    by = {}
    for r in runs:
        g = group_of(r)
        if g:
            by.setdefault(g, []).append(r)
    # the ablation table also needs the references of the same datasets (same splits and seeds)
    if "abl" in by:
        ds = {r["dataset"] for r in by["abl"]}
        by["abl"] += [r for r in by.get("main", []) if r["dataset"] in ds and r["label"] in (OURS, "GeFL-F", "FedDF")]
    if "comp" in by:   # the full FedECS-LT of the same datasets is the components' reference
        ds = {r["dataset"] for r in by["comp"]}
        by["comp"] += [r for r in by.get("main", []) if r["dataset"] in ds and r["label"] == OURS]
    shutil.rmtree(os.path.join(GEN, "tables"), ignore_errors=True)
    nums, made = {}, []
    status = ["# Result slots: status", "",
              f"Written by `python docs/papers/fill.py` from {len(runs)} runs in fedecs_lt/results "
              "(NB00 excluded). Slots not listed under 'filled' still show a red placeholder.", ""]
    E_by = {}
    with tempfile.TemporaryDirectory() as tmp:
        for g in ("main", "nat", "t2", "abl", "comp", "sweep"):
            E_by[g] = exp_of(by[g], tmp, g) if by.get(g) else None
            if E_by[g] is not None and g in ("main", "nat", "t2"):
                fill_group(g, E_by[g], nums, made, status)
        fill_extras(E_by, nums, made)
    if "--figures" in sys.argv:
        status.append("## Figures\n")
        fill_figures(made, status, nums)
        status.append("")
    os.makedirs(GEN, exist_ok=True)
    if "--figures" not in sys.argv:   # numbers computed only with the figures (repr/...): keep the last values
        old = os.path.join(GEN, "numbers.tex")
        if os.path.exists(old):
            for k, v in re.findall(r"\\defnum\{(repr/[^}]*)\}\{(.*)\}\s*$", open(old).read(), re.M):
                nums.setdefault(k, v)
    with open(os.path.join(GEN, "numbers.tex"), "w") as f:
        f.write("% written by docs/papers/fill.py; do not edit\n")
        for k in sorted(nums):
            f.write(f"\\defnum{{{k}}}{{{nums[k]}}}\n")
    text = "".join(open(p).read() for p in content_files())
    used_tabs = set(re.findall(r"\\slottab\{([^}]+)\}", text))
    used_nums = set(re.findall(r"\\R\{([^}]+)\}", text))
    used_figs = set(re.findall(r"\\slotfig\{([^}]+)\}", text))
    have_figs = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(GEN, "figures", "*.pdf"))}
    status += ["## Slots in the paper", "",
               f"- Tables filled: {len(used_tabs & set(made))} of {len(used_tabs)}; still empty: "
               + (", ".join(f"`{t}`" for t in sorted(used_tabs - set(made))) or "none"),
               f"- Numbers filled: {len(used_nums & set(nums))} of {len(used_nums)}; still empty: "
               + (", ".join(f"`{t}`" for t in sorted(used_nums - set(nums))) or "none"),
               f"- Figures filled: {len(used_figs & have_figs)} of {len(used_figs)}; still empty: "
               + (", ".join(f"`{t}`" for t in sorted(used_figs - have_figs)) or "none"), "",
               "## Notebooks not returned yet (by folder)", ""]
    for folder, names in sorted(missing_notebooks().items()):
        status.append(f"- {folder}: " + ", ".join(names))
    with open(os.path.join(GEN, "STATUS.md"), "w") as f:
        f.write("\n".join(status) + "\n")
    print(f"{len(nums)} numbers, {len(made)} tables/figures written; see docs/papers/generated/STATUS.md")
    print(f"slots in the paper: tables {len(used_tabs & set(made))}/{len(used_tabs)}, numbers "
          f"{len(used_nums & set(nums))}/{len(used_nums)}, figures {len(used_figs & have_figs)}/{len(used_figs)}")


if __name__ == "__main__":
    main()
