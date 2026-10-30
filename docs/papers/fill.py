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
from baselines import GROUPED  # noqa: E402

OURS = "FedECS-LT"
SIG = {}   # (group, setting) -> margin over the strongest other method significant after Holm (unrounded p)
DIAG = "FedECS-LT (200 extractor rounds)"
HEADS = "FedECS-LT (heads only)"
MATCHED = ("GeFL-F (10 synthetic epochs)", "FedDF (10 synthetic epochs)")   # as many synthetic epochs per round as FedECS-LT   # FedECS-LT before its last step: the heads as trained, without the exact readout
NOTEST = {DIAG, HEADS}   # shown in the tables, never in the test family, "best", "strongest" or the baseline counts
# references, not competitors: shown in the tables, never "best", "strongest" or counted among the baselines
REFS = {"Centralised (pooled data)"}
# runs kept out of the paper by the user's decision (30 Oct 2026): the Gaussian-sampler component is reported in a
# separate note (docs/study/gaussian_sampler.tex); its numbers are written under note/<ds>/ and comp/<ds>/
NOT_IN_PAPER = {"FedECS-LT w/ Gaussian sampler"}
# number slots per method: (short name in the key, runs.jsonl key)
METRICS = [("bal", "final_bal"), ("er_bal", "er_bal"), ("f1", "final_f1"), ("tail", "final_tail"),
           ("worst", "final_worst"), ("mcc", "final_mcc"), ("gmean", "final_gmean"), ("ece", "final_ece"),
           ("worst_arch", "final_worst_arch"), ("er_tail", "er_tail"), ("local", "final_local"),
           ("gefl", "gefl_best_acc"), ("gefl_drop", "gefl_drop")]
# tables across settings: (slot suffix, runs.jsonl key)
ACROSS = [("bal", "final_bal"), ("er", "er_bal"), ("ertail", "er_tail"), ("f1", "final_f1"), ("tail", "final_tail"),
          ("worst", "final_worst"), ("ece", "final_ece"), ("worst_arch", "final_worst_arch"),
          ("gefl", "gefl_best_acc"),   # GeFL's measure: every model's best round (fedecs_lt/gefl_measure.py)
          ("local", "final_local")]   # accuracy on each model's own clients' class mix
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
    if main_setting(r) and r["seed"] in bn.FRESH_SEEDS:
        return "fresh" if lab in bn.FRESH_METHODS else None   # the check on new client splits (seeds 6-8)
    if not main_setting(r):
        return "sweep" if lab in bn.SWEEP_METHODS else None
    if lab in bn.GEFL_VARIANTS + bn.MATCHED_TS and not lab.startswith("FedECS-LT w"):
        return "abl"   # variants of GeFL-F and FedDF (the Gaussian sampler variant of FedECS-LT is a component)
    if lab.startswith("FedECS-LT w"):
        return "comp"
    if lab in bn.TRACK1_MAIN:
        return "main" if ds in bn.TIER1 else "nat" if ds in bn.TIER2 else None
    if (lab in GROUPED or lab in REFS) and ds in bn.TIER1:
        return "main"   # GeFL's grouped baselines and the pooled-data reference (action plan section 24)
    return None


def k50_setting(r):
    """Fifty clients, five per architecture (GeFL's second setting): every method, otherwise the main setting."""
    return (DATASETS.get(r["dataset"], {}).get("track") != 2 and r["K"] == 50 and r["alpha"] == 0.5
            and r["IF"] == 0.01 and r.get("pfrac", 1.0) == 1.0 and not r.get("tail_perm")
            and r["dataset"] in bn.TIER1)


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


def with_readout(r):
    """FedECS-LT ends with the exact readout (paper, sec:method-readout), so its reported metrics (final_*, val_final_*)
    are those of the readout (er_*, val_er_*), computed in the same run. Its heads as trained become a second run,
    labelled HEADS, with GeFL's per-round measure (heads only: the readout is computed once, at the end). The
    as-trained values also stay in the FedECS-LT run under heads_final_*. Other methods are unchanged."""
    if r["label"] != OURS or r.get("er_bal") is None:
        return [r]
    heads = {k: v for k, v in r.items() if not k.startswith(("er_", "val_er_"))}
    heads["label"] = HEADS
    full = {k: v for k, v in r.items() if not k.startswith("gefl_") and k != "final_per_arch"}
    for k in [k for k in r if k.startswith(("er_", "val_er_"))]:
        f = k.replace("er_", "final_", 1)
        full["heads_" + f] = r.get(f)
        full[f] = r[k]
    return [full, heads]


def load_runs():
    runs = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*", "runs.jsonl"))):
        if os.path.basename(os.path.dirname(f)).startswith(("NB00", "_")):
            continue
        gm = os.path.join(os.path.dirname(f), "gefl_measure.jsonl")   # GeFL's measure from the per-round logs
        gefl = {o["run_id"]: o for o in map(json.loads, filter(str.strip, open(gm)))} if os.path.exists(gm) else {}
        for line in open(f):
            if line.strip():
                r = correct_fedgrab(json.loads(line))
                r["_folder"] = os.path.basename(os.path.dirname(f))
                if r["run_id"] in gefl:
                    r["gefl_best_acc"] = gefl[r["run_id"]]["gefl_best_acc"]
                    if gefl[r["run_id"]].get("gefl_last_acc") is not None:   # lost after the best round
                        r["gefl_drop"] = r["gefl_best_acc"] - gefl[r["run_id"]]["gefl_last_acc"]
                runs[r["run_id"]] = r
    return list(runs.values())


def repeat_numbers(nums):
    """Run-to-run variation: a notebook run a second time with the same run list (same splits, seeds and code apart
    from scheduling; results/_repeats/), against the first run (the paper's): the standard deviation over runs of the
    difference between the two runs of one configuration, and the largest such difference."""
    for d in sorted(glob.glob(os.path.join(RESULTS, "_repeats", "*", "runs.jsonl"))):
        name = os.path.basename(os.path.dirname(d)).split("__")[0]
        first = os.path.join(RESULTS, name, "runs.jsonl")
        if not os.path.exists(first):
            continue
        a = {r["run_id"]: r for r in map(json.loads, filter(str.strip, open(first)))}
        b = {r["run_id"]: r for r in map(json.loads, filter(str.strip, open(d)))}
        common = sorted(set(a) & set(b))
        for key, short in (("final_bal", "bal"), ("er_bal", "er")):
            diff = [(100 * (b[k][key] - a[k][key]), a[k]["label"]) for k in common
                    if a[k].get(key) is not None and b[k].get(key) is not None]
            if len(diff) < 2:
                continue
            v = np.array([x for x, _ in diff])
            nums[f"repeat/{short}/sd"] = f"{v.std(ddof=1):.1f}"
            i = int(np.argmax(np.abs(v)))
            nums[f"repeat/{short}/max"] = f"{abs(v[i]):.1f}"
            nums[f"repeat/{short}/max_method"] = tex(diff[i][1])
            nums[f"repeat/{short}/n"] = str(len(v))
            # GPU non-determinism alone in a paired difference of two methods averaged over three seeds: each seed's
            # difference varies by the SD of a two-run difference, the 3-seed mean by that / sqrt(3); twice that is the
            # smallest difference the paper treats as resolved
            nums[f"repeat/{short}/floor"] = f"{2 * v.std(ddof=1) / np.sqrt(3):.1f}"
        ds = a[common[0]]["dataset"] if common else None
        if ds:
            nums["repeat/dataset"] = tex(DATASETS[ds]["name"])
        break   # one repeated notebook so far (BloodMNIST)


def reproduction_numbers(nums):
    """GeFL's own setting (balanced IID data, 10 clients) in the earlier implementation (experiments/core.py,
    results K16_kaggle_recalibration__*_iid, seeds 0-2): GeFL-F and FedECS-LT at the last round and with GeFL's
    measure, against GeFL-F's published CIFAR-10 accuracy (GeFL paper, Fig. 4, 10 clients; core.DATASETS)."""
    root = os.path.join(ROOT, "experiments", "results")
    sys.path.insert(0, os.path.join(ROOT, "experiments"))
    try:
        import core
        nums["repro/cifar10/published"] = f"{core.DATASETS['cifar10']['paper']['cvae'][0]:.1f}"
    except Exception:
        pass
    for ds in ("cifar10", "cifar100"):
        f = os.path.join(root, f"K16_kaggle_recalibration__{ds}_iid", "runs.jsonl")
        if not os.path.exists(f):
            continue
        rs = [json.loads(x) for x in open(f) if x.strip()]
        for label, key in (("GeFL-F", "GeFL-F"), ("Ours (gated)", "FedECS-LT")):
            sel = [r for r in rs if r["label"] == label]
            if sel:
                nums[f"repro/{ds}/{key}/final"] = f"{100 * np.mean([r['final_bal'] for r in sel]):.1f}"
                nums[f"repro/{ds}/{key}/best"] = f"{100 * np.mean([r['best_mean_acc'] for r in sel]):.1f}"
                nums[f"repro/{ds}/{key}/seeds"] = str(len(sel))


def exp_of(runs, tmp, name):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "runs.jsonl"), "w") as f:
        for r in runs:
            f.write(json.dumps({k: v for k, v in r.items() if k != "_folder"}) + "\n")
    E = plots.Exp(d)
    if HEADS in E.methods:   # the heads-only row right after FedECS-LT
        E.methods.remove(HEADS)
        E.methods.insert(E.methods.index(OURS) + 1 if OURS in E.methods else 0, HEADS)
    return E


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
    if abs(x) < 0.005:
        return "+0.0"
    return f"{x:+.2f}" if abs(x) < 0.05 else f"{x:+.1f}"   # a small difference keeps its sign ("-0.03", not "+0.0")


# ---------------------------------------------------------------- statistics
def paired(E, S, key, ours_label=OURS):
    """FedECS-LT (or ours_label) minus each method on `key`, paired over common seeds:
    {method: (n, mean, half-width, p, p_Holm)}."""
    from scipy import stats
    import representation as rp
    ours = {r["seed"]: r for r in E.of(ours_label, S)}
    rows = []
    for m in E.methods:
        if m == OURS or m in NOTEST or m in REFS:   # references and diagnostics are not in the test family
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


def matched(E, S, key, exclude=(), ours_label=OURS):
    """FedECS-LT minus each method on `key` (points), averaged over the seeds both have: {method: (difference,
    seeds)}. Means over different seed sets are not comparable (some methods came back with fewer seeds), so every
    best / strongest / margin statement uses this."""
    ours = {r["seed"]: r.get(key) for r in E.of(ours_label, S)}
    out = {}
    for m in E.methods:
        if m == OURS or m in NOTEST or m in REFS or m in exclude:
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
            col = np.array([np.nan if r[0] in REFS or r[0] in NOTEST else v for r, v in zip(rows, A[:, j])],
                           dtype=float)
            if not np.all(np.isnan(col)):
                b = np.nanmin(col) if lower else np.nanmax(col)
                bold |= {(int(i), j + 1) for i in np.where(np.isclose(col, b, rtol=0, atol=1e-9))[0]}
    return header, rows, bold


def paired_table(E, settings, key="final_bal"):
    header = ["Method"] + [pt.column_name(S, settings) for S in settings]
    P = {S: paired(E, S, key) for S in settings}
    rows, bold = [], set()
    for m in E.methods:
        if m == OURS or m in NOTEST or not any(m in P[S] for S in settings):
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
    if g == "k50" and any(not E.of(OURS, S) for S in settings):   # the comparison needs FedECS-LT in every setting
        status.append("  - not written to the paper yet: FedECS-LT has no run in every setting\n")
        return
    for S in settings:
        s = sid(S)
        nums[f"{g}/{s}/seeds"] = str(len({r["seed"] for r in E.runs if plots.setting_key(r) == S}))
        for m in E.methods:
            for short, key in METRICS:
                mu, sd, n = plots.metric(E, m, key, S)
                if not np.isnan(mu):
                    nums[f"{g}/{s}/{m}/{short}"] = tex(report.fmt(mu, sd, key, n))
            nz = [sum(1 for x in r["final_recall_c"] if x == 0) for r in E.of(m, S) if r.get("final_recall_c")]
            if nz:   # classes that none of the client models recognises (pooled recall 0), mean over seeds
                nums[f"{g}/{s}/{m}/nzero"] = f"{np.mean(nz):.1f}".rstrip("0").rstrip(".")
            b = mean_of(E, m, "heads_final_bal" if m == OURS else "final_bal", S)   # as trained
            e = mean_of(E, m, "er_bal", S)
            if not (np.isnan(b) or np.isnan(e)):
                nums[f"{g}/{s}/{m}/gain_er"] = signed(100 * (e - b))
        for key, pre in (("final_bal", ""), ("er_bal", "er_"), ("final_head", "head_"), ("final_medium", "med_"),
                         ("final_tail", "tail_"), ("er_tail", "ertail_")):
            for m, (n, d, half, p, ph) in paired(E, S, key).items():
                nums[f"{g}/{s}/{pre}diff/{m}"] = signed(d)
                nums[f"{g}/{s}/{pre}ci/{m}"] = f"[{d - half:+.1f}, {d + half:+.1f}]"
                nums[f"{g}/{s}/{pre}p/{m}"] = f"{ph:.2g}"
        d = matched(E, S, "final_bal")   # FedECS-LT minus each method over common seeds
        means = {m: mean_of(E, m, "final_bal", S) for m in E.methods if m not in NOTEST and m not in REFS}
        means = {m: v for m, v in means.items() if not np.isnan(v)}
        if d:
            nums[f"{g}/{s}/best"] = tex(OURS if ours_best(E, S) else min(d, key=lambda m: d[m][0]))
        elif means:
            nums[f"{g}/{s}/best"] = tex(max(means, key=means.get))
        others = {m: v for m, v in d.items() if m != "Local"}
        if others:   # which methods are above / below FedECS-LT on the seeds both have (for statements of scope)
            ab = sorted((m for m in others if others[m][0] < 0), key=lambda m: others[m][0])
            nums[f"{g}/{s}/above_list"] = tex(and_list([m.replace(" (LongTailLib)", "") for m in ab])) if ab else "none"
            nums[f"{g}/{s}/n_above"] = str(len(ab))
            nums[f"{g}/{s}/n_below"] = str(sum(1 for m in others if others[m][0] > 0))
        if others:
            st = min(others, key=lambda m: others[m][0])
            nums[f"{g}/{s}/strongest"] = tex(st)
            nums[f"{g}/{s}/margin"] = signed(others[st][0])
            if f"{g}/{s}/{st}/nzero" in nums:
                nums[f"{g}/{s}/strongest_nzero"] = nums[f"{g}/{s}/{st}/nzero"]
            for short, _ in METRICS:   # every metric of the strongest other method
                if f"{g}/{s}/{st}/{short}" in nums:
                    nums[f"{g}/{s}/strongest/{short}"] = nums[f"{g}/{s}/{st}/{short}"]
            nums[f"{g}/{s}/margin_seeds"] = and_list([str(x) for x in others[st][1]])
            pt_ = paired(E, S, "final_bal").get(st)   # paired over seeds, Holm-corrected over all methods
            if pt_:
                nums[f"{g}/{s}/strongest_p"] = f"{pt_[4]:.2g}"
                SIG[(g, s)] = pt_[4] < 0.05   # decided on the unrounded p-value
        hd = matched(E, S, "final_bal", exclude=("Local",), ours_label=HEADS)
        if hd:   # FedECS-LT's heads before the readout against every other method as trained
            st = min(hd, key=lambda m: hd[m][0])
            nums[f"{g}/{s}/heads_strongest"] = tex(st)
            nums[f"{g}/{s}/heads_margin"] = signed(hd[st][0])
            pt_ = paired(E, S, "final_bal", ours_label=HEADS).get(st)
            if pt_:
                nums[f"{g}/{s}/heads_strongest_p"] = f"{pt_[4]:.2g}"
        ers = matched(E, S, "er_bal", exclude=("Local",))
        if ers:   # the same comparison with every last layer replaced by the exact readout
            st = min(ers, key=lambda m: ers[m][0])
            nums[f"{g}/{s}/er_strongest"] = tex(st)
            nums[f"{g}/{s}/er_margin"] = signed(ers[st][0])
            pt_ = paired(E, S, "er_bal").get(st)
            if pt_:
                nums[f"{g}/{s}/er_strongest_p"] = f"{pt_[4]:.2g}"
                nums[f"{g}/{s}/er_strongest_ci"] = f"[{pt_[1] - pt_[2]:+.1f}, {pt_[1] + pt_[2]:+.1f}]"
    mg = {S: matched(E, S, "final_bal") for S in settings}
    mg = {S: min(v[0] for m, v in d.items() if m != "Local") for S, d in mg.items() if any(m != "Local" for m in d)}
    if mg:   # range of the margin over the strongest other method across the settings, with each end's dataset
        for end, S in (("lo", min(mg, key=mg.get)), ("hi", max(mg, key=mg.get))):
            n = len(E.of(OURS, S))
            nums[f"{g}/margin_{end}"] = signed(mg[S])
            nums[f"{g}/margin_{end}_ds"] = tex(pt.column_name(S, settings)) + (", one seed" if n == 1 else "")
    me = {S: matched(E, S, "er_bal", exclude=("Local",)) for S in settings}
    who = {S: min(d, key=lambda m: d[m][0]) for S, d in me.items() if d}
    me = {S: min(v[0] for v in d.values()) for S, d in me.items() if d}
    if me:   # the same range with the exact readout for every method of the GeFL-F pipeline
        for end, S in (("lo", min(me, key=me.get)), ("hi", max(me, key=me.get))):
            nums[f"{g}/er_margin_{end}"] = signed(me[S])
            nums[f"{g}/er_margin_{end}_ds"] = tex(pt.column_name(S, settings))
            nums[f"{g}/er_margin_{end}_who"] = tex(who[S])
        behind = [S for S in me if me[S] < 0]   # settings where a method with the readout has the higher mean
        nums[f"{g}/er_n_behind"] = word(len(behind))
        # settings where the lead with the readout is significant after Holm (over the GeFL-F pipeline methods)
        sig = [S for S in me if (lambda d: d and d[4] < 0.05)(paired(E, S, "er_bal").get(who[S]))]
        nums[f"{g}/er_n_sig"] = word(len(sig)) if sig else "none"
        nums[f"{g}/er_sig_list"] = and_list([tex(pt.column_name(S, settings)) for S in sig]) if sig else "none"
    if g == "t2":   # the strongest method at imbalance ratio 100 on both datasets, in one phrase
        pair = [S for S in settings if round(1 / S[1]) == 100]
        st = {S: min(((m, v) for m, v in matched(E, S, "final_bal").items() if m != "Local"),
                     key=lambda x: x[1][0], default=None) for S in pair}
        if len(pair) == 2 and all(st.values()):
            (m1, (d1, _)), (m2, (d2, _)) = st[pair[0]], st[pair[1]]
            who = f"the strongest method, {m1}, is" if m1 == m2 else f"the strongest methods, {m1} and {m2}, are"
            side = lambda d: "behind" if d >= 0 else "ahead of"   # d = ours minus theirs
            gap = (f"{abs(d1):.1f} and {abs(d2):.1f} points {side(d1)}" if side(d1) == side(d2) else
                   f"{abs(d1):.1f} points {side(d1)} \\ours on {tex(pt.column_name(pair[0], settings))} and "
                   f"{abs(d2):.1f} points {side(d2)}")
            nums["t2/r100/strongest_phrase"] = tex(who) + " " + gap
    above = None   # methods above FedECS-LT on their clients' own class mixes in every setting (common seeds)
    for S in settings:
        loc = matched(E, S, "final_local", exclude=("Local",))
        if loc:   # how many other federated methods are above FedECS-LT on the own mixes, in this setting
            nums[f"{g}/{sid(S)}/local_n_above"] = str(sum(1 for v, _ in loc.values() if v < 0))
            nums[f"{g}/{sid(S)}/local_n_other"] = str(len(loc))
        d = {m for m, (v, _) in matched(E, S, "final_local").items() if v < 0}
        above = d if above is None else above & d
    if above is not None:
        nums[f"{g}/local_above"] = tex(and_list([m for m in E.methods if m in above])) if above else "no method"
    sr = [x for r in E.runs if r["label"] == OURS for x in (r.get("spread_ratio") or [])]
    if sr:   # generated over exact within-class spread after the moment correction, every class and run
        nums[f"{g}/mc_spread_lo"], nums[f"{g}/mc_spread_med"], nums[f"{g}/mc_spread_hi"] = (
            f"{np.min(sr):.2f}", f"{np.median(sr):.2f}", f"{np.max(sr):.2f}")
    for S in settings:   # rank of FedECS-LT by calibration error among every method of the setting (1 = best)
        ece = {m: mean_of(E, m, "final_ece", S) for m in E.methods if m not in NOTEST and m not in REFS}
        ece = {m: v for m, v in ece.items() if not np.isnan(v)}
        if OURS in ece:
            nums[f"{g}/{sid(S)}/ece_rank"] = str(1 + sum(1 for v in ece.values() if v < ece[OURS]))
            nums[f"{g}/{sid(S)}/ece_n"] = word(len(ece))
    with_ours = [S for S in settings if not np.isnan(mean_of(E, OURS, "final_bal", S))]
    one = [S for S in with_ours if len(E.of(OURS, S)) == 1]
    nums[f"{g}/one_seed"] = tex(and_list([pt.column_name(S, settings) for S in one])) if one else "none"
    nums[f"{g}/n_one_seed"] = word(len(one))
    nums[f"{g}/nsettings"] = str(len(with_ours))
    nums[f"{g}/nsettings_w"] = word(len(with_ours))
    nums[f"{g}/datasets"] = tex(and_list([pt.column_name(S, with_ours) for S in with_ours])) if with_ours else "none"
    if with_ours and g != "t2":   # the same list with each dataset's citation
        nums[f"{g}/datasets_cite"] = and_list([tex(pt.column_name(S, with_ours)) + "~\\cite{" + DATA_CITE[S[0]] + "}"
                                               for S in with_ours if S[0] in DATA_CITE])
    base = sorted({r["label"] for r in E.runs} - {OURS, "Local"} - NOTEST - REFS)
    nums[f"{g}/nbaselines_w"] = word(len(base))
    nums[f"{g}/nlt_w"] = word(len(set(base) & LT_METHODS))
    if g == "t2":
        ratios = sorted({int(round(1 / S[1])) for S in settings}, reverse=True)
        nums["t2/ratios"] = and_list([str(x) for x in ratios])
    sig = [S for S in with_ours if SIG.get((g, sid(S)))]
    nums[f"{g}/n_sig"] = word(len(sig))   # settings whose margin over the strongest other method is significant (Holm)
    nums[f"{g}/n_sig_of"] = word(sum(1 for S in with_ours if f"{g}/{sid(S)}/strongest_p" in nums))
    nums[f"{g}/nbest"] = str(sum(1 for S in with_ours if ours_best(E, S)))
    nb, ns = int(nums[f"{g}/nbest"]), len(with_ours)   # "all six", "both", "5 of the six", "none of the six"
    nums[f"{g}/nbest_of"] = ("both" if ns == 2 else f"all {word(ns)}") if nb == ns and ns > 1 else \
        (("neither" if ns == 2 else f"none of the {word(ns)}") if nb == 0 else f"{nb} of the {word(ns)}")
    if g == "main":   # generator fidelity: the referee's per-class accuracy on generated features (server.fidelity)
        from settings import class_groups
        rows = []
        for S in settings:
            for m in E.methods:
                for key, what in (("fidelity", "generator"), ("fidelity_mc", "after moment correction")):
                    fs = [r[key] for r in E.of(m, S) if r.get(key)]
                    if not fs or m == HEADS:
                        continue
                    F = np.array(fs)   # seeds x classes, classes in order of decreasing training size
                    hd, md, tl = class_groups(F.shape[1])
                    cells = []
                    for grp, idx in (("all", slice(None)), ("head", hd), ("medium", md), ("tail", tl)):
                        v = F[:, idx].mean(1)
                        cells.append(report.fmt(v.mean(), v.std(ddof=1) if len(v) > 1 else float("nan"),
                                                "final_bal", len(v)))
                        nums[f"{g}/{sid(S)}/{m}/{key}_{grp}"] = tex(cells[-1])
                    rows.append([pt.column_name(S, settings), f"{m} ({what})"] + cells)
        if rows:
            made.append(write_table("fidelity", ["Dataset", "Generator", "All", "Head", "Medium", "Tail"], rows,
                                    set()))
    # tables
    for suffix, key in ACROSS:
        h, rows, bold = across_table(E, settings, key, [m for m in E.methods if m != DIAG], key in LOWER)
        if rows:
            made.append(write_table(f"{g}_{suffix}", h, rows, bold))
    h, rows, bold = paired_table(E, settings)
    if rows:
        made.append(write_table(f"{g}_paired", h, rows, bold))
    h, rows, bold = paired_table(E, settings, "er_bal")   # every method of the pipeline with the readout
    if rows:
        made.append(write_table(f"{g}_er_paired", h, rows, bold))
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
    T = [r for r in ours_runs if r.get("class_sep_F") is not None]
    if T:   # both tracks (the shared ResNet-8 features of track 2 are a different feature space)
        F = [r["class_sep_F"] for r in T]
        nums["gate/F/min_all"], nums["gate/F/max_all"] = f"{min(F):.3f}", f"{max(F):.3f}"
        name = lambda r: DATASETS[r["dataset"]]["name"] + (
            "-LT with one shared model" if DATASETS[r["dataset"]].get("track") == 2 else "")
        rows_all = sorted({name(r) for r in T if r.get("gate_branch") == "rows"})
        nums["gate/rows_all"] = tex(and_list(rows_all)) if rows_all else "none of them"
    labels = {r["label"] for E in allE for r in E.runs}
    published = labels - {OURS, "Local"} - NOTEST - REFS - set(bn.GEFL_VARIANTS) - set(bn.MATCHED_TS)
    nums["count/methods"] = str(len(published))
    nums["count/datasets"] = str(len({r["dataset"].replace("_t2", "") for r in ours_runs}))
    nums["count/datasets_w"] = word(len({r["dataset"].replace("_t2", "") for r in ours_runs}))
    nums["count/methods_w"] = word(len(published))
    E = E_by.get("abl")
    if E is not None:
        for S in ordered(E):
            s = sid(S)
            g, la, us = (mean_of(E, m, "final_bal", S) for m in ("GeFL-F", "GeFL-F + LA", HEADS))   # as trained
            for m in E.methods:   # each variant against the method it changes (FedDF's against FedDF), both ways
                base = "FedDF" if m.startswith("FedDF") else "GeFL-F"
                if m in (base, OURS) or m in NOTEST or m.startswith("FedECS-LT"):
                    continue
                for key, pre in (("final_bal", ""), ("er_bal", "er_")):
                    a, b = mean_of(E, m, key, S), mean_of(E, base, key, S)
                    if not (np.isnan(a) or np.isnan(b)):
                        nums[f"abl/{s}/{m}/{pre}gain"] = signed(100 * (a - b))
                # paired over the seeds both have (uncorrected: one planned comparison per variant)
                both = sorted({r["seed"] for r in E.of(m, S)} & {r["seed"] for r in E.of(base, S)})
                if len(both) >= 2:
                    from scipy import stats
                    for key, pre in (("final_bal", ""), ("final_tail", "tail_"), ("er_bal", "er_")):
                        v, w = ({r["seed"]: r.get(key) for r in E.of(x, S)} for x in (m, base))
                        d = [100 * (v[x] - w[x]) for x in both if v[x] is not None and w[x] is not None]
                        if len(d) >= 2 and np.std(d) > 0:
                            nums[f"abl/{s}/{m}/{pre}paired"] = signed(float(np.mean(d)))
                            nums[f"abl/{s}/{m}/{pre}p"] = f"{float(stats.ttest_1samp(d, 0.0).pvalue):.2g}"
                    nums[f"abl/{s}/{m}/seeds"] = word(len(both))
            for m in MATCHED:   # FedECS-LT's heads against GeFL-F and FedDF with as many synthetic epochs
                o = mean_of(E, m, "final_bal", S)
                if not (np.isnan(o) or np.isnan(us)):
                    nums[f"abl/{s}/heads_vs/{m}"] = signed(100 * (us - o))
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
            d, o = mean_of(E, DIAG, "final_bal", S), mean_of(E, HEADS, "final_bal", S)   # both as trained
            if not (np.isnan(d) or np.isnan(o)):
                nums[f"t2/{sid(S)}/fe200_diff"] = signed(100 * (d - o))
    E = E_by.get("main")
    if E is not None:
        for S in ordered(E):
            s = sid(S)
            for key, short in (("time_s", "time_ratio"), ("comm_mb_per_client", "comm_ratio")):
                for ref, suf in (("GeFL-F", ""), ("Balanced Softmax", "_bsm")):   # Balanced Softmax: no generator
                    a, b = mean_of(E, OURS, key, S), mean_of(E, ref, key, S)
                    if not (np.isnan(a) or np.isnan(b)) and b > 0:
                        nums[f"cost/{s}/{short}{suf}"] = f"{a / b:.2f}" if a / b < 10 else f"{a / b:.0f}"
            if s == "cifar10":
                keys = [k for k in report.SYSTEM if k[0] in ("comm_mb_per_client", "time_s", "fwd_mflops",
                                                           "train_gflops_round")]
                rows, bold = pt.metric_rows(E, S, keys)
                keep = [i for i, r in enumerate(rows) if r[0] != HEADS]   # one run: its cost is FedECS-LT's
                bold = {(keep.index(i), j) for i, j in bold if i in keep}
                rows = [rows[i] for i in keep]
                if rows:
                    made.append(write_table("cost", ["Method"] + [n for _, n in keys], rows, bold))
    E = E_by.get("comp")
    if E is not None:   # components of FedECS-LT: rows = variants, columns = (dataset, as trained / ER)
        sets = ordered(E)
        header = ["Variant"] + [f"{pt.column_name(S, sets)} {w}" for S in sets for w in ("as trained", "ER")]
        rows = []
        for m in [OURS] + [m for m in E.methods if m.startswith("FedECS-LT w") and m not in NOT_IN_PAPER]:
            cells = []
            for S in sets:
                for key in ("final_bal", "er_bal"):
                    key = "heads_final_bal" if (m == OURS and key == "final_bal") else key   # its heads as trained
                    mu, sd, n = plots.metric(E, m, key, S)
                    cells.append(report.fmt(mu, sd, key, n))
            if any(c != "-" for c in cells):
                rows.append([("FedECS-LT (full)" if m == OURS else m.replace("FedECS-LT ", ""))] + cells)
        if rows:
            made.append(write_table("abl_components", header, rows, set()))
        every = [m for m in E.methods if m.startswith("FedECS-LT w")]
        variants = [m for m in every if m not in NOT_IN_PAPER]
        nums["comp/n_variants"] = word(len(variants))
        for S in sets:   # what each change costs the full method (with the readout) and its heads as trained
            s, drop = sid(S), {}
            for m in every:
                for key, ref, pre in (("er_bal", "er_bal", "er_"), ("final_bal", "heads_final_bal", "")):
                    a, b = mean_of(E, m, key, S), mean_of(E, OURS, ref, S)
                    if not (np.isnan(a) or np.isnan(b)):
                        nums[f"comp/{s}/{m.replace('FedECS-LT ', '')}/{pre}drop"] = signed(100 * (b - a))
                        if pre == "er_" and m in variants:
                            drop[m] = 100 * (b - a)
            if drop:   # paired drops with the readout, Holm over the paper's variants (the components' test family)
                import representation as rp
                from scipy import stats
                full = {r["seed"]: r.get("er_bal") for r in E.of(OURS, S)}
                tested = []
                for m in variants:
                    v = {r["seed"]: r.get("er_bal") for r in E.of(m, S)}
                    d = [100 * (full[x] - v[x]) for x in sorted(set(full) & set(v)) if None not in (full[x], v[x])]
                    nums[f"comp/{s}/{m.replace('FedECS-LT ', '')}/seeds"] = word(len(d))
                    if len(d) >= 2 and np.std(d) > 0:
                        tested.append((m, float(stats.ttest_1samp(d, 0.0).pvalue)))
                for (m, p), a in zip(tested, rp.holm([p for _, p in tested]) if tested else []):
                    nums[f"comp/{s}/{m.replace('FedECS-LT ', '')}/er_p"] = f"{a:.2g}"
                fl = float(nums.get("repeat/er/floor", "nan"))   # resolved: above the noise floor and significant
                nums[f"comp/{s}/n_resolved"] = word(sum(1 for (m, _), a in zip(tested, rp.holm([p for _, p in tested]))
                                                       if a < 0.05 and abs(drop[m]) >= fl)) if tested else "none"
                for m in NOT_IN_PAPER:   # for the separate note: paired tests against the full method, cost
                    v = {r["seed"]: r for r in E.of(m, S)}
                    both = sorted(set(full) & set(v))
                    key = f"note/{s}/{m.replace('FedECS-LT ', '')}"
                    nums[f"{key}/seeds"] = word(len(both))
                    for k, ref, pre in (("er_bal", "er_bal", "er_"), ("final_bal", "heads_final_bal", ""),
                                        ("er_tail", "er_tail", "er_tail_")):
                        o = {r["seed"]: r.get(ref) for r in E.of(OURS, S)}
                        d = [100 * (v[x].get(k) - o[x]) for x in both if None not in (v[x].get(k), o[x])]
                        if d:
                            nums[f"{key}/{pre}diff"] = signed(float(np.mean(d)))
                        if len(d) >= 2 and np.std(d) > 0:
                            nums[f"{key}/{pre}p"] = f"{float(stats.ttest_1samp(d, 0.0).pvalue):.2g}"
                    for k in ("comm_mb_per_client", "time_s"):
                        a, b = mean_of(E, m, k, S), mean_of(E, OURS, k, S)
                        if not (np.isnan(a) or np.isnan(b)) and a > 0:
                            nums[f"{key}/{k}"] = f"{a:,.0f}"
                            nums[f"{key}/{k}_ours"] = f"{b:,.0f}"
                            nums[f"{key}/{k}_ratio"] = f"{b / a:.1f}"
                    for k in ("final_bal", "er_bal"):
                        mu, sd, n = plots.metric(E, m, k, S)
                        nums[f"{key}/{k}"] = report.fmt(mu, sd, k, n).replace("±", "$\\pm$")
                    # Holm over all seven changes, the note's family (the six of the paper and this one)
                    allp = []
                    for x in every:
                        w = {r["seed"]: r.get("er_bal") for r in E.of(x, S)}
                        d = [100 * (full[y] - w[y]) for y in sorted(set(full) & set(w)) if None not in (full[y], w[y])]
                        if len(d) >= 2 and np.std(d) > 0:
                            allp.append((x, float(stats.ttest_1samp(d, 0.0).pvalue)))
                    for (x, _), a in zip(allp, rp.holm([p for _, p in allp]) if allp else []):
                        if x == m:
                            nums[f"{key}/er_p_holm"] = f"{a:.2g}"
                nums[f"comp/{s}/n_one_seed"] = word(sum(1 for m in variants if len(
                    {r["seed"] for r in E.of(m, S)} & set(full)) == 1))
                big = max(drop, key=drop.get)
                nums[f"comp/{s}/largest"] = tex(big.replace("FedECS-LT ", ""))
                nums[f"comp/{s}/largest_drop"] = signed(drop[big])
                fl = nums.get("repeat/er/floor")
                if fl:
                    nums[f"comp/{s}/n_small"] = word(sum(1 for v in drop.values() if abs(v) < float(fl)))
    E = E_by.get("sweep")
    if E is not None:   # robustness: rows = methods, columns = the changed setting
        sets = ordered(E)

        def label(S):
            IF, alpha, K = S[1], S[2], S[3]
            return (f"Dirichlet {alpha}" if alpha != 0.5 else f"K = {K}" if K != 10 else
                    f"ratio {int(round(1 / IF))}" if IF else "natural")
        header = ["Method"] + [label(S) for S in sets]
        rows = []
        for m in bn.SWEEP_METHODS[:1] + [HEADS] + bn.SWEEP_METHODS[1:]:
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
            r = next(r for r in E.runs if plots.setting_key(r) == S and r["label"] not in REFS)   # not the pooled split
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
            cells = {}   # (method) -> {(seed, architecture): leave-one-out 10-NN accuracy of the embeddings}
            for m in (OURS, "GeFL-F", "FedDF"):
                cells[m] = {}
                for r in E.of(m):
                    d = E.pred(r)
                    if d is None or "y" not in d.files:
                        continue
                    for k in sorted(k for k in d.files if k.startswith("embed_")):   # every architecture
                        cells[m][(r["seed"], k)] = rp.knn_correct(d[k].astype(np.float64), d["y"]).mean()
                if cells[m]:
                    nums[f"repr/cifar10/{m}/knn"] = tex(f"{100 * np.mean(list(cells[m].values())):.1f}%")
            for m in ("GeFL-F", "FedDF"):
                common = sorted(set(cells.get(OURS, {})) & set(cells.get(m, {})))
                if common:   # architecture x seed cells in which FedECS-LT's embeddings are better
                    nums[f"repr/cifar10/knn_wins/{m}"] = str(sum(cells[OURS][c] > cells[m][c] for c in common))
                    nums["repr/cifar10/knn_cells"] = str(len(common))


# figure slots drawn here from runs.jsonl (no results zip needed): written on every fill, removed when their runs are gone
RUN_FIGS = ("fig_components", "fig_variants", "fig_clients")


def bar_panels(name, panels, series, xlabel, status, made, height_per_row=0.2):
    """Horizontal grouped bars, one panel per dataset. panels: [(dataset label, [row label], {series: [(mean, sd)]})];
    series: [(series label, colour)]. Saved with the paper's plot style and layout check."""
    import plot_style as ps
    ps.apply()
    order = list(dict.fromkeys(lab for _, labs, _ in panels for lab in labs))   # the same rows in every panel
    nan = (float("nan"), float("nan"))
    panels = [(ds, order, {sl: [dict(zip(labs, v)).get(lab, nan) for lab in order] for sl, v in vals.items()})
              for ds, labs, vals in panels]
    rows = len(order)
    fig, axes = ps.figure("page", height_per_row * rows * len(series) / 2 + 1.1, ncols=len(panels), sharey=False)
    axes = np.atleast_1d(axes)
    data = []
    for i, (ax, (ds, labels, vals)) in enumerate(zip(axes, panels)):
        y = np.arange(len(labels))
        h = 0.8 / len(series)
        for j, (sl, col) in enumerate(series):
            mu = [0.0 if np.isnan(v[0]) else v[0] for v in vals[sl]]   # a method without runs here: no bar
            sd = [0 if np.isnan(v[1]) else v[1] for v in vals[sl]]
            ax.barh(y + (j - (len(series) - 1) / 2) * h, mu, h, xerr=sd, color=col, label=sl, edgecolor="black",
                    linewidth=0.3, error_kw=dict(lw=0.6))
            data += [[ds, lab, sl, m, s_] for lab, m, s_ in zip(labels, mu, sd)]
        ax.set_yticks(y)
        ax.set_yticklabels(labels if i == 0 else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_xlabel(f"{ds}: {xlabel}")
        ax.axvline(0, color="black", lw=0.5)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5 * len(panels) + 0.1 * (len(panels) - 1), -0.18),
                   ncol=len(series))
    before = len(ps.LAYOUT_PROBLEMS)
    ps.save(fig, os.path.join(GEN, "figures"), name, data, ["dataset", "row", "series", "mean", "sd"])
    for p_ in ps.LAYOUT_PROBLEMS[before:]:
        status.append(f"- `{name}`: layout problem {p_[1]}")
    made.append(name)


def fill_run_figures(E_by, made, status):
    import plot_style as ps
    out = os.path.join(GEN, "figures")
    for name in RUN_FIGS:   # a slot whose runs are not (or no longer) there must not keep an old file
        for ext in (".pdf", ".png", ".csv"):
            if os.path.exists(os.path.join(out, name + ext)):
                os.remove(os.path.join(out, name + ext))
    TRAINED, READOUT = ("heads as trained", ps.PALETTE[8]), ("with the exact readout", ps.PALETTE[0])

    def ms(v):
        v = [x for x in v if x is not None and not np.isnan(x)]
        return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")) if v else (np.nan, np.nan)
    E = E_by.get("comp")
    if E is not None:   # what each change costs the full method, paired over seeds
        panels = []
        for S in ordered(E):
            full = {r["seed"]: r for r in E.of(OURS, S)}
            var = [m for m in E.methods if m.startswith("FedECS-LT w") and m not in NOT_IN_PAPER]
            vals = {TRAINED[0]: [], READOUT[0]: []}
            keep = []
            for m in var:
                rs = {r["seed"]: r for r in E.of(m, S)}
                common = sorted(set(full) & set(rs))
                if not common:
                    continue
                keep.append(m.replace("FedECS-LT ", ""))
                vals[TRAINED[0]].append(ms([100 * (full[x].get("heads_final_bal", np.nan) - rs[x]["final_bal"])
                                            for x in common]))
                vals[READOUT[0]].append(ms([100 * (full[x]["er_bal"] - rs[x].get("er_bal", np.nan)) for x in common]))
            if keep:
                panels.append((pt.column_name(S, ordered(E)), keep, vals))
        if panels:
            bar_panels("fig_components", panels, [TRAINED, READOUT], "Drop in balanced accuracy (points)", status, made)
    E = E_by.get("abl")
    if E is not None:   # variants of GeFL-F and FedDF, as trained and with the readout
        panels = []
        for S in ordered(E):
            labels, vals = [], {TRAINED[0]: [], READOUT[0]: []}
            for m in E.methods:
                if m in NOTEST or m == "Local" or not E.of(m, S):
                    continue
                labels.append(m)
                tk = "heads_final_bal" if m == OURS else "final_bal"   # FedECS-LT: its heads, then its readout
                vals[TRAINED[0]].append(ms([100 * r.get(tk, np.nan) for r in E.of(m, S)]))
                vals[READOUT[0]].append(ms([100 * r.get("er_bal", np.nan) for r in E.of(m, S)]))
            if labels:
                panels.append((pt.column_name(S, ordered(E)), labels, vals))
        if panels:
            bar_panels("fig_variants", panels, [TRAINED, READOUT], "Balanced accuracy (%)", status, made)
    E10, E50 = E_by.get("main"), E_by.get("k50")
    # every method at 10 and at 50 clients (FedECS-LT with its readout), once FedECS-LT has 50-client runs
    if E10 is not None and E50 is not None and all(E50.of(OURS, S) for S in ordered(E50)):
        panels = []
        for S50 in ordered(E50):
            S10 = next((S for S in E10.settings if S[0] == S50[0]), None)
            if S10 is None:
                continue
            labels = [m for m in E50.methods if m not in NOTEST and m not in REFS and E50.of(m, S50) and E10.of(m, S10)]
            if not labels:
                continue
            vals = {"10 clients": [ms([100 * r["final_bal"] for r in E10.of(m, S10)]) for m in labels],
                    "50 clients": [ms([100 * r["final_bal"] for r in E50.of(m, S50)]) for m in labels]}
            panels.append((pt.column_name(S50, ordered(E50)), labels, vals))
        if panels:
            bar_panels("fig_clients", panels, [("10 clients", ps.PALETTE[8]), ("50 clients", ps.PALETTE[1])],
                       "Balanced accuracy (%)", status, made)


def content_files():
    return [p for p in glob.glob(os.path.join(HERE, "content", "*.tex")) if not p.endswith("macros.tex")]


def missing_notebooks():
    """Notebooks with runs that are in no results folder (by run id, as build_notebooks.rerun_notebooks counts them:
    a notebook can be complete through another notebook's or a rerun's folder), with their missing run counts."""
    from runner import run_id
    done = set()
    for f in glob.glob(os.path.join(RESULTS, "*", "runs.jsonl")):
        if not os.path.basename(os.path.dirname(f)).startswith(("NB00", "_")):
            done |= {json.loads(line)["run_id"] for line in open(f) if line.strip()}
    out = {}
    for n, nb in bn.all_notebooks().items():
        miss = sum(run_id(r) not in done for r in nb["runs"])
        if miss and not n.startswith("NB00"):
            out.setdefault(bn.FOLDERS[bn.priority(n)], []).append(f"{n} ({miss} of {len(nb['runs'])} runs missing)")
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
        g, k50 = group_of(r), k50_setting(r)   # decided on the run as recorded, before the readout relabelling
        for x in with_readout(r):
            if g:
                by.setdefault(g, []).append(x)
            if k50:   # also in `sweep` (the robustness table's K=50 column) for the sweep methods
                by.setdefault("k50", []).append(x)
    # the ablation table also needs the references of the same datasets (same splits and seeds)
    if "abl" in by:
        ds = {r["dataset"] for r in by["abl"]}
        by["abl"] += [r for r in by.get("main", []) if r["dataset"] in ds and r["label"] in (OURS, HEADS, "GeFL-F", "FedDF")]
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
        for g in ("main", "nat", "t2", "abl", "comp", "sweep", "k50", "fresh"):
            E_by[g] = exp_of(by[g], tmp, g) if by.get(g) else None
            if E_by[g] is not None and g in ("main", "nat", "t2", "k50", "fresh"):
                fill_group(g, E_by[g], nums, made, status)
        repeat_numbers(nums)   # first: the noise floor is used by fill_extras
        fill_extras(E_by, nums, made)
        fill_run_figures(E_by, made, status)
    reproduction_numbers(nums)
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
