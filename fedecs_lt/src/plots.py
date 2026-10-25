"""All figures of one experiment (action plan sections 15 and 17b), in IEEE format (plot_style), each with a formal
caption (captions.py) and a CSV of its data. A figure whose data the experiment does not contain (e.g. privacy plots in
an experiment without DP runs) is listed as not applicable, never silently dropped.

Catalogue (numbering fixed here and in docs/review/action_plan.md section 15):
 A  training curves        1 bal. acc. vs round (Stage III)   2 Stage I curves      3 generator loss      4 head losses
                           5 macro-F1 and tail recall vs round 6 ECE vs round         7 consensus entropy   8 client drift
 B  class imbalance        9 per-class recall   10 recall vs class size   11 confusion matrices   12 per-class precision
                           13 per-class F1      14 head/medium/tail recall 15 precision-recall curves (rare classes)
                           16 reliability diagram 17 class-wise ECE      18 worst-class recall and G-mean
 C  representation         19 t-SNE embeddings (ours) 20 t-SNE embeddings (GeFL-F) 21 PCA embeddings
                           22 t-SNE real vs generated (ours) 23 t-SNE real vs generated (GeFL-F)
                           24 real vs generated class means 25 classifier norm per class 26 classifier cosine similarity
                           27 neural-collapse statistics
 D  generator quality      28 per-class fidelity 29 spread ratio 30 class-row norms 31 fidelity over rounds
                           32 moment-correction scales
 E  mechanism              33 class-row norm over rounds 34 holder gain vs non-holder shrink 35 row norm vs holders
                           36 Adam second moment of the rows 37 readout gap
 F  comparison             38 balanced accuracy bars 39 forest plot 40 critical-difference diagram 41 radar chart
                           42 accuracy vs communication 43 accuracy vs time 44 metric heatmap 45 ablation waterfall
                           46 sensitivity lines
 G  federated system       47 communication by message type 48 time per stage 49 GPU memory 50 participation
                           51 accuracy vs cumulative communication
 H  privacy                52 accuracy vs DP epsilon 53 membership-inference ROC 54 release noise vs class size
 I  diagram                55 pipeline diagram
 17b aggregation effect    56 aggregation effect per class over rounds 57 fidelity before vs after aggregation
                           58 row norms of holders' copies, non-holders' copies and the average 59 effect vs holders
                           60 federated vs local-only per class
"""
import glob
import json
import os

import numpy as np

import captions as cap
import plot_style as ps
from settings import DATASETS, class_groups

OURS = ps.OURS
REF = "GeFL-F"


# ---------------------------------------------------------------- loading
class Exp:
    """Everything one experiment produced: runs (one record per run), per-round logs and saved predictions."""

    def __init__(self, res_dir):
        self.dir = res_dir
        self.runs = []
        p = os.path.join(res_dir, "runs.jsonl")
        if os.path.exists(p):
            for line in open(p):
                try:
                    self.runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        self.logs = {}
        for f in glob.glob(os.path.join(res_dir, "logs", "*.jsonl")):
            recs = []
            for line in open(f):
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            if recs and recs[0].get("event") == "run_start":
                self.logs[recs[0]["run_id"]] = recs
        self.methods = [m for m in ps.ORDER if any(r["label"] == m for r in self.runs)]
        self.methods += sorted({r["label"] for r in self.runs} - set(self.methods))
        self.settings = sorted({setting_key(r) for r in self.runs})

    def of(self, label, setting=None):
        return [r for r in self.runs if r["label"] == label and (setting is None or setting_key(r) == setting)]

    def log(self, run, event):
        return [e for e in self.logs.get(run["run_id"], []) if e["event"] == event]

    def pred(self, run):
        safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in run["run_id"])
        p = os.path.join(self.dir, "predictions", safe + ".npz")
        return np.load(p) if os.path.exists(p) else None


def setting_key(r):
    return (r["dataset"], r["IF"], r["alpha"], r["K"], r.get("pfrac", 1.0), bool(r.get("tail_perm", False)))


def ms(vals):
    v = np.array([x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))], dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v)


def metric(E, label, key, setting):
    return ms([r.get(key) for r in E.of(label, setting)])


# ---------------------------------------------------------------- figure registry
FIGS = []


def fig(num, group):
    def deco(fn):
        FIGS.append((num, group, fn))
        return fn
    return deco


def _curve(E, label, setting, event, getter):
    """Mean over seeds of a per-round quantity: rounds, mean, sd."""
    per = []
    for r in E.of(label, setting):
        pts = [(e["round"], getter(e)) for e in E.log(r, event)]
        pts = [(a, b) for a, b in pts if b is not None]
        if pts:
            per.append(dict(pts))
    if not per:
        return None
    rounds = sorted(set.intersection(*[set(p) for p in per]))
    if not rounds:
        return None
    vals = np.array([[p[x] for x in rounds] for p in per], dtype=float)
    return np.array(rounds), vals.mean(0), vals.std(0, ddof=1) if len(per) > 1 else np.zeros(len(rounds))


def _one_client_per_arch(r):
    """True when every head architecture has a single client (K <= 10 with ten architectures): aggregating a head
    then averages one copy and leaves it unchanged, so the aggregation-effect figures say nothing."""
    from settings import DATASETS, N_ARCH
    return not DATASETS.get(r["dataset"], {}).get("single_arch") and r["K"] <= N_ARCH


def _round_ticks(ax, rounds, n=6):
    """Heatmap columns are evaluation rounds (not evenly spaced: round 1, then every 5th): ticks at the column
    centres, labelled with the round numbers."""
    idx = sorted(set(np.linspace(0, len(rounds) - 1, min(n, len(rounds))).round().astype(int)))
    ax.set_xticks(idx)
    ax.set_xticklabels([str(rounds[i]) for i in idx])


def _lines(E, S, out, num, event, getter, ylabel, name, what, scale=100.0, methods=None, width="column",
           split="Test-set"):
    f, ax = ps.figure(width, 2.4)
    rows = []
    for m in methods or E.methods:
        c = _curve(E, m, S, event, getter)
        if c is None:
            continue
        x, mu, sd = c
        st = ps.style(m)
        ax.plot(x, mu * scale, label=m, markevery=max(1, len(x) // 6), **st)
        ax.fill_between(x, (mu - sd) * scale, (mu + sd) * scale, color=st["color"], alpha=0.12, linewidth=0)
        rows += [[m, int(a), float(b * scale), float(s * scale)] for a, b, s in zip(x, mu, sd)]
    if not rows:
        return []
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=1 if len(set(r[0] for r in rows)) < 6 else 2)
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["method", "round", "mean", "sd"])
    return [(p, cap.curve(num, E, S, what, split))]


@fig(1, "A")
def f01(E, S, out):
    return _lines(E, S, out, 1, "stage3_round", lambda e: (e.get("post") or {}).get("bal"),
                  "Balanced accuracy (%)", "balanced_accuracy_rounds", "balanced accuracy of the aggregated heads "
                  "during head training (Stage III)", width="page")


@fig(2, "A")
def f02(E, S, out):
    return _lines(E, S, out, 2, "stage1_round", lambda e: (e.get("post") or {}).get("bal"), "Balanced accuracy (%)",
                  "stage1_rounds", "balanced accuracy of the shared feature extractor with the architecture heads "
                  "during Stage I", methods=[m for m in E.methods if m in (OURS,)] or E.methods[:1])


@fig(3, "A")
def f03(E, S, out):
    return _lines(E, S, out, 3, "stage2_round", lambda e: np.mean(list(e["loss"].values())) if e.get("loss") else None,
                  "Generator loss per sample", "generator_loss", "Mean training loss of the clients' generator copies "
                  "(reconstruction + KL) during Stage II", scale=1.0, split=None)


@fig(4, "A")
def f04(E, S, out):
    a = _lines(E, S, out, 4, "stage3_round",
               lambda e: np.mean([v for v in e["real_loss"].values() if v is not None]) if e.get("real_loss") else None,
               "Real-phase loss", "head_real_loss", "Mean real-feature training loss of the clients' heads during "
               "Stage III", scale=1.0, split=None)
    return a


@fig(5, "A")
def f05(E, S, out):
    return _lines(E, S, out, 5, "stage3_round", lambda e: (e.get("post") or {}).get("tail"), "Tail-class recall (%)",
                  "tail_recall_rounds", "mean recall of the tail classes (the rarest 40%) of the aggregated heads "
                  "during Stage III")


@fig(6, "A")
def f06(E, S, out):
    return _lines(E, S, out, 6, "stage3_round", lambda e: (e.get("post") or {}).get("ece"),
                  "Expected calibration error (%)", "ece_rounds", "expected calibration error (15 bins) of the "
                  "aggregated heads during Stage III")


@fig(7, "A")
def f07(E, S, out):
    return _lines(E, S, out, 7, "stage3_round", lambda e: e.get("consensus_entropy"), "Consensus entropy (nats)",
                  "consensus_entropy", "Mean entropy of the consensus soft labels on the round's generated features",
                  scale=1.0, methods=[OURS], split=None)


@fig(8, "A")
def f08(E, S, out):
    return _lines(E, S, out, 8, "stage1_round", lambda e: float(np.mean(e["drift"])) if e.get("drift") else None,
                  "Client drift (L2)", "client_drift", "Mean L2 distance of the clients' feature extractors from "
                  "their average after each Stage I round (client drift)", scale=1.0,
                  methods=E.methods[:1], split=None)


# ---------------------------------------------------------------- B: class imbalance
def _per_class(E, S, key, methods):
    mat = {}
    for m in methods:
        rs = [r.get(key) for r in E.of(m, S) if r.get(key)]
        if rs:
            mat[m] = np.array(rs, dtype=float)
    return mat


def _class_lines(E, S, out, num, key, ylabel, name, what, scale=100.0, ref=None, prob_metric=False):
    """One line per method over the classes (most to least frequent). ref = (label, values) adds a grey reference
    line (e.g. the true class share)."""
    mat = _per_class(E, S, key, E.methods)
    if not mat:
        return []
    C = len(next(iter(mat.values()))[0])
    f, ax = ps.figure("page", 2.6)
    rows = []
    x = np.arange(C)
    for m, v in mat.items():
        mu = np.nanmean(v, 0) * scale
        sd = np.nanstd(v, 0, ddof=1) * scale if len(v) > 1 else np.zeros(C)
        st = ps.style(m)
        ax.plot(x, mu, label=m, markevery=max(1, C // 12), **st)
        rows += [[m, int(c), float(a), float(b)] for c, a, b in zip(x, mu, sd)]
    if ref is not None:
        ax.plot(x, np.asarray(ref[1]) * scale, color="0.5", linestyle=(0, (1, 1)), linewidth=1.2, label=ref[0])
        rows += [[ref[0], int(c), float(a), 0.0] for c, a in zip(x, np.asarray(ref[1]) * scale)]
    ax.set_xlabel("Class (ordered from most to least frequent)")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, loc="best")
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["method", "class", "mean", "sd"])
    return [(p, cap.per_class(num, E, S, what, note=cap.proto_note(mat) if prob_metric else ""))]


@fig(9, "B")
def f09(E, S, out):
    return _class_lines(E, S, out, 9, "final_recall_c", "Recall (%)", "per_class_recall", "recall")


@fig(12, "B")
def f12(E, S, out):
    return _class_lines(E, S, out, 12, "final_precision_c", "Precision (%)", "per_class_precision", "precision")


@fig(13, "B")
def f13(E, S, out):
    return _class_lines(E, S, out, 13, "final_f1_c", "F1 score (%)", "per_class_f1", "F1 score")


# every other metric with a per-class form (metrics.PER_CLASS), one figure each
@fig(61, "B")
def f61(E, S, out):
    return _class_lines(E, S, out, 61, "final_ap_c", "Average precision (%)", "per_class_auc_pr",
                        "average precision (area under the precision-recall curve)")


@fig(62, "B")
def f62(E, S, out):
    return _class_lines(E, S, out, 62, "final_auc_c", "AUC-ROC (%)", "per_class_auc_roc",
                        "area under the ROC curve of the class against all others")


@fig(63, "B")
def f63(E, S, out):
    return _class_lines(E, S, out, 63, "final_mcc_c", "MCC (%)", "per_class_mcc",
                        "Matthews correlation coefficient of the class against all others")


@fig(64, "B")
def f64(E, S, out):
    return _class_lines(E, S, out, 64, "final_ece_c", "Calibration error (%)", "per_class_calibration",
                        "calibration error of the predicted probability of the class (15 bins; lower is better)",
                        prob_metric=True)


@fig(65, "B")
def f65(E, S, out):
    return _class_lines(E, S, out, 65, "final_nll_c", "NLL (nats)", "per_class_nll",
                        "negative log-likelihood on the test images of the class (lower is better)", scale=1.0,
                        prob_metric=True)


@fig(66, "B")
def f66(E, S, out):
    return _class_lines(E, S, out, 66, "final_brier_c", "Brier score", "per_class_brier",
                        "Brier score on the test images of the class (lower is better)", scale=1.0,
                        prob_metric=True)


@fig(67, "B")
def f67(E, S, out):
    return _class_lines(E, S, out, 67, "final_top5_c", "Top-5 recall (%)", "per_class_top5",
                        "top-5 recall (true class among the five most probable)")


@fig(68, "B")
def f68(E, S, out):
    true = None
    for r in E.runs:
        if setting_key(r) == S and r.get("final_confusion"):
            cm = np.array(r["final_confusion"], dtype=float)
            true = cm.sum(1) / cm.sum()
            break
    return _class_lines(E, S, out, 68, "final_pred_share_c", "Share of predictions (%)", "per_class_prediction_share",
                        "share of all test predictions that go to the class; the dotted grey line is the class's "
                        "true share of the test set (a line above it: the class is over-predicted)",
                        ref=("Test set (true share)", true) if true is not None else None)


@fig(69, "B")
def f69(E, S, out):
    return _class_lines(E, S, out, 69, "er_recall_c", "Recall, exact readout (%)", "per_class_recall_exact_readout",
                        "recall after the exact class-balanced readout")


@fig(70, "B")
def f70(E, S, out):
    """Per-class recall of the aggregated heads at every evaluation round of Stage III (FedECS-LT and GeFL-F)."""
    res = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        rs = E.of(m, S)
        if not rs:
            continue
        ev = [e for e in E.log(rs[0], "stage3_round") if (e.get("post") or {}).get("recall_c")]
        if len(ev) < 2:
            continue
        A = np.array([e["post"]["recall_c"] for e in ev], dtype=float).T * 100
        rr = [e["round"] for e in ev]
        f, ax = ps.figure("column", 2.6)
        im = ax.imshow(A, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=100)
        _round_ticks(ax, rr)
        ax.set_xlabel("Round")
        ax.set_ylabel("Class (0 = most frequent)")
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Recall (%)")
        name = "".join(ch for ch in m if ch.isalnum()).lower()
        p = ps.save(f, out, f"fig_70_per_class_recall_rounds_{name}",
                    [[rr[j], c, float(A[c, j])] for c in range(A.shape[0]) for j in range(A.shape[1])],
                    ["round", "class", "recall"])
        res.append((p, cap.simple(70, E, S, f"Recall of every class at each evaluation round of head training "
                                            f"(Stage III) for {cap.mname(m)}: aggregated heads, test set, first seed. "
                                            "Every per-class metric of every evaluation round is in the run logs.")))
    return res


# ---------------------------------------------------------------- Track 2: over all communication rounds
def _global_series(E, r):
    """(round, evaluation) pairs of a run on one round axis: LongTailLib's own evaluations (every 5th round); for the
    suite's pipeline, Stage I's evaluations (extractor and heads) and Stage III's, the latter counted after the T_FE
    rounds of Stage I (the generator stage in between is not a training round of the classifier)."""
    out = []
    if r["method"].get("pipeline") == "longtaillib":
        out = [(e["round"], e["post"]) for e in E.log(r, "round_eval") if e.get("post")]
    else:
        T_FE = (r.get("rounds") or {}).get("T_FE", 0)
        out = [(e["round"], e["post"]) for e in E.log(r, "stage1_round") if e.get("post")]
        out += [(T_FE + e["round"], e["post"]) for e in E.log(r, "stage3_round") if e.get("post")]
    return sorted(out, key=lambda t: t[0])


def _track2(S):
    return DATASETS.get(S[0], {}).get("track") == 2


def _global_lines(E, S, out, num, key, ylabel, name, what):
    if not _track2(S):
        return []
    f, ax = ps.figure("page", 2.8)
    rows = []
    for m in E.methods:
        per = []
        for r in E.of(m, S):
            pts = {g: p.get(key) for g, p in _global_series(E, r) if p.get(key) is not None}
            if pts:
                per.append(pts)
        if not per:
            continue
        rr = sorted(set.intersection(*[set(p) for p in per]))
        if not rr:
            continue
        V = np.array([[p[g] for g in rr] for p in per], dtype=float) * 100
        mu, sd = V.mean(0), (V.std(0, ddof=1) if len(V) > 1 else np.zeros(len(rr)))
        st = ps.style(m)
        ax.plot(rr, mu, label=m, markevery=max(1, len(rr) // 8), **st)
        ax.fill_between(rr, mu - sd, mu + sd, color=st["color"], alpha=0.12, linewidth=0)
        rows += [[m, int(g), float(a), float(b)] for g, a, b in zip(rr, mu, sd)]
    if not rows:
        return []
    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3)
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["method", "round", "mean", "sd"])
    return [(p, cap.simple(num, E, S, f"Test-set {what} of the global model over the communication rounds, mean over "
                                      "seeds (band: one standard deviation). LongTailLib methods: every 5th round; "
                                      "FedECS-LT: its extractor stage, then its head stage counted after the "
                                      "extractor's rounds (its generator stage in between is not shown)."))]


@fig(71, "F")
def f71(E, S, out):
    return _global_lines(E, S, out, 71, "bal", "Balanced accuracy (%)", "track2_balanced_accuracy_rounds",
                         "balanced accuracy")


@fig(72, "F")
def f72(E, S, out):
    return _global_lines(E, S, out, 72, "tail", "Tail-class recall (%)", "track2_tail_recall_rounds",
                         "recall of the tail classes (the rarest 40%)")


@fig(73, "F")
def f73(E, S, out):
    """Per-class recall over the rounds, one heatmap per method (Track 2; first seed)."""
    if not _track2(S):
        return []
    res = []
    for m in E.methods:
        rs = E.of(m, S)
        if not rs:
            continue
        ser = [(g, p["recall_c"]) for g, p in _global_series(E, rs[0]) if p.get("recall_c")]
        if len(ser) < 2:
            continue
        A = np.array([v for _, v in ser], dtype=float).T * 100
        rr = [g for g, _ in ser]
        f, ax = ps.figure("column", 2.6)
        im = ax.imshow(A, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=100)
        _round_ticks(ax, rr)
        ax.set_xlabel("Round")
        ax.set_ylabel("Class (0 = most frequent)")
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Recall (%)")
        name = "".join(ch for ch in m if ch.isalnum()).lower()
        p = ps.save(f, out, f"fig_73_track2_per_class_recall_rounds_{name}",
                    [[rr[j], c, float(A[c, j])] for c in range(A.shape[0]) for j in range(A.shape[1])],
                    ["round", "class", "recall"])
        res.append((p, cap.simple(73, E, S, f"Recall of every class of {cap.mname(m)}'s global model at each "
                                            "evaluation over the communication rounds (test set, first seed). Every "
                                            "per-class metric of every evaluation is in the run logs.")))
    return res


@fig(10, "B")
def f10(E, S, out):
    f, ax = ps.figure("column", 2.4)
    rows = []
    for m in E.methods:
        rs = E.of(m, S)
        if not rs or not rs[0].get("final_recall_c"):
            continue
        n = np.array(rs[0]["client_counts"]).sum(0)
        rec = np.mean([r["final_recall_c"] for r in rs], 0) * 100
        st = ps.style(m)
        ax.scatter(n, rec, s=8, marker=st["marker"], color=st["color"], label=m, linewidths=0.4)
        rows += [[m, int(a), float(b)] for a, b in zip(n, rec)]
    if not rows:
        return []
    ax.set_xscale("log")
    ax.set_xlabel("Training images of the class (federation)")
    ax.set_ylabel("Recall (%)")
    ax.legend(ncol=2)
    p = ps.save(f, out, "fig_10_recall_vs_class_size", rows, ["method", "class_size", "recall"])
    return [(p, cap.simple(10, E, S, "Per-class recall against the number of training images of the class in the "
                                     "federation (logarithmic axis); each point is one class, mean over seeds."))]


@fig(11, "B")
def f11(E, S, out):
    res = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        cms = [np.array(r["final_confusion"], dtype=float) for r in E.of(m, S) if r.get("final_confusion")]
        if not cms:
            continue
        cm = np.sum(cms, 0)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        C = len(cm)
        f, ax = ps.figure("column", 3.0)
        im = ax.imshow(cmn * 100, vmin=0, vmax=100, cmap="viridis")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        if C <= 20:
            ax.set_xticks(range(C))
            ax.set_yticks(range(C))
        cb = f.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Share of the true class (%)")
        name = "ours" if m == OURS else "geflf"
        rows = [[i, j, float(cmn[i, j])] for i in range(C) for j in range(C)]
        p = ps.save(f, out, f"fig_11_confusion_{name}", rows, ["true", "pred", "share"], raster=C > 50)
        res.append((p, cap.simple(11, E, S, f"Row-normalised confusion matrix of {cap.mname(m)} on the test set "
                                            "(all client models, summed over seeds); classes ordered from most "
                                            "to least frequent.")))
    return res


@fig(14, "B")
def f14(E, S, out):
    rows = []
    for m in E.methods:
        vals = [metric(E, m, f"final_{g}", S) for g in ("head", "medium", "tail")]
        if np.isnan(vals[0][0]):
            continue
        rows.append([m] + [v[0] * 100 for v in vals] + [v[1] * 100 for v in vals])
    if not rows:
        return []
    f, ax = ps.figure("page", 2.6)
    w = 0.8 / 3
    x = np.arange(len(rows))
    for i, (g, hatch) in enumerate(zip(("Head", "Medium", "Tail"), ("", "///", "..."))):
        ax.bar(x + (i - 1) * w, [r[1 + i] for r in rows], w, yerr=[r[4 + i] for r in rows], label=g, hatch=hatch,
               color=["#BBBBBB", "#777777", "#333333"][i], edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("Recall (%)")
    ax.legend(ncol=3)
    p = ps.save(f, out, "fig_14_head_medium_tail", rows, ["method", "head", "medium", "tail", "sd_head", "sd_medium",
                                                         "sd_tail"])
    return [(p, cap.simple(14, E, S, "Mean recall of the head (most frequent 30%), medium (next 30%) and tail "
                                     "(rarest 40%) classes; mean over seeds, error bars one standard deviation."))]


def _probs(E, m, S):
    """Test-set probabilities of every architecture's head (all seeds), with labels."""
    P, Y = [], []
    for r in E.of(m, S):
        d = E.pred(r)
        if d is None:
            continue
        for k in d.files:
            if k.startswith("probs_"):
                P.append(d[k].astype(np.float64))
                Y.append(d["y_full"] if "y_full" in d.files else None)
    return P, Y


@fig(15, "B")
def f15(E, S, out):
    from sklearn.metrics import precision_recall_curve
    res = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        rs = E.of(m, S)
        d = E.pred(rs[0]) if rs else None
        if d is None or "y_full" not in d.files:
            continue
        y = d["y_full"]
        C = int(y.max()) + 1
        probs = np.mean([d[k].astype(np.float64) for k in d.files if k.startswith("probs_")], 0)
        tail = list(range(C))[class_groups(C)[2]][-3:]
        f, ax = ps.figure("column", 2.4)
        rows = []
        for i, c in enumerate(tail):
            pr, rc, _ = precision_recall_curve(y == c, probs[:, c])
            ax.plot(rc * 100, pr * 100, linestyle=ps.DASHES[i], color=ps.PALETTE[i], label=f"Class {c}")
            rows += [[c, float(a), float(b)] for a, b in zip(rc[::max(1, len(rc) // 200)], pr[::max(1, len(pr) // 200)])]
        ax.set_xlabel("Recall (%)")
        ax.set_ylabel("Precision (%)")
        ax.legend()
        name = "ours" if m == OURS else "geflf"
        p = ps.save(f, out, f"fig_15_pr_curves_{name}", rows, ["class", "recall", "precision"])
        res.append((p, cap.simple(15, E, S, f"Precision-recall curves of the three rarest classes for {cap.mname(m)} "
                                            "(averaged softmax of all client models, first seed).")))
    return res


@fig(16, "B")
def f16(E, S, out):
    f, ax = ps.figure("column", 2.6)
    rows = []
    for m in [x for x in (OURS, REF, "GeFL-F + AdamW + LA") if x in E.methods]:
        rs = E.of(m, S)
        d = E.pred(rs[0]) if rs else None
        if d is None or "y_full" not in d.files:
            continue
        y = d["y_full"]
        conf_all, acc_all = [], []
        for k in d.files:
            if k.startswith("probs_"):
                p = d[k].astype(np.float64)
                conf_all.append(p.max(1))
                acc_all.append(p.argmax(1) == y)
        conf, acc = np.concatenate(conf_all), np.concatenate(acc_all)
        edges = np.linspace(0, 1, 16)
        mids, accs = [], []
        for i in range(15):
            sel = (conf > edges[i]) & (conf <= edges[i + 1])
            if sel.sum() > 20:
                mids.append(conf[sel].mean())
                accs.append(acc[sel].mean())
        ax.plot(np.array(mids) * 100, np.array(accs) * 100, label=m, **ps.style(m))
        rows += [[m, float(a), float(b)] for a, b in zip(mids, accs)]
    if not rows:
        return []
    ax.plot([0, 100], [0, 100], color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Confidence (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    p = ps.save(f, out, "fig_16_reliability", rows, ["method", "confidence", "accuracy"])
    return [(p, cap.simple(16, E, S, "Reliability diagram: accuracy against confidence in 15 equal-width bins "
                                     "(bins with more than 20 test images); the dotted line is perfect calibration."))]


@fig(17, "B")
def f17(E, S, out):
    rows = []
    for m in E.methods:
        v = metric(E, m, "final_cw_ece", S)
        if not np.isnan(v[0]):
            rows.append([m, v[0] * 100, v[1] * 100])
    return _bars(E, S, out, 17, rows, "Class-wise ECE (%)", "classwise_ece",
                 "Class-wise expected calibration error (mean over classes of the calibration error of p(c|x)); mean "
                 "over seeds, error bars one standard deviation.")


@fig(18, "B")
def f18(E, S, out):
    res = []
    for key, lab, name, txt in (("final_worst", "Worst-class recall (%)", "worst_class", "Recall of the worst class"),
                                ("final_gmean", "G-mean of recalls (%)", "gmean",
                                 "Geometric mean of the class recalls (each floored at 1%)")):
        rows = []
        for m in E.methods:
            v = metric(E, m, key, S)
            if not np.isnan(v[0]):
                rows.append([m, v[0] * 100, v[1] * 100])
        res += _bars(E, S, out, 18, rows, lab, name, txt + "; mean over seeds, error bars one standard deviation.")
    return res


def _bars(E, S, out, num, rows, ylabel, name, text, width="page"):
    if not rows:
        return []
    f, ax = ps.figure(width, 2.4)
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        st = ps.style(r[0])
        ax.bar(i, r[1], 0.7, yerr=r[2], color=st["color"], edgecolor="black", linewidth=0.4,
               hatch="///" if r[0] == OURS else None, error_kw=dict(lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["method", "mean", "sd"])
    return [(p, cap.simple(num, E, S, text))]


# ---------------------------------------------------------------- C: representation
def _tsne(X, seed=0):
    from sklearn.manifold import TSNE
    X = X.astype(np.float32)
    if X.shape[1] > 50:
        from sklearn.decomposition import PCA
        X = PCA(50, random_state=seed).fit_transform(X)
    return TSNE(2, init="pca", random_state=seed, perplexity=30).fit_transform(X)


def _scatter_classes(ax, Z, y, C, marker="o", alpha=0.8, label_prefix=None):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if C <= 10 else "tab20")
    for c in range(C):
        sel = y == c
        if sel.any():
            # filled markers need no edge; line markers ("x") are drawn only by their edge, so they need a width
            lw = 0.6 if marker in ("x", "+", "1", "2", "3", "4") else 0
            ax.scatter(Z[sel, 0], Z[sel, 1], s=2.5 if lw == 0 else 6, color=cmap(c % cmap.N), marker=marker,
                       alpha=alpha, linewidths=lw,
                       label=f"{label_prefix or ''}{c}" if C <= 10 else None, rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _class_legend(ax, C, sources=False):
    """Class legend inside the axes, top right (embedding scatter plots): one entry per class (10 classes or fewer;
    with more, the caption says the points are coloured by class), plus the marker of real and generated features.
    The y range gets 35% headroom so that the legend sits above the points instead of on them."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if C > 10 and not sources:
        return
    cmap = plt.get_cmap("tab10")
    handles = [Line2D([], [], marker="o", linestyle="", color=cmap(c), markersize=4, label=str(c))
               for c in range(C)] if C <= 10 else []
    if sources:
        handles += [Line2D([], [], marker="o", linestyle="", color="0.45", markersize=4, label="real"),
                    Line2D([], [], marker="x", linestyle="", color="0.45", markersize=4, label="generated")]
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.35 * (y1 - y0))
    leg = ax.legend(handles=handles, loc="upper right", ncol=min(5, len(handles)), frameon=True, framealpha=0.9,
                    edgecolor="0.7", fancybox=False, handletextpad=0.1, columnspacing=0.6, borderpad=0.3,
                    labelspacing=0.25, borderaxespad=0.3)
    leg.get_frame().set_linewidth(0.5)
    leg.set_gid("keep")   # plot_style.place_legends leaves it where it is


def _emb_fig(E, S, out, num, m, name):
    rs = E.of(m, S)
    d = E.pred(rs[0]) if rs else None
    if d is None:
        return []
    keys = sorted(k for k in d.files if k.startswith("embed_"))
    if not keys:
        return []
    y = d["y"]
    C = int(y.max()) + 1
    g = keys[0]
    Z = _tsne(d[g].astype(np.float32))
    f, ax = ps.figure("column", 3.0)
    _scatter_classes(ax, Z, y, C)
    _class_legend(ax, C)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    p = ps.save(f, out, f"fig_{num:02d}_{name}", [[float(a), float(b), int(c)] for (a, b), c in zip(Z, y)],
                ["tsne1", "tsne2", "class"], raster=True)
    arch = int(g.split("_")[1]) + 1
    return [(p, cap.simple(num, E, S, f"t-SNE of the test-set embeddings of {cap.mname(m)}'s head CNN-{arch} "
                                      f"(2,000 test images, first seed), coloured by true class."))]


@fig(19, "C")
def f19(E, S, out):
    return _emb_fig(E, S, out, 19, OURS, "tsne_ours")


@fig(20, "C")
def f20(E, S, out):
    return _emb_fig(E, S, out, 20, REF, "tsne_geflf")


@fig(21, "C")
def f21(E, S, out):
    from sklearn.decomposition import PCA
    res = []
    rows = []
    ms_ = [m for m in (OURS, REF) if m in E.methods]
    if not ms_:
        return []
    f, axes = ps.figure("page", 2.6, ncols=len(ms_))
    axes = np.atleast_1d(axes)
    ok = False
    for ax, m in zip(axes, ms_):
        rs = E.of(m, S)
        d = E.pred(rs[0]) if rs else None
        keys = sorted(k for k in d.files if k.startswith("embed_")) if d is not None else []
        if not keys:
            continue
        y = d["y"]
        Z = PCA(2, random_state=0).fit_transform(d[keys[0]].astype(np.float32))
        _scatter_classes(ax, Z, y, int(y.max()) + 1)
        _class_legend(ax, int(y.max()) + 1)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        rows += [[m, float(a), float(b), int(c)] for (a, b), c in zip(Z, y)]
        ok = True
    if not ok:
        return []
    ps.panel_labels(axes, y=-0.1)   # no tick labels on embedding plots: the labels sit just under the axis title
    p = ps.save(f, out, "fig_21_pca_embeddings", rows, ["method", "pc1", "pc2", "class"], raster=True)
    res.append((p, cap.simple(21, E, S, "First two principal components of the test-set embeddings: (a) "
                              + " (b) ".join(cap.mname(m) for m in ms_) + "; coloured by true class.")))
    return res


def _real_vs_gen(E, S, out, num, m, name):
    rs = E.of(m, S)
    d = E.pred(rs[0]) if rs else None
    if d is None or "gen_x" not in d.files:
        return []
    Xr, yr, Xg, yg = d["val_x"].astype(np.float32), d["val_y"], d["gen_x"].astype(np.float32), d["gen_y"]
    Z = _tsne(np.concatenate([Xr, Xg]))
    C = int(max(yr.max(), yg.max())) + 1
    f, ax = ps.figure("column", 3.0)
    _scatter_classes(ax, Z[:len(Xr)], yr, C, marker="o", alpha=0.5)
    _scatter_classes(ax, Z[len(Xr):], yg, C, marker="x", alpha=0.9)
    _class_legend(ax, C, sources=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    rows = [[float(a), float(b), int(c), "real"] for (a, b), c in zip(Z[:len(Xr)], yr)] + \
           [[float(a), float(b), int(c), "generated"] for (a, b), c in zip(Z[len(Xr):], yg)]
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["tsne1", "tsne2", "class", "source"], raster=True)
    return [(p, cap.simple(num, E, S, f"t-SNE of real validation features (circles) and features generated by "
                                      f"{cap.mname(m)}'s generator (crosses) in the frozen feature space, coloured by "
                                      "class (first seed)."))]


@fig(22, "C")
def f22(E, S, out):
    return _real_vs_gen(E, S, out, 22, OURS, "tsne_real_vs_generated_ours")


@fig(23, "C")
def f23(E, S, out):
    return _real_vs_gen(E, S, out, 23, REF, "tsne_real_vs_generated_geflf")


@fig(24, "C")
def f24(E, S, out):
    res = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        rs = E.of(m, S)
        d = E.pred(rs[0]) if rs else None
        if d is None or "gen_x" not in d.files:
            continue
        Xr, yr, Xg, yg = d["val_x"].astype(np.float64), d["val_y"], d["gen_x"].astype(np.float64), d["gen_y"]
        C = int(max(yr.max(), yg.max())) + 1
        mr = np.stack([Xr[yr == c].mean(0) if (yr == c).any() else np.zeros(Xr.shape[1]) for c in range(C)])
        mg = np.stack([Xg[yg == c].mean(0) if (yg == c).any() else np.zeros(Xg.shape[1]) for c in range(C)])
        D = np.sqrt(((mr[:, None, :] - mg[None, :, :]) ** 2).sum(-1))
        f, ax = ps.figure("column", 3.0)
        im = ax.imshow(D, cmap="cividis")
        ax.set_xlabel("Generated class")
        ax.set_ylabel("Real class")
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Distance of class means")
        name = "ours" if m == OURS else "geflf"
        p = ps.save(f, out, f"fig_24_class_mean_distances_{name}",
                    [[i, j, float(D[i, j])] for i in range(C) for j in range(C)], ["real", "generated", "distance"])
        res.append((p, cap.simple(24, E, S, f"Euclidean distance between the mean real feature of each class "
                                            f"(validation set) and the mean generated feature of each class for "
                                            f"{cap.mname(m)}; a dark diagonal means generated classes sit at their real "
                                            "class.")))
    return res


@fig(25, "C")
def f25(E, S, out):
    f, ax = ps.figure("page", 2.4)
    rows = []
    for m in E.methods:
        vals = []
        for r in E.of(m, S):
            ev = E.log(r, "stage3_round")
            if ev and ev[-1].get("fc_norm"):
                vals.append(np.mean([np.array(v) for v in ev[-1]["fc_norm"].values()], 0))
        if not vals:
            continue
        v = np.mean(vals, 0)
        ax.plot(np.arange(len(v)), v, label=m, markevery=max(1, len(v) // 12), **ps.style(m))
        rows += [[m, i, float(x)] for i, x in enumerate(v)]
    if not rows:
        return []
    ax.set_xlabel("Class (ordered from most to least frequent)")
    ax.set_ylabel("Last-layer weight norm")
    ax.legend(ncol=3)
    p = ps.save(f, out, "fig_25_classifier_norms", rows, ["method", "class", "norm"])
    return [(p, cap.simple(25, E, S, "L2 norm of each class's last-layer weight vector after Stage III, averaged over "
                                     "architectures and seeds; a falling line means the classifier favours frequent "
                                     "classes."))]


@fig(26, "C")
def f26(E, S, out):
    return []  # needs the trained last layers: produced from models.zip by tools/figures_from_models.py


@fig(27, "C")
def f27(E, S, out):
    rows = []
    for m in E.methods:
        vals = []
        for r in E.of(m, S):
            d = E.pred(r)
            if d is None:
                continue
            for k in d.files:
                if k.startswith("embed_"):
                    H, y = d[k].astype(np.float64), d["y"]
                    C = int(y.max()) + 1
                    mu_g = H.mean(0)
                    mus = np.stack([H[y == c].mean(0) for c in range(C) if (y == c).any()])
                    Sw = np.mean([np.cov(H[y == c].T) if (y == c).sum() > 1 else 0 for c in range(C)], 0)
                    Sb = np.cov((mus - mu_g).T)
                    nc1 = float(np.trace(Sw @ np.linalg.pinv(Sb)) / len(mus))
                    vals.append(nc1)
        if vals:
            rows.append([m, float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0])
    return _bars(E, S, out, 27, rows, "NC1 (within / between)", "neural_collapse_nc1",
                 "Neural-collapse statistic NC1 = tr(S_W S_B^+)/C of the test-set embeddings (lower = tighter, better "
                 "separated classes), mean over architectures and seeds.")


# ---------------------------------------------------------------- D: generator
@fig(28, "D")
def f28(E, S, out):
    f, ax = ps.figure("page", 2.4)
    rows = []
    for m in E.methods:
        for key, suffix in (("fidelity", ""), ("fidelity_mc", " (moment-corrected)")):
            v = [r[key] for r in E.of(m, S) if r.get(key)]
            if not v:
                continue
            mu = np.mean(v, 0) * 100
            st = ps.style(m)
            if suffix:
                st = dict(st, linestyle=":")
            ax.plot(np.arange(len(mu)), mu, label=m + suffix, markevery=max(1, len(mu) // 12), **st)
            rows += [[m + suffix, i, float(x)] for i, x in enumerate(mu)]
    if not rows:
        return []
    ax.set_xlabel("Class (ordered from most to least frequent)")
    ax.set_ylabel("Fidelity (%)")
    ax.legend(ncol=2)
    p = ps.save(f, out, "fig_28_generator_fidelity", rows, ["generator", "class", "fidelity"])
    return [(p, cap.simple(28, E, S, "Per-class fidelity of the trained generator: share of 300 generated class-c "
                                     "features that a referee classifier trained on the server's validation features "
                                     "assigns to class c; mean over seeds."))]


@fig(29, "D")
def f29(E, S, out):
    f, ax = ps.figure("column", 2.4)
    rows = []
    for m in E.methods:
        v = [r["spread_ratio"] for r in E.of(m, S) if r.get("spread_ratio")]
        if not v:
            continue
        mu = np.mean(v, 0)
        ax.plot(np.arange(len(mu)), mu, label=m, **ps.style(m))
        rows += [[m, i, float(x)] for i, x in enumerate(mu)]
    if not rows:
        return []
    ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Class")
    ax.set_ylabel("Generated / real spread")
    ax.legend()
    p = ps.save(f, out, "fig_29_spread_ratio", rows, ["method", "class", "ratio"])
    return [(p, cap.simple(29, E, S, "Within-class spread of the generated features over the exact real spread "
                                     "(1 = realistic diversity, 0 = copies of the class mean), per class."))]


@fig(30, "D")
def f30(E, S, out):
    f, ax = ps.figure("column", 2.4)
    rows = []
    for m in E.methods:
        v = [r["cond_norms_end"] for r in E.of(m, S) if r.get("cond_norms_end")]
        if not v:
            continue
        mu = np.mean(v, 0)
        ax.plot(np.arange(len(mu)), mu, label=m, **ps.style(m))
        rows += [[m, i, float(x)] for i, x in enumerate(mu)]
    if not rows:
        return []
    ax.set_xlabel("Class (ordered from most to least frequent)")
    ax.set_ylabel("Class-row norm")
    ax.legend()
    p = ps.save(f, out, "fig_30_class_row_norms", rows, ["method", "class", "norm"])
    return [(p, cap.simple(30, E, S, "L2 norm of each class's conditioning row of the generator after Stage II "
                                     "(class-row generators only); mean over seeds."))]


@fig(31, "D")
def f31(E, S, out):
    return _lines(E, S, out, 31, "stage2_round",
                  lambda e: (np.mean(e["fidelity_post"][class_groups(len(e["fidelity_post"]))[2]])
                             if e.get("fidelity_post") else None),
                  "Tail-class fidelity (%)", "fidelity_rounds", "Fidelity of the aggregated generator on the tail "
                  "classes during Stage II (share of generated features the server's referee assigns to their class)",
                  split=None)


@fig(32, "D")
def f32(E, S, out):
    f, ax = ps.figure("column", 2.4)
    rows = []
    for m in E.methods:
        v = [r["mc_scale"] for r in E.of(m, S) if r.get("mc_scale")]
        if not v:
            continue
        mu = np.mean(v, 0)
        ax.plot(np.arange(len(mu)), mu, label=m, **ps.style(m))
        rows += [[m, i, float(x)] for i, x in enumerate(mu)]
    if not rows:
        return []
    ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Class")
    ax.set_ylabel("Spread scale s_c")
    p = ps.save(f, out, "fig_32_mc_scales", rows, ["method", "class", "scale"])
    return [(p, cap.simple(32, E, S, "Moment-correction scale s_c = sqrt(V_c / V~_c) per class: the factor by which "
                                     "the generator's within-class spread is stretched to the exact spread."))]


# ---------------------------------------------------------------- E: mechanism
@fig(33, "E")
def f33(E, S, out):
    res = []
    for m in E.methods:
        rs = E.of(m, S)
        if not rs:
            continue
        ev = E.log(rs[0], "stage2_round")
        ev = [e for e in ev if e.get("row_norm_global")]
        if not ev:
            continue
        A = np.array([e["row_norm_global"] for e in ev]).T
        rr = [e["round"] for e in ev]
        f, ax = ps.figure("column", 2.4)
        im = ax.imshow(A, aspect="auto", cmap="viridis", origin="lower")
        _round_ticks(ax, rr)
        ax.set_xlabel("Round")
        ax.set_ylabel("Class")
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Class-row norm")
        name = "".join(ch for ch in m if ch.isalnum()).lower()
        p = ps.save(f, out, f"fig_33_row_norm_rounds_{name}",
                    [[rr[j], c, float(A[c, j])] for c in range(A.shape[0]) for j in range(A.shape[1])],
                    ["round", "class", "norm"])
        res.append((p, cap.simple(33, E, S, f"Norm of each class's conditioning row of the aggregated generator over "
                                            f"the rounds of Stage II for {cap.mname(m)} (first seed).")))
    return res


@fig(34, "E")
def f34(E, S, out):
    rows = []
    for m in E.methods:
        for r in E.of(m, S)[:1]:
            counts = np.array(r["client_counts"])
            ev = E.log(r, "stage2_round")
            prev = None
            for e in ev:
                if not e.get("row_norm_local") or not e.get("row_norm_global"):
                    continue
                if prev is not None:
                    gain = np.zeros(counts.shape[1])
                    shrink = np.zeros(counts.shape[1])
                    nh = np.zeros(counts.shape[1])
                    nn_ = np.zeros(counts.shape[1])
                    for k, v in e["row_norm_local"].items():
                        hold = counts[int(k)] > 0
                        d = np.array(v) - np.array(prev)
                        gain[hold] += d[hold]
                        nh[hold] += 1
                        shrink[~hold] -= d[~hold]
                        nn_[~hold] += 1
                    rows.append([m, e["round"], float(np.mean(gain / np.maximum(nh, 1))),
                                 float(np.mean(shrink / np.maximum(nn_, 1)))])
                prev = e["row_norm_global"]
    if not rows:
        return []
    f, ax = ps.figure("column", 2.4)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        st = ps.style(m)
        ax.plot([r[1] for r in rr], [r[2] for r in rr], label=f"{m}: holders' gain", **st)
        ax.plot([r[1] for r in rr], [r[3] for r in rr], label=f"{m}: non-holders' shrink", **dict(st, linestyle=":"))
    ax.set_xlabel("Round")
    ax.set_ylabel("Row-norm change per round")
    ax.legend()
    p = ps.save(f, out, "fig_34_holder_gain_vs_shrink", rows, ["method", "round", "holder_gain", "nonholder_shrink"])
    return [(p, cap.simple(34, E, S, "Mean change of a class row's norm in one round of local training: growth on "
                                     "clients that hold the class against shrinkage on clients that do not (class-row "
                                     "generators, first seed)."))]


@fig(35, "E")
def f35(E, S, out):
    rows = []
    for m in E.methods:
        for r in E.of(m, S):
            if r.get("cond_norms_end"):
                holders = (np.array(r["client_counts"]) > 0).sum(0)
                rows += [[m, int(h), float(v)] for h, v in zip(holders, r["cond_norms_end"])]
    if not rows:
        return []
    f, ax = ps.figure("column", 2.4)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        st = ps.style(m)
        ax.scatter([r[1] for r in rr], [r[2] for r in rr], s=8, marker=st["marker"], color=st["color"], label=m)
    ax.set_xlabel("Clients holding the class")
    ax.set_ylabel("Final class-row norm")
    ax.legend()
    p = ps.save(f, out, "fig_35_row_norm_vs_holders", rows, ["method", "holders", "norm"])
    return [(p, cap.simple(35, E, S, "Final norm of each class's conditioning row against the number of clients that "
                                     "hold the class; one point per class and seed."))]


@fig(36, "E")
def f36(E, S, out):
    return _lines(E, S, out, 36, "stage2_round",
                  lambda e: (float(np.mean([np.mean(v) for v in e["row_v"].values()])) if e.get("row_v") else None),
                  "Adam second moment of rows", "adam_row_v", "Mean Adam second-moment estimate of the class rows over "
                  "clients during Stage II", scale=1.0, split=None)


@fig(37, "E")
def f37(E, S, out):
    rows = []
    for m in E.methods:
        a, b = metric(E, m, "final_bal", S), metric(E, m, "er_bal", S)
        if not np.isnan(a[0]) and not np.isnan(b[0]):
            rows.append([m, a[0] * 100, b[0] * 100, a[1] * 100, b[1] * 100])
    if not rows:
        return []
    f, ax = ps.figure("page", 2.5)
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, yerr=[r[3] for r in rows], label="Heads as trained",
           color="#BBBBBB", edgecolor="black", linewidth=0.4, error_kw=dict(lw=0.6))
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, yerr=[r[4] for r in rows], label="Exact readout", color="#333333",
           edgecolor="black", linewidth=0.4, hatch="///", error_kw=dict(lw=0.6))
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=35, ha="right")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.legend(ncol=2)
    p = ps.save(f, out, "fig_37_readout_gap", rows, ["method", "as_trained", "exact_readout", "sd_trained", "sd_er"])
    return [(p, cap.simple(37, E, S, "Balanced accuracy of the heads as trained and after the exact class-balanced "
                                     "readout (last layer recomputed from secure sums of the clients' embeddings); "
                                     "mean over seeds, error bars one standard deviation."))]


# ---------------------------------------------------------------- F: comparison
@fig(38, "F")
def f38(E, S, out):
    rows = []
    for m in E.methods:
        v = metric(E, m, "final_bal", S)
        if not np.isnan(v[0]):
            rows.append([m, v[0] * 100, v[1] * 100])
    return _bars(E, S, out, 38, rows, "Balanced accuracy (%)", "balanced_accuracy",
                 "Balanced accuracy on the test set (mean of the class recalls) of every method; mean over seeds, "
                 "error bars one standard deviation.")


@fig(39, "F")
def f39(E, S, out):
    from scipy import stats
    ours = {r["seed"]: r for r in E.of(OURS, S)}
    if not ours:
        return []
    rows = []
    for m in E.methods:
        if m == OURS:
            continue
        other = {r["seed"]: r for r in E.of(m, S)}
        common = sorted(set(ours) & set(other))
        d = [100 * (ours[s]["final_bal"] - other[s]["final_bal"]) for s in common
             if ours[s].get("final_bal") is not None and other[s].get("final_bal") is not None]
        if len(d) < 1:
            continue
        mu = float(np.mean(d))
        half = float(stats.t.ppf(0.975, len(d) - 1) * np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        rows.append([m, mu, mu - half, mu + half, len(d)])
    if not rows:
        return []
    f, ax = ps.figure("column", 0.25 * len(rows) + 0.8)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.errorbar(r[1], yi, xerr=[[r[1] - r[2]], [r[3] - r[1]]], fmt="o", color="black", markersize=3,
                    elinewidth=0.8, capsize=2)
    ax.axvline(0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("Difference in balanced accuracy (points)")
    p = ps.save(f, out, "fig_39_forest", rows, ["baseline", "mean_diff", "ci_low", "ci_high", "n_seeds"])
    if max(r[4] for r in rows) == 1:   # one seed: a difference on the same split, no interval
        return [(p, cap.simple(39, E, S, "Difference in balanced accuracy, FedECS-LT minus each baseline, on the same "
                                         "client split (one seed, so no confidence interval); right of the dotted line "
                                         "FedECS-LT is better."))]
    return [(p, cap.simple(39, E, S, "Paired difference in balanced accuracy, FedECS-LT minus each baseline, over the "
                                     "same seeds and splits; points are means, bars 95% t confidence intervals (none "
                                     "for a baseline with one common seed); right of the dotted line FedECS-LT is "
                                     "better."))]


@fig(40, "F")
def f40(E, S, out):
    """Average rank over seeds x main metrics (critical-difference style, Nemenyi distance shown)."""
    keys = ["final_bal", "final_f1", "final_precision", "final_mcc", "final_gmean", "final_tail", "final_worst",
            "final_auc_pr"]
    seeds = sorted({r["seed"] for r in E.runs if setting_key(r) == S})
    blocks = []
    for s in seeds:
        for k in keys:
            vals = {}
            for m in E.methods:
                rr = [r for r in E.of(m, S) if r["seed"] == s and r.get(k) is not None]
                if rr:
                    vals[m] = rr[0][k]
            if len(vals) == len(E.methods):
                blocks.append(vals)
    if len(blocks) < 2 or len(E.methods) < 3:
        return []
    from scipy.stats import rankdata
    R = np.array([rankdata([-b[m] for m in E.methods]) for b in blocks])
    avg = R.mean(0)
    k, N = len(E.methods), len(blocks)
    q = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}.get(k, 3.164 + 0.02 * (k - 10))
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    order = np.argsort(avg)
    f, ax = ps.figure("page", 0.18 * k + 1.0)
    for i, j in enumerate(order):
        ax.plot([avg[j]], [i], "o", color=ps.style(E.methods[j])["color"], markersize=4)
        ax.text(avg[j] + 0.08, i, E.methods[j], va="center")
    ax.plot([1, 1 + cd], [-1, -1], color="black", linewidth=1.0)
    ax.text(1 + cd / 2, -1.6, f"CD = {cd:.2f}", ha="center")
    ax.set_yticks([])
    ax.set_xlabel("Average rank (1 = best)")
    ax.set_ylim(-2.2, k)
    p = ps.save(f, out, "fig_40_critical_difference", [[E.methods[j], float(avg[j])] for j in order],
                ["method", "average_rank"])
    return [(p, cap.simple(40, E, S, f"Average rank of every method over {N} (seed, metric) pairs (balanced accuracy, "
                                     "macro-F1, macro precision, MCC, G-mean, tail recall, worst-class recall, macro "
                                     f"AUC-PR); the bar is the Nemenyi critical difference CD = {cd:.2f} at the 5% "
                                     "level."))]


@fig(41, "F")
def f41(E, S, out):
    keys = [("final_bal", "Bal. acc."), ("final_f1", "Macro-F1"), ("final_precision", "Precision"),
            ("final_mcc", "MCC"), ("final_gmean", "G-mean"), ("final_tail", "Tail recall"),
            ("final_worst", "Worst class"), ("final_auc_pr", "AUC-PR")]
    # FedECS-LT and the five other methods with the highest balanced accuracy
    others = sorted((m for m in E.methods if m != OURS and not np.isnan(metric(E, m, "final_bal", S)[0])),
                    key=lambda m: -metric(E, m, "final_bal", S)[0])
    ms_ = ([OURS] if OURS in E.methods else []) + others[:5]
    vals = {m: [metric(E, m, k, S)[0] for k, _ in keys] for m in ms_}
    vals = {m: v for m, v in vals.items() if not any(np.isnan(v))}
    if len(vals) < 2:
        return []
    import matplotlib.pyplot as plt
    ps.apply()
    f = plt.figure(figsize=(ps.COLUMN_IN, 3.0))
    ax = f.add_subplot(111, polar=True)
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    for m, v in vals.items():
        st = ps.style(m)
        ax.plot(ang + ang[:1], [x * 100 for x in v] + [v[0] * 100], color=st["color"], linestyle=st["linestyle"],
                linewidth=st["linewidth"], label=m)
    ax.set_xticks(ang)
    ax.set_xticklabels([n for _, n in keys])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    p = ps.save(f, out, "fig_41_radar", [[m] + v for m, v in vals.items()], ["method"] + [n for _, n in keys])
    return [(p, cap.simple(41, E, S, "Main metrics (%, test set, mean over seeds) of FedECS-LT and the five other "
                                     "methods with the highest balanced accuracy; larger is better on every axis."))]


@fig(42, "F")
def f42(E, S, out):
    return _pareto(E, S, out, 42, "comm_mb_per_client", "Communication per client (MB)", "accuracy_vs_communication",
                   "communication per client over the whole training (logarithmic axis)", logx=True)


@fig(43, "F")
def f43(E, S, out):
    return _pareto(E, S, out, 43, "time_s", "Wall-clock time per run (s)", "accuracy_vs_time",
                   "wall-clock time of one run on one GPU (logarithmic axis)", logx=True)


def _pareto(E, S, out, num, key, xlabel, name, what, logx=False):
    rows = []
    f, ax = ps.figure("column", 2.4)
    for m in E.methods:
        x, y = metric(E, m, key, S), metric(E, m, "final_bal", S)
        if np.isnan(x[0]) or np.isnan(y[0]):
            continue
        st = ps.style(m)
        ax.scatter([x[0]], [y[0] * 100], marker=st["marker"], color=st["color"], s=16, label=m, linewidths=0.4)
        rows.append([m, x[0], y[0] * 100])
    if not rows:
        return []
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Balanced accuracy (%)")
    ax.legend(ncol=2)
    p = ps.save(f, out, f"fig_{num:02d}_{name}", rows, ["method", "x", "balanced_accuracy"])
    return [(p, cap.simple(num, E, S, f"Balanced accuracy against {what}; mean over seeds. Up and left is better."))]


@fig(44, "F")
def f44(E, S, out):
    keys = [("final_bal", "Bal."), ("final_acc", "Acc."), ("final_f1", "F1"), ("final_precision", "Prec."),
            ("final_mcc", "MCC"), ("final_kappa", "Kappa"), ("final_gmean", "G-mean"), ("final_tail", "Tail"),
            ("final_worst", "Worst"), ("final_auc_pr", "AUC-PR"), ("final_auc_roc", "AUC-ROC"),
            ("er_bal", "Bal. (ER)")]
    M = np.array([[metric(E, m, k, S)[0] * 100 for k, _ in keys] for m in E.methods])
    if M.size == 0 or np.all(np.isnan(M)):
        return []
    f, ax = ps.figure("page", 0.22 * len(E.methods) + 0.9)
    Z = (M - np.nanmean(M, 0)) / np.where(np.nanstd(M, 0) > 0, np.nanstd(M, 0), 1)
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="white" if abs(Z[i, j]) > 1.6 else "black")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([n for _, n in keys], rotation=35, ha="right")
    ax.set_yticks(range(len(E.methods)))
    ax.set_yticklabels(E.methods)
    f.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Standardised score")
    p = ps.save(f, out, "fig_44_metric_heatmap", [[m] + list(r) for m, r in zip(E.methods, M)],
                ["method"] + [n for _, n in keys])
    return [(p, cap.simple(44, E, S, "Every main metric (%, test set, mean over seeds) of every method; colours are "
                                     "standardised per metric (red = better than the column mean)."))]


@fig(45, "F")
def f45(E, S, out):
    return []  # ablation waterfall: made from the ablation experiment (its own notebook)


@fig(46, "F")
def f46(E, S, out):
    return []  # sensitivity lines: made from the sweep experiments (phase C)


# ---------------------------------------------------------------- G: system
@fig(47, "G")
def f47(E, S, out):
    rows = []
    for m in E.methods:
        for r in E.of(m, S)[:1]:
            for t, b in (r.get("bytes") or {}).items():
                rows.append([m, t, (b["down"] + b["up"]) / 2 ** 20 / r["K"]])
    if not rows:
        return []
    types = sorted({r[1] for r in rows})
    ms_ = [m for m in E.methods if any(r[0] == m for r in rows)]
    f, ax = ps.figure("page", 2.6)
    bottom = np.zeros(len(ms_))
    for i, t in enumerate(types):
        v = np.array([sum(r[2] for r in rows if r[0] == m and r[1] == t) for m in ms_])
        ax.bar(np.arange(len(ms_)), v, 0.7, bottom=bottom, label=t, color=ps.PALETTE[i % len(ps.PALETTE)],
               hatch=["", "///", "...", "xx", "\\\\"][i % 5], edgecolor="black", linewidth=0.3)
        bottom += v
    ax.set_xticks(np.arange(len(ms_)))
    ax.set_xticklabels(ms_, rotation=35, ha="right")
    ax.set_ylabel("MB per client")
    ax.legend(ncol=3)
    p = ps.save(f, out, "fig_47_communication", rows, ["method", "message_type", "mb_per_client"])
    return [(p, cap.simple(47, E, S, "Communication per client over the whole training, by message type (model "
                                     "weights, statistics, secure sums); first seed. Cached shared stages count once."))]


@fig(48, "G")
def f48(E, S, out):
    rows = []
    for m in E.methods:
        for st_ in ("fe_time_s", "gen_time_s", "head_time_s", "time_s"):
            v = metric(E, m, st_, S)
            if not np.isnan(v[0]):
                rows.append([m, st_, v[0]])
    if not rows:
        return []
    ms_ = [m for m in E.methods if any(r[0] == m for r in rows)]
    f, ax = ps.figure("page", 2.5)
    for i, (st_, lab) in enumerate((("fe_time_s", "Stage I"), ("gen_time_s", "Stage II"), ("time_s", "Whole run"))):
        v = [next((r[2] for r in rows if r[0] == m and r[1] == st_), 0) for m in ms_]
        ax.bar(np.arange(len(ms_)) + (i - 1) * 0.27, v, 0.27, label=lab, color=["#BBBBBB", "#777777", "#333333"][i],
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(ms_)))
    ax.set_xticklabels(ms_, rotation=35, ha="right")
    ax.set_ylabel("Time (s)")
    ax.legend(ncol=3)
    p = ps.save(f, out, "fig_48_time_per_stage", rows, ["method", "stage", "seconds"])
    return [(p, cap.simple(48, E, S, "Wall-clock time per stage on one GPU (simulation of all clients), mean over "
                                     "seeds; shared stages loaded from cache report the time of their first run."))]


@fig(49, "G")
def f49(E, S, out):
    rows = []
    for m in E.methods:
        v = metric(E, m, "gpu_peak_mb", S)
        if not np.isnan(v[0]):
            rows.append([m, v[0], v[1]])
    return _bars(E, S, out, 49, rows, "Peak GPU memory (MB)", "gpu_memory",
                 "Peak GPU memory of the server process during one run (mean over seeds).")


@fig(50, "G")
def f50(E, S, out):
    rows = []
    for r in E.runs:
        if setting_key(r) != S or r.get("pfrac", 1.0) >= 1.0:
            continue
        for e in E.log(r, "stage3_round"):
            rows.append([r["label"], e["round"], len(e["clients"])])
    if not rows:
        return []
    f, ax = ps.figure("column", 2.2)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        ax.plot([r[1] for r in rr], [r[2] for r in rr], label=m, **ps.style(m))
    ax.set_xlabel("Round")
    ax.set_ylabel("Participating clients")
    p = ps.save(f, out, "fig_50_participation", rows, ["method", "round", "clients"])
    return [(p, cap.simple(50, E, S, "Number of clients taking part in each round of Stage III (partial "
                                     "participation: the same seeded subset for every method)."))]


@fig(51, "G")
def f51(E, S, out):
    rows = []
    f, ax = ps.figure("column", 2.4)
    for m in E.methods:
        rs = E.of(m, S)
        if not rs or not rs[0].get("comm_mb_per_client"):
            continue
        c = _curve(E, m, S, "stage3_round", lambda e: (e.get("post") or {}).get("bal"))
        if c is None:
            continue
        x, mu, _ = c
        total = rs[0]["comm_mb_per_client"]
        xx = total * x / x.max()
        ax.plot(xx, mu * 100, label=m, markevery=max(1, len(x) // 6), **ps.style(m))
        rows += [[m, float(a), float(b * 100)] for a, b in zip(xx, mu)]
    if not rows:
        return []
    ax.set_xlabel("Cumulative communication per client (MB)")
    ax.set_ylabel("Balanced accuracy (%)")
    ax.legend(ncol=2)
    p = ps.save(f, out, "fig_51_accuracy_vs_cumulative_communication", rows, ["method", "mb", "bal"])
    return [(p, cap.simple(51, E, S, "Balanced accuracy of the aggregated heads against cumulative communication "
                                     "per client (Stage III rounds placed proportionally on the total)."))]


# ---------------------------------------------------------------- H, I: privacy and diagram (own experiments)
@fig(52, "H")
def f52(E, S, out):
    rows = []
    for r in E.runs:
        if setting_key(r) == S and r.get("dp_eps") is not None:
            rows.append([r["label"], r["dp_eps"], r.get("final_bal")])
    if not rows:
        return []
    f, ax = ps.figure("column", 2.3)
    eps = sorted({r[1] for r in rows})
    ax.plot(eps, [np.mean([r[2] for r in rows if r[1] == e]) * 100 for e in eps], **ps.style(OURS))
    ax.set_xscale("log")
    ax.set_xlabel("Privacy budget epsilon")
    ax.set_ylabel("Balanced accuracy (%)")
    p = ps.save(f, out, "fig_52_accuracy_vs_epsilon", rows, ["method", "epsilon", "bal"])
    return [(p, cap.simple(52, E, S, "Balanced accuracy of FedECS-LT with differentially private statistics "
                                     "release against the privacy budget (delta = 1e-5)."))]


@fig(53, "H")
def f53(E, S, out):
    return []  # membership-inference ROC: privacy experiment


@fig(54, "H")
def f54(E, S, out):
    return []  # release noise vs class size: privacy experiment


@fig(55, "I")
def f55(E, S, out):
    return []  # pipeline diagram: drawn once (docs/latex), not per experiment


# ---------------------------------------------------------------- section 17b: aggregation effect
@fig(56, "17b")
def f56(E, S, out):
    res = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        rs = E.of(m, S)
        if not rs:
            continue
        r = rs[0]
        if _one_client_per_arch(r):
            continue
        rows, M, rounds = [], [], []
        for e in E.log(r, "stage3_round"):
            if not e.get("pre") or not e.get("post"):
                continue
            pre = np.mean([np.array(v["recall_c"]) for v in e["pre"].values()], 0)
            post = np.array(e["post"]["recall_c"])
            M.append((post - pre) * 100)
            rounds.append(e["round"])
            rows += [[e["round"], c, float(x)] for c, x in enumerate((post - pre) * 100)]
        if not M:
            continue
        A = np.array(M).T
        f, ax = ps.figure("column", 2.4)
        lim = max(1e-6, float(np.abs(A).max()))
        im = ax.imshow(A, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
        _round_ticks(ax, rounds)
        ax.set_xlabel("Round")
        ax.set_ylabel("Class")
        f.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Recall after - before (points)")
        name = "ours" if m == OURS else "geflf"
        p = ps.save(f, out, f"fig_56_aggregation_effect_{name}", rows, ["round", "class", "effect"])
        res.append((p, cap.simple(56, E, S, f"Aggregation effect per class for {cap.mname(m)}: recall of the aggregated "
                                            "heads minus the mean recall of the clients' local heads just before "
                                            "aggregation, at every evaluation round of Stage III (blue = aggregation "
                                            "helps); first seed.")))
    return res


@fig(57, "17b")
def f57(E, S, out):
    rows = []
    for m in E.methods:
        for r in E.of(m, S)[:1]:
            for e in E.log(r, "stage2_round"):
                if e.get("fidelity_pre") and e.get("fidelity_post"):
                    C = len(e["fidelity_post"])
                    tl = class_groups(C)[2]
                    pre = np.mean([np.array(v)[tl].mean() for v in e["fidelity_pre"].values()])
                    rows.append([m, e["round"], float(pre) * 100, float(np.mean(np.array(e["fidelity_post"])[tl])) * 100])
    if not rows:
        return []
    f, ax = ps.figure("column", 2.4)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        st = ps.style(m)
        ax.plot([r[1] for r in rr], [r[2] for r in rr], label=f"{m}: client copies", **dict(st, linestyle=":"))
        ax.plot([r[1] for r in rr], [r[3] for r in rr], label=f"{m}: aggregate", **st)
    ax.set_xlabel("Round")
    ax.set_ylabel("Tail-class fidelity (%)")
    ax.legend()
    p = ps.save(f, out, "fig_57_fidelity_pre_post", rows, ["method", "round", "pre", "post"])
    return [(p, cap.simple(57, E, S, "Tail-class fidelity of the clients' generator copies just before aggregation "
                                     "(mean over clients) and of the aggregated generator, during Stage II."))]


@fig(58, "17b")
def f58(E, S, out):
    rows = []
    for m in E.methods:
        for r in E.of(m, S)[:1]:
            counts = np.array(r["client_counts"])
            C = counts.shape[1]
            tl = class_groups(C)[2]
            for e in E.log(r, "stage2_round"):
                if not e.get("row_norm_local") or not e.get("row_norm_global"):
                    continue
                h, n_ = [], []
                for k, v in e["row_norm_local"].items():
                    v = np.array(v)
                    hold = counts[int(k)] > 0
                    for c in range(C)[tl]:
                        (h if hold[c] else n_).append(v[c])
                rows.append([m, e["round"], float(np.mean(h)) if h else None, float(np.mean(n_)) if n_ else None,
                             float(np.mean(np.array(e["row_norm_global"])[tl]))])
    if not rows:
        return []
    f, ax = ps.figure("column", 2.4)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        st = ps.style(m)
        ax.plot([r[1] for r in rr], [r[2] for r in rr], label=f"{m}: holders' copies", **st)
        ax.plot([r[1] for r in rr], [r[3] for r in rr], label=f"{m}: non-holders' copies", **dict(st, linestyle=":"))
        ax.plot([r[1] for r in rr], [r[4] for r in rr], label=f"{m}: aggregate", **dict(st, linestyle="--"))
    ax.set_xlabel("Round")
    ax.set_ylabel("Tail class-row norm")
    ax.legend()
    p = ps.save(f, out, "fig_58_row_norm_holders_nonholders", rows,
                ["method", "round", "holders", "non_holders", "aggregate"])
    return [(p, cap.simple(58, E, S, "Norm of the tail classes' conditioning rows in the copies of clients that hold "
                                     "the class, of clients that do not, and in the aggregate, during Stage II "
                                     "(class-row generators, first seed)."))]


@fig(59, "17b")
def f59(E, S, out):
    rows = []
    for m in [x for x in (OURS, REF) if x in E.methods]:
        for r in [x for x in E.of(m, S)[:1] if not _one_client_per_arch(x)]:
            holders = (np.array(r["client_counts"]) > 0).sum(0)
            effs = []
            for e in E.log(r, "stage3_round"):
                if e.get("pre") and e.get("post"):
                    pre = np.mean([np.array(v["recall_c"]) for v in e["pre"].values()], 0)
                    effs.append(np.array(e["post"]["recall_c"]) - pre)
            if effs:
                eff = np.mean(effs, 0) * 100
                rows += [[m, int(h), float(x)] for h, x in zip(holders, eff)]
    if not rows:
        return []
    f, ax = ps.figure("column", 2.4)
    for m in sorted({r[0] for r in rows}):
        rr = [r for r in rows if r[0] == m]
        st = ps.style(m)
        ax.scatter([r[1] for r in rr], [r[2] for r in rr], s=9, marker=st["marker"], color=st["color"], label=m)
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Clients holding the class")
    ax.set_ylabel("Mean aggregation effect (points)")
    ax.legend()
    p = ps.save(f, out, "fig_59_effect_vs_holders", rows, ["method", "holders", "effect"])
    return [(p, cap.simple(59, E, S, "Mean aggregation effect on a class's recall (aggregate minus local heads, over "
                                     "Stage III) against the number of clients holding the class; one point per "
                                     "class."))]


@fig(60, "17b")
def f60(E, S, out):
    loc = E.of("Local only", S)
    if not loc:
        return []
    l_rec = np.mean([r["final_recall_c"] for r in loc], 0) * 100
    rows = []
    f, ax = ps.figure("page", 2.4)
    x = np.arange(len(l_rec))
    for m in [x_ for x_ in (OURS, REF) if x_ in E.methods]:
        v = np.mean([r["final_recall_c"] for r in E.of(m, S)], 0) * 100
        ax.plot(x, v - l_rec, label=m, **ps.style(m))
        rows += [[m, int(c), float(d)] for c, d in zip(x, v - l_rec)]
    if not rows:
        return []
    ax.axhline(0, color="#999999", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Class (ordered from most to least frequent)")
    ax.set_ylabel("Recall gain over local (points)")
    ax.legend()
    p = ps.save(f, out, "fig_60_federated_vs_local", rows, ["method", "class", "gain"])
    return [(p, cap.simple(60, E, S, "Per-class benefit of federation: recall of each federated method minus the "
                                     "recall of local-only training (each client trains on its own data), mean over "
                                     "seeds."))]


def _arch(K):
    blk = max(K // 10, 1)
    return [min(k // blk, 9) for k in range(K)]


# ---------------------------------------------------------------- driver
def make_all(res_dir, out_dir):
    """Every figure of the catalogue for every setting of the experiment. Returns (figures, not_applicable), where
    figures = [(number, group, pdf path, caption)] and not_applicable = [(number, group, reason)]."""
    E = Exp(res_dir)
    figs, na = [], []
    for S in E.settings:
        sub = os.path.join(out_dir, cap.setting_slug(S))
        for num, group, fn in sorted(FIGS):
            try:
                made = fn(E, S, sub)
            except Exception as e:  # a failing figure is reported in the report, never hides the others
                na.append((num, group, f"error: {e!r}"[:300], S))
                continue
            if made:
                figs += [(num, group, p, c) for p, c in made]
            else:
                na.append((num, group, cap.not_applicable(num), S))
    return figs, na, E
