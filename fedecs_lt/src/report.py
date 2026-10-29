"""report.pdf of one experiment: every table and every figure the experiment produced, with captions, plus the run
list, failures and a file index. A completeness check fails the notebook if any figure file or any logged scalar
metric is missing from the PDF.

    python report.py --results RESULTS_DIR --title "..." --out report.pdf
"""
import argparse
import datetime
import glob
import json
import os
import platform
import re

import numpy as np

import plot_style as ps
import plots
import captions as cap
from captions import proto_note, setting_slug, setting_text
from settings import CITE, DATASETS

MAIN = [("final_bal", "Bal. acc."), ("final_f1", "Macro-F1"), ("final_precision", "Precision"), ("final_mcc", "MCC"),
        ("final_gmean", "G-mean"), ("final_tail", "Tail rec."), ("final_worst", "Worst rec."),
        ("final_auc_pr", "AUC-PR")]
SECOND = [("final_acc", "Accuracy"), ("final_kappa", "Kappa"), ("final_top5", "Top-5"), ("final_auc_roc", "AUC-ROC"),
          ("final_head", "Head rec."), ("final_medium", "Med. rec."), ("final_rec_std", "Recall SD")]
CLIENTS = [("final_worst_arch", "Worst-arch. bal."), ("final_local", "Own-mix acc."),
           ("final_own_mix_sd", "Own-mix SD"), ("final_own_mix_min", "Own-mix min.")]
CALIB = [("final_ece", "ECE"), ("final_cw_ece", "Class-wise ECE"), ("final_nll", "NLL"), ("final_brier", "Brier")]
READOUT = [("er_bal", "Bal. acc."), ("er_f1", "Macro-F1"), ("er_tail", "Tail rec."), ("er_worst", "Worst rec."),
           ("er_mcc", "MCC"), ("er_gmean", "G-mean")]
VAL = [("val_final_bal", "Val. bal. acc."), ("val_er_bal", "Val. bal. acc., exact readout")]
SYSTEM = [("comm_mb_per_client", "Comm./client (MB)"), ("time_s", "Time (s)"), ("fe_time_s", "Stage I (s)"),
          ("gen_time_s", "Stage II (s)"), ("gpu_peak_mb", "Peak GPU (MB)"), ("fwd_mflops", "MFLOPs/img"),
          ("train_gflops_round", "Train. GFLOPs/round")]
NOT_PCT = {"final_nll", "final_brier", "comm_mb_per_client", "time_s", "fe_time_s", "gen_time_s", "gpu_peak_mb",
           "class_sep_F", "fwd_mflops", "train_gflops_round"}


REFS = {
    "FedECS-LT": ("This work", "Our implementation (Flower)"),
    "GeFL-F": ("Kang et al., IEEE TMC 2025", "Re-implementation of the authors' Algorithm 3, checked against the "
                                             "earlier experiments"),
    "GeFL-F + LA": ("Ablation: GeFL-F with logit adjustment (Menon et al., ICLR 2021) and decoupled weight decay "
                    "(Loshchilov and Hutter, ICLR 2019)", "Our implementation"),
    "GeFL-F w/o generator": ("Ablation: GeFL-F without its generator (heads trained on real features only)",
                             "Our implementation"),
    "Balanced Softmax": ("Ren et al., NeurIPS 2020 (each client's head trained with its own class prior; equal to "
                         "logit adjustment, Menon et al., ICLR 2021, with tau = 1)", "Our implementation (loss only)"),
    "FedLC": ("Zhang et al., ICML 2022", "Re-implemented from the paper (loss only)"),
    "FedRS": ("Li and Zhan, KDD 2021", "Re-implemented from the paper (loss only)"),
    "GeFL-F w/ Gaussian generator + LA": ("Ablation: GeFL-F with its generator replaced by class Gaussians from the "
                                          "exact class statistics, with logit adjustment", "Our implementation"),
    "FedDF": ("Lin et al., NeurIPS 2020", "Re-implementation (server ensemble distillation on generated features)"),
    "FedDF + LA": ("Ablation: FedDF (Lin et al., NeurIPS 2020) with logit adjustment", "Re-implementation"),
    "Local": ("Lower bound: each client trains its own model alone, without federation", "Our implementation"),
    "FedECS-LT (200 extractor rounds)": ("This work, with the feature extractor trained for 200 rounds (diagnosis run)",
                               "Our implementation (Flower)"),
}


# (runs.jsonl key, name in the caption, shown in %): every per-class metric (metrics.PER_CLASS), heads as trained,
# then the exact readout's per-class recall
PER_CLASS_TABLES = [
    ("final_recall_c", "recall (%)", True), ("final_precision_c", "precision (%)", True),
    ("final_f1_c", "F1 score (%)", True), ("final_ap_c", "average precision, AUC-PR (%)", True),
    ("final_auc_c", "AUC-ROC, class against the rest (%)", True), ("final_mcc_c", "MCC, class against the rest (%)", True),
    ("final_ece_c", "calibration error of p(c|x) (%)", True), ("final_nll_c", "NLL on the images of the class (nats)", False),
    ("final_brier_c", "Brier score on the images of the class", False),
    ("final_top5_c", "top-5 recall (%)", True), ("final_pred_share_c", "share of all predictions (%)", True),
    ("er_recall_c", "recall with the exact class-balanced readout (%)", True)]


def method_ref(m):
    if m in REFS:
        return REFS[m]
    if m.startswith("FedECS-LT"):
        return ("This work (ablation / variant)", "Our implementation (Flower)")
    base = m.split(" (")[0]
    return (CITE.get(base, "see the experiment description"), "see the method's source field")


def roman(n):
    """IEEE table numbers."""
    out = ""
    for v, r in ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                 (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + r, n - v
    return out


def sub_super(text):
    """Captions write subscripts and superscripts as _{...} and ^{...} (escaped text in, reportlab markup out)."""
    return re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", text))


def fmt(mu, sd, key, n=2):
    """mean ± SD of n values; the SD is shown only for two or more (one run has no spread)."""
    if np.isnan(mu):
        return "-"
    spread = n > 1 and not np.isnan(sd)
    if key in NOT_PCT:
        if abs(mu) >= 100 or (spread and sd >= 100):   # times, bytes, FLOPs: whole numbers, never an exponent
            return f"{mu:,.0f} ± {sd:,.0f}" if spread and sd > 0 else f"{mu:,.0f}"
        return f"{mu:.3g} ± {sd:.2g}" if spread and sd > 0 else f"{mu:.3g}"
    return f"{100 * mu:.1f} ± {100 * sd:.1f}" if spread else f"{100 * mu:.1f}"


def build(res_dir, title, out_pdf, fig_dir):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)
    from xml.sax.saxutils import escape

    figs, na, E = plots.make_all(res_dir, fig_dir)
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Times-Roman", fontSize=9, leading=11, alignment=TA_JUSTIFY)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, leading=9)
    capst = ParagraphStyle("c", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Times-Bold", fontSize=14, spaceAfter=6)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Times-Bold", fontSize=11, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=6.5, leading=7.5, alignment=0)
    P = lambda t, st=body: Paragraph(escape(str(t)), st)  # noqa: E731
    story = []
    used_keys = set()

    def table(rows, header, widths=None, bold_best=None):
        data = [[Paragraph(f"<b>{sub_super(escape(h))}</b>", cell) for h in header]]
        for r in rows:
            data.append([Paragraph(escape(str(x)), cell) for x in r])
        t = Table(data, colWidths=widths, repeatRows=1)
        st = [("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
              ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]
        for (i, j) in (bold_best or []):
            st.append(("BACKGROUND", (j, i + 1), (j, i + 1), colors.HexColor("#E6E6E6")))
        t.setStyle(TableStyle(st))
        return t

    tab_no = [0]
    table_errors = []
    caption_problems = []

    def tcap(text):
        tab_no[0] += 1
        n = roman(tab_no[0])
        caption_problems.extend(f"Table {n}: {p_}" for p_ in cap.lint(text, figure=False))
        return Paragraph(f"<b>TABLE {n}</b><br/>{sub_super(escape(text))}", ParagraphStyle("tc", parent=capst,
                                                                                          alignment=1))

    def metric_table(S, keys, caption):
        rows, cells = [], []
        for i, m in enumerate(E.methods):
            r = [m]
            vals = []
            for k, _ in keys:
                mu, sd, n = plots.metric(E, m, k, S)
                used_keys.add(k)
                r.append(fmt(mu, sd, k, n))
                vals.append(mu)
            if all(np.isnan(v) for v in vals):
                continue
            rows.append(r)
            cells.append(vals)
        if not rows:
            return []
        best = []
        A = np.array(cells, dtype=float)
        for j, (k, _) in enumerate(keys):
            col = A[:, j]
            if np.all(np.isnan(col)):
                continue
            lower_better = k in ("final_ece", "final_cw_ece", "final_nll", "final_brier", "final_rec_std",
                                 "comm_mb_per_client", "time_s", "fe_time_s", "gen_time_s", "gpu_peak_mb",
                                 "fwd_mflops", "train_gflops_round", "final_own_mix_sd")
            b = np.nanmin(col) if lower_better else np.nanmax(col)
            best += [(i, j + 1) for i in np.where(np.isclose(col, b, rtol=0, atol=1e-9))[0]]   # ties: all shaded
        w = [1.55 * inch] + [(7.0 - 1.55) * inch / len(keys)] * len(keys)
        return [tcap(caption), Spacer(1, 3), table(rows, ["Method"] + [n for _, n in keys], w, best), Spacer(1, 10)]

    def paired_table(S, st):
        """FedECS-LT minus each method in balanced accuracy, paired over the common seeds: mean, 95% t interval,
        paired t-test p and Holm-adjusted p over the methods."""
        from scipy import stats
        import representation as rp
        ours = {r["seed"]: r for r in E.of(plots.OURS, S)}
        rows_ = []
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
                rows_.append([m, len(d), float(np.mean(d)), half, pv])
        if not ours:
            return []
        if not rows_:
            return [P("Paired tests: FedECS-LT and the other methods share fewer than two seeds in this experiment, "
                      "so no paired test is computed (the merged experiment computes them over seeds 3-5).", small),
                    Spacer(1, 8)]
        adj = rp.holm([r[4] for r in rows_])
        body = [[r[0], r[1], f"{r[2]:+.1f}", f"[{r[2] - r[3]:+.1f}, {r[2] + r[3]:+.1f}]", f"{r[4]:.3g}", f"{a:.3g}"]
                for r, a in zip(rows_, adj)]
        return [tcap("Balanced accuracy of FedECS-LT minus each method (points, test set, heads as trained), paired "
                     "over the common seeds: mean, 95% t confidence interval, two-sided paired t-test p-value and "
                     "Holm-adjusted p-value over the methods. " + st), Spacer(1, 3),
                table(body, ["Method", "Seeds", "Difference", "95% CI", "p", "p (Holm)"],
                      [1.9 * inch, 0.6 * inch, 1.0 * inch, 1.4 * inch, 1.0 * inch, 1.1 * inch]), Spacer(1, 10)]

    def rounds_table(S, st):
        """Rounds each method needs to reach a target balanced accuracy: the final balanced accuracy of the
        reference (GeFL-F, or FedAvg in the one-shared-model setting)."""
        shared = DATASETS.get(S[0], {}).get("track") == 2
        ref = "FedAvg" if shared else plots.REF
        tgt = plots.metric(E, ref, "final_bal", S)[0]
        if np.isnan(tgt):
            return []
        body = []
        for m in E.methods:
            first = []
            for r in E.of(m, S):
                series = plots._global_series(E, r) if shared else \
                    [(e["round"], e["post"]) for e in E.log(r, "stage3_round") if e.get("post")]
                if not series:      # no rounds (Local trains once, alone)
                    continue
                # the per-round logs keep 4 decimals: a round reaches the target within that rounding
                hit = [rd for rd, p in series if p.get("bal") is not None and p["bal"] >= tgt - 5e-5]
                first.append(min(hit) if hit else None)
            if not first:
                continue
            got = [x for x in first if x is not None]
            body.append([m, f"{np.mean(got):.0f}" if got else "not reached", f"{len(got)} of {len(first)}"])
        if not body:
            return []
        stage = "communication rounds of the one shared model (FedECS-LT: extractor rounds, then head rounds)" \
            if shared else "head-training rounds (Stage III, or the method's own rounds)"
        return [tcap(f"Rounds to the target balanced accuracy {100 * tgt:.1f}% (the final balanced accuracy of {ref}): "
                     f"first evaluated round, in {stage}, at which a method's test balanced accuracy reaches it; mean "
                     "over the seeds that reach it. " + st), Spacer(1, 3),
                table(body, ["Method", "Rounds", "Seeds reaching it"], [2.4 * inch, 1.6 * inch, 1.6 * inch]),
                Spacer(1, 10)]

    def representation_table(S, st):
        import representation as rp
        gen_mmd = plots._gen_class_stat(E, S, rp.mmd_rbf)
        gen_mis = plots._gen_class_stat(E, S, rp.mean_mismatch)
        body = []
        for m in E.methods:
            col = plots._collapse(E, m, S)
            mmd = [x[2] for x in gen_mmd if x[0] == m]
            mis = [x[2] for x in gen_mis if x[0] == m]
            au = [r["active_units"] / r["latent_dim"] for r in E.of(m, S) if r.get("active_units") is not None
                  and r.get("latent_dim")]

            def mean_of(k):
                v = [x[k] for x in col if x.get(k) is not None]
                return f"{np.mean(v):.3f}" if v else "-"
            def pct_of(k):
                v = [x[k] for x in col if x.get(k) is not None]
                return f"{100 * np.mean(v):.1f}" if v else "-"
            row = [m, pct_of("knn"), pct_of("knn_tail"), mean_of("nc2_equinorm"), mean_of("nc2_equiangle"), mean_of("nc3"),
                   (f"{np.mean([x['minority_angle'] for x in col if x.get('minority_angle') is not None]):.1f}"
                    if any(x.get("minority_angle") is not None for x in col) else "-"),
                   f"{100 * np.mean(au):.0f}" if au else "-",
                   f"{np.nanmean(mmd):.3f}" if mmd and not np.all(np.isnan(mmd)) else "-",
                   f"{np.nanmean(mis):.2f}" if mis and not np.all(np.isnan(mis)) else "-"]
            if any(x != "-" for x in row[1:]):
                body.append(row)
        if not body:
            return []
        return [tcap("Representation and generator: leave-one-out 10-nearest-neighbour accuracy of the test embeddings "
                     "(%, all classes and the tail classes), NC2 equinorm and equiangularity and NC3 self-duality of "
                     "the test "
                     "embeddings and last layers (lower = closer to neural collapse), mean angle between the tail "
                     "classes' classifier vectors (degrees; small = minority collapse), active latent units of the "
                     "generator (%), and the mean over classes of the squared MMD and of the mean mismatch (in "
                     "within-class standard deviations) between generated and real validation features; mean over "
                     "architectures and seeds. " + st), Spacer(1, 3),
                table(body, ["Method", "10-NN acc.", "10-NN tail", "NC2 norm", "NC2 angle", "NC3", "Tail angle",
                             "Active units", "MMD^{2}", "Mean mism."], [1.55 * inch] + [(7.0 - 1.55) * inch / 9] * 9),
                Spacer(1, 10)]

    # ---------------------------------------------------------------- cover
    done = len(E.runs)
    fails = []
    fp = os.path.join(res_dir, "failures.jsonl")
    if os.path.exists(fp):
        fails = [json.loads(line) for line in open(fp) if line.strip()]
    done_ids = {r["run_id"] for r in E.runs}
    lost = sorted({f["run_id"] for f in fails} - done_ids)        # failed and not recovered by the retry
    recovered = sorted({f["run_id"] for f in fails} & done_ids)
    story += [Paragraph(escape(title), h1),
              P(f"Report generated {datetime.datetime.utcnow():%Y-%m-%d %H:%M} UTC. {done} runs completed"
                + (f" ({len(recovered)} of them on a second attempt)" if recovered else "")
                + f", {len(lost)} failed. Framework: Flower simulation (one server, one client process per simulation); "
                  "every client trains on its own data only."
                + (" The long-tail baselines of the one-shared-model setting run in their authors' own code "
                   "(LongTailLib, FedLTLib benchmark) on the same client split (Table II)."
                   if any("LongTailLib" in str(r.get("framework", "")) for r in E.runs) else "")),
              Spacer(1, 6)]
    aud = None
    if os.path.exists(os.path.join(res_dir, "audit.json")):
        aud = json.load(open(os.path.join(res_dir, "audit.json")))
        story += [P("Logging audit: " + (
            f"every run has every metric for every class on every evaluation ({aud['checked']['runs']} runs, "
            f"{aud['checked']['round_evaluations']} per-round evaluations checked)." if aud["n_problems"] == 0 else
            f"{aud['n_problems']} problems found; see the appendix 'Logging audit'.")), Spacer(1, 6)]
    n_err = sum(1 for _, _, reason, _ in na if str(reason).startswith("error:"))
    if n_err:
        story += [P(f"Figure errors: {n_err} figures could not be drawn; Appendix A gives the error of each."),
                  Spacer(1, 6)]
    cover_at = len(story)   # table errors are known only after the tables: their note is inserted here
    try:
        import torch
        tv, gpu = torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    except Exception:          # a merged report (merge_results.py) can be built on a computer without PyTorch
        tv, gpu = "not installed", "CPU only"
    try:
        import flwr
        fv = flwr.__version__
    except Exception:
        fv = "unknown"
    env = [["Python", platform.python_version()], ["PyTorch", tv], ["Flower", fv], ["GPU", gpu],
           ["Code version", os.environ.get("FEDECS_CODE_VERSION", "see results/code_version.txt")]]
    merged = os.environ.get("FEDECS_MERGED")   # set by merge_results.py
    env_cap = ("Software and hardware of the computer that joined the experiment's parts and drew this report; the "
               "runs themselves ran on Kaggle GPUs (each part's own report gives its software and GPU)."
               if merged else "Software and hardware.")
    story += [tcap(env_cap), Spacer(1, 3), table(env, ["Item", "Value"], [1.6 * inch, 5.4 * inch]),
              Spacer(1, 8)]
    meth = []
    for m in E.methods:
        r = E.of(m)[0]
        ref, src = method_ref(m)
        meth.append([m, ref, r["method"].get("source", src)])
    story += [tcap("Methods in this experiment, their reference and the origin of their code."), Spacer(1, 3),
              table(meth, ["Method", "Reference", "Code"], [1.9 * inch, 2.3 * inch, 2.8 * inch]), Spacer(1, 8)]
    for S in E.settings:
        story.append(P("Setting: " + setting_text(S, sorted({r['seed'] for r in E.runs if plots.setting_key(r) == S}))))
    acc = os.path.join(res_dir, "acceptance.json")
    v = None
    if os.path.exists(acc):
        try:
            v = json.load(open(acc))
        except ValueError as e:
            story.append(P(f"The acceptance check's output could not be read ({escape(str(e))})."))

    def pct(d):
        return "-" if d.get("new") is None else (f"{100 * d['new']:.1f} / {100 * d['old_seed3']:.1f} "
                                                f"(±{100 * d['tolerance']:.1f})")
    if v is not None:
        rows = []
        for c in v["checks"]:
            if "final_bal" not in c:
                rows.append([c["check"], c["verdict"], "-", "-", "-", "-"])
                continue
            rows.append([c["check"], c["verdict"], "yes" if c["counts_identical"] else "NO",
                         pct(c["final_bal"]), pct(c["er_bal"]),
                         f"{c['F']['new']:.3f} / {c['F']['old_seed3']:.3f}" if "F" in c else "-"])
        story += [Spacer(1, 8), tcap(f"Acceptance of the Flower implementation against the earlier experiments "
                                     f"(seed 3): overall {v['overall']}. Balanced accuracy new / old (tolerance), "
                                     "as trained and with the exact readout; class separation F new / old."),
                  Spacer(1, 3), table(rows, ["Check", "Verdict", "Same splits", "Bal. acc.", "Bal. acc. (ER)", "F"],
                                      [1.6 * inch, 0.6 * inch, 0.7 * inch, 1.4 * inch, 1.4 * inch, 1.3 * inch])]
    story.append(PageBreak())

    # ---------------------------------------------------------------- tables
    story.append(Paragraph("1. Results", h1))
    for S in E.settings:
        st = setting_text(S, sorted({r['seed'] for r in E.runs if plots.setting_key(r) == S}))
        story.append(Paragraph(escape(st), h2))
        # the classes of this setting, in the order every per-class table and figure uses
        r0 = next((r for r in E.runs if plots.setting_key(r) == S and r.get("class_order")), None)
        if r0 is not None:
            C = len(r0["class_order"])
            ntr = np.array(r0.get("client_counts") or [[0] * C]).sum(0)
            nte = r0.get("test_class_counts") or [None] * C   # images of each class in the test set
            names = r0.get("class_names") or [None] * C
            rows = [[str(c), str(o), escape(names[c] or "-"), str(int(ntr[c])), "-" if nte[c] is None else str(int(nte[c]))]
                    for c, o in enumerate(r0["class_order"])]
            story += [tcap("Classes in the order used by every per-class table and figure (0 = most frequent in the "
                           "federation's training data): original label in the dataset's files, name, training images "
                           "held by all clients together, test images. " + st), Spacer(1, 3),
                      table(rows, ["Class", "Original label", "Name", "Training images", "Test images"],
                            [0.6 * inch, 1.0 * inch, 3.4 * inch, 1.0 * inch, 1.0 * inch]), Spacer(1, 10)]
        seeds = sorted({r['seed'] for r in E.runs if plots.setting_key(r) == S})
        one = lambda t: cap.one_seed(t, seeds)  # noqa: E731
        shared = DATASETS.get(S[0], {}).get("track") == 2      # Track 2: one model shared by all clients
        trained = "model as trained" if shared else "heads as trained"
        story += metric_table(S, MAIN, one(f"Main metrics (%, test set, {trained}; mean ± standard deviation over "
                                           "seeds; shaded = best). Bal. acc.: mean of the class recalls; Macro-F1 "
                                           "and Precision: means over the classes (precision 0 for a class that is "
                                           "never predicted); G-mean: geometric mean of the class recalls, each "
                                           "floored at 1%; Tail rec.: mean recall of the rarest 40% of the classes; "
                                           "Worst rec.: lowest class recall; AUC-PR: macro average precision. " + st))
        story += metric_table(S, READOUT, ("The same model" if shared else "The same heads") + " with the exact "
                              "class-balanced readout (last layer recomputed from secure sums of every client's "
                              "embeddings). " + st)
        story += metric_table(S, SECOND, f"Further metrics (%, test set, {trained}). Top-5: share of the test images "
                                         "whose class is among the five most probable; AUC-ROC: one class against the "
                                         "rest, macro average; Head rec. and Med. rec.: mean recall of the most "
                                         "frequent 30% and of the next 30% of the classes; Recall SD: standard "
                                         "deviation of the class recalls. " + st)
        if not shared:   # one shared model: no architectures, every client evaluates the same model
            story += metric_table(S, CLIENTS, f"Fairness across clients (%, test set, {trained}): balanced accuracy of "
                                  "the weakest head architecture, and the accuracy re-weighted to each architecture's "
                                  "own clients' class mix (own-mix accuracy; mean, standard deviation and minimum over "
                                  "the architectures, or over the clients' own models). " + st)
        story += metric_table(S, CALIB, "Calibration (ECE and class-wise ECE in %, NLL in nats, Brier score). ECE: 15 "
                              "equal-width bins of the highest probability; class-wise ECE: the same for p(c|x) of "
                              "every class, averaged (Kull et al., NeurIPS 2019); NLL with probabilities floored at "
                              "10^{-6}; Brier score summed over the classes (0 to 2)."
                              + proto_note([m for m in E.methods if E.of(m, S)]) + " " + st)
        story += metric_table(S, VAL, "Balanced accuracy on the validation set (images no client holds), used to select "
                                      "settings without the test set. " + st)
        story += metric_table(S, SYSTEM, "Cost: communication per client over the whole training, wall-clock time, "
                                         "peak GPU memory on one GPU, forward FLOPs per image (extractor and the mean "
                                         "head, or the whole model) and training FLOPs of one round over all clients "
                                         "(3 x forward; the head stage for the methods on the frozen extractor, every "
                                         "round of the one shared model otherwise). " + st)
        for fn in (paired_table, rounds_table, representation_table):
            try:
                story += fn(S, st)
            except Exception as e:   # a failing table is reported, never stops the report
                table_errors.append(f"{fn.__name__} ({setting_slug(S)}): {e!r}"[:400])
        # per-seed table
        rows = []
        for m in E.methods:
            for r in sorted(E.of(m, S), key=lambda r: r["seed"]):
                rows.append([m, r["seed"], fmt(r.get("final_bal", np.nan) or np.nan, 0, "final_bal", 1),
                             fmt(r.get("er_bal", np.nan) or np.nan, 0, "er_bal", 1),
                             fmt(r.get("final_tail", np.nan) or np.nan, 0, "final_tail", 1),
                             fmt(r.get("final_f1", np.nan) or np.nan, 0, "final_f1", 1), f"{r.get('time_s', 0):.0f}"])
        if rows:
            story += [tcap("Every run (one row per method and seed): balanced accuracy as trained and with the exact "
                           "readout, tail recall, macro-F1 (%), time (s). " + st), Spacer(1, 3),
                      table(rows, ["Method", "Seed", "Bal.", "Bal. (ER)", "Tail", "F1", "Time"],
                            [1.9 * inch, 0.5 * inch] + [0.9 * inch] * 5), Spacer(1, 10)]
        # per-class tables: every metric that has a per-class form, for every class (blocks of 20 classes)
        for key, name, pct in PER_CLASS_TABLES:
            recs = []
            for m in E.methods:
                v = [r.get(key) for r in E.of(m, S) if r.get(key)]
                if v:
                    mu = np.nanmean(np.array(v, dtype=float), 0) * (100 if pct else 1)
                    recs.append((m, mu))
            if not recs:
                continue
            if key == "final_pred_share_c":   # reference row: each class's true share of the test set
                cm = next((np.array(r["final_confusion"], dtype=float) for r in E.of(recs[0][0], S)
                           if r.get("final_confusion")), None)
                if cm is not None:
                    recs.append(("Test set (true share)", cm.sum(1) / cm.sum() * 100))
            C = len(recs[0][1])
            for c0 in range(0, C, 20):
                cols = list(range(c0, min(C, c0 + 20)))
                def cell_(x):   # no "-0" for a value that rounds to zero
                    return "-" if np.isnan(x) else (f"{int(round(x))}" if pct else f"{x:.2f}".replace("-0.00", "0.00"))
                rows = [[m] + [cell_(mu[c]) for c in cols]
                        for m, mu in recs]
                part = "" if C <= 20 else f" Classes {cols[0]}-{cols[-1]} of {C}."
                note = proto_note([m for m, _ in recs]) if key in ("final_ece_c", "final_nll_c", "final_brier_c") else ""
                story += [tcap(one(f"Per-class {name} (test set, mean over seeds); class 0 is the most frequent."
                                   f"{part}{note} " + st)), Spacer(1, 3),
                          table(rows, ["Method"] + [str(c) for c in cols],
                                [1.55 * inch] + [(7.0 - 1.55) * inch / len(cols)] * len(cols)), Spacer(1, 10)]
    story.append(PageBreak())

    # ---------------------------------------------------------------- figures
    story.append(Paragraph("2. Figures", h1))
    story.append(P("Every figure is drawn at its printed IEEE size (one column 3.5 in, two columns 7.16 in, Times "
                   "8 pt); shown here as previews. The vector PDF and the plotted data (CSV) of each are in "
                   "results.zip, folder figures/.", small))
    included = set()
    for i, (num, group, pdf, caption) in enumerate(figs, 1):
        png = pdf[:-4] + ".png"
        from PIL import Image as PILImage
        with PILImage.open(png) as im:
            w, h = im.size
        width_in = min(7.0, w / 200.0)
        img = Image(png, width=width_in * inch, height=width_in * inch * h / w)
        story.append(KeepTogether([img, Spacer(1, 3), Paragraph(f"<b>Fig. {i}.</b> {sub_super(escape(caption))}", capst),
                                   Spacer(1, 12)]))
        included.add(os.path.abspath(pdf))
        caption_problems.extend(f"Fig. {i} (catalogue {num}): {p_}" for p_ in cap.lint(caption))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix
    story.append(Paragraph("Appendix A. Figures not applicable to this experiment", h1))
    rows = [[n, g, reason, setting_slug(S)] for n, g, reason, S in na]
    caption_problems.extend(f"not-applicable reason of figure {n}: {p_}" for n, g, reason, S in na
                            if not str(reason).startswith("error:") for p_ in cap.lint(str(reason), figure=False))
    if rows:
        story += [table(rows, ["Catalogue no.", "Group", "Reason", "Setting"],
                        [0.9 * inch, 0.6 * inch, 4.0 * inch, 1.5 * inch]), Spacer(1, 8)]
    story.append(Paragraph("Appendix B. Figure sources", h1))
    rows = [[i, n, os.path.relpath(p, os.path.dirname(fig_dir))] for i, (n, g, p, c) in enumerate(figs, 1)]
    story += [table(rows, ["Fig.", "Catalogue no.", "File"], [0.5 * inch, 0.9 * inch, 5.6 * inch]), Spacer(1, 8)]
    story.append(Paragraph("Appendix C. All logged values of every run", h1))
    for r in E.runs:
        scal = sorted((k, v) for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
        used_keys.update(k for k, _ in scal)
        rows = [[k, f"{v:.6g}" if isinstance(v, float) else str(v)] for k, v in scal]
        vec = sorted(k for k, v in r.items() if isinstance(v, (list, dict)))
        story += [Paragraph(escape(r["run_id"]), h2), table(rows, ["Quantity", "Value"], [3.0 * inch, 4.0 * inch]),
                  P("Vector-valued fields (in runs.jsonl): " + ", ".join(vec), small), Spacer(1, 6)]
    if fails:
        story.append(Paragraph("Appendix D. Failed runs", h1))
        story += [table([[f["run_id"] + (" (completed on retry)" if f["run_id"] in done_ids else ""),
                          f["error"][:600]] for f in fails], ["Run", "Error"], [2.0 * inch, 5.0 * inch])]
    if aud is not None and aud["n_problems"]:
        story.append(Paragraph("Appendix E. Logging audit", h1))
        story += [P(f"{aud['n_problems']} problems (at most 200 listed)."),
                  table([[escape(p_)] for p_ in aud["problems"]], ["Problem"], [7.0 * inch])]

    # ---------------------------------------------------------------- completeness check
    all_pdfs = {os.path.abspath(p) for p in glob.glob(os.path.join(fig_dir, "**", "*.pdf"), recursive=True)}
    missing = sorted(all_pdfs - included)
    if missing:
        raise RuntimeError(f"report incomplete: {len(missing)} figure files not in the PDF: {missing[:5]}")

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"{title} - page {doc.page}")
        canvas.restoreState()
    layout = [f"{n}: {p_}" for n, p_ in ps.LAYOUT_PROBLEMS]
    if layout or caption_problems:   # presentation checks: on the cover, and listed in an appendix
        story.insert(cover_at, P(f"Presentation checks: {len(layout)} figure-layout and {len(caption_problems)} "
                                 "caption problems; Appendix G lists them."))
        story.append(Paragraph("Appendix G. Presentation checks", h1))
        story += [table([[escape(t)] for t in (layout + caption_problems)[:300]], ["Problem"], [7.0 * inch])]
    if table_errors:   # on the cover, and listed in an appendix
        story.insert(cover_at, P(f"Table errors: {len(table_errors)} tables could not be computed; Appendix F lists "
                                 "them."))
        story.append(Paragraph("Appendix F. Table errors", h1))
        story += [table([[escape(t)] for t in table_errors], ["Error"], [7.0 * inch])]
    doc = SimpleDocTemplate(out_pdf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch, title=title)
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return dict(figures=len(figs), not_applicable=len(na), runs=len(E.runs), failures=len(lost),
                figure_errors=sum(1 for _, _, reason, _ in na if str(reason).startswith("error:")),
                table_errors=len(table_errors), layout_problems=len(ps.LAYOUT_PROBLEMS),
                caption_problems=len(caption_problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    info = build(a.results, a.title, a.out, os.path.join(a.results, "figures"))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
