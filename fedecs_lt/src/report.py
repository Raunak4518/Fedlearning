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

import numpy as np

import plots
from captions import proto_note, setting_slug, setting_text
from settings import CITE

MAIN = [("final_bal", "Bal. acc."), ("final_f1", "Macro-F1"), ("final_precision", "Precision"), ("final_mcc", "MCC"),
        ("final_gmean", "G-mean"), ("final_tail", "Tail rec."), ("final_worst", "Worst rec."),
        ("final_auc_pr", "AUC-PR")]
SECOND = [("final_acc", "Accuracy"), ("final_kappa", "Kappa"), ("final_top5", "Top-5"), ("final_auc_roc", "AUC-ROC"),
          ("final_head", "Head rec."), ("final_medium", "Med. rec."), ("final_rec_std", "Recall SD"),
          ("final_worst_arch", "Worst arch."), ("final_local", "Own-mix acc.")]
CALIB = [("final_ece", "ECE"), ("final_cw_ece", "Class-wise ECE"), ("final_nll", "NLL"), ("final_brier", "Brier")]
READOUT = [("er_bal", "Bal. acc."), ("er_f1", "Macro-F1"), ("er_tail", "Tail rec."), ("er_worst", "Worst rec."),
           ("er_mcc", "MCC"), ("er_gmean", "G-mean")]
VAL = [("val_final_bal", "Val. bal. acc."), ("val_er_bal", "Val. bal. acc. (ER)")]
SYSTEM = [("comm_mb_per_client", "Comm./client (MB)"), ("time_s", "Time (s)"), ("fe_time_s", "Stage I (s)"),
          ("gen_time_s", "Stage II (s)"), ("gpu_peak_mb", "Peak GPU (MB)")]
NOT_PCT = {"final_nll", "final_brier", "comm_mb_per_client", "time_s", "fe_time_s", "gen_time_s", "gpu_peak_mb",
           "class_sep_F"}


REFS = {
    "FedECS-LT": ("This work", "This suite (Flower)"),
    "GeFL-F": ("Kang et al., IEEE TMC 2025", "Re-implementation of the authors' Algorithm 3, checked against the "
                                             "earlier experiments"),
    "GeFL-F + AdamW + LA": ("GeFL-F with decoupled weight decay (Loshchilov and Hutter, ICLR 2019) and logit "
                            "adjustment (Menon et al., ICLR 2021)", "This suite"),
    "Real only": ("Control: heads trained on real features only (no generator)", "This suite"),
    "Real only + LA (balanced softmax)": ("Balanced softmax (Ren et al., NeurIPS 2020), i.e. logit adjustment "
                                          "(Menon et al., ICLR 2021) with each client's prior", "This suite"),
    "FedLC": ("Zhang et al., ICML 2022", "Re-implemented from the paper (loss only)"),
    "FedRS": ("Li and Zhan, KDD 2021", "Re-implemented from the paper (loss only)"),
    "Gaussian generator + LA": ("Control: class Gaussian from exact statistics, with logit adjustment", "This suite"),
    "FedDF": ("Lin et al., NeurIPS 2020", "Re-implementation (server ensemble distillation on generated features)"),
    "FedDF + LA": ("FedDF (Lin et al., NeurIPS 2020) with logit adjustment", "Re-implementation"),
    "Local only": ("Control: no federation (each client trains alone)", "This suite"),
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
        return ("This work (ablation / variant)", "This suite (Flower)")
    base = m.split(" (")[0]
    return (CITE.get(base, "see the experiment description"), "see the method's source field")


def fmt(mu, sd, key):
    if np.isnan(mu):
        return "-"
    if key in NOT_PCT:
        if abs(mu) >= 1000:   # times in seconds, bytes: whole numbers, no exponent
            return f"{mu:,.0f}" if sd == 0 or np.isnan(sd) else f"{mu:,.0f} ± {sd:,.0f}"
        return f"{mu:.3g}" if sd == 0 or np.isnan(sd) else f"{mu:.3g} ± {sd:.2g}"
    return f"{100 * mu:.1f} ± {100 * sd:.1f}"


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
        data = [[Paragraph(f"<b>{escape(h)}</b>", cell) for h in header]]
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

    def tcap(text):
        tab_no[0] += 1
        roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
                 "XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII",
                 "XXIX", "XXX"]
        n = roman[tab_no[0] - 1] if tab_no[0] <= len(roman) else str(tab_no[0])
        return Paragraph(f"<b>TABLE {n}</b><br/>{escape(text)}", ParagraphStyle("tc", parent=capst, alignment=1))

    def metric_table(S, keys, caption):
        rows, cells = [], []
        for i, m in enumerate(E.methods):
            r = [m]
            vals = []
            for k, _ in keys:
                mu, sd, n = plots.metric(E, m, k, S)
                used_keys.add(k)
                r.append(fmt(mu, sd, k))
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
                                 "comm_mb_per_client", "time_s", "fe_time_s", "gen_time_s", "gpu_peak_mb")
            i = int(np.nanargmin(col) if lower_better else np.nanargmax(col))
            best.append((i, j + 1))
        w = [1.55 * inch] + [(7.0 - 1.55) * inch / len(keys)] * len(keys)
        return [tcap(caption), Spacer(1, 3), table(rows, ["Method"] + [n for _, n in keys], w, best), Spacer(1, 10)]

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
                + (" Methods marked LongTailLib run in the authors' own code (LongTailLib, FedLTLib benchmark) on the "
                   "same client split." if any("LongTailLib" in str(r.get("framework", "")) for r in E.runs) else "")),
              Spacer(1, 6)]
    import torch
    try:
        import flwr
        fv = flwr.__version__
    except Exception:
        fv = "unknown"
    env = [["Python", platform.python_version()], ["PyTorch", torch.__version__], ["Flower", fv],
           ["GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"],
           ["Code version", os.environ.get("FEDECS_CODE_VERSION", "see results/code_version.txt")]]
    story += [tcap("Software and hardware."), Spacer(1, 3), table(env, ["Item", "Value"], [1.6 * inch, 5.4 * inch]),
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
            cm = r0.get("final_confusion")
            nte = np.array(cm).sum(1) if cm else [None] * C
            names = r0.get("class_names") or [None] * C
            rows = [[str(c), str(o), escape(names[c] or "-"), str(int(ntr[c])), "-" if nte[c] is None else str(int(nte[c]))]
                    for c, o in enumerate(r0["class_order"])]
            story += [tcap("Classes in the order used by every per-class table and figure (0 = most frequent in the "
                           "federation's training data): original label in the dataset's files, name, training images "
                           "held by all clients together, test images. " + st), Spacer(1, 3),
                      table(rows, ["Class", "Original label", "Name", "Training images", "Test images"],
                            [0.6 * inch, 1.0 * inch, 3.4 * inch, 1.0 * inch, 1.0 * inch]), Spacer(1, 10)]
        story += metric_table(S, MAIN, "Main metrics (%, test set, heads as trained; mean ± standard deviation over "
                                       "seeds; shaded = best). " + st)
        story += metric_table(S, READOUT, "The same heads with the exact class-balanced readout (last layer recomputed "
                                          "from secure sums of every client's embeddings). " + st)
        story += metric_table(S, SECOND, "Further metrics (%, test set, heads as trained). Own-mix accuracy: accuracy "
                                         "re-weighted to each architecture's own clients' class mix. " + st)
        story += metric_table(S, CALIB, "Calibration (ECE and class-wise ECE in %, NLL in nats, Brier score)."
                              + proto_note([m for m in E.methods if E.of(m, S)]) + " " + st)
        story += metric_table(S, VAL, "Balanced accuracy on the validation set (images no client holds), used to select "
                                      "settings without the test set. " + st)
        story += metric_table(S, SYSTEM, "Cost: communication per client over the whole training, wall-clock time and "
                                         "peak GPU memory on one GPU. " + st)
        # per-seed table
        rows = []
        for m in E.methods:
            for r in sorted(E.of(m, S), key=lambda r: r["seed"]):
                rows.append([m, r["seed"], fmt(r.get("final_bal", np.nan) or np.nan, 0, "final_bal"),
                             fmt(r.get("er_bal", np.nan) or np.nan, 0, "er_bal"),
                             fmt(r.get("final_tail", np.nan) or np.nan, 0, "final_tail"),
                             fmt(r.get("final_f1", np.nan) or np.nan, 0, "final_f1"), f"{r.get('time_s', 0):.0f}"])
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
                rows = [[m] + ["-" if np.isnan(mu[c]) else (f"{mu[c]:.0f}" if pct else f"{mu[c]:.2f}") for c in cols]
                        for m, mu in recs]
                part = "" if C <= 20 else f" Classes {cols[0]}-{cols[-1]} of {C}."
                note = proto_note([m for m, _ in recs]) if key in ("final_ece_c", "final_nll_c", "final_brier_c") else ""
                story += [tcap(f"Per-class {name} (test set, mean over seeds); class 0 is the most frequent.{part}{note} "
                               + st), Spacer(1, 3),
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
        story.append(KeepTogether([img, Spacer(1, 3), Paragraph(f"<b>Fig. {i}.</b> {escape(caption)}", capst),
                                   Spacer(1, 12)]))
        included.add(os.path.abspath(pdf))
    story.append(PageBreak())

    # ---------------------------------------------------------------- appendix
    story.append(Paragraph("Appendix A. Figures not applicable to this experiment", h1))
    rows = [[n, g, reason, setting_slug(S)] for n, g, reason, S in na]
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
    doc = SimpleDocTemplate(out_pdf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.8 * inch, title=title)
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return dict(figures=len(figs), not_applicable=len(na), runs=len(E.runs), failures=len(lost))


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
