// Builds the evaluation report (Word) from figures in ./fig. Run: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, PageBreak, TableOfContents, Footer, Header, PageNumber, LevelFormat, BorderStyle,
} = require("docx");

const FIG = path.join(__dirname, "fig");
const CONTENT = 9638; // A4 width 11906 minus 2 x 1134 margins (DXA)
const BLUE = "1C4E9A", GREY = "5B6676", LIGHT = "EEF2F8", GREEN = "E9F7EC", RED = "FDECEC";

// ---------- helpers
function runs(text, base = {}) {
  const out = [];
  for (const part of text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/)) {
    if (!part) continue;
    if (part.startsWith("**")) out.push(new TextRun({ ...base, text: part.slice(2, -2), bold: true }));
    else if (part.startsWith("*")) out.push(new TextRun({ ...base, text: part.slice(1, -1), italics: true }));
    else out.push(new TextRun({ ...base, text: part }));
  }
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 276 }, alignment: o.align || AlignmentType.JUSTIFIED, ...o.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true, spacing: { after: 200 } });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], keepNext: true, spacing: { before: 240, after: 120 } });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], keepNext: true, spacing: { before: 180, after: 80 } });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 264 } });
const N = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });
const caption = (t) => new Paragraph({ children: runs(t, { size: 18, color: GREY, italics: true }), alignment: AlignmentType.CENTER, spacing: { after: 220 } });
const br = () => new Paragraph({ children: [new PageBreak()] });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function img(name, widthPx = 640, cap = null) {
  const f = path.join(FIG, name);
  const { w, h } = pngSize(f);
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: widthPx, height: Math.round(widthPx * h / w) },
      altText: { title: name, description: cap || name, name } })] })];
  if (cap) out.push(caption(cap));
  return out;
}
function eq(name, label = null) {
  const f = path.join(FIG, name + ".png");
  const { w, h } = pngSize(f);
  let width = Math.round(w * 96 / 220 * 0.95);
  if (width > 600) width = 600;
  const kids = [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width, height: Math.round(width * h / w) },
    altText: { title: name, description: "equation", name } })];
  if (label) kids.push(new TextRun({ text: "     (" + label + ")", color: GREY }));
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 }, children: kids });
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFC7D5" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(head, rows, fr, opts = {}) {
  const tot = fr.reduce((a, b) => a + b, 0);
  const widths = fr.map((x) => Math.round(CONTENT * x / tot));
  widths[widths.length - 1] += CONTENT - widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, hdr, shade) => new TableCell({
    borders, width: { size: widths[i], type: WidthType.DXA },
    shading: { fill: hdr ? BLUE : (shade || "FFFFFF"), type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: [new Paragraph({ alignment: (i === 0 || opts.left) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), hdr ? { bold: true, color: "FFFFFF", size: 18 } : { size: 18 }) })],
  });
  const trs = [new TableRow({ tableHeader: true, children: head.map((t, i) => cell(t, i, true)) })];
  rows.forEach((r, ri) => {
    const shade = (opts.highlight || []).includes(ri) ? "DCEBFF" : (opts.good || []).includes(ri) ? GREEN : (opts.bad || []).includes(ri) ? RED : (ri % 2 ? "F7F9FC" : "FFFFFF");
    trs.push(new TableRow({ children: r.map((t, i) => cell(t, i, false, shade)) }));
  });
  return [new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: widths, rows: trs }),
          new Paragraph({ spacing: { after: opts.cap ? 40 : 160 }, children: [] }), ...(opts.cap ? [caption(opts.cap)] : [])];
}
function callout(title, lines, fill = LIGHT) {
  const c = new TableCell({ borders: { top: { style: BorderStyle.SINGLE, size: 12, color: BLUE }, bottom: border, left: border, right: border },
    width: { size: CONTENT, type: WidthType.DXA }, shading: { fill, type: ShadingType.CLEAR, color: "auto" },
    margins: { top: 120, bottom: 120, left: 160, right: 160 },
    children: [new Paragraph({ children: runs(title, { bold: true, color: BLUE }), spacing: { after: 80 } }),
      ...lines.map((l) => new Paragraph({ children: runs(l), spacing: { after: 60 } }))] });
  return [new Table({ width: { size: CONTENT, type: WidthType.DXA }, columnWidths: [CONTENT], rows: [new TableRow({ children: [c] })] }),
          new Paragraph({ spacing: { after: 160 }, children: [] })];
}

// ---------- content
const C = [];
// Title page
C.push(new Paragraph({ spacing: { before: 2400, after: 200 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Research Progress Report", size: 28, color: GREY })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({
  text: "Exact-Statistics Anchoring for Generator-Based Heterogeneous Federated Learning under Long-Tailed Data", bold: true, size: 40, color: BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({
  text: "Why rare classes vanish from GeFL-F, and how Ours-A brings them back", italics: true, size: 24, color: GREY })] }));
for (const l of ["Submitted by: [Student Name]   ·   [Roll No.]", "Supervisor: [Professor Name]", "[Department], [Institute]", "Version 2 · 15 October 2026"])
  C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: l, size: 24 })] }));
C.push(br());

// Summary
C.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Summary")], spacing: { after: 200 } }));
C.push(B("**Problem:** under long-tailed clients GeFL-F loses 21–25 points; rare classes are almost never predicted."));
C.push(B("**Cause:** FedAvg with Adam's weight decay shrinks the generator's class rows of rare classes, even when 70–90 % of the clients hold them (Theorem 1, measured on the real model)."));
C.push(B("**Our architecture (Ours-A):** class identity from the exact class mean, fake features corrected to exact class statistics, consensus labels, client-prior loss."));
C.push(...table(["Result (3 seeds, same seeds and splits for every method)", "GeFL-F", "Ours-A", "Gain"], [
  ["MNIST, long tail (balanced accuracy)", "75.25", "**94.82**", "+19.6"],
  ["FashionMNIST, long tail", "58.47", "**79.67**", "+21.2"],
  ["CIFAR-10, long tail", "33.74", "**44.38**", "+10.6"],
  ["SVHN, long tail", "50.49", "**54.41**", "+3.9"],
  ["SVHN, long tail, with the automatic switch to class rows (Part 5.7)", "50.49", "**65.60**", "+15.1"],
  ["MNIST, 100 clients, long tail", "75.98", "**94.30**", "+18.3"],
  ["FashionMNIST, 50 / 100 clients, long tail", "61.9", "**79.5 / 79.0**", "+17.6 / +17.1"],
  ["Paper's balanced setting vs the best of its 10 methods", "—", "above on MNIST (10/50/100 clients) and CIFAR-10; level on FMNIST; below on SVHN", ""],
], [5, 1.2, 2.4, 1.0], { highlight: [0, 1, 2, 4, 5, 6] }));
C.push(...callout("Honest comparison point (new in this version, Part 6)", [
  "Most of GeFL-F's long-tail loss sits in the **last layer** of each head. Re-fitting only that layer on real data (a known fix: CCVR, CReFF, Fed3R) lifts GeFL-F to 90.7 / 79.2 / 64.8 (MNIST / FMNIST / SVHN, long tail).",
  "Ours-A still beats this stronger baseline (94.8 / 79.7 / 65.6), but by **+4.1 / +0.5 / +0.8**, not +20. The paper must report it.",
]));
C.push(...callout("What changed since version 1", [
  "**Collapse theorem corrected** (Part 2.2): the old \"fewer than half the clients\" rule was wrong in our own data. New exact bound (Theorem 1) and the role of Adam's decay (Lemma 2), checked on the real model.",
  "**New Part 6:** where the remaining error lives (the last layer), a federated fix from exact statistics (ER), and the stronger baseline above.",
  "**New results:** leave-one-out ablation (Part 5.6), SVHN switch (Part 5.7), FashionMNIST at 50 / 100 clients (Part 5.5), covariance version MC-B (Part 8).",
  "**Claims made more precise:** which parts of Ours-A are new and which are known techniques (Parts 4.1, 4.8).",
], GREEN));

// ---------------- PART 1
C.push(H1("1. The baseline: GeFL-F and the full lifecycle of its data"));
C.push(H2("1.1 Setting and notation"));
C.push(B("**Clients.** K clients (K = 10, 50 or 100). Client k holds private data D_k and a classifier head of architecture a(k), one of G = 10 CNNs (CNN-1 … CNN-10), assigned contiguously (K/10 clients per architecture)."));
C.push(B("**Data.** MNIST, FashionMNIST and SVHN use a 10% fraction of the training set (6,000 / 6,000 / 7,326 images); CIFAR-10 uses 50% (25,000). In the paper's setting the data are split IID across clients. Test sets are the full balanced test sets."));
C.push(B("**Models.** One shared feature extractor φ, one shared conditional generator G(h | y), and K private heads h_k."));
C.push(B("**Metric.** The paper's *best_mean_acc*: each architecture's test accuracy at its best round, averaged over the 10 architectures. Under the long tail we report balanced accuracy at the final round, and *tail recall* (mean recall of the four rarest classes)."));
C.push(...img("f01_overview.png", 640, "Figure 1. GeFL-F system view: what is local, what is shared, and which messages travel in each stage."));
C.push(H2("1.2 Model architectures and tensor shapes"));
C.push(...img("f03_architectures.png", 640, "Figure 2. Architectures. The CVAE-F's label projection W (C × 512) holds the only class-specific parameters; this matters in Part 2."));
C.push(...table(["Model", "Input", "Output", "Parameters", "Shared?"], [
  ["Feature extractor φ (Table XIV)", "image resized to 1×32×32 (MNIST, FMNIST) / 3×32×32", "feature h: 3×16×16 = 768 (MNIST, FMNIST, SVHN); CIFAR-10 10×16×16 = 2,560", "36 / 90 / 390", "yes (FedAvg in Stage I, then frozen)"],
  ["Generator G, CVAE-F (Table XX)", "h (+ label map) → z (l = 16; 50 for CIFAR-10)", "ĥ, same shape as h", "≈ 5.5 M", "yes (FedAvg in Stage II, then frozen)"],
  ["Heads CNN-1 … CNN-10 (Tables XXI–XXIII)", "feature h", "10 logits", "5.3 k – 109.8 k", "no (averaged only within an architecture group)"],
], [2.2, 2.0, 3.0, 1.2, 2.0]));
C.push(H2("1.3 Lifecycle: three sequential stages"));
C.push(...img("f02_timeline.png", 640, "Figure 3. The three stages, with rounds, local epochs, data, optimiser and loss of each."));
C.push(H3("Stage I: feature-extractor warm-up (T_FE = 20 rounds; 50 for SVHN and CIFAR-10)"));
C.push(B("**Local data:** the client's real images x with labels y."));
C.push(B("**Local training:** 5 epochs per round; φ and the client's head trained jointly; SGD, learning rate 0.1, batch 64 (128 for CIFAR-10); loss = cross-entropy CE(h_k(φ(x)), y)."));
C.push(B("**Uplink:** φ's weights (36–390 numbers) and the head's weights (5.3 k–110 k numbers)."));
C.push(B("**Aggregation:** φ is averaged over all clients (FedAvg, equation below); heads are averaged within each architecture group."));
C.push(eq("eq_fedavg", "1"));
C.push(B("**Downlink:** the averaged φ and heads. At the end of Stage I, φ is frozen for the rest of training."));
C.push(H3("Stage II: federated feature generator (T_KA = 100 rounds; 200 for CIFAR-10)"));
C.push(B("**Local data:** real features h = φ(x) of the client's own images, with their labels."));
C.push(B("**Local training:** 5 epochs per round of the conditional VAE, Adam with learning rate 10⁻³ and coupled weight decay 10⁻³, batch 64. The loss is the evidence lower bound:"));
C.push(eq("eq_cvae", "2"));
C.push(B("The label enters the decoder through W^T e_y, i.e. the class-c row w_c of the label projection: these C × 512 numbers are the generator's *only* class-specific parameters."));
C.push(B("**Uplink:** every generator weight, about 5.5 million numbers (≈ 22 MB at fp32) per client per round, ≈ 2.2 GB per client over Stage II."));
C.push(B("**Aggregation:** FedAvg of all weights, including W (equation 1). **Downlink:** the averaged generator. At the end of Stage II, G is frozen and every client keeps a copy."));
C.push(H3("Stage III: heterogeneous head training (T_TN = 50 rounds; 100 for SVHN and CIFAR-10)"));
C.push(B("**Synthetic phase:** T_s = 1 epoch on generated features h̃ = G(z, y), with y drawn uniformly over the 10 classes and z ~ N(0, I), sampled locally from the client's own copy of G; loss CE(h_k(h̃), y)."));
C.push(B("**Real phase:** T_r = 5 epochs on the client's real features φ(x); loss CE(h_k(φ(x)), y); SGD, learning rate 0.1."));
C.push(B("**Uplink and aggregation:** head weights, averaged within an architecture group only when several clients share an architecture (K = 50, 100). At K = 10 each architecture has one client, and nothing is exchanged in this stage."));
C.push(...img("f04_round.png", 620, "Figure 4. One Stage-III round. Classes a client does not hold reach its head only through the generator."));
C.push(H2("1.4 Communication summary"));
C.push(...table(["Stage", "Uplink per client per round", "Server operation", "Downlink"], [
  ["I", "φ (36–390) + head (5.3 k–110 k numbers)", "FedAvg(φ); heads within groups", "averaged φ, heads"],
  ["II", "all generator weights, ≈ 5.5 M numbers (≈ 22 MB)", "FedAvg(G)", "averaged G"],
  ["III", "head weights (only if an architecture has several clients)", "average within architecture groups", "averaged heads"],
], [0.6, 3.2, 2.6, 1.6]));
C.push(P("**Privacy:** images, features and labels never leave a client; only model weights move (same as FedAvg)."));
C.push(H2("1.5 Validation of our reproduction"));
C.push(...table(["Setting", "Paper", "Authors' code", "Ours"], [
  ["MNIST IID, K = 10 (best_mean_acc)", "95.47", "96.07", "96.02 ± 0.04"],
  ["MNIST IID, K = 50", "95.04", "–", "95.16"],
  ["MNIST IID, K = 100", "94.63", "–", "94.61"],
  ["FashionMNIST IID, K = 10", "83.14", "–", "82.67 ± 0.31"],
  ["MNIST long tail, seed 0 (final balanced accuracy)", "–", "75.84", "73.92"],
], [4, 1.2, 1.4, 1.6]));
C.push(P("Our GeFL-F matches the paper within 0.5 points in the balanced setting."));

// ---------------- PART 2
C.push(H1("2. The problem: rare classes vanish under long-tailed clients"));
C.push(H2("2.1 The long-tailed federation"));
C.push(P("Class sizes n_c ∝ IF^{c/(C−1)}, IF = 100 (MNIST: 24 to ≈2,400 samples). Client shares: Dirichlet(0.5). Test sets balanced."));
C.push(...img("f05_problem.png", 640, "Figure 5. (a) GeFL-F's accuracy under the long tail. (b) Its tail classes' conditioning rows and fidelity. (c) The collapse threshold of Theorem 1 (Part 2.2)."));
C.push(H2("2.2 Diagnosis: the conditioning-row recursion"));
C.push(P("Write r_c for the generator's row of class c. In one round, m_c of the K clients hold class c and K − m_c do not. A holder's local training changes the row's length by +G_k (data pull it outward); a non-holder sees no class-c data, only the weight-decay gradient, and shortens it by D_k. FedAvg then takes the plain average of the K rows. With p_c = m_c / K, Ḡ_c the average holder gain and D̄_c the average non-holder loss:"));
C.push(eq("eq_collapse", "Theorem 1"));
C.push(B("**Theorem 1 (exact, no assumption on the optimiser).** The row's length after averaging is at most the right-hand side, with equality when all client rows point the same way. It shrinks whenever the holder share is below θ_c."));
C.push(B("**Lemma 2 (why the loss D̄ is large).** Adam divides the gradient by its own running size, so the decay gradient λ·r_c becomes a step of about the learning rate η per coordinate, **whatever λ is** (10⁻⁴ or 10⁻³ give the same step). Only λ = 0 removes it. With plain SGD the same decay would be about 1,000 times weaker here."));
C.push(B("**Consequence.** Rare classes appear in few batches, so their holders gain little (Ḡ small) while non-holders lose a full step: θ_c is close to 1, not ½. The version-1 rule \"rows shrink when fewer than half the clients hold the class\" is the special case Ḡ = D̄, and it is too optimistic."));
C.push(P("Figure 5(c) plots θ against Ḡ / D̄. Measured on MNIST long tail: Ḡ / D̄ = 0.07–0.64, so θ = 0.61–0.93; the K/2 rule would predict no collapse at all."));
C.push(H2("2.3 Evidence"));
C.push(...table(["Measurement (IF = 100, Dir 0.5, K = 10)", "MNIST", "FashionMNIST", "SVHN"], [
  ["Tail-to-head row-norm ratio, GeFL-F", "0.33", "0.22", "0.18–0.49"],
  ["Same, with non-holders excluded from the average (HWA)", "0.80", "1.01", "0.96–1.05"],
  ["Tail fidelity of GeFL-F's generator (referee accuracy on synthetic tail features)", "8.8 %", "19.6 %", "–"],
  ["GeFL-F balanced accuracy: IID → long tail", "96.0 → 75.3", "82.8 → 58.5", "75.8 → 50.5"],
], [4.5, 1.4, 1.6, 1.4], { bad: [0, 2, 3] }));
C.push(P("Ratio far below 1 under the long tail, and restored when non-holders are removed from the average."));
C.push(H3("Direct test of Theorem 1 on the real generator (T01: MNIST long tail, K = 10, 40 rounds, 4 arms)"));
C.push(...img("t01_collapse.png", 640, "Figure 5b. (a) Tail rows shrink to 15 % of head rows with decay 10⁻³ or 10⁻⁴ (identical curves), not without decay, and not with HWA. (b) The bound holds in all 1,200 row-rounds. (c) Rows of classes held by 7–9 of 10 clients still collapse; under HWA they grow."));
C.push(...table(["Arm (same data, same seed)", "Tail / head row length at round 40", "What it shows"], [
  ["GeFL-F, decay 10⁻³ (paper's setting)", "0.152 (tail rows at 4–10 % of start)", "collapse although every class has ≥ 7 of 10 holders"],
  ["GeFL-F, decay 10⁻⁴", "0.155", "the size of the decay does not matter (Lemma 2)"],
  ["GeFL-F, decay 0", "0.848", "only the presence of decay matters"],
  ["HWA, decay 10⁻³", "tail rows grow to 1.17–1.25×", "averaging over holders only removes the D̄ term"],
], [3.0, 3.0, 3.6], { bad: [0, 1], good: [2, 3] }));
C.push(P("**Is this new?** The collapse of class-specific weights under missing classes is known (FedRS, FedLC, FedAwS for classifier layers). What we add is the exact bound for a *generator's* conditioning rows and the Adam-decay explanation. The obvious existing fixes (AdamW, no decay on the rows, lazy decay) are compared with HWA in the prepared job K15 (Part 8)."));
C.push(H2("2.4 Why this breaks knowledge transfer"));
C.push(P("A head trained on fake features is only as good on real data as the fake features are close to real ones (TV distance). For rare classes this gap is large:"));
C.push(eq("eq_tv", "3"));

// ---------------- PART 3
C.push(H1("3. The research journey: every approach we tested"));
C.push(...img("f06_journey.png", 640, "Figure 6. Research map. Green: kept in the final method. Red: tested and dropped on the evidence."));
C.push(H2("3.1 Repairing the generator's aggregation"));
C.push(H3("HWA: holder-weighted aggregation (kept)"));
C.push(P("*Where:* server, Stage II, class rows only. *How:* weight each client's row by its effective sample count (Cui et al., CVPR 2019); non-holders get weight 0. This is an established re-weighting, applied here to the generator's rows; it removes the D̄ term of Theorem 1:"));
C.push(eq("eq_hwa", "4"));
C.push(P("*Result:* +9.0 / +11.2 / +3.4 / +6.0 (MNIST / FMNIST / CIFAR-10 / SVHN, long tail); row ratio back to 0.80–1.05. Later replaced in Ours-A by the mean anchor, which has no class rows at all."));
C.push(H3("LCD: lazy conditioning decay (dropped)"));
C.push(P("*Where:* clients, Stage II. *How:* update and decay only rows of classes present in the batch. *Result:* 78.6 vs 79.1 for HWA; adds nothing."));
C.push(H2("3.2 Training the heads"));
C.push(H3("LA: logit adjustment with the client's own prior (kept; established technique)"));
C.push(P("*Where:* each client, real-data phase. *How:* Menon et al. (ICLR 2021) with the client's own private prior π_k:"));
C.push(eq("eq_la", "5"));
C.push(P("*Result:* +3.6 to +6.3 on all four datasets. Needed because the 5 real epochs pull each head back to its own skew. No effect in IID."));
C.push(H3("CSL: consensus soft labels on synthetic features (kept)"));
C.push(P("*Where:* synthetic phase; clients send only prediction sums on a shared fake pool. *How:* target = mix of the label and the average prediction of all architectures (FedMD, FedDF):"));
C.push(eq("eq_csl", "6"));
C.push(P("*Result:* +2.5 MNIST long tail; +0.5 to +0.8 IID. β = 0.5 better than β = 1."));
C.push(H3("T_s = 10: a larger synthetic budget (kept)"));
C.push(P("*Why (Proposition 5):* more fake training helps only when the generator is faithful (low label bias b)."));
C.push(eq("eq_budget", "Proposition 5"));
C.push(P("*Result:* with the anchored generator +1.7 (MNIST long tail) and +3.9 at K = 100; with the original generator −0.05, as predicted."));
C.push(...table(["Dropped head-side idea", "Placement and model", "Result", "Why it failed"], [
  ["Bayesian CSL", "Replace the fixed mixture by a Bayes posterior under symmetric generator noise, fidelity ρ̂ estimated from ensemble agreement", "83.16 vs 84.22 (FMNIST IID, p = 0.004); 87.3 vs 90.5 (MNIST LT)", "ρ̂ is circular: heads trained on the generator agree with it (ρ̂ → 0.999), so the target collapses to the hard label"],
  ["CSLM", "Interleave consensus-labelled synthetic features into every real batch", "SVHN LT 49.9 vs 62.4", "Mixes the client-prior logit adjustment with balanced synthetic labels in one batch"],
  ["PCM", "Fill each real batch's missing classes with synthetic samples", "FMNIST LT 59.6 vs 65.2 (with LA)", "Needs faithful tail features, which do not exist before the generator is fixed"],
  ["RSA", "Align head embeddings of synthetic and real features on held classes", "synthetic–real gap grew on unheld classes", "Fits the alignment to the held classes only"],
  ["VCT / MixUp", "Mix real features with consensus-labelled synthetic partners (VCT) or with each other (MixUp)", "CIFAR-10 IID: VCT 60.29, MixUp 59.98, neither 60.93; LT tail recall +10 with VCT", "Feature-space mixing does not regularise past the frozen extractor (Part 5.4)"],
], [1.4, 3.0, 2.4, 3.0], { bad: [0, 1, 2, 3, 4] }));
C.push(H2("3.3 Server-side corrections"));
C.push(...table(["Approach", "Placement and model", "Result", "Verdict"], [
  ["SED: server-side ensemble distillation", "Server distils the heads' ensemble into each head on fresh synthetic features", "FMNIST IID 81.69 → 80.96; LT 70.65 → 71.30", "Dropped: bounded by equation 3; the ensemble's knowledge reaches heads only as faithfully as the generator allows"],
  ["BCR: balanced classifier re-training", "Server re-fits each head's last layer on balanced synthetic features (CReFF-like)", "62.0 vs 65.2 for LA (FMNIST LT, quick)", "Dropped: weaker than LA"],
  ["GF: gradient filtering", "Filter conflicting synthetic gradients", "58.5 vs GeFL-F 58.8", "Dropped: no effect"],
  ["BBC: balanced bias calibration (Saerens et al., 2002)", "Fit C logit offsets per head on balanced fake features", "+0.35 to +1.4 under the long tail (MNIST/FMNIST); stays below the last-layer refit in every run", "Kept as an optional, gated post-hoc step; overlaps with LA (Part 5.6); cannot fix a weight error (Part 6)"],
], [2.0, 3.2, 2.4, 2.2], { bad: [0, 1, 2] }));
C.push(H2("3.4 Alternative generators"));
C.push(...table(["Generator", "Placement and model", "Result", "Verdict and reason"], [
  ["FSG: Gaussian from exact statistics", "Server builds N(m_c, Σ_pooled) from secure-aggregated sums; no generator training", "MNIST LT 87.0; CIFAR-10 LT 44.2; SVHN LT 44.2 < GeFL-F 50.5", "Dropped: its Bayes head is LDA (eq. 8), which reaches only 21 % on SVHN pixels"],
  ["Hybrid CVAE / Gaussian", "Mixture of the two generators", "87.1 vs 87.9 (full schedule; the quick pass had favoured it)", "Dropped"],
  ["KME-Gen", "Generator trained only at the server, by MMD to exact kernel embeddings", "MNIST 89.9, FMNIST 75.1 (+15 over GeFL-F) at 1/3000 of the communication", "Dropped as main method; noted as an efficient variant"],
  ["DDPM-F (paper's diffusion generator)", "Our components on the paper's best feature generator", "SVHN IID 70.37 vs DDPM GeFL-F 69.77 (p = 0.10); far below 76–77 with CVAE", "Dropped: slow (hours per seed) and weaker here"],
  ["ZP: ex-post latent prior", "Fit z ~ N(m, S) to the encoder means (exact sums)", "PC + ZP 91.69 vs PC + MC 92.31", "Dropped: MC fixes the same issue better"],
  ["PCR: class rows + mean anchor", "Both conditioning signals in one decoder", "Ties Ours-A on MNIST/FMNIST; SVHN LT 60.9 vs 65.6 for rows alone", "Dropped: the anchor under-disperses samples where means are uninformative"],
], [1.7, 2.8, 2.7, 2.6], { bad: [0, 1, 2, 3, 4, 5] }));
C.push(eq("eq_lda", "8"));
C.push(H2("3.5 Moment correction"));
C.push(P("**MC (kept in Ours-A).** Moves each fake class onto its exact mean and spread with the smallest change (Proposition 3): +1.6 MNIST, +1.6 FMNIST over the plain anchored generator."));
C.push(P("**Covariance version (MC-S / MC-B): future scope.** Helps FMNIST and SVHN but not yet complete on every dataset (Part 8)."));

// ---------------- PART 4
const card = (a, b, rows) => table([a, b], rows, [1.5, 8.1], { left: true });
C.push(H1("4. The final architecture: Ours-A"));
C.push(...callout("The single idea", ["FedAvg *estimates* class information and, under a long tail, the estimate collapses.",
  "Secure aggregation *adds up* class sums exactly. Ours-A takes class information from these exact sums."]));

C.push(H2("4.1 Ours-A at a glance"));
C.push(...img("f07_esa.png", 640, "Figure 7. Ours-A inside GeFL-F. Blue: what changes."));
C.push(...table(["#", "Component", "Stage", "Runs on", "Input → output", "Learned?"], [
  ["0", "Feature extractor (FE), same as GeFL-F", "I", "every client", "image 1×32×32 → h 3×16×16 = 768", "yes, then frozen"],
  ["1", "Class statistics", "after I", "client → server", "own features → 4 sums per class (0.3 MB)", "no (exact sums)"],
  ["2", "PC-VAE generator", "II", "every client, FedAvg", "z (16) + m_y (768) → h̃ (768)", "yes, then frozen"],
  ["3", "MC: moment fix", "before III", "server", "fake h̃ → fake with exact mean and spread", "no"],
  ["4", "KH: fingerprint selection", "before III", "server", "4n candidates → best n per class", "no"],
  ["5", "CSL: consensus labels", "III", "server (sums)", "fake h̃ → soft target t (10)", "no"],
  ["6", "LA: client-prior loss", "III", "every client", "real h, y → loss with offset log π_k", "no"],
  ["7", "Private head (10 architectures)", "III", "every client", "h (768) → 10 logits", "yes"],
], [0.4, 2.6, 0.9, 1.6, 3.2, 1.3], { highlight: [1, 2, 3, 4, 5, 6] }));
C.push(P("Rows 1–4 are our contribution (taking class information from exact sums, in three places). Rows 5 and 6 are known techniques (FedMD/FedDF, logit adjustment) that we adopt. Rows 3–6 have no trainable weights."));

C.push(H2("4.2 Step 1: class statistics (one upload, after Stage I)"));
C.push(...card("Step 1", "Class statistics", [
  ["**What**", "Each client sends, for each of the 10 classes, four sums of its own frozen features."],
  ["**When**", "Once, right after Stage I (the FE is frozen, so features never change again)."],
  ["**How**", "Secure aggregation (Bonawitz et al., CCS 2017): the server receives only the total over all clients."],
  ["**Why**", "These totals are exact. Nothing is learned and nothing is averaged, so FedAvg cannot shrink them."],
]));
C.push(...table(["Sum sent by client k for class c", "Size", "Server computes", "Meaning"], [
  ["n_kc: number of samples", "1", "n_c = Σ_k n_kc", "class size"],
  ["S_kc = Σ h", "768", "m_c = S_c / n_c", "class mean (used by PC-VAE and MC)"],
  ["Q_kc = Σ ‖h‖²", "1", "V_c = Q_c / n_c − ‖m_c‖²", "class spread (used by MC)"],
  ["Φ_kc = Σ φ(h), φ = random Fourier features", "6,144", "μ_c = Φ_c / n_c", "class fingerprint (used by KH)"],
], [3.4, 0.8, 2.6, 3.0]));
C.push(eq("eq_stats", "9"));
C.push(P("Total: 10 × (1 + 768 + 1 + 6,144) ≈ 69,000 numbers ≈ **0.3 MB per client, once**. GeFL-F already sends 22 MB per round."));

C.push(H2("4.3 Step 2: the generator PC-VAE (replaces CVAE-F in Stage II)"));
C.push(...img("f13_pcvae.png", 640, "Figure 8. PC-VAE with every shape. Blue boxes: where the exact class mean enters."));
C.push(...card("Step 2", "PC-VAE", [
  ["**Problem it fixes**", "In CVAE-F the class enters only through a learned row w_y. A client without class c only shrinks w_c (weight decay), so rows of rare classes collapse (Theorem 1, Part 2.2)."],
  ["**What changes**", "The class enters through the exact mean m_y. It goes in twice: as 3 extra input channels of the encoder, and through a linear layer A (768 → 512) added to the decoder input."],
  ["**Output**", "h̃ = ReLU(m_y + decoder output). The network learns only how a sample *differs* from its class mean."],
  ["**Why it works**", "There is no class-specific weight, so there is nothing for FedAvg to shrink. The shape of 'difference from the mean' is shared by all classes, so every client trains it, rare classes included."],
  ["**Source**", "Class prototypes as shared knowledge: FedProto (Tan et al., AAAI 2022)."],
  ["**Cost**", "5.94 M parameters (CVAE-F: 5.55 M). Same training as GeFL-F: loss ‖h̃ − h‖² + KL, Adam lr 0.001, wd 0.001, batch 64, 5 local epochs, 100 rounds (CIFAR-10: 200)."],
]));
C.push(eq("eq_pc", "10"));
C.push(...table(["Layer", "MNIST / FMNIST / SVHN", "CIFAR-10"], [
  ["Encoder input [h ; m_y]", "6×16×16", "20×16×16"],
  ["Conv 4×4 stride 2 → BN → ReLU (×4)", "64×8×8 → 128×4×4 → 256×2×2 → 512×1×1", "same"],
  ["Linear → μ, log σ²", "16 + 16", "50 + 50"],
  ["Decoder input: Linear(z) + A m_y → ReLU", "512×1×1", "512×1×1"],
  ["Transposed Conv 4×4 stride 2 (×4)", "256×2×2 → 128×4×4 → 64×8×8 → 3×16×16", "… → 10×16×16"],
  ["Output ReLU(m_y + ·)", "3×16×16 = 768", "10×16×16 = 2560"],
], [3.4, 4.0, 2.2]));

C.push(H2("4.4 Step 3: moment fix (MC)"));
C.push(...card("Step 3", "MC", [
  ["**Problem it fixes**", "A VAE's samples are too narrow (under-dispersed), and the mean of a rare class can be slightly off."],
  ["**What**", "For each class, measure the fake mean m̃_c and spread Ṽ_c (from 500 generator samples), then move every fake sample x:  x′ = ReLU( m_c + √(V_c / Ṽ_c) · (x − m̃_c) )."],
  ["**Result**", "The fake class now has the *exact* real mean m_c and spread V_c."],
  ["**Why this map (Proposition 3)**", "Of all maps that give the exact mean and spread, this one moves the samples the least. It corrects the class but keeps the shape the generator learned."],
  ["**Source**", "Optimal transport between distributions, scalar case (Olkin and Pukelsheim, 1982)."],
  ["**Cost**", "No training. Two numbers per class (a shift and a scale), computed by the server."],
]));
C.push(eq("eq_mc", "11"));

C.push(H2("4.5 Step 4: fingerprint selection (kernel herding, KH)"));
C.push(...card("Step 4", "KH", [
  ["**Problem it fixes**", "Mean and spread can be right while the *shape* is still wrong (e.g. a fake sub-cluster that real data does not have)."],
  ["**What**", "Draw 4n corrected candidates of class c and keep the n whose average fingerprint is closest to the real fingerprint μ_c."],
  ["**Why it works (Lemma 1)**", "A head's error on real data differs from its error on fake data by at most B · MMD(fake, real). KH directly lowers this MMD."],
  ["**Source**", "Kernel herding (Chen, Welling and Smola, UAI 2010); MMD (Gretton et al., JMLR 2012)."],
  ["**Cost**", "No training. One matrix product per selected sample; uses only μ_c, already uploaded."],
]));
C.push(P("**Algorithm** (greedy, without replacement):"));
C.push(N("compute fingerprints φ(x_i) of the 4n candidates (6,144 numbers each); set w = μ_c;"));
C.push(N("pick the candidate with the largest ⟨φ(x_i), w⟩;"));
C.push(N("update w ← w + μ_c − φ(x_picked); repeat step 2 until n samples are picked."));
C.push(eq("eq_kh", "12"));
C.push(...img("ev07_rare_class_mnist.png", 640, "Figure 9. Real data, MNIST's rarest class (24 samples): the generator's output, after MC (right centre and width) and after KH (right shape). Details in Part 5.9."));

C.push(H2("4.6 Step 5: training the heads (Stage III)"));
C.push(...img("f14_round3.png", 640, "Figure 10. One Stage III round. The fake pool is made fresh every round."));
C.push(H3("(a) 10 epochs on the fake pool, with consensus soft labels (CSL)"));
C.push(...card("Step 5a", "CSL", [
  ["**What**", "Target t = ½ · one-hot(y) + ½ · (average of softmax predictions of the 10 architectures' heads on the same fake sample)."],
  ["**How**", "All clients use one shared fake pool per round. Clients upload only *sums* of predictions (≈ 24 KB per fake epoch), never their heads' outputs on real data."],
  ["**Why it works**", "Different architectures make different errors. Their average is a softer and more reliable label than the generator's label alone. β = 0.5 was better than β = 1."],
  ["**Source**", "FedMD (Li and Wang, 2019); FedDF (Lin et al., NeurIPS 2020)."],
]));
C.push(eq("eq_csl", "6"));
C.push(H3("(b) 5 epochs on the client's own real features, with logit adjustment (LA)"));
C.push(...card("Step 5b", "LA", [
  ["**What**", "Loss = CE( head(h) + τ · log π_k , y ), τ = 1, where π_k is the client's own label distribution."],
  ["**Why it works**", "A client with 90 % of one class learns to predict that class. Adding log π_k during training removes this bias; at test time the offset is dropped."],
  ["**Privacy**", "π_k never leaves the client."],
  ["**Source**", "Logit adjustment (Menon et al., ICLR 2021); FedLC (Zhang et al., ICML 2022)."],
]));
C.push(eq("eq_la", "5"));
C.push(H3("(c) Why 10 fake epochs instead of GeFL-F's 1"));
C.push(P("**Proposition 5:** more fake training helps only when the generator is faithful. Measured on MNIST (long tail): +1.7 with PC-VAE + MC + KH, +3.9 at 100 clients, but −0.05 with the original generator."));
C.push(H3("(d) Aggregation and test time"));
C.push(B("Heads are averaged only with heads of the **same architecture**, as in GeFL-F."));
C.push(B("Test: image → frozen FE → 3×16×16 → the client's own head → 10 logits. No generator is needed."));

C.push(H2("4.7 How each step adds up (same seeds, long tail, balanced accuracy %)"));
C.push(...table(["Step", "MNIST", "FMNIST", "What it fixes"], [
  ["GeFL-F", "75.25", "58.47", "—"],
  ["+ LA", "78.92", "64.78", "client's own class skew"],
  ["+ HWA (class rows averaged over holders)", "87.89", "76.02", "rare rows no longer pulled down"],
  ["+ CSL", "90.40", "76.60", "better labels on fake features"],
  ["PC-VAE replaces HWA", "90.75", "75.96", "no class rows at all"],
  ["+ MC", "92.31", "77.56", "exact mean and spread"],
  ["+ KH", "93.11", "78.22", "fake ≈ real (low MMD)"],
  ["+ T_s = 10  = **Ours-A**", "**94.82**", "**79.67**", "uses the now-faithful generator more"],
], [3.6, 1.2, 1.2, 3.6], { highlight: [7] }));

C.push(H2("4.8 Where each part comes from"));
C.push(...table(["Part of Ours-A", "Source (research paper)", "What we use from it"], [
  ["Class sums sent safely", "Bonawitz et al., Practical Secure Aggregation, ACM CCS 2017", "server gets only the total over clients"],
  ["Class mean as class identity", "Tan et al., FedProto, AAAI 2022", "class prototypes (means) as shared knowledge"],
  ["Moment fix (MC)", "Optimal transport, scalar case (Olkin and Pukelsheim 1982)", "smallest change that gives the exact mean and spread"],
  ["Fingerprint selection (KH)", "Chen, Welling and Smola, Kernel Herding, UAI 2010; Gretton et al., MMD, JMLR 2012", "pick samples whose average fingerprint matches the class"],
  ["Client-prior loss (LA)", "Menon et al., Logit Adjustment, ICLR 2021; Zhang et al., FedLC, ICML 2022", "loss that undoes a client's own class skew"],
  ["Consensus labels (CSL)", "Li and Wang, FedMD, 2019; Lin et al., FedDF, NeurIPS 2020", "average prediction of different models as soft label"],
], [2.4, 3.8, 3.4]));
C.push(P("**What is new in Ours-A:** taking class information from exact secure-aggregation sums instead of from FedAvg-trained weights, and using it in three places (generator input, moment fix, sample selection). Each building block exists; the combination, its use inside a heterogeneous generator-based pipeline, and the collapse analysis that motivates it are ours. LA, CSL and HWA are adopted, not claimed."));

C.push(H2("4.9 Communication and privacy"));
C.push(...table(["Release", "Size", "When", "Protection"], [
  ["Class counts, sums of h and of ‖h‖², sums of random features", "≈ 0.3 MB per client", "once, after Stage I", "secure aggregation; optional (ε, δ)-DP"],
  ["Consensus labels: prediction sums on a shared fake pool", "≈ 24 KB per fake epoch", "each Stage-III round", "sums only; no real data"],
  ["Generator", "5.94 M params ≈ 23.7 MB (GeFL-F: 22 MB)", "each Stage-II round", "same as GeFL-F"],
  ["Feature extractor, heads", "as in GeFL-F", "as in GeFL-F", "unchanged"],
], [3.6, 2.6, 1.6, 1.8]));
C.push(P("For formal privacy, features are clipped and each release gets Gaussian noise (zero-concentrated DP):"));
C.push(eq("eq_dp", "13"));

// ---------------- PART 5
C.push(H1("5. Results"));
C.push(H2("5.1 Protocol"));
C.push(P("All runs follow the paper's appendix: the same models, optimisers, rounds and data fractions; 3 seeds; each seed fixes the split and the initialisation for every method, so comparisons are paired; p-values come from paired t-tests. Local runs used one laptop GPU; heavy runs used Kaggle T4 × 2 sessions."));
C.push(H2("5.2 Main result: long-tailed clients"));
C.push(...table(["Method (IF = 100, Dir 0.5, K = 10; balanced accuracy)", "MNIST", "FashionMNIST", "CIFAR-10", "SVHN"], [
  ["GeFL-F (Kang et al., 2025)", "75.25", "58.47", "33.74", "50.49"],
  ["GeFL-F + logit adjustment (≈ FedLC)", "78.92", "64.78", "37.30", "56.44"],
  ["+ HWA (count-anchored rows) + LA", "87.89", "76.02", "40.73", "62.45"],
  ["+ HWA + LA + consensus labels", "90.40", "76.60", "42.10", "–"],
  ["Gaussian generator from exact statistics (FSG) + LA", "87.04", "73.60", "44.17", "44.23"],
  ["GeFL-F + last-layer refit on pooled real data (upper bound of CCVR / CReFF / Fed3R-type recalibration)", "90.7", "79.2", "–", "64.8"],
  ["**Ours-A**", "**94.82**", "**79.67**", "**44.38**", "54.41"],
  ["**Ours-A with the SVHN switch** (class rows + HWA + MC + KH when class means are uninformative)", "94.82", "79.67", "44.38", "**65.60**"],
  ["oracle: Ours-A's heads, last layer re-fit on pooled real data", "94.93", "81.55", "50.64", "59.13 / 68.4 (switch)"],
], [4.6, 1.2, 1.4, 1.2, 1.2], { highlight: [6, 7], bad: [5] }));
C.push(P("The red row is the strongest baseline we found, and it is new in this version: it shows how much of GeFL-F's loss a known classifier recalibration would recover. Ours-A beats it by +4.1 (MNIST), +0.5 (FashionMNIST) and +0.8 (SVHN with the switch). On FashionMNIST the margin is within noise; on CIFAR-10 the number is not yet logged (Part 8)."));
C.push(P("Tail recall: 51.9 → 91.1 (MNIST), 35.4 → 73.6 (FashionMNIST). SVHN is the weak case (Part 5.7)."));
C.push(...img("f08_longtail.png", 640, "Figure 11. Long-tailed clients: Ours-A against GeFL-F and the strongest alternatives."));
C.push(H2("5.3 The paper's own balanced setting, against all ten of its methods"));
C.push(...table(["best_mean_acc, IID", "GeFL-F (paper)", "Best of the paper's ten methods", "Our GeFL-F", "**Ours-A**", "Ours-A − best"], [
  ["MNIST, K = 10", "95.47", "96.44 (GeFL, DDPM)", "96.01", "**97.62**", "+1.18"],
  ["MNIST, K = 50", "95.04", "95.04 (GeFL-F, CVAE-F)", "95.43", "**96.93**", "+1.89"],
  ["MNIST, K = 100", "94.63", "94.63 (GeFL-F, CVAE-F)", "94.88", "**96.77**", "+2.14"],
  ["FashionMNIST, K = 10", "83.14", "84.28 (GeFL-F, DDPM-F)", "82.76", "83.96", "−0.32"],
  ["FashionMNIST, K = 50", "82.21", "82.96 (GeFL-F, DDPM-F)", "81.50", "**82.96**", "±0.00"],
  ["CIFAR-10, K = 10", "55.86", "59.36 (GeFL, DDPM)", "59.10", "**60.93**", "+1.57"],
  ["SVHN, K = 10 (mean anchor)", "76.26", "76.26 (GeFL-F, CVAE-F)", "75.82", "74.38", "−1.88"],
  ["SVHN, K = 10 (with the switch to class rows)", "76.26", "76.26 (GeFL-F, CVAE-F)", "75.82", "**77.07**", "+0.81"],
  ["FashionMNIST, K = 10, with MC-B (covariance version)", "83.14", "84.28 (GeFL-F, DDPM-F)", "82.76", "**85.75**", "+1.47"],
], [2.0, 1.2, 2.2, 1.2, 1.0, 1.1], { highlight: [0, 1, 2, 5, 7, 8] }));
C.push(P("Above the best of the paper's 10 methods on MNIST (all client counts), CIFAR-10 and, with the switch, SVHN; level on FashionMNIST with the main method and above it with MC-B. Note: best_mean_acc picks each architecture's best round on the test set, which flatters every method equally; our long-tail results use the final round instead."));
C.push(...img("f09_iid.png", 640, "Figure 12. The paper's balanced setting: Ours-A against the paper's GeFL-F and the best of its ten methods."));
C.push(H2("5.4 Data augmentation (paper's Table IV), and why MixUp does not transfer"));
C.push(...table(["CIFAR-10, IID, K = 10", "Accuracy"], [
  ["FedAvg / + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "55.65 / 60.07 / 58.95 / 53.96 / 56.99"],
  ["Image-space GeFL + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "62.67 / 61.66 / 56.47 / 59.97"],
  ["Ours-A, no augmentation", "60.93"],
  ["Ours-A + feature-space MixUp (our run, same seeds)", "59.98"],
  ["Ours-A + VCT (our MixUp variant)", "60.29"],
], [6, 3]));
C.push(P("Ours-A beats 7 of 9 without augmentation. The two higher ones train full models on raw images; in GeFL-F the extractor is frozen, and feature-space MixUp lowered our accuracy (59.98 vs 60.93)."));
C.push(H2("5.5 More clients, and formal privacy"));
C.push(...img("f11_scaling_privacy.png", 640, "Figure 13. (a) 100 clients, MNIST long tail. (b) Ours-A with every statistic made (ε, 10⁻⁵)-differentially private."));
C.push(...table(["Setting", "GeFL-F", "HWA + LA", "FSG + LA", "**Ours-A**"], [
  ["MNIST long tail, K = 100", "75.98", "83.07", "86.05", "**94.30**"],
], [3.6, 1.4, 1.4, 1.4, 1.4], { highlight: [0] }));
C.push(P("Only −0.5 from 10 to 100 clients (the CVAE with HWA loses 4.8): no class weights to estimate from shrinking client data. FashionMNIST long tail (3 seeds): Ours-A 79.7 / 79.5 / 79.0 at K = 10 / 50 / 100, against GeFL-F 58.5 / 61.9 / 61.9."));
C.push(...table(["Long tail, K = 10", "GeFL-F (no formal guarantee)", "Ours-A exact", "Ours-A, ε = 8", "Ours-A, ε = 2"], [
  ["MNIST", "75.25", "94.82", "90.36", "85.06"],
  ["FashionMNIST", "58.47", "79.67", "72.77", "65.59"],
], [2.2, 2.4, 1.4, 1.4, 1.4]));
C.push(P("Privacy costs 4.5–6.9 points at ε = 8 and 9.8–14.1 at ε = 2, mostly on the rarest classes; still +7 to +15 over GeFL-F."));
C.push(H2("5.6 What each component contributes"));
C.push(...img("f10_ablation.png", 640, "Figure 14. Adding components one at a time on the same seeds (long tail)."));
C.push(P("Each step adds on the same seeds. Kernel herding helped in all 9 paired tests (p = 0.004)."));
C.push(H3("Leave-one-out ablation (K09: remove one part from the full Ours-A; long tail, 3 seeds, paired)"));
C.push(...table(["Removed", "MNIST (change, points)", "FashionMNIST (change, points)", "Reading"], [
  ["PC-VAE (back to CVAE + HWA)", "−4.72 (p = 0.02)", "−0.67", "large on MNIST, small on FMNIST"],
  ["LA", "−2.98 (p = 0.02)", "−6.18 (p = 0.02)", "largest; mostly recovered by post-hoc BBC"],
  ["CSL", "−1.63", "−1.38 (p = 0.008)", "medium"],
  ["MC", "−0.96 (p = 0.03)", "−0.49", "small; works through the spread"],
  ["KH", "−0.71", "−0.30", "small"],
], [2.6, 2.0, 2.2, 2.8]));
C.push(P("Full Ours-A in this run: 95.18 ± 0.08 (MNIST), 79.79 ± 0.31 (FMNIST). All 30 seed-level differences are negative (every part, every seed, both datasets; sign test p = 2·10⁻⁹). With 3 seeds no single part survives a multiple-test correction; two more seeds (job K14, ready) are needed for per-part significance."));
C.push(H2("5.7 The weak case: SVHN"));
C.push(...table(["SVHN", "GeFL-F", "+ HWA + LA", "Ours-A"], [
  ["long tail, balanced accuracy", "50.49", "62.45", "54.41"],
  ["balanced setting, best_mean_acc", "75.82", "75.64", "74.38"],
], [3.8, 1.6, 1.6, 1.6]));
C.push(P("SVHN's class means are almost identical (nearest-mean accuracy 12 %, chance 10 %), so a mean-anchored generator cannot separate its classes."));
C.push(P("**Switch (implemented):** the server measures how well the exact class means separate the classes (F, from the same sums). When F < 0.05 it keeps GeFL-F's class rows, protected by HWA, and still applies MC and KH. On SVHN this gives **65.60 ± 1.50** under the long tail (+15.1 over GeFL-F) and **77.07 ± 0.19** in the balanced setting (above the paper's 76.26). MNIST, FMNIST and CIFAR-10 are unaffected (F = 0.07–0.60)."));
C.push(H2("5.8 Mechanism measurements"));
C.push(...table(["Quantity (long tail)", "GeFL-F", "Ours-A"], [
  ["Tail fidelity (referee accuracy on synthetic tail features), MNIST / FMNIST", "10.3 / 19.9 %", "88.6 / 78.7 %"],
  ["Tail spread relative to real data, raw → after correction and herding", "–", "0.39–0.49 → 0.93–0.98"],
  ["IID fidelity vs the referee's accuracy on real data (FMNIST)", "–", "0.837 vs 0.839"],
], [5, 2, 2]));
C.push(P("After the fixes, fake features are as hard as real ones, not artificially clean."));

C.push(H2("5.9 Proof on real data: what each component does"));
C.push(P("Measured on real features: MNIST and FashionMNIST, long tail, 3 seeds, the same splits as every run (Kaggle T4; a second run on another GPU gives the same numbers within about 1 point). The *referee* is a head trained on held-out real features; it never takes part in training. Code: experiments/D01_component_evidence.py."));
C.push(...img("ev01_collapse.png", 640, "Rows of rare classes under FedAvg end at 0.33 (MNIST) and 0.44 (FMNIST) of their start value; with HWA they grow (×1.7, ×2.4). They shrink even when 70–90 % of clients hold the class."));
C.push(...img("ev02_fidelity.png", 640, "Rare-class fake samples recognised as their class: GeFL-F 10 % / 20 %, + HWA 31 % / 71 %, PC-VAE 93 % / 88 %, Ours-A 90 % / 79 % (real test features: 97 % / 84 %)."));
C.push(...img("ev03_moments.png", 640, "Spread of fake features (1 = real): PC-VAE 0.47 / 0.45, + MC 0.85 / 0.90, + KH 0.95 / 0.98. Rare-class mean error: 0.22 → 0.09 → 0.05 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Class separability F of fake features: GeFL-F 0.09 / 0.32 (classes overlap), PC-VAE 0.78 / 1.35 (too clean), Ours-A 0.32 / 0.61, real 0.30 / 0.60."));
C.push(...img("ev04_mmd.png", 640, "Distance to real test features (MMD, all classes): GeFL-F 0.22 / 0.25 → PC-VAE 0.20 / 0.19 → + MC 0.089 / 0.093 → + KH 0.071 / 0.078. Lower MMD goes with higher real accuracy (r = −0.69 / −0.47), as Lemma 1 says."));
C.push(...img("ev05_tstr.png", 640, "Head trained only on fake features, tested on real data (balanced accuracy): GeFL-F 57 / 48, + HWA 81 / 76, PC-VAE 92 / 76, + MC 94 / 77, + KH 94 / 76. Rare classes on MNIST: 32 → 89. KH does not add in this test (−0.2 / −1.1); its gain shows only inside the full pipeline (+0.8 / +0.7)."));
C.push(...img("ev07_rare_class_mnist.png", 640, "MNIST's rarest class (24 real samples): GeFL-F's fake class 9 lies on other classes; PC-VAE is on the class but too narrow; MC and KH spread it over the real class."));
C.push(...img("ev08_la_csl.png", 640, "LA (seed 0): balanced accuracy 92.0 → 95.2 (MNIST), 75.5 → 79.9 (FMNIST); correlation between a client's own class shares and its predictions 0.64 → 0.44, 0.67 → 0.52. CSL: the soft target is closer to the real-data referee than the one-hot label, lower or equal in every class (TV 0.100 → 0.092, 0.212 → 0.202)."));
C.push(...img("ev09_ts.png", 520, "More fake training helps more with the better generator (MNIST: −0.05 vs +1.71)."));
C.push(...img("ev06_separability.png", 640, "How well the exact class mean identifies the class (3 seeds): F = 0.30 / 0.60 / 0.07 / 0.012 and nearest-mean accuracy 82 / 69 / 41 / 12 % on MNIST / FMNIST / CIFAR-10 / SVHN (chance 10 %). SVHN's means carry almost no class information: the reason Ours-A is weak there."));

// ---------------- PART 6 (new in v2)
C.push(H1("6. Where the remaining error lives: the last layer"));
C.push(P("Every run also logs two diagnostics. **Last-layer refit (oracle):** keep each head's body, re-fit only its last layer on the pooled real training features with balanced sampling. **Ensemble:** average the 10 heads' predictions. The difference *refit − head* is the error in the last layer alone (the *readout gap*)."));
C.push(...img("f16_readout.png", 600, "Figure 15. Head as trained vs the same head with its last layer re-fit on real data (long tail, 3 seeds)."));
C.push(...table(["Run (long tail, 3 seeds)", "Head", "Last layer refit", "Readout gap"], [
  ["GeFL-F, MNIST / FMNIST / SVHN", "74.2 / 58.7 / 50.5", "90.7 / 79.2 / 64.8", "**16.5 / 20.5 / 14.3**"],
  ["+ HWA + LA, MNIST / FMNIST", "87.8 / 76.0", "92.4 / 80.1", "4.6 / 4.1"],
  ["Ours-A, MNIST / FMNIST", "94.8 / 79.7", "94.9 / 81.6", "0.1 / 1.9"],
  ["Ours-A, FMNIST at K = 10 / 50 / 100", "79.7 / 79.5 / 79.0", "81.6 / 81.2 / 81.0", "1.9 / 1.7 / 2.0"],
  ["Ours-A (switch), SVHN; Ours-A, CIFAR-10", "65.6; 46.1", "68.4; 52.3", "2.8; **6.2**"],
], [3.6, 2.0, 2.0, 2.0], { bad: [0], highlight: [2] }));
C.push(B("**GeFL-F's long-tail failure is mostly in the last layer.** Its bodies are good; a last-layer refit adds 14–20 points."));
C.push(B("**Ours-A removes most of it** (16.5 → 0.1 on MNIST) *and* gives better bodies: Ours-A's refit beats GeFL-F's refit by +4.2 / +2.4 / +3.6."));
C.push(B("**What is left does not shrink with more clients** (1.7–2.0 on FMNIST at every K) and is largest on CIFAR-10 (6.2)."));
C.push(H2("6.1 A federated way to close the gap: exact readout (ER)"));
C.push(P("The refit above needs pooled real data, which federated learning forbids. But the same answer can be built from sums. For each architecture g, every client passes its real features through head g's body and sends, by secure aggregation, per-class sums of the embeddings e, the class counts and the sum of e·eᵀ. The server then sets the last layer in closed form (linear discriminant analysis with a uniform class prior):"));
C.push(B("class means μ_c and shared covariance Σ from the sums (shrunk: 0.9·Σ + 0.1·tr(Σ)/d·I, fixed in advance);"));
C.push(B("weights w_c = Σ⁻¹ μ_c, bias b_c = −½ μ_cᵀ Σ⁻¹ μ_c."));
C.push(P("Cost: one extra round; under 1.1 M numbers per architecture for the largest head. It discloses federation-level class statistics of the embeddings, the same kind of release as the class sums of Part 4.2."));
C.push(H3("Test (T03, pre-registered; 1 seed, shortened schedule on CPU)"));
C.push(...table(["Run", "Head", "Refit (pooled data)", "**ER (sums only)**", "BBC (offsets only)"], [
  ["FMNIST GeFL-F", "61.2", "78.7", "**79.1**", "65.1"],
  ["FMNIST + HWA + LA", "73.0", "79.4", "**79.9**", "73.6"],
  ["FMNIST Ours-A", "75.7", "80.7", "**80.3**", "77.9"],
  ["MNIST GeFL-F", "72.4", "91.0", "**90.8**", "75.5"],
  ["MNIST + HWA + LA", "82.1", "91.8", "**91.3**", "82.0"],
  ["MNIST Ours-A", "91.1", "93.6", "**92.8**", "92.2"],
], [2.6, 1.2, 1.9, 1.9, 1.9]));
C.push(B("**Passed:** ER is within 0.8 points of the pooled-data refit in all runs (9 of 9 including the follow-up), +18 points on plain GeFL-F. Logit offsets (BBC, LA) stay below the refit in every run: the error is in the weights, not only in a class prior."));
C.push(B("**Where the error sits (measured):** on classes a client does **not hold** (FMNIST Ours-A: −17 recall points against the refit), less on classes held with 1–20 samples (−7), none on well-held classes."));
C.push(B("**Failed prediction (reported honestly):** we predicted the cause was a wrong synthetic class mean; that did not explain the per-class gap. Mis-scaled weight lengths explain about a quarter (re-normalising them recovers 16–34 % of the gap). The full mechanism is still open."));
C.push(H2("6.2 What this means for the paper"));
C.push(B("**ER is not a new idea**: it is the known classifier recalibration from federated statistics (CCVR, NeurIPS 2021; CReFF, IJCAI 2022; Fed3R, ICML 2024), extended to heterogeneous heads (one set of statistics per architecture). We present it that way."));
C.push(B("**GeFL-F + recalibration must be a baseline.** Against it Ours-A wins by +4.1 / +0.5 / +0.8, not +19.6 / +21.2 / +15.1. With recalibration applied to both, Ours-A keeps +4.2 / +2.4 / +3.6."));
C.push(B("ER is now logged automatically (er_bal) in every new run, at no extra GPU cost."));

// ---------------- PART 7
C.push(H1("7. Why it works, and its limits"));
C.push(B("**Removes the cause:** the generator has no class-specific weight, so FedAvg cannot shrink rare classes (holds at 100 clients)."));
C.push(B("**Smallest correction:** the moment fix changes fake features as little as possible (Proposition 3)."));
C.push(B("**Targets the transfer error:** fingerprint selection reduces the term that bounds real-data error (Lemma 1)."));
C.push(B("**Same protocol and privacy model as GeFL-F:** one small extra upload of class sums; formal DP possible."));
C.push(H2("Limitations"));
C.push(B("SVHN: class means carry no class information, so Ours-A is below GeFL-F in the balanced setting (74.4 vs 75.8)."));
C.push(B("Balanced data: gains are +1 to +2 points; the method is built for imbalance."));
C.push(B("The frozen feature extractor limits every method (CIFAR-10 oracle ≈ 51 under the long tail)."));
C.push(B("On CIFAR-10 and SVHN, 10 fake epochs give a higher peak but a lower final accuracy than 1."));
C.push(B("Against a known last-layer recalibration of GeFL-F, the margin is large only on MNIST (+4.1); on FashionMNIST it is +0.5 (Part 6)."));
C.push(B("Per-component significance needs 2 more seeds (K14); the collapse test T01 has 1 seed (the bound itself is exact)."));
C.push(B("Two audit points to state: averaging BatchNorm running variances drops the between-client part of the variance; best_mean_acc selects the best round on the test set."));

// ---------------- PART 8
C.push(H1("8. Status and next steps"));
C.push(...table(["Item", "Why", "Status"], [
  ["Automatic switch to class rows when class means are weak", "fix SVHN", "**done**: SVHN 54.4 → 65.6 (LT), 77.07 (IID)"],
  ["Leave-one-out ablation (K09)", "prove each part is needed", "**done**: 30/30 negative; per-part significance needs K14"],
  ["FashionMNIST at 50 / 100 clients", "complete the grid", "**done**: 79.5 / 79.0"],
  ["Collapse theorem rewritten and tested (T01)", "the version-1 rule was wrong", "**done** (1 seed)"],
  ["Last-layer analysis and ER (T03)", "where the remaining error is", "**done** (1 seed, short schedule); mechanism open"],
  ["K14: ablation seeds 3–4 and MC-B (Kaggle, ≈ 4 GPU-h)", "per-part significance; MC-B stability", "notebooks ready, not run"],
  ["K15: existing collapse fixes (AdamW, no decay on rows, lazy decay) vs HWA at K = 10/50/100 (≈ 7 GPU-h)", "the first question a reviewer will ask", "notebooks ready, not run"],
  ["CIFAR-10 / SVHN long tail with ER logged (≈ 6 GPU-h)", "GeFL-F + recalibration baseline on the hard datasets", "proposed; needs approval (re-runs existing settings)"],
  ["Covariance correction MC-B", "exact covariance where data suffice", "FMNIST: 81.57 (LT), 85.75 (IID); MNIST unstable on one seed"],
  ["Paper's other FL baselines under the long tail", "complete comparisons", "optional"],
], [4.2, 2.6, 2.8]));

// ---------------- APPENDIX
C.push(H1("Appendix A. Proofs"));
C.push(P("**Theorem 1.** Let r_k be client k's row after local training and r⁺ = (1/K) Σ_k r_k. Define G_k = ‖r_k‖ − ‖r‖ for holders and D_k = ‖r‖ − ‖r_k‖ for non-holders. By the triangle inequality ‖r⁺‖ ≤ (1/K) Σ_k ‖r_k‖ = ‖r‖ + p Ḡ − (1 − p) D̄, with equality if and only if all r_k point the same way. The right side is below ‖r‖ exactly when p < D̄ / (Ḡ + D̄). ∎"));
C.push(P("**Lemma 2.** A non-holder's gradient on r_c is exactly λ r_c. Adam's step is η·m̂/(√v̂ + ε); multiplying every gradient by λ multiplies m̂ and √v̂ by λ, so the step is unchanged as long as λ|r| ≫ ε. Hence D̄ does not depend on λ > 0, and D̄ = 0 at λ = 0. Under SGD the same step would be η λ ‖r‖. ∎"));
C.push(P("**Proposition 3.** With Y = T(X), E‖Y − X‖² = V_c + Ṽ_c + ‖m_c − m̃_c‖² − 2E⟨Y − m_c, X − m̃_c⟩. By Cauchy–Schwarz the last term is at most √(V_c Ṽ_c), with equality if and only if Y − m_c = √(V_c/Ṽ_c)(X − m̃_c). ∎"));
C.push(P("**Proposition 5.** With w = λn_s/(n_r + λn_s), the error becomes σ²(1 − w)/n_r + w²b², a convex parabola minimised at w* = σ²/(2n_rb²). The optimum is clipped to the feasible range, so the optimal synthetic share is non-increasing in b and reaches its maximum as b → 0. ∎"));
C.push(P("**Lemma 1** is the standard RKHS bound: |E_G ℓ − E_P ℓ| = |⟨ℓ, μ_G − μ_P⟩| ≤ ‖ℓ‖·MMD(G, P)."));
C.push(H1("Appendix B. Hyperparameters and experiment inventory"));
C.push(...table(["Dataset", "Fraction / train size", "Batch", "T_FE / T_KA / T_TN", "Latent l", "Feature h"], [
  ["MNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×16×16"],
  ["FashionMNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×16×16"],
  ["SVHN", "0.1 / 7,326", "64", "50 / 100 / 100", "16", "3×16×16"],
  ["CIFAR-10", "0.5 / 25,000", "128", "50 / 200 / 100", "50", "10×16×16"],
], [1.6, 2.0, 0.9, 2.0, 1.1, 1.6]));
C.push(P("Ours-A settings: 6,144 random Fourier features; herding pool 4n; consensus weight β = 0.5; LA temperature τ = 1; T_s = 10, T_r = 5."));
C.push(...table(["Experiment series", "Question answered"], [
  ["F01, F10, E20 (local)", "Main long-tail comparison and the first anchored generator"],
  ["E16, E21, E22 (local)", "Synthetic budget (Proposition 5); 100 clients"],
  ["E24, E25 (local)", "Kernel herding, KME-Gen, the anchored stack"],
  ["E26, E27, E29, E31, E32 (local)", "Formal DP; IID at 50/100 clients; hybrid"],
  ["K01–K04 (Kaggle)", "CIFAR-10 and SVHN baselines; regimes, privacy, ablations; client scaling; diffusion generator"],
  ["K08, K10, K11, K13 (Kaggle)", "Ours-A on CIFAR-10 and SVHN; SVHN switch; hybrid; VCT vs MixUp"],
  ["K09 (Kaggle)", "Leave-one-out ablation of Ours-A"],
  ["E32 (local)", "MC-B, covariance version of the moment fix"],
  ["T01, T03 (local CPU, pre-registered)", "Collapse theorem on the real generator; last-layer gap and ER"],
  ["K14, K15 (Kaggle, ready, not run)", "Ablation seeds 3–4; existing collapse fixes vs HWA"],
], [3.5, 6]));
C.push(P("All code, logs and per-seed results are in the project repository (run ledger: experiments/LEDGER.md), and every number in this report can be regenerated from the logged runs."));

// ---------- document
const doc = new Document({
  creator: "[Student Name]", title: "Ours-A for Generator-Based Heterogeneous FL",
  
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 32, bold: true, color: BLUE, font: "Calibri" }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, color: "243B63", font: "Calibri" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, color: "3B5B8F", font: "Calibri" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
                                 { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Ours-A · Research Progress · v2", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "OursA_Backup_Handout.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024 / 1024).toFixed(1), "MB");
});
