// Builds the handout (Word) from figures in ./fig. Run: python figures.py && node build_handout.js
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
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], keepNext: true, spacing: { before: 360, after: 200 } });
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
  text: "Where Does Heterogeneous Federated Learning Lose Rare Classes?", bold: true, size: 40, color: BLUE })] }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({
  text: "GeFL-F under long-tailed clients: three causes, a rule that predicts the first, and the fixes", italics: true, size: 24, color: GREY })] }));
for (const l of ["Submitted by: [Student Name]   ·   [Roll No.]", "Supervisor: [Professor Name]", "[Department], [Institute]", "Version 4 · 22 October 2026"])
  C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: l, size: 24 })] }));
C.push(br());

// Summary
C.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Summary")], spacing: { after: 200 } }));
C.push(P("GeFL-F lets clients with different model architectures learn from each other through a shared feature generator. It was tested only on balanced clients. We studied it under **long-tailed clients** (a few common classes, many rare ones) and compared every stage with the same pipeline trained on all the data in one place."));
C.push(B("**Three causes of the lost accuracy, each measured:** (1) Adam's weight decay on clients that lack a class shrinks that class's generator parameters to zero (Part 2); (2) each model's **last layer** is fitted to its own client's skew, the largest loss: 7–20 points (Part 3); (3) synthetic data help a model only if they carry their label (Part 4)."));
C.push(B("**A rule computed from class counts alone predicts the first cause before training:** all 352 flagged class rows collapsed, and a prediction written down before new runs held for 16 of 16 rows (Part 2)."));
C.push(B("**Each cause has a direct fix**, and with all three the federated pipeline is within 0–2 points of centralised training on the three datasets where we could measure it: what remains is the long tail itself (Part 6.2)."));
C.push(B("**Honest margin:** on new seeds with every setting chosen on validation data, our method is −0.3 to +1.9 points from the strongest of nine baselines, significant on two of five datasets (Part 6.1). The work is therefore written as an **analysis paper**, not a new state of the art. A 9-page LaTeX version exists."));
C.push(B("**The mechanism is general:** the same collapse removes rare item embeddings in federated recommendation (MovieLens-1M, Part 7)."));
C.push(...callout("Changed since version 3", [
  "**Kernel herding is now optional.** It adds only +0.1 to +0.4 points with the exact readout (positive in 8 of 8 paired seeds) but makes the head stage about 6× slower (Part 5).",
  "**New results:** CIFAR-10 with 50 clients and SVHN under other imbalance settings (Part 6.3).",
  "**This report is shorter:** the development history, superseded variants and per-component detail are removed; they remain in the project's working draft.",
]));

// 1. Setting
C.push(H1("1. The setting"));
C.push(...img("f01_overview.png", 620, "Figure 1. GeFL-F. Stage I trains a small shared feature extractor; Stage II trains a conditional feature generator by FedAvg; Stage III trains each client's private head on generated and real features."));
C.push(B("**Data:** MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100-LT. Long tail: imbalance 100 : 1 between the most and least common class, split across 10 clients by a Dirichlet(0.5) draw, so clients see very different class mixes."));
C.push(B("**Models:** as in the GeFL-F paper (ten head architectures, so at 10 clients each architecture has one client). Our GeFL-F reproduces the published balanced-data numbers within 0.5 points."));
C.push(B("**Metric:** balanced accuracy (mean recall over classes) on balanced test sets, final round. A difference of accuracies is given in *points*."));
C.push(B("**Fairness of the main comparison:** new seeds 3–5, every method's settings chosen on 2,000 held-out training images, the analysis and predictions fixed before any run, 95% confidence intervals with Holm correction."));
C.push(H2("Where the accuracy goes (GeFL-F vs the same pipeline trained centrally)"));
C.push(...table(["Loss", "Size", "Cause", "Fix"], [
  ["Last layer", "7–20 points", "fitted to one client's class mix", "exact readout from aggregated sums (Part 3)"],
  ["Generator collapse", "7–11 points (10 clients)", "Adam's coupled weight decay on clients that lack a class", "decoupled decay, or holder-weighted averaging (Part 2)"],
  ["Model body on one client's data", "3–4 points", "small sample, missing classes", "synthetic data that carry their label (Parts 4, 5)"],
  ["Label-free synthetic data", "up to 2 points", "they flatten the model's outputs", "a better generator (Part 4)"],
  ["Shared feature extractor", "0–1 point", "trained on long-tailed data, as centrally", "not addressed"],
], [2.2, 1.6, 3.2, 3.2], { left: true }));

// 2. Generator collapse
C.push(H1("2. Cause 1: the generator loses rare classes"));
C.push(P("The generator stores each class as one row of 512 numbers (its *conditioning row*); nothing else in it is class-specific. In one round each client trains its copy and the server averages the copies. A client that has no samples of class c still applies weight decay to row c. Under **Adam**, which rescales every step to roughly the learning rate, that decay moves the row toward zero at full speed, while clients that hold the class barely grow it. Averaging then shrinks the row."));
C.push(eq("eq_collapse", "1"));
C.push(P("Here r_c is the row, p_c the share of clients holding class c, Ḡ_c the holders' mean growth and D̄_c the non-holders' mean shrinkage in one round. Because D̄_c ≫ Ḡ_c under Adam, rare-class rows shrink **even when most clients hold the class**. On MNIST every class is held by 7 or more of 10 clients, yet the three rarest rows shrank to 4–10% of their length."));
C.push(H2("A rule from counts alone"));
C.push(P("Each client's number of local steps and the chance that a batch contains class c are known before training. Their ratio predicts the outcome:"));
C.push(eq("eq_kappa", "2"));
C.push(...img("v01_kappa.png", 640, "Figure 2. (a) 900 class rows from all logged GeFL-F runs: every row with κ < 1 collapsed (352 of 352). (b) A prediction written down before the SVHN runs with 50 and 100 clients: 16 of 16 flagged rows collapsed under GeFL-F's averaging, none under holder-weighted averaging."));
C.push(P("**Why this does not happen centrally:** with all data in one place, Adam's step size for a row is set by its data gradients, as long as the class appears at least once every ~1,000 steps (Adam's memory). Every class does here. In federation a non-holding client sees only the decay."));
C.push(H2("Two fixes, both work"));
C.push(eq("eq_hwa", "3"));
C.push(P("**Holder-weighted averaging (HWA)** averages each row only over clients that hold the class (equation 3; it needs only sums, so it works under secure aggregation). **Decoupled decay (AdamW)** removes the cause instead. Balanced accuracy, long tail, 3 seeds:"));
C.push(...table(["", "MNIST 10", "MNIST 50", "MNIST 100", "FMNIST 10", "FMNIST 50", "FMNIST 100"], [
  ["GeFL-F (coupled decay)", "79.2", "81.4", "79.5", "65.2", "66.5", "66.6"],
  ["+ AdamW (decoupled decay)", "86.4", "84.7", "81.6", "75.9", "74.0", "71.8"],
  ["+ HWA (decay kept)", "**87.9**", "**86.5**", "**83.1**", "76.0", "**75.5**", "**73.3**"],
], [2.6, 1, 1, 1, 1, 1, 1], { cap: "Removing the decay recovers 2–11 points; HWA adds 1.3–1.8 more at 50–100 clients on MNIST, where rare rows otherwise also learn more slowly." }));

// 3. Last layer
C.push(H1("3. Cause 2: the last layer"));
C.push(P("At 10 clients each architecture's head is one client's model, so its last layer reflects that client's class mix: classes the client lacks are almost never predicted. This is the **largest single loss**."));
C.push(B("**Per-client logit corrections cannot fix the model body.** Logit adjustment, FedLC, FedRS and balanced softmax are exact reparametrisations of a single-client head: they change only the last-layer bias (or a learning rate), never the body. Measured: heads trained with any of the four reach the same accuracy after the exact readout, within 0.5 points on every seed and dataset."));
C.push(B("**The exact readout (ER) fixes it exactly.** Each client sends per-class sums of its embeddings; the server computes a class-balanced linear discriminant (equation 4). Since sums do not depend on how data are split across clients, this equals the centralised readout for the same model bodies."));
C.push(eq("eq_er", "4"));
C.push(...table(["", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], [
  ["GeFL-F, as trained", "74.5", "59.2", "49.6", "33.9", "7.8"],
  ["GeFL-F + ER", "90.4", "79.3", "64.7", "51.3", "15.2"],
  ["Ours + ER", "**94.4**", "**81.5**", "**66.7**", "**53.9**", "**18.6**"],
  ["Ours − GeFL-F, both + ER", "+3.9", "+2.2", "+2.0", "+2.6", "+3.4"],
], [2.6, 1, 1, 1, 1, 1], { cap: "Development seeds 0–2. ER alone lifts GeFL-F by 7–20 points; with the same readout, our model bodies are 2–4 points better. Every comparison below uses a readout for every method." }));

// 4. Synthetic data
C.push(H1("4. Cause 3: synthetic data must carry their label"));
C.push(P("If a synthetic feature carries no information about its label, training on it with uniformly drawn labels is exactly a penalty on the spread of the model's outputs (equation 5): it teaches the model to be unsure, and it does so precisely for the classes the client lacks."));
C.push(eq("eq_logitvar", "5"));
C.push(P("So the useful quantity is the **signal fraction** s, the share of generated features that carry their label (0 = none, 1 = like real data). The theory predicts a sign change, and Figure 3 shows it: on CIFAR-100, GeFL-F's generator (s = 0.002) makes the models 2.2 points *worse* than using no synthetic data; ours (s = 0.09) makes them 1.8 points better."));
C.push(...img("v03_signal_fraction.png", 520, "Figure 3. Gain of the synthetic phase (accuracy with ER minus that of heads trained on real data only) against the signal fraction, new seeds 3–5."));

// 5. Method
C.push(H1("5. The method"));
C.push(P("The common principle: wherever the data decide the result, use a statistic that secure aggregation computes **exactly** (a sum over clients) instead of a parameter that averaging only estimates."));
C.push(...img("m01_method.png", 640, "Figure 4. The method inside GeFL-F."));
C.push(...table(["Step", "What it does", "Why"], [
  ["Mean anchor", "The generator receives the exact class mean and models only the variation around it (no class row)", "Nothing class-specific left to collapse (Part 2)"],
  ["Gate", "If class means are too close to tell apart (separation F < 0.05, only SVHN), use class rows with HWA instead", "A mean anchor cannot separate classes whose means coincide"],
  ["Moment correction (MC)", "Shift and scale each class's samples to the exact class mean and spread", "The smallest change that gives the right first two moments"],
  ["Logit adjustment (LA)", "Each client trains on its real data with its own class prior added to the outputs", "Keeps the real phase from erasing what synthetic data taught about missing classes"],
  ["Consensus labels (CSL)", "Synthetic features are labelled ½ by their class and ½ by the average prediction of all architectures", "Lowers the label error of imperfect synthetic features"],
  ["Exact readout (ER)", "Last layer from aggregated sums, once after training", "Recovers the largest loss exactly (Part 3)"],
], [2, 4.5, 3.5], { left: true }));
C.push(H2("Kernel herding: now optional"));
C.push(P("Kernel herding (KH) keeps, out of 4n generated samples per class, the n whose average random-feature fingerprint is closest to the real class's. It lowers the distribution distance (MMD) but barely changes accuracy:"));
C.push(...table(["Removing KH", "With ER", "Heads as trained", "Head-stage time with KH"], [
  ["MNIST (5 seeds)", "−0.26", "−0.67 (5 of 5 seeds)", "5.8×"],
  ["FashionMNIST (5 seeds)", "−0.13", "−0.65", "6.1×"],
  ["CIFAR-10 (2 seeds)", "−0.37", "−0.22", "6.8×"],
  ["KH replaced by a random subset (FMNIST / CIFAR-10)", "−0.16 / −0.23", "−0.49 / −0.23", "3.9× / 3.7×"],
], [3.6, 1.6, 2.4, 2.4], { cap: "Existing ablation runs, paired by seed. With ER all 8 paired seeds favour KH (sign test p = 0.008), by 0.05–0.43 points." }));
C.push(P("A larger fingerprint or a larger candidate pool changes nothing, so a better approximation of the same kernel would not help either. **Decision:** KH is optional. The new-seed results in Part 6 still include it; without it our margins would be about 0.1–0.4 points lower (measured on MNIST, FashionMNIST and CIFAR-10)."));

// 6. Results
C.push(H1("6. Results"));
C.push(H2("6.1 Main comparison: new seeds, nine baselines"));
C.push(...img("v02_fair_retest.png", 640, "Figure 5. Ours minus each baseline, every method with its own validation-chosen readout; black edge: significant after Holm correction."));
C.push(...table(["Ours minus", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], [
  ["GeFL-F", "**+3.4**", "+1.0", "+1.5", "**+2.4**", "**+4.0**"],
  ["GeFL-F + AdamW + LA (decay fix)", "**+2.0**", "+0.5", "+1.5", "+1.5", "**+3.0**"],
  ["Real data only + LA (= FedLC = FedRS = balanced softmax)", "**+4.7**", "**+2.6**", "**+11.3**", "+1.5", "**+1.6**"],
  ["Gaussian generator from exact statistics + LA", "**+2.9**", "**+1.3**", "**+12.4**", "+0.5", "**+1.6**"],
  ["FedDF (server-side distillation)", "**+1.9**", "+0.3", "+0.5", "+2.3", "**+3.5**"],
  ["FedDF + LA", "**+2.0**", "+0.3", "−0.3", "+1.7", "**+3.5**"],
  ["**Margin over the strongest**", "**+1.9**", "+0.3", "−0.3", "+0.5", "**+1.6**"],
], [4, 1, 1, 1, 1, 1], { cap: "Bold: significant (Holm-adjusted p < 0.05 over nine baselines). Our pre-registered prediction of ≥ 2 points over every baseline on four datasets failed." }));
C.push(P("**Reading.** Removing the decay and adding the exact readout recovers most of GeFL-F's loss; our generator adds 0.5–3.0 on top. FedDF is strong once it also gets an exact readout. Part 6.2 explains why no method can be far ahead here."));
C.push(H2("6.2 Federated versus centralised"));
C.push(...img("v04_centralised.png", 640, "Figure 6. The same pipeline trained on all data in one place (green) versus federated (blue), exact readout, seed 0."));
C.push(...table(["", "MNIST", "FashionMNIST", "CIFAR-100*"], [
  ["Long-tail drop, our federated pipeline", "2.9", "3.3", "2.7"],
  ["Long-tail drop, centralised pipeline", "3.2", "2.7", "4.1"],
  ["Remaining federated loss (centralised − ours)", "0.0", "1.9", "−0.6"],
], [4.6, 1.6, 1.6, 1.6], { cap: "*CIFAR-100: 3 of the 10 architectures. SVHN and CIFAR-10 are not yet measured (in notebook NB71)." }));
C.push(P("Our federated pipeline loses about as much to the long tail as centralised training does. This also bounds what any better generator could add over the decay fix: at most 1.9 points on MNIST and 1.4 on FashionMNIST."));
C.push(H2("6.3 More clients and other imbalance settings"));
C.push(...table(["Exact readout", "SVHN, 50 clients", "SVHN, 100 clients", "CIFAR-10, 50 clients"], [
  ["GeFL-F", "58.0", "58.8", "–"],
  ["Decay fix (AdamW + LA)", "60.3", "59.9", "45.1"],
  ["HWA + LA", "60.3", "60.8", "45.2"],
  ["Ours", "**61.8**", "**61.0**", "**46.1**"],
], [3.4, 2, 2, 2], { cap: "With 5 or 10 clients per architecture. On CIFAR-10 every method loses 6.5–7 points to the 10-client setting, and HWA ≈ decay fix." }));
C.push(...table(["Ours − decay fix (FMNIST / SVHN)", "Imbalance 10", "Dirichlet 0.1", "Dirichlet 1.0", "Random rare classes"], [
  ["Exact readout", "+1.0 / +2.6", "+1.5 / +1.7", "+0.8 / +1.0", "+0.6 / +1.1"],
], [3.2, 1.7, 1.7, 1.7, 1.7], { cap: "All three pre-registered predictions hold on both datasets; margins are small (three seeds)." }));

// 7. Recommendation
C.push(H1("7. The same collapse in federated recommendation"));
C.push(P("Cause 1 needs only parameter rows that few samples use, clients that never use some of them, local Adam with coupled weight decay, and plain averaging. Federated matrix factorisation has all four: each movie has an embedding row that is averaged across clients. MovieLens-1M, 10 or 50 clients, 3 seeds, predictions fixed before the runs:"));
C.push(...table(["", "Rare items collapsed (10 / 50 clients)", "Error on those items (RMSE)", "Error overall"], [
  ["Adam + coupled decay, FedAvg", "85% / 83%", "1.094 / 1.064", "0.859 / 0.908"],
  ["AdamW, FedAvg", "0% / 0%", "1.036 / 0.992", "0.904 / 0.907"],
  ["Adam + decay, HWA", "23% / 9%", "1.042 / **0.967**", "**0.857** / **0.894**"],
  ["Centralised Adam + coupled decay", "78%", "1.099", "0.888"],
], [3.2, 2.4, 2.2, 2.2]));
C.push(...img("v06_movielens.png", 520, "Figure 7. Final item-embedding length against how often an item appears in a batch. Below Adam's memory (one in 1,000 steps) items collapse even centrally; federation makes it about ten times worse."));
C.push(P("**Two of four predictions failed:** not every flagged item collapses (83–85% do), and centralised training also collapses rare items that appear less than once per 1,000 steps. In recommendation, federation makes the collapse much more severe rather than creating it. This is one dataset."));

// 8. What did not work, limitations, status
C.push(H1("8. What did not work, limitations, next steps"));
C.push(...table(["Idea", "Result", "Why"], [
  ["Kernel herding as a core component", "+0.1 to +0.4 with ER, 6× head-stage time", "Closer in distribution is not the same as more useful; now optional"],
  ["Gate that drops label-free synthetic classes", "Helps the decay-fixed GeFL-F (+0.8, +1.3 on the CIFARs); no change for ours", "The server's estimate saturates for our generator"],
  ["Class-balanced loss for the shared extractor", "Neutral on MNIST / FashionMNIST, −8.2 / −4.2 on the CIFARs", "Unnormalised weights act as a 20× step size"],
  ["Exact readout recomputed during training", "No gain, much worse calibration", "Training under a fixed readout harms the body"],
  ["Covariance version of moment correction", "+0.4 / +0.6 with ER (MNIST / FashionMNIST); on SVHN beaten by class rows + HWA", "Small gain for its cost"],
], [3, 3.6, 3.4], { left: true }));
C.push(H2("Limitations"));
C.push(B("Margins over strong baselines are small; the claim is the analysis, not a new state of the art."));
C.push(B("The shared extractor is small and frozen, as in GeFL-F (CIFAR-100 accuracy is 15–19% for every method)."));
C.push(B("The centralised comparison covers three datasets with one seed; SVHN and CIFAR-10 are pending."));
C.push(B("The released class sums leak membership for rare classes (attack success 0.63–0.73 against 0.5 for chance); adding Gaussian noise (ε = 8) brings tail classes to 0.50–0.58 at a cost of 4.5–6.9 points."));
C.push(H2("Status and next steps"));
C.push(...table(["Item", "State"], [
  ["9-page paper (LaTeX) and full working draft", "Written; numbers checked against the result files"],
  ["NB71: centralised comparison on SVHN / CIFAR-10, remaining partial-participation runs", "Built, not yet run. Decides whether a better generator can still gain anything on those datasets"],
  ["K24: DCGAN generator under the long tail", "Not reached in NB70; to be confirmed whether it ran elsewhere"],
  ["Target venue", "To be chosen; the paper then moves to that venue's template"],
], [5, 5], { left: true }));
C.push(P("All code, per-seed results and the run ledger are in the project repository; every number in this report can be regenerated from the logged runs."));

// ---------- document
const doc = new Document({
  creator: "[Student Name]", title: "Where Does Heterogeneous Federated Learning Lose Rare Classes?",
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
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Rare classes in heterogeneous FL · Research Progress · v4", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Handout.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024 / 1024).toFixed(1), "MB");
});
