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
  text: "Why rare classes vanish from GeFL-F, how Ours-A brings them back, and how far federated training is from centralised training", italics: true, size: 24, color: GREY })] }));
for (const l of ["Submitted by: [Student Name]   ·   [Roll No.]", "Supervisor: [Professor Name]", "[Department], [Institute]", "Version 3 · 21 October 2026"])
  C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: l, size: 24 })] }));
C.push(br());

// Summary
C.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Summary")], spacing: { after: 200 } }));
C.push(B("**Problem:** under long-tailed clients GeFL-F loses 21–25 points; rare classes are almost never predicted."));
C.push(B("**Three causes, located and measured:** (1) the generator's class rows of rare classes are shrunk to zero by Adam's weight decay on clients that lack the class (Part 2); (2) each head's **last layer** is fitted to its own client's skew, the largest loss (7–20 points, Part 6); (3) each head's body sees one client's data, and fake data help it only if they carry their label (Part 7)."));
C.push(B("**A rule from counts alone predicts which classes the generator loses** before training: all 352 class rows it flagged collapsed, and a prediction written down before the SVHN 50 / 100-client runs came true for all 16 flagged rows (Part 2.5)."));
C.push(B("**Ours-A** (exact class statistics in the generator, moment correction, sample selection, consensus labels, logit adjustment) plus an **exact readout** (the last layer from secure-aggregated sums) is close to centralised training on MNIST, FashionMNIST and CIFAR-100: the remaining loss is the long tail itself, present without federation too (Part 7)."));
C.push(...table(["Result (long tail, 10 clients)", "MNIST", "FashionMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], [
  ["GeFL-F as published (heads as trained)", "74.5", "59.2", "49.6", "33.9", "7.8"],
  ["GeFL-F + exact readout (a strong, fair baseline)", "90.4", "79.3", "64.7", "51.3", "15.2"],
  ["**Ours-A + exact readout**", "**94.4**", "**81.5**", "**66.7**", "**53.9**", "**18.6**"],
  ["Ours-A − GeFL-F, both with the exact readout (seeds 0–2)", "+3.9", "+2.2", "+2.0", "+2.6", "+3.4"],
  ["**Fair re-test:** new seeds 3–5, every setting chosen on validation data, Ours − strongest of 9 baselines", "+1.9 (sig.)", "+0.3", "−0.3", "+0.5", "+1.6 (sig.)"],
], [4.4, 1.1, 1.3, 1.0, 1.1, 1.1], { highlight: [2], bad: [4] }));
C.push(...callout("Honest standing (new in this version)", [
  "Against GeFL-F with the same last-layer fix, Ours-A is ahead by **+2.0 to +3.9** points on all five datasets. Against the **strongest** published baseline on new seeds with validation-chosen settings (server-side distillation FedDF, the Gaussian generator FSG, the decay fix), the margin is **−0.3 to +1.9**, significant only on MNIST and CIFAR-100 (Part 5.3).",
  "The reason is now measured: on MNIST, FashionMNIST and CIFAR-100 the federated pipeline is already within 0–2 points of the same pipeline trained on all the data in one place (Part 7). No federated method can beat centralised training, so the room for a better method is small in this setting.",
  "The paper is therefore best written as an **analysis**: where federated learning loses rare classes, a count rule that predicts it, and the fixes that remove each loss. Two new methods suggested by the theory did not improve Ours-A (Part 3.6).",
]));
C.push(...callout("What changed since version 2", [
  "**New results (Kaggle, 18–20 Oct):** the decay fix vs holder-weighted averaging at 10 / 50 / 100 clients (K15); every method under eight last-layer readouts on five datasets including CIFAR-100 (K16); a fair re-test on new seeds against nine baselines (K17, K19); SVHN with 50 / 100 clients (K18); mechanism controls (K20); five-seed ablation (K14); other imbalance settings and partial participation (K22, K23); per-round measurements (K25); membership attacks on the released sums.",
  "**New Part 7:** the mathematics of federated versus centralised training, stage by stage, with the measurements that test each statement.",
  "**New Part 2.5:** a rule from counts alone that predicts the collapse, confirmed on a dataset it had not seen.",
  "**New Part 9:** the same collapse in federated recommendation (MovieLens), where it is partly present in centralised training too.",
  "**Corrected:** the main comparison is now against a recalibrated GeFL-F and validation-selected baselines, not against GeFL-F as published.",
], GREEN));


// ---------------- HOW TO READ THE NUMBERS (v2)
C.push(H1("How to read the numbers"));
C.push(P("This report uses many measurements. Only four of them decide whether the method works; one exists only to compare with the original paper; the rest are *diagnostics*: they explain why something works or fails (Part 8 walks through each failure), and are not goals in themselves. Every accuracy is in percent; a difference between two accuracies is in *points* (79.7 − 58.5 = +21.2 points)."));
const gl = (rows, cap) => table(["Measurement", "What it measures, in plain words", "How to read it"], rows, [2.2, 4.4, 3.0], { left: true, cap });
C.push(H2("A. The numbers that decide whether the method works"));
C.push(...gl([
  ["**Balanced accuracy**", "Accuracy on each of the 10 classes separately, then averaged, so a rare class counts as much as a common one. Measured on the test set after the last training round.", "10 = guessing, 100 = perfect. **The headline number of this report.**"],
  ["**Rare-class accuracy** (\"tail recall\")", "The same, but averaged over the 4 rarest classes only.", "Shows whether the rare classes came back. Close to balanced accuracy = rare classes are no longer neglected."],
  ["**Gain** (points)", "Accuracy of one method minus another, on the same data splits and starting weights.", "+21 = 21 percentage points better."],
  ["**± and p-value**", "Every result is repeated 3 times with different random splits (\"seeds\"); ± is the spread over the 3. The p-value asks: if the two methods were really equal, how likely is a difference this consistent?", "± small compared with the gain = trustworthy. p below 0.05 is the usual bar. With only 3 repeats, p-values are rough."],
]));
C.push(H2("B. Only for comparison with the original paper"));
C.push(...gl([
  ["**Paper's accuracy** (best_mean_acc)", "For each of the 10 model types, its best test accuracy over all rounds; then the average of the 10.", "Optimistic, because it picks the best round by looking at the test set. We use it only where we compare with the paper's own tables; our own claims use balanced accuracy at the last round."],
]));
C.push(H2("C. Diagnostics: why it works or fails"));
C.push(...gl([
  ["**Row length ratio** (rare / common)", "The original generator stores each class as a row of 512 numbers. We compare the length of the rows of rare classes with those of common classes.", "1 = rare classes treated like common ones. 0.2 = rare-class rows shrank to a fifth: the generator can no longer tell rare classes apart. **Proof of the collapse** (Part 2)."],
  ["**Holder gain / non-holder loss, and the threshold θ**", "Per round: how much the clients that have a class lengthen its row, against how much the clients without it shorten it. θ is the share of clients that must have the class for the row not to shrink.", "Measured θ = 0.61–0.93: a rare class shrinks unless 61–93 % of clients have it, not 50 % as we first claimed."],
  ["**Fake-sample correctness** (\"fidelity\")", "A separate judge classifier, trained only on real data and never used in training, labels the generator's fake samples of a rare class. Share it labels correctly.", "10 % = fakes look like random classes. Ceiling = the judge's score on real data (97 % MNIST, 84 % FashionMNIST)."],
  ["**Spread ratio**", "How varied the fake samples of a class are, divided by how varied the real ones are.", "1 = right variety. 0.45 = fakes too similar to each other."],
  ["**Fake-to-real distance** (maximum mean discrepancy)", "One number for how different the whole cloud of fake features is from the cloud of real features.", "0 = identical; lower is better. Compare only within one dataset."],
  ["**Train-on-fake, test-on-real accuracy**", "Train a classifier only on fake features, test it on real data.", "The most direct test of whether fake features are useful."],
  ["**Class separability** (F) and **nearest-mean accuracy**", "Can the *average* feature of each class tell the classes apart? F: distance between class averages relative to the spread inside a class. Nearest-mean accuracy: label each real sample by the closest class average.", "F near 0 and nearest-mean accuracy near 10 % = class averages are almost identical. Explains why SVHN fails (12 %)."],
  ["**Last-layer gap** (\"readout gap\")", "Accuracy after re-fitting only each model's last layer on pooled real data, minus the accuracy as trained.", "How much accuracy is lost in the last layer alone. GeFL-F loses 7–20 points there (Part 6)."],
  ["**Exact readout** (ER)", "Each model's last layer computed in closed form from sums of real features that secure aggregation returns exactly (linear discriminant analysis).", "Equals the centralised answer for the same model bodies (Part 7). Every main comparison now uses it for every method."],
  ["**κ (kappa)**", "For one class: the steps on which clients that hold the class can grow its generator row, divided by the steps on which clients without it shrink it. Computed from the client class counts before training.", "κ < 1: the row collapses (Part 2.5)."],
  ["**Signal fraction** s", "How much class information the fake features carry: 0 = none (a judge cannot tell their class), 1 = as much as real data.", "Fake data help a model when s is high and hurt when s ≈ 0 (Part 7.4)."],
  ["**Validation-selected**", "A setting (the readout, the number of fake epochs) chosen on 2,000 held-out training images no client holds, never on the test set.", "Used in the fair re-test (Part 5.3). Earlier results chose settings on the test set and are labelled \"development\"."],
  ["**Class-share correlation**", "Correlation between how often a client sees each class and how often its model predicts it.", "0 = predictions not biased toward the client's own common classes. Used to show what logit adjustment does."],
  ["**Total-variation distance of labels**", "How far the soft training labels are from a real-data judge's answers.", "0 = same answers. Used to show what consensus labels do."],
  ["**Privacy budget** ε", "Strength of the formal privacy guarantee when noise is added to the uploaded class sums.", "Smaller = more private. 8 = moderate, 2 = strong."],
]));
C.push(H2("D. The settings"));
C.push(...gl([
  ["**Long tail** (imbalance 100 : 1)", "The most common class has 100 times as many samples as the rarest.", "The setting this work is built for."],
  ["**Balanced** setting", "Every class has the same number of samples, split evenly over clients: the original paper's setting.", "Checks that the method does no harm when there is no imbalance."],
  ["**Dirichlet 0.5**", "How unevenly each class is shared among clients. Smaller = more uneven.", "At 0.5 most clients have several classes missing entirely."],
  ["**Number of clients** (10, 50, 100)", "Devices taking part. There are always 10 model types, so 50 clients means 5 clients per model type.", "More clients = less data each."],
]));
C.push(...callout("If you remember only four numbers", [
  "**Balanced accuracy with the exact readout** (does it work), **margin over the strongest baseline on new seeds** (is it better than what exists), **κ** (which classes the generator loses, predicted from counts), and **last-layer gap** (where the error was).",
  "Short names used in tables: Ours-A = our final method; GeFL-F = the original paper's method; holder-weighted averaging, moment correction, sample selection, consensus labels and logit adjustment are the parts described in Part 4.",
]));

// ---------------- PART 1
C.push(H1("1. The baseline: GeFL-F and the full lifecycle of its data"));
C.push(H2("1.1 Setting and notation"));
C.push(B("**Clients.** K clients (10 clients, 50 or 100). Client k holds private data D_k and a classifier head of architecture a(k), one of G = 10 CNNs (CNN-1 … CNN-10), assigned contiguously (K/10 clients per architecture)."));
C.push(B("**Data.** MNIST, FashionMNIST and SVHN use a 10% fraction of the training set (6,000 / 6,000 / 7,326 images); CIFAR-10 uses 50% (25,000). In the paper's setting the data are split evenly (balanced) across clients. Test sets are the full balanced test sets."));
C.push(B("**Models.** One shared feature extractor φ, one shared conditional generator G(h | y), and K private heads h_k."));
C.push(B("**Metric.** The paper's *best_mean_acc*: each architecture's test accuracy at its best round, averaged over the 10 architectures. Under the long tail we report balanced accuracy at the final round, and *rare-class accuracy* (average accuracy on the four rarest classes; see \"How to read the numbers\")."));
C.push(...img("f01_overview.png", 640, "Figure 1. GeFL-F system view: what is local, what is shared, and which messages travel in each stage."));
C.push(H2("1.2 Model architectures and tensor shapes"));
C.push(...img("f03_architectures.png", 640, "Figure 2. Architectures. The original generator's label projection W (C × 512) holds the only class-specific parameters; this matters in Part 2."));
C.push(...table(["Model", "Input", "Output", "Parameters", "Shared?"], [
  ["Feature extractor φ (Table XIV)", "image resized to 1×32×32 (MNIST, FashionMNIST) / 3×32×32", "feature h: 3×16×16 = 768 (MNIST, FashionMNIST, SVHN); CIFAR-10 10×16×16 = 2,560", "36 / 90 / 390", "yes (FedAvg in Stage I, then frozen)"],
  ["Generator G, original generator (Table XX)", "h (+ label map) → z (l = 16; 50 for CIFAR-10)", "ĥ, same shape as h", "≈ 5.5 M", "yes (FedAvg in Stage II, then frozen)"],
  ["Heads CNN-1 … CNN-10 (Tables XXI–XXIII)", "feature h", "10 logits", "5.3 k – 109.8 k", "no (averaged only within an architecture group)"],
], [2.2, 2.0, 3.0, 1.2, 2.0]));
C.push(H2("1.3 Lifecycle: three sequential stages"));
C.push(...img("f02_timeline.png", 640, "Figure 3. The three stages, with rounds, local epochs, data, optimiser and loss of each."));
C.push(H3("Stage I: feature-extractor warm-up (T_FE = 20 rounds; 50 for SVHN and CIFAR-10)"));
C.push(B("**Local data:** the client's real images x with labels y."));
C.push(B("**Local training:** 5 epochs per round; φ and the client's head trained jointly; stochastic gradient descent, learning rate 0.1, batch 64 (128 for CIFAR-10); loss = cross-entropy cross-entropy(h_k(φ(x)), y)."));
C.push(B("**Uplink:** φ's weights (36–390 numbers) and the head's weights (5.3 k–110 k numbers)."));
C.push(B("**Aggregation:** φ is averaged over all clients (FedAvg, equation below); heads are averaged within each architecture group."));
C.push(eq("eq_fedavg", "1"));
C.push(B("**Downlink:** the averaged φ and heads. At the end of Stage I, φ is frozen for the rest of training."));
C.push(H3("Stage II: federated feature generator (T_KA = 100 rounds; 200 for CIFAR-10)"));
C.push(B("**Local data:** real features h = φ(x) of the client's own images, with their labels."));
C.push(B("**Local training:** 5 epochs per round of the conditional variational autoencoder, Adam with learning rate 10⁻³ and coupled weight decay 10⁻³, batch 64. The loss is the evidence lower bound:"));
C.push(eq("eq_cvae", "2"));
C.push(B("The label enters the decoder through W^T e_y, i.e. the class-c row w_c of the label projection: these C × 512 numbers are the generator's *only* class-specific parameters."));
C.push(B("**Uplink:** every generator weight, about 5.5 million numbers (≈ 22 MB at fp32) per client per round, ≈ 2.2 GB per client over Stage II."));
C.push(B("**Aggregation:** FedAvg of all weights, including W (equation 1). **Downlink:** the averaged generator. At the end of Stage II, G is frozen and every client keeps a copy."));
C.push(H3("Stage III: heterogeneous head training (T_TN = 50 rounds; 100 for SVHN and CIFAR-10)"));
C.push(B("**Synthetic phase:** 1 synthetic epoch epoch on generated features h̃ = G(z, y), with y drawn uniformly over the 10 classes and z ~ N(0, I), sampled locally from the client's own copy of G; loss cross-entropy(h_k(h̃), y)."));
C.push(B("**Real phase:** T_r = 5 epochs on the client's real features φ(x); loss cross-entropy(h_k(φ(x)), y); stochastic gradient descent, learning rate 0.1."));
C.push(B("**Uplink and aggregation:** head weights, averaged within an architecture group only when several clients share an architecture (50 clients, 100). At 10 clients each architecture has one client, and nothing is exchanged in this stage."));
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
  ["MNIST balanced, 10 clients (best_mean_acc)", "95.47", "96.07", "96.02 ± 0.04"],
  ["MNIST balanced, 50 clients", "95.04", "–", "95.16"],
  ["MNIST balanced, 100 clients", "94.63", "–", "94.61"],
  ["FashionMNIST balanced, 10 clients", "83.14", "–", "82.67 ± 0.31"],
  ["MNIST long tail, seed 0 (final balanced accuracy)", "–", "75.84", "73.92"],
], [4, 1.2, 1.4, 1.6]));
C.push(P("Our GeFL-F matches the paper within 0.5 points in the balanced setting."));

// ---------------- PART 2
C.push(H1("2. The problem: rare classes vanish under long-tailed clients"));
C.push(H2("2.1 The long-tailed federation"));
C.push(P("The most common class has 100 times as many training samples as the rarest, with sizes falling evenly (on a log scale) in between (MNIST: about 2,400 down to 24 samples). Each class is shared out among the clients with a Dirichlet(0.5) draw, so most clients have many samples of a few classes and none of several others. Test sets stay balanced."));
C.push(...img("f05_problem.png", 640, "Figure 5. (a) GeFL-F's accuracy under the long tail. (b) Its tail classes' conditioning rows and fidelity. (c) The collapse threshold of Theorem 1 (Part 2.2)."));
C.push(H2("2.2 Diagnosis: the conditioning-row recursion"));
C.push(P("Write r_c for the generator's row of class c. In one round, m_c of the K clients hold class c and K − m_c do not. A holder's local training changes the row's length by +G_k (data pull it outward); a non-holder sees no class-c data, only the weight-decay gradient, and shortens it by D_k. FedAvg then takes the plain average of the K rows. With p_c = m_c / K, Ḡ_c the average holder gain and D̄_c the average non-holder loss:"));
C.push(eq("eq_collapse", "Theorem 1"));
C.push(B("**Theorem 1 (exact, no assumption on the optimiser).** The row's length after averaging is at most the right-hand side, with equality when all client rows point the same way. It shrinks whenever the holder share is below θ_c. Because Ḡ and D̄ are only known after the round, θ_c explains a measured shrinkage after the fact; it does not predict it in advance. A class held by every client can still shrink if the holders' updates point in different directions."));
C.push(B("**Lemma 2 (why the loss D̄ is large).** Adam divides the gradient by its own running size, so the decay gradient λ·r_c becomes a step of about the learning rate η per coordinate, **whatever λ is** (10⁻⁴ or 10⁻³ give the same step), as long as λ·|r| is much larger than Adam's ε. Only λ = 0 removes it. Adam's running averages are kept per client across rounds, so a single step can also exceed η. With plain stochastic gradient descent the same decay would be about 1,000 times weaker here."));
C.push(B("**Consequence.** Rare classes appear in few batches, so their holders gain little (Ḡ small) while non-holders lose a full step: θ_c is close to 1, not ½. The version-1 rule \"rows shrink when fewer than half the clients hold the class\" is the special case Ḡ = D̄, and it is too optimistic."));
C.push(P("Figure 5(c) plots θ against Ḡ / D̄. Measured on MNIST long tail: Ḡ / D̄ = 0.07–0.64, so θ = 0.61–0.93; the K/2 rule would predict no collapse at all."));
C.push(H2("2.3 Evidence"));
C.push(...table(["Measurement (imbalance 100 : 1, Dirichlet 0.5, 10 clients)", "MNIST", "FashionMNIST", "SVHN"], [
  ["Tail-to-head row-norm ratio, GeFL-F", "0.33", "0.22", "0.18–0.49"],
  ["Same, with non-holders excluded from the average ", "0.80", "1.01", "0.96–1.05"],
  ["Tail fidelity of GeFL-F's generator (referee accuracy on synthetic tail features)", "8.8 %", "19.6 %", "–"],
  ["GeFL-F balanced accuracy: balanced → long tail", "96.0 → 75.3", "82.8 → 58.5", "75.8 → 50.5"],
], [4.5, 1.4, 1.6, 1.4], { bad: [0, 2, 3] }));
C.push(P("Ratio far below 1 under the long tail, and restored when non-holders are removed from the average."));
C.push(H3("Direct test of Theorem 1 on the real generator (T01: MNIST long tail, 10 clients, 40 rounds, 4 arms)"));
C.push(...img("t01_collapse.png", 640, "Figure 5b. (a) Tail rows shrink to 15 % of head rows with decay 10⁻³ or 10⁻⁴ (identical curves), not without decay, and not with holder-weighted averaging. (b) The bound holds in all 1,200 row-rounds. (c) Rows of classes held by 7–9 of 10 clients still collapse; under holder-weighted averaging they grow."));
C.push(...table(["Arm (same data, same seed)", "Tail / head row length at round 40", "What it shows"], [
  ["GeFL-F, decay 10⁻³ (paper's setting)", "0.152 (tail rows at 4–10 % of start)", "collapse although every class has ≥ 7 of 10 holders"],
  ["GeFL-F, decay 10⁻⁴", "0.155", "the size of the decay does not matter (Lemma 2)"],
  ["GeFL-F, decay 0", "0.848", "only the presence of decay matters"],
  ["Holder-weighted averaging, decay 10⁻³", "tail rows grow to 1.17–1.25×", "averaging over holders only removes the D̄ term"],
], [3.0, 3.0, 3.6], { bad: [0, 1], good: [2, 3] }));
C.push(P("**Is this new?** The collapse of class-specific weights under missing classes is known (FedRS, FedLC, FedAwS for classifier layers). What we add is the bound for a *generator's* conditioning rows, the Adam-decay explanation, and a rule that predicts the collapse from counts alone (Part 2.5). The obvious existing fixes are compared with holder-weighted averaging in Part 2.6."));
C.push(H2("2.4 Why this breaks knowledge transfer"));
C.push(P("A head trained on fake features is only as good on real data as the fake features are close to real ones (total-variation distance). For rare classes this gap is large:"));
C.push(eq("eq_tv", "3"));
C.push(H2("2.5 Predicting the collapse from counts alone (new in v3)"));
C.push(P("Theorem 1 explains a shrinkage after it has happened. A predictor must use only what is known before training: how many samples of each class every client has, and how many local steps it takes. One Adam step moves each coordinate by about the learning rate. A client without class c takes all its τ_k steps toward zero; a client with it moves the row toward its target only on the steps whose batch contains c (a fraction f_kc of its steps, computable from the counts). With plain averaging the drift per round is proportional to"));
C.push(P("**κ_c = (steps that can grow the row) / (steps that shrink it) = Σ_{holders} τ_k f_kc / Σ_{non-holders} τ_k.**", { align: AlignmentType.CENTER }));
C.push(B("**κ_c < 1:** the shrinking steps outnumber the growing ones in every coordinate, so the row goes to zero whatever the data say."));
C.push(B("**κ_c > 1:** the row settles at a shrunk length that grows with κ_c; a large κ is needed to survive intact."));
C.push(B("**Why centralised Adam with the same decay does not do this:** there, the step for row c is divided by the size of its own *data* gradients, so decay steps are tiny. The collapse needs a client whose only gradient on the row is decay, Adam's normalisation of it, and averaging that client's row back in."));
C.push(...img("v01_kappa.png", 640, "Figure 5c. (a) 900 class rows from every logged GeFL-F run (MNIST, FashionMNIST at 10 / 50 / 100 clients, CIFAR-10, CIFAR-100): all 352 rows with κ < 1 ended below 0.15 of their starting length; Spearman 0.94 between κ and the final length. (b) A prediction written into the paper before the SVHN 50 / 100-client runs returned: 16 of 16 flagged rows collapsed under GeFL-F (largest final length 0.04), none under holder-weighted averaging (Spearman 0.99)."));
C.push(H2("2.6 Fixing the optimiser vs fixing the averaging (K15, new in v3)"));
C.push(P("If the decay on clients without the class causes the collapse, removing the decay should restore the rows as well as holder-weighted averaging does. Long tail, 3 seeds, every arm with logit adjustment, balanced accuracy:"));
C.push(...table(["Arm", "MNIST 10", "MNIST 50", "MNIST 100", "FashionMNIST 10", "FashionMNIST 50", "FashionMNIST 100"], [
  ["GeFL-F + logit adjustment (coupled decay, as in the paper)", "79.2", "81.4", "79.5", "65.2", "66.5", "66.6"],
  ["+ AdamW (decoupled decay)", "86.4", "84.7", "81.6", "75.9", "74.0", "71.8"],
  ["+ no decay on the class rows only", "86.8", "84.4", "81.8", "75.9", "74.5", "71.9"],
  ["+ lazy decay (only rows in the batch)", "85.1", "83.9", "81.7", "75.8", "74.2", "71.6"],
  ["+ holder-weighted averaging (decay kept)", "**87.9**", "**86.5**", "**83.1**", "76.0", "**75.5**", "**73.3**"],
], [3.6, 0.9, 0.9, 0.9, 1.1, 1.1, 1.1], { good: [4] }));
C.push(B("**The diagnosis holds:** every fix restores the rows and recovers 7–11 points at 10 clients. The size of the decay does not matter (AdamW 10⁻² equals 10⁻³); the decay term *on the rows* is the cause."));
C.push(B("**Holder-weighted averaging adds a further 1.3–1.8 points at 50–100 clients on MNIST** (95% intervals exclude zero; same direction, not significant, on FashionMNIST). Without decay a rare row still learns slowly, diluted by the share of clients that hold it; holder averaging removes that too, which matters more as clients multiply. On SVHN at 100 clients it adds +1.8 points over the decay fix (K18)."));
C.push(B("**Practical reading:** removing coupled decay is the first fix and costs nothing; holder-weighted averaging is a refinement worth 1–2 points that grows with the number of clients."));

// ---------------- PART 3
C.push(H1("3. The research journey: every approach we tested"));
C.push(...img("f06_journey.png", 640, "Figure 6. Research map. Green: kept in the final method. Red: tested and dropped on the evidence."));
C.push(H2("3.1 Repairing the generator's aggregation"));
C.push(H3("Holder-weighted averaging (kept)"));
C.push(P("*Where:* server, Stage II, class rows only. *How:* weight each client's row by its effective sample count (Cui et al., CVPR 2019); non-holders get weight 0. This is an established re-weighting, applied here to the generator's rows; it removes the D̄ term of Theorem 1:"));
C.push(eq("eq_hwa", "4"));
C.push(P("*Result:* +9.0 / +11.2 / +3.4 / +6.0 (MNIST / FashionMNIST / CIFAR-10 / SVHN, long tail); row ratio back to 0.80–1.05. Later replaced in Ours-A by the mean anchor, which has no class rows at all."));
C.push(H3("Lazy decay (dropped)"));
C.push(P("*Where:* clients, Stage II. *How:* update and decay only rows of classes present in the batch. *Result:* 78.6 vs 79.1 for holder-weighted averaging; adds nothing."));
C.push(H2("3.2 Training the heads"));
C.push(H3("Logit adjustment with the client's own class shares (kept; established technique)"));
C.push(P("*Where:* each client, real-data phase. *How:* Menon et al. (ICLR 2021) with the client's own private prior π_k:"));
C.push(eq("eq_la", "5"));
C.push(P("*Result:* +3.6 to +6.3 on all four datasets. Needed because the 5 real epochs pull each head back to its own skew. No effect in balanced."));
C.push(H3("Consensus labels on synthetic features (kept)"));
C.push(P("*Where:* synthetic phase; clients send only prediction sums on a shared fake pool. *How:* target = mix of the label and the average prediction of all architectures (FedMD, FedDF):"));
C.push(eq("eq_csl", "6"));
C.push(P("*Result:* +2.5 MNIST long tail; +0.5 to +0.8 balanced. β = 0.5 better than β = 1."));
C.push(H3("10 synthetic epochs: a larger synthetic budget (kept)"));
C.push(P("*Why (Proposition 5):* more fake training helps only when the generator is faithful (low label bias b)."));
C.push(eq("eq_budget", "Proposition 5"));
C.push(P("*Result:* with the anchored generator +1.7 (MNIST long tail) and +3.9 at 100 clients; with the original generator −0.05, as predicted."));
C.push(...table(["Dropped head-side idea", "Placement and model", "Result", "Why it failed"], [
  ["Bayesian consensus labels", "Replace the fixed mixture by a Bayes posterior under symmetric generator noise, fidelity ρ̂ estimated from ensemble agreement", "83.16 vs 84.22 (FashionMNIST balanced, p = 0.004); 87.3 vs 90.5 (MNIST long tail)", "ρ̂ is circular: heads trained on the generator agree with it (ρ̂ → 0.999), so the target collapses to the hard label"],
  ["Consensus labels inside real batches", "Interleave consensus-labelled synthetic features into every real batch", "SVHN long tail 49.9 vs 62.4", "Mixes the client-prior logit adjustment with balanced synthetic labels in one batch"],
  ["Prior-completing batches", "Fill each real batch's missing classes with synthetic samples", "FashionMNIST long tail 59.6 vs 65.2 (with logit adjustment)", "Needs faithful tail features, which do not exist before the generator is fixed"],
  ["Real–synthetic alignment", "Align head embeddings of synthetic and real features on held classes", "synthetic–real gap grew on unheld classes", "Fits the alignment to the held classes only"],
  ["Our mixing variant / MixUp", "Mix real features with consensus-labelled synthetic partners (our mixing variant) or with each other (MixUp)", "CIFAR-10 balanced: our mixing variant 60.29, MixUp 59.98, neither 60.93; long tail tail recall +10 with our mixing variant", "Feature-space mixing does not regularise past the frozen extractor (Part 5.8)"],
], [1.4, 3.0, 2.4, 3.0], { bad: [0, 1, 2, 3, 4] }));
C.push(H2("3.3 Server-side corrections"));
C.push(...table(["Approach", "Placement and model", "Result", "Verdict"], [
  ["Server-side ensemble distillation", "Server distils the heads' ensemble into each head on fresh synthetic features", "FashionMNIST balanced 81.69 → 80.96; long tail 70.65 → 71.30", "Dropped: bounded by equation 3; the ensemble's knowledge reaches heads only as faithfully as the generator allows"],
  ["Balanced classifier re-training", "Server re-fits each head's last layer on balanced synthetic features (CReFF-like)", "62.0 vs 65.2 for logit adjustment (FashionMNIST long tail, quick)", "Dropped: weaker than logit adjustment"],
  ["Gradient filtering", "Filter conflicting synthetic gradients", "58.5 vs GeFL-F 58.8", "Dropped: no effect"],
  ["Bias calibration (Saerens et al., 2002)", "Fit C logit offsets per head on balanced fake features", "+0.35 to +1.4 under the long tail (MNIST/FashionMNIST); stays below the last-layer refit in every run", "Kept as an optional, gated post-hoc step; overlaps with logit adjustment (Part 5.10); cannot fix a weight error (Part 6)"],
], [2.0, 3.2, 2.4, 2.2], { bad: [0, 1, 2] }));
C.push(H2("3.4 Alternative generators"));
C.push(...table(["Generator", "Placement and model", "Result", "Verdict and reason"], [
  ["Gaussian generator from exact statistics", "Server builds N(m_c, Σ_pooled) from secure-aggregated sums; no generator training", "MNIST long tail 87.0; CIFAR-10 long tail 44.2; SVHN long tail 44.2 < GeFL-F 50.5", "Dropped: its Bayes head is linear discriminant analysis (eq. 8), which reaches only 21 % on SVHN pixels"],
  ["Hybrid original generator / Gaussian", "Mixture of the two generators", "87.1 vs 87.9 (full schedule; the quick pass had favoured it)", "Dropped"],
  ["Server-trained generator (kernel matching)", "Generator trained only at the server, by fake-to-real distance to exact kernel embeddings", "MNIST 89.9, FashionMNIST 75.1 (+15 over GeFL-F) at 1/3000 of the communication", "Dropped as main method; noted as an efficient variant"],
  ["Diffusion generator (paper's diffusion generator)", "Our components on the paper's best feature generator", "SVHN balanced 70.37 vs diffusion GeFL-F 69.77 (p = 0.10); far below 76–77 with original generator", "Dropped: slow (hours per seed) and weaker here"],
  ["Latent prior fitted afterwards", "Fit z ~ N(m, S) to the encoder means (exact sums)", "Mean anchor + latent prior 91.69 vs mean anchor + moment correction 92.31", "Dropped: moment correction fixes the same issue better"],
  ["Class rows plus mean anchor", "Both conditioning signals in one decoder", "Ties Ours-A on MNIST/FashionMNIST; SVHN long tail 60.9 vs 65.6 for rows alone", "Dropped: the anchor under-disperses samples where means are uninformative"],
], [1.7, 2.8, 2.7, 2.6], { bad: [0, 1, 2, 3, 4, 5] }));
C.push(eq("eq_lda", "8"));
C.push(H2("3.5 Moment correction"));
C.push(P("**Moment correction (kept in Ours-A).** Moves each fake class onto its exact mean and spread with the smallest change (Proposition 3): +1.6 MNIST, +1.6 FashionMNIST over the plain anchored generator."));
C.push(P("**Covariance correction: promising, not yet final.** Uses each class's exact covariance where the class has enough data: FashionMNIST 81.57 (long tail) and 85.75 (balanced, above the paper's best); unstable on one MNIST seed. Its spread-only variant (Ours-A-B) is +0.75 / +1.71 points better than Ours-A as trained on seeds 3–4 (K14)."));
C.push(H2("3.6 Ideas tested after version 2 (new in v3)"));
C.push(...table(["Idea", "What it does", "Result", "Verdict and reason"], [
  ["Exact readout during training", "Recompute the exact last layer every round and train only the bodies under it", "MNIST 93.9 vs 94.1, FashionMNIST 79.4 vs 79.8 for the readout computed once afterwards (1 seed); worse calibration", "Dropped by its pre-registered rule: compute the readout once, after training"],
  ["Signal-fraction gate on fake data (from the theory, Part 7.4)", "The server estimates, per class, how much label information the fakes carry and either drops classes with none (gate) or labels them by the models' consensus (per-class consensus)", "Decay-fixed GeFL-F: +0.8 (CIFAR-10) and +1.3 (CIFAR-100) points with the exact readout, 3 of 3 seeds. Ours-A: no change", "Confirms the theory on a weak generator. For Ours-A the server's estimate reads ≈ 1 because the moment correction puts fakes exactly on the class means, all a linear judge sees; a better estimator is needed"],
  ["Class-balanced loss in Stage I (from the theory, Part 7.2)", "Weight each sample's Stage-I loss by N / (C · n_class) from the exact global counts", "CPU pilot: +1.0 (FashionMNIST), +0.4 (MNIST). Full pipeline: +0.2 MNIST, 0.0 FashionMNIST, **−8.2 CIFAR-10, −4.2 CIFAR-100** (one CIFAR-100 seed diverged)", "Dropped: the unnormalised weights (up to 20×) act as a much larger step size on clients that hold rare classes"],
], [1.8, 2.8, 2.8, 2.6], { bad: [0, 2] }));

// ---------------- PART 4
const card = (a, b, rows) => table([a, b], rows, [1.5, 8.1], { left: true });
C.push(H1("4. The final architecture: Ours-A"));
C.push(...callout("The single idea", ["FedAvg *estimates* class information and, under a long tail, the estimate collapses.",
  "Secure aggregation *adds up* class sums exactly. Ours-A takes class information from these exact sums."]));

C.push(H2("4.1 Ours-A at a glance"));
C.push(...img("f07_esa.png", 640, "Figure 7. Ours-A inside GeFL-F. Blue: what changes."));
C.push(...table(["#", "Component", "Stage", "Runs on", "Input → output", "Learned?"], [
  ["0", "Feature extractor, same as GeFL-F", "I", "every client", "image 1×32×32 → h 3×16×16 = 768", "yes, then frozen"],
  ["1", "Class statistics", "after I", "client → server", "own features → 4 sums per class (0.3 MB)", "no (exact sums)"],
  ["2", "Mean-anchored generator generator", "II", "every client, FedAvg", "z (16) + m_y (768) → h̃ (768)", "yes, then frozen"],
  ["3", "Moment correction: moment fix", "before III", "server", "fake h̃ → fake with exact mean and spread", "no"],
  ["4", "Sample selection: fingerprint selection", "before III", "server", "4n candidates → best n per class", "no"],
  ["5", "Consensus labels: consensus labels", "III", "server (sums)", "fake h̃ → soft target t (10)", "no"],
  ["6", "Logit adjustment: client-prior loss", "III", "every client", "real h, y → loss with offset log π_k", "no"],
  ["7", "Private head (10 architectures)", "III", "every client", "h (768) → 10 logits", "yes"],
], [0.4, 2.6, 0.9, 1.6, 3.2, 1.3], { highlight: [1, 2, 3, 4, 5, 6] }));
C.push(P("Rows 1–4 are our contribution (taking class information from exact sums, in three places). Rows 5 and 6 are known techniques (FedMD/FedDF, logit adjustment) that we adopt. Rows 3–6 have no trainable weights."));

C.push(H2("4.2 Step 1: class statistics (one upload, after Stage I)"));
C.push(...card("Step 1", "Class statistics", [
  ["**What**", "Each client sends, for each of the 10 classes, four sums of its own frozen features."],
  ["**When**", "Once, right after Stage I (the feature extractor is frozen, so features never change again)."],
  ["**How**", "Secure aggregation (Bonawitz et al., CCS 2017): the server receives only the total over all clients."],
  ["**Why**", "These totals are exact. Nothing is learned and nothing is averaged, so FedAvg cannot shrink them."],
]));
C.push(...table(["Sum sent by client k for class c", "Size", "Server computes", "Meaning"], [
  ["n_kc: number of samples", "1", "n_c = Σ_k n_kc", "class size"],
  ["S_kc = Σ h", "768", "m_c = S_c / n_c", "class mean (used by mean-anchored generator and moment correction)"],
  ["Q_kc = Σ ‖h‖²", "1", "V_c = Q_c / n_c − ‖m_c‖²", "class spread (used by moment correction)"],
  ["Φ_kc = Σ φ(h), φ = random Fourier features", "6,144", "μ_c = Φ_c / n_c", "class fingerprint (used by sample selection)"],
], [3.4, 0.8, 2.6, 3.0]));
C.push(eq("eq_stats", "9"));
C.push(P("Total: 10 × (1 + 768 + 1 + 6,144) ≈ 69,000 numbers ≈ **0.3 MB per client, once**. GeFL-F already sends 22 MB per round."));

C.push(H2("4.3 Step 2: the generator mean-anchored generator (replaces original generator in Stage II)"));
C.push(...img("f13_pcvae.png", 640, "Figure 8. Mean-anchored generator with every shape. Blue boxes: where the exact class mean enters."));
C.push(...card("Step 2", "Mean-anchored generator", [
  ["**Problem it fixes**", "In original generator the class enters only through a learned row w_y. A client without class c only shrinks w_c (weight decay), so rows of rare classes collapse (Theorem 1, Part 2.2)."],
  ["**What changes**", "The class enters through the exact mean m_y. It goes in twice: as 3 extra input channels of the encoder, and through a linear layer A (768 → 512) added to the decoder input."],
  ["**Output**", "h̃ = ReLU(m_y + decoder output). The network learns only how a sample *differs* from its class mean."],
  ["**Why it works**", "There is no class-specific weight, so there is nothing for FedAvg to shrink. The shape of 'difference from the mean' is shared by all classes, so every client trains it, rare classes included."],
  ["**Source**", "Class prototypes as shared knowledge: FedProto (Tan et al., AAAI 2022)."],
  ["**Cost**", "5.94 M parameters (original generator: 5.55 M). Same training as GeFL-F: loss ‖h̃ − h‖² + KL, Adam lr 0.001, wd 0.001, batch 64, 5 local epochs, 100 rounds (CIFAR-10: 200)."],
]));
C.push(eq("eq_pc", "10"));
C.push(...table(["Layer", "MNIST / FashionMNIST / SVHN", "CIFAR-10"], [
  ["Encoder input [h ; m_y]", "6×16×16", "20×16×16"],
  ["Conv 4×4 stride 2 → batch norm → ReLU (×4)", "64×8×8 → 128×4×4 → 256×2×2 → 512×1×1", "same"],
  ["Linear → μ, log σ²", "16 + 16", "50 + 50"],
  ["Decoder input: Linear(z) + A m_y → ReLU", "512×1×1", "512×1×1"],
  ["Transposed Conv 4×4 stride 2 (×4)", "256×2×2 → 128×4×4 → 64×8×8 → 3×16×16", "… → 10×16×16"],
  ["Output ReLU(m_y + ·)", "3×16×16 = 768", "10×16×16 = 2560"],
], [3.4, 4.0, 2.2]));

C.push(H2("4.4 Step 3: moment fix "));
C.push(...card("Step 3", "Moment correction", [
  ["**Problem it fixes**", "A variational autoencoder's samples are too narrow (under-dispersed), and the mean of a rare class can be slightly off."],
  ["**What**", "For each class, measure the fake mean m̃_c and spread Ṽ_c (from 500 generator samples), then move every fake sample x:  x′ = ReLU( m_c + √(V_c / Ṽ_c) · (x − m̃_c) )."],
  ["**Result**", "The fake class now has the *exact* real mean m_c and spread V_c."],
  ["**Why this map (Proposition 3)**", "Of all maps that give the exact mean and spread, this one moves the samples the least. It corrects the class but keeps the shape the generator learned."],
  ["**Source**", "Optimal transport between distributions, scalar case (Olkin and Pukelsheim, 1982)."],
  ["**Cost**", "No training. Two numbers per class (a shift and a scale), computed by the server."],
]));
C.push(eq("eq_mc", "11"));

C.push(H2("4.5 Step 4: fingerprint selection (kernel herding, sample selection)"));
C.push(...card("Step 4", "Sample selection", [
  ["**Problem it fixes**", "Mean and spread can be right while the *shape* is still wrong (e.g. a fake sub-cluster that real data does not have)."],
  ["**What**", "Draw 4n corrected candidates of class c and keep the n whose average fingerprint is closest to the real fingerprint μ_c."],
  ["**Why it works (Lemma 1)**", "A head's error on real data differs from its error on fake data by at most B × (fake-to-real distance). Sample selection directly lowers this fake-to-real distance."],
  ["**Source**", "Kernel herding (Chen, Welling and Smola, UAI 2010); maximum mean discrepancy (Gretton et al., JMLR 2012)."],
  ["**Cost**", "No training. One matrix product per selected sample; uses only μ_c, already uploaded."],
]));
C.push(P("**Algorithm** (greedy, without replacement):"));
C.push(N("compute fingerprints φ(x_i) of the 4n candidates (6,144 numbers each); set w = μ_c;"));
C.push(N("pick the candidate with the largest ⟨φ(x_i), w⟩;"));
C.push(N("update w ← w + μ_c − φ(x_picked); repeat step 2 until n samples are picked."));
C.push(eq("eq_kh", "12"));
C.push(...img("ev07_rare_class_mnist.png", 640, "Figure 9. Real data, MNIST's rarest class (24 samples): the generator's output, after moment correction (right centre and width) and after sample selection (right shape). Details in Part 5.13."));

C.push(H2("4.6 Step 5: training the heads (Stage III)"));
C.push(...img("f14_round3.png", 640, "Figure 10. One Stage III round. The fake pool is made fresh every round."));
C.push(H3("(a) 10 epochs on the fake pool, with consensus soft labels "));
C.push(...card("Step 5a", "Consensus labels", [
  ["**What**", "Target t = ½ · one-hot(y) + ½ · (average of softmax predictions of the 10 architectures' heads on the same fake sample)."],
  ["**How**", "All clients use one shared fake pool per round. Clients upload only *sums* of predictions (≈ 24 KB per fake epoch), never their heads' outputs on real data."],
  ["**Why it works**", "Different architectures make different errors. Their average is a softer and more reliable label than the generator's label alone. β = 0.5 was better than β = 1."],
  ["**Source**", "FedMD (Li and Wang, 2019); FedDF (Lin et al., NeurIPS 2020)."],
]));
C.push(eq("eq_csl", "6"));
C.push(H3("(b) 5 epochs on the client's own real features, with logit adjustment "));
C.push(...card("Step 5b", "Logit adjustment", [
  ["**What**", "Loss = cross-entropy( head(h) + τ · log π_k , y ), τ = 1, where π_k is the client's own label distribution."],
  ["**Why it works**", "A client with 90 % of one class learns to predict that class. Adding log π_k during training removes this bias; at test time the offset is dropped."],
  ["**Privacy**", "π_k never leaves the client."],
  ["**Source**", "Logit adjustment (Menon et al., ICLR 2021); FedLC (Zhang et al., ICML 2022)."],
]));
C.push(eq("eq_la", "5"));
C.push(H3("(c) Why 10 fake epochs instead of GeFL-F's 1"));
C.push(P("**Proposition 5:** more fake training helps only when the generator is faithful. Measured on MNIST (long tail): +1.7 with mean-anchored generator + moment correction + sample selection, +3.9 at 100 clients, but −0.05 with the original generator."));
C.push(H3("(d) Aggregation and test time"));
C.push(B("Heads are averaged only with heads of the **same architecture**, as in GeFL-F."));
C.push(B("Test: image → frozen feature extractor → 3×16×16 → the client's own head → 10 logits. No generator is needed."));

C.push(H2("4.7 How each step adds up (development runs: seeds 0–2, settings chosen on the test set; long tail, balanced accuracy %, heads as trained)"));
C.push(...table(["Step", "MNIST", "FashionMNIST", "What it fixes"], [
  ["GeFL-F", "75.25", "58.47", "—"],
  ["+ logit adjustment", "78.92", "64.78", "client's own class skew"],
  ["+ holder-weighted averaging (class rows averaged over holders)", "87.89", "76.02", "rare rows no longer pulled down"],
  ["+ consensus labels", "90.40", "76.60", "better labels on fake features"],
  ["Mean-anchored generator replaces holder-weighted averaging", "90.75", "75.96", "no class rows at all"],
  ["+ moment correction", "92.31", "77.56", "exact mean and spread"],
  ["+ sample selection", "93.11", "78.22", "fake ≈ real (low fake-to-real distance)"],
  ["+ 10 synthetic epochs  = **Ours-A**", "**94.82**", "**79.67**", "uses the now-faithful generator more"],
], [3.6, 1.2, 1.2, 3.6], { highlight: [7] }));

C.push(H2("4.8 Where each part comes from"));
C.push(...table(["Part of Ours-A", "Source (research paper)", "What we use from it"], [
  ["Class sums sent safely", "Bonawitz et al., Practical Secure Aggregation, ACM CCS 2017", "server gets only the total over clients"],
  ["Class mean as class identity", "Tan et al., FedProto, AAAI 2022", "class prototypes (means) as shared knowledge"],
  ["Moment fix ", "Optimal transport, scalar case (Olkin and Pukelsheim 1982)", "smallest change that gives the exact mean and spread"],
  ["Fingerprint selection ", "Chen, Welling and Smola, Kernel Herding, UAI 2010; Gretton et al., JMLR 2012", "pick samples whose average fingerprint matches the class"],
  ["Client-prior loss ", "Menon et al., Logit Adjustment, ICLR 2021; Zhang et al., FedLC, ICML 2022", "loss that undoes a client's own class skew"],
  ["Consensus labels ", "Li and Wang, FedMD, 2019; Lin et al., FedDF, NeurIPS 2020", "average prediction of different models as soft label"],
], [2.4, 3.8, 3.4]));
C.push(P("**What is new in Ours-A:** taking class information from exact secure-aggregation sums instead of from FedAvg-trained weights, and using it in three places (generator input, moment fix, sample selection). Each building block exists; the combination, its use inside a heterogeneous generator-based pipeline, and the collapse analysis that motivates it are ours. Logit adjustment, consensus labels and holder-weighted averaging are adopted, not claimed."));

C.push(H2("4.9 Communication and privacy"));
C.push(...table(["Release", "Size", "When", "Protection"], [
  ["Class counts, sums of h and of ‖h‖², sums of random features", "≈ 0.3 MB per client", "once, after Stage I", "secure aggregation; optional (ε, δ) differential privacy"],
  ["Consensus labels: prediction sums on a shared fake pool", "≈ 24 KB per fake epoch", "each Stage-III round", "sums only; no real data"],
  ["Generator", "5.94 M params ≈ 23.7 MB (GeFL-F: 22 MB)", "each Stage-II round", "same as GeFL-F"],
  ["Feature extractor, heads", "as in GeFL-F", "as in GeFL-F", "unchanged"],
], [3.6, 2.6, 1.6, 1.8]));
C.push(P("For formal privacy, features are clipped and each release gets Gaussian noise (zero-concentrated differential privacy):"));
C.push(eq("eq_dp", "13"));

// ---------------- PART 5
C.push(H1("5. Results"));
C.push(H2("5.1 Protocol"));
C.push(P("All runs follow the paper's appendix: the same models, optimisers, rounds and data fractions. Long tail: imbalance 100 : 1, Dirichlet 0.5, 10 clients unless stated. Each seed fixes the data split and the initialisation for every method, so comparisons are paired. **New in v3:** every trained model is also scored under eight federated last-layer readouts (the exact readout, Fed3R, FedNCM, CCVR, each from all clients or from the architecture's own clients), and the fair re-test (Part 5.3) uses new seeds 3–5 with every setting chosen on a held-out validation split, with the selection rule and predictions committed before any run. Heavy runs used Kaggle T4 × 2 sessions."));
C.push(H2("5.2 Main comparison: every method with the same exact readout (K16, seeds 0–2)"));
C.push(...table(["Long tail, balanced accuracy", "MNIST", "FashionMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], [
  ["GeFL-F, heads as trained", "74.5", "59.2", "49.6", "33.9", "7.8"],
  ["GeFL-F + exact readout", "90.4", "79.3", "64.7", "51.3", "15.2"],
  ["GeFL-F + CCVR (all clients)", "91.9", "80.5", "64.8", "49.9", "14.3"],
  ["minimal variant (holder-weighted averaging + logit adjustment + consensus labels) + exact readout", "92.4", "80.6", "66.4", "53.4", "16.5"],
  ["Ours-A, heads as trained", "95.1", "80.1", "65.4", "45.7", "15.4"],
  ["**Ours-A + exact readout**", "**94.4**", "**81.5**", "**66.7**", "**53.9**", "**18.6**"],
  ["Ours-A − GeFL-F, both + exact readout [95% CI]", "+3.9 [3.0, 4.8]", "+2.2 [0.1, 4.4]", "+2.0 [0.8, 3.2]", "+2.6 [2.1, 3.1]", "+3.4 [3.1, 3.8]"],
], [3.8, 1.2, 1.3, 1.2, 1.2, 1.2], { highlight: [5] }));
C.push(B("**The last layer is where most of GeFL-F's long-tail loss sits:** the exact readout lifts GeFL-F by 7–20 points. Any method for this setting must be compared with a recalibrated GeFL-F."));
C.push(B("**Ours-A's bodies support a better last layer on all five datasets** (+2.0 to +3.9 with the exact readout, +2.1 to +3.6 with CCVR). CIFAR-100 (100 classes) shows the effect is not specific to ten classes; every method is weak there in absolute terms (15–19 %) because the frozen 10-channel extractor limits everyone."));
C.push(B("Published per-architecture readouts (CCVR, Fed3R, FedNCM over an architecture's own clients) see one client's data at 10 clients and lose 4–26 points against their all-client versions."));
C.push(H2("5.3 Fair re-test on new seeds against nine baselines (K17, K19)"));
C.push(P("Seeds 3–5 (never used before); for every method its readout, and for Ours-A its number of fake epochs, chosen on the validation split; test accuracy compared paired over seeds, with a Holm correction over the baselines. The baselines: GeFL-F; the decay fix (GeFL-F + AdamW + logit adjustment); heads trained on real data only with plain loss, logit adjustment (balanced softmax), FedLC or FedRS; the Gaussian generator from exact statistics (FSG) + logit adjustment; server-side ensemble distillation (FedDF), with and without logit adjustment."));
C.push(...img("v02_fair_retest.png", 640, "Figure 11a. Ours-A minus each baseline on new seeds (test, validation-selected readouts)."));
C.push(...table(["", "MNIST", "FashionMNIST", "SVHN", "CIFAR-10", "CIFAR-100"], [
  ["Ours-A (selected readout)", "95.6", "82.1", "65.6", "53.1", "18.7"],
  ["strongest baseline", "FedDF 93.6", "FedDF / FedDF + LA 81.8", "FedDF + LA 65.9", "FSG + LA 52.6", "FSG + LA 17.1"],
  ["Ours-A − strongest", "**+1.9** (sig.)", "+0.3", "−0.3", "+0.5", "**+1.6** (sig.)"],
  ["Ours-A − decay fix", "**+2.0**", "+0.5", "+1.5", "+1.5", "**+3.0**"],
], [2.6, 1.3, 1.9, 1.5, 1.4, 1.4], { highlight: [2] }));
C.push(B("**Ours-A is not ahead of every baseline everywhere.** The pre-registered prediction (≥ 2 points over every baseline on 4 of 5 datasets) failed: it holds against GeFL-F and real-data heads, not against FedDF or FSG + LA on FashionMNIST, SVHN and CIFAR-10."));
C.push(B("**LA, FedLC, FedRS and balanced softmax give the same model** when a head is trained on real data only (within 0.5 points on every seed): they are the same correction written differently (Part 7.3)."));
C.push(B("**Reproducibility:** MNIST was run twice in two sessions; the 21 paired runs differ by −0.06 ± 0.43 points."));
C.push(H2("5.4 More clients, and only part of them each round (K18, K23)"));
C.push(...table(["SVHN long tail, exact readout (final balanced accuracy as trained)", "50 clients", "100 clients"], [
  ["GeFL-F", "58.0 (42.2)", "58.8 (42.2)"],
  ["decay fix (AdamW + LA)", "60.3 (54.3)", "59.9 (51.4)"],
  ["holder-weighted averaging + LA", "60.3 (54.8)", "60.8 (53.2)"],
  ["**Ours-A**", "**61.8 (58.2)**", "**61.0 (55.2)**"],
], [5, 2.3, 2.3], { highlight: [3] }));
C.push(P("Ours-A beats GeFL-F by +3.8 / +2.2 with the same readout (predicted ≥ 2: holds). Holder-weighted averaging gains over the decay fix as clients grow (+0.5 at 50, +1.8 at 100, heads as trained). **Partial participation** (100 clients, 10 % or 20 % per round; 31 of 48 runs so far): on FashionMNIST at 10 %, holder-weighted averaging beats the decay fix by +3.6 / +2.7 points as trained (equal with the exact readout), and Ours-A reaches 73.2 as trained (79.8 with the exact readout). The remaining runs are in notebook NB71."));
C.push(H2("5.5 Other imbalance settings and random rare classes (K22, FashionMNIST)"));
C.push(...table(["FashionMNIST, exact readout", "imbalance 10", "Dirichlet 0.1", "Dirichlet 1.0", "random rare classes"], [
  ["real-data heads + LA", "80.8", "76.6", "78.4", "77.7"],
  ["decay fix (AdamW + LA)", "83.2", "78.9", "80.8", "79.5"],
  ["**Ours-A**", "**84.2**", "**80.4**", "**81.5**", "**80.1**"],
], [3.2, 1.6, 1.6, 1.6, 1.8], { highlight: [2] }));
C.push(P("All three pre-registered predictions hold on FashionMNIST: Ours-A is ahead in every setting; its margin is larger when clients lack more classes (Dirichlet 0.1: +1.5) than when they lack fewer (1.0: +0.8); and choosing the rare classes at random changes nothing (+0.5, as in the main setting). SVHN is in notebook NB70."));
C.push(H2("5.6 Development results (seeds 0–2, settings chosen on the test set, heads as trained)"));
C.push(P("These are the results the method was built from. They compare against GeFL-F *without* a readout, so their gains are much larger than the fair margins above."));
C.push(...table(["Method (imbalance 100 : 1, Dirichlet 0.5, 10 clients; balanced accuracy)", "MNIST", "FashionMNIST", "CIFAR-10", "SVHN"], [
  ["GeFL-F (Kang et al., 2025)", "75.25", "58.47", "33.74", "50.49"],
  ["GeFL-F + logit adjustment (similar to FedLC)", "78.92", "64.78", "37.30", "56.44"],
  ["+ holder-weighted averaging + logit adjustment", "87.89", "76.02", "40.73", "62.45"],
  ["+ holder-weighted averaging + logit adjustment + consensus labels", "90.40", "76.60", "42.10", "–"],
  ["Gaussian generator from exact statistics + logit adjustment", "87.04", "73.60", "44.17", "44.23"],
  ["GeFL-F + last-layer refit on pooled real data (upper bound of CCVR / CReFF / Fed3R-type recalibration)", "90.7", "79.2", "–", "64.8"],
  ["**Ours-A**", "**94.82**", "**79.67**", "**44.38**", "54.41"],
  ["**Ours-A with the SVHN switch** (class rows + holder-weighted averaging + moment correction + sample selection when class means are uninformative)", "94.82", "79.67", "44.38", "**65.60**"],
  ["oracle: Ours-A's heads, last layer re-fit on pooled real data", "94.93", "81.55", "50.64", "59.13 / 68.4 (switch)"],
], [4.6, 1.2, 1.4, 1.2, 1.2], { highlight: [6, 7], bad: [5] }));
C.push(P("The red row was the strongest baseline in version 2: GeFL-F with its last layer refit on pooled real data. Part 5.2 now gives the fair, federated version of this comparison (the exact readout) on all five datasets, and Part 5.3 against nine baselines on new seeds."));
C.push(P("Tail recall: 51.9 → 91.1 (MNIST), 35.4 → 73.6 (FashionMNIST). SVHN is the weak case (Part 5.11)."));
C.push(...img("f08_longtail.png", 640, "Figure 11. Long-tailed clients: Ours-A against GeFL-F and the strongest alternatives."));
C.push(H2("5.7 The paper's own balanced setting, against all ten of its methods"));
C.push(...table(["best_mean_acc, balanced", "GeFL-F (paper)", "Best of the paper's ten methods", "Our GeFL-F", "**Ours-A**", "Ours-A − best"], [
  ["MNIST, 10 clients", "95.47", "96.44 (GeFL, diffusion)", "96.01", "**97.62**", "+1.18"],
  ["MNIST, 50 clients", "95.04", "95.04 (GeFL-F, original generator)", "95.43", "**96.93**", "+1.89"],
  ["MNIST, 100 clients", "94.63", "94.63 (GeFL-F, original generator)", "94.88", "**96.77**", "+2.14"],
  ["FashionMNIST, 10 clients", "83.14", "84.28 (GeFL-F, diffusion generator)", "82.76", "83.96", "−0.32"],
  ["FashionMNIST, 50 clients", "82.21", "82.96 (GeFL-F, diffusion generator)", "81.50", "**82.96**", "±0.00"],
  ["CIFAR-10, 10 clients (final round: Ours-A 56.82 vs our GeFL-F 56.68)", "55.86", "59.36 (GeFL, diffusion)", "59.10", "**60.93**", "+1.57"],
  ["SVHN, 10 clients (mean anchor)", "76.26", "76.26 (GeFL-F, original generator)", "75.82", "74.38", "−1.88"],
  ["SVHN, 10 clients (with the switch to class rows)", "76.26", "76.26 (GeFL-F, original generator)", "75.82", "**77.07**", "+0.81"],
  ["FashionMNIST, 10 clients, with covariance correction (covariance version)", "83.14", "84.28 (GeFL-F, diffusion generator)", "82.76", "**85.75**", "+1.47"],
], [2.0, 1.2, 2.2, 1.2, 1.0, 1.1], { highlight: [0, 1, 2, 5, 7, 8] }));
C.push(P("On the paper's metric, above the best of its 10 methods on MNIST (all client counts), CIFAR-10 and, with the switch, SVHN; level on FashionMNIST with the main method and above it with covariance correction. Two cautions: the paper's numbers are single published values without error bars, and best_mean_acc picks the best round on the test set. At the final round the CIFAR-10 gain over our GeFL-F is only +0.14, so we claim no balanced-data gain there; on MNIST it survives (97.58 vs 95.75)."));
C.push(P("**With the exact readout (K16, 3 seeds):** GeFL-F → Ours-A is 96.9 → 97.4 (MNIST), 84.5 → 84.8 (FashionMNIST), 61.4 → 61.6 (CIFAR-10), 76.2 → 77.3 (SVHN), 19.8 → 21.4 (CIFAR-100): no harm in balanced data, and a small gain on the hardest datasets."));
C.push(...img("f09_iid.png", 640, "Figure 12. The paper's balanced setting: Ours-A against the paper's GeFL-F and the best of its ten methods."));
C.push(H2("5.8 Data augmentation (paper's Table IV), and why MixUp does not transfer"));
C.push(...table(["CIFAR-10, balanced, 10 clients", "Accuracy"], [
  ["FedAvg / + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "55.65 / 60.07 / 58.95 / 53.96 / 56.99"],
  ["Image-space GeFL + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "62.67 / 61.66 / 56.47 / 59.97"],
  ["Ours-A, no augmentation", "60.93"],
  ["Ours-A + feature-space MixUp (our run, same seeds)", "59.98"],
  ["Ours-A + our mixing variant", "60.29"],
], [6, 3]));
C.push(P("Ours-A beats 7 of 9 without augmentation. The two higher ones train full models on raw images; in GeFL-F the extractor is frozen, and feature-space MixUp lowered our accuracy (59.98 vs 60.93)."));
C.push(H2("5.9 More clients (development), and privacy"));
C.push(...img("f11_scaling_privacy.png", 640, "Figure 13. (a) 100 clients, MNIST long tail. (b) Ours-A with every statistic made differentially private (ε, δ = 10⁻⁵)."));
C.push(...table(["Setting", "GeFL-F", "Holder-weighted averaging + logit adjustment", "Gaussian generator + logit adjustment", "**Ours-A**"], [
  ["MNIST long tail, 100 clients", "75.98", "83.07", "86.05", "**94.30**"],
], [3.6, 1.4, 1.4, 1.4, 1.4], { highlight: [0] }));
C.push(P("Only −0.5 from 10 to 100 clients (the original generator with holder-weighted averaging loses 4.8): no class weights to estimate from shrinking client data. FashionMNIST long tail (3 seeds): Ours-A 79.7 / 79.5 / 79.0 at 10 / 50 / 100 clients, against GeFL-F 58.5 / 61.9 / 61.9."));
C.push(...table(["Long tail, 10 clients", "GeFL-F (no formal guarantee)", "Ours-A exact", "Ours-A, ε = 8", "Ours-A, ε = 2"], [
  ["MNIST", "75.25", "94.82", "90.36", "85.06"],
  ["FashionMNIST", "58.47", "79.67", "72.77", "65.59"],
], [2.2, 2.4, 1.4, 1.4, 1.4]));
C.push(P("Privacy costs 4.5–6.9 points at ε = 8 and 9.8–14.1 at ε = 2, mostly on the rarest classes; at ε = 2 still +9.8 (MNIST) and +7.1 (FashionMNIST, p = 0.084, not significant) over GeFL-F. The guarantee covers only the four added class statistics; the generator's and heads' federated training, the consensus-label sums and the switch's inputs are not private, and secure aggregation is simulated."));
C.push(P("**Membership attack on the released class sums (new in v3, 1 seed).** An attacker who knows a sample's label and has public reference data checks whether the released class mean has moved toward it. Success (AUC; 0.5 = guessing):"));
C.push(...table(["Release", "MNIST: all / rare classes / rarest", "FashionMNIST: all / rare / rarest"], [
  ["exact sums", "0.52 / 0.64 / 0.69", "0.52 / 0.63 / 0.73"],
  ["Gaussian noise, ε = 8", "0.50 / 0.58 / 0.64", "0.52 / 0.54 / 0.45"],
  ["Gaussian noise, ε = 2", "0.50 / 0.53 / 0.61", "0.51 / 0.50 / 0.41"],
], [2.6, 3.4, 3.4]));
C.push(P("Common classes are safe (0.49–0.51). Rare classes leak through the exact sums, because a mean over 24–112 samples moves visibly with each one; noise removes most of it. Secure aggregation alone does not protect rare classes: the added statistics should be released with noise whenever a class is rare."));
C.push(H2("5.10 What each component contributes"));
C.push(...img("f10_ablation.png", 640, "Figure 14. Adding components one at a time on the same seeds (long tail)."));
C.push(P("Each step adds on the same seeds. Kernel herding's own effect is small: removing it costs 0.71 (MNIST, p = 0.06) and 0.30 (FashionMNIST, p = 0.10) in the leave-one-out ablation below, and it does not improve train-on-fake, test-on-real accuracy."));
C.push(H3("Leave-one-out ablation (K09: remove one part from the full Ours-A; long tail, 3 seeds, paired)"));
C.push(...table(["Removed", "MNIST (change, points)", "FashionMNIST (change, points)", "Reading"], [
  ["Mean-anchored generator (back to original generator + holder-weighted averaging)", "−4.72 (p = 0.02)", "−0.67", "large on MNIST, small on FashionMNIST"],
  ["Logit adjustment", "−2.98 (p = 0.02)", "−6.18 (p = 0.02)", "largest; mostly recovered by post-hoc bias calibration"],
  ["Consensus labels", "−1.63", "−1.38 (p = 0.008)", "medium"],
  ["Moment correction", "−0.96 (p = 0.03)", "−0.49", "small; works through the spread"],
  ["Sample selection", "−0.71", "−0.30", "small"],
], [2.6, 2.0, 2.2, 2.8]));
C.push(P("Full Ours-A in this run: 95.18 ± 0.08 (MNIST), 79.79 ± 0.31 (FashionMNIST). All 30 seed-level differences are negative, but they share one reference run per dataset and seed, so they are not 30 independent tests: the correct cluster-level sign test gives p = 0.031 (not 2·10⁻⁹). With 3 seeds no single part survives a multiple-test correction."));
C.push(H3("Five seeds (K09 + K14, new in v3) and with the exact readout"));
C.push(...table(["Removed (change in points)", "MNIST as trained", "FashionMNIST as trained", "MNIST with exact readout", "FashionMNIST with exact readout"], [
  ["Logit adjustment", "−3.05 (5 of 5 seeds worse)", "−5.78 (5 of 5)", "+0.11", "+0.05"],
  ["Mean-anchored generator", "−3.89 (5 of 5)", "−0.50 (4 of 5)", "−1.05", "+0.15"],
  ["Consensus labels", "−1.35 (5 of 5)", "−1.50 (5 of 5)", "−0.44", "−0.53"],
  ["Moment correction", "−1.13 (5 of 5)", "−0.63 (4 of 5)", "−0.53", "−0.14"],
  ["Sample selection", "−0.67 (5 of 5)", "−0.65 (5 of 5)", "−0.26", "−0.13"],
], [2.6, 1.9, 1.9, 1.6, 1.6]));
C.push(P("As trained, every removal hurts on MNIST in every seed. With the exact readout (seeds 3–4) logit adjustment's effect vanishes, as the theory says it must (it is a last-layer correction, Part 7.3); consensus labels and the mean anchor keep part of their effect, which is in the bodies."));
C.push(H3("Mechanism controls (K20, new in v3; the same generator for every arm, seeds 3–4)"));
C.push(...table(["Change to Ours-A", "FashionMNIST (as trained / exact readout)", "CIFAR-10 (as trained / exact readout)", "Pre-registered prediction"], [
  ["herding → random subset of the same candidate pool", "−0.49 / −0.16", "−0.23 / −0.23", "herding beats random: **holds** (small)"],
  ["moment correction: mean only", "−0.89 / −0.39", "−0.97 / −0.39", "the spread is what matters: **holds**"],
  ["moment correction: spread only", "+0.29 / +0.09", "+0.15 / +0.11", "(spread alone matches the full correction)"],
  ["fingerprint size 1,536 instead of 6,144; pool 8n instead of 4n", "within ±0.2", "within ±0.2", "flat in both: **holds**"],
], [3.2, 2.2, 2.2, 2.4]));
C.push(P("Moving only the class mean *raises* the judge's recognition of the fakes (+4 to +5 points) but *lowers* accuracy: recognisable fakes are not the same as useful fakes; the right spread is."));
C.push(H2("5.11 The weak case: SVHN"));
C.push(...table(["SVHN", "GeFL-F", "+ holder-weighted averaging + logit adjustment", "Ours-A"], [
  ["Long tail, balanced accuracy", "50.49", "62.45", "54.41"],
  ["Balanced setting, paper's accuracy (best_mean_acc)", "75.82", "75.64", "74.38"],
], [3.8, 1.6, 1.6, 1.6]));
C.push(P("SVHN's class means are almost identical (nearest-mean accuracy 12 %, chance 10 %), so a mean-anchored generator cannot separate its classes."));
C.push(P("**Switch (implemented):** the server measures how well the exact class means separate the classes (F, from the same sums). When F < 0.05 it keeps GeFL-F's class rows, protected by holder-weighted averaging, and still applies moment correction and sample selection. On SVHN this gives **65.60 ± 1.50** under the long tail (+15.1 over GeFL-F) and **77.07 ± 0.19** in the balanced setting (above the paper's 76.26). MNIST and FashionMNIST are far from the threshold (F = 0.25–0.71). CIFAR-10 is close to it (F = 0.055–0.078) and only its anchor branch has been run. The threshold was chosen after SVHN failed, so it is a heuristic, not a tuned or validated rule."));
C.push(H2("5.12 Mechanism measurements"));
C.push(...table(["Quantity (long tail)", "GeFL-F", "Ours-A"], [
  ["Tail fidelity (referee accuracy on synthetic tail features), MNIST / FashionMNIST", "10.3 / 19.9 %", "88.6 / 78.7 %"],
  ["Tail spread relative to real data, raw → after correction and herding", "–", "0.39–0.49 → 0.93–0.98"],
  ["Fake-sample correctness in the balanced setting vs the judge's accuracy on real data (FashionMNIST)", "–", "0.837 vs 0.839"],
], [5, 2, 2]));
C.push(P("After the fixes, fake features are as hard as real ones, not artificially clean."));

C.push(H2("5.13 Proof on real data: what each component does"));
C.push(P("Measured on real features: MNIST and FashionMNIST, long tail, 3 seeds, the same splits as every run (Kaggle T4; a second run on another GPU gives the same numbers within about 1 point). The *referee* is a head trained on held-out real features; it never takes part in training. Code: experiments/D01_component_evidence.py."));
C.push(...img("ev01_collapse.png", 640, "Rows of rare classes under FedAvg end at 0.33 (MNIST) and 0.44 (FashionMNIST) of their start value; with holder-weighted averaging they grow (×1.7, ×2.4). They shrink even when 70–90 % of clients hold the class."));
C.push(...img("ev02_fidelity.png", 640, "Rare-class fake samples recognised as their class: GeFL-F 10 % / 20 %, + holder-weighted averaging 31 % / 71 %, mean-anchored generator 93 % / 88 %, Ours-A 90 % / 79 % (real test features: 97 % / 84 %)."));
C.push(...img("ev03_moments.png", 640, "Spread of fake features (1 = real): mean-anchored generator 0.47 / 0.45, + moment correction 0.85 / 0.90, + sample selection 0.95 / 0.98. Rare-class mean error: 0.22 → 0.09 → 0.05 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Class separability F of fake features: GeFL-F 0.09 / 0.32 (classes overlap), mean-anchored generator 0.78 / 1.35 (too clean), Ours-A 0.32 / 0.61, real 0.30 / 0.60."));
C.push(...img("ev04_mmd.png", 640, "Distance to real test features (fake-to-real distance, all classes): GeFL-F 0.22 / 0.25 → mean-anchored generator 0.20 / 0.19 → + moment correction 0.089 / 0.093 → + sample selection 0.071 / 0.078. Lower fake-to-real distance goes with higher real accuracy (r = −0.69 / −0.47), as Lemma 1 says."));
C.push(...img("ev05_tstr.png", 640, "Head trained only on fake features, tested on real data (balanced accuracy): GeFL-F 57 / 48, + holder-weighted averaging 81 / 76, mean-anchored generator 92 / 76, + moment correction 94 / 77, + sample selection 94 / 76. Rare classes on MNIST: 32 → 89. Sample selection does not add in this test (−0.2 / −1.1); its gain shows only inside the full pipeline (+0.8 / +0.7)."));
C.push(...img("ev07_rare_class_mnist.png", 640, "MNIST's rarest class (24 real samples): GeFL-F's fake class 9 lies on other classes; mean-anchored generator is on the class but too narrow; moment correction and sample selection spread it over the real class."));
C.push(...img("ev08_la_csl.png", 640, "Logit adjustment (seed 0): balanced accuracy 92.0 → 95.2 (MNIST), 75.5 → 79.9 (FashionMNIST); correlation between a client's own class shares and its predictions 0.64 → 0.44, 0.67 → 0.52. Consensus labels: the soft target is closer to the real-data referee than the one-hot label, lower or equal in every class (total-variation distance 0.100 → 0.092, 0.212 → 0.202)."));
C.push(...img("ev09_ts.png", 520, "More fake training helps more with the better generator (MNIST: −0.05 vs +1.71)."));
C.push(...img("ev06_separability.png", 640, "How well the exact class mean identifies the class (3 seeds): F = 0.30 / 0.60 / 0.07 / 0.012 and nearest-mean accuracy 82 / 69 / 41 / 12 % on MNIST / FashionMNIST / CIFAR-10 / SVHN (chance 10 %). SVHN's means carry almost no class information: the reason Ours-A is weak there."));

// ---------------- PART 6 (new in v2)
C.push(H1("6. Where the remaining error lives: the last layer"));
C.push(P("Every run also logs two diagnostics. **Last-layer refit (oracle):** keep each head's body, re-fit only its last layer on the pooled real training features with balanced sampling. **Ensemble:** average the 10 heads' predictions. The difference *refit − head* is the error in the last layer alone (the *readout gap*)."));
C.push(...img("f16_readout.png", 600, "Figure 15. Head as trained vs the same head with its last layer re-fit on real data (long tail, 3 seeds)."));
C.push(...table(["Run (long tail, 3 seeds)", "Head", "Last layer refit", "Readout gap"], [
  ["GeFL-F, MNIST / FashionMNIST / SVHN", "74.2 / 58.7 / 50.5", "90.7 / 79.2 / 64.8", "**16.5 / 20.5 / 14.3**"],
  ["+ holder-weighted averaging + logit adjustment, MNIST / FashionMNIST", "87.8 / 76.0", "92.4 / 80.1", "4.6 / 4.1"],
  ["Ours-A, MNIST / FashionMNIST", "94.8 / 79.7", "94.9 / 81.6", "0.1 / 1.9"],
  ["Ours-A, FashionMNIST at 10 / 50 / 100 clients", "79.7 / 79.5 / 79.0", "81.6 / 81.2 / 81.0", "1.9 / 1.7 / 2.0"],
  ["Ours-A (switch), SVHN; Ours-A, CIFAR-10", "65.6; 46.1", "68.4; 52.3", "2.8; **6.2**"],
], [3.6, 2.0, 2.0, 2.0], { bad: [0], highlight: [2] }));
C.push(B("**GeFL-F's long-tail failure is mostly in the last layer.** Its bodies are good; a last-layer refit adds 14–20 points."));
C.push(B("**Ours-A removes most of it** (16.5 → 0.1 on MNIST) *and* gives better bodies: Ours-A's refit beats GeFL-F's refit by +4.2 / +2.4 / +3.6."));
C.push(B("**What is left does not shrink with more clients** (1.7–2.0 on FashionMNIST at every K) and is largest on CIFAR-10 (6.2)."));
C.push(H2("6.1 A federated way to close the gap: exact readout"));
C.push(P("The refit above needs pooled real data, which federated learning forbids. But the same answer can be built from sums. For each architecture g, every client passes its real features through head g's body and sends, by secure aggregation, per-class sums of the embeddings e, the class counts and the sum of e·eᵀ. The server then sets the last layer in closed form (linear discriminant analysis with a uniform class prior):"));
C.push(B("class means μ_c and shared covariance Σ from the sums (shrunk: 0.9·Σ + 0.1·tr(Σ)/d·I, fixed in advance);"));
C.push(B("weights w_c = Σ⁻¹ μ_c, bias b_c = −½ μ_cᵀ Σ⁻¹ μ_c."));
C.push(P("Cost: one extra round; under 1.1 M numbers per architecture for the largest head. It discloses federation-level class statistics of the embeddings, the same kind of release as the class sums of Part 4.2."));
C.push(H3("Test (T03, pre-registered; 1 seed, shortened schedule on CPU)"));
C.push(...table(["Run", "Head", "Refit (pooled data)", "**Exact readout (sums only)**", "Bias calibration (offsets only)"], [
  ["FashionMNIST GeFL-F", "61.2", "78.7", "**79.1**", "65.1"],
  ["FashionMNIST + holder-weighted averaging + logit adjustment", "73.0", "79.4", "**79.9**", "73.6"],
  ["FashionMNIST Ours-A", "75.7", "80.7", "**80.3**", "77.9"],
  ["MNIST GeFL-F", "72.4", "91.0", "**90.8**", "75.5"],
  ["MNIST + holder-weighted averaging + logit adjustment", "82.1", "91.8", "**91.3**", "82.0"],
  ["MNIST Ours-A", "91.1", "93.6", "**92.8**", "92.2"],
], [2.6, 1.2, 1.9, 1.9, 1.9]));
C.push(B("**Passed:** exact readout is within 0.8 points of the pooled-data refit in all runs (9 of 9 including the follow-up), +18 points on plain GeFL-F. Logit offsets (bias calibration, logit adjustment) stay below the refit in every run: the error is in the weights, not only in a class prior."));
C.push(B("**Where the error sits (measured):** on classes a client does **not hold** (FashionMNIST Ours-A: −17 recall points against the refit), less on classes held with 1–20 samples (−7), none on well-held classes."));
C.push(B("**Failed prediction (reported honestly):** we predicted the cause was a wrong synthetic class mean; that did not explain the per-class gap. Mis-scaled weight lengths explain about a quarter (re-normalising them recovers 16–34 % of the gap). The full mechanism is still open."));
C.push(H2("6.2 What this means for the paper"));
C.push(B("**Exact readout is not a new idea**: it is the known classifier recalibration from federated statistics (CCVR, NeurIPS 2021; CReFF, IJCAI 2022; Fed3R, ICML 2024), extended to heterogeneous heads (one set of statistics per architecture). We present it that way."));
C.push(B("**GeFL-F + recalibration must be a baseline.** Done in v3 on five datasets (Part 5.2): with the exact readout on both, Ours-A keeps +2.0 to +3.9 points; on new seeds against the strongest of nine baselines, −0.3 to +1.9 (Part 5.3)."));
C.push(B("**The exact readout is exactly the centralised readout** of the same model bodies (Part 7.5), and it is now computed for every run."));

// ---------------- PART 7 (new in v3): federated vs centralised, the mathematics
C.push(H1("7. How far is federated training from centralised training? (new in v3)"));
C.push(P("This part compares every stage of GeFL-F with the same stage trained on all the clients' data in one place, with the same frozen-extractor design. Each statement is either proven (short proofs in Appendix A and in docs/research/federated_vs_centralized_math.md) or measured; checks are in docs/research/math_checks/."));
C.push(H2("7.1 The clients differ only in which classes they have"));
C.push(P("**Lemma (pure label shift, proven from the splitting code).** A client's images of class c are a random subset of all images of class c. So for anything that depends only on a class (a generator's class row, a class mean, a last layer), every client's data point at the *same* target as the pooled data. Federation can therefore cost **weighting, noise and speed**, but never a wrong target. A wrong target can only come from (i) the optimiser (the decay of Part 2), (ii) parameters tied to the client's own class mix (biases, last layers), or (iii) shared parts trained on a long-tailed loss, which are equally long-tailed centrally."));
C.push(H2("7.2 Stage I: the extractor is trained as if on pooled data"));
C.push(P("FedAvg with different amounts of local work optimises a weighted mix of the clients' losses (FedNova, NeurIPS 2020). Because each client runs a number of local steps proportional to its data size, the weights come out within 1–2 % of the pooled weights at 10 clients: to first order the federated extractor is trained on the pooled long-tailed loss. **Measured:** replacing it by a centrally trained, class-balanced extractor changes the final accuracy by 0.0 (MNIST), +1.1 (FashionMNIST) and −1.0 (CIFAR-100), one seed each. A federated class-balanced Stage-I loss gave +1.0 / +0.4 in a CPU pilot but broke on CIFAR in the full pipeline (Part 3.6)."));
C.push(H2("7.3 Per-client loss corrections only move the last layer"));
C.push(B("**Proposition (exact).** For a head with a free last-layer bias, training with logit adjustment, FedLC or balanced softmax gives *exactly* the same model as plain training started from a shifted bias; FedRS is plain training with a different step size on absent classes. So for a head trained on one client's real data these published losses cannot change the body. **Measured:** real-data heads with the four losses agree within 0.5 points with the exact readout on every seed and dataset (K17)."));
C.push(B("**Why logit adjustment still matters during training.** Each round ends with 5 real epochs. Without the adjustment they push the scores of the classes a client lacks down without limit, erasing what the fake epochs taught; with it those classes get almost no push. Measured share of the fake phase's effect that survives the real phase, for absent classes: 0.96–1.0 with logit adjustment, 0.76–0.98 without (MNIST, K25). Once the exact readout is applied its effect vanishes (Part 5.10)."));
C.push(H2("7.4 Fake data help only if they carry their label"));
C.push(P("**Identity (exact).** If a fake feature's label is unrelated to the feature, training on it is the same as penalising the spread of the model's scores: it teaches the model to *not* separate classes, exactly in the directions the client's real data never correct (the classes it lacks). With a share s of informative fakes the loss is s × (useful term) + (1 − s) × (that penalty). Prediction: the fake phase hurts when s ≈ 0 and helps when s is large."));
C.push(...img("v03_signal_fraction.png", 520, "Figure 7.4. Gain of the fake phase over heads trained on real data only (same readout) against the signal fraction s measured by a referee: it changes sign near s ≈ 0.03–0.07. CIFAR-100: GeFL-F (s = 0.002) −2.2 points, decay-fixed GeFL-F −1.2, Ours-A (s = 0.09) +1.8."));
C.push(P("This is the clearest account of what Ours-A's generator buys: it raises s by 7× (CIFAR-10) and 44× (CIFAR-100) over GeFL-F. Consensus labels turn the penalty into distillation of the models' consensus, which is why they help most when fakes are poor."));
C.push(H2("7.5 The exact readout is the centralised readout"));
C.push(P("**Theorem (trivial but useful).** Any quantity computed from sums over all samples is the same however the data are split across clients. The exact readout, the class means and spreads behind the anchor and the moment correction, the fingerprints and the switch statistic are all such sums: they equal their centralised values for the same networks. The extractor, the generator's shared weights and the bodies are trained by FedAvg and are not. So once the exact readout is applied, the last layer is no longer a source of federated loss; what remains is in the bodies."));
C.push(H2("7.6 The remaining gap, measured against centralised training"));
C.push(...img("v04_centralised.png", 640, "Figure 7.6. Long tail, seed 0, exact readout. Grey: each body trained on its own client only. Blue: Ours-A. Green: the same bodies trained on all clients' data in one place (centralised). Light: balanced data."));
C.push(...table(["", "MNIST", "FashionMNIST", "CIFAR-100*"], [
  ["one client's data only → all clients' data (federated cost of one client per architecture)", "91.5 → 94.4", "78.1 → 81.9", "15.1 → 18.1"],
  ["Ours-A (federated, synthetic data from the other clients)", "94.4", "81.1", "18.7"],
  ["federated residual of Ours-A against centralised", "0.0", "1.9 (≈ 1.1 extractor + 0.8 body)", "≈ −0.6"],
  ["long-tail cost of Ours-A (balanced − long tail)", "2.9", "3.3", "2.7"],
  ["long-tail cost of **centralised** training (balanced − long tail)", "3.2", "2.7", "4.1"],
], [4.4, 1.6, 2.4, 1.6], { highlight: [4] }));
C.push(B("A body that sees only its own client's data loses 3.0–4.2 points against one that sees everyone's (MNIST, FashionMNIST, two seeds). Ours-A's fake data recover all of it on MNIST and most of it on FashionMNIST."));
C.push(B("**The long-tail loss that remains is not federated:** centralised training on the same long-tailed data loses as much. No federated method can recover it."));
C.push(B("*CIFAR-100's centralised arms used 3 of the 10 architectures; SVHN and CIFAR-10 are being measured (notebook NB71). One or two seeds per dataset."));
C.push(H2("7.7 What happens at each aggregation (K25, measured on the real pipeline)"));
C.push(...img("v05_instrumentation.png", 660, "Figure 7.7. Left, middle: per round and class, how much clients without the class shrink its generator row (bars) and clients with it grow it (ticks). With the decay fixed the shrinking is ≈ 0. Right: the fake phase flattens the models' scores, most for GeFL-F, whose fakes carry the least label information."));
C.push(B("Clients without a class shrink its row 2–10 times faster than the clients holding it grow it (GeFL-F, MNIST and CIFAR-10); with AdamW the shrinking is about 0.001 or less per round (prediction: holds)."));
C.push(B("The extractor's client gradients disagree more as training goes on in CIFAR-10 (heterogeneity 0.02 → 0.58) but not in MNIST (0.12 → 0.007); averaged BatchNorm statistics stay within 0.86–1.10 of the pooled ones."));
C.push(H2("7.8 What the mathematics changes in the paper"));
C.push(B("Two of the three losses are removed exactly (the collapse, by the decay fix or holder averaging; the last layer, by the exact readout). The third (one client per architecture) is closed by fake data only to the extent they carry their label."));
C.push(B("The ceiling is close on easy data: on MNIST the margin over the decay fix (+2.0) already equals the gap between the decay fix and centralised bodies (≈ 1.9). This explains why no further method gave large gains."));

// ---------------- PART 7
C.push(H1("8. Why it fails, and why each fix works"));
C.push(P("Parts 4 and 5 say what each part of Ours-A does and how many points it adds. This part says **why** (Part 7 gives the mathematics behind it). There are eight failures, in the order they occur. For each one we give: what goes wrong; the cause, step by step; the measurement that shows it; what we would have measured *if the explanation were wrong*; why the fix removes the cause; and how sure we are."));
C.push(...table(["#", "What goes wrong", "Cause in one line", "Fix", "Before → after"], [
  ["1", "Rare classes are erased from the generator", "Clients without the class pull its list to zero every step; clients with it push back only rarely", "Holder-weighted averaging, then the mean-anchored generator", "Rare lists 4–10 % → 117–125 % of start; rare fakes recognised 10 % → 93 % (MNIST)"],
  ["2", "Fakes too narrow and too clean", "The generator learns to output the \"typical\" sample", "Moment correction", "Spread 0.47 → 0.85 of real; separability 0.78 → 0.32 (real 0.30)"],
  ["3", "Fakes have the wrong shape", "Average and spread are only two numbers", "Sample selection", "Fake-to-real distance 0.089 → 0.071"],
  ["4", "Each model leans to its client's common classes", "The last 5 epochs of every round are on skewed real data", "Logit adjustment", "Skew 0.64 → 0.44; +3 to +6 points"],
  ["5", "Ambiguous fakes get a confident label", "A fake drawn as class c can look like two classes", "Consensus labels", "Closer to a real-data judge in every class; +1.4 to +1.6 points"],
  ["6", "More fake training did not help", "Training longer on wrong fakes teaches the wrong thing longer", "Only after the generator is fixed", "−0.05 → +1.7 points"],
  ["7", "Ours-A weak on SVHN", "SVHN's class averages are nearly identical", "Automatic switch back to class lists", "54.4 → 65.6"],
  ["8", "Some error stays in the last layer", "Weights of classes the client lacks have the wrong size (about a quarter of it); the rest unknown", "Exact readout from sums", "Within 0.8 points of the best possible last layer"],
], [0.3, 2.0, 3.0, 2.0, 2.6], { left: true }));

// 7.1
C.push(H2("8.1 Rare classes are erased from the generator"));
C.push(P("**What goes wrong.** The original generator forgets how to draw rare classes. Its fake samples of a rare class look like random classes, so every model trained on them learns nothing about that class, and rare-class accuracy stays at 52 % (MNIST) and 35 % (FashionMNIST). (Part 2.5 shows the collapse is predictable from counts; Part 7.7 shows it happening round by round.)"));
C.push(P("**The cause, step by step.**"));
C.push(B("**The generator keeps one list of 512 numbers per class** (its *class list*). To draw a sample of class c it reads list c. A long, distinctive list says \"class c\" clearly; a list near zero says nothing, and the sample comes out as a mix of all classes."));
C.push(B("**Every training step also applies weight decay**: each number is pulled a little toward zero. It is meant as a mild brake against over-fitting."));
C.push(B("**A client that has class c** sees it in some of its batches. Each time, training pushes list c to describe class c better, which in practice makes it longer. **A client that does not have class c** never sees it, so the only thing that ever touches list c on that client is the pull toward zero."));
C.push(B("**The optimiser turns the mild brake into a full step.** Adam divides each update by the typical size of that number's recent updates. A number that only ever receives the small, steady pull is divided by that same small size, so the pull becomes a step as large as real learning. The size of the decay no longer matters, only whether it is there."));
C.push(B("**Rare classes lose the tug-of-war.** A rare class appears in few batches even on the clients that have it (MNIST's rarest class has 24 samples in the whole federation). Its pushes are small and occasional; the pulls from clients without it come at full size on every step."));
C.push(B("**The server averages all clients' lists**, pulls outweigh pushes, and list c ends every round shorter than it began. After a few dozen rounds the rare lists are close to zero."));
C.push(B("**The exact condition (Theorem 1, Part 2.2):** list c shrinks unless the share of clients holding class c is above θ = loss ÷ (gain + loss). Because rare classes gain little, θ is 0.61–0.93: a rare class needs 61–93 % of clients to survive. On MNIST the rarest classes are held by 7 or 8 of 10 clients and are still erased."));
C.push(...img("w01_erasing.png", 640, "Figure 16. (a) One change at a time on the real generator. (b) Share of rare-class fakes that a judge trained only on real data labels correctly."));
C.push(P("**How we know.** We trained the real generator four times, changing one thing each time (Figure 16a). Normal training: the three rarest lists end at 4–10 % of their starting length. Decay ten times smaller: *identical* curves. Decay switched off: they grow to 112–120 %. Only clients that hold the class averaged: they grow to 117–125 %. The consequence shows in the fakes (Figure 16b): a judge trained on real data recognises only 10 % (MNIST) and 20 % (FashionMNIST) of the original method's rare-class fakes, against 97 % and 84 % of real samples."));
C.push(P("**If the explanation were wrong.** If rare classes simply had too little data, switching off the decay would not help, but it restores them fully. If the decay acted as a mild brake (as with plain gradient descent), ten times less decay would mean ten times slower erasing, but the curves are identical, as the optimiser explanation predicts. If the half-the-clients rule of version 1 were right, classes held by 7–9 of 10 clients would be safe, but they are erased."));
C.push(P("**The fix, and why it removes the cause.**"));
C.push(B("**Holder-weighted averaging** (first fix): when the server averages list c, only clients that have class c vote, weighted by how many samples they have. The clients that could only pull toward zero no longer count, so the loss term of Theorem 1 is gone and the list can only grow by what holders teach it. Result: rare lists 117–125 % of start; judge recognises 31 % / 71 % of rare fakes; +9.0 points (MNIST), +11.2 (FashionMNIST)."));
C.push(B("**Mean-anchored generator** (final fix): removes class lists altogether. The class enters as its exact average feature, which each client sends once as a sum (secure aggregation: the server sees only the total). A sum cannot shrink. The generator's own weights are shared by all classes and learn only *how a sample differs from its class average*, which every client trains on every class it has. Result: the judge recognises 93 % / 88 % of rare fakes."));
C.push(B("**Why the anchor beats holder averaging:** holder averaging still has to *learn* each list from the few rare samples; the class average is exact from the first round."));
C.push(P("**How sure.** The bound is proven (Appendix A). The cause is tested by changing one thing at a time on the real generator (1 seed; the bound itself is exact). The accuracy consequences hold over 3 seeds and four datasets."));

// 7.2
C.push(H2("8.2 The fakes are in the right place but too narrow and too clean"));
C.push(P("**What goes wrong.** After the mean anchor, fakes sit around the right class average, but they are bunched far too tightly: their spread is 0.47 (MNIST) and 0.45 (FashionMNIST) of the real spread. And their classes are separated much more cleanly than real classes: separability 0.78 / 1.35 against 0.30 / 0.60 for real data."));
C.push(P("**The cause, step by step.**"));
C.push(B("The generator (a variational autoencoder) learns by rebuilding each real feature from a compressed code. When it is unsure, the output with the lowest error is the *average* of the plausible answers, and averaging removes the sample-to-sample variety."));
C.push(B("Its random input is meant to put the variety back, but in practice it is under-used. The result is \"textbook\" samples: correct, typical, too similar to each other."));
C.push(B("**Why that hurts:** a model trained on textbook samples draws its class boundaries to fit tight, well-separated clusters. Real samples spread further and overlap more, so the ones near the edge of a class fall on the wrong side, like learning handwriting only from neat examples and then reading a doctor's note."));
C.push(...img("w02_narrow.png", 640, "Figure 17. (a) Spread of fake features divided by the spread of real ones (1 = right). (b) How cleanly the fake classes are separated, against real data."));
C.push(P("**How we know.** Spread and separability are measured directly against the real features (Figure 17). In the leave-one-out ablation, removing moment correction cuts the rare-class spread from 0.93 to 0.59 (MNIST) and costs 0.5–1.0 points, while the judge's recognition of the fakes does *not* fall (FashionMNIST 83.9 vs 79.2). So moment correction does not help by making fakes more recognisable; it helps by giving them the right variety."));
C.push(P("**If the explanation were wrong.** If narrowness did not matter, correcting only the spread and centre would not change accuracy. It adds 1.6 points on both datasets, and the gain comes with the spread returning to that of real data, not with any change in recognisability."));
C.push(P("**The fix, and why it removes the cause.** **Moment correction:** for each class, measure the fakes' average and spread (500 samples), then shift every fake so its average equals the exact real average and stretch it so its spread equals the exact real spread (both from the uploaded sums). Of all ways to reach the right average and spread, this one moves the samples the least (Proposition 3), so it keeps whatever shape the generator learned and corrects only the two numbers that were wrong. No training, two numbers per class."));
C.push(P("**Result.** Spread 0.85 / 0.90; separability 0.32 / 0.61, matching real data (0.30 / 0.60); fake-to-real distance 0.20 → 0.089 (MNIST); +1.6 points on both datasets. **How sure:** measured on 3 seeds; the minimum-change property is proven."));

// 7.3
C.push(H2("8.3 Right centre and spread, but the wrong shape"));
C.push(P("**What goes wrong.** Two clouds of points can have the same centre and the same spread and still look different: one may have a sub-cluster the other lacks, or miss a style (for example sevens written with a cross-bar)."));
C.push(P("**The cause.** The centre and the spread are only two numbers per class; the shape needs many more."));
C.push(P("**The fix, and why it works.** **Sample selection:** each client also sends, once, a sum of *fingerprints* of its features (6,144 random features that together describe the shape of a class). The server draws four times as many fakes as it needs and keeps the subset whose average fingerprint is closest to the real one: at each step it adds the candidate that best fills what is still missing. A known bound says a model's error on real data is at most its error on fakes plus a constant times the fake-to-real distance (Lemma 1); selection lowers exactly that distance."));
C.push(...img("w03_shape.png", 520, "Figure 18. Distance between fake and real features after each step (lower is better)."));
C.push(P("**How we know.** The distance falls from 0.089 to 0.071 (MNIST) and 0.093 to 0.078 (FashionMNIST). Across all runs, lower distance goes with higher real accuracy (correlation −0.69 and −0.47). In the full method, selection helped in 9 of 9 paired tests (+0.7 to +0.8 points)."));
C.push(P("**What we cannot yet explain.** In the simpler test (train a model only on fakes, test on real data) selection adds nothing (−0.2 / −1.1). Its gain appears only together with the real-data epochs and consensus labels. It is also the smallest part in the ablation (−0.3 to −0.7 when removed). **How sure:** the effect is consistent but small, and its exact route is open."));

// 7.4
C.push(H2("8.4 Each model leans toward its client's common classes"));
C.push(P("**What goes wrong.** A client's model predicts the classes that are common *on that client* far too often, and the client's rare classes too rarely."));
C.push(P("**The cause, step by step.**"));
C.push(B("Every round ends with 5 epochs on the client's own real data. If 90 % of that data is one class, the cheapest way to lower the training error is to predict that class more often."));
C.push(B("The balanced fake epochs at the start of the round are partly undone by these 5 skewed epochs, every round."));
C.push(...img("w04_skew.png", 600, "Figure 19. (a) Correlation between how often a client sees each class and how often its model predicts it (0 = no lean). (b) Balanced accuracy with and without logit adjustment."));
C.push(P("**How we know.** The correlation between a client's class shares and its model's predictions is 0.64 (MNIST) and 0.67 (FashionMNIST): models predict what their client has."));
C.push(P("**The fix, and why it removes the cause.** **Logit adjustment:** during the real-data epochs, add the logarithm of the class's share on that client to the model's score for that class before computing the loss. A common class already starts ahead in the score, so the model gains nothing by leaning toward it and stops learning the lean. At test time the head start is removed and the scores are balanced. The client's class shares never leave the client."));
C.push(P("**Result.** Correlation 0.44 / 0.52; balanced accuracy 92.0 → 95.2 (MNIST) and 75.5 → 79.9 (FashionMNIST); the largest single part in the ablation (−3.0 / −6.2 points when removed). **Note:** correcting the lean after training (bias calibration) recovers most of this too: the job is \"remove the client's skew\", and either tool does it. **How sure:** measured on 3 seeds; an established technique (Menon et al., 2021)."));

// 7.5
C.push(H2("8.5 Ambiguous fakes get a confident label"));
C.push(P("**What goes wrong.** Every fake comes with the label it was drawn for. Some fakes look like two classes (a shirt halfway between a T-shirt and a pullover). A hard \"100 % T-shirt\" label teaches the model to be confidently wrong near the boundary between the two."));
C.push(P("**The fix, and why it works.** **Consensus labels:** the training target becomes half the drawn label plus half the average prediction of the 10 different model types on that same fake. Different architectures make different mistakes, so their average is a better estimate of what the sample really looks like. Clients send only sums of predictions on a shared pool of fakes."));
C.push(P("**How we know.** Compared with a judge trained on real data, the mixed target is closer than the hard label in every class (total-variation distance 0.100 → 0.092 on MNIST, 0.212 → 0.202 on FashionMNIST). Removing consensus labels costs 1.4–1.6 points (FashionMNIST p = 0.008)."));
C.push(P("**If the explanation were wrong.** If agreement between models were good evidence of a correct label, trusting it more would help. A Bayesian version that trusted agreement more did *worse* (83.2 vs 84.2): models trained on the same generator agree with the generator, so their agreement is partly circular. Half-and-half worked better than full trust. **How sure:** measured on 3 seeds; the gain is medium-sized and consistent."));

// 7.6
C.push(H2("8.6 More training on fakes helps only once the fakes are good"));
C.push(P("**What goes wrong.** The original method trains 1 fake epoch per round. More fake training looks like an easy win, but with the original generator it adds nothing."));
C.push(P("**The cause.** A model's error has two parts: noise from having little real data, which falls as you add fake data, and bias from fakes that are wrong, which grows the more you rely on them. The best share of fake training falls as the fakes get worse (Proposition 5). With bad fakes, extra fake epochs add bias as fast as they remove noise."));
C.push(...img("w06_budget.png", 500, "Figure 20. Gain from 10 fake epochs per round instead of 1."));
C.push(P("**How we know.** Exactly as predicted: −0.05 points with the original generator, +1.7 with ours (MNIST long tail), +3.9 with 100 clients, where each client has less real data and the noise part is larger. **Limit:** on CIFAR-10 and SVHN, where fakes are still worse, 10 epochs give a higher peak but a lower final accuracy than 1. **How sure:** proven model, measured on 3 seeds."));

// 7.7
C.push(H2("8.7 Why the main method is weak on SVHN"));
C.push(P("**What goes wrong.** On SVHN, Ours-A with the mean anchor reaches only 54.4, below the simpler holder-weighted averaging plus logit adjustment (62.5)."));
C.push(P("**The cause, step by step.**"));
C.push(B("The mean-anchored generator tells classes apart *only* through their class averages: two generated classes can differ at most as much as their averages do."));
C.push(B("SVHN digits come with varying colours, backgrounds and neighbouring digits. After the frozen feature extractor, the average feature of a \"3\" is almost the same as that of an \"8\"."));
C.push(B("So the generator receives nearly the same instruction for every class, and its fakes cannot be told apart by class."));
C.push(...img("w07_svhn.png", 640, "Figure 21. (a) Guessing each real sample's class from the nearest class average: high on MNIST, near chance on SVHN. (b) SVHN long tail: the automatic switch fixes it."));
C.push(P("**How we know.** Guessing a real sample's class from the nearest class average works 82 % of the time on MNIST, 69 % on FashionMNIST, 41 % on CIFAR-10, and only 12 % on SVHN (chance is 10 %). The prediction is that the anchor fails exactly where this is near chance, and it fails only on SVHN."));
C.push(P("**The fix.** The server computes how well the class averages separate the classes, from sums it already has. Below a threshold (separability 0.05) it keeps the original class lists, protected by holder-weighted averaging, and still applies moment correction and sample selection. Result: 65.6 under the long tail (+15.1 over the original method) and 77.07 in the balanced setting (above the paper's 76.26). The other three datasets are above the threshold and unaffected. **How sure:** measured on 3 seeds; the threshold was chosen from these four datasets and needs a fifth dataset to be confirmed."));

// 7.8
C.push(H2("8.8 What still fails: the last layer"));
C.push(P("**What goes wrong.** Even with Ours-A, each model's last layer loses some accuracy compared with the best last layer for the same model (Part 6): 0.1 points on MNIST, 1.9 on FashionMNIST, 6.2 on CIFAR-10. It does not shrink with more clients."));
C.push(...img("w08_lastlayer.png", 560, "Figure 22. Accuracy lost in the last layer, split by how much of the class the model's own client has (FashionMNIST, 1 seed, shortened schedule)."));
C.push(P("**Where it sits (measured).** Almost entirely on classes the model's own client does **not have**: there Ours-A loses 17 points (the original method 74). Classes the client has with 1–20 samples lose 7; classes it has with more than 20 samples lose nothing."));
C.push(P("**The cause, as far as we know.**"));
C.push(B("For a class its client lacks, a model's last layer learns that class **only from fakes**. For classes the client has, it also learns from real samples in the last 5 epochs of every round."));
C.push(B("So the weights for missing classes end up the **wrong size** relative to the others, and those classes are predicted too rarely. Rescaling all class weights to the same size recovers 16–34 % of the loss: this explains about a quarter."));
C.push(B("**The rest is not explained yet.** Our first explanation, that the fake class averages are slightly off inside each model, was tested and was wrong: it did not predict which classes lose the most (correlation 0.10–0.33, against 0.33–0.90 for the simple \"how much of the class came from fakes\")."));
C.push(P("**What fixes it anyway.** The exact readout (Part 6.1) sets the last layer from exact sums of real features and never sees a fake. It lands within 0.8 points of the best possible last layer. **How sure:** the location of the error and the fix are measured (1 seed, shortened schedule; consistent with the 3-seed numbers of Part 6); the cause is only a quarter explained."));

// 7.9
C.push(H2("8.9 Why the dropped ideas failed"));
C.push(...table(["Idea", "Why it failed"], [
  ["Server-side ensemble distillation", "It teaches each model on fakes. It cannot pass on anything the fakes do not contain, so it is limited by the same fake-to-real gap (equation 3)."],
  ["Re-training the last layer on balanced fakes", "Same limit, and the fakes are worse than real data: weaker than logit adjustment (62.0 vs 65.2). The exact readout works because it uses real sums instead of fakes."],
  ["Gaussian generator from exact statistics", "A model trained on its samples becomes a straight-line classifier. Straight lines work on MNIST but reach only 21 % on SVHN features."],
  ["Bayesian consensus labels", "Trusted model agreement more; agreement is partly circular (8.5)."],
  ["Filling each real batch's missing classes with fakes", "Needs good rare-class fakes, which did not exist before the generator was fixed."],
  ["Feature mixing (MixUp and our variant)", "Lowered accuracy (59.98 vs 60.93). Mixing features after a frozen extractor cannot add the image-level variety that the paper's image mixing adds while training the whole model."],
  ["Lazy decay (decay only the rows of classes in the batch)", "Removes the same pull toward zero as holder-weighted averaging, so on top of it adds nothing (78.6 vs 79.1)."],
], [3.0, 6.6], { left: true }));

C.push(H2("Limitations"));
C.push(B("**Margins over the strongest baselines are small:** −0.3 to +1.9 on new seeds with validation-chosen settings, significant only on MNIST and CIFAR-100 (Part 5.3)."));
C.push(B("**The frozen extractor limits every method** (CIFAR-100 about 19 %, CIFAR-10 about 54 % with the exact readout). Results are on small images and the paper's backbone sizes."));
C.push(B("**Scope:** at 10 clients each architecture has one client; the rare classes are always 6–9 in the main runs (random rare classes tested on FashionMNIST, Part 5.5); CIFAR-10's real imbalance after subsampling is 48 : 1."));
C.push(B("**Selection on the test set** for the development runs (seeds 0–2); the fair re-test (seeds 3–5) fixes this for the main comparison, but the label mixing weight, the logit-adjustment strength and the switch threshold were still set from earlier runs."));
C.push(B("**The centralised comparison (Part 7.6)** uses one or two seeds and three datasets; SVHN and CIFAR-10 are pending (NB71)."));
C.push(B("**Privacy:** differential privacy covers only the added class statistics; rare classes leak through exact sums (Part 5.9); secure aggregation is simulated."));
C.push(B("**Cost:** Ours-A is 2.2–2.8× slower than GeFL-F end to end; a batched version of sample selection (same selected samples) now removes most of the per-sample overhead."));
C.push(B("**Not reproducible bit for bit:** GPU training was not deterministic; a repeated MNIST run differs by −0.06 ± 0.43 points."));

// ---------------- PART 9 (new in v3): beyond GeFL
C.push(H1("9. Beyond GeFL: the same collapse in federated recommendation (new in v3, first results)"));
C.push(P("The collapse of Part 2 needs only three things: a row of parameters that gets a gradient only from samples that use it (an embedding row), clients that never use it, and local Adam with ordinary (coupled) L2 decay followed by averaging. Federated recommendation has all three: item embeddings are shared, and most clients never rate most items. Test: matrix factorisation on MovieLens-1M (6,040 users, 3,706 films; user embeddings stay on their client, item embeddings are averaged), 10 or 50 clients, 3 seeds, 30 rounds; predictions written before the runs."));
C.push(...table(["Rare items (κ < 1)", "collapsed, 10 / 50 clients", "rating error on them (RMSE), 10 / 50"], [
  ["local Adam + L2, plain averaging (common practice)", "85 % / 83 %", "1.09 / 1.06"],
  ["same, optimiser state reset every round", "99.7 % / 100 %", "1.11 / 1.11"],
  ["AdamW (decay decoupled)", "0 % / 0 %", "1.04 / 0.99"],
  ["holder-weighted averaging (each item averaged over the clients that rated it)", "23 % / 9 %", "1.04 / **0.97**"],
  ["**centralised** Adam + L2 (no federation)", "**78 %**", "1.10"],
], [4.6, 2.4, 2.6], { bad: [0, 1, 4], good: [3] }));
C.push(...img("v06_movielens.png", 520, "Figure 9.1. Item-embedding length after training against how often an item appears in the optimiser's batches. Centrally, items seen less often than about once per 500 steps are wiped out (Adam forgets in about 1,000 steps); federated training with 50 clients wipes out moderately rare items about 10 times harder."));
C.push(B("**Partly confirmed, partly corrected.** The count rule ranks the final lengths well (Spearman 0.93–0.96) and the collapse grows with the number of clients (22 % → 48 % of all items). But **centralised Adam with L2 also collapses rare items** (78 %): our prediction \"no collapse centrally\" was wrong here, and \"every κ < 1 row collapses\" held for 83–85 %, not all (the rest end at about 0.2 of their length)."));
C.push(B("**One rule covers both worlds:** the share of optimiser steps on which an item gets any data predicts its collapse centrally (Spearman 0.91) and federated (0.94–0.97). Federation adds the clients that never see the item, which matters more as clients multiply."));
C.push(B("**Practical:** holder-weighted averaging keeps L2's regularisation for popular items (best overall error) and gives the best rare-item error at 50 clients."));

// ---------------- PART 10
C.push(H1("10. Status and next steps"));
C.push(...table(["Item", "Status"], [
  ["Decay fix vs holder-weighted averaging at 10 / 50 / 100 clients (K15)", "**done** (Part 2.6)"],
  ["Every method under eight last-layer readouts, five datasets incl. CIFAR-100, long tail and balanced (K16)", "**done** (Part 5.2)"],
  ["Fair re-test on new seeds against nine baselines, validation-selected (K17, K19)", "**done**: margins −0.3 to +1.9 (Part 5.3)"],
  ["SVHN with 50 / 100 clients and the prospective κ test (K18)", "**done**: κ prediction 16 of 16 (Parts 2.5, 5.4)"],
  ["Five-seed ablation (K14), mechanism controls (K20)", "**done** (Part 5.10)"],
  ["Signal-fraction gate and class-balanced Stage-I loss (K21)", "**done**: no gain for Ours-A; the gate helps weak generators (Part 3.6)"],
  ["Per-round measurements (K25); membership attack; batched sample selection", "**done** (Parts 7.7, 5.9)"],
  ["Federated recommendation (MovieLens)", "**done**, first results (Part 9)"],
  ["Other imbalance settings: FashionMNIST (K22)", "**done**; SVHN in notebook NB70"],
  ["Partial participation (K23)", "31 of 48 runs; the rest in NB71"],
  ["Centralised comparison on SVHN / CIFAR-10 / CIFAR-100 (X3)", "NB71 (the first attempt failed on a device bug)"],
  ["CIFAR-10 with 50 clients (K26); DCGAN generator under the long tail (K24)", "NB70"],
  ["Paper: analysis framing, LaTeX version, one-command regeneration of every number", "draft rewritten; LaTeX and tagging to do"],
], [6.4, 3.2]));
C.push(P("**Recommended framing for the paper:** an analysis of where generator-based heterogeneous federated learning loses rare classes, with a count rule that predicts the generator's collapse, the exact readout as the fix for the last layer, a stage-by-stage comparison with centralised training, and Ours-A as a consistent but modest gain. Realistic targets: TMLR, a federated-learning workshop, or an IEEE / Elsevier journal; a main-track conference needs the recommendation result to hold up and extend to a second embedding setting."));

// ---------------- APPENDIX
C.push(H1("Appendix A. Proofs"));
C.push(P("**Theorem 1.** Let r_k be client k's row after local training and r⁺ = (1/K) Σ_k r_k. Define G_k = ‖r_k‖ − ‖r‖ for holders and D_k = ‖r‖ − ‖r_k‖ for non-holders. By the triangle inequality ‖r⁺‖ ≤ (1/K) Σ_k ‖r_k‖ = ‖r‖ + p Ḡ − (1 − p) D̄, with equality if and only if all r_k point the same way. The right side is below ‖r‖ exactly when p < D̄ / (Ḡ + D̄). ∎"));
C.push(P("**Lemma 2.** A non-holder's gradient on r_c is exactly λ r_c. Adam's step is η·m̂/(√v̂ + ε); multiplying every gradient by λ multiplies m̂ and √v̂ by λ, so the step is unchanged as long as λ|r| ≫ ε. Hence D̄ does not depend on λ > 0, and D̄ = 0 at λ = 0. Under stochastic gradient descent the same step would be η λ ‖r‖. ∎"));
C.push(P("**Proposition 3.** With Y = T(X), E‖Y − X‖² = V_c + Ṽ_c + ‖m_c − m̃_c‖² − 2E⟨Y − m_c, X − m̃_c⟩. By Cauchy–Schwarz the last term is at most √(V_c Ṽ_c), with equality if and only if Y − m_c = √(V_c/Ṽ_c)(X − m̃_c). ∎"));
C.push(P("**Proposition 5.** With w = λn_s/(n_r + λn_s), the error becomes σ²(1 − w)/n_r + w²b², a convex parabola minimised at w* = σ²/(2n_rb²). The optimum is clipped to the feasible range, so the optimal synthetic share is non-increasing in b and reaches its maximum as b → 0. ∎"));
C.push(P("**Per-client loss corrections (Part 7.3).** With a client constant a_k added to the logits, ℓ(Wu + b + a_k, y) is ℓ(Wu + b′, y) with b′ = b + a_k. The map b ↦ b + a_k is a translation, so every gradient agrees at corresponding points and, by induction over steps, every parameter except the bias follows the same trajectory. FedRS multiplies the logits of absent classes by α, which is plain training in the variables αw_c, αb_c with step size multiplied by α². ∎"));
C.push(P("**Uninformative fakes (Part 7.4).** If y is uniform on the C classes and independent of the feature, E_y[ℓ(f, y)] = log Σ_c e^{f_c} − (1/C) Σ_c f_c = log C + KL(uniform ‖ softmax f) = log C + (1/2C) Σ_c (f_c − f̄)² + O(‖f − f̄‖³). ∎"));
C.push(P("**Sums do not depend on the split (Part 7.5).** Σ_k Σ_{i∈D_k} T(x_i, y_i) = Σ_i T(x_i, y_i) because addition is associative and commutative; any estimate computed from these sums equals its centralised value. ∎"));
C.push(P("**Lemma 1** is the standard kernel bound: |E_G ℓ − E_P ℓ| = |⟨ℓ, μ_G − μ_P⟩| ≤ ‖ℓ‖·fake-to-real distance(G, P)."));
C.push(H1("Appendix B. Hyperparameters and experiment inventory"));
C.push(...table(["Dataset", "Fraction / train size", "Batch", "T_FE / T_KA / T_TN", "Latent l", "Feature h"], [
  ["MNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×16×16"],
  ["FashionMNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×16×16"],
  ["SVHN", "0.1 / 7,326", "64", "50 / 100 / 100", "16", "3×16×16"],
  ["CIFAR-10", "0.5 / 25,000", "128", "50 / 200 / 100", "50", "10×16×16"],
], [1.6, 2.0, 0.9, 2.0, 1.1, 1.6]));
C.push(P("Ours-A settings: 6,144 random Fourier features; herding pool 4n; consensus weight β = 0.5; logit adjustment temperature τ = 1; 10 synthetic epochs, T_r = 5."));
C.push(...table(["Experiment series", "Question answered"], [
  ["F01, F10, E20 (local)", "Main long-tail comparison and the first anchored generator"],
  ["E16, E21, E22 (local)", "Synthetic budget (Proposition 5); 100 clients"],
  ["E24, E25 (local)", "Kernel herding, server-trained generator (kernel matching), the anchored stack"],
  ["E26, E27, E29, E31, E32 (local)", "Formal differential privacy; balanced at 50/100 clients; hybrid"],
  ["K01–K04 (Kaggle)", "CIFAR-10 and SVHN baselines; regimes, privacy, ablations; client scaling; diffusion generator"],
  ["K08, K10, K11, K13 (Kaggle)", "Ours-A on CIFAR-10 and SVHN; SVHN switch; hybrid; our mixing variant vs MixUp"],
  ["K09 (Kaggle)", "Leave-one-out ablation of Ours-A"],
  ["E32 (local)", "Covariance correction, covariance version of the moment fix"],
  ["T01, T03 (local CPU, pre-registered)", "Collapse theorem on the real generator; last-layer gap and exact readout"],
  ["K14, K15 (Kaggle)", "Ablation seeds 3–4; existing collapse fixes vs holder-weighted averaging"],
  ["K16, K17, K19 (Kaggle)", "Every method under every readout; fair re-test on new seeds; further baselines"],
  ["K18, K20, K21 (Kaggle)", "SVHN 50 / 100 clients and the prospective κ test; mechanism controls; signal-fraction gate and balanced Stage-I loss"],
  ["K22, K23, K25 (Kaggle)", "Other imbalance settings; partial participation; per-round measurements"],
  ["N1–N7, X2, X4, X5, X14 (CPU, docs/research/math_checks)", "Checks of the mathematics; centralised comparison; κ test; balanced Stage-I pilot; MovieLens step frequency; membership attack"],
  ["E40 (CPU)", "Federated matrix factorisation on MovieLens-1M"],
], [3.5, 6]));
C.push(P("All code, logs and per-seed results are in the project repository (run ledger: experiments/LEDGER.md), and every number in this report can be regenerated from the logged runs."));

// ---------- document
const doc = new Document({
  creator: "[Student Name]", title: "Ours-A for Generator-Based Heterogeneous Federated Learning",
  
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
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Ours-A · Research Progress · v3", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "OursA_Backup_Handout.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024 / 1024).toFixed(1), "MB");
});
