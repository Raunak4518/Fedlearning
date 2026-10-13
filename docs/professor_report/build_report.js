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
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], spacing: { before: 240, after: 120 } });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], spacing: { before: 180, after: 80 } });
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
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
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
  text: "Why rare classes vanish from GeFL-F, and how exactly aggregated statistics bring them back", italics: true, size: 24, color: GREY })] }));
for (const l of ["Submitted by: [Student Name]   ·   [Roll No.]", "Supervisor: [Professor Name]", "[Department], [Institute]", "October 2026"])
  C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: l, size: 24 })] }));
C.push(br());

// Executive summary
C.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Executive summary")], spacing: { after: 200 } }));
C.push(P("**Setting.** GeFL-F (Kang, Cha and Kang, IEEE TMC 2025) lets federated clients with *different* model architectures learn from one another. The clients share a small frozen feature extractor and a conditional feature generator trained by FedAvg, and each client trains its own private classifier head on real and generated features."));
C.push(P("**Problem.** Real federations are long-tailed. Under a 100:1 class imbalance with non-IID clients, GeFL-F loses 21–25 points of accuracy against its balanced setting, and its rare classes are almost never recognised. We proved why. Under FedAvg the generator's class-specific parameters shrink, round after round, for every class held by fewer than half of the clients (Proposition 1). Rare classes vanish from the generator, and every head is then trained on class-agnostic synthetic data."));
C.push(P("**Our method: ESA (Exact-Statistics Anchoring).** Class-level information should not be a parameter that FedAvg *estimates*. It should be anchored to statistics that secure aggregation computes *exactly*: per-class counts, sums, second moments and kernel features, which are sums over clients. ESA uses them at each point where class information enters the pipeline:"));
C.push(B("the generator's class identity comes from the exact class means;"));
C.push(B("generated samples are moved to the exact class mean and covariance by the minimum-displacement (Gelbrich) map;"));
C.push(B("they are then herded to the exact class kernel embedding;"));
C.push(B("where the class means do not identify the classes, the same statistics detect it, and class rows aggregated with exact counts are used instead."));
C.push(P("Each step carries a guarantee: the collapse cannot occur, the corrections displace samples as little as possible, and herding tightens each head's risk bound."));
C.push(...table(["Headline result (3 seeds, same seeds and splits for every method)", "GeFL-F", "ESA (ours)", "Gain"], [
  ["MNIST, long tail (balanced accuracy)", "75.25", "**94.99**", "+19.7"],
  ["FashionMNIST, long tail", "58.47", "**81.96**", "+23.5"],
  ["CIFAR-10, long tail", "33.74", "**≈ 47**", "≈ +13"],
  ["SVHN, long tail", "50.49", "**≈ 67**", "≈ +16"],
  ["MNIST, 100 clients, long tail", "75.98", "**94.30**", "+18.3"],
  ["Paper's own balanced setting (vs the best of the paper's ten methods)", "—", "above on MNIST, FMNIST, CIFAR-10, SVHN", "+0.8 to +2.1"],
], [5, 1.2, 1.8, 1.2], { highlight: [0, 1, 2, 3, 4] }));
C.push(P("**Work done.** 32 local experiment series and 13 Kaggle experiment series (about 40 notebooks) across 4 datasets, 3 seeds each. About 20 candidate techniques were tested; 9 were kept and 11 dropped, and each decision is backed by measurements and a mathematical explanation. The paper's baseline was validated against the authors' released code, and seven propositions were proved. A paper draft exists. Section 7 lists the few runs still in progress."));
C.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("Contents")], pageBreakBefore: true }));
C.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
C.push(P("*(If the contents list is empty, right-click it in Word and choose Update Field.)*", { run: { color: GREY, size: 18 } }));

// ---------------- PART 1
C.push(H1("1. The baseline: GeFL-F and the full lifecycle of its data"));
C.push(P("This part describes GeFL-F exactly as specified in the paper's appendix and as reproduced in our code: which models exist, what data each stage uses, how many epochs, which losses, what travels to the server (with dimensions), how it is aggregated and how knowledge returns to the clients."));
C.push(H2("1.1 Setting and notation"));
C.push(B("**Clients.** K clients (K = 10, 50 or 100). Client k holds private data D_k and a classifier head of architecture a(k), one of G = 10 CNNs (CNN-1 … CNN-10), assigned contiguously (K/10 clients per architecture)."));
C.push(B("**Data.** MNIST, FashionMNIST and SVHN use a 10% fraction of the training set (6,000 / 6,000 / 7,326 images); CIFAR-10 uses 50% (25,000). In the paper's setting the data are split IID across clients. Test sets are the full balanced test sets."));
C.push(B("**Models.** One shared feature extractor φ, one shared conditional generator G(h | y), and K private heads h_k."));
C.push(B("**Metric.** The paper's *best_mean_acc*: each architecture's test accuracy at its best round, averaged over the 10 architectures. Under the long tail we report balanced accuracy at the final round, and *tail recall* (mean recall of the four rarest classes)."));
C.push(...img("f01_overview.png", 640, "Figure 1. GeFL-F system view: what is local, what is shared, and which messages travel in each stage."));
C.push(H2("1.2 Model architectures and tensor shapes"));
C.push(...img("f03_architectures.png", 640, "Figure 2. Architectures. The CVAE-F's label projection W (C × 512) holds the only class-specific parameters; this matters in Part 2."));
C.push(...table(["Model", "Input", "Output", "Parameters", "Shared?"], [
  ["Feature extractor φ (Table XIV)", "image 1×28×28 (MNIST) / 3×32×32", "feature h: 3×14×14 = 588; SVHN 3×16×16 = 768; CIFAR-10 10×16×16 = 2,560", "36 / 90 / 390", "yes (FedAvg in Stage I, then frozen)"],
  ["Generator G, CVAE-F (Table XX)", "h (+ label map) → z (l = 16; 50 for CIFAR-10)", "ĥ, same shape as h", "≈ 5.5 M", "yes (FedAvg in Stage II, then frozen)"],
  ["Heads CNN-1 … CNN-10 (Tables XXI–XXIII)", "feature h", "10 logits", "3.9 k – 108.8 k", "no (averaged only within an architecture group)"],
], [2.2, 2.0, 3.0, 1.2, 2.0]));
C.push(H2("1.3 Lifecycle: three sequential stages"));
C.push(...img("f02_timeline.png", 640, "Figure 3. The three stages, with rounds, local epochs, data, optimiser and loss of each."));
C.push(H3("Stage I: feature-extractor warm-up (T_FE = 20 rounds; 50 for SVHN and CIFAR-10)"));
C.push(B("**Local data:** the client's real images x with labels y."));
C.push(B("**Local training:** 5 epochs per round; φ and the client's head trained jointly; SGD, learning rate 0.1, batch 64 (128 for CIFAR-10); loss = cross-entropy CE(h_k(φ(x)), y)."));
C.push(B("**Uplink:** φ's weights (36–390 numbers) and the head's weights (3.9 k–109 k numbers)."));
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
  ["I", "φ (36–390) + head (3.9 k–109 k numbers)", "FedAvg(φ); heads within groups", "averaged φ, heads"],
  ["II", "all generator weights, ≈ 5.5 M numbers (≈ 22 MB)", "FedAvg(G)", "averaged G"],
  ["III", "head weights (only if an architecture has several clients)", "average within architecture groups", "averaged heads"],
], [0.6, 3.2, 2.6, 1.6]));
C.push(P("**Privacy model.** Raw images, real features and labels never leave a client; only model parameters move. This is the same threat model as FedAvg, with secure aggregation as the standard protection of the uploads (Bonawitz et al., 2017)."));
C.push(H2("1.5 Validation of our reproduction"));
C.push(...table(["Setting", "Paper", "Authors' code", "Ours"], [
  ["MNIST IID, K = 10 (best_mean_acc)", "95.47", "96.07", "96.02 ± 0.04"],
  ["MNIST IID, K = 50", "95.04", "–", "95.16"],
  ["MNIST IID, K = 100", "94.63", "–", "94.61"],
  ["FashionMNIST IID, K = 10", "83.14", "–", "82.67 ± 0.31"],
  ["MNIST long tail, seed 0 (final balanced accuracy)", "–", "75.84", "73.92"],
], [4, 1.2, 1.4, 1.6]));
C.push(P("Our baseline matches the paper and the authors' code. Under the long tail it is 1.9 points *below* the authors' code, so every improvement we report is, if anything, understated."));

// ---------------- PART 2
C.push(H1("2. The problem: rare classes vanish under long-tailed clients"));
C.push(H2("2.1 The long-tailed federation"));
C.push(P("Class sizes follow n_c ∝ IF^{c/(C−1)} with imbalance factor IF = 100, so the rarest MNIST class has 24 training samples and the most frequent about 2,400. Client shares follow a Dirichlet(0.5) distribution, so most classes are held by only a few clients. Test sets stay balanced."));
C.push(...img("f05_problem.png", 640, "Figure 5. (a) GeFL-F's accuracy under the long tail. (b) Its tail classes' conditioning rows and fidelity. (c) The collapse bound of Proposition 1."));
C.push(H2("2.2 Diagnosis: the conditioning-row recursion"));
C.push(P("Consider class c held by m_c of the K clients. A client that does not hold c receives no data gradient on the class row w_c, only the weight-decay gradient λw_c. Adam normalises each coordinate's step by its running second moment, so a pure-decay gradient moves a coordinate by about η·sign(w_c) per step, regardless of λ. Over S local steps a holder can grow |w_c| by at most Sη, and a non-holder shrinks it by Sη until it reaches zero. Flat averaging then gives:"));
C.push(eq("eq_collapse", "Proposition 1"));
C.push(P("The row of every class held by fewer than half of the clients therefore shrinks **every round**. The decoder loses the class signal, and samples labelled c become class-agnostic. Without weight decay (or with SGD) the data step is still scaled by m_c/K, so rare classes learn m_c/K times slower: dilution instead of collapse."));
C.push(H2("2.3 Evidence"));
C.push(...table(["Measurement (IF = 100, Dir 0.5, K = 10)", "MNIST", "FashionMNIST", "SVHN"], [
  ["Tail-to-head row-norm ratio, GeFL-F", "0.33", "0.22", "0.18–0.49"],
  ["Same, with non-holders excluded from the average (HWA)", "0.80", "1.01", "0.96–1.05"],
  ["Tail fidelity of GeFL-F's generator (referee accuracy on synthetic tail features)", "8.8 %", "19.6 %", "–"],
  ["GeFL-F balanced accuracy: IID → long tail", "96.0 → 75.3", "82.8 → 58.5", "75.8 → 50.5"],
], [4.5, 1.4, 1.6, 1.4], { bad: [0, 2, 3] }));
C.push(P("All three predictions of Proposition 1 hold. Under a long tail the ratio is far below 1. It worsens with K. And removing non-holders from the rows' average restores it."));
C.push(H2("2.4 Why this breaks knowledge transfer"));
C.push(P("A head trained on synthetic data obeys the domain-shift bound below, where TV is the total-variation distance. If the generator's class-c samples are class-agnostic, the second term is large exactly for the rare classes, and no amount of synthetic training can teach them. The rest of this report replaces the parameter that collapses with a quantity that cannot."));
C.push(eq("eq_tv", "3"));

// ---------------- PART 3
C.push(H1("3. The research journey: every approach we tested"));
C.push(P("We followed one discipline throughout. Each idea was derived from a measured failure, first tested on a reduced schedule, and promoted to 3-seed full runs only when its mechanism was confirmed. Figure 6 places every approach in the pipeline. The subsections give each one's placement, model, result and verdict, kept short for the ones that failed."));
C.push(...img("f06_journey.png", 640, "Figure 6. Research map. Green: kept in the final method. Red: tested and dropped on the evidence."));
C.push(H2("3.1 Repairing the generator's aggregation"));
C.push(H3("HWA: holder-weighted aggregation (kept)"));
C.push(P("*Placement:* the server's Stage II aggregation of the class rows; every other weight is still FedAvg-averaged. *Model:* each client's row is weighted by the effective number of samples it holds (Cui et al., 2019), so non-holders get weight 0:"));
C.push(eq("eq_hwa", "4"));
C.push(P("*Result:* +9.0 (MNIST), +11.2 (FashionMNIST), +3.4 (CIFAR-10) and +6.0 (SVHN) over GeFL-F + LA under the long tail, with the row ratio restored to 0.80–1.05. When client counts are equal, HWA *is* FedAvg. The server needs only count-weighted sums, which secure aggregation provides; Laplace-noised counts at ε = 1 cost at most 1.3 points."));
C.push(H3("LCD: lazy conditioning decay (dropped)"));
C.push(P("*Placement:* clients, Stage II. *Model:* decay and update only the rows of classes present in a batch, which removes the collapse without sharing counts. *Result:* 78.6 against 79.1 for HWA (MNIST long tail, quick pass, 3 seeds). It adds nothing on top of HWA. It remains a fallback if even aggregated counts are forbidden."));
C.push(H2("3.2 Training the heads"));
C.push(H3("LA: logit adjustment with the client's own prior (kept; established technique)"));
C.push(P("*Placement:* each client's real phase. *Model:* Menon et al. (ICLR 2021), applied per client with its private prior π_k, which never leaves the client:"));
C.push(eq("eq_la", "5"));
C.push(P("*Result:* +3.6 to +6.3 points on all four datasets. It is necessary because the real phase (5 epochs) follows the synthetic phase every round and pulls the head back toward the client's own skew. In IID data π_k is uniform and LA has no effect."));
C.push(H3("CSL: consensus soft labels on synthetic features (kept)"));
C.push(P("*Placement:* the synthetic phase; clients upload only prediction sums on a shared synthetic pool. *Model:* the target mixes the generator's label with the mean prediction of the G architectures' heads, a cross-architecture distillation in the spirit of FedMD and FedDF:"));
C.push(eq("eq_csl", "6"));
C.push(P("*Result:* +2.5 on MNIST under the long tail; +0.5 to +0.8 in IID (MNIST 96.82 vs 96.01; CIFAR-10 59.85 vs 59.10; SVHN 76.30 vs 75.82). β = 0.5 beats β = 1, the bias–variance optimum predicted by Proposition 5."));
C.push(H3("T_s = 10: a larger synthetic budget (kept)"));
C.push(P("*Model (Proposition 5):* in a stylised estimator, the optimal synthetic share falls with the generator's label bias b. More synthetic training therefore pays only once the generator is faithful."));
C.push(eq("eq_budget", "Proposition 5"));
C.push(P("*Result:* with the anchored generator, T_s = 10 adds +1.7 on MNIST under the long tail and +3.9 at K = 100, where it added −0.05 to the original generator. This is the predicted interaction. On CIFAR-10 and SVHN, where generator fidelity is low, it raises the peak accuracy but lowers the final-round accuracy, as the proposition also predicts."));
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
  ["BBC: balanced bias calibration (Saerens et al., 2002)", "Fit C logit offsets per head so it predicts all classes equally often on balanced *anchored* synthetic data; on only when exact counts are imbalanced (max/min > 2)", "+0.35 to +1.4 under the long tail; ≈ +3 on CIFAR-10 LT; switched off in IID", "Kept (gated)"],
], [2.0, 3.2, 2.4, 2.2], { good: [3], bad: [0, 1, 2] }));
C.push(eq("eq_bbc", "7"));
C.push(H2("3.4 Alternative generators"));
C.push(...table(["Generator", "Placement and model", "Result", "Verdict and reason"], [
  ["FSG: Gaussian from exact statistics", "Server builds N(m_c, Σ_pooled) from secure-aggregated sums; no generator training", "MNIST LT 87.0; CIFAR-10 LT 44.2 (best before ESA); SVHN LT 44.2 < GeFL-F 50.5", "Dropped: its Bayes head is LDA (eq. 8), which reaches only 21 % on SVHN pixels"],
  ["Hybrid CVAE / Gaussian", "Mixture of the two generators", "87.1 vs 87.9 (full schedule; the quick pass had favoured it)", "Dropped"],
  ["KME-Gen", "Generator trained only at the server, by MMD to exact kernel embeddings", "MNIST 89.9, FMNIST 75.1 (+15 over GeFL-F) at 1/3000 of the communication", "Dropped as main method; noted as an efficient variant"],
  ["DDPM-F (paper's diffusion generator)", "Our components on the paper's best feature generator", "SVHN IID 70.37 vs DDPM GeFL-F 69.77 (p = 0.10); far below 76–77 with CVAE", "Dropped: slow (hours per seed) and weaker here"],
  ["ZP: ex-post latent prior", "Fit z ~ N(m, S) to the encoder means (exact sums)", "PC + ZP 91.69 vs PC + MC 92.31", "Dropped: MC fixes the same issue better"],
  ["PCR: class rows + mean anchor", "Both conditioning signals in one decoder", "Ties ESA on MNIST/FMNIST; SVHN LT 60.9 vs 65.6 for rows alone", "Dropped: the anchor under-disperses samples where means are uninformative"],
], [1.7, 2.8, 2.7, 2.6], { bad: [0, 1, 2, 3, 4, 5] }));
C.push(eq("eq_lda", "8"));
C.push(H2("3.5 Moment corrections, and how one operator replaced three"));
C.push(P("**MC (kept, then generalised).** It moves each generated class onto its exact mean and spread with the map that displaces samples least (Proposition 3): +1.6 (MNIST) and +1.6 (FashionMNIST) over the plain anchored generator."));
C.push(P("**MC-S (dropped in its first form).** It also matches the exact class covariance in a 256-dimensional principal subspace (Gelbrich map, Proposition 4). Results:"));
C.push(B("on SVHN, where class identity lives in second moments (pixel-space QDA 54 % against LDA 21 %), it added +6.0 under the long tail and +2.7 in IID;"));
C.push(B("on MNIST under the long tail it *lost* 1.6, all of it on rare classes (−4.2 tail recall). Their 24-sample covariances were shrunk toward the *pooled* covariance, overwriting the shape the generator had learned."));
C.push(P("**MC-B (kept).** It shrinks toward the *generator's own* class covariance instead, an empirical-Bayes estimate weighted by sample size. A rare class keeps the generator's shape and the map reduces exactly to MC; a common class gets its exact covariance."));
C.push(B("FashionMNIST long tail: +1.89 over MC (p = 0.016, every seed)."));
C.push(B("FashionMNIST IID: +1.78 (p < 0.001), reaching 85.75."));
C.push(B("MNIST long tail: a tie (−0.06)."));

// ---------------- PART 4
C.push(H1("4. The final method: ESA (Exact-Statistics Anchoring)"));
C.push(H2("4.1 Where the idea comes from"));
C.push(...table(["Source (research paper)", "What it gives us", "How ESA uses it"], [
  ["Bonawitz et al., Practical Secure Aggregation, ACM CCS 2017", "The server learns sums over clients exactly, no individual term", "Every class statistic ESA uses is such a sum"],
  ["Tan et al., FedProto, AAAI 2022; Shang et al., CReFF, IJCAI 2022", "Class prototypes and federated features as shared knowledge", "Exact class means become the generator's class identity, not a regulariser"],
  ["Gelbrich 1990; Olkin and Pukelsheim 1982 (optimal transport)", "Minimum-displacement map between two given covariances", "Moment correction of generated samples (Propositions 3–4)"],
  ["Chen, Welling and Smola, Kernel Herding, UAI 2010; Gretton et al., MMD, JMLR 2012", "Deterministic sample selection matching a kernel mean embedding", "Herding toward the exact class embedding (Lemma 1)"],
  ["Menon et al., Logit Adjustment, ICLR 2021; Zhang et al., FedLC, ICML 2022", "Fisher-consistent loss for label shift", "Client-prior logit adjustment of the heads"],
  ["Li and Wang, FedMD, 2019; Lin et al., FedDF, NeurIPS 2020", "Ensemble distillation across heterogeneous models", "Consensus soft labels on the anchored synthetic pool"],
  ["Cui et al., Class-Balanced Loss, CVPR 2019", "Effective number of samples", "Weights of the count-anchored rows (HWA)"],
  ["Saerens et al., Neural Computation 2002", "Prior correction of a classifier's outputs", "Gated balanced bias calibration"],
], [3.4, 3.0, 3.2]));
C.push(...callout("The single idea", ["Federated averaging *estimates* class-level parameters, and under a long tail the estimate collapses (Proposition 1).",
  "Secure aggregation *computes* class-level sums exactly, for any partition of the data: the pooled class mean from summed client sums equals the centralised one.",
  "ESA therefore anchors every class-level quantity of the generator to these exact statistics instead of to averaged parameters."]));
C.push(H2("4.2 Architecture"));
C.push(...img("f07_esa.png", 640, "Figure 7. ESA inside GeFL-F. Blue: components that change. Everything new is computed from sums returned by secure aggregation."));
C.push(H2("4.3 Mathematical model"));
C.push(H3("Step 1: exact statistics (one upload per client, after Stage I)"));
C.push(P("Each client uploads, under secure aggregation, its per-class counts n_kc, feature sums S_c, sums of squared norms Q_c and sums of bounded random Fourier features Φ_c (D = 6144, ‖φ‖ ≤ √2). For the covariance step it also uploads the pooled second moment and per-class second moments in a broadcast 256-dimensional basis. The server derives the class means, spreads, kernel embeddings and the separation ratio F:"));
C.push(eq("eq_stats", "9"));
C.push(H3("Step 2: class identity, with the separation gate"));
C.push(P("The generator models only the within-class residual around the exact mean, with a class-shared projection A, so no class-specific parameter exists and Proposition 1 has nothing to act on:"));
C.push(eq("eq_pc", "10"));
C.push(P("**Proposition 2 (separation bound).** If the decoder is L-Lipschitz in its second argument, two synthetic classes can differ only in proportion to how far apart their exact means are:"));
C.push(eq("eq_sep", "11"));
C.push(P("The gate follows from this. If F < 0.05, the means cannot carry class identity, and ESA uses the paper's class rows aggregated with exact counts (equation 4) instead. F is 0.29 (MNIST), 0.53–0.60 (FashionMNIST), about 0.14 (CIFAR-10) and 0.01 (SVHN), so any threshold between 0.02 and 0.1 makes the same choices: the decision is not tuned."));
C.push(H3("Step 3: moment anchoring (MC-B)"));
C.push(P("**Proposition 3.** Among all maps that give class c its exact mean and spread, the scalar map below displaces the samples least (proof by Cauchy–Schwarz, Appendix A):"));
C.push(eq("eq_mc", "12"));
C.push(P("**Proposition 4.** Among all maps that give the exact mean and covariance, the linear map with the matrix below is displacement-optimal, and its cost is the Gelbrich (Bures) distance:"));
C.push(eq("eq_gelbrich", "13"));
C.push(P("MC-B uses the empirical-Bayes covariance target below. It reduces to Proposition 3 for a class with few samples and to Proposition 4 for a class with many:"));
C.push(eq("eq_mcb", "14"));
C.push(H3("Step 4: distribution anchoring (kernel herding)"));
C.push(P("**Lemma 1.** For a head whose loss lies in an RKHS ball of radius B, the synthetic-to-real risk gap is bounded by the maximum mean discrepancy. ESA draws 4n corrected candidates per class and keeps n by Frank–Wolfe herding toward the exact embedding μ_c, which lowers exactly this term:"));
C.push(eq("eq_kh", "15"));
C.push(H3("Step 5: heads and calibration"));
C.push(P("Heads train T_s = 10 epochs on the anchored pool with consensus labels (equation 6), then 5 real epochs with logit adjustment (equation 5). After training, gated BBC (equation 7) removes the residual bias when the exact counts are imbalanced."));
C.push(H2("4.4 What ESA adds to communication and disclosure"));
C.push(...table(["Release", "Size", "When", "Protection"], [
  ["Class counts, sums of h and of ‖h‖², sums of RFF features", "≈ 0.3 MB per client", "once, after Stage I", "secure aggregation; optional (ε, δ)-DP"],
  ["Pooled second moment, class second moments in the 256-d basis (MC-B)", "D² + C·256² numbers (≈ 4 MB MNIST, ≈ 29 MB CIFAR-10)", "once", "secure aggregation"],
  ["Consensus labels: prediction sums on a shared synthetic pool", "≈ 24 KB per synthetic epoch", "each Stage-III round", "sums only; no real data"],
  ["Generator, extractor, heads", "as in GeFL-F", "as in GeFL-F", "unchanged"],
], [3.6, 2.6, 1.6, 1.8]));
C.push(P("The one-time additions are small compared with Stage II's ≈ 2.2 GB per client. For formal privacy, features are clipped to a public norm bound, and each release receives Gaussian noise composed under zero-concentrated DP:"));
C.push(eq("eq_dp", "16"));

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
  ["**ESA (ours)**", "**94.99**", "**81.96**", "**≈ 47**", "**≈ 67**"],
  ["oracle: ESA's heads, last layer re-fit on pooled real data", "94.93", "81.55", "50.6", "–"],
], [4.6, 1.2, 1.4, 1.2, 1.2], { highlight: [5] }));
C.push(P("Tail recall rises from 51.9 to 90.7 (MNIST) and from 35.4 to 74.5 (FashionMNIST). On MNIST ESA reaches the accuracy of its own heads re-fit on pooled real data. On SVHN the gate selects the count-anchored rows, as F = 0.01. *Notes:* MNIST and FashionMNIST use MC-B; CIFAR-10 and SVHN use the MC version (MC-B there is listed in Part 7). Values marked ≈ were printed with two significant digits by the Kaggle summaries."));
C.push(...img("f08_longtail.png", 640, "Figure 8. Long-tailed clients: ESA against GeFL-F and the strongest alternatives."));
C.push(H2("5.3 The paper's own balanced setting, against all ten of its methods"));
C.push(...table(["best_mean_acc, IID", "GeFL-F (paper)", "Best of the paper's ten methods", "Our GeFL-F", "**ESA**", "ESA − best"], [
  ["MNIST, K = 10", "95.47", "96.44 (GeFL, DDPM)", "96.01", "**97.62**", "+1.18"],
  ["MNIST, K = 50", "95.04", "95.04 (GeFL-F, CVAE-F)", "95.43", "**96.93**", "+1.89"],
  ["MNIST, K = 100", "94.63", "94.63 (GeFL-F, CVAE-F)", "94.88", "**96.77**", "+2.14"],
  ["FashionMNIST, K = 10", "83.14", "84.28 (GeFL-F, DDPM-F)", "82.76", "**85.75**", "+1.47"],
  ["FashionMNIST, K = 50", "82.21", "82.96 (GeFL-F, DDPM-F)", "81.50", "**82.96**", "±0.00"],
  ["CIFAR-10, K = 10", "55.86", "59.36 (GeFL, DDPM)", "59.10", "**60.93**", "+1.57"],
  ["SVHN, K = 10", "76.26", "76.26 (GeFL-F, CVAE-F)", "75.82", "**77.07**", "+0.81"],
], [2.0, 1.2, 2.2, 1.2, 1.0, 1.1], { highlight: [0, 1, 2, 3, 5, 6] }));
C.push(P("The ten methods are GeFL and GeFL-F with DCGAN, CVAE and DDPM generators (two guidance settings), plus FedProx and FedALA. ESA beats the best of them in 6 of 7 cells and ties the seventh. Gains are smaller than under the long tail, as expected: with balanced clients, FedAvg has no rare class to lose. FashionMNIST K = 100, SVHN K = 50/100 and CIFAR-10 K = 50/100 are in progress (Part 7)."));
C.push(...img("f09_iid.png", 640, "Figure 9. The paper's balanced setting: ESA against the paper's GeFL-F and the best of its ten methods."));
C.push(H2("5.4 Data augmentation (paper's Table IV), and why MixUp does not transfer"));
C.push(...table(["CIFAR-10, IID, K = 10", "Accuracy"], [
  ["FedAvg / + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "55.65 / 60.07 / 58.95 / 53.96 / 56.99"],
  ["Image-space GeFL + MixUp / + CutMix / + AugMix / + AutoAugment (paper)", "62.67 / 61.66 / 56.47 / 59.97"],
  ["ESA, no augmentation", "60.93"],
  ["ESA + feature-space MixUp (our run, same seeds)", "59.98"],
  ["ESA + VCT, our MixUp variant with federated partners", "60.29"],
], [6, 3]));
C.push(P("ESA beats 7 of the 9 augmented baselines without any augmentation. The two that remain higher train full models on raw images with a DCGAN; they have no frozen feature extractor, so augmentation can regularise every layer. Inside GeFL-F's protocol the extractor is frozen, and we measured that MixUp in feature space *lowers* accuracy (59.98 against 60.93). The gap is therefore a property of the protocol, not of the method."));
C.push(H2("5.5 More clients, and formal privacy"));
C.push(...img("f11_scaling_privacy.png", 640, "Figure 10. (a) 100 clients, MNIST long tail. (b) ESA with every statistic made (ε, 10⁻⁵)-differentially private."));
C.push(...table(["Setting", "GeFL-F", "HWA + LA", "FSG + LA", "**ESA**"], [
  ["MNIST long tail, K = 100", "75.98", "83.07", "86.05", "**94.30**"],
], [3.6, 1.4, 1.4, 1.4, 1.4], { highlight: [0] }));
C.push(P("ESA loses only 0.5 points from K = 10 to K = 100, while the count-anchored CVAE loses 4.8: the mean-anchored generator has no class parameter to estimate from shrinking per-client data. The gains are +18.3 over GeFL-F (p = 0.0004) and +8.3 over the Gaussian generator (p = 0.001), on every seed."));
C.push(...table(["Long tail, K = 10", "GeFL-F (no formal guarantee)", "ESA exact", "ESA, ε = 8", "ESA, ε = 2"], [
  ["MNIST", "75.25", "94.82", "90.36", "85.06"],
  ["FashionMNIST", "58.47", "79.67", "72.77", "65.59"],
], [2.2, 2.4, 1.4, 1.4, 1.4]));
C.push(P("Formal privacy costs 4.5–6.9 points at ε = 8 and 9.8–14.1 at ε = 2, almost all of it on the rarest classes, whose 24 samples cannot be both private and accurate. Even at ε = 2 ESA stays 7–10 points above GeFL-F, which gives no formal guarantee. Feature-space memorisation (MND, the paper's privacy metric) equals GeFL-F's."));
C.push(H2("5.6 What each component contributes"));
C.push(...img("f10_ablation.png", 640, "Figure 11. Adding components one at a time on the same seeds (long tail)."));
C.push(P("Kernel herding helped in all 9 paired comparisons (sign test p = 0.004), and MC-B over MC is significant on FashionMNIST. A leave-one-out ablation of the final method is running (Part 7)."));
C.push(H2("5.7 The hard case: SVHN, where class means carry no class information"));
C.push(...img("f12_svhn.png", 560, "Figure 12. SVHN (F = 0.01). The separation gate chooses count-anchored class rows, the best option measured."));
C.push(P("In the shared feature space, SVHN's nearest-class-mean accuracy is 12 % (chance is 10 %), so Proposition 2 forbids a mean-anchored generator from separating the classes. This explains why the plain anchored stack scored 55.8. The covariance correction recovered most of the gap (+6.0, p = 0.016), and the gate's choice of class rows recovers all of it (65.6, +15 over GeFL-F)."));
C.push(H2("5.8 Mechanism measurements"));
C.push(...table(["Quantity (long tail)", "GeFL-F", "ESA"], [
  ["Tail fidelity (referee accuracy on synthetic tail features), MNIST / FMNIST", "10.3 / 19.9 %", "88.6 / 78.7 %"],
  ["Tail spread relative to real data, raw → after correction and herding", "–", "0.39–0.49 → 0.93–0.98"],
  ["IID fidelity vs the referee's accuracy on real data (FMNIST)", "–", "0.837 vs 0.839"],
], [5, 2, 2]));
C.push(P("After correction and herding the synthetic features are exactly as hard as real ones: realistic, not artificially pure. That is the condition under which training on them transfers to real data."));

// ---------------- PART 6
C.push(H1("6. Justification, alternatives and limitations"));
C.push(H2("6.1 Why this approach"));
C.push(B("**It removes the cause, not a symptom.** The failure is a FedAvg-estimated class parameter (Proposition 1). ESA's generator has none, so the collapse is impossible for any number of clients, which the K = 100 result confirms."));
C.push(B("**Every step is optimal for its target.** The moment correction is displacement-minimal (Propositions 3–4), herding minimises the term that bounds transfer (Lemma 1), and the gate follows from a proved limit (Proposition 2)."));
C.push(B("**It keeps GeFL-F's protocol and privacy model.** It adds one small secure-aggregated upload. Every release can be made formally differentially private."));
C.push(B("**The evidence is strong.** Results are paired over 3 seeds against a baseline validated against the authors' code, with all failures reported."));
C.push(H2("6.2 Anticipated questions"));
const qa = [
  ["Why not simply run FedAvg longer, or tune the generator?", "Proposition 1 holds per round, so more rounds shrink the rare rows further. Removing weight decay only turns collapse into m_c/K-slower dilution."],
  ["Why not a stronger generator, such as the paper's diffusion model DDPM-F?", "Any generator whose class conditioning is FedAvg-trained inherits the recursion. With our components DDPM-F reached 70.4 on SVHN IID against 77.1 for ESA, at many times the compute."],
  ["Why not generate directly from the statistics (a Gaussian, or the server-only KME-Gen)?", "A Gaussian with a shared covariance can teach heads only the LDA rule (21 % on SVHN pixels). KME-Gen is 15 points above GeFL-F but below the learned, anchored decoder. Statistics are best used to *anchor* a learned generator, not to replace it."],
  ["Why not existing federated long-tail methods (FedLC, CReFF, CLIMB, FedLoGe)?", "They assume a shared, averageable classifier, which model-heterogeneous FL does not have. Their principles are covered: logit adjustment (FedLC) is part of every baseline we report, and classifier re-training on synthetic features (CReFF) was tested and was weaker."],
  ["Why not data augmentation (MixUp, CutMix)?", "It is orthogonal to the imbalance problem, and in GeFL-F's feature space MixUp lowers accuracy (Part 5.4). The paper's augmentation gains come from image-space training that GeFL-F does not allow."],
  ["Does sharing class statistics leak privacy?", "They are federation-level sums under secure aggregation, the same kind as FedAvg's. A class held by a single client is the exception, and formal DP closes it at the measured cost of Part 5.5."],
  ["Is the separation gate dataset tuning?", "No. It is computed from released statistics without labels, follows from Proposition 2, and its threshold can move by a factor of 5 without changing any decision."],
  ["Why report balanced accuracy rather than the paper's metric under the long tail?", "With a 100:1 tail, plain accuracy rewards ignoring rare classes. Balanced accuracy and tail recall measure what the federation needs. In the balanced setting we use the paper's own metric."],
];
for (const [q, a] of qa) { C.push(P("**" + q + "**")); C.push(P(a)); }
C.push(H2("6.3 Limitations"));
C.push(B("The gate threshold was set from four datasets. A dataset with intermediate separation would need both branches evaluated."));
C.push(B("Gains in balanced data are +1 to +2 points. The method is designed for, and mainly helps, label skew and imbalance."));
C.push(B("The frozen feature extractor bounds every method, as GeFL-F's protocol fixes it (the CIFAR-10 oracle is about 51 under the long tail)."));
C.push(B("Strong formal privacy (ε = 2) costs 10–14 points, mostly on the rarest classes. The generator's FedAvg training is not DP, in GeFL-F either."));
C.push(B("On hard data (CIFAR-10, SVHN) the 10-epoch synthetic budget trades final-round accuracy for peak accuracy, as Proposition 5 predicts."));

// ---------------- PART 7
C.push(H1("7. Status and next steps"));
C.push(...table(["Item", "Status", "What it adds"], [
  ["Leave-one-out ablation of the final method (Kaggle NB18)", "running", "Proves each component is necessary"],
  ["SVHN: count-anchored rows with the covariance correction (NB25)", "running", "Final detail of the low-separation branch"],
  ["SVHN and CIFAR-10 at 50 / 100 clients (NB26–NB29); FashionMNIST at 100 (NB17)", "running", "Remaining cells of the paper's main comparison"],
  ["The paper's other federated baselines under the long tail (NB13)", "running", "Completes the baseline families under imbalance"],
  ["MC-B on CIFAR-10 and SVHN", "planned", "One moment operator for every dataset"],
  ["Paper write-up (draft exists: single idea, Propositions 1–5)", "in progress", "Submission to IEEE TMC (GeFL's venue) or an FL venue"],
], [5, 1.4, 3.2]));
C.push(P("**Plan for the paper.** Present ESA as one principle: anchor class-level information to exactly aggregated statistics. The main text has the diagnosis, the three anchors and the gate, Propositions 1–5, the four result tables and the privacy analysis. The approaches that did not work go into an appendix, since they explain what the method actually fixes."));

// ---------------- APPENDIX
C.push(H1("Appendix A. Proofs"));
C.push(P("**Proposition 1.** A non-holder's gradient on w_c is λw_c. Adam's update is η·m̂/√v̂, and for a constant-sign gradient m̂/√v̂ → sign(w_c), so each step moves |w_c| by η toward zero. A holder's normalised step is at most η in magnitude. With weights m_c/K for holders and 1 − m_c/K for non-holders, the averaged change is at most (2m_c/K − 1)Sη. ∎"));
C.push(P("**Proposition 2.** Couple both classes through the same latent z. ReLU is 1-Lipschitz, so ‖X_c − X_c'‖ ≤ ‖m_c − m_c'‖ + L‖A(m_c − m_c')‖. Taking expectations bounds W₁. ∎"));
C.push(P("**Proposition 3.** With Y = T(X), E‖Y − X‖² = V_c + Ṽ_c + ‖m_c − m̃_c‖² − 2E⟨Y − m_c, X − m̃_c⟩. By Cauchy–Schwarz the last term is at most √(V_c Ṽ_c), with equality if and only if Y − m_c = √(V_c/Ṽ_c)(X − m̃_c). ∎"));
C.push(P("**Proposition 4.** Let K = E[(Y − m_c)(X − m̃_c)ᵀ]. The joint covariance is positive semi-definite, so K = Σ_c^{1/2} R Σ̃_c^{1/2} with ‖R‖₂ ≤ 1. Hence tr K ≤ ‖Σ̃_c^{1/2}Σ_c^{1/2}‖_* = tr(Σ̃_c^{1/2}Σ_cΣ̃_c^{1/2})^{1/2}. The map T*(x) = m_c + A_c(x − m̃_c) attains this bound and gives Cov T*(X) = Σ_c. In the subspace form, the displacement splits orthogonally, and the scalar map on the complement attains Proposition 3. ∎"));
C.push(P("**Proposition 5.** With w = λn_s/(n_r + λn_s), the error becomes σ²(1 − w)/n_r + w²b², a convex parabola minimised at w* = σ²/(2n_rb²). The optimum is clipped to the feasible range, so the optimal synthetic share is non-increasing in b and reaches its maximum as b → 0. ∎"));
C.push(P("**Lemma 1** is the standard RKHS bound: |E_G ℓ − E_P ℓ| = |⟨ℓ, μ_G − μ_P⟩| ≤ ‖ℓ‖·MMD(G, P)."));
C.push(P("**Proposition 6 (MixUp, used for VCT).** Cross-entropy is linear in the target, so a second-order expansion in δ = (1 − λ)(x' − x) shows that the MixUp objective depends on its partners only through the class prior, class means and class covariances (checked numerically to within 0.001 of the loss):"));
C.push(eq("eq_mixup", "17"));
C.push(H1("Appendix B. Hyperparameters and experiment inventory"));
C.push(...table(["Dataset", "Fraction / train size", "Batch", "T_FE / T_KA / T_TN", "Latent l", "Feature h"], [
  ["MNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×14×14"],
  ["FashionMNIST", "0.1 / 6,000", "64", "20 / 100 / 50", "16", "3×14×14"],
  ["SVHN", "0.1 / 7,326", "64", "50 / 100 / 100", "16", "3×16×16"],
  ["CIFAR-10", "0.5 / 25,000", "128", "50 / 200 / 100", "50", "10×16×16"],
], [1.6, 2.0, 0.9, 2.0, 1.1, 1.6]));
C.push(P("ESA settings: separation threshold F = 0.05; 256-dimensional covariance basis with prior strength k = 256; 6,144 random Fourier features; herding pool 4n; consensus weight β = 0.5; LA temperature τ = 1; BBC gate max/min > 2; T_s = 10, T_r = 5."));
C.push(...table(["Experiment series", "Question answered"], [
  ["F01, F10, E20 (local)", "Main long-tail comparison and the first anchored generator"],
  ["E16, E21, E22 (local)", "Synthetic budget (Proposition 5); 100 clients"],
  ["E24, E25 (local)", "Kernel herding, KME-Gen, the anchored stack"],
  ["E26, E27, E29, E31, E32 (local)", "Formal DP; IID at 50/100 clients; hybrid; MC-S; MC-B"],
  ["K01–K04 (Kaggle)", "CIFAR-10 and SVHN baselines; regimes, privacy, ablations; client scaling; diffusion generator"],
  ["K08, K10, K11, K13 (Kaggle)", "Anchored stack on CIFAR-10 and SVHN; hybrid; covariance anchoring; VCT vs MixUp"],
], [3.5, 6]));
C.push(P("All code, logs and per-seed results are in the project repository, and every number in this report can be regenerated from the logged runs."));

// ---------- document
const doc = new Document({
  creator: "[Student Name]", title: "Exact-Statistics Anchoring for Generator-Based Heterogeneous FL",
  features: { updateFields: true },
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
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "ESA: Exact-Statistics Anchoring · Research Progress Report", size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: GREY })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "ESA_Research_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, (buf.length / 1024 / 1024).toFixed(1), "MB");
});
