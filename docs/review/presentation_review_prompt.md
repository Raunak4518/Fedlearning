# Presentation review prompt: metrics, logs, figures, tables and captions of the FedECS-LT suite

Use this prompt for the last check before notebooks are run (and again on the first returned results). It assumes
the reviewer can read `fedecs_lt/src/` and the outputs of a CPU notebook test (`results/` folder with `runs.jsonl`,
`logs/`, `predictions/`, `figures/`, and the notebook's `report.pdf`). Every item has an acceptance criterion; a
finding is either fixed in `fedecs_lt/src/` (then tests, rebuild) or written down with the reason it stays.

---

## Prompt

You are reviewing the measurement and presentation layer of a federated-learning experiment suite whose results
will go into an IEEE journal paper (IEEE Transactions style, two-column, figures 3.5 in or 7.16 in wide). Check
every metric, every log record, every figure and every table against the items below, on the outputs of **two**
fresh CPU tests: one main-comparison notebook (several clients with different head architectures; FedECS-LT,
GeFL-F, a loss-only baseline, an HtFLlib baseline and Local) and one one-shared-model notebook (FedECS-LT and a
LongTailLib baseline). Do not trust what the code says it does: open the files, render the pages, measure. Report
each finding with the file, the figure or table number, what is wrong, and the fix; then fix it.

### A. Metric definitions (against the literature, not only against the code's own docstring)

For each metric, find its line in `metrics.py` and confirm the formula, the averaging and the edge cases.

1. Balanced accuracy = mean of the per-class recalls over the classes present in the test set.
2. Macro-F1 and macro precision: mean over all classes; a class that is never predicted has precision 0 (as
   scikit-learn with `zero_division=0`); state this in the methods text.
3. MCC (multiclass, Gorodkin 2004) and Cohen's kappa from the confusion matrix; both 0 when undefined.
4. G-mean of the recalls with each recall floored at 1% (otherwise one zero recall makes it 0): the floor must be
   stated in every caption or table that shows G-mean.
5. Worst-class recall, head / medium / tail recall (frequency-rank groups: first 30%, next 30%, last 40% of the
   classes), recall SD: the group definition must be stated where they appear.
6. AUC-ROC (one-vs-rest, macro over present classes) and AUC-PR (average precision, macro): scikit-learn.
7. Top-5 accuracy: for C <= 5 it is trivially 100%; it must not be reported (or must be marked) when C <= 5.
8. ECE: 15 equal-width bins on the maximum probability; class-wise ECE (Kull et al., 2019) on p(c|x).
9. NLL with probabilities floored at 1e-6; Brier score summed over classes (range 0-2).
10. Own-mix accuracy: accuracy re-weighted to the evaluated model's own clients' class mix.
11. For prototype methods (FedProto, FedTGP) the "probabilities" are a softmax of negative distances: every
    calibration figure and table that shows them must say so.
12. Multi-architecture summaries: averaged over architectures, each counting once; the confusion matrix summed over
    architectures (counts are then architectures x test images: any table that shows counts must use the test
    set's own class counts).
13. Representation statistics (NC1, NC2, NC3, minority-collapse angle, linear CKA), generator statistics (MMD with
    median-bandwidth Gaussian kernel, mean mismatch, active units with threshold 0.01) and FLOPs (2 per
    multiply-accumulate; training = 3 x forward): definitions in the captions, citations given.
14. Statistics over seeds: mean and sample SD (ddof = 1); paired differences over common seeds with t intervals;
    Holm adjustment over the compared methods; with one seed no SD, no "+-", no interval.

Acceptance: every definition matches a cited source or is stated explicitly where it is shown; `tests/` has a
check for every metric that has a reference implementation.

### B. Logs (what the paper's numbers come from)

1. One `run_start` record per run with the full configuration (method, dataset, IF, alpha, K, seed, rounds,
   participation, class order, class names); one `run_end`.
2. Every per-round evaluation (`stage1_round`, `stage3_round`, `round_eval`) holds every per-class metric for every
   class for the aggregated / global model (`post`) and, where clients are evaluated, for each client (`pre`). The
   audit (`audit.json`) reports 0 problems.
3. Training-time quantities per round: losses, client drift, update norms, BatchNorm divergence (Stage I);
   reconstruction / KL loss, class-row norms (local and global), Adam second moments, fidelity before / after
   aggregation (Stage II); synthetic and real loss, consensus entropy, confidence and agreement, logit spread and
   real-data balanced accuracy before / after the synthetic phase, last-layer norms, predicted-class counts
   (Stage III).
4. Units and scales: fractions in [0, 1] in the logs (percent only in figures and tables); times in seconds; bytes
   in bytes; one key name per quantity across pipelines (no `bal` here and `balanced_acc` there).
5. Values rounded to 4 decimals in per-round logs; anything compared against a final value (rounds to target)
   allows for that rounding.
6. Reproducibility: seed, code version (commit), package versions, hardware and run time per stage are recorded.
7. `summary.zip` (sent back) holds `runs.jsonl`, every log, every figure CSV and the audit, but not predictions or
   models; it stays small (a few MB).

### C. Figures (every figure file, rendered at print size)

Check every figure in `results/figures/` at 100% zoom of its print size.

1. Size: width 3.5 in (column) or 7.16 in (page); height no more than needed; aspect such that text is not
   stretched.
2. Text: at least 8 pt after scaling (axis labels, ticks, legends, colour-bar labels, panel labels); one serif font
   (Times-like) throughout, matching the paper.
3. Lines at least 0.5 pt; markers visible; hatching readable.
4. No overlapping text anywhere: tick labels with each other, with axis labels, with panel labels, with a
   neighbouring panel, with a colour bar; no text cut off at the figure edge.
5. Legends: inside the axes only where they cover no data (t-SNE / PCA: top right inside, as the user asked; with
   headroom above the points); otherwise outside (below, or above for bar charts); never more than about 6
   entries in one column; entries in the same order as the methods everywhere.
6. Every method keeps one colour, marker and line style in every figure; ours is drawn first-colour, solid,
   thicker, hatched in bar charts; colours distinguishable for colour-blind readers and in greyscale (markers /
   dashes / hatching carry the distinction too).
7. Axis labels: quantity and unit ("Balanced accuracy (%)", "Time (s)", "Communication per client (MB)"), sentence
   case, no internal keys (no `final_bal`, `er_`, `T_FE`); mathematical symbols typeset ($s_c$, MMD$^2$).
8. Ticks: integer rounds on round axes; percent axes 0-100 where the quantity is a percent; log axes marked as such
   in the caption; no more than about 8 tick labels per axis; class axes ordered from most to least frequent.
9. No figure titles inside the image (IEEE puts the title in the caption); panels labelled (a), (b), ... below each
   panel, referenced in the caption.
10. Heatmaps: colour bar with label and unit; diverging colour maps centred at 0 for differences; sequential ones
    for magnitudes; cell annotations readable or omitted.
11. Error bars and bands: defined in the caption (one SD over seeds); absent (not zero-width) with one seed.
12. Figures with many classes (CIFAR-100, EMNIST, GTSRB): per-class lines readable (markers thinned), no 100-entry
    legends, class-colour repetition stated in the caption.
13. Empty or degenerate figures (all values equal, one point, a single bar) are not drawn, or are listed as not
    applicable with a reason.
14. Every figure has PDF (vector, fonts embedded), PNG and CSV of its data.
15. Embedding maps (t-SNE / PCA), following Wattenberg et al. (Distill 2016) and Kobak and Berens (Nat. Commun.
    2019): PCA to 50 dimensions, PCA initialisation, perplexity 30 for ~2,000 points, learning rate n/12; the same
    images and settings for every method compared; a baseline-vs-ours panel pair (as in the FedLTLib benchmark's
    feature-space figure); tail classes emphasised; a quantitative companion computed on the embeddings themselves
    (leave-one-out 10-NN accuracy, all and tail classes); a perplexity check (10 / 30 / 100); a UMAP map of the same
    embeddings (15 neighbours, minimum distance 0.1); the caption says that cluster sizes and distances between
    clusters are not interpretable.

### D. Captions

1. Each caption says what is plotted, on which data and split (test / validation / training), for which models
   (aggregated, global, clients' models), the setting (dataset, imbalance, clients, split, participation), the
   seeds, and what the bands / bars mean.
2. Formal names only: method names as published (GeFL-F, FedDF, FedLC, FedRS, Balanced Softmax, Local, FedProto,
   FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL, FedAvg, FedProx, MOON, CCVR, CReFF, FedETF, Fed-GraB, FedLoGe,
   FedYoYo, RUCR); ablations in "w/ ... / w/o ... / + ..." form; no internal words ("suite", "Track", "notebook",
   "cache", "pipeline", "holder", "ER", "copies" for models, file names, run ids).
3. Abbreviations defined at first use in each caption (LA, CSL, CCWA, MC, NC1-NC3, CKA, MMD, ECE, AUC-PR).
4. Numbers in captions (rounds, bins, sample counts, thresholds) match the code that drew the figure.
5. Not-applicable reasons are true for the experiment they are printed in.

### E. Tables (render the report pages)

1. Booktabs style (three rules), caption above the table, numbered with Roman numerals (IEEE).
2. Column headers short, with units; no header broken in the middle of a word; abbreviations defined in the caption.
3. Consistent decimals per column (percent with one decimal; times as whole seconds with thousands separators;
   NLL / Brier with three significant digits); no "-0"; "-" for not available; mean +- SD only with two or more
   seeds.
4. The best value of each column marked (shaded or bold) with the right direction (lower is better for ECE, NLL,
   Brier, recall SD, time, communication, FLOPs, own-mix SD).
5. Per-class tables: classes ordered from most to least frequent, blocks of at most 20 classes, the class table
   (index, original label, name, training and test images) first.
6. Statistical tables: number of seeds, interval type, test and multiple-comparison correction stated.

### F. Report document

1. The cover states what was run, the logging-audit result, figure and table errors (none), failed runs and the
   code origin of every method; the methods table gives the reference and code origin of each method.
2. Every figure in `figures/` appears in the PDF (completeness check) with its caption; not-applicable figures
   are listed with reasons; appendices list figure sources and every logged scalar of every run.
3. The report never stops on one failing figure or table, and the results are packed even if the report fails.

---

## How to run this review

1. Fresh CPU tests of a main-comparison notebook and a one-shared-model notebook with the current code
   (`fedecs_lt/tests/run_notebook_on_cpu.py`).
2. Automated checks: `plot_style.layout_problems` (overlapping or clipped text, legends over data) on every figure,
   the caption lint in `captions.lint` (internal words, missing setting or seeds), the audit, the report's figure /
   table error counts. All must be zero.
3. Render every figure and every report page and go through sections C-F by eye.
4. Fix, run `python -m pytest fedecs_lt/tests -q`, rebuild the notebooks, repeat 1-3 until clean.
