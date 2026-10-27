# From notebook results to the paper: a step-by-step guide

This guide is for writing the results of the Kaggle notebooks into the paper without help. It says what each
notebook gives, which paper table or figure it feeds, the exact commands that turn the returned files into LaTeX,
and the rules the text must follow. Read sections 1 to 3 once; afterwards sections 4 to 6 are the day-to-day
reference.

Files this guide refers to:

| File | What it is |
|---|---|
| `fedecs_lt/notebooks/START_HERE.md` | the run sheet: every notebook, its folder (priority), runs and hours |
| `fedecs_lt/merge_results.py` | joins several notebooks' results into one experiment; one report, one set of tables |
| `fedecs_lt/paper_tables.py` | writes the paper's numbers as LaTeX (`.tex`) and CSV from a results folder |
| `docs/paper_draft.md` | the full working record of the paper; update it first |
| `docs/latex/paper.tex` | the submission paper (9 pages + appendix); update it second |
| `docs/review/adversarial_review_dossier.md` | claims that are false or unsupported: never write them |

---

## 1. What comes back from each notebook

Every notebook writes four files to its Kaggle *Output* tab (`NAME` = the notebook's name):

| File | Size | Keep where | Contains |
|---|---|---|---|
| `NAME__report.pdf` | 1-8 MB | Kaggle (and your Drive) | every table and figure of that notebook, with captions; the cover says whether anything failed |
| `NAME__summary.zip` | 2-60 MB | Kaggle; its small files in the repository (below) | every number and log as text (`runs.jsonl`, `runs.csv`, per-round logs, figure data as CSV); `runs.jsonl` alone is enough for **every table** |
| `NAME__results.zip` | large | Google Drive | the summary plus predictions and embeddings; needed to **redraw figures** (confusion matrices, t-SNE, UMAP, calibration) |
| `NAME__models.zip` | large | Google Drive | trained weights; only needed if a reviewer asks for a new analysis |

Do not commit the zips or the PDFs (together several hundred MB). Kaggle keeps every output as long as the notebook
exists, and `fetch_kaggle.py` (below) downloads it again at any time.

### First check of every returned notebook (2 minutes)

Open `NAME__report.pdf`. Its cover must show:
1. **Failed runs: 0.** If a run failed, its error is in the report's appendix and in `failures.jsonl`
   inside the zip. Re-run the same notebook; it skips nothing on a new Kaggle session, so it runs everything again.
   If only a timeout cut it short (no new run starts after 11 h), the finished runs are all in the files: keep
   them and run the missing ones later.
2. **Logging audit: 0 problems**, **figure errors: 0**, **table errors: 0**.
3. The number of runs equals the "Runs" column of `START_HERE.md`.

Then file it in the repository with one command (the notebook must be public, or shared, on Kaggle):

```
python fedecs_lt/fetch_kaggle.py https://www.kaggle.com/code/ACCOUNT/NOTEBOOK-SLUG
```

It writes `fedecs_lt/results/NAME/`: `runs.jsonl`, `runs.csv`, `failures.jsonl`, `audit.json`,
`report_info.json`, `code_version.txt` and a `summary.md` (runs finished against the plan, failed runs grouped by
cause). It adds the Kaggle link to `fedecs_lt/results/KAGGLE_SOURCES.md`, and puts the report and the zips in
`fedecs_lt/results/_downloads/` (not committed). `python fedecs_lt/fetch_kaggle.py --all` downloads every recorded
notebook again; add `--results` to also get the large `NAME__results.zip`, which figures need. Commit the new
folder. Every command in section 3 accepts these folders instead of the zips.

---

## 2. Which notebook feeds which part of the paper

A notebook's own report only knows its own runs. A paper table often needs several notebooks: the parts of a split
notebook, the three seeds of a one-shared-model setting, or an ablation notebook plus its reference runs. Section 3
shows how to join them. Here is the map.

### Track 1: the main setting (a different model per client, shared frozen extractor)

| Notebooks (folder) | Gives | Paper |
|---|---|---|
| `NB10_main_mnist`, `NB10_main_fmnist`, `NB10_main_svhn` (folder 1) | 13 methods × seeds 3-5 on each dataset | **Main comparison table** (replaces `tab:fair`); **readout table** (replaces `tab:readout`); main-text figures |
| `NB10_main_cifar10_part1of3` + `part2of3` + `part3of3` (folder 1) | the same on CIFAR-10-LT: join the three parts | same tables, CIFAR-10 column |
| `NB10_main_cifar100_part1of2` + `part2of2` (folder 1) | the same on CIFAR-100-LT: join the two parts | same tables, CIFAR-100 column |
| `NB10_main_bloodmnist`, `dermamnist`, `organamnist`, `emnist`; `pathmnist`, `gtsrb`, `fer2013` (3 parts each) (folder 3) | the same on datasets with natural class imbalance | a second comparison table "datasets with natural imbalance" (main text if space, else appendix) |
| `NB30_ablation_cifar10` + `NB10_main_cifar10_part1of3` (folders 2 and 1) | GeFL-F w/o generator, GeFL-F + LA, GeFL-F w/ Gaussian generator + LA, FedDF + LA, paired with FedECS-LT and GeFL-F on the same splits and seeds | **ablation table** (`app:ablation`, and one line in the main text) |
| `NB30_ablation_cifar100` + `NB10_main_cifar100_part1of2` | the same on CIFAR-100-LT | ablation table, second column |

Part 1 of every split Track-1 notebook holds FedECS-LT, GeFL-F, Balanced Softmax, FedLC, FedRS, FedDF and Local.
The later parts hold FedProto, FedTGP, FedGen, FedGH, LG-FedAvg and FedMRL. The ablations need part 1 only.

### Track 2: one shared model (the standard federated long-tail protocol, FedLTLib)

| Notebooks (folder) | Gives | Paper |
|---|---|---|
| `NB20_track2_cifar10_IF100_s3_part1of2` + `part2of2`, the same for `s4` and `s5` (6 notebooks, folder 2) | FedECS-LT and 11 long-tail methods (authors' LongTailLib code), CIFAR-10-LT ratio 100, seeds 3-5 | **one-shared-model table** (new; main text), CIFAR-10 column; round curves (Fig. 71, 72) |
| the same 6 for `cifar100` (folder 2) | CIFAR-100-LT ratio 100, seeds 3-5 | same table, CIFAR-100 column |
| `NB20_track2_cifar10_IF050_s3_*`, `..._IF010_s3_*`, the same for `cifar100`, and `NB29_track2_long_runs_1` (folder 4) | ratios 50 and 10, **seed 3 only**; NB29 holds FedYoYo at ratio 10 | extra columns "ratio 50 / ratio 10" of the one-shared-model table, marked "one seed" |

`NB20_track2_cifar10_IF100_s3_part2of2` also holds **"FedECS-LT (200 extractor rounds)"**, a diagnosis run. Our method
normally trains its extractor for 100 rounds and the baselines train for 200. This run shows what changes when the
extractor gets the same 200 rounds. Report it as a footnote or one row, never as the main FedECS-LT number.

### Not from these notebooks (keep as they are)

The paper's analysis part comes from the earlier experiments (`experiments/results/`, numbers in
`docs/paper_assets/`): `tab:fixes`, `tab:central`, `tab:robust`, `tab:recsys`, `fig_kappa`, `fig_signal`, the
privacy table and the proofs. Keep them. Their protocol paragraph must still say they come from the earlier
(PyTorch) implementation with seeds 0-2 / 3-5 as stated there. **Never mix the two implementations in one table:** a
table is all Flower runs or all earlier runs.

`NB00_acceptance` is not a paper table. It is the evidence that the Flower code reproduces the earlier results. Use
one sentence in the protocol paragraph: "the Flower implementation gives the same client splits as the original code
and the same accuracy within seed noise (MNIST 94.3 vs 94.5, CIFAR-10 40.7 vs 42.3)". The numbers are in
`fedecs_lt/results/NB00_acceptance/summary.md`.

---

## 3. Turning returned files into tables and figures

### One-time setup on your computer (no GPU needed)

```
git pull
pip install numpy scipy scikit-learn matplotlib reportlab
pip install umap-learn        # optional: only for the UMAP figure (Fig. 85)
python -m pytest fedecs_lt/tests/test_paper_tables.py -q     # must print "1 passed"
```

Python 3.10 or newer. PyTorch is not needed for tables or reports.

### Tables (fast, from the folders in `fedecs_lt/results/`)

One command per paper table. It joins the inputs and writes `OUT_DIR/tables/`. `ALL.md` there shows every table
readably. Each table also comes as `.tex` (paste into the paper) and `.csv` (open in Excel). `--no-report` skips
the PDF, so it takes seconds.

Main comparison, folder 1, all five datasets in **one** call. The paper's table then has one column per dataset:

```
python fedecs_lt/merge_results.py --no-report paper_build/main_tier1 "Main comparison" \
  fedecs_lt/results/NB10_main_mnist \
  fedecs_lt/results/NB10_main_fmnist \
  fedecs_lt/results/NB10_main_svhn \
  fedecs_lt/results/NB10_main_cifar10_part1of3 \
  fedecs_lt/results/NB10_main_cifar10_part2of3 \
  fedecs_lt/results/NB10_main_cifar10_part3of3 \
  fedecs_lt/results/NB10_main_cifar100_part1of2 \
  fedecs_lt/results/NB10_main_cifar100_part2of2
```

(On Windows write the command on one line without the `\`.) In `paper_build/main_tier1/tables/` you then find:

| File | Paper |
|---|---|
| `across_track1_final_bal.tex` | **main comparison**: methods × datasets, balanced accuracy, heads as trained, mean ± SD over seeds 3-5, best in bold |
| `across_track1_paired.tex` | **FedECS-LT minus each method**, per dataset: difference [95% CI], bold where Holm-adjusted p < 0.05 (the layout of the current `tab:fair`) |
| `across_track1_er_bal.tex` | the same with the exact readout (the layout of the current `tab:readout`) |
| `across_track1_final_f1.tex`, `..._final_tail.tex`, `..._final_worst.tex` | macro-F1, tail-class recall, worst-class recall: main text if space, else appendix |
| `<dataset>_main.tex` | per dataset: every method × 8 main metrics (balanced accuracy, macro-F1, precision, MCC, G-mean, tail and worst-class recall, AUC-PR): appendix, one per dataset |
| `<dataset>_readout.tex`, `<dataset>_more.tex`, `<dataset>_paired.tex` | the exact readout; accuracy, kappa, top-5, AUC-ROC, head / medium recall, recall SD, calibration and cost; the paired test with p-values: appendix |

The same for the other tables. Only the inputs change:

| Paper table | Inputs (folders in `fedecs_lt/results/`) |
|---|---|
| Natural-imbalance comparison | the 13 notebooks of folder 3 |
| Ablation | `NB30_ablation_cifar10`, `NB10_main_cifar10_part1of3`, `NB30_ablation_cifar100`, `NB10_main_cifar100_part1of2` |
| One shared model, ratio 100 | the 12 `NB20_track2_*_IF100_*` notebooks of folder 2 (gives `across_track2_*.tex`) |
| One shared model, all ratios | the 12 above + the 11 of folder 4 (columns "CIFAR-10 (ratio 100)", "(ratio 50)", "(ratio 10)", ...) |

For the ablation table take `across_track1_final_bal.tex` / `_er_bal.tex`. Then **delete the rows of methods
that are not ablations or references**: part 1 of the main notebook also brings Balanced Softmax, FedLC, FedRS,
FedDF and Local. Delete whole rows only; never edit a number.

A run that appears in two inputs is taken from the **last** input. The command prints which. NB00's runs are an
earlier code version, so never pass NB00 together with the others.

### Figures and full reports (slow, from the large results zips)

```
python fedecs_lt/merge_results.py paper_build/cifar10_full "CIFAR-10-LT, every method" \
  NB10_main_cifar10_part1of3__results.zip NB10_main_cifar10_part2of3__results.zip NB10_main_cifar10_part3of3__results.zip
```

This writes `merged__report.pdf`: one report with every method and seed of CIFAR-10, every figure redrawn over
all methods, and the paired tests. The figures are in `paper_build/cifar10_full/results/figures/<setting>/` as
`fig_NN_name.pdf` (vector, for LaTeX), `.png` and `.csv` (the plotted data). It can take up to an hour on a laptop.
A notebook that was not split needs no merge: its own `NAME__results.zip` already has every figure in
`results/figures/`.

Put a figure into the paper: copy its PDF to `docs/latex/figures/` (create the folder once; keep the file name) and write

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/fig_39_forest.pdf}   % column figure; use figure* and \textwidth for page-wide ones
\caption{...}   % start from the caption printed under the figure in the report (section 5)
\label{fig:forest}
\end{figure}
```

Figures are drawn at their print size: 3.5 in for one column, 7.16 in for the full page. The report names the
width. Never scale a column figure to the page width; use `figure*` for page-wide figures.

### Pasting a table

```latex
\begin{table}[t]
\centering\small
\caption{Main comparison: test balanced accuracy (\%), heads as trained, mean $\pm$ standard deviation over seeds
3--5; best in bold. 10 clients, Dirichlet label split ($\alpha=0.5$), imbalance ratio 100.}
\label{tab:main}
\input{tables/across_track1_final_bal.tex}
\end{table}
```

Copy the `.tex` files you use into `docs/latex/tables/`. If a table is too wide, add `\setlength{\tabcolsep}{3pt}`
or `\resizebox{\linewidth}{!}{...}` around the `\input`. Better still, move columns to the appendix.

---

## 4. Which report table and figure to use where

Each report (one notebook's, or a merged one) has, per setting, these tables in this order. Roman numbers differ
between reports, so look for the start of the caption:

| Report table (caption starts with) | Use in the paper |
|---|---|
| "Main metrics (%, test set, ..." | main comparison (balanced accuracy column) and the per-dataset appendix tables |
| "The same heads with the exact readout" / "The same model with the exact readout" | readout table |
| "Further metrics" | appendix |
| "Fairness across clients" | appendix; one sentence in the main text (worst architecture, accuracy on each client's own class mix) |
| "Calibration" | appendix; ECE in the main text only if it supports a claim |
| "Balanced accuracy on the validation set" | appendix (shows what selection used) |
| "Cost" | cost / privacy appendix (`app:privacy`): communication per client, time, FLOPs |
| "Balanced accuracy of FedECS-LT minus each method" | the paired table (main text) |
| "Rounds to the target balanced accuracy" | appendix (convergence speed) |
| "Representation and generator" | mechanism discussion: 10-NN accuracy, neural collapse, generator MMD |
| "Every run" | appendix (all seeds), or supplementary material |
| "Per-class ..." | supplementary material only |

Recommended figures for the main text (numbers as in the reports; all others go to the appendix or the
supplement):

| Fig. | File | Shows | Paper section |
|---|---|---|---|
| 39 | `fig_39_forest` | each method's difference to FedECS-LT with 95% CI | main comparison |
| 10 | `fig_10_recall_vs_class_size` | recall against training images per class | where the long-tail loss sits |
| 14 | `fig_14_head_medium_tail` | head / medium / tail recall per method | main comparison |
| 83 | `fig_83_tsne_comparison` | GeFL-F and FedECS-LT embeddings side by side, tail classes emphasised | representation |
| 22 | `fig_22_tsne_real_vs_generated_ours` | real vs generated features | generator |
| 30 / 33 | `fig_30_class_row_norms`, `fig_33_row_norm_rounds_geflf` | rare-class rows shrinking in GeFL-F (the collapse of Sec. 3) | Stage II |
| 44 | `fig_44_metric_heatmap` | every method × every main metric | main comparison (or appendix) |
| 71 | `fig_71_track2_balanced_accuracy_rounds` | one shared model: balanced accuracy over rounds | one-shared-model section |

For t-SNE and UMAP (Fig. 19-23, 83-85) the caption must say that **cluster sizes and distances between clusters
mean nothing**. Only the grouping does. The quantitative claim goes with the 10-NN accuracy in the same caption.
The report's caption already says both; keep them.

---

## 5. Writing rules (what a reviewer checks)

**Names.** Use the methods' published names only: FedECS-LT, GeFL-F, Balanced Softmax, FedLC, FedRS, FedDF, Local,
FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL; FedAvg, FedProx, MOON, CCVR, CReFF, FedETF, Fed-GraB, FedLoGe,
FedYoYo, RUCR. Write ablations as "GeFL-F w/o generator", "GeFL-F + LA", "GeFL-F w/ Gaussian generator + LA",
"FedDF + LA". Never write file names, notebook names, "Track 1/2", "suite", "pipeline", "run", "part" or "seed 3
notebook" in the paper. Say "main setting" and "one-shared-model setting" (or "the FedLTLib protocol").

**Every caption** says what is shown, on which data (test set), for which models (heads as trained / exact
readout), the setting (dataset, imbalance ratio, 10 clients, Dirichlet α = 0.5; one shared model: 20 clients, 40%
per round), the seeds, and what ± or the bars mean (one standard deviation over seeds 3-5). Define abbreviations
at first use in each caption (LA, ECE, MCC, AUC-PR, NC1, CKA, MMD). Start from the report's caption and shorten it.

**Numbers.**
- Never type a number by hand. Paste the `.tex` from `paper_tables.py`, or copy from its `.csv`.
- One decimal for percentages.
- "± SD" only with two or more seeds. Single-seed columns (ratios 50 and 10) get "one seed" in the caption and no
  ± at all.
- When the text quotes a number, copy it from the same `.csv` the table came from.

**Claims.**
- "Better" only where the paired test is bold, i.e. Holm-adjusted p < 0.05. Otherwise write "comparable", or
  "higher on average but not significant".
- Report where FedECS-LT is not the best, as the current paper does ("Ours is not ahead of every baseline"). Keep
  the pre-registered predictions and say which failed.
- Do not claim results across imbalance ratios if folder 4 was not run. Say "at imbalance ratio 100".
- Check every new claim against `docs/review/adversarial_review_dossier.md`.

**Protocol paragraph (update once).**
- The experiments run in Flower 1.42 (simulation), on Kaggle T4 GPUs.
- Seeds 3-5.
- Main setting: 10 clients with different head architectures, a shared frozen extractor, Dirichlet α = 0.5,
  imbalance ratio 100 (natural imbalance for the second dataset group). The model-heterogeneous baselines are the
  HtFLlib implementations (Apache-2.0), ported to Flower, with their authors' hyperparameters.
- One-shared-model setting: the FedLTLib protocol. ResNet-8, 20 clients, 40% per round, 200 rounds, 10 local
  epochs. The baselines run in the authors' LongTailLib code (commit 4141678) on our splits. FedECS-LT trains its
  extractor for 100 rounds and its heads for 100 more; the generator's rounds come on top and are reported in the
  cost table.
- Selection is on 2,000 validation images that no client holds.

**Order of work.**
1. Update `docs/paper_draft.md` with the new numbers and the reasoning.
2. Then update the matching numbers in `docs/latex/paper.tex`.
3. Build with `pdflatex paper.tex` three times in `docs/latex/`.
4. Commit both files.

---

## 6. Checklist before submission

- [ ] Every notebook used in the paper is in `fedecs_lt/results/NAME/` (report + summary zip + summary.md).
- [ ] Every report cover shows 0 failed runs, a logging audit with 0 problems, 0 figure errors and 0 table errors.
- [ ] Every paper table is a pasted `.tex` from `paper_tables.py`; the command that made it is noted in
      `docs/paper_draft.md` next to the table.
- [ ] No table mixes Flower runs with earlier runs; NB00 is not an input of any table.
- [ ] Seeds stated in every caption; single-seed results marked "one seed".
- [ ] "Better" appears only for bold (Holm p < 0.05) differences.
- [ ] The t-SNE / UMAP captions say that cluster sizes and distances are not interpretable.
- [ ] Every method's citation is in the bibliography, and the code origin is in the protocol / appendix (HtFLlib,
      LongTailLib, PFLlib for FedLC, our re-implementations of FedDF, FedRS and Balanced Softmax).
- [ ] Nothing from `docs/review/adversarial_review_dossier.md` came back.
- [ ] `paper.tex` builds without undefined references; page limit (9 + appendix) kept.

---

## 7. If something goes wrong

| Problem | What to do |
|---|---|
| `merge_results.py` says "OUT_DIR/results exists" | choose a new output folder (it never overwrites) |
| A table lacks a method | that method's notebook (or part) is not among the inputs, or its runs failed: see `failures.jsonl` |
| Paired table empty for a dataset | FedECS-LT and the other method share fewer than two seeds there |
| Figures listed as "not applicable" in a merged report | you merged `summary.zip` files; figures need the `results.zip` files |
| A Kaggle notebook stopped at 12 h | the finished runs are saved; run the notebook again on another account and merge both (the later input wins for runs in both) |
| A number in the paper disagrees with a report | the paper is wrong: re-paste the `.tex`; the report and `paper_tables.py` use the same functions |
