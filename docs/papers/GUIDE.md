# The paper guide: from every experiment to the finished paper

This guide is for finishing the FedECS-LT paper on your own. It covers:
- how the two versions of the paper are built;
- what each experiment gives the paper;
- the commands that move a returned notebook's results into the paper;
- how to change the text when a result is better or worse than hoped;
- how to submit to an Elsevier or an IEEE journal, and how to keep the paper free of plagiarism problems.

Read sections 0 to 3 once. Sections 4 and 5 are the reference you will use every time a notebook comes back.

| Section | What |
|---|---|
| 0 | The loop: six commands after every returned notebook |
| 1 | Where everything is |
| 2 | One-time setup (your computer, or Overleaf) |
| 3 | How the paper fills itself: slots |
| 4 | Every experiment: notebooks, what the paper gets, what to write for each outcome |
| 5 | Changing the text when results change |
| 6 | Figures you draw yourself (architecture diagram, graphical abstract) |
| 7 | Building, checking and the page limit |
| 8 | Submitting: Elsevier (single column) or IEEE (two columns) |
| 9 | Originality: plagiarism, similarity checks and the use of AI tools |
| 10 | References |
| 11 | Final checklist |
| 12 | If something goes wrong |
| 13 | Reference: every slot, every number key |

---

## 0. The loop (after every returned notebook)

```
git pull
python fedecs_lt/fetch_kaggle.py https://www.kaggle.com/code/ACCOUNT/NOTEBOOK   # file the results (no login)
python docs/papers/fill.py                    # tables and numbers into the paper (seconds)
docs/papers/build.sh                          # both PDFs (or recompile on Overleaf)
# read docs/papers/generated/STATUS.md, then fix the blue sentences that changed (section 5)
git add -A fedecs_lt/results docs/papers && git commit -m "Results of NOTEBOOK" && git push
```

- Several notebooks at once: give `fetch_kaggle.py` several links.
- Figures: when you also want figures, download the large results with `fetch_kaggle.py --results LINK` and run
  `python docs/papers/fill.py --figures` (section 4, E8).

The rule behind it: **never type a result into the paper.** Every number, table and figure that comes from an
experiment is a *slot* that `fill.py` fills from the returned runs. If a result changes, run the loop again and the
paper changes with it.

---

## 1. Where everything is

| Path | What |
|---|---|
| `docs/papers/content/*.tex` | **the text of the paper, written once**, used by both versions |
| `docs/papers/content/refs.bib` | the references (both versions) |
| `docs/papers/content/macros.tex` | the slot commands (`\R`, `\slottab`, `\slotfig`, `\claim`) |
| `docs/papers/content/declarations.tex` | CRediT, competing interests, funding, data availability, AI use |
| `docs/papers/elsevier/main.tex` | Elsevier single-column wrapper (elsarticle); `highlights.tex` |
| `docs/papers/ieee/main.tex` | IEEE two-column wrapper (IEEEtran journal) |
| `docs/papers/generated/` | written by `fill.py`: `numbers.tex`, `tables/`, `figures/`, `STATUS.md` |
| `docs/papers/fill.py` | fills the slots from `fedecs_lt/results/` |
| `docs/papers/build.sh` | builds both PDFs |
| `fedecs_lt/results/NAME/` | one folder per returned notebook (`fetch_kaggle.py` writes it) |
| `fedecs_lt/notebooks/` | the notebooks, one folder per priority; `START_HERE.md` is the run sheet |
| `fedecs_lt/merge_results.py`, `fedecs_lt/paper_tables.py` | combined reports and tables for your own analyses (section 12) |
| `docs/review/adversarial_review_dossier.md` | claims that reviewers showed to be false or unsupported: never write them |

The two versions share every sentence. **Edit only `content/*.tex`**, never a copy of a section in a wrapper.
The wrappers hold only what differs: the document class, the title block, where the declarations go, and how the
page-wide floats are made. `widefigure` and `widetable` span both columns in IEEE and are normal floats in Elsevier.

The paper is: **FedECS-LT: Exact Class Statistics for Long-Tailed Federated Learning with Heterogeneous Client
Models**. Sections:
1. introduction;
2. related work;
3. problem setting;
4. where generator-based heterogeneous FL loses rare classes (the analysis, proofs in the appendix);
5. FedECS-LT (the method);
6. experimental setup;
7. results;
8. discussion and limitations;
9. conclusion.

The appendix holds the proofs, the experimental details and further tables.

`docs/latex/paper.tex` is the earlier analysis paper, built on the earlier implementation (`experiments/core.py`).
It stays as the record. The new paper uses **only the Flower runs**; section 4.10 says what may be taken over from
the earlier one.

---

## 2. One-time setup

**On your computer.**
1. Python 3.10 or newer: `pip install numpy scipy scikit-learn matplotlib reportlab pyarrow`. Add `umap-learn` for
   the UMAP figure only.
2. A TeX distribution with `pdflatex` and `bibtex`. TeX Live (Linux, macOS) and MiKTeX (Windows) both include the
   `elsarticle` and `IEEEtran` classes.
3. Check: `python -m pytest fedecs_lt/tests/test_paper_fill.py fedecs_lt/tests/test_paper_tables.py -q` must pass.

On Windows, run `build.sh` from Git Bash, or build by hand in `docs/papers/elsevier` and `docs/papers/ieee`:
`pdflatex main`, `bibtex main`, `pdflatex main`, `pdflatex main`.

**On Overleaf** (no TeX installation needed).
1. Run `fill.py` on your computer first.
2. Upload the whole `docs/papers` folder (zip it) as a new project.
3. Set the main document to `elsevier/main.tex` or `ieee/main.tex` (Menu → Main document).
4. After each `fill.py`, upload `docs/papers/generated/` again.

Overleaf has both classes built in.

---

## 3. How the paper fills itself: slots

There are four commands in the text (defined in `content/macros.tex`).

| Command | Shows | Filled from |
|---|---|---|
| `\R{main/cifar10/FedECS-LT/bal}` | a number, e.g. `42.8 ± 2.0` | `generated/numbers.tex` |
| `\slottab{main_bal}{placeholder}` | a whole table body | `generated/tables/main_bal.tex` |
| `\slotfig{fig_tsne}{wide}{what it is}` | a figure at its print size | `generated/figures/fig_tsne.pdf` |
| `\claim{sentence}` | a sentence whose wording depends on the results | you (section 5) |

- **Until a slot is filled**, the PDF shows it in **red**: `[main/mnist/margin]`, a table of `--`, or a framed
  "FIGURE PLACEHOLDER" box that says what belongs there.
- **`\claim{...}` sentences are blue.** Every blue sentence makes a statement about the results. Read it whenever a
  result changes and make sure it is still true (section 5).
- **Final build.** Uncomment `\finalslotstrue` in the wrapper before submitting. Then any red slot left is a LaTeX
  error, and the blue colour disappears.
- **`generated/STATUS.md`** (written by `fill.py`) lists:
  - every experiment group with its settings, methods and seeds, flagged where fewer than three seeds came back;
  - the slots of the paper that are still empty;
  - the notebooks that have not come back yet, by folder.

**How `fill.py` decides what goes where.** It reads every run in `fedecs_lt/results/NB*/runs.jsonl` except NB00's
(an earlier code version). It sorts each run into an experiment by its method and setting, so parts, reruns and the
order of the notebooks do not matter:

| Group | Runs | Paper |
|---|---|---|
| `main` | main setting (10 clients, Dirichlet 0.5, ratio 100) on MNIST, Fashion-MNIST, SVHN, CIFAR-10, CIFAR-100; the 13 main methods | Tables main, paired, readout, tail; appendix tables |
| `nat` | the same setting on the seven datasets with natural imbalance | Table natural |
| `t2` | one shared ResNet-8 (FedLTLib protocol), any ratio | Table shared |
| `abl` | GeFL-F variants (NB30) + FedECS-LT, GeFL-F, FedDF of the same datasets | Table ablation |
| `comp` | FedECS-LT component variants (NB31) + the full FedECS-LT of the same datasets | Table components |
| `sweep` | FedECS-LT, GeFL-F, FedDF, Balanced Softmax with one setting changed (NB40) | Table robustness |

A run counts once. If the same run came back twice, the later folder in alphabetical order wins; `STATUS.md` shows
the seeds per setting.

**Adding a new slot.**
- *A number:* write `\R{group/setting/method/metric}` in the text. Keys are listed in section 13. Run `fill.py`;
  every listed key is written for every method and setting that has runs.
- *A table:* use `\slottab{name}{...}` with a name `fill.py` writes (section 13). For a table that does not exist
  yet, add a few lines to `fill_extras()` in `fill.py`, modelled on the cost table.
- *A figure:* add the slot to the `FIGS` dictionary at the top of `fill.py`'s figure code:
  `"fig_name": (figure number in the reports, dataset key, imbalance factor)`. The figure numbers are those in every
  notebook report (`fig_NN_name.pdf`).

---

## 4. Every experiment

Each entry gives:
- the notebooks (with their folder in `fedecs_lt/notebooks/`);
- what the paper gets (slots, sections);
- the state on 27 Oct 2026;
- what to write for each possible outcome.

Run the loop of section 0 after each notebook. Status codes: done / partial / not run.

### E1. Main comparison, long-tailed benchmarks (folder 1)

- **Notebooks:** `NB10_main_mnist`, `NB10_main_fmnist`, `NB10_main_svhn`, `NB10_main_cifar10_part1of3`–`part3of3`,
  `NB10_main_cifar100_part1of2`–`part2of2`.
- **Paper:**
  - Tables `tab:main` (`main_bal`), `tab:paired` (`main_paired`), `tab:readout` (`main_er`) and `tab:tail`
    (`main_tail`), plus appendix tables `main_cifar10_main`, `main_cifar100_main` and `main_ece`;
  - the numbers in the abstract, introduction (contributions), Section 7.1–7.2, discussion and conclusion;
  - figures `fig_forest`, `fig_hmt` and `fig_recall_size` (E8).
- **State:** CIFAR-10-LT and CIFAR-100-LT are done (13 methods × seeds 3–5); MNIST, Fashion-MNIST and SVHN are not
  run. Until they are, Tables main/paired/readout/tail have two columns and the abstract's CIFAR numbers are filled.
- **What to write.** Use the paired table: a difference is "significant" only when it is bold (Holm p < 0.05).

| Outcome | Wording (Section 7.1, abstract, introduction) |
|---|---|
| FedECS-LT best on every benchmark, bold against every method | "FedECS-LT has the highest balanced accuracy on all N benchmarks, significantly above every compared method (paired t-test, Holm-corrected)." |
| best on all, not bold everywhere | "FedECS-LT has the highest mean balanced accuracy on all N benchmarks; the difference to the strongest other method is significant on X of them." |
| best on some | "FedECS-LT has the highest balanced accuracy on X of N benchmarks; on D it is below S by M points (Table …), which we discuss in Section 8." |
| below a baseline on most | Rewrite the contribution: the paper's claim becomes the analysis (Section 4) and the gain over GeFL-F; say plainly which methods are stronger and where. |

Always keep:
- the comparison with GeFL-F, the method FedECS-LT extends;
- the readout table. A reviewer will ask how much of the gain is the readout. The paper answers with
  `tab:readout` and the numbers `main/<dataset>/er_diff/GeFL-F`.

### E2. Datasets with natural class imbalance (folder 3)

- **Notebooks:**
  - `NB10_main_dermamnist`, `organamnist`, `bloodmnist` and `emnist`, plus `NB10_main_emnist_rerun_7bfd7d` from
    folder 5;
  - `NB10_main_pathmnist_part1of3`–`3of3`, `NB10_main_gtsrb_part1of3`–`3of3` and `NB10_main_fer2013_part1of3`–`3of3`.
- **Paper:** `tab:natural` (`nat_bal`), the numbers `nat/nbest` and `nat/nsettings`, and the gate numbers
  `gate/F/min`, `gate/F/max` and `gate/rows` (Section 5.2).
- **State:**
  - DermaMNIST, OrganAMNIST, BloodMNIST, PathMNIST and FER-2013 are done.
  - EMNIST has 13 of 39 runs; the rerun notebook completes it.
  - GTSRB is not run. All three parts must be (re)run with the fixed download.
- **What to write:** as E1. Also say on which datasets the gate chose class rows (`gate/rows`, from the runs). On
  FER-2013 it chose class rows; that is the gate working as designed, not a failure.

### E3. One shared model, imbalance ratio 100 (folder 2)

- **Notebooks:**
  - `NB20_track2_cifar10_IF100_s3_part1of2`/`part2of2`, the same for `s4` and `s5`;
  - the same six for `cifar100`;
  - and the rerun in folder 5 (`NB20_track2_rerun_*`).
- **Paper:** `tab:shared` (`t2_bal`), Section 7.4, the numbers `t2/cifar10-r100/…` and `t2/cifar100-r100/…`, and
  figure `fig_shared_rounds`.
- **State:** CIFAR-10 seeds 3–4 and CIFAR-100 seed 4 are done. **Current numbers put FedECS-LT well below the best
  long-tail methods here** (e.g. FedYoYo, CCVR, FedLoGe on CIFAR-10-LT). The sentence in Section 7.4 is written as a
  `\claim` and will read badly as it is. Decide how to present it:
  - **Option A (recommended): report it as a limitation of the design.** FedECS-LT is built for clients with
    different models, where a shared frozen extractor and a generator are the only way to transfer knowledge. With
    one shared model, end-to-end training of the whole network wins. Text: "With one shared model, end-to-end
    long-tail methods are stronger: FedECS-LT reaches X against Y for the strongest (Table …). Its frozen extractor,
    trained for half of the rounds, limits the representation; with the extractor trained for all 200 rounds the
    gap changes by Z points." The last part uses `t2/cifar10-r100/fe200_diff`, the diagnosis run. Add a bullet to
    the limitations in Section 8.
  - Option B: move the one-shared-model results to the appendix and say in Section 6.2 that the main claims concern
    clients with different models. Keep the table; removing it would look like hiding a result.
  - Never drop the result or the setting silently.
- Update the abstract sentence about the one-shared-model comparison to match (it currently only says what was
  compared).

### E4. One shared model, ratios 50 and 10 (folder 4)

- **Notebooks:** `NB20_track2_cifar10_IF050_s3_*`, `..._IF010_s3_*`, the same for `cifar100`, and
  `NB29_track2_long_runs_1`.
- **Paper:** extra columns of `tab:shared` (`fill.py` adds them by itself, headed "CIFAR-10 (ratio 50)" etc.).
- **State:** CIFAR-10 ratio 10 parts 1–2 and CIFAR-100 ratio 10 part 1 are done. FedECS-LT failed there and is in
  the rerun of folder 5.
- **What to write:** one seed only. The caption already says "one seed (3) at ratios 50 and 10"; keep it.
- **If folder 4 is not run:** delete "50 and 10" from the caption and Section 6.2, keep ratio 100 only, and say
  "at imbalance ratio 100".

### E5. Variants of GeFL-F (folder 2: `NB30_ablation_cifar10`, `NB30_ablation_cifar100`)

- **Paper:** `tab:ablation` (`abl_bal`) and the numbers `abl/cifar10/GeFL-F + LA/gain` and `abl/cifar10/rest`
  (Section 7.5).
- **State:** not run. The references (FedECS-LT, GeFL-F, FedDF) come from E1's part 1 automatically.
- **What to write:** "Removing the coupled decay and adding logit adjustment (GeFL-F + LA) recovers G points; FedECS-LT
  adds R points on top." If R is small or negative, say so. It means the decay fix is most of the gain on that
  dataset, which the analysis of Section 4.1 predicts.

### E6. Components of FedECS-LT (folder 2: `NB31_components_cifar10`, `NB31_components_cifar100`)

- **Variants**, each changing one component:
  - class rows averaged by CCWA instead of the gate;
  - class rows averaged flat (GeFL-F's generator);
  - without moment correction;
  - without consensus soft labels;
  - without logit adjustment;
  - one synthetic epoch instead of ten.
- **Paper:** `tab:components` (`abl_components`: as trained and with the exact readout), Section 7.5, paragraph
  "Components of FedECS-LT". Replace the bracketed blue sentence there with the findings.
- **State:** not run (built 27 Oct 2026). The full FedECS-LT comes from E1.
- **What to write:** for each component, the drop when it is removed, as trained and with the readout. Components
  that matter only as trained (typically logit adjustment) act on the last layer. Components that matter with the
  readout improve the representation. Name which is which. A component whose removal changes nothing within the seed
  spread is "not needed on this dataset". Say it, and consider removing it from the method in the next revision.

### E7. Robustness sweeps (folder 6: `NB40_sweep_cifar10_a0.1`, `a1.0`, `K50`, `K100`, `IF050`, `IF010`)

- **Setup:** CIFAR-10-LT, one setting changed at a time: Dirichlet α = 0.1 or 1.0, K = 50 or 100 clients, imbalance
  ratio 50 or 10. Methods: FedECS-LT, GeFL-F, FedDF, Balanced Softmax (the strongest others on CIFAR when built).
- **Paper:** `tab:robust` (`robust`), the numbers `sweep/<setting>/<method>/bal` (e.g. `sweep/Dirichlet0.1/FedECS-LT/bal`),
  and the bracketed blue sentence in Section 7.5 "Robustness".
- **State:** not run. K = 100 takes longer than the planner estimates. Runs cut off by the 11-h limit come back as a
  rerun in folder 5 after the loop.
- **What to write:** whether the ranking of Table main holds in every column. "The ranking is the same under stronger
  and weaker label skew and with 50 and 100 clients", or name the column where it changes and give the likely
  reason. For example, with more clients per architecture CCWA averages over more holders.
- **If E1 shows other strongest methods:** change `SWEEP_METHODS` in `fedecs_lt/build_notebooks.py` before running
  E7, then rebuild the notebooks (`python fedecs_lt/build_notebooks.py`).

### E8. Figures (any notebook; needs the large results files)

- **Commands:** `python fedecs_lt/fetch_kaggle.py --results LINK` for the notebooks of a dataset (for CIFAR-10-LT:
  all three parts), then `python docs/papers/fill.py --figures`.
- **Slots:**

| Slot | Report figure | Dataset | Section |
|---|---|---|---|
| `fig_rows` | Fig. 30, class-row norms of the generator | CIFAR-10-LT | 4.1 |
| `fig_spread` | Fig. 75, logit spread in the synthetic phase | CIFAR-10-LT | 4.3 |
| `fig_forest` | Fig. 39, difference to FedECS-LT with 95% CI | CIFAR-10-LT | 7.1 |
| `fig_recall_size` | Fig. 10, recall against class size | CIFAR-100-LT | 7.2 |
| `fig_hmt` | Fig. 14, head / medium / tail recall | CIFAR-10-LT | 7.2 |
| `fig_tsne` | Fig. 83, t-SNE of GeFL-F and FedECS-LT side by side | CIFAR-10-LT | 7.6 |
| `fig_realgen` | Fig. 22, real and generated features | CIFAR-10-LT | 7.6 |
| `fig_shared_rounds` | Fig. 71, balanced accuracy over rounds, one shared model | CIFAR-100-LT ratio 100 | 7.4 |

- **What else this writes:** the figure's own caption from the report, as `generated/figures/<slot>.caption.txt`.
  Compare it with the caption in the paper; the paper's caption must state the same setting and seeds. It also
  writes the 10-NN accuracies `repr/cifar10/<method>/knn` used in Section 7.6.
- **Another dataset:** to show a figure for another dataset, change the dataset in `FIGS` in `fill.py`.
- **t-SNE and UMAP captions:** they must keep the sentence that cluster sizes and distances between clusters are not
  interpretable.

### E9. Reruns (folder 5)

- **What they are:** notebooks named `*_rerun_<id>` hold runs that failed elsewhere.
- **How they are handled:** `fill.py` merges them automatically; nothing to do except run the loop.
- **When new reruns are made:** after every loop, `python fedecs_lt/build_notebooks.py` makes new reruns for
  anything still missing.

### E10. What may be taken from the earlier analysis paper (`docs/latex/paper.tex`)

These carry over unchanged:
- the proofs (already in the appendix);
- the theory of Section 4;
- the qualitative findings.

Never take over its numbers or tables (`tab:fixes`, `tab:central`, `tab:robust` there, `tab:recsys`, `fig_kappa`,
`fig_signal`). They come from the earlier implementation with other seeds. If you want one of them, for example the
recommendation result or the count predictor κ, put it in the appendix under its own heading. Its caption must say
"earlier implementation (`experiments/core.py`), seeds 0–2". It must never be in a table next to Flower results.

---

## 5. Changing the text when results change

1. Run the loop (section 0) and open the PDF. Red slots are still empty, blue sentences are claims.
2. Read every blue sentence. It contains `\R{...}` numbers that are now filled. Ask:
   - Is the sentence still true? "has the highest balanced accuracy on 2 of the 2 benchmarks" is true;
     "on 3 of 5" changes "highest" to "highest on most".
   - Does it claim significance? Check the bold cells of the paired table.
   - Does a number contradict the words? A negative margin with "above" is wrong.
3. Rewrite the sentence in `content/*.tex`. Keep the `\R{...}` numbers inside it, and keep `\claim{...}` around it.
4. Where the outcome changes the story, change it everywhere it is told:
   - **the abstract**, `content/abstract.tex` (≤ 250 words);
   - **the introduction**: contributions, last item;
   - **Section 7**: the paragraph of the experiment;
   - **Section 8**: discussion and limitations;
   - **the conclusion**;
   - **the highlights**, `elsevier/highlights.tex`: ≤ 85 characters each,
     `python docs/papers/fill.py --check-highlights`.
5. Rebuild and read the changed pages.

**Sentence patterns** (from the writing guidance in section 9):

| You want to say | Write | Not |
|---|---|---|
| a significant gain | "X points higher (95% CI [a, b]; Holm-corrected p < 0.05)" | "significantly better" without the test |
| a gain that is not significant | "higher on average (X points), but not significantly so with three seeds" | "better", "outperforms" |
| a loss | "X points below S" | silence |
| a mechanism you measured | "the rare rows shrink to … (Fig. …)" | "obviously", "clearly" |
| a mechanism you believe | "we expect …, because …" (keep it apart from results) | presenting it as a result |

Words to avoid in the paper: "suite", "track", "notebook", "pipeline", "run" (say "experiment" or "training"), file
names, "state of the art", "novel" (say what is new instead), "significantly" without a test, "huge", "dramatic".
Use the methods' published names only (section 13).

---

## 6. Figures you draw yourself

**Architecture diagram (`fig_overview`, Section 5, page-wide).** Draw it in draw.io / diagrams.net, PowerPoint,
Inkscape or TikZ, and export a **PDF**. Save it as `docs/papers/generated/figures/fig_overview.pdf`; `fill.py` never
deletes figures.

- **Size:** 7.16 in wide (IEEE page width), at most about 3 in high.
- **Text:** Times-like font, at least 8 pt at that size.
- **Content, left to right:**
  1. clients with different head architectures (boxes CNN-1 … CNN-10);
  2. Stage I, the shared frozen extractor;
  3. the statistics release: counts, sums, squared norms, through a "secure aggregation" box to the server;
  4. the server: gate F, anchored generator or class rows with CCWA, moment correction;
  5. Stage III on the clients: synthetic phase with consensus soft labels, real phase with logit adjustment;
  6. the exact readout: embedding sums, again through secure aggregation.
- **Arrows:** label each one with what is sent (statistics, generator, heads, consensus predictions, readout sums).
- **Style:** one colour per stage; do not use colour alone to distinguish things (the paper may be printed in grey).

**Graphical abstract (Elsevier, optional).** One image, at least 531 × 1328 pixels (height × width), readable at
5 × 13 cm. A simplified version of the overview with one result: the tail recall of GeFL-F against FedECS-LT.
Upload it as a separate file; it is not in the PDF.

---

## 7. Building, checking and the page limit

- `docs/papers/build.sh` builds `elsevier/main.pdf`, `ieee/main.pdf` and `elsevier/highlights.pdf`. On failure
  it names the log; search the log for `!`.
- **Before submission:**
  - uncomment `\finalslotstrue` in the wrapper and build: every unfilled slot is an error;
  - for Elsevier, switch to `\documentclass[review,12pt]{elsarticle}` and uncomment `\linenumbers`.
- **IEEE TNNLS length:** 10 pages are free, each further page costs about USD 200, and 15 pages is the maximum.
  The appendix can go to a separate **supplementary file**, which does not count. To make it:
  1. copy `ieee/main.tex` to `ieee/supplement.tex`;
  2. keep only `\input{../content/appendix.tex}` in it;
  3. delete the appendix from `main.tex`;
  4. refer to it as "the supplementary material".

  If the main text is still too long, move per-dataset tables and the robustness table to the supplement.
- **Elsevier (Neural Networks)** has no page limit, but reviewers prefer about 25–35 pages in review format.

---

## 8. Submitting

Submit to **one journal at a time**. Both versions are the same paper. Submitting both at once is duplicate
submission, which every publisher forbids. Use the other version only if the first journal rejects the paper.

### Elsevier, single column (`elsevier/main.tex`)

- **Target:** *Neural Networks*. Alternatives with the same template: *Knowledge-Based Systems*, *Information
  Sciences*, *Pattern Recognition*, *Expert Systems with Applications*.
  - Change `\journal{...}` in `main.tex`.
  - Check the journal's own Guide for Authors; limits differ slightly.
- **Abstract:** at most 250 words, no citations.
- **Highlights:** 3–5 bullets, each at most 85 characters including spaces, as a separate editable file
  (`highlights.tex`, or paste the bullets into a Word file) with "highlights" in the file name.
- **Keywords:** in `main.tex` (1–7).
- **Required statements** (all in `main.tex`, texts in `content/declarations.tex`):
  - CRediT author contributions;
  - declaration of competing interest;
  - funding;
  - data and code availability;
  - **Declaration of generative AI and AI-assisted technologies in the writing process**: a section before the
    references, saying which tool was used for what. Basic grammar or spelling checks need no declaration.
- **Files to upload:** the PDF (or the LaTeX source as a zip, with `generated/` and `content/`), highlights, the
  graphical abstract (optional), a cover letter, and suggested reviewers if asked.
- **Review:** check whether the journal is double-anonymised. If so, remove the author names and acknowledgments
  from the reviewed version, and avoid "our earlier work [x]" phrasing.

### IEEE, two columns (`ieee/main.tex`)

- **Target:** *IEEE Transactions on Neural Networks and Learning Systems* (TNNLS). Alternatives: IEEE TKDE, IEEE TMC
  (the journal GeFL appeared in), IEEE TPAMI (very selective).
- **Format:** IEEEtran journal style, single-spaced, double column, PDF.
- **Length:** 10 pages free, at most 15 with over-length charges (section 7).
- **Abstract:** one paragraph, about 150–250 words. **Index terms** are in `main.tex`.
- **AI use:** IEEE asks for AI-generated content (text, figures, code) to be disclosed in the **Acknowledgments**,
  naming the tool and the parts it was used for. `\AIText` is placed there.
- **Biographies:** with a photo, for the final version; templates are at the end of `main.tex`.
- **Before submitting:** check the TNNLS Information for Authors page; page charges change.

**The cover letter** (both journals): one page. It covers:
- the problem;
- what is new: the three losses and the exact-statistics repairs;
- the evidence (12 datasets, two protocols, 21 published methods, three seeds with paired tests);
- that the paper is not under review elsewhere;
- any related earlier work of yours (for example a preprint or the earlier analysis), with how this paper differs.

---

## 9. Originality: plagiarism, similarity checks and AI tools

**How it is checked.** Journals run every submission through a similarity checker: Crossref Similarity Check, which
is iThenticate, at both Elsevier and IEEE. The report lists overlapping passages per source. Editors look at the
**largest single sources**, not only the total percentage. A large overlap with one paper, especially in the
methods or results, leads to rejection unless it is explained.

**What to do.**
1. **Write every sentence yourself, or rewrite it until it is yours.**
   - The text in `content/` is a draft: read every paragraph and change what you would say differently.
   - Text from the earlier analysis paper (`docs/latex/paper.tex`) was reused in Sections 3–4 and the appendix.
     If that paper is or will be public (arXiv, a thesis, a workshop), cite it, rewrite the shared passages, and
     mention it in the cover letter.
2. **Cite at the point of use.** Every method, dataset, metric and theorem from elsewhere has a citation where it
   is used (section 10). Definitions copied from a paper (e.g. of a metric) are written in your own words, or quoted
   with the citation.
3. **Run a similarity check yourself before submitting**, if your institute has iThenticate or Turnitin (most
   universities do). Rewrite passages with long matches to a single source. Standard phrases (dataset names, "we
   follow the protocol of …") are fine.
4. **Figures and tables** are your own (generated from your runs). If you adapt someone else's figure, say
   "adapted from [x]" and check its licence.

**AI tools.** Both publishers allow AI tools for writing and code, and both require **disclosure**:
- Elsevier: the declaration section.
- IEEE: the acknowledgments.

The authors stay responsible for every sentence and every number. AI tools cannot be authors.

The honest way through is disclosure, not hiding. Rewriting text with "AI humaniser" or paraphrasing tools to get
past AI detectors is not a solution:
- the detectors are unreliable, and editors know it;
- such tools introduce errors and garbled technical terms;
- hiding AI use breaks the journals' policies, which is a reason for rejection or retraction.

What protects you is what is already set up here:
- the results come from your own experiments;
- every number is generated, not typed;
- every claim is checked against a test;
- you rewrite the text in your own words;
- the AI use is declared accurately: say what was done, e.g. "to write and test experiment code and to draft
  passages, which the authors revised".

---

## 10. References

- All references are in `content/refs.bib`. Authors, titles, venues and years were checked against the publishers'
  or authors' pages on 27 Oct 2026.
- Page numbers were left out where they could not be confirmed. Before submission, add missing pages and DOIs from
  each paper's official page (search the title on the publisher's site or on dblp.org), and check every arXiv entry
  for a published version.
- To add a reference: copy its BibTeX from dblp.org or the publisher (not from a search engine summary), check
  authors and venue against the paper itself, add it to `refs.bib`, and cite it with `\cite{key}`.
- **Never cite a paper you have not opened.**

---

## 11. Final checklist

- [ ] Every notebook the paper uses is in `fedecs_lt/results/` and its `summary.md` says "complete" (or the missing
      runs came back in a rerun).
- [ ] `fill.py` run after the last result; `STATUS.md` shows no empty slot in the paper, and every setting has seeds
      3–5 (or the caption says "one seed").
- [ ] `fill.py --figures` run; every figure caption matches its `generated/figures/*.caption.txt` in setting and seeds.
- [ ] The architecture diagram is drawn (section 6).
- [ ] Every blue sentence re-read and true; abstract, introduction, results, discussion, conclusion and highlights
      tell the same story.
- [ ] "Significant" only where the paired table is bold.
- [ ] Where FedECS-LT is not the best, the paper says so (Sections 7 and 8).
- [ ] No claim from `docs/review/adversarial_review_dossier.md` came back. In particular: no "ahead of every method",
      no "first to use statistics in a generator", and no privacy claim beyond "the server sees sums".
- [ ] No result from the earlier implementation next to Flower results.
- [ ] Author block, CRediT, funding, competing interests, data availability and AI declaration filled in
      (`content/declarations.tex`, wrappers).
- [ ] References complete (pages / DOIs), every one opened at least once.
- [ ] Similarity check run; long single-source matches rewritten.
- [ ] `\finalslotstrue` build passes with no error and no undefined reference.
- [ ] IEEE: ≤ 10 pages (or accept the charges, ≤ 15), supplement separate. Elsevier: review format, line numbers,
      highlights within limits.
- [ ] Cover letter written; submitted to one journal only.

---

## 12. If something goes wrong

| Problem | What to do |
|---|---|
| A red slot stays red after `fill.py` | `STATUS.md` lists it under "still empty"; its experiment (section 4) has not come back, or has no run of that method |
| A table has a column with only "-" | that setting has runs of other methods only; check `STATUS.md` |
| A notebook's `summary.md` says INCOMPLETE | the failed runs are listed with their cause; `python fedecs_lt/build_notebooks.py` makes a rerun notebook in folder 5 |
| `fetch_kaggle.py` says "no NAME__summary.zip" | the notebook is still running, was not saved as a version, or is private (share it, or make it public) |
| A number in the text looks wrong | do not edit `generated/`; check the runs in `fedecs_lt/results/NAME/runs.csv` and the notebook's report; then run `fill.py` again |
| You need a table the paper does not have | build it with `python fedecs_lt/merge_results.py --no-report OUT "title" fedecs_lt/results/NB...` (tables in `OUT/tables`, readable in `ALL.md`), then add it as a slot (section 3) |
| You need a combined report with all figures of one dataset | `python fedecs_lt/merge_results.py OUT "title" A__results.zip B__results.zip ...` (up to an hour) |
| Build error "Result slot … is not filled" | you are building with `\finalslotstrue`; fill the slot or remove it from the text |
| Overleaf shows old numbers | upload `docs/papers/generated/` again after `fill.py` |

---

## 13. Reference: slots and number keys

**Number keys** (`\R{key}`), written for every method and setting that has runs:

| Key | Value |
|---|---|
| `G/S/M/bal`, `er_bal`, `f1`, `tail`, `worst`, `mcc`, `gmean`, `ece`, `worst_arch`, `er_tail` | mean ± SD over seeds (%), e.g. `main/cifar10/FedECS-LT/bal` |
| `G/S/M/gain_er` | exact readout minus as trained (points) |
| `G/S/diff/M`, `G/S/ci/M`, `G/S/p/M` | FedECS-LT minus M: mean paired difference, 95% CI, Holm p (as trained) |
| `G/S/er_diff/M`, `er_ci`, `er_p` | the same with the exact readout |
| `G/S/best`, `G/S/strongest`, `G/S/margin` | best method; strongest other method (Local excluded); FedECS-LT minus it |
| `G/S/seeds` | number of seeds |
| `G/nbest`, `G/nsettings` | settings where FedECS-LT is best; settings with FedECS-LT |
| `gate/F/min`, `gate/F/max`, `gate/rows` | class separation F over the datasets; where the gate chose class rows |
| `count/methods`, `count/datasets` | published methods compared; datasets with FedECS-LT results |
| `abl/S/GeFL-F + LA/gain`, `abl/S/rest` | GeFL-F + LA minus GeFL-F; FedECS-LT minus GeFL-F + LA |
| `t2/S/fe200_diff` | FedECS-LT with 200 extractor rounds minus FedECS-LT |
| `cost/S/time_ratio`, `cost/S/comm_ratio` | FedECS-LT over GeFL-F |
| `sweep/T/M/bal` | robustness, T = `Dirichlet0.1`, `Dirichlet1.0`, `K=50`, `K=100`, `ratio50`, `ratio10` |
| `repr/cifar10/M/knn` | 10-NN accuracy of the test embeddings (with `--figures`) |

- **G** is the group: `main`, `nat`, `t2`.
- **S** is the setting: the dataset key (`mnist`, `fmnist`, `svhn`, `cifar10`, `cifar100`, `dermamnist`,
  `organamnist`, `bloodmnist`, `pathmnist`, `gtsrb`, `emnist`, `fer2013`), or for `t2` e.g. `cifar10-r100`.
- **M** is the method's published name: FedECS-LT, GeFL-F, Balanced Softmax, FedLC, FedRS, FedDF, Local, FedProto,
  FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL; FedAvg, FedProx, MOON, CCVR, CReFF, FedETF, Fed-GraB, FedLoGe, FedYoYo,
  RUCR; "FedECS-LT (200 extractor rounds)"; ablations as in Table ablation.

**Table slots:**
- per group: `G_bal`, `G_er`, `G_f1`, `G_tail`, `G_worst`, `G_ece`, `G_worst_arch`, `G_paired`, and `G_S_main`
  for every setting (G = `main`, `nat`, `t2`);
- other tables: `abl_bal`, `abl_er`, `abl_components`, `robust`, `cost`, `datasets`.

Each is also written as `.csv` next to the `.tex`.

**Figure slots:** `fig_rows`, `fig_spread`, `fig_forest`, `fig_recall_size`, `fig_hmt`, `fig_tsne`, `fig_realgen` and
`fig_shared_rounds` come from `fill.py --figures`; `fig_overview` and `graphical_abstract` you draw yourself
(section 6).
