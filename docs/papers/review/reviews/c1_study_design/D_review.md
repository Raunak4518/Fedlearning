# Agent D, cycle 1: review of the Study design (sec 6), status of earlier D findings, and fill.py changes

Manuscript v02 (git 248170e). `sec6_setup.tex` is the same in v01 (87b55d6) and v02. I compared the text before
(`review/manuscript_v00_original/manuscript_v00.txt`) and after (`pdftotext docs/papers/elsevier/main.pdf`), the
code in `fedecs_lt/src` and `docs/papers/fill.py`, and the slot values in `generated/numbers.tex`. The review was
read-only; recomputations are those of cycle 0 (scripts in
`/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/agentD/`).

**Note.** The uncommitted working tree contains further changes to fill.py and STATUS.md that are not in 248170e.
This review covers the committed code only.

**Verified as correct against the code:**
- Head optimiser: SGD 0.1, no momentum (`local_train.py`).
- FedRS α = 0.5; FedLC τ·(n+1)^(−1/4); add-one smoothed prior for logit adjustment and Balanced Softmax
  (`fedecs.py`).
- FedProx μ = 0.01 (`grouped_server.py`); FedDF 20 server SGD steps of 128 (`server.feddf_distill`).
- Validation hold-out of min(200, ⌊2000/C⌋) per class, capped at a quarter of a class (`data.make_partition`).
- Natural-imbalance datasets relabelled by frequency.
- Worst-class definition min_c of the recall averaged over models (`metrics.summarize`).
- Geometric mean computed per model and then averaged.
- Secure sum counted as 8 bytes per value per client, upload only (`server.ssum`).
- Splits identical across methods.
- Repeat SDs: 1.2 points as trained over 39 configurations and 0.6 with the exact readout over 18 configurations
  match my cycle-0 values (1.23 and 0.58).

---

## 1. Study design (sec 6): findings

**D-21 (P1, high confidence). The tuning disclosure contradicts itself, and one sentence about the data is false.**
- Location: sec 6.5 "Settings and tuning" and sec 6.10 "Tuning".
- Problem:
  - sec 6.5 opens with "No method was tuned on our data". Three sentences later it says that FedECS-LT's settings
    were fixed "on earlier runs of the same benchmarks with other seeds (0–2)".
  - sec 6.10 says "The test splits of the reported runs are new". They are not new: every result is on the
    official test set (sec 6.3), the same set that was used during development. Seeds 3–5 change only the
    training subsample and the client partition.
- Why it matters: the reader is told twice that there was no selection on the evaluation data. The development
  history is exactly the asymmetry that dossier F5 and cycle-0 D-8 raise.
- Proposed action:
  - in sec 6.5, write "No baseline was tuned; the settings of FedECS-LT were chosen on earlier runs of the same
    benchmarks (seeds 0–2, same official test sets)";
  - in sec 6.10, replace "test splits … are new" with "the training subsamples and client partitions are new; the
    test sets are the same".
- Verification: neither sentence remains, and both sections say "same test sets".

**D-22 (P2, high confidence). The research-question table points to the wrong sections and does not say what is
still pending.**
- Location: Tab. 2 (`tab:rq`).
- Problem:
  - RQ5 lists "sweeps of K, α, ρ" under Sections 7.3 and 7.4. The sweeps are reported in sec 7.5
    (Robustness, `tab:robust`).
  - RQ4 is unanswered: NB30 and NB31 have not returned and sec 7.5 contains only placeholders.
  - RQ1's causal part (removing the coupled decay) depends on NB30. The current RQ1 evidence is observational
    (row norms, readout gain, logit spread on CIFAR-10 only).
  - In the draft build, RQ2 shows "; 50 clients" because draft mode prints `\whentab` content without data.
- Why it matters: a reviewer checks each RQ against an experiment.
- Proposed action: add `sec:res-ablation` to RQ5. Either mark RQ4 and the sweep part of RQ5 as pending until the
  notebooks return, or keep them out of the submitted version.
- Verification: every RQ row points to a section that contains its result in the final build.

**D-23 (P2, high confidence). The fairness table is incomplete, and one row is wrong.**
- Location: Tab. `tab:fairness`.
- Problem:
  - The **Local** row says "0 / 5". Local trains its own extractor and head jointly for 5·(T_FE+T_TN) epochs
    (750 on CIFAR, `server.run_local`), with augmentation. It does not use the frozen shared extractor.
  - There are no rows for the grouped baselines (FedAvg, FedProx, FedALA, AvgKD, CCVR, FedETF), although
    sec 6.5 lists them as a method group.
  - Two columns that a fairness table needs are missing:
    - **total rounds**: 350 for FedECS-LT, GeFL-F and FedDF; 150 for the methods without a generator;
    - **communication per client**: 11.3 GB for FedECS-LT against 61 MB for Balanced Softmax, FedLC and FedRS,
      and 17–128 MB for the HtFLlib methods (`tab:cost`).
  - FedECS-LT's exchanges are listed as "consensus labels; heads". The secure-sum releases (class statistics,
    readout sums) are missing.
  - The HtFLlib optimiser (SGD 0.01, batch 10) differs from the head stage (SGD 0.1, batch 64/128). This is stated
    in the text but not in the table.
- Why it matters: the table is the paper's answer to cycle-0 D-6. As it stands, it understates how unequal the
  budgets are.
- Proposed action: fix the Local row; add the grouped rows; add a rounds column and a communication column (from
  `tab:cost`); add "statistics, readout sums" to FedECS-LT's exchanges.
- Verification: the table rows match `runner.PIPELINE`, `server.run_local` and `baselines.GROUPED`.

**D-24 (P2, high confidence). The "below about one point is unresolved" rule is not derived from the repeat, and it
is too low for differences as trained.**
- Location: sec 6.7 (repeat slots) and sec 6.9.
- Evidence: `repeat/bal/sd` is the SD over 39 configurations of d = run₂ − run₁. Per-run noise is therefore
  σ_run = SD_d/√2 = 0.87 points as trained and 0.41 with the exact readout. A paired method difference averaged
  over n = 3 seeds has a run-noise SD of √2·σ_run/√3 = SD_d/√3, which gives:

  | Evaluation | SD_d (repeat) | Run-noise SD of a 3-seed paired difference | Noise-only band, ±1.96 SD (normal) | Same band with a t quantile, df = 2 |
  |---|---|---|---|---|
  | As trained | 1.2 | 0.71 | ±1.4 | ±3.1 |
  | Exact readout | 0.6 | 0.34 | ±0.7 | ±1.4 |

- The noise floor as trained is therefore about 1.4 points, not 1.
- With the exact readout it is about 0.7 points. The natural-dataset readout margins (0.8–0.9) lie just above it,
  so "within the run-to-run variation" (sec 7.3, now in the text) overstates the case. "Of the same order as" is
  accurate.
- Four further limits:
  - The SD comes from one dataset (BloodMNIST, accuracy 60–87 %), where noise may differ from CIFAR-100
    (accuracy about 14 %).
  - It is heterogeneous across methods: FedDF up to 3.6 points, FedECS-LT 0.1–2.0.
  - The repeat also differed in scheduling and memory code (`_repeats/README.md`).
  - "Treat as unresolved" is applied after the results are known. It is not a test.
- Why it matters: the rule is used to set aside the natural-dataset readout margins. A reviewer will ask where the
  number 1 comes from.
- Proposed action:
  - state the derivation and give the per-run SD (σ_run = SD_d/√2), then a noise floor for each evaluation (about
    1.4 points as trained, 0.7 with the readout);
  - say that the floor comes from one dataset;
  - prefer "differences within the noise floor are not interpreted" over a fixed one-point rule;
  - in sec 7.3, replace "within the run-to-run variation" with "of the same order as".
  - Optionally, add a slot `repeat/bal/floor` = 1.96·SD_d/√3.
- Verification: the sec 6.9 text and the sec 7.3 sentence agree with the computed floor.

**D-25 (P2, high confidence). The statistics paragraph hard-codes the Holm family sizes and omits facts about
small-sample inference.**
- Location: sec 6.9.
- Problem:
  - "twelve methods as trained" is correct for the main and natural-imbalance tables today. It is 11 for
    one shared model, and it becomes 18 for the main group once the grouped baselines return. The readout family
    stays at 5, because the grouped methods have no readout and the pooled reference is excluded (correct).
  - The paragraph does not state that with n = 3 a distribution-free test cannot reach p < 0.05: the minimum exact
    two-sided p is 0.25. The t-test's normality assumption cannot be checked.
  - It does not state that the CIFAR-10 verdict depends on the family: Holm p = 0.066 over 12 methods, 0.050 over
    the 5 GeFL-F-pipeline methods.
  - "Little power" is not quantified. At α = 0.05/12 and n = 3, power is about 0.12 for the CIFAR-10 margin as
    trained and 0.04–0.19 for the readout margins.
  - "Every seed favours FedECS-LT" (sec 7.1) is not evidence at n = 3, and the paragraph should say so.
- Proposed action: write the family size from a slot (for example `main/n_family`, `er/n_family`); add one sentence
  each on the permutation floor, on the family sensitivity of the CIFAR-10 verdict and on power.
- Verification: the family sizes come from fill.py.

**D-26 (P2, medium confidence). The ablation does not test whether the baselines would gain from T_s = 10.**
- Location: sec 6.8 ("one synthetic epoch … separates the effect of the longer synthetic phase from that of the
  exact statistics") and threats "Compute".
- Problem: FedECS-LT with T_s = 1 shows how much T_s matters for FedECS-LT. It does not show whether GeFL-F or FedDF
  would gain from T_s = 10, and the effects can interact. The fairness question is the second one.
- Proposed action: add GeFL-F (T_s = 10) and FedDF (T_s = 10) to NB30 (about 12 runs). If not, soften the sentence
  to "measures how much FedECS-LT depends on the longer synthetic phase".
- Verification: an arm with T_s = 10 and no exact statistics exists, or the wording is changed.

**D-27 (P2, high confidence). The threats to validity are incomplete.**
- Location: sec 6.10.
- Missing items:
  1. **Selective failures**: in the one-shared-model setting only FedECS-LT failed (CIFAR-100 ρ = 100 seed 3; both
     ρ = 10 settings). The data are not missing at random. "Run at fewer seeds than planned" does not say this.
  2. **Server-held labelled data**: the validation block is used by the server to set the secure-aggregation
     bounds and to train the referee. That is an assumption of the system model, and it is not in the threats.
  3. **Timing**: wall-clock time includes GPU sharing. Stages loaded from the cache are not timed, and waiting for
     another run is (cycle-0 D-11). sec 6.6 mentions only the sharing. The cost-table caption still says "one run
     on one T4 GPU", which contradicts sec 6.6.
  4. **Holm family choice**: see D-25.
  5. **Participation not paired** in the one-shared-model setting: this is stated in sec 6.4, but not among the
     threats.
  6. **Fed-GraB**: only its accuracy is valid; its other metrics are dropped.
  7. **One-seed and mixed-seed cells**: EMNIST, where FedMRL and LG-FedAvg ran on another seed.
- Proposed action: add items 1–3 at least; fix the cost caption.
- Verification: sec 6.10 contains them, and the `tab:cost` caption matches sec 6.6.

**D-28 (P3, high confidence). Smaller accuracy points.**
- "The draw is repeated until every client holds at least ten images": the code tries up to 100 redraws, then moves
  images from the largest client to the smallest (`data.make_partition`). Add "then images are moved from the
  largest client".
- RQ3 wording "only the heads' bodies differ" is now hedged in sec 8, which is fine. RQ3 itself should keep "for
  the GeFL-F-pipeline methods".
- sec 7.1, own mix: "below 8 of the 11 other federated methods …; Balanced Softmax, …, Local and FedMRL are above
  it". Local is not federated; write "methods" or drop Local from the list.
- The Fig. `fig:shared-rounds` caption says "mean over seeds 3–5" for CIFAR-100-LT, where FedECS-LT has seed 4
  only.

**What is good.** The six method groups, the statement that the loss-only methods exchange nothing at K = 10, the
tuning sources, the run-to-run paragraph and the threats on ports and compute answer most of cycle-0 D-6, D-7, D-8
and D-10.

---

## 2. Status of earlier D findings (v00 → v02)

| Finding | Before (v00) | After (v02 PDF) | Verdict |
|---|---|---|---|
| D-3 tail concentration | "gain concentrated in the tail" (CIFAR-100 example) | CIFAR-10 tail +17.5 vs head +3.0; CIFAR-100 spread (+7.0 / +8.7 / +4.7); against FedDF tail +20.8, head −8.1; readout tail gain over BSM +0.0 / −0.1; the conclusion limits the tail claim to CIFAR-10. Slot values match my recomputation | **PASS** |
| D-4 published numbers | CReFF and Fed-GraB only | FedLC (69.0) added; 300 vs 200 rounds stated | **PASS** (residual: CIFAR-100-LT comparison and its realised ρ of 120 not discussed; FedLoGe −3.8 is at the edge of "about three") |
| D-5 "end-to-end LT methods stronger" | general statement plus an untested representation story | above_list "FedYoYo, FedLoGe, CCVR and FedETF" / "FedYoYo, FedLoGe and CCVR"; 7 and 8 methods below; explanation marked "not tested"; abstract, intro and conclusion say "the strongest long-tail methods" | **PASS** |
| D-9 significance wording | "not significant" without nuance | readout: strongest not significant after Holm, CIFAR-100 uncorrected CI excludes 0, GeFL-F and FedDF still significant | **PARTIAL**: wording fixed. Family sensitivity (0.066 vs 0.050), the permutation floor and power are not stated, and no extra seeds were run (D-25) |
| D-12 natural imbalance | "highest on all six" | "highest mean …; significant after Holm on two of the five datasets with three seeds" (Derma 0.036, Organ 0.023; Blood, Path, FER not significant); sec 6.3 notes the small realised ratios | **PASS** (minor: "within the run-to-run variation" overstated, D-24) |
| D-13 "three seeds everywhere" | intro and sec 6 universal | intro "(fewer where a table says so)"; sec 6.6 "settings that came back with fewer seeds say so"; Tab. 10 caption "seeds that came back" | **PASS** (Fig. shared-rounds caption still says seeds 3–5, D-28) |
| D-14 fill.py drops folder-7 and pooled runs | dropped | grouped runs (K = 10, tier-1) and the pooled reference go to `main`; K = 50 to `k50`; references excluded from `paired` | **PARTIAL**: the code paths now exist (§3). The Holm family size is hard-coded in the text, the methods table is gated on `k50_bal` instead of the 10-client grouped results, and the fairness table has no grouped rows |
| D-16 10-NN on one architecture | 30.9 vs 26.0, architecture 0 | 40.9 vs 37.7 (GeFL-F) and 38.9 (FedDF), mean over 10 architectures; 29/30 and 24/30 cells. My all-architecture recomputation gives 40.9 / 37.7 / 29 of 30 | **PASS** |
| D-19 readout in one shared model | not reported | −3.0 (CIFAR-10-LT) and −0.1 (CIFAR-100-LT) reported | **PASS** |
| D-20 own-mix count | intersection list only | "below 8 of the 11 other federated methods (CIFAR-10-LT), 5 of 11 (CIFAR-100-LT)"; counts verified | **PASS** (wording on Local, D-28) |
| "D-22" | – | – | My cycle-0 report has no D-22. If it means claim row #22 (the representation explanation in the one-shared-model setting), it is now marked "not tested": **PASS** |
| "D-27, part" | – | – | No such ID in cycle 0. If it means claim row #27 (cost "mainly through the longer synthetic phase"), sec 8 now separates time (Stage III) from communication (generator): **PASS**. Cycle-0 D-11 is not fixed: the cost caption says "one run on one T4 GPU" and the cache effects are undisclosed (D-27 item 3 above) |

---

## 3. fill.py changes (committed 87b55d6 and 248170e)

**`group_of`**
- Grouped methods and references on tier-1 datasets join `main` only if `main_setting(r)` holds, which means K = 10.
  That is correct.
- The REFS label must match `runner.PIPELINE["Centralised (pooled data)"]` exactly; it does.
- No other bug found.
- **D-29 (P3).** Two downstream fragilities:
  - Once grouped runs exist, the main tables, the Holm family (12 → 18) and "strongest" include them. That is
    intended, but the sec 6.9 text hard-codes "twelve" (D-25).
  - The datasets table takes the first run of each setting. A pooled run (every client holds all data) would give
    wrong client counts. Today this is safe only because `sorted(glob)` puts "NB…" before lower-case folder names.
    Skip REFS there: `next(r for r in E.runs if setting_key(r) == S and r["label"] not in REFS)`.

**`k50_setting`**
- Correct: track 1, K = 50, α 0.5, IF 0.01, tier-1.
- K = 50 runs of the four sweep methods go to both `sweep` and `k50`. This is intended, and they pair with the
  folder-7 runs because the splits and seeds are the same.
- No bug found.

**`repeat_numbers`**
- Correct, and the definition matches the text.
- Minor points:
  - it pools all methods, including Local, into one SD (state "over 39 configurations, 13 methods");
  - it reads the raw labels without `correct_fedgrab`, which does not matter for BloodMNIST;
  - the `break` after the first repeat silently ignores later repeats. Pool them, or emit per-dataset slots.
- Add a slot for the noise floor (D-24).

**`paired` exclusion**
- `m in (OURS, DIAG) or m in REFS` is correct.
- `paired_table` skips the excluded methods because they are absent from `P`.
- `matched` already excluded both.

**New slots**
- `head_/med_/tail_/ertail_` diffs: computed with `paired()` on the common seeds. The values match my independent
  recomputation (for example CIFAR-100 vs GeFL-F: +7.0 / +8.7 / +4.7; readout tail vs BSM: +0.0 / −0.1).
- `above_list`, `n_above`, `n_below`: based on `matched` without Local and REFS. They count Fed-GraB and non-LT
  methods (FedAvg, MOON, FedProx) among "below", which fits the text ("methods are below it").
- `local_n_above`, `local_n_other`: correct, and they exclude Local, so the text should not then name Local among
  "federated methods" (D-28).
- **`n_sig` (P3 bug).** It compares `float(nums[".../strongest_p"]) < 0.05`, but that string is rounded to two
  significant figures. A p of 0.0496 is stored as "0.05" and would count as not significant. Compare the unrounded
  Holm p: keep it in a dictionary, or recompute `paired(...)[st][4]`.
- `repr` 10-NN:
  - all architectures × seeds, values reproduced (40.9 / 37.7, 29/30);
  - `cells` is filled from `E.of(m)` over every setting in the CIFAR-10 cache, which today holds only the main
    setting;
  - if NB10 zips with other settings were ever merged (for example `tail_perm`), restrict to `S`.

---

## 4. Summary of new findings

| ID | Severity | Topic |
|---|---|---|
| D-21 | P1 | "no method tuned on our data" contradiction; "test splits are new" false |
| D-22 | P2 | RQ table references; RQ4 and part of RQ5 pending |
| D-23 | P2 | fairness table: Local row wrong; grouped rows, rounds, communication and statistics missing |
| D-24 | P2 | one-point rule vs the repeat-derived floor (about 1.4 as trained, about 0.7 with the readout) |
| D-25 | P2 | Holm family size hard-coded; permutation floor, family sensitivity and power not stated |
| D-26 | P2 | the T_s = 1 ablation does not answer the matched-compute question |
| D-27 | P2 | missing threats (selective failures, server-held data, timing, cost caption) |
| D-28 | P3 | wording and caption details |
| D-29 | P3 | fill.py fragilities (datasets table and REFS; `n_sig` rounding) |
