# Reviewer 9: Reproducibility and code audit

**Specialty:** ML reproducibility. I traced each headline number to a run, recomputed it, audited the evaluation and statistics code, and checked determinism and provenance.
**Method:** Read-only. I recomputed every value directly from `experiments/results/*/runs.jsonl` with my own scripts (`r9_work/agg.py`, `dump.py`, `dups.py`), not with `paper_assets.py`. I used `summary.md` only where no jsonl exists (K01 CIFAR-10). Git history was read only.

## Recommendation

**Major revision.** Confidence: high on the findings below, medium on the overall verdict.

Most headline numbers can be traced and recomputed exactly: about 90% of the cells in the matrix match or differ only by rounding. The authors' own `paper_assets.py`/`CHECK.md` mechanism is a real asset. The problems are elsewhere:

1. **The same label hides different configurations.** "Ours-A" is T_s=10 in some places and T_s=1 in others. The SVHN "final method" is a T_s=1 run.
2. **Some quoted numbers come from runs the authors themselves invalidated.** These are the F04 K=50/100 consensus-label (CSL) rows.
3. **Runs are not deterministic.** A seed fixes only the data split. Each job retrains its own feature extractor (FE), so pairing by seed across jobs pairs only the splits.
4. **The code that produced the Kaggle results cannot be recovered exactly.** There is no commit hash in any run, the generated scripts are overwritten on every rebuild, and Kaggle notebooks NB1–NB31 were never committed.
5. **One central claim is contradicted by the authors' own logs.** The paper says the F-gate makes the same choice for any threshold in 0.02–0.1. The logs show F = 0.055–0.078 on CIFAR-10.

---

## 1. Traceability matrix

Status codes: **M** = matches to 2 decimals. **R** = rounding only. **X** = mismatch. **C** = the value matches its source, but the source is a different configuration from the one the label says. **U** = cannot be traced. **I** = traced, but the source run is invalid.

LT = long tail, IF=100, Dir 0.5, K=10, metric `final_bal`. IID columns use `best_mean_acc`. All entries have n=3 seeds unless stated.

### 1a. Main long-tail table (handout Part 5.2, paper_single_idea Table 1, make_figures f08, paper_draft §5.11)

| Claim | Reported (where) | Source run / arm | Recomputed mean ± sd [seeds] | Status |
|---|---|---|---|---|
| GeFL-F MNIST | 75.25 (handout, T11, single-idea) | E20 `GeFL-F` | 75.25±1.12 [74.19,75.11,76.43] | M. The same config gives 74.62 (F01), 74.24 (F10), 74.94 (E16) and 75.31 (K02) in other jobs. |
| GeFL-F FMNIST | 58.47 | E20 `GeFL-F` | 58.47±3.59 | M. Other jobs give 58.62 / 58.74 / 59.13 / 58.90. |
| GeFL-F CIFAR-10 | 33.74 | K01 cifar10 **summary.md only** | 33.74±0.51 | M, but there are no per-seed data. |
| GeFL-F SVHN | 50.49 | K01 svhn `GeFL-F` | 50.49±2.24 [51.68,51.89,47.92] | M |
| +LA MNIST / FMNIST | 78.92 / 64.78 | F01 `+LA` (F01's own GeFL-F is 74.62 / 58.62) | 78.92±2.32 / 64.78±3.35 | M. Comes from a different job than the GeFL-F row. |
| +LA CIFAR / SVHN | 37.30 / 56.44 | K01 summary / K01 jsonl | 37.30±0.59 / 56.44±0.78 | M |
| +HWA+LA MNIST / FMNIST / CIFAR / SVHN | 87.89 / 76.02 / 40.73 / 62.45 | F01; F01; K01 summary; K01 `+HWA+LA (ours)` | 87.89±1.61 / 76.02±0.67 / 40.73±0.78 / 62.45±1.76 | M |
| +HWA+LA+CSL MNIST / FMNIST / CIFAR | 90.40 / 76.60 / 42.10 | E20 `Ours (HWA+LA+CSL)`; K01 summary | 90.40±1.42 / 76.60±0.21 / 42.10±0.74 | M. paper_draft §5.5 reports the same arm as 89.87 / 76.37 (F10). |
| FSG+LA (Gaussian) | 87.04 / 73.60 / 44.17 / 44.23 | F01; F01; K01 summary; K01 | 87.04±1.63 / 73.60±0.84 / 44.17±0.87 / 44.23±2.71 | M |
| GeFL-F + last-layer refit | 90.7 / 79.2 / – / 64.8 | F10 `GeFL-F` oracle_bal; K01 svhn oracle | 90.73 / 79.17 / – / 64.82 | R. This GeFL-F job is not the one used in the GeFL-F row: E20's GeFL-F oracle is 91.21 / 78.94. |
| Ours-A MNIST | 94.82 | E25 `Ours-A Ts=10` | 94.82±0.57 [95.38,94.84,94.23] | M. The same config in K09 gives 95.18±0.08. |
| Ours-A FMNIST | 79.67 | E25 `Ours-A Ts=10` | 79.67±0.54 | M. K09 gives 79.79. |
| Ours-A CIFAR-10 | 44.38 (handout, f08) / 46.1 (abstract, readout_report) | K08 `Ours-A Ts=10` / K08 `Ours-A (PC+MC+KH+LA+CSL)` (T_s=1) | 44.38±1.07 / 46.06±1.29 | C. Two configurations are reported under one name. |
| Ours-A SVHN | 54.41 (handout, f08) / 55.76 (draft T14, f12) / 55.89 (draft T13) | K08 Ts=10 / K11 Ts=1 / K08 Ts=1 | 54.41±1.13 / 55.76±1.68 / 55.89±1.20 | C |
| "Ours-A with SVHN switch" | 65.60 | K10 `Ours+MC+KH (CVAE+HWA)`, **ts=1** | 65.60±1.50 [64.02,65.77,67.01] | C. The stated final method has T_s=10; this run uses T_s=1. |
| ESA ≈ 67 SVHN, ≈ 47 CIFAR (single-idea) | ≈67 / ≈47 | K10 bbc_bal (Ts=1) / K08 Ts=10 bbc_bal | 66.96 / 47.26 | R / C |
| ESA (+gated BBC) MNIST / FMNIST | 95.17 / 80.55 (also 95.2 / 80.6 in the abstract) | E25 Ts=10 bbc_bal | 95.17±0.55 / 80.55±0.65 | M |
| Oracle MNIST / FMNIST | 94.93 / 81.55 | E25 Ts=10 oracle_bal | 94.93±0.70 / 81.55±0.76 | M. On seed 2 the oracle (94.19) is below the head (94.23); see L10. |
| Oracle CIFAR / SVHN | 50.64 / 59.13, 68.4 (switch) | K08 Ts10 oracle; K10 oracle | 50.64 / 59.13 / 68.37 | M |
| Tail recall 52→91, 35→74 (abstract) | | E20 GeFL-F tail; E25 Ts10 **without BBC** tail | 51.9→91.1; 35.5→73.6 | R. The abstract pairs balanced accuracy with BBC and tail recall without BBC. |

### 1b. IID table (handout 5.3, single-idea Tables 2 and 2b, f09)

| Claim | Reported | Source | Recomputed | Status |
|---|---|---|---|---|
| Ours-A MNIST K10 | 97.62 | E25 Ts=10 bma | 97.62±0.06 [97.56,97.67,97.62] | M |
| Ours-A MNIST K50 / K100 | 96.93 / 96.77 | E27 `Ours-A Ts=10` | 96.93±0.21 / 96.77±0.18 | M |
| Ours-A FMNIST K10 / K50 | 83.96 / 82.96 | E25 / E27 | 83.96±0.39 / 82.96±0.57 | M |
| Ours-A FMNIST K100 | "PENDING (NB17)" | K08 fmnist IF=1 K=100 Ts=10 | 82.04±0.52 (exists) | X (stale). The result exists but is not reported. |
| Ours-A CIFAR-10 | 60.93 | K08 cifar10_iid Ts=10 bma | 60.93±1.27 | M. On final_bal the same run scores 56.82, against GeFL-F 56.68; see L9. |
| Ours-A SVHN (mean anchor) | 74.38 | K08 Ts=10 bma | 74.38±0.79 | M. K11's T_s=1 run also gives 74.38, a coincidence. |
| SVHN with switch | 77.07 | K10 rows+MC+KH (Ts=1) bma | 77.07±0.19 | C (T_s=1) |
| Our GeFL-F MNIST K10 | 96.01 (T11, handout) / 96.02±0.04 (T01, T04) | F10 / F09 | 96.01±0.14 / 96.02±0.04 | M. Two different jobs; the CHECK.md flag here is a source-mapping artifact. |
| Our GeFL-F MNIST K50 / K100 | 95.16 / 94.61 (draft, 1 seed) vs 95.43 / 94.88 (handout, single-idea) | F09 s0 vs K03 (3 seeds) | 95.16 / 94.61 vs 95.43±0.27 / 94.88±0.29 | X across documents (different runs) |
| Our GeFL-F FMNIST K10 | 82.67 (T01) / 82.76 (T11) / 82.80 (T05) / 82.96 (E16) | F09 / E20 / F10 / E16 | all M to their own source | Four values for one quantity in one paper |
| MNIST K10 +1.18, CIFAR +1.57, SVHN +0.81 over paper best | | | 1.18 / 1.57 / 0.81 | M |

### 1c. Other headline numbers

| Claim | Reported | Source | Recomputed | Status |
|---|---|---|---|---|
| K=100 Ours-A (MNIST LT) | 94.30 | E22 Ts=10 | 94.30±0.31 [94.55,93.95,94.41] | M |
| K=100 GeFL-F / HWA+LA / FSG | 75.98 / 83.07 / 86.05 | F04 | 75.98±0.95 / 83.07±0.56 / 86.05±0.84 | M. Compared across jobs. |
| FMNIST K50 / K100 Ours-A | 79.5 / 79.0 | K08 fmnist Ts=10 | 79.47±0.38 / 79.02±0.98 | R |
| FMNIST K50 / K100 GeFL-F | 61.9 | K03 | 61.92±1.31 / 61.95±1.52 | R |
| DP ε=8 MNIST / FMNIST | 90.36 / 72.77 | E26 | 90.36±0.30 / 72.77±1.67 | M |
| DP ε=2 MNIST / FMNIST | 85.06 / 65.59 | E26 | 85.06±0.54 / 65.59±0.58 | M |
| DP: MC used on 7 / 5 classes | | E26 mc_classes_used | 7 / 5 (KH on 5 / 3) | M |
| K09 ablation, full method | 95.18±0.08 / 79.79±0.31 | K09 `Ours-A Ts=10` | 95.18±0.08 / 79.79±0.31 | M |
| K09 differences −4.72, −2.98, −1.63, −0.96, −0.71 (MNIST); −0.67, −6.18, −1.38, −0.49, −0.30 (FMNIST) | | K09 within-job paired | consistent with arm means (e.g. 90.46−95.18 = −4.72) | M. The −PC arm used a different FE (see L4). |
| MC-B FMNIST LT / IID | 81.57 / 85.75 | E32 `Ours-A-B Ts=10` | 81.57±0.31 / 85.75±0.45 | M. The MNIST seed-2 "instability" (92.50) coincides with an outlier FE (referee 0.958 against 0.968–0.975 in every other job). |
| SVHN gate result | 65.60±1.50 | K10 | as above | C (T_s=1) |
| MC-S vs Ours-A on SVHN: +6.0 (+4.5..+7.0, p=0.016), +2.7 IID | | K11 within-job, shared FE | +4.53, +7.03, +6.54 → 6.03; IID +2.67 | M |
| Readout gap: GeFL-F 74.2/58.7/50.5 → refit 90.7/79.2/64.8; gaps 16.5/20.5/14.3 | | F10 GeFL-F, K01 svhn | 74.24→90.73 (16.49); 58.74→79.17 (20.43); 50.49→64.82 (14.33) | R |
| Readout gap: HWA+LA 87.8/76.0 → 92.4/80.1 | | F10 | 87.78→92.37; 75.99→80.08 | R |
| Readout gap: Ours-A 94.8/79.7 → 94.9/81.6; FMNIST K50/K100 gaps 1.7 / 2.0 | | E25; K08 | K50 gap is 81.25−79.47 = **1.78** (reported 1.7); K100 2.01 | R / minor X |
| Readout gap: "Ours-A CIFAR-10 46.1/52.3, gap 6.2" | | K08 **Ts=1** | 46.06 / 52.25 | C (the CIFAR IID row in the same table is T_s=10) |
| T03 (12 head / refit / ER / BBC values) | e.g. FMNIST GeFL-F 61.2 / 78.7 / 79.1 / 65.1 | T03 runs.jsonl (1 seed, quick_scale 0.4, CPU) | all R; ER−oracle ∈ [−0.75, +0.60] across 9 runs | M |
| T01: 0/1,200 violations; θ 0.61–0.93; 139/197 contractions; tail/head ratio 0.152 vs 0.155; 0.848 at λ=0 | | T01 summary.md | as quoted. The mean slack 0.983 is from the λ=1e-3 arm only, not all three. | M / R |
| §3.3 row ratios 0.33 (0.09, 0.17, 0.72) / fidelity 8.8 | | F01 | 0.33 / 8.7 | R. Seed 2 has ratio 0.72, i.e. essentially no collapse. |
| "+12 to +20 points from HWA on MNIST, FMNIST, SVHN **and CIFAR-10**" (abstract) | | F01, K01 | +13.3 / +17.4 / +12.0 / **+7.0** (+8.4 with CSL) | X |
| §5.3 CIFAR text: "HWA+LA 40.4, Gaussian 43.4" | | K01 | 40.73 / 44.17 | X (stale partial results; not caught by CHECK.md) |
| §5.3 K=50: +HWA+LA+CSL 86.8, FSG+LA+CSL 87.2, MIX 87.9; K=100: 82.7 / 85.5 / 85.2 | | **F04 runs_before_cslfix.jsonl** | 86.78 / 87.17 / 87.86; 82.71 / 85.45 / 85.22 | I. The runs exist (PROGRESS.md says "no run"), but the authors moved them aside as confounded by the shared-pool bug (commit ea148a8 / 348f09a). |
| §5.9 LT budget: 61.33 / 76.98 / 79.33 (2 seeds) | | E16 (now 3 seeds) | 59.13 / 76.94 / 79.23 | X (stale; listed in CHECK.md). "+18.0 over GeFL-F" is now +20.1. |
| §5.8 DDPM "Ours 83.10" | | F11 | 82.82 (n=2) | X (in CHECK.md) |
| "With BBC +4.8 over FSG" (CIFAR-10) | | K08 Ts1 bbc vs K01 | 48.71−44.17 = 4.54 | X (computed from the rounded "≈49") |
| "F is 0.14–0.60 on the other datasets; any threshold in 0.02–0.1 gives the same choice" | | `class_sep_F` in K13 cifar10 | IID 0.055 / 0.063 / 0.055; LT 0.067 / 0.072 / 0.078 | **X**. This contradicts the authors' own logs. |
| K02 ablations ("removing HWA costs 7–8, LA 5–8") | | K02, **n=2** for the ablation arms | 7.07 / 8.07; 4.69 / 7.80 | R. The text does not say n=2, and "Ours" averages 3 seeds while each ablation averages 2. |

---

## 2. Cross-document inconsistencies, with explanations

| # | Quantity | Values and where | Explanation | Verdict |
|---|---|---|---|---|
| I1 | CIFAR-10 LT "Ours-A" | 44.38 (handout table, f08, draft T13 Ts=10 row); **46.1** (draft abstract; readout_report; handout Part 6) | T_s=10 vs T_s=1 runs of the same K08 job. The abstract headlines the better T_s=1 run while the method is defined with T_s=10 (draft §4.6; single-idea §4.5). | Configuration difference reported under one label. Selecting T_s per dataset on test results is a form of test-set selection. |
| I2 | SVHN LT "Ours-A" | 54.41 (f08, handout) vs 55.76 (f12, draft T14) vs 55.89 (draft T13) | K08 T_s=10 / K11 T_s=1 / K08 T_s=1. The last two are the same config in two jobs and differ by 0.13 from non-determinism. | Configuration difference plus re-run noise |
| I3 | Ours-A MNIST LT | 94.82 (E25) vs 95.18 (K09); FMNIST 79.67 vs 79.79 | Same config and split; different hardware, code version (K09 has extra fields) and FE. Seed 2 differs by 0.97. The sd is 0.57 against 0.08. | Non-determinism, not an error. The paper should state both values or pool them (n=6). |
| I4 | GeFL-F MNIST LT | 75.25 (abstract/T11/handout), 74.6 (§5.3), 74.2 (§5.3 K-table, readout), 74.24 (§5.5), 75.3 (§5.6), 73.92 (parity, seed 0) | E20 / F01 / F10 / F10 / K02 / F01 seed 0. All the same configuration. Per-seed spread across jobs reaches 3.1 points (seed 1). | Same quantity reported 5 ways. Each gain is computed against a different baseline. |
| I5 | GeFL-F FMNIST LT | 58.47 (E20), 58.62 (F01), 58.7/58.74 (F10), 58.9 (K02), 61.33 (stale E16 2-seed; now 59.13); 61.9 at K=50/100 (K03) | Same as I4. 61.9 is a different K, so it is legitimate. | Same as I4 |
| I6 | Minimal variant MNIST LT | 89.87 (§5.5 F10) vs 90.40 (§5.11, handout step table; E20) vs 90.2 / 90.15 (E16 / K02) | Re-run noise | Two values in one paper |
| I7 | "CVAE+HWA loses X from K=10 to K=100" | 4.7 (draft, F10 baseline) vs 4.8 (handout, single-idea, F01 baseline) | Different K=10 reference job | Minor |
| I8 | Our GeFL-F MNIST IID K50 / K100 | 95.16 / 94.61 (draft, F09, 1 seed) vs 95.43 / 94.88 (handout, single-idea, K03, 3 seeds) | Different runs | The comparison with the paper's best changes sign of nothing, but the parity table should use one source. |
| I9 | Gate statistic F | draft and single-idea "0.14–0.60, any threshold 0.02–0.1"; handout "F = 0.07–0.60"; logs: CIFAR 0.055–0.078 | The 0.14 figure came from a reduced-schedule run | Genuine error. The gate margin on CIFAR-10 IID is 0.005. |
| I10 | FMNIST F | draft "0.53–0.60"; logs 0.40–0.71; D01 local vs Kaggle 0.42 vs 0.57 on the same seed | FE non-determinism | Report a range or mean ± sd |
| I11 | f12 vs f08 SVHN "Ours-A" | 55.76 vs 54.41 | Same as I2 | Figures disagree with each other |
| I12 | Step-wise table (handout 4.7, f10, single-idea Table 4) | GeFL-F 75.25 (E20) → +LA 78.92 (F01) → +HWA 87.89 (F01) → +CSL 90.40 (E20) → PC 90.75 (E20) → +MC 92.31 (E20) → +KH 93.11 (E25) → T_s 94.82 (E25) | Three jobs with distinct FEs. Within F01, the step to +HWA+LA starts from 74.62, not 75.25. A consistent alternative exists in F10 (74.24 / 87.78 / 89.87). | **Not one consistent set of runs.** The "same seeds" claim is true for the splits only. Step sizes are mixed with ~0.5-point job noise. |
| I13 | "Our GeFL-F" MNIST IID parity | CHECK.md flags 96.02 vs 96.01 | paper_assets maps T01 to F10; the draft quotes F09 | Mapping artifact, not a data error |
| I14 | K=50 CSL rows | draft 86.8 / 87.2 / 87.9 vs PROGRESS "no run exists" | They do exist, in runs_before_cslfix.jsonl, and are invalid | PROGRESS and the draft are both wrong, in opposite ways |
| I15 | "Ours-A FMNIST K100 IID" | single-idea "PENDING (NB17)" | K08 has it (82.04, above the paper's 81.65) | Stale |

---

## 3. Figure scripts with typed numbers

- `make_figures.py` f05, f08, f09, f10, f11, f12: every value matches a source run to 2 decimals. The problems are labelling ones: I1, I2 and I11 (T_s=1 and T_s=10 values under the same name), and f12 marks "ESA's gate choice" with a T_s=1 run. The IID values in f09 (74.38 vs 77.07 for SVHN) are correct for their sources.
- `fig_readout.py`: MNIST and FMNIST values match F10 / E25 after rounding.
- `fig_why.py`:
  - w01 (10/10/120/125; 4/4/114/119; 9/9/112/117) matches T01.
  - w07 [50.5, 54.4, 65.6] mixes T_s=10 (54.4) with T_s=1 (65.6). The honest same-T_s comparison is 55.8/55.9 → 65.6.
  - ev06 / w07(a) NCM [82, 69, 41, 12] matches D01.
- PROGRESS.md itself says these hand-made figures are "superseded by docs/paper_assets/figures", yet the handout still embeds them.
- **Step-wise ablation table (handout 4.7):** not from one consistent set of runs (I12). The KH "9/9 paired comparisons, p=0.004" result is cross-job: E25 against E20, with different FEs (referee accuracy differs per seed). Its sign pattern recomputes correctly: +1.24/+0.65/+0.52; +0.35/+0.44/+1.18; +0.07/+0.95/+0.25.

## 4. Determinism and seeds

- **What a seed fixes:** `make_partition` uses `np.random.RandomState(seed)`, so the split and held-out set are deterministic. I confirmed that `class_counts` are identical between D01 local and D01 Kaggle. `set_seed(seed)`, `seed+1000` and `seed+2000` seed the global torch/numpy RNG per stage.
- **What it does not fix:**
  - `torch.backends.cudnn.benchmark = True` is set; `cudnn.deterministic` and `use_deterministic_algorithms` are not.
  - Nothing records the hardware, torch version or commit.
- **Demonstrated non-determinism:**
  1. *D01 local vs Kaggle, same seed:* only 42–266 of 2,345–2,851 logged numbers are identical. FMNIST seed-2 F is 0.424 vs 0.565. A train-on-fake recall moved from 55% to 86%. The final_bal of Ours-A without LA differs by 1.5 points. The handout's claim "same numbers within about 1 point" is optimistic.
  2. *Every job retrains a different FE for the same seed:* referee_test_acc differs across F01, F10, E20, E25, E26, E16, K02, E29, E31, E32 and K09 for every seed.
  3. *Inside one Kaggle job, the two GPU workers trained different FEs for the same seed:*
     - K09: the −PC arm has F=0.2938 against 0.3073 for the other arms on seed 0.
     - K10 and K11: two F values per seed.

     So within-job pairing is also broken for arms placed on different workers.
  4. *Identical configs re-run across jobs:* GeFL-F MNIST LT per-seed range is 0.9–3.1 points, job means 74.24–75.31. Ours-A MNIST job means are 94.82 vs 95.18.
- **CPU (T03):** nearly but not exactly deterministic. A GeFL-F re-run gives 61.20→61.19 and ER 79.12→79.22; +HWA+LA gives 72.96→73.12.
- **Consequence:** "paired by seed" tests across jobs pair splits, not models. This affects T08, T11, T12, T13, T14 and the abstract's p ≤ 0.011. Effects of ≤ 0.5 points cannot be distinguished from job-to-job noise: FMNIST margin over GeFL-F+refit +0.5, PCR +0.22, hybrid −0.07, FMNIST LT MC-S "tie". In the cross-job reported claims the noise sd is about 0.3–0.7 per seed.

## 5. Can an external researcher reproduce this?

| Item | State | Blocker? |
|---|---|---|
| Data | Downloaded automatically: torchvision for MNIST/FMNIST; Hugging Face `resolve/main` with no revision pin for SVHN/CIFAR-10, falling back to torchvision. SVHN order from HF and from the torchvision .mat may differ, which would change the splits. CIFAR is claimed "verified identical"; there is no such check for SVHN. | Medium |
| Environment | `requirements.txt` has lower bounds only (torch>=2.1). It omits scipy, pyarrow and Pillow, which the code needs. No lockfile, no CUDA/driver record, no Kaggle image version. | High for exact reproduction |
| FE / generator checkpoints | None saved; every stage is retrained in each job. | High (combined with non-determinism) |
| Configs | Full `method` dict stored in every jsonl row. Good. | – |
| Code version | `build.py` regenerates `K*.py` / `E*.py` from the **current** `core.py` (K01 script last changed Oct 14 23:47, after the Kaggle run). Runs carry no commit hash. Kaggle notebooks NB1–NB31, which produced K01–K13, were never committed (only NB32–39 are in git). Resume-by-`run_id` silently reuses old rows after code changes; this is the F04 contamination mechanism. | **High** |
| K01 CIFAR-10 runs.jsonl | Missing; only the pasted summary.md exists. The LEDGER marks it "COMPLETE 27/27", counting from the summary. CIFAR-10 LT and IID baselines (GeFL-F, +LA, HWA+LA, minimal variant, FSG) have no per-seed data, and their p-values are Welch from mean/sd. | Unavailable evidence (not misreporting) |
| LEDGER "NOT RUN" notebooks | K05–K07, K10/K11 on CIFAR-10, K12 (NB25–29, incl. SVHN T_s=10 for the final method and SVHN/CIFAR at K=50/100), K14 (seeds 3–4), K15 (collapse-fix baselines), CIFAR DDPM. The paper correctly marks most as pending. Two gaps matter: (a) the final SVHN method at T_s=10 (NB25) is "superseded" in PROGRESS although nothing replaces it; (b) K15 is the key baseline (AdamW / no decay on rows) for the novelty of HWA. | Unavailable evidence |
| Partial jobs | F04 K=50 CSL arms (invalidated, not re-run); F09 / F11 / E24 partial (E24 KME-Gen n=1; DDPM FMNIST n=1–2). | Medium |
| T01 / T03 | 1 seed; reduced schedule; CPU. Disclosed. | Low |

## 6. Code-level issues that could bias results

1. **Shared synthetic pool at K>10 (fixed in ea148a8).** Clients of the same architecture trained on identical synthetic slices. Contaminated: F04 K=50/100 CSL rows (moved aside, but still quoted in draft §5.3) and the K03 CSL rows in the repo (K03 K50 summary is from Oct 9 23:14, before the fix; the "NB4b/NB5b" re-runs were never run). Headline E22 / E27 / E29 / K08 at K>10 are post-fix by commit date. Without a code hash this is an inference, not proof.
2. **runs.csv overwrite (multi-GPU).** Each worker writes runs.csv from its own rows only. The jsonl is the source of truth, so no number is biased. Separately, workers append to a shared jsonl, which carries a small interleaving risk; I found no malformed lines.
3. **Metric code (`summarize_eval`, `eval_headers`, `Tracker`).**
   - `bal` = per-class recall averaged over classes and over architectures, on the full test set. This is correct for SVHN's imbalanced test set.
   - `tail` is hard-coded to classes 6–9; in IID it is meaningless but unused.
   - `best_mean_acc` is the **plain** accuracy maximised per architecture over all evaluated rounds, *including FE-warm-up rounds*, evaluated every 10 rounds. Selecting the best round on the test set inflates it; acknowledged as an "audit point". The inflation is material where it carries claims: CIFAR-10 IID Ours-A T_s=10 has bma 60.93 but final_bal 56.82, which equals GeFL-F's 56.68; CIFAR LT bma 48.07 vs final 44.38; SVHN IID Ours-A T_s=10 final 71.82 vs GeFL-F 74.54. The "above the paper's best on CIFAR-10" claim depends entirely on test-selected rounds. Evaluating every 10 rounds also differs from the paper's protocol.
4. **Test-set use in selection.**
   - BBC is fitted on synthetic samples only (no leakage). Its gate uses global counts.
   - However, the final method's choices were made after seeing test results across arms: the F-gate threshold of 0.05, the SVHN branch (rows+MC+KH picked from 5 SVHN variants evaluated on test), T_s per dataset (the abstract quotes T_s=1 on CIFAR), and BBC on/off. There is no validation split, although 2,000 held-out images exist and are used only for the referee.
5. **"Oracle" is not an upper bound.** `classifier_oracle` runs 300 SGD steps (lr 0.05, batch 128) starting from the trained fc. It can end below the trained head (E25 seed 2: 94.19 < 94.23) and below BBC (94.93 < 95.17). Two claims rely on it: "Ours-A reaches its own oracle" and "upper bound of CCVR/CReFF/Fed3R". The exact readout (LDA) sometimes beats it, which shows the refit is under-converged.
6. **paper_assets.py.**
   - The store key omits `alpha`. It is safe today, because no job mixes alpha at equal IF, but fragile.
   - Duplicate (label, seed) rows silently overwrite one another.
   - The CHECK tolerance is ±0.05 for 1-decimal drafts and an extra 0.5 for draft values 49/57/62/67, which hides the 4.8 vs 4.54 kind of error.
   - CHECK only scans paper_draft, not paper_single_idea or the handout, which is where most of I1–I15 live.
7. **DP clipping bound.** Taken from the 90th percentile of the held-out pool, which is drawn from the same training distribution and called "public". This is an assumption that must be stated.
8. **`-PC` and other cross-worker arms do not share an FE** (see §4), contrary to the K09 analysis note.

---

## 7. Evidence ledger

| ID | Claim and location | Evidence | Reasoning | Consequence | Severity (1 = fatal … 4 = minor) | Confidence | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| L1 | Draft §5.3 K-table: K=50 86.8 / 87.2 / 87.9; K=100 82.7 / 85.5 / 85.2 | Values = means in `F04_clients/runs_before_cslfix.jsonl`; commit 348f09a calls them "confounded" | Quoted numbers come from runs the authors invalidated | One table row set is invalid | 2 | High | Demonstrated | Remove the rows or re-run post-fix (K=100 CSL post-fix n=2: 83.04) |
| L2 | "Ours-A" CIFAR 46.1 (abstract) vs 44.38; SVHN final 65.60 / 77.07 from T_s=1 while the method specifies T_s=10 | K08/K10 `method.head.ts` | T_s chosen per dataset after seeing test results; the label hides it | Headline numbers on 2 of 4 datasets are not the stated method | 2 | High | Demonstrated | Fix one T_s (or state the per-dataset rule chosen without test data) and report it everywhere; run NB25 (SVHN T_s=10) |
| L3 | "F gate: any threshold 0.02–0.1 makes the same choice; F ≥ 0.14 elsewhere" | `class_sep_F` CIFAR 0.055–0.078 (K13); D01 0.063–0.076 | The claim is false; CIFAR IID sits 0.005 above the threshold | The gate's robustness claim fails; the choice on CIFAR is fragile | 2 | High | Demonstrated | Correct the text; report F ± sd; test sensitivity |
| L4 | Paired tests across jobs ("same seeds and splits") | referee_test_acc / F differ per job and even per worker within a job | Seed fixes the split only | p-values overstate pairing; sub-0.5-point claims are not resolvable | 2 | High | Demonstrated | Share FE/generator checkpoints per seed, or run all compared arms in one job; set deterministic flags |
| L5 | Exact code that produced the K01–K13 results | No commit hash in rows; K*.py regenerated Oct 14; NB1–31 not in git | Cannot re-run the exact code | Exact reproduction is impossible; only approximate | 2 | High | Demonstrated | Store the git SHA plus environment in every row; archive the notebooks and scripts as run |
| L6 | Step-wise table (handout 4.7, f10, single-idea T4) | Rows from E20, F01 and E25 with different baselines | Not one consistent run set | Step sizes contaminated by ~0.5-point job noise | 3 | High | Demonstrated | Rebuild from one job (e.g. F10 + E25 with a shared FE) |
| L7 | Abstract: "HWA +12 to +20 on MNIST, FMNIST, SVHN and CIFAR-10" | K01: CIFAR +7.0 (+8.4 with CSL) | Misreport | Overstated CIFAR gain | 3 | High | Demonstrated | Correct to "+7 to +17" |
| L8 | Stale numbers: §5.3 CIFAR 40.4 / 43.4; §5.9 61.33 / 79.33 / "+18.0" / "2 seeds"; DDPM 83.10; "+4.8 over FSG"; FMNIST K100 IID "pending" | T13, T10, T16, K08 | Text not updated | Small numeric errors | 3 | High | Demonstrated | Regenerate the text from numbers.csv; extend CHECK to all documents |
| L9 | IID "above the paper's best" on CIFAR (60.93) and SVHN (77.07) | final_bal: Ours-A CIFAR IID 56.82 ≈ GeFL-F 56.68 | Rests on test-selected best rounds | IID superiority on CIFAR is metric-dependent | 3 | High | Demonstrated | Report final-round metrics next to bma |
| L10 | "Oracle = upper bound"; "MNIST reaches its oracle"; "GeFL-F + refit = upper bound of CCVR/CReFF/Fed3R" | 300-step refit; oracle < head on E25 s2; oracle < BBC | Under-converged refit | The stronger-baseline margin (+4.1 / +0.5 / +0.8) may be over- or under-stated | 3 | Medium | Suspected | Converge the refit (closed-form, or many epochs with a validation split) |
| L11 | GeFL-F baseline reported as 75.25 / 74.6 / 74.2 / 75.3 and FMNIST 58.47–59.13 | Five jobs, same config | Re-run noise; each gain computed against a different baseline | Confusing; gains differ by up to 1 point | 3 | High | Demonstrated | Pool the re-runs or fix one canonical baseline per table |
| L12 | Readout table / handout Part 6 mixes T_s=1 (CIFAR LT) and T_s=10 (CIFAR IID, others) "Ours-A" | K08 | Label hides configuration | Minor | 4 | High | Demonstrated | Label T_s |
| L13 | K01 CIFAR runs.jsonl missing | summary.md only | Unavailable, not misreported | CIFAR baseline tests unpaired; cross-job anyway | 3 | High | Unresolved | Copy back the zip or re-run |
| L14 | D01 "Kaggle rerun agrees within ~1 point" | F 0.42 vs 0.57; recall 55 vs 86% | Overstated agreement | Mechanism figures carry FE noise | 3 | High | Demonstrated | Report the replication spread |
| L15 | MC-B "unstable on MNIST seed 2" | E32 seed-2 referee 0.958 (outlier FE) | Instability may be an FE draw, not MC-B | Interpretation | 4 | Medium | Suspected | Re-run E32 s2 with a shared FE |
| L16 | K02 ablations presented without n | n=2, unpaired with 3-seed "Ours" | Disclosure | Weak evidence | 4 | High | Demonstrated | State n=2; compare on shared seeds |
| L17 | Env and data pins | requirements unpinned; scipy/pyarrow/PIL missing; HF without revision | Reproducibility | Approximate reproduction only | 3 | High | Demonstrated | Lockfile, dataset hashes, revision pins |
| L18 | K03 CSL rows (repo) | Pre-fix by date; re-run never done | Possibly confounded | Not used in headlines | 4 | Medium | Suspected | Mark invalid in the ledger |
| L19 | paper_assets: alpha missing from the key, overwrite on duplicates, loose CHECK tolerance | Code lines 52, 460 | Fragile tooling | Could hide future errors | 4 | High | Demonstrated | Add alpha to the key; assert on duplicates; tighten the tolerance |

## 8. Questions to the authors

1. Which T_s defines "Ours-A / ESA"? Why does the abstract quote CIFAR-10 at T_s=1 (46.1) and SVHN at T_s=1 (65.6 / 77.07), when the method specifies T_s=10? Was the choice made without looking at test accuracy?
2. Will you remove or re-run the F04 K=50/100 CSL rows that you moved to `runs_before_cslfix.jsonl` but still quote?
3. How do you reconcile the gate claim ("0.14–0.60; any threshold 0.02–0.1") with logged CIFAR-10 F = 0.055–0.078?
4. Can you provide, for each result folder, the git commit and the exact notebook/script that produced it, plus the torch/CUDA versions and GPU type? Can NB1–NB31 be recovered?
5. Will you make the pipeline deterministic (`cudnn.deterministic`, no benchmark, seeded workers), or share FE/generator checkpoints per seed, so that cross-job "paired" tests compare the same models?
6. Within K09/K10/K11, arms on different GPU workers used different FEs. Which ablation differences are affected?
7. Will you report final-round balanced accuracy alongside `best_mean_acc` for every IID comparison with the paper (CIFAR-10 IID final 56.82 vs GeFL-F 56.68)?
8. Is the 300-step `classifier_oracle` converged? Please give a closed-form or validation-tuned refit before calling it an upper bound or the "GeFL-F + recalibration" baseline.
9. Which GeFL-F run is canonical (E20 75.25, F01 74.62, F10 74.24, K02 75.31)? Will you pool the five re-runs to report re-run variance?
10. Can you restore the K01 CIFAR-10 runs.jsonl? Will you run K15 (AdamW / no-decay-on-rows baselines) and NB25 (SVHN final method at T_s=10)?
11. Was SVHN loaded from the HF mirror or from torchvision in each environment, and does the image order match (it determines the splits)?
