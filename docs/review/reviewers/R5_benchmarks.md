# Reviewer 5: Benchmarking and fairness of the evaluation

**Specialty:** long-tailed recognition, federated learning (label skew and long tails), evaluation of generative models.
**Manuscript:** `docs/paper_draft.md` (GeFL-F under long-tailed clients; minimal variant HWA+LA+CSL; anchored "Ours-A"/ESA).

## Recommendation

**Major revision. Do not accept in the current form. Confidence: high (4/5).**

The internal comparison is clean. Each arm uses the same seed-fixed split, frozen extractor and schedule (`core.py:2216-2250`), so "repaired GeFL-F beats unrepaired GeFL-F" is well supported on MNIST and FMNIST.

The external benchmark is not a fair test of the paper's claims. Under the long tail there are only two kinds of comparison method:

- generator-free methods that, at K = 10, reduce to *local training* (one client per architecture);
- variants of GeFL-F that the authors built themselves.

No published federated long-tail method is run (CCVR, CReFF, FedLC, FedRS, Fed3R, BalanceFL, FEDIC, FedLoGe, Fed-GraB). The authors' own diagnostic shows that a federated classifier-recalibration step applied to plain GeFL-F (CCVR/Fed3R-style "exact readout") lands within 0–1 point of the final method on FMNIST and SVHN. That arm was never run at full schedule.

Model selection uses test-set results throughout: there is no validation split. The headline IID comparison also relies on `best_mean_acc`, which selects the best round on the test set.

---

## A. Dataset construction (verified in code)

| Item | Finding | Evidence |
|---|---|---|
| IF definition | `n_c ∝ IF^{c/(C-1)}` with `IF=0.01` in every spec. The head is class 0 and the tail is class 9; "IF = 100" in the paper is the 100:1 ratio. The text formula (§5.1) would invert head and tail if read literally with IF = 100. | `core.py:205-207`; specs, e.g. `F01_main_longtail.py:16` |
| Per-class counts (recomputed) | **MNIST/FMNIST:** 2417, 1449, 868, 520, 312, 187, 112, 67, 40, 24 (total 5996). **SVHN:** 2951 … 29. **CIFAR-10:** 4800, 4800, 3620, 2170, 1301, 779, 467, 280, 168, 100. On CIFAR-10 the head is capped at the 4800 available, so the *effective IF is 48*, not 100, and the total is 18,485, not 25,000. This contradicts "IF = 100 … total data budget fixed" (§5.1). | `core.py:204-210`; `r5_work/inventory.txt` (n_train = 18485 for CIFAR-10 LT) |
| Class order | Fixed and identical across all seeds: tail = classes 6–9 always. Examples: MNIST 6/7/8/9; FMNIST shirt / sneaker / bag / ankle boot, so FMNIST's hardest class, *shirt*, is always a tail class; SVHN digits 6–9. Seeds vary only the subsample, the Dirichlet draw and the initialisation. Reported SDs therefore exclude class-permutation variance. The design choices (BBC, τ, F-gate) were all made against the same tail identities. | `core.py:206-210`, `772-773` (tail = `rec[6:10]`) |
| Dirichlet | Dir(0.5) per class over K clients, resampled until each client has ≥ 10 samples, then topped up from the largest client. Only α = 0.5 is ever used for long-tail runs. | `core.py:215-231` |
| Data fractions | 10% of the MNIST/FMNIST/SVHN training sets and 50% of CIFAR-10, as stated. | `core.py:47-61` |
| Test set | MNIST/FMNIST/CIFAR-10 test sets are balanced. SVHN's is **not**: `bal` is the mean per-class recall (correct), but `best_mean_acc` uses plain accuracy, so SVHN `best_mean_acc` is head-weighted (e.g. HWA+LA 66.86 bma vs 62.45 bal). | `core.py:743-777` |
| Train / val / test | **There is no validation split.** 200 images per class are held out of the training pool. They are used only for the referee classifier, MND, and the DP clipping bound (labelled "public" although it comes from the same training pool). All selections were made by looking at test numbers. Examples: T_s per setting, the generator per dataset (F-gate added after SVHN failed), the BBC gate, τ, MC vs MC-S vs MC-B, and the 40+ variants on MNIST LT. | `core.py:209`, `1441`, `1738`, `2269-2272`; draft §5.11 "registered prediction that failed" (l. 829-831) |
| Feature extractor | Stage I: FE and heads trained by FedAvg with plain CE on the *long-tailed* federated data (no LA unless LAFE), T_FE rounds, then frozen. It is cached and shared by every generator arm with the same seed, so all GeFL-F-family methods use the same FE (fair). FedAvg and LG-FedAvg train their own FE for T_FE + T_TN rounds. | `core.py:796-845`, `2235-2250`, `2227-2232` |
| Extra information given to the method | Ours-A uses secure-aggregated exact class statistics: counts, Σh, Σ‖h‖², Σφ(h) (RFF), and for MC-S/MC-B also class covariances. It adds per-round consensus soft labels and a server-side BBC fitted on the generator. Baselines get none of this, except FSG+LA (Gaussian means and covariances) and GeFL-F+LA (each client's own prior). No baseline gets the classifier-side use of the same statistics: CCVR/Fed3R/ER would use exactly this release. BBC is *computed* for GeFL-F (`bbc_bal`), but the paper reports BBC only on Ours-A (MNIST LT: GeFL-F+BBC 79.24 vs GeFL-F 75.25). | `core.py:2259-2266`, `2401-2415`; `inventory.txt` (E20 GeFL-F bbc 79.24 / 60.49) |

## B. Baseline inventory: what was actually run under the long tail

Full-schedule runs, IF = 100 (`IF=0.01`), Dir 0.5 unless noted. Values are final balanced accuracy (mean of 3 seeds unless n is shown) from `runs.jsonl`, or `summary.md` for K01 CIFAR-10.

| Dataset | K | Method (label) | Seeds | Final bal | Source folder | Notes |
|---|---|---|---|---|---|---|
| MNIST | 10 | FedAvg (grouped) | 0-2 | 62.86 | F01_main_longtail | = local training (1 client per architecture) |
| MNIST | 10 | LG-FedAvg / +LA | 0-2 | 63.22 / 67.84 | F01 | heads unshared at K = 10 |
| MNIST | 10 | GeFL-F | 0-2 | 74.24–75.31 | F01, F10, E16, E20, K02 | 5 reruns, same seeds |
| MNIST | 10 | GeFL-F + LA | 0-2 | 78.92 | F01 | only "tuned for LT" GeFL-F |
| MNIST | 10 | FSG + LA (federated Gaussian, CCVR-like generator) | 0-2 | 87.04 | F01 | **omitted from the main table T02/§5.3** |
| MNIST | 10 | HWA+LA / +CSL | 0-2 | 87.89 / 89.87 | F01, F10 | |
| MNIST | 10 | Ours-A Ts=10 (+BBC) | 0-2 | 94.82 (95.17) | E25 | |
| MNIST | 10 | GeFL-F + NOWD / ADAMW / ROWNOWD (optimiser fixes) | 0 (quick) / none | – | E02 quick; K15 NOT RUN | obvious alternative fix never run at full schedule |
| MNIST | 50 | GeFL-F, +LA, HWA+LA, FSG+LA | 0-2 | 78.10 / 81.35 / 86.46 / 87.66 | F04 | **no Ours-A or CSL rows at K = 50** (the draft's 86.8 / 87.2 / 87.9 have no source; PROGRESS.md claim 4) |
| MNIST | 100 | GeFL-F, +LA, HWA+LA, FSG+LA | 0-2 | 75.98 / 79.51 / 83.07 / 86.05 | F04 | |
| MNIST | 100 | Ours-A Ts=10 | 0-2 | 94.30 | E22 | |
| FMNIST | 10 | FedAvg / LG-FedAvg / +LA | 0-2 | 52.79 / 52.45 / 57.57 | F01 | |
| FMNIST | 10 | GeFL-F / +LA / FSG+LA / HWA+LA | 0-2 | 58.62 / 64.78 / 73.60 / 76.02 | F01 | |
| FMNIST | 10 | Ours-A Ts=10 (+BBC); Ours-A-B | 0-2 | 79.67 (80.55); 81.57 | E25; E32 | |
| FMNIST | 50, 100 | GeFL-F, +LA, HWA+LA, FSG+LA(+CSL), minimal | 0-2 | 61.9 / 66.5 / 75.5–73.3 / 75.1–74.6 | K03 | |
| FMNIST | 50, 100 | Ours-A Ts=10 | 0-2 | 79.47 / 79.02 | K08 fmnist | |
| SVHN | 10 | GeFL-F / +LA / HWA+LA / FSG+LA | 0-2 | 50.49 / 56.44 / 62.45 / 44.23 | K01 svhn | **no FedAvg / LG-FedAvg** (K07 = NB13 NOT RUN) |
| SVHN | 10 | Ours-A / Ours-A-S / rows+MC+KH (final) | 0-2 | 55.89 / 61.80 / 65.60 | K08, K11, K10 | |
| CIFAR-10 | 10 | GeFL-F / +LA / HWA+LA / minimal / FSG+LA | 0-2 | 33.74 / 37.30 / 40.73 / 42.10 / 44.17 | K01 cifar10 (summary only, runs.jsonl missing) | Welch tests only; **no FedAvg / LG-FedAvg** |
| CIFAR-10 | 10 | Ours-A Ts=1 (+BBC) / Ts=10 | 0-2 | 46.06 (48.71) / 44.38 | K08 cifar10_lt | |
| MNIST, FMNIST | 10 | GeFL-F → minimal at IF = 1 (Dir 0.5) and IF = 10 | 0-2 | T06 | K02 | **only the minimal variant; Ours-A never at IF ≠ 100** |
| all | – | DCGAN-F / DDPM-F under long tail | – | – | F11 LT arms missing | none of the paper's other 9 GeFL variants under long tail |
| all | – | CCVR, CReFF, FedLC, FedRS, Fed3R, BalanceFL, FEDIC, FedLoGe, Fed-GraB, CLIP2FL | – | – | – | **cited only.** `baselines/creff.py` is CCVR-style Gaussian re-training on a different homogeneous backbone, mislabelled as CReFF, and is never invoked by any experiment |
| MNIST, FMNIST | 10 | ER (federated exact-statistics readout) on GeFL-F | **seed 0, 0.4× schedule, CPU** | 90.8 / 79.1 | T03_exact_readout | not a full run; `er_bal` absent from every full `runs.jsonl` |
| all LT | – | "oracle" = each head's fc refit on **pooled real** features (non-federated) | 0-2 | GeFL-F 90.9 / 79.2 / 64.8 (SVHN) | every run (`oracle_bal`) | diagnostic, not a baseline |

**Literature check.** CReFF (IJCAI'22) uses CIFAR-10/100-LT at IF 10/50/100 with Dir 0.5 and 20 clients. It compares against FedAvg, FedAvgM, FedProx, FedDF, FedBE, CCVR, FedNova, Focal, Ratio Loss and τ-norm. Fed-GraB, FedLoGe and CLIP2FL follow the same CIFAR-LT/ImageNet-LT protocol and baseline family. GeFL (TMC 2025) partitions data evenly and reports the mean accuracy over the 10 heterogeneous target networks. Its baselines are FedAvg, FedProx, FedALA, AvgKD, FedDF and LG-FedAvg, with no non-IID or long-tailed experiment. Against this standard, the paper has:

- no published FL-LT method;
- no FedDF / FedProto-style heterogeneous knowledge transfer under the long tail;
- no CIFAR-100-LT or larger-C dataset.

## C. Metric consistency

1. **Two metrics, split by regime.** Long tail uses `final_bal` (last round, mean per-class recall averaged over heads). IID uses `best_mean_acc`: per head, the maximum test accuracy over evaluated rounds, averaged over heads (`core.py:780-792`). The second is test-set model selection. It is inherited from GeFL, but it is exactly where the IID "beats the best of ten" claims live.
2. **The CIFAR-10 IID claim depends on the metric.** Ours-A Ts=10 has bma 60.93 but final_bal 56.82. GeFL-F has bma 59.10 and final_bal 56.68. On final balanced accuracy the gain is **+0.14**, not +1.8 (`inventory.txt`; `K01…cifar10/summary.md`). T_s = 10 raises bma while lowering final accuracy on CIFAR-10 and SVHN (draft l. 835-839). So T_s = 10 is chosen for IID headlines (bma) and T_s = 1 for the CIFAR-10 LT headline (46.06; with T_s = 10, 44.38 ≈ FSG 44.17). The abstract mixes the two: "46.1, or about 49 with the calibration" (T_s = 1) beside "60.93 on CIFAR-10" (T_s = 10).
3. **Comparison with the paper's numbers.** "Best of all ten methods" values (96.44, 84.28, 59.36, 76.26) are copied from the GeFL paper, while ours are reproductions. Our GeFL-F is +0.55 above the paper on MNIST and +3.24 above on CIFAR-10 (§5.2/5.10). Margins of +0.04 to +1.2 over copied numbers are therefore within implementation offset. Only our own GeFL-F on matched seeds is a fair reference. The image-space GeFL variants (GeFL-DDPM, GeFL+MixUp) were never reproduced.
4. **Ensemble vs per-head.** All headline numbers are the mean over heads (correct, matching GeFL). The ensemble (`final_ens_bal`) is a diagnostic. It is consistently higher (e.g. CIFAR-10 LT 57.1 vs 46.1) and is not used to inflate claims. Fair.
5. **Statistics.** Most comparisons are paired by seed, with n = 3 and many arms. There is no multiplicity correction, although PROGRESS.md concedes that no K09 component survives Holm. K01 CIFAR-10 tests are Welch on pasted summaries.

## D. Oracle comparisons

- `classifier_oracle` refits each head's last layer on the **pooled real training features** with class-balanced sampling, keeping the body frozen (`core.py:2133-2154`). It is non-federated and uses data no server holds.
- "Reaching the oracle" (abstract l. 34; Ours-A MNIST 94.82 vs oracle 94.93) shows only that the readout gap *of that method's own bodies* is ≈ 0. It does not show that Ours-A reaches centralised, pooled-data performance. No pipeline-level centralised oracle was run: `stage_i_oracle` and E09 are NOT RUN.
- The oracle numbers *are* informative against the paper. Plain GeFL-F's bodies, refit, give 90.9 (MNIST), 79.2 (FMNIST) and 64.8 (SVHN). The final method's heads score 94.8 / 79.7 / 65.6. The authors' T03 shows a **federated**, secure-aggregatable readout (ER, LDA from per-architecture class sums) reaching the oracle to within 0.8 in 9/9 runs (`docs/research/readout_report.md` §9). It is the published CCVR/Fed3R principle. So "GeFL-F + recalibration" is a legitimate federated baseline, and on FMNIST and SVHN it plausibly ties the final method (+0.4–0.8 margin). Its CIFAR-10 value is unknown. The readout report itself concludes "the paper must add this baseline" (l. 248-251). It has not been run at full schedule with 3 seeds.

## E. Generalisation

- **Datasets.** Only 4 datasets, all 10 classes at 32×32. The FE is 1 conv block (3 channels at 16×16), or 2 blocks / 10 channels on CIFAR-10. The heads are tiny CNNs.
  - CIFAR-10 LT absolute accuracy is 44–49 (oracle ≈ 50–52), at effective IF 48.
  - No CIFAR-100-LT, ImageNet-LT or any many-class benchmark, where both the collapse mechanism (more rows, fewer holders) and the anchor's F-statistic would be stressed.
- **SVHN.** The mean-anchor method fails (55.89, or 54.41 at T_s = 10, vs HWA+LA 62.45), so a dataset-level gate was introduced *after* the failure. The gate threshold is set from four points (Limitations l. 998).
- **Regimes.**
  - IF ∈ {1, 10, 100} only for the minimal variant on MNIST/FMNIST (K02). Ours-A is only at IF = 100 and IID, and α only 0.5.
  - K = 50/100 runs exist only for MNIST and FMNIST: Ours-A at MNIST K = 100 and FMNIST K = 50/100, with no MNIST K = 50 Ours-A.
  - No SVHN or CIFAR-10 at K > 10 (K12 NOT RUN).
  - Every client participates every round (no partial participation).
- **"Heterogeneous FL."** At K = 10 each of the 10 architectures belongs to exactly one client (`assign_arch`, `core.py:238-241`). The model heterogeneity is 10 small header widths on one shared frozen FE. "FedAvg (per architecture)" is therefore local training, and FedProx/FedALA coincide with it (K07 docstring). The claim generalises to "heterogeneous small heads on a shared tiny FE", not to heterogeneous FL broadly.

---

## Most serious fairness concerns (ranked)

1. **The strongest competing family, classifier recalibration on GeFL-F from federated statistics (CCVR/Fed3R/ER), is missing.** The authors' own diagnostics show it closes 14–20 points on GeFL-F and lands within about 1 point of the final method on FMNIST and SVHN.
2. **No published FL long-tail method is run.** The long-tail "baselines" outside GeFL-F are local training. SVHN and CIFAR-10 have no non-GeFL baseline at all.
3. **Test-set selection everywhere.**
   - There is no validation split.
   - The configuration differs by dataset and regime (T_s, generator branch, BBC, MC variant).
   - The IID headline uses best-round test accuracy, and the CIFAR-10 IID gain disappears under final accuracy.
4. **Asymmetric post-processing and information.** BBC is applied only to the proposed method in the tables, although GeFL-F+BBC is logged. Ours-A consumes richer exact statistics than any baseline is allowed to use.
5. **GeFL-F is not tuned for the long tail beyond LA.** The trivial optimiser fixes the paper's own Lemma 2 implies (no decay / AdamW on rows) were never run at full schedule (K15 NOT RUN).
6. **The CIFAR-10 "IF = 100" is actually IF = 48.** Fixed tail identities across seeds and the narrow regime coverage limit generality.

## Narrowest supported empirical claim

On MNIST and FashionMNIST (and SVHN with the dataset-specific rows branch), 10 classes, IF = 100, Dir(0.5), K = 10–100, with GeFL-F's frozen one-block feature extractor:

- replacing flat FedAvg of the CVAE-F class rows with holder-weighted aggregation, and training heads with client-prior logit adjustment, raises final balanced accuracy over the authors' own GeFL-F reproduction by about 13 (MNIST) to 17 (FMNIST) points, and by 12 on SVHN (3 paired seeds, same splits);
- the anchored generator stack adds a further 4–5 points on MNIST and 2–4 on FMNIST.

It has **not** been shown to beat GeFL-F followed by a federated classifier recalibration, any published FL-LT method, or the GeFL paper's best variants under a non-test-selected metric.

---

## Evidence ledger

Severity: 1 = fatal, 4 = minor. Status: D = demonstrated, S = suspected, U = unresolved.

| ID | Claim and location | Evidence (file:line) | Reasoning | Consequence | Sev | Conf | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| R5-1 | Gains are "ahead of every other method" (abstract l. 37; §6.7 positions against CCVR/CReFF only in prose) | `docs/research/readout_report.md:22-33,236-251`; `core.py:2100-2154`; no `er_bal` in any full runs.jsonl; LEDGER NB13 NOT RUN (l. 591) | GeFL-F oracle 79.2 (FMNIST) and 64.8 (SVHN) vs final method 79.7 and 65.6. A federated ER reaches the oracle within 0.8 (T03, 1 seed, reduced schedule) | The main improvement may largely be reproducible by a known post-hoc recalibration of the baseline | 1 | high | U (strongly indicated) | Run GeFL-F + ER/CCVR (and Fed3R ridge) at full schedule, 3 seeds, all 4 datasets. Report the method also with ER on top. Reuse seeds 0-2, so only the readout pass is needed |
| R5-2 | Long-tail baseline table (§5.3 l. 508-515) represents FL baselines | `core.py:238-241`, `2227-2232`; `K07_kaggle_baselines_cifar10_svhn.py:1-9`; LEDGER l. 20-33, 558-574 | At K = 10, FedAvg grouped and LG-FedAvg heads are single-client. No FedLC/FedRS/CReFF/CCVR/BalanceFL/FedProto/FedDF. SVHN and CIFAR-10 have none at all | Straw-man external comparison | 1 | high | D | Add ≥ 3 FL-LT baselines (e.g. CCVR, FedLC, CReFF), adapted to per-architecture heads on the same frozen FE and splits, plus a heterogeneous KD baseline (FedDF/FedProto) |
| R5-3 | IID: "60.93 on CIFAR-10 … +1.8 over our GeFL-F" (abstract l. 49-50; §5.11 l. 825-828) | `inventory.txt`: Ours-A Ts=10 CIFAR IID bal 56.82 vs bma 60.93; `K01…cifar10/summary.md` GeFL-F bal 56.68; `core.py:780-792` | The gain exists only under the best-round-on-test metric | IID superiority claim on CIFAR-10 is unsupported | 2 | high | D | Report final accuracy (or validation-selected rounds) for all IID claims |
| R5-4 | Final method is a single method (§4.6) | Draft l. 37 vs 48-49, 815-818; T13 table | T_s = 1 for CIFAR-10 LT, T_s = 10 for IID/MNIST/FMNIST. Generator branch, BBC and MC-S/MC-B were chosen after test results. No validation split (`core.py:209`) | Optimistic selection bias of several points on harder datasets | 2 | high | D | Fix one configuration a priori, or select on a held-out validation split, and report that number |
| R5-5 | BBC gives "95.2 / 80.6" (abstract l. 31-32) | `core.py:2409-2415`; `inventory.txt` E20 GeFL-F bbc 79.24 / 60.49 | Post-hoc calibration applied only to the proposed method in reported tables | Inflates margin by ≈ 0.4–1 points relative to a symmetric comparison | 3 | high | D | Report GeFL-F+BBC and HWA+LA+BBC alongside, or drop BBC from headline numbers |
| R5-6 | FSG (CCVR-like) is a weak alternative (§5.7 l. 643-645) | `F01_main_longtail/summary.md` (FSG+LA 87.04 / 73.60); `T02_longtail_minimal.md` omits it; K01 CIFAR-10 FSG 44.17 vs Ours-A Ts=10 44.38 | The closest statistic-based competitor ties the minimal method on MNIST and Ours-A on CIFAR-10 LT, but is excluded from the main table | Selective table composition | 3 | high | D | Put FSG+LA, with and without CSL/BBC, in all main tables |
| R5-7 | GeFL-F fails structurally, "not a tuning issue" (§1 l. 65) | `core.py:599-602` (NOWD/ADAMW/ROWNOWD exist); LEDGER NB38/39 NOT RUN (l. 612-613); E02 quick only | Lemma 2 predicts that removing the coupled decay fixes the collapse. That arm is the obvious tuned baseline and has only a 1-seed quick pass | HWA's contribution vs trivially fixing the optimiser is unknown | 2 | high | U | Run K15 (already prepared, references reused) |
| R5-8 | "IF = 100 … total budget fixed" (§5.1 l. 487) | `core.py:204-210`; recomputed counts (CIFAR-10 head capped at 4800, tail 100) | Effective IF on CIFAR-10 is 48 and the total is 18,485 | Misreported setting | 3 | high | D | Correct the text or rebalance the profile |
| R5-9 | Tail = "four rarest classes" | `core.py:206-210`, `772-773` | Fixed class order across seeds; FMNIST tail includes intrinsically hard *shirt* | Variance understated; method tuned to one tail assignment | 3 | med | D | Randomise the class-to-frequency permutation per seed (or report ≥ 2 permutations) |
| R5-10 | "Reaches the accuracy of its own heads re-fit on pooled real data" (abstract l. 34) | `core.py:2133-2154`; E09 NOT RUN | Oracle is relative to the method's own bodies. No centralised pipeline oracle exists | Wording overstates; readers may read it as "reaches the centralised bound" | 3 | high | D | Rephrase; add a centralised (pooled-data) upper bound |
| R5-11 | "+1.2/+1.6/+0.8 over the best of all ten methods" (abstract l. 48-50; §5.10) | §5.2 parity table (our GeFL-F +0.55 MNIST, +3.24 CIFAR-10 over paper) | Copied paper numbers vs our reproduction; offsets are of the same size as the margins | Not apples-to-apples | 2 | high | D | Compare only with reproduced baselines on the same seeds, or subtract the parity offset |
| R5-12 | §5.3 "More clients" K = 50 entries 86.8 / 87.2 / 87.9 | `docs/PROGRESS.md` claim 4; LEDGER F04 missing arms (l. 652) | No run exists for those cells | Unsourced numbers in the manuscript | 2 | high | D | Delete or run |
| R5-13 | Generality: "heterogeneous FL", K-robustness, regimes (abstract, §5.6) | LEDGER: K12 NOT RUN; Ours-A only at IF = 100/IID; α = 0.5 only; SVHN gate post hoc (l. 829-864) | 4 small 10-class datasets, tiny FE, 1 client per architecture at K = 10, full participation | Claims exceed evidence | 2 | high | D | Add CIFAR-100-LT (or similar), α ∈ {0.1, 1}, partial participation, IF sweep for Ours-A, SVHN/CIFAR at K = 50/100 |
| R5-14 | SVHN `best_mean_acc` comparisons (§5.3 l. 543-545) | `core.py:743-777` | Plain accuracy on imbalanced SVHN test set, combined with best-round selection | bma overstates SVHN by 4–5 points relative to bal | 4 | high | D | Use balanced metric for SVHN |
| R5-15 | CIFAR-10 p-values (§5.11) | `K01_kaggle_cifar10_svhn__cifar10/summary.md:2` ("runs.jsonl not yet copied back") | Unpaired Welch on pasted summaries; no multiplicity control across ~40 arms | Weak inference | 4 | high | D | Recover runs.jsonl (not a rerun); apply Holm within each table |
| R5-16 | DP clipping uses "public" held-out data (§5.6) | `core.py:2269-2272`, `209` | Held-out images come from the same training distribution, so not truly public | Minor optimism in the DP arm | 4 | med | D | State the assumption or use a disjoint public set |

## Questions to the authors

1. What is GeFL-F + ER (or CCVR / Fed3R) at full schedule on seeds 0–2 for all four datasets? What is Ours-A + ER? Only the readout pass is needed: cached FE and heads, no new training.
2. Why are FedLC, FedRS, CReFF (gradient-matched features), CCVR and BalanceFL not run on the same frozen FE with per-architecture heads? `baselines/creff.py` implements Gaussian re-training (CCVR-style), not CReFF. Was it ever used?
3. Please report every IID claim with final (or validation-selected) accuracy. Does any IID margin over your own GeFL-F survive on CIFAR-10 and SVHN?
4. Which configuration is "the" method? Name a single T_s, generator rule and BBC rule fixed before seeing test numbers, and report it on all datasets.
5. Does turning off coupled weight decay (or AdamW / row-wise no-decay) on GeFL-F recover most of HWA's gain (K15)?
6. Why is FSG+LA excluded from the main long-tail table when it ties HWA+LA on MNIST and Ours-A T_s = 10 on CIFAR-10?
7. Will you randomise which classes are tail classes across seeds, and correct the CIFAR-10 effective IF (48)?
8. Can you show one many-class benchmark (CIFAR-100-LT) and α ≠ 0.5, so that "heterogeneous FL under long tails" is supported beyond 10-class digit and clothing data?

*Sources consulted (literature):* [GeFL arXiv 2412.18460](https://arxiv.org/html/2412.18460v2); [CReFF, IJCAI'22](https://ar5iv.labs.arxiv.org/html/2204.13399); [Fed-GraB](https://arxiv.org/pdf/2310.07587); [FedLoGe](https://arxiv.org/pdf/2401.08977); [CLIP2FL](https://arxiv.org/pdf/2312.08648); [Towards Federated Long-Tailed Learning](https://arxiv.org/pdf/2206.14988).
