# Reviewer 4: statistics, benchmarking and reproducibility

**Specialty.** Statistician and experimental ML researcher: benchmarking protocols, paired designs, multiple comparisons, selection bias and reproducibility.

**Manuscript.** `docs/paper_draft.md` ("Rare Classes Vanish from Federated Feature Generators…", GeFL-F under long-tailed clients; proposed Ours-A / ESA).

**Recommendation: major revision. As written it is a weak reject on the evidential standard for the secondary claims. Confidence: high (4/5) for the items marked demonstrated.**

The large headline effects under the long tail hold up when I recompute them from per-seed data, and they do not depend on which GeFL-F replicate is chosen as the reference. Those effects are Ours-A against GeFL-F on MNIST and FMNIST (+19.6, +21.2) and the SVHN rows variant against GeFL-F (+15.1). Almost everything below that tier is statistically fragile or rests on post-hoc selection on the test set. This covers the component-wise claims, the claims to beat the paper's best, the "matches" and "ties", the CIFAR-10 IID win, KH, MC-B and the gate threshold. Several of these claims are contradicted by the raw data.

---

## 0. How to reproduce

All scripts are in `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r4_work/`. They read only `experiments/results/*/runs.jsonl`. The K01 CIFAR-10 values come from `summary.md` because that folder has no jsonl file. Run each script from that directory with `python <script>`.

| script | what it computes |
|---|---|
| `load.py` | loader: (folder, label, dataset, IF, K) → {seed: record} (skips T03 records) |
| `dump.py` → `dump.txt` | per-seed final_bal / best_mean_acc of every configuration |
| `headline.py` → `headline.md` | all headline comparisons: means, sds, per-seed differences, paired t, 95% CI, Welch |
| `k09.py` → `k09.md` | K09 ablation: paired t, Holm, BH, sign tests, swap of the reference replicate, intersection-union test |
| `replic.py` → `replic.md` | same configuration and same seed, different jobs: replicate SD vs seed SD |
| `mult.py` | Holm/BH over the 204 p-values in `numbers.csv` and the ≥ 33 quoted in the draft |
| `power.py` | power and minimum detectable effect of a paired t-test at n = 3 and n = 5 |
| `bestof.py` | optimism of best-over-rounds: best_mean_acc − last-round accuracy, per arm |
| `t03.py` | T03 Spearman over 100 head×class pairs: naive, within-class, class-mean, partial |
| `k02.py` | K02 privacy and ablation arms and the regime trend |
| `dof.py`, `collide.py` | number of runs, methods and settings; whether the loader keys collide |

`git -C /home/user/Fedlearning log --format='%h %ad %s' --date=iso -- experiments/results/T03_exact_readout` gave the prereg timeline in §6.

---

## A. Statistical unit, pairing and pseudo-replication

**What a seed fixes, from the code.**
* `make_partition(..., seed)` uses `np.random.RandomState(seed)`. The split, including the held-out 2000 images, depends only on (dataset, K, frac, IF, alpha, seed). So the same seed gives the same partition for every arm and every job. Pairing by split is valid.
* Initialisation goes through `set_seed(seed)`, `seed+1000` and `seed+2000`. However, `core.py:42` sets `torch.backends.cudnn.benchmark = True` and turns on no deterministic algorithms. Training is therefore **not reproducible** for a fixed seed.
* Within a job, arms share the cached stage-I feature extractor (`key_of(...,1)`) and, where the generator spec matches, the stage-II generator. Across jobs both are retrained.

**The manuscript says "each seed fixes the split and the initialisation for every method" (§5.1).** That is only half true. The split is fixed. The training trajectory is not, and the replicate noise is of the same size as the small effects the paper claims (`replic.py`):

| configuration (same seeds 0-2, different jobs) | jobs | job means | replicate (within-seed) sd | seed sd |
|---|---|---|---|---|
| GeFL-F MNIST LT | 5 | 74.62, 74.24, 74.94, 75.25, 75.31 | 0.83 | 1.45 |
| GeFL-F FMNIST LT | 5 | 58.62, 58.74, 59.13, 58.47, 58.90 | 0.48 | 3.69 |
| Ours-A T_s=10 MNIST LT (E25 vs K09) | 2 | 94.82, 95.18 | **0.44** | 0.26 |
| Ours-A T_s=10 FMNIST LT (E25 vs K09) | 2 | 79.67, 79.79 | 0.32 | 0.36 |
| Ours-A T_s=1 SVHN LT (K08 vs K11) | 2 | 55.89, 55.76 | **0.96** | 1.20 |
| GeFL-F FMNIST IID best_mean_acc | 4 | 82.67–82.96 | 0.13 | 0.29 |

For Ours-A on MNIST, re-running the same seed varies more than changing the seed does. Many key comparisons pair arms across jobs:
* Ours-A (E25) against GeFL-F (E20 / F10);
* KH as E25 against E20;
* MC-B (E32) against K09;
* CIFAR-10 and SVHN as K08 against K01, K10 and K11.

In those comparisons the pairing removes split variance only, not training noise. That is acceptable for 20-point effects and decisive for effects of 0.3–1 point.

**Are the paired tests justified?**
* For the large effects, yes. I recomputed Ours-A T_s=10 on MNIST/FMNIST LT against each of the 5 independent GeFL-F replicates. All give +19.5…+20.6 with p = 0.0025–0.0055 on MNIST, and +20.5…+21.2 with p = 0.010–0.013 on FMNIST. The headline is robust.
* The paired t-test with n = 3 has df = 2. Normality of the differences cannot be checked with 3 points.
* Rank and sign tests cannot reach significance. The minimum two-sided p of a Wilcoxon or sign-flip test is 0.25 at n = 3.

**Pseudo-replication: K09 sign test (demonstrated).** `K09_ablation_analysis.md` reports "all 30 seed-level differences negative … two-sided sign test on the 30: p = 2e-9" (`2*0.5**30 = 1.86e-9`).
* The 30 values are 5 arms × 3 seeds × 2 datasets. Within each (dataset, seed) cell, all 5 differences subtract **the same Ours-A run**, so a lucky reference draw makes all five negative together. The independent units are the 6 (dataset, seed) clusters. The cluster-level sign test gives p = 2/64 = **0.031**.
* Shown directly: replacing the K09 Ours-A reference with the E25 Ours-A run, which has the same label, configuration and seeds, leaves **24/30 negative, not 30/30**.
  * MNIST -CSL, -MC and -KH flip at seed 2, because K09 Ours-A scores 95.20 and E25 Ours-A scores 94.23 on that seed.
  * FMNIST -PC and -KH flip as well.
  * Per-arm p-values rise to 0.27–0.56 for CSL, MC and KH (`k09.md`).
* "Removing any one component lowers accuracy" is an intersection hypothesis. The correct test is intersection-union, which takes the maximum p: **0.106**. A pooled sign test answers a different question.

**Pseudo-replication: T03 Spearman (demonstrated).** Each ρ in T03 uses 100 (head, class) pairs from **one run** (seed 0, reduced schedule 0.4). The pairs share 10 classes and 10 heads, so they are not 100 independent observations. scipy p-values computed on 100 pairs (e.g. 1e-37) would be meaningless. The report quotes ρ without p, which is acceptable. Its post-hoc conclusion, "(1−α) explains the readout gap (ρ 0.33–0.90)", is fragile, however:
* Aggregated to 10 class means, ρ(B, dR) is 0.45–0.52 with p = 0.13–0.19 for the FMNIST LA runs, and 0.41 (p = 0.24) for MNIST Ours-A.
* The partial ρ(B, dR | 1−α) is **negative in all six runs** (−0.05 to −0.23). B carries no information beyond (1−α), which confirms that P3 failed.
* There is one seed per configuration, so nothing can be said about seed variability.

---

## B. Recomputed numbers vs. reported (from `runs.jsonl`)

LT columns use final_bal (last round); IID columns use best_mean_acc. Each cell gives mean ± sd over n = 3. "paired" is a paired t-test by seed; "CI" is the 95% t-interval of the paired difference.

| comparison | recomputed | per-seed diffs | paired p / 95% CI | reported in draft | verdict |
|---|---|---|---|---|---|
| MNIST LT Ours-A T_s10 vs GeFL-F | 94.82±0.57 vs 75.25±1.12 | +21.18 +19.73 +17.80 | 0.0025, [15.4, 23.8] | 94.82 vs 75.25, p=0.002 | ✔ |
| FMNIST LT | 79.67±0.54 vs 58.47±3.59 | +17.16 +21.54 +24.91 | 0.011, [11.6, 30.9] | p=0.011 | ✔ |
| MNIST / FMNIST LT + BBC | 95.17 / 80.55 | | 0.0023 / 0.011 | abstract 95.2 / 80.6 | ✔ |
| CIFAR-10 LT T_s10 vs GeFL-F | 44.38±1.07 vs 33.74±0.51 (summary only) | – | Welch 0.00076 | 44.38 | ✔ (Welch only) |
| CIFAR-10 LT **T_s1** vs GeFL-F | 46.06±1.29 | – | Welch 0.0012 | abstract "46.1" | ✔ but uses a different T_s than the stated final method (T_s=10) |
| CIFAR-10 LT T_s1 + BBC vs FSG | 48.71 vs 44.17 = **+4.54** | – | Welch 0.0038 | "+4.8 (p=0.003)" | ✘ (+4.8 is computed from the rounded "≈49") |
| CIFAR-10 LT T_s10 vs T_s1 | −1.67 | −1.19 −1.65 −2.18 | 0.028 | p=0.03 | ✔ |
| SVHN LT Ours-A T_s10 vs GeFL-F | 54.41±1.13 vs 50.49±2.24 | +2.91 +1.32 +7.54 | **0.17**, [−4.1, 12.0] | not significant (acknowledged) | ✔ |
| SVHN LT rows+MC+KH vs GeFL-F | 65.60±1.50 vs 50.49 | +12.3 +13.9 +19.1 | 0.018 | 65.6 vs 50.5 | ✔ |
| SVHN LT rows vs hybrid PCR+MC-S | +2.71 | +2.15 +1.78 +4.22 | 0.070 | "Welch p = 0.07 … per-seed files not copied back" | stale text: per-seed files exist; paired p = 0.070 |
| MNIST IID Ours-A vs GeFL-F | 97.62±0.06 vs 96.01±0.14 | +1.69 +1.62 +1.49 | 0.0014 | p=0.001 | ✔ |
| FMNIST IID Ours-A vs GeFL-F | 83.96±0.39 vs 82.76±0.33 | +0.86 +1.30 +1.45 | 0.021 | p=0.021 | ✔ |
| FMNIST IID Ours-A vs minimal T_s10 | −0.38 | −0.75 −0.20 −0.18 | 0.18, [−1.18, +0.43] | "−0.4, n.s." | ✔ but see §E (power) |
| CIFAR-10 IID Ours-A T_s10 vs GeFL-F | 60.93 vs 59.10 | – | Welch 0.11 | "+1.8, p≈0.1" | ✔; **last-round: 56.82 vs 56.68, +0.14, p = 0.93** |
| SVHN IID rows+MC+KH vs GeFL-F | 77.07 vs 75.82 | +2.04 +1.14 +0.58 | **0.098** | "+0.8 over the paper's best" | not significant against our own GeFL-F |
| MNIST LT Ours-A vs minimal T_s10 | +4.66 | +6.57 +4.70 +2.72 | **0.052** | "+4.7, positive on every seed" | true, but n.s. |
| FMNIST LT Ours-A vs minimal T_s10 | +0.45 | +0.26 +0.41 +0.67 | **0.065** | "+0.5, positive on every seed" | true, but n.s. |
| **KH**, MNIST LT (E25 vs E20) | +0.80 | +1.24 +0.65 +0.51 | 0.070 | "all 9 positive, sign test p = 0.004" | per setting n.s.; see §B2 |
| KH, FMNIST LT | +0.66 | +0.35 +0.43 +1.19 | 0.133 | | |
| KH, FMNIST IID (best_mean_acc) | +0.43 | +0.07 +0.96 +0.25 | 0.254 | | |
| **MC-B** FMNIST LT vs Ours-A (K09) | 81.57 vs 79.79 | +1.99 +2.19 +1.16 | 0.030 | 81.57, p=0.03 | ✔ (vs E25: p = 0.016) |
| MC-B MNIST LT vs Ours-A | −0.42 | +0.81 +0.63 **−2.70** | 0.75 | "unstable" | ✔ |
| MC-B FMNIST IID vs Ours-A (E25) | +1.78 | +1.74 +1.77 +1.85 | 0.0003 | 85.75 | ✔ (post-hoc variant, §C) |
| Ours-A vs **GeFL-F + oracle last-layer refit**, FMNIST LT | 79.67 vs 78.94 = **+0.74** | −0.11 +1.25 +1.08 | **0.23**, [−1.1, 2.6] | readout report "+0.5" | no significant advantage |
| same, SVHN LT (rows variant) | 65.60 vs 64.82 = +0.78 | +0.30 +0.15 +1.88 | **0.29** | "+0.8" | no significant advantage |
| same, MNIST LT | +3.61 | | 0.028 | "+4.1" | ✔ |
| DP ε=8 / ε=2 cost, MNIST | −4.46 / −9.76 | | 0.006 / 0.002 | 4.5 / 9.8 | ✔ |
| DP ε=8 / ε=2 cost, FMNIST | −6.90 / −14.09 | | 0.022 / 0.002 | 6.9 / 14.1 | ✔ |
| DP ε=2 vs GeFL-F, FMNIST | +7.12 | +3.78 +6.28 +11.29 | **0.084**, [−2.4, 16.6] | "even at ε=2 the method is 7–10 points above GeFL-F" | not significant on FMNIST |
| K02 "HWA weighting rule irrelevant: within 0.4 points" | MNIST uniform-over-holders −0.99 vs Ours (**n = 2 seeds**) | | 0.35 | "within 0.4 points of each other" | ✘ contradicted (spread 1.0 on MNIST); every K02 ablation arm has n = 2 |
| Gate statistic F "≥ 0.14 on every other dataset; any threshold in 0.02–0.1 makes the same choice" | CIFAR-10 F = **0.055–0.063 (IID), 0.066–0.078 (LT)** in `class_sep_F` | | | §4.6, §7 | ✘ contradicted (see C3) |

### B2. Sign test "KH helps in all 9 paired comparisons, p = 0.004"

The arithmetic is right: 9/9 positive gives two-sided 2/512 = 0.0039. The inference has three problems:
1. **Cross-job comparison.** Ours-A comes from E25 and PC+MC from E20. The FE and generator are retrained in each job, and same-seed replicate SD is 0.3–0.9 (§A), comparable to the KH effect (0.4–0.8). Code changes between E20 and E25 are possible confounds (suspected, not shown).
2. **Mixed metrics.** final_bal in two settings and best_mean_acc in the third.
3. **The within-job test is weaker.** In K09, where -KH shares FE and generator with Ours-A, the effect is −0.71 (p = 0.059) on MNIST and −0.30 (p = 0.097) on FMNIST, and neither survives Holm. With the E25 reference instead, KH's sign flips on 3 of 6 seeds.

The defensible statement is "KH: +0.3–0.7, consistent in direction within job, not individually significant."

### B3. Multiple comparisons (`mult.py`)

* `numbers.csv` holds **204 p-values** (191 paired, 13 Welch). 137 are nominally < 0.05, **6 survive Holm** and 129 survive BH.
* The draft text quotes at least 33 p-values. 23 are < 0.05 and 5 survive Holm over that family.
* K09 alone: no component survives Holm over its 10 tests. The smallest adjusted p is 0.085 (FMNIST -CSL). Under BH, 5 of 10 survive at 0.044–0.056 (`k09.md`).
* The paper never says what family it controls. It reports raw p-values next to claims of "significance" and "consistent" effects.

### B4. Power at n = 3 (`power.py`)

* A two-sided paired t-test at α = 0.05 needs a standardised effect **d_z ≥ 3.26** for 80% power at n = 3. It needs 1.68 at n = 5 and 1.00 at n = 10.
* Using the observed SD of the paired differences, the minimum detectable effects at n = 3 are:
  * KH: 1.0 pt (MNIST), 0.56 pt (FMNIST);
  * MC on FMNIST: 0.95 pt;
  * Ours-A vs minimal, FMNIST LT: 0.68 pt;
  * Ours-A vs minimal, FMNIST IID: 1.06 pt;
  * rows vs hybrid on SVHN: 4.3 pt.
* The power to detect the observed effect at n = 3 is 0.22–0.56.
* Every "n.s. / tie / within noise / changes nothing measurable" statement is therefore uninformative. Examples: FMNIST IID −0.4, PCR "+0.22 (p = 0.29)", the K = 50 "within noise (p = 0.47)", "MC never hurts", the HWA weighting rule. None of these is supported by an equivalence test (TOST).
* K09's own follow-up plan has a problem. It says "five seeds (10/10 negative → p = 0.002)", which pools the two datasets. Per component and per dataset, 5/5 negative gives a two-sided sign-test p of **0.0625**. That can never reach 0.05, so the planned K14 cannot deliver per-component significance by a sign test.

---

## C. Researcher degrees of freedom and selection on the test set

### C1. Size of the search (`dof.py`)

* 1150 runs in `runs.jsonl`, plus 27 in the K01 CIFAR-10 summary, gives **1177** (matching the ledger).
* 406 of the jsonl runs are quick passes.
* **103 distinct method labels**, 73 of them at the full schedule, over 20 (dataset, IF, α, K) settings.
* Planned job families: E00–E32, F01–F11, K01–K15, T01–T03 and D01, with 73 experiment scripts.

### C2. No validation split

* `load_dataset` returns only the official train and test splits.
* `make_partition` sets aside 2000 training images (200 per class) as `heldout`. These are used only for the referee classifier, MND and the DP clipping bound (`run_one`, `train_referee`). They are never used for model or hyperparameter selection.
* Every evaluation (`eval_headers`, `Tracker.update`, BBC, the oracle and ER) uses `data["xte"]`, the test set.
* The quick passes that chose which ideas to keep (§5.7) were also scored on the test set.

### C3. Design choices made after seeing test results (git log, demonstrated)

| choice | evidence that it was chosen from test results |
|---|---|
| **BBC gate** (global imbalance > 2) | commit `eaa133e`: "BBC +0.9/+1.5 under LT, −0.45 in IID → gated by exact global imbalance". The gate's input is label-free, but the decision to gate was made from test accuracy. |
| **Generator gate F < 0.05** | commit `b9657e0`: SVHN anchored stack loses. Then commit `6f9e1a3` (after the K10/K11 SVHN test results): "Final method gated by … F: anchor if F ≥ 0.05, HWA rows otherwise". The gated final method was **never run as one pipeline**: K12 (NB25–NB29) shows NOT RUN in the ledger. The SVHN "final method" number 65.6 is the K10 arm picked post hoc. |
| **The gate is not robust as claimed** | The `class_sep_F` stored in every run is 0.005–0.013 (SVHN), **0.055–0.078 (CIFAR-10)**, 0.25–0.36 (MNIST) and 0.40–0.71 (FMNIST). §4.6 and the abstract claim "at least 0.14 on every other dataset, so any value between 0.02 and 0.1 makes the same choice everywhere". That is false: a threshold of 0.06 switches CIFAR-10 IID to rows, and 0.08 switches all CIFAR-10. The chosen 0.05 sits 0.005 below CIFAR-10 IID. (The 0.067 CIFAR value in §4.5 is a *pixel*-space F, while the gate uses feature space.) |
| **MC-B** | commit `360ec6c`: MC-S hurt MNIST LT by −1.56, "all on rare classes (tail −4.2)". MC-B was then designed to shrink tail covariances (E32). It was designed on per-class test recall and has been evaluated on the same test sets only (E32, 3 seeds; K14 not run). |
| **T_s per dataset** | T_s = 10 was chosen from E16 test results. The final method is stated with T_s = 10, but the abstract's CIFAR-10 LT figure (46.1) is T_s = 1, the better of the two on final_bal. At the stated T_s = 10 it is 44.38. |
| **β = 0.5, τ = 1, HWA β = 0.999, k = 256, γ (T03)** | β = 0.5 vs 1 was compared on test (F09). τ and β grids were run in K02. γ = 0.1 was fixed in a preregistration (good). |
| **Which result is "Ours"** | The headline row switches between Ours-A (MNIST/FMNIST), Ours-A T_s = 1 (CIFAR-10 LT in the abstract), the rows variant (SVHN) and MC-B (PROGRESS, "FMNIST IID 85.75 above the paper's best"). That is a maximum over variants, reported per dataset. |

**Consequence.** The "final method" was assembled from 70+ variants, with the best variant per dataset chosen on the test set it is reported on. The improvement against GeFL-F is too large to be an artefact of this selection. Two kinds of claim are not robust to it: claims of beating the paper's best (margins of 0.06–1.6 pt) and component-level claims. A held-out validation split, or untouched new seeds and datasets for the final configuration, is required.

---

## D. Test-set protocol

* `Tracker.update` keeps, **per head**, the maximum plain test accuracy over all evaluations, including stage-I warm-up evaluations, because `tracker.best = s1["tr"].best`. `best_mean_acc` is the mean of these per-head maxima. That is an oracle early stop on the test set for each architecture separately. There are 9 evaluation points for MNIST/FMNIST and 17 for SVHN/CIFAR-10.
* `final_bal` is taken from the last evaluation. The same code applies to all methods (`finish`), so the protocol is identical across arms. The optimism is not.

| arm | best_mean_acc | last-round acc | optimism |
|---|---|---|---|
| CIFAR-10 IID Ours-A T_s=10 | 60.93 | 56.82 | **+4.12** (peak at stage-III round 10) |
| CIFAR-10 IID GeFL-F (summary) | 59.10 | ≈56.68 (final_bal) | ≈+2.4 |
| SVHN IID Ours-A T_s=10 | 74.38 | 72.35 | +2.03 |
| CIFAR-10 LT Ours-A T_s=10 | 48.07 | 44.38 | +3.69 |
| MNIST IID Ours-A / GeFL-F | 97.62 / 96.01 | 97.59 / 95.79 | +0.03 / +0.23 |
| SVHN IID rows+MC+KH / GeFL-F | 77.07 / 75.82 | 76.96 / 75.46 | +0.11 / +0.36 |

Consequences:
1. **The CIFAR-10 IID claim** ("60.93, +1.6 over the best of all ten methods; +1.8 over our GeFL-F") **depends entirely on test-set early stopping.** On the last round, Ours-A T_s = 10 is 56.82 against GeFL-F's 56.68 (+0.14, Welch p = 0.93). It peaks at round 10 and loses 4 points afterwards.
2. All the IID comparisons with the paper use best_mean_acc. That matches the paper's protocol, but the GeFL paper's numbers are single published values with no variance. Our own GeFL-F differs from them by +0.55 (MNIST), −0.47 (FMNIST), **+3.24** (CIFAR-10) and −0.44 (SVHN). The implementation offsets are as large as, or larger than, the claimed margins (+0.06 to +1.6).
3. The long-tail headline claims use final_bal, which is fine. The draft also reports "On the paper's own metric (best_mean_acc) the gain is …" and §5.3 SVHN "66.86 vs 57.36 (p = 0.045)". These best-of-rounds values are inflated by about 3–6 points on SVHN.
4. BBC, the oracle and ER are all scored on the test set. BBC is fitted on synthetic data only (checked in `balanced_bias_calibration`), so it involves no test leakage.

---

## E. Practical significance

* **FMNIST IID:**
  * Ours-A 83.96 against the paper's best 84.28 (−0.32);
  * Ours-A against our minimal variant at T_s = 10: −0.38 (CI [−1.18, +0.43]);
  * the minimal variant against the paper's best: 84.34 vs 84.28, presented as a "tie" or "match".

  All these differences are below the replicate SD (0.13–0.4) and the implementation offset (0.47). Neither "matches" nor "beats" can be established.
* **Against a simple recalibration of GeFL-F** (last-layer refit on pooled real embeddings; T03's federated ER-LDA comes within 0.2–0.8 of it):
  * FMNIST: +0.74 (p = 0.23);
  * SVHN: +0.78 (p = 0.29);
  * MNIST: +3.6 (p = 0.03).

  Most of the gain over GeFL-F on FMNIST and SVHN can be obtained by a known classifier-recalibration principle. The paper does not include this baseline. The readout report itself says "it must".
* **CSL in IID:** +0.4 to +0.8, about 1–2 replicate SDs.
* **KH and MC:** 0.3–1.0 points, at or below the run-to-run SD of the same configuration.
* **SVHN IID "+0.8 over the paper's best":** against our own GeFL-F it is +1.25 (p = 0.098), and against our + CSL it is +0.77 (p = 0.11).

---

## F. T03 preregistration

* **Timeline (git).** `b9f1d27` at 21:32:16 commits PREREGISTRATION.md and the harness. The run log's first line is `21:32:08 loading fmnist`, **8 s before** the commit. The first result is at 21:51:45. The prereg was never edited afterwards: only `b9f1d27` touches the file.
* PREREGISTRATION_2 (H-R′) was committed in `4cfb5ec` at 23:47:30, together with the T03 results. `run_b.log` starts at 23:47:46. So it was written after seeing T03 and before T03b. The prereg is therefore plausibly honest. Git cannot show that the text was written before the run started; at best it is contemporaneous.
* **Scoring**, recomputed from `runs*.jsonl` (`t03.py`):
  * P1 pass: ER ≥ head in all 6 runs and within 0.75 of the oracle.
  * P2 pass: +17.9 / +18.4.
  * P4 pass.
  * P3 fail: MNIST Ours-A ρ(B) = 0.29 < 0.3, and B is never better than (1−α); the partial ρ is negative throughout.
  * T03b: Q1 pass (ρ 0.44–0.69); Q2 fail (τ-norm recovers 26%, 16% and 34% of the gap); Q3 pass.
  * The report scores these honestly.
* **Limits.**
  * There is one seed, at the reduced schedule.
  * The "pre-registered" T03b re-tests a hypothesis generated from the T03 data on the **same seed and configuration**. The report itself says "the T03 numbers reproduce to within 0.2: same seed, same code". It is not an independent confirmation.
  * The falsification rule for H-R′ is conjunctive ("Q1 ρ < 0.1 **and** Q2 < 25% in all runs"), which makes falsification unlikely by design.
  * The ρ values are computed over 100 non-independent pairs (§A).
  * These results support descriptive diagnostics only, not mechanism claims.

---

## Missing uncertainty and reporting gaps

1. No confidence intervals for differences anywhere. Only mean ± sd and p are given, and the sd is over 3 seeds.
2. Single-seed cells in tables: KME-Gen; MNIST K = 50 / 100 IID ("+0.37 over the best in the paper", 1 seed); DDPM-F seed 0.
3. K02 ablation arms have **n = 2**, not 3, and this is not stated.
4. CIFAR-10 K01 has summary statistics only (Welch). §5.11 still says the K08/K10/K11 per-seed files "have not been copied back". They now exist; the paired results are in §B.
5. The paper does not state the family of tests or any correction.
6. Paper baselines are point values with no variance, yet they are compared with three-seed means.
7. `CHECK.md` lists 13 draft numbers that do not match the data. Examples: §5.9 says GeFL-F 61.33 where the data say 59.13, and "2 seeds" where there are 3; CIFAR-10 IID "+CSL 60.02" is really the minimal variant, while + CSL is 59.85.
8. `paper_assets.Store` keys runs without α or `quick`, so 74 records in `_quick` folders collide (IID vs Dir(0.5) at IF = 1). No paper table is affected today, but the key is fragile.

---

## Minimum additional experiments

1. **A validation protocol.** Use a validation split carved from the clients' training data (or the existing held-out 2000 images) for every selection: T_s, the gate threshold, MC vs MC-B vs MC-S, β, τ, and the round chosen for best_mean_acc. Re-report final numbers on the untouched test set. Alternatively, run the frozen final configuration on **new seeds (3–7)** and one unseen dataset (e.g. EMNIST or CIFAR-100-coarse) without further changes.
2. **The gated final method run end to end** (K12) on all four datasets with 5 seeds, at the stated T_s = 10, reporting final_bal. Report CIFAR-10 LT at T_s = 10, or justify T_s = 1 by validation.
3. **≥ 5 seeds (preferably 10) for every component and variant claim** (K09/K14). Analyse with a mixed model (arm fixed effect, seed random effect, job random effect), or with paired differences against a within-job reference. Correct for multiplicity (Holm within each table). Use TOST equivalence bounds (e.g. ±0.5 pt) for every "tie" or "no cost" claim.
4. **Determinism or replicate runs.** Set `cudnn.benchmark = False` and `torch.use_deterministic_algorithms(True)`. Otherwise run ≥ 2 replicates per seed for the small-effect arms and report replicate variance. Put every compared arm in the same job.
5. **A GeFL-F + recalibration baseline**: federated ER-LDA or CCVR/CReFF-style, full schedule, 3–5 seeds, all four datasets.
6. **Last-round (final) metrics next to best_mean_acc in every IID table.** For the comparison with the paper, also re-run the authors' code for more than one seed.

---

## Evidence ledger

Severity runs from 1 (fatal) to 4 (minor). Status is D (demonstrated), S (suspected) or U (unresolved).

| ID | claim (location) | evidence | reasoning | consequence | sev | conf | status | minimum action |
|---|---|---|---|---|---|---|---|---|
| R4-1 | Final method, thresholds and variants are not selected on test (§4.6 "threshold is not tuned"; BBC gate "uses no test data") | git `eaa133e`, `6f9e1a3`, `360ec6c`; no validation split in `core.py`; 103 labels / 1177 runs | Selection (gate, MC-B, T_s, which variant is "ours") was done on the same test sets that are reported | Small margins (beats-paper, ties, components) are biased upward; the large GeFL-F gaps survive | 2 | high | D | Validation split, or frozen config on new seeds and datasets |
| R4-2 | "F ≥ 0.14 on every other dataset; any threshold in 0.02–0.1 gives the same choice" (§4.6, §5.11, abstract) | `class_sep_F` on CIFAR-10 = 0.055–0.078 in all 12 runs | The gate margin on CIFAR-10 IID is 0.005; a threshold of 0.06–0.08 flips CIFAR-10 to rows | Gate robustness claim is false; CIFAR results depend on the exact threshold | 2 | high | D | Correct the claim; report CIFAR-10 under both branches; set the threshold by validation |
| R4-3 | CIFAR-10 IID 60.93, +1.6 over the paper's best, +1.8 over GeFL-F (abstract, §5.11) | `bestof.py`: last round 56.82 vs GeFL-F 56.68 (+0.14, p = 0.93); peak at round 10, then −4.1 | The gain exists only under per-head max over 17 test evaluations | Claim not supported under a non-oracle stopping rule | 2 | high | D | Report last-round or validation-selected numbers |
| R4-4 | "All 30 differences negative, sign test p = 2e-9" (K09 analysis; "every component helps on 30/30" in the readout report) | Shared reference per (dataset, seed); cluster sign test p = 0.031; with the E25 replicate as reference 24/30 negative; IUT p = 0.106 | Dependent observations treated as independent | The per-component claim is not established; overstated by about 7 orders of magnitude | 2 | high | D | Mixed model or IUT; ≥ 5 seeds; within-job references |
| R4-5 | "KH helps in all 9 paired comparisons, sign test p = 0.004" (§5.11) | Cross-job E25 vs E20; per-setting p = 0.07 / 0.13 / 0.25; mixed metrics; within-job K09 p = 0.059 / 0.097; replicate sd 0.3–0.9 | Effect at the noise floor; the comparison mixes jobs | KH contribution unproven | 3 | high | D | Within-job ablation with ≥ 5 seeds |
| R4-6 | "Each seed fixes the split and the initialisation for every method" (§5.1) | `cudnn.benchmark = True`; same-seed reruns differ (sd 0.3–0.96; Ours-A MNIST replicate sd > seed sd) | Pairing removes split variance only; cross-job pairs carry training noise | p-values for sub-1-point effects are unreliable | 3 | high | D | Deterministic mode or replicates; same-job arms |
| R4-7 | Equivalence statements: "tie", "matches", "changes nothing measurable", "within noise", "−0.4 n.s." (§5.3, §5.10, §5.11, §7) | Power 0.22–0.56; MDE 0.6–4.3 pt at n = 3; no TOST | Absence of evidence presented as equivalence | Unsupported | 3 | high | D | TOST with pre-stated margins; more seeds |
| R4-8 | "HWA weighting rule irrelevant, within 0.4 points" (§5.6) | K02: MNIST uniform −0.99 vs Ours; ablation arms n = 2 | Contradicted by data; n not stated | False statement | 3 | high | D | Correct; run 3–5 seeds |
| R4-9 | Multiplicity unaddressed (all tables) | 204 p-values in numbers.csv: 137 nominal, 6 Holm, 129 BH; K09: 0/10 Holm | No family-wise or FDR control stated | Secondary "significant" claims inflated | 3 | high | D | State families; Holm within tables |
| R4-10 | The method's advantage is generator-side (abstract, §4.5) | vs GeFL-F + oracle refit: FMNIST +0.74 (p = 0.23), SVHN +0.78 (p = 0.29) | Most of the gain is matched by known last-layer recalibration | Practical significance over a simple baseline is unestablished on 2 of 3 datasets | 2 | med-high | D (oracle) / U (federated ER at full schedule) | Add a GeFL-F + ER/CCVR baseline, 3–5 seeds |
| R4-11 | IID "+1.2 / +1.6 / +0.8 over the best of all ten methods" (abstract, §5.10) | Paper values are single points with no variance; our GeFL-F offset from the paper is −0.47…+3.24; SVHN vs own GeFL-F p = 0.098 | Cross-implementation comparison with best-of-rounds | Not a valid significance claim | 3 | high | D | Compare against re-run baselines; report CIs |
| R4-12 | CIFAR-10 LT "33.7 → 46.1" (abstract) while the final method is T_s = 10 | T_s = 10 gives 44.38; T_s chosen per dataset | Best-of-variants headline | Overstates by 1.7 pt | 4 | high | D | Report the stated configuration |
| R4-13 | SVHN final method 65.6 (abstract) | K12 (the gated method end to end) NOT RUN; 65.6 is the post-hoc K10 arm | Not a run of the claimed method | Headline is a component-assembled estimate | 3 | high | D | Run K12 |
| R4-14 | MC-B FMNIST LT 81.57 (p = 0.03), "= oracle"; FMNIST IID 85.75 > paper's best (PROGRESS, T11/T15) | Recomputed ✔ (p = 0.030; 0.016 vs E25); designed after MC-S failure on the same test sets; MNIST seed 2 −2.7 | Post-hoc variant; one confirmation job | Promising but exploratory | 3 | med | D | Validate on fresh seeds and datasets |
| R4-15 | DP: "even at ε = 2 … 7–10 points above GeFL-F" (§5.6) | FMNIST +7.12, p = 0.084, CI [−2.4, 16.6] | Not significant on FMNIST | Qualify | 4 | high | D | Report CI; more seeds |
| R4-16 | T03 mechanism ρ (readout report) | 100 non-independent pairs from 1 run; class-level ρ p = 0.13–0.24; partial ρ(B, dR \| 1−α) < 0; T03b re-uses the same seed | Pseudo-replication; HARK-adjacent re-test | Diagnostic only | 3 | high | D | Multiple seeds; cluster-level analysis |
| R4-17 | T03 "written before any T03 run" | prereg commit 21:32:16; run.log 21:32:08 | 8-s precedence of the run; content cannot be verified as prior | Minor; scoring is honest | 4 | high | D | Commit preregs before launching; timestamp hash |
| R4-18 | Stale and mismatched numbers (§5.9 "2 seeds", 61.33; CIFAR IID 60.02 label; "per-seed files not copied back") | `CHECK.md` (13 rows); `dump.txt` | Draft is not synchronised with the data | Reproducibility of the text | 4 | high | D | Regenerate all numbers from `paper_assets.py` |
| R4-19 | CIFAR BBC "+4.8 over FSG" (§5.11) | +4.54 from per-seed bbc_bal (48.71 vs 44.17) | Computed from the rounded "≈49" | Minor overstatement | 4 | high | D | Use exact values |
| R4-20 | Large LT gains vs GeFL-F (MNIST/FMNIST +19.6/+21.2; SVHN rows +15.1) | Robust to all 5 GeFL-F replicates (p 0.0025–0.013) | — | **Supported** | – | high | D | Add CIs |

---

## Questions to the authors

1. Which data were used to choose T_s = 10, β = 0.5, the F threshold 0.05, the BBC gate, MC-B's shrinkage and the per-dataset variant? Was any of these chosen on data other than the test set?
2. The stored `class_sep_F` on CIFAR-10 is 0.055–0.078. How do you reconcile this with "≥ 0.14 on every other dataset"? What does the method do on CIFAR-10 at thresholds 0.06–0.1?
3. Will you report last-round (or validation-selected) accuracy for every IID comparison? Does the CIFAR-10 IID claim survive?
4. Why is the stated final configuration (T_s = 10) not the one reported for CIFAR-10 LT in the abstract?
5. Will you add GeFL-F + federated recalibration (ER-LDA / CCVR) at the full schedule? On FMNIST and SVHN your method's advantage over the oracle refit is +0.7–0.8 and not significant.
6. Why is training non-deterministic (`cudnn.benchmark = True`)? Can all compared arms be put in one job, or can replicate variance be reported?
7. What family does each reported p-value belong to, and what correction applies? How will K14 reach per-component significance when a 5/5 sign test cannot fall below p = 0.0625?
8. Why do the K02 ablation arms have only 2 seeds, and why does the text say the HWA weighting rules agree "within 0.4" when MNIST differs by 1.0?
9. Has the gated final method (K12) been run end to end on any dataset?
10. Can the GeFL comparison use re-runs of the authors' code with several seeds, instead of single published values?
