# Project progress: what is done, what each claim rests on, what is left

*Last updated 2026-10-15 (K16 built; after the adversarial review; the plan is `docs/review/action_plan.md`). Regenerate the numbers with `python experiments/paper_assets.py` and the run ledger with
`python experiments/ledger.py`; update this file by hand only for decisions and status.*

## 0. How to use the records (so nothing is run or drawn twice)

| Record | What it is | Regenerate with |
|---|---|---|
| `experiments/LEDGER.md` | every completed run (1 177), and every job's planned-vs-done status | `python experiments/ledger.py` |
| `python experiments/ledger.py --check <script> [args]` | duplicate check for a planned job; **run before every new notebook** | – |
| `docs/paper_assets/INDEX.md` | every paper table (16) and figure (8), with paper section and sources | `python experiments/paper_assets.py` |
| `docs/paper_assets/numbers.csv` | every number in those tables: mean, sd, seeds, p-value, source folder + arm | same |
| `docs/paper_assets/CHECK.md` | numbers quoted in `docs/paper_draft.md` that differ from the data | same |
| `experiments/results/<job>/` | `summary.md` + `runs.jsonl` per job (raw per-seed records) | from the Kaggle zip |

Tables and figures are generated, never typed: after new results arrive, re-run `paper_assets.py` and every
table, figure and p-value updates. Old hand-made charts (`docs/professor_report/make_figures.py`, f08–f12) use
typed-in numbers and are superseded by `docs/paper_assets/figures/`.

## 1. Claims of the paper and their evidence

Status: **DONE** = full schedule, ≥ 3 seeds, generated table/figure exists. **WEAK** = done but n < 3 or not
significant. **FIX DRAFT** = data exist but the draft quotes a different number. **PENDING** = a run is still needed.

| # | Claim (paper §) | Evidence (table / figure) | Source runs | Status |
|---|---|---|---|---|
| 1 | Under FedAvg + Adam coupled decay, the row-length change decomposes exactly into holder gain and non-holder loss (Theorem 1, an ex-post decomposition, not a prediction); tail rows held by 7–8 of 10 clients collapse; HWA prevents it (§3) | F01_collapse; `experiments/results/T01_collapse_threshold/summary.md` | T01_collapse_threshold (local CPU, MNIST LT, 40 rounds, 1 seed, 4 arms) | DONE for 1 seed (0 violations is expected: the bound is an identity-based inequality, so it is not evidence); λ = 1e-4 ≡ 1e-3, none at 0; HWA tail rows grow 1.17–1.25× vs 0.04–0.10 |
| 2 | Our GeFL-F reproduces the paper's GeFL-F (§5.2) | T01_parity | F09, F10, reference_check | DONE |
| 3 | HWA + LA repairs GeFL-F under the long tail: +13 / +17 (§5.3) | T02_longtail_minimal, F02_longtail | F01 | DONE |
| 4 | Gain holds at K = 50 / 100 (MNIST LT) (§5.3) | T03_clients_minimal, F07_clients | F04 | DONE for GeFL-F / + LA / + HWA + LA / FSG + LA; the CSL arms have no valid post-fix run at K = 50 (draft shows "–") and n = 1–2 at K = 100 |
| 5 | CSL helps in the paper's IID setting (§5.4) | T04_iid_csl | F09, K01 | DONE (MNIST, FMNIST, SVHN K=10; K = 50/100 one seed) |
| 6 | One method for both regimes (§5.5) | T05_combined | F10 | DONE |
| 7 | Gain grows with imbalance (§5.6) | T06_regimes, F06_regimes | K02 | DONE |
| 8 | Minimal-variant ablations; HWA weighting rule matters little (within 1.1 points); DP on counts costs 0.2–1.3 at ε = 1 (§5.6) | T07_k02_ablations | K02 | WEAK: ablation arms have 2 seeds (by design of K02) |
| 9 | Formal DP on the four added statistics only: ε = 8 costs 4.5–6.9 pts; at ε = 2 still above GeFL-F (+9.8 MNIST; +7.1 FMNIST, p = 0.084) (§5.6) | T08_dp, F08_privacy | E26, E25, E20 | DONE; scope stated (generator, heads, consensus sums not covered) |
| 10 | CSL makes a larger synthetic budget useful (Prop. 3) (§5.9) | T09_budget_iid, T10_budget_lt, F05_budget | E16 | DONE (draft now uses the 3-seed table) |
| 11 | Anchored method Ours-A beats GeFL-F by +19.6 / +21.2 (LT), +1.6 / +1.2 (IID, best round) (§5.11) | T11_anchored | E20, E25, F10 | DONE, but **against GeFL-F with its last layer re-fit the margin is only +4.1 (p = 0.01) / +0.5 (p = 0.35) / +0.8 (SVHN, p = 0.29)**: the federated recalibration baseline (B1) decides the framing |
| 12 | Ours-A robust to K (MNIST K=100: 94.3; **FMNIST K=50/100: 79.5 / 79.0 vs GeFL-F 61.9**) | T12_clients_anchored, F07_clients | E22, F04, K03, K08 FMNIST | DONE (FMNIST part is new, was "pending" in the draft) |
| 13 | CIFAR-10: anchored method wins in both regimes; SVHN: anchor fails, rows win (§5.11) | T13_cifar_svhn, T14_svhn_gate, F02, F09 | K01, K08, K10, K11 | DONE; CIFAR-10 K01 tests are Welch (its runs.jsonl was never copied back), everything else paired. CIFAR-10 IID gain is +0.14 at the final round (no claim). The gate threshold was chosen after SVHN; CIFAR-10 (F = 0.055–0.078) is near it with only the anchor branch run |
| 14 | Component ablation of the final method (§5.11) | T15_ablation, F04_ablation | K09 (+ K14 pending) | WEAK until K14: every removal hurts (cluster-level sign test p = 0.031; the old "30/30, p = 2e-9" was pseudo-replicated); KH alone −0.71 / −0.30, not significant |
| 15 | Generator-agnostic (DDPM-F) (§5.8) | T16_ddpm | F11, K04 SVHN | WEAK: FMNIST 1–2 seeds and Ours does not beat GeFL-F on DDPM-F (82.82 vs 83.16); SVHN Ours 70.37 vs 69.77; our DDPM-F is 6.5 below the paper's SVHN CVAE number. Claim reduced to "runs on DDPM-F" |
| 16 | **New: MC-B** (exact covariance where data suffice) | T11, T15, F04 | E32 (+ K14 pending) | PROMISING: FMNIST LT 81.57 (= oracle 81.55, p = 0.03 paired vs Ours-A); **FMNIST IID 85.75, above the paper's best of all ten methods (84.28)**; MNIST LT unstable (seed 2: −2.7) |
| 18 | **New (T03): GeFL-F's long-tail residual is mostly a readout gap; exact-statistics readout (ER) closes it** | `docs/research/readout_report.md` | T03 (CPU, 1 seed, reduced schedule), oracle numbers of all runs | ER validated (9/9 runs within 0.8 of oracle); mechanism open; paper must add the GeFL-F + recalibration baseline (CCVR/CReFF/Fed3R) |
| 17 | VCT beats MixUp (target: GeFL + MixUp 62.67 on CIFAR-10 IID) | T13_cifar_svhn | K13 | **NEGATIVE**: VCT 60.29 vs MixUp 59.98 (IID), 45.65 vs 46.20 (LT); neither reaches 62.67. Move to "what did not work" |

## 1b. Gate 1, readout half: DECIDED (K16 long tail, all 5 datasets; tables T19 / T20 / T21)

Ours (gated) minus GeFL-F, long-tail balanced accuracy, both with the **same** readout (paired, 3 seeds, 95% CI):

| readout | MNIST | FMNIST | SVHN | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|
| heads as trained | +20.6 [16.5, 24.7] | +20.9 [14.1, 27.7] | +15.8 [4.9, 26.7] | +11.8 [8.2, 15.4] | +7.6 [6.3, 8.9] |
| **ER (LDA from exact sums, all clients)** | **+3.9 [3.0, 4.8]** | **+2.2 [0.1, 4.4]** | **+2.0 [0.8, 3.2]** | **+2.6 [2.1, 3.1]** | **+3.4 [3.1, 3.8]** |
| CCVR (all clients) | +3.6 | +2.1 | +2.4 | +2.8 | +3.5 |
| oracle (pooled real, not federated) | +4.1 | +2.6 | +4.0 | +3.4 | +4.5 |
| absolute, Ours + ER | 94.4 | 81.5 | 66.7 | 53.9 | 18.6 |

* **Rule (ER margin ≥ 2, CI excluding 0) passes on 4 of 5 datasets** (SVHN +1.98 misses the threshold by 0.02, CI
  excludes 0) → outcome **A1**: a method paper. The margin is 2–4 points over a recalibrated GeFL-F, not 15–20.
* **The readout matters for our method too.** Ours' heads as trained are near their oracle on MNIST / SVHN / CIFAR-100,
  but not on CIFAR-10 (45.7 as trained, 53.9 with ER): there GeFL-F + ER (51.3) beats Ours as trained. The practical
  recipe is therefore "our generator + an exact-statistics readout"; this answers D4.
* **Balanced data (IID; MNIST, FMNIST, CIFAR-10):** no harm. Heads: +1.7 (MNIST, p = 0.003), +0.9 and +1.0 (n.s.).
  With ER the methods are within 0.6. On CIFAR-10 IID the minimal variant (58.6) is above Ours (57.8).
* CIFAR-100 absolute accuracy is low for every method (15–19% with a readout): the frozen 10-channel extractor is the
  bottleneck. Report it as such.
* The published per-architecture readouts ("own clients") are weak at K = 10 for every method (one client per
  architecture); the all-client readouts need each body broadcast, which is a protocol change to state.
* Still open: Gate 1 second half (K15, NB38 / NB39: HWA vs the optimiser fixes); SVHN / CIFAR-100 IID (NB46 / NB48).

## 1b2. Gate 1, optimiser half: K15 (NB38 / NB39, 16 Oct; tables T17 / T18)

Long tail, balanced accuracy, 3 seeds (references F01 / F04 / K03, same splits):

| | MNIST K=10 | K=50 | K=100 | FMNIST K=10 | K=50 | K=100 |
|---|---|---|---|---|---|---|
| GeFL-F + LA (coupled decay) | 79.2 | 81.4 | 79.5 | 65.2 | 66.5 | 66.6 |
| best optimiser fix + LA (AdamW / no decay on rows / lazy decay) | 86.8 | 84.7 | 81.8 | 76.1 | 74.5 | 71.9 |
| HWA + LA | 87.9 | 86.5 | 83.1 | 76.0 | 75.5 | 73.3 |
| **HWA minus best fix** [95% CI] | +1.1 [-0.9, 3.1] | **+1.8 [0.3, 3.2]** | **+1.3 [0.6, 2.0]** | -0.1 | +0.9 (n.s.) | +1.4 (n.s.) |

* **The diagnosis is confirmed:** every fix that removes the coupled decay on non-holders restores the tail rows
  (tail/head norm 0.03-0.22 -> 0.73-0.97) and recovers most of the loss (+7 to +11 points over GeFL-F + LA).
  AdamW at 1e-2 = AdamW at 1e-3 (86.5 vs 86.4; 76.1 vs 75.9), as Lemma 2 predicts.
* **HWA adds a smaller, K-dependent gain on top:** +1.3 to +1.8 on MNIST at K = 50 / 100 (CIs exclude 0), and the
  same direction on FMNIST without significance (+0.9 / +1.4). Tail recall shows it more clearly (MNIST K = 50:
  74.6 vs 69.9). This is the dilution term (m_c / K) that the optimiser fixes leave in place.
* **Gate 1 rule (HWA beats the best fix by >= 1 at K = 50 or 100):** met on MNIST, not significant on FMNIST, so
  borderline (between B1 and B2). Paper framing: "remove coupled decay" is the main, simple fix for the collapse;
  HWA is a secondary refinement that matters as the number of clients grows. Note the anchored generator has no
  class rows, so HWA matters only for the minimal variant and the SVHN row branch.

## 1d. K17 fair re-test: DONE (all 5 datasets, new seeds 3-5; T22; draft §5.6)

Validation-selected Ours vs each baseline's validation-selected readout (test set, paired, Holm):

| margin over | MNIST | FMNIST | SVHN | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|
| GeFL-F | +3.4 * | +1.0 | +1.5 * | +2.4 * | +4.0 * |
| GeFL-F + AdamW + LA | +2.0 * | +0.5 | +1.5 * | +1.5 (p = 0.051) | +3.0 * |
| strongest baseline | +2.0 * | +0.5 | +1.5 * | +1.5 | +1.6 * |

(* Holm p < 0.05.) P1 (>= 2 over every baseline on >= 4 datasets) FAILS (MNIST only). P2 fails (3 of 5). P3 holds
(CIFAR-10 anchor vs rows: -0.13). P4 fails pooled (0.28; within-dataset 0.82-0.97, exploratory). MNIST run twice:
replicate differences -0.06 +- 0.43 (ER). **Reading:** decay fix + exact readout recover most of the loss; the anchored
generator adds 0.5-2.0. This is scenario 2-3 of the publication plan (method paper with modest, honest gains).

K14 (C1, NB60; 20 of 24 runs, 4 MNIST seed-4 runs lost when the session ended): with K09, 5 seeds. As trained every
removal hurts on MNIST in every seed (LA -3.1, PC -3.9, CSL -1.4, MC -1.1, KH -0.7; p <= 0.05); on FMNIST LA (-5.8)
and CSL (-1.5) significant, MC / KH / PC not. With ER, LA's effect vanishes. MC-B on FMNIST: +1.7 heads, +0.56 ER (n = 2).

## 1c. ERL pilot (E33, 16 Oct): negative, dropped by its pre-registered kill rule

Idea: recompute the balanced LDA last layer from exact sums every round and train only the body under it.
Reduced-schedule CPU pilot, seed 0, long tail (`experiments/results/E33_exact_readout_loop_quick/`):

| balanced accuracy | MNIST | FMNIST |
|---|---|---|
| Ours + ER after training (K16 recipe) | 94.14 | 79.80 |
| Ours + ERL (every round) | 93.90 | 79.41 |
| Ours + ERL5 (every 5 rounds) | 93.82 | 79.35 |
| GeFL-F + ER after training | 91.50 | 78.94 |
| GeFL-F + ERL | 90.03 | 78.40 |

* P1 (heads under ERL match ER) holds by construction. **P2 (ERL beats the after-training readout by >= 1) fails on
  both datasets** (-0.2 / -0.4; GeFL-F -1.5 / -0.5); P3 mixed. A pooled-real refit of the ERL bodies is lower than of
  normally trained bodies (MNIST 89.2 vs 94.9), so training under a fixed balanced readout makes the representation
  slightly worse, not better. Calibration is much worse (ECE 41% on FMNIST).
* Conclusion: computing the readout once, after training, is enough. Code kept (off by default) for the record.
* First attempt collapsed the body (raw LDA weights too large); fixed by rescaling before this pilot.

## 2. Draft corrections

All corrections from `docs/paper_assets/CHECK.md` and items A1–A20 of `docs/review/action_plan.md` are applied in
`docs/paper_draft.md` (15 Oct 2026); `CHECK.md` reports 0 mismatches. The handout (`docs/professor_report/`) was
updated to match. The remaining weaknesses are experimental, not textual: the recalibration baseline (B1), the
optimiser-fix comparison (K15), external long-tail FL baselines, and a validation-selected, pre-specified method run
on new seeds (see the action plan).

## 3. What is still to run (and what is NOT)

The complete, ordered list (built notebooks, measurements, new methods, breadth, privacy) is
`docs/review/action_plan.md` §10 (18 Oct 2026). In short: NB63 (K19 + K20 + the 4 K14 runs NB60 lost) and NB62
(K18) are built; the centralised comparator on SVHN / CIFAR-10 (X3) decides whether a body-side method is worth
building; the signal-fraction gate (X7) is the new method the theory supports. K15, K16 (all of it), K17 and NB60 are
done.

**Superseded: do not run** (their arms are covered by later jobs or were dropped on evidence):
NB9–NB12 (K05, K06 new-generator arms: superseded by K08), NB20–NB21 (K10 hybrid on CIFAR-10: the gate selects the
anchor there), NB23–NB24 (K11 MC-S on CIFAR-10: MC-B is the follow-up), NB25 (K12 SVHN K = 10: K10 already has the
rows + MC + KH), all local E-specs listed as NOT RUN / QUICK ONLY in the ledger, and the PARTIAL arms of K01 / K03 /
F01 / F04 / F09 / F11 / E16 / E24 (arms added to specs after they ran; see the ledger) – except the F04 K = 50 rows if
claim 4 keeps them.

## 3b. Results of 18 Oct (CPU, this container)

- **X4 (P2 pilot, class-balanced Stage-I loss):** positive on every metric, both seeds (validation split). Mean
  balanced-minus-plain: MNIST raw LDA +1.75, one-client ER +0.36, pooled ER +0.08; FashionMNIST +0.45, +0.96, +1.06.
  Prediction (+1 ± 1 FMNIST, 0 ± 0.5 MNIST) held. Full test: K21 (BALFE) in NB68 / NB69. Files:
  `docs/research/math_checks/x4_*.jsonl`.
- **X14 (membership inference on the released class sums):** head classes at chance; exact release leaks for rare
  classes (tail AUC 0.63–0.64, rarest class 0.69–0.73); Gaussian mechanism ε = 8 / 2 brings tail AUC to 0.54–0.58 /
  0.50–0.53. Paper §6.9. Files: `x14_*.json`.
- **X3 on CPU** was stopped (≈ 5 h per dataset-seed here); it runs on GPU in NB69.

## 4. Theory status

Numbering follows the 18 Oct draft (old Theorem 1 → 3, Lemma 2 → 4, Prop. 3 → 11, 4 → 13, 5 → 12, 6 → 14, 7 → 15).
Proofs and numerical checks: `docs/research/federated_vs_centralized_math.md`, `docs/research/math_checks/`.

| Result | Status |
|---|---|
| Lemma 1 (pure label shift) | proved from the partition code |
| Corollary 2 (Stage I is first-order pooled ERM) | first order; surrogate weights within TV 0.009–0.018 at K = 10; regime not perturbative (ητ = 1–17) |
| Theorem 3 (contraction bound) | exact, ex-post; 0 violations in 1,200 row-rounds (T01) |
| Lemma 4 (Adam λ-invariance) | holds for λ‖r‖ ≫ ε; optimiser state persists across rounds |
| Approximation 5 (count-only κ) | 352 / 352 rows with κ < 1 collapsed over 900 rows; Spearman 0.94; prospective SVHN test in K18 |
| Proposition 6 (row fixed points without decay; V_c) | exact in the quadratic row model; simulated ratios match to 2 decimals |
| Proposition 7 (logit corrections are reparametrisations) | exact; K17 real-only arms equal within 0.5 ER points |
| Proposition 8 (alternation) | exact for quadratics; retention measured in a toy (0.97–0.99 with LA); real-pipeline measurement is X6 |
| Proposition 9 (label-independent synthetic data = logit-variance penalty) | exact; K17 sign change with s on the CIFARs |
| Theorem 10 (partition invariance) | exact (trivial); ER = centralised LDA on the same bodies |
| Prop. 11 (synthetic budget) | subsampling model; consistent with E16, K08 |
| Prop. 13 (MC), Prop. 14 (MC-S Gelbrich map) | idealised maps; implemented maps approximate |
| Prop. 12 (mean anchor separates classes only as far as their means) | proved; SVHN F = 0.005–0.013 matches |
| Prop. 15 (MixUp second order) | proved; VCT built on it did not win |
| BatchNorm averaging, best-on-test selection | stated in Limitations (§8) |
