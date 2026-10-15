# Project progress: what is done, what each claim rests on, what is left

*Last updated 2026-10-15 (after the adversarial review; the plan is `docs/review/action_plan.md`). Regenerate the numbers with `python experiments/paper_assets.py` and the run ledger with
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

## 2. Draft corrections

All corrections from `docs/paper_assets/CHECK.md` and items A1–A20 of `docs/review/action_plan.md` are applied in
`docs/paper_draft.md` (15 Oct 2026); `CHECK.md` reports 0 mismatches. The handout (`docs/professor_report/`) was
updated to match. The remaining weaknesses are experimental, not textual: the recalibration baseline (B1), the
optimiser-fix comparison (K15), external long-tail FL baselines, and a validation-selected, pre-specified method run
on new seeds (see the action plan).

## 3. What is still to run (and what is NOT)

**Needed for the paper:**

| Job | Why | Cost | Status |
|---|---|---|---|
| **NB38 / NB39 (K15)**, existing fixes for row collapse vs HWA, K = 10/50/100: AdamW matched (1e-3), no decay on class rows, lazy decay; AdamW 1e-2 at K = 10 | the question a reviewer asks first: is HWA better than just fixing the optimiser? Pre-registered predictions in `docs/collapse_explainer/collapse_theorem_explained.pdf` §8; tables T17/T18 fill in automatically | ~3.3 h each, T4 × 2 (60 runs) | ready in `kaggle_notebooks/`, not run; references reused (0 duplicates) |
| NB36 / NB37 (K14), seeds 3–4 | per-component significance (claim 14) and MC-B stability on MNIST (claim 16) | ~2 h each, T4 × 2 | ready in `kaggle_notebooks/`, not run |
| **B1 ("K16")**: GeFL-F and Ours-A re-run with models saved and the exact readout logged, MNIST / FMNIST / CIFAR-10 / SVHN long tail | the decisive baseline: GeFL-F + federated recalibration (reviewer objection F1) | ~8–12 GPU h | **needs approval (D2)**: re-runs existing configurations because no models were saved before commit 9a16e54 |
| T01 extra seeds (local CPU or Kaggle) | claim 1 has 1 seed | ~1.5 h per seed on CPU | optional; theorem itself is exact, the seeds only add robustness |

**Optional (only if the paper keeps the corresponding claim):**

| Job | Use | Decision |
|---|---|---|
| NB13 (K07) FedAvg / LG-FedAvg on CIFAR-10 + SVHN under imbalance | baseline rows of T02 for the hard datasets | run only if a reviewer-proof baseline table on CIFAR/SVHN is wanted |
| NB26–NB29 (K12) SVHN / CIFAR-10 at K = 50 / 100 | §5.10 "against every GeFL variant" at K = 50/100 on hard data | run only if that claim stays in the paper |
| NB7 / NB8 DDPM-F on CIFAR-10 | claim 15 on a third dataset | low priority |
| K01 CIFAR-10 runs.jsonl | paired tests for T13 CIFAR column | **not a run**: copy the file back from Kaggle if it still exists |

**Superseded: do not run** (their arms are covered by later jobs or were dropped on evidence):
NB9–NB12 (K05, K06 new-generator arms: superseded by K08), NB20–NB21 (K10 hybrid on CIFAR-10: the gate selects the
anchor there), NB23–NB24 (K11 MC-S on CIFAR-10: MC-B is the follow-up), NB25 (K12 SVHN K = 10: K10 already has the
rows + MC + KH), all local E-specs listed as NOT RUN / QUICK ONLY in the ledger, and the PARTIAL arms of K01 / K03 /
F01 / F04 / F09 / F11 / E16 / E24 (arms added to specs after they ran; see the ledger) – except the F04 K = 50 rows if
claim 4 keeps them.

## 4. Theory status

| Result | Status |
|---|---|
| Contraction bound for class rows (Theorem 1, §3) | rewritten; 0 violations in 1,200 row-rounds on the real CVAE-F (T01); λ-invariance confirmed (10⁻⁴ ≡ 10⁻³, none at 0) |
| Prop. 3 (synthetic budget) | stated; consistent with E16, K08 (T_s = 10 hurts final acc. where fidelity is low) |
| Prop. 4 (MC is minimum-displacement), Prop. 6 (Gelbrich map for MC-S) | proved for the idealised map; the implemented map (ReLU, plug-in moments, shrinkage) is approximate, now stated |
| Prop. 5 (mean anchor separates classes only as far as their means) | proved; SVHN F = 0.005–0.013 matches |
| Prop. 7 (MixUp second order) | proved; VCT built on it did not win (claim 17) |
| Known audit points (from the ChatGPT review, verified): BatchNorm running-variance averaging drops between-client variance; best-on-test selection biases `best_mean_acc` (we headline final balanced accuracy) | stated in Limitations (§7) |
| Lemma 2 (Adam λ-invariance) | holds only for λ‖r‖ ≫ ε; per-client optimiser state persists across rounds; now stated |
| Prop. 3 (synthetic budget) model | restated as a subsampling model |
