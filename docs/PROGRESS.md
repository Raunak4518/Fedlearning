# Project progress: what is done, what each claim rests on, what is left

*Last updated 2026-10-14. Regenerate the numbers with `python experiments/paper_assets.py` and the run ledger with
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
| 1 | Under FedAvg + Adam coupled decay, class rows contract by an exact bound; tail rows collapse even when held by ≥ K/2 clients (§3) | F01_collapse; T01 analysis | T01_collapse_threshold (local CPU, MNIST LT, 40 rounds, 1 seed) | DONE for 1 seed (theorem holds in every round); see §4 |
| 2 | Our GeFL-F reproduces the paper's GeFL-F (§5.2) | T01_parity | F09, F10, reference_check | DONE |
| 3 | HWA + LA repairs GeFL-F under the long tail: +13 / +17 (§5.3) | T02_longtail_minimal, F02_longtail | F01 | DONE |
| 4 | Gain holds at K = 50 / 100 (MNIST LT) (§5.3) | T03_clients_minimal, F07_clients | F04 | FIX DRAFT: K = 50 rows for +CSL, FSG+CSL, MIX have **no run** (draft quotes 86.8 / 87.2 / 87.9 with no source); K = 100 +CSL n = 2 |
| 5 | CSL helps in the paper's IID setting (§5.4) | T04_iid_csl | F09, K01 | DONE (MNIST, FMNIST, SVHN K=10; K = 50/100 one seed) |
| 6 | One method for both regimes (§5.5) | T05_combined | F10 | DONE |
| 7 | Gain grows with imbalance (§5.6) | T06_regimes, F06_regimes | K02 | DONE |
| 8 | Minimal-variant ablations, HWA weighting irrelevant, DP on counts cheap (§5.6) | T07_k02_ablations | K02 | WEAK: ablation arms have 2 seeds (by design of K02) |
| 9 | Formal DP on anchoring statistics: ε = 8 costs 4.5–6.9 pts, still ≫ GeFL-F (§5.6) | T08_dp, F08_privacy | E26, E25, E20 | DONE |
| 10 | CSL makes a larger synthetic budget useful (Prop. 3) (§5.9) | T09_budget_iid, T10_budget_lt, F05_budget | E16 | FIX DRAFT: long-tail table now has 3 seeds (GeFL-F 59.13, not 61.33; Ts=10 79.23, not 79.33) |
| 11 | Anchored method Ours-A beats GeFL-F by +19.6 / +21.2 (LT), +1.6 / +1.2 (IID) (§5.11) | T11_anchored | E20, E25, F10 | DONE |
| 12 | Ours-A robust to K (MNIST K=100: 94.3; **FMNIST K=50/100: 79.5 / 79.0 vs GeFL-F 61.9**) | T12_clients_anchored, F07_clients | E22, F04, K03, K08 FMNIST | DONE (FMNIST part is new, was "pending" in the draft) |
| 13 | CIFAR-10: anchored method wins in both regimes; SVHN: anchor fails, rows win (§5.11) | T13_cifar_svhn, T14_svhn_gate, F02, F09 | K01, K08, K10, K11 | DONE; CIFAR-10 K01 tests are Welch (its runs.jsonl was never copied back), everything else paired |
| 14 | Component ablation of the final method (§5.11) | T15_ablation, F04_ablation | K09 (+ K14 pending) | WEAK until K14: every removal hurts on 30/30 seeds, but no single component survives Holm with n = 3 |
| 15 | Generator-agnostic (DDPM-F) (§5.8) | T16_ddpm | F11, K04 SVHN | WEAK: FMNIST 1–2 seeds; SVHN shows +CSL +0.5 (p = 0.045); our DDPM-F is 6.5 below the paper's SVHN CVAE number |
| 16 | **New: MC-B** (exact covariance where data suffice) | T11, T15, F04 | E32 (+ K14 pending) | PROMISING: FMNIST LT 81.57 (= oracle 81.55, p = 0.03 paired vs Ours-A); **FMNIST IID 85.75, above the paper's best of all ten methods (84.28)**; MNIST LT unstable (seed 2: −2.7) |
| 17 | VCT beats MixUp (target: GeFL + MixUp 62.67 on CIFAR-10 IID) | T13_cifar_svhn | K13 | **NEGATIVE**: VCT 60.29 vs MixUp 59.98 (IID), 45.65 vs 46.20 (LT); neither reaches 62.67. Move to "what did not work" |

## 2. Draft corrections needed (from `docs/paper_assets/CHECK.md`)

* §5.3 "More clients" table: delete or re-source the K = 50 entries 86.8 / 87.2 / 87.9 (no run exists); K = 100 entries
  are now 83.04 (n = 2), 86.34 (1 seed), 85.29 (n = 2).
* §5.9 long-tail budget table: replace the 2-seed numbers by the 3-seed ones in `tables/T10_budget_lt.md`.
* §5.8 DDPM-F: "Ours 83.10" is now 82.82 ± 0.39 over 2 seeds.
* §5.11 CIFAR-10/SVHN: "tests are unpaired because the per-seed files have not been copied back" is now true only for
  K01 CIFAR-10; K01 SVHN, K08, K10, K11 are paired (p-values in `tables/T13`, `T14`).
* §5.11 "Pending" list: FMNIST K = 50/100 (done: claim 12), VCT (done: negative, claim 17), formal DP (done) and the
  component ablation (done, K09) are no longer pending.
* §3: DONE. The K/2 theorem was wrong in our own setting (every tail class has ≥ 7 of 10 holders and still
  collapses). Rewritten as Theorem 1 (exact contraction bound) + Lemma 2 (Adam's coupled decay is λ-invariant) in
  `paper_draft.md` §3.1–3.2 and `paper_single_idea.md` Prop. 1 / A.1; abstract, contributions and related work updated.

## 3. What is still to run (and what is NOT)

**Needed for the paper:**

| Job | Why | Cost | Status |
|---|---|---|---|
| NB36 / NB37 (K14), seeds 3–4 | per-component significance (claim 14) and MC-B stability on MNIST (claim 16) | ~2 h each, T4 × 2 | ready in `kaggle_notebooks/`, not run |
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
| Prop. 4 (MC is minimum-displacement), Prop. 6 (Gelbrich map for MC-S) | proved in the draft |
| Prop. 5 (mean anchor separates classes only as far as their means) | proved; SVHN F = 0.01 matches |
| Prop. 7 (MixUp second order) | proved; VCT built on it did not win (claim 17) |
| Known audit points (from the ChatGPT review, verified): BatchNorm running-variance averaging drops between-client variance; best-on-test selection biases `best_mean_acc` (we headline final balanced accuracy) | to state in Limitations |
