# Action plan after all reviews (15 October 2026)

**Built from every review we have:**
- our ten independent reviewers and the cross-examined dossier (`docs/review/adversarial_review_dossier.md`, issues F1–F20);
- GPT's package: 43 issues and 100 simulated reviews (`docs/review/external_gpt/`; comparison in `docs/review/gpt_review_comparison.md`).

Every distinct request from both sources is listed below once, with its source IDs. Nothing is run until you approve
it. Every new job must pass `python experiments/ledger.py --check` first (`CLAUDE.md`).

**Names used.**
- GeFL-F: the original method.
- Ours-A: the anchored method.
- Minimal variant: holder-weighted averaging + logit adjustment + consensus labels.
- Readout refit / ER: a last layer computed from exact sums of real features.
- K14 / K15: the two prepared Kaggle jobs, not yet run.

---

## 0. What all the reviews agree on

Both review sets reach the same verdict: **reject as submitted**, confidence 4/5. Three things are behind it, and
**two experiments decide most of the outcome**:

1. **Recalibration baseline (F1; GPT EXP-02, raised by 30+ of its reviewers).** Does Ours-A still beat GeFL-F once *both*
   get a last layer computed from real-data statistics? This decides whether the paper is a **method paper** or an
   **analysis paper**.
2. **Optimiser fix vs holder-weighted averaging (F3; GPT EXP-01 / K15).** Does simply removing or decoupling weight decay
   repair GeFL-F as well as holder-weighted averaging does? This decides whether holder-weighted averaging is a contribution.

Everything else is either:
- text that must be corrected however those two come out (§2), or
- evidence that strengthens whichever paper we end up writing (§4–§6).

---

## 1. Decisions only you can make (needed before any compute)

| # | Decision | Options | My recommendation |
|---|---|---|---|
| D1 | Target venue and deadline | (a) NeurIPS/ICML/ICLR main track; (b) TMLR or AISTATS; (c) FL workshop first, then a journal (IEEE TMC/TNNLS) | Decide after Gate 1 (§3). Plan for (b)/(c) now: both reviews say the current contribution is below main-track level |
| D2 | Approve re-running existing configurations to log the readout refit | Required because **no trained models were saved**, so the refit cannot be computed from old runs. This breaks the no-duplicate rule on purpose | Approve, and add model saving (task A12) so this never happens again |
| D3 | Headline metric | final-round balanced accuracy, or the same after post-hoc bias calibration | Final-round balanced accuracy for every method. Report bias calibration as a separate column, applied to **all** methods |
| D4 | Which variant is "the method" | Ours-A with 10 synthetic epochs; or a rule; or MC-B as the main variant | Fix it before the new runs: Ours-A, one synthetic-epoch rule chosen on validation data, the gate executed in code. MC-B is a secondary variant |
| D5 | Many-class dataset (CIFAR-100-LT) | Needs code work: the class count is hard-coded at 10 in places | Only if D1 = main track. Otherwise list it as a limitation |

---

**Decided 15 Oct 2026** (your answers; your "D1" answered the re-run question, which is D2 here):
- **D1 venue:** aim as high as possible (main track: NeurIPS / ICML / ICLR). Gate 1 still decides whether that is
  realistic; the Phase C/D generality evidence is therefore needed, not optional.
- **D2 re-runs:** approved. Built as job **K16** (`experiments/specs/K16_kaggle_recalibration.py`, notebooks
  NB40–NB48). Models are now saved to their own folder (`models/<job>/`, zipped separately as
  `<folder>__models.zip`) and any saved run can be re-evaluated without training (`--from_models`).
- **D3 metrics:** report both final-round balanced accuracy and the paper's best-round metric, plus metrics not used
  so far: macro-F1, G-mean of class recalls, worst architecture, spread of class recalls, accuracy on each head's own
  clients' label distribution, and calibration (ECE, NLL, Brier). All are now logged for the heads and every readout.
- **D4 the method:** not yet answered (the question was unclear). K16 uses the method as it now runs in code:
  GATE + MC + KH + LA + CSL, T_s = 10 on MNIST / FashionMNIST and 1 elsewhere. C4 must still fix the T_s rule on
  validation data and run it on new seeds.
- **D5 many-class dataset:** yes. CIFAR-100-LT is implemented (100 classes; IF = 100 gives 10,926 images, 480 to 5
  per class, the standard shape) and is part of K16.

## 2. Phase A: corrections with no compute (do first, about 1–2 days of writing)

Each item fixes a statement that is demonstrably false or overstated. Source IDs: F = our dossier, R# = our reviewers,
GPT = GPT issue ID.

| ID | What to change | Where | Source |
|---|---|---|---|
| A1 | Gate statistic: CIFAR-10 is 0.055–0.078, not "≥ 0.14". Delete "any threshold 0.02–0.1 gives the same choice". Call the gate a heuristic chosen after the SVHN failure | abstract l.42; §4.6 l.465 | F6; R3, R4, R5, R8, R10; GPT THEORY-05 and GATE-02 repeat the error |
| A2 | Delete "a class held by every client cannot be contracted". Call the bound a *conditional decomposition* with an ex-post threshold, and stop treating 0 violations as evidence | §3.1 l.125; §3.2; collapse-explainer PDF; handout Part 2 / 7.1 | F4; R2; GPT TH-01 |
| A3 | Restrict Lemma 2 to λ‖r‖ ≫ ε. Note that a single step can exceed η and that optimiser state persists | §3.1 l.133–139 | R2; GPT TH-02; GPT R022, R024, R025, R033, R034 |
| A4 | Correct moment correction and the covariance version: "exact" and "minimum displacement" hold for the idealised map only. State the ReLU, clamping and subspace limits | §4.5 l.331–372 | F4; R2; GPT MATH-01, MATH-02; R027, R028, R036 |
| A5 | Kernel herding: the risk bound is conditional, the loss of a ReLU head is not in the space, so call it motivation. Drop "Frank–Wolfe" | §4.5 l.400–404 | F4; R2; GPT MATH-03; R016, R029 |
| A6 | Synthetic-budget proposition: fix the variance model, or call it subsampling | §4.4 l.257–274 | R2; GPT THEORY-04 |
| A7 | Replace the K = 50/100 rows (they quote runs set aside after the bug fix) with the post-fix values. Remove K = 50 cells that have no valid run | §5.3 l.520–532 | F10; R5, R9; GPT STAT-03 |
| A8 | Abstract: holder-weighted averaging gives +7.0 on CIFAR-10, not "+12 to +20" on all four. Use T_s = 10 numbers for CIFAR-10 LT (44.38), or state the per-dataset rule. Use exact values (+4.54, not +4.8) | abstract; §5.11 | F10; R4, R9 |
| A9 | Replace "within 0.4 points" (the weighting rules differ by 1.0 on MNIST, and n = 2) | §5.6 l.602 | R4; GPT HWA-01 |
| A10 | Cost section: report Ours-A (2.2–2.8× slower; kernel herding 83–88% of Stage III), not the minimal variant's +1.9%. Add a full byte table per variant, including the covariance version's D×D release and the readout statistics. Specify the consensus-label protocol (who generates the pool, what is sent). Remove "1/3000" | §5.12; §4.3 l.243–253; handout | F13; R3, R7; GPT COMM-01, SYS-03; R039, R085, R086 |
| A11 | Privacy wording: "DP for the four added statistics only, per example, add/remove neighbours, trusted noiser; generator, heads, consensus sums and gate counts are not covered". Delete "formal privacy is cheap". State that secure aggregation is simulated | abstract l.51; §5.6; §4.7 | F12; R6; GPT PRIV-01…05; R081–R083, R089, R097 |
| A12 | **Code:** save the trained heads, feature extractor and generator per run, so future readouts and baselines need no re-runs | `experiments/core.py` `run_one` | new (lesson from D2) |
| A13 | **Code:** implement the gate as a real runtime switch on `class_sep_F`, or describe the per-dataset variant honestly | `core.py` stage II | F5; GPT ALG-01; R037, R090, R099 |
| A14 | **Code:** add `torch.backends.cudnn.deterministic = True` (and `benchmark = False`) as an option. Record the git commit hash in every `runs.jsonl` record. Pin `requirements.txt` | `core.py:42`; `requirements.txt` | F15; R4, R9; GPT REPRO-01 |
| A15 | Regenerate every summary from `runs.jsonl`: E25's `summary.md` shows only 1 of its 4 settings, E20's says n = 2 where there are 3, and K09's `runs.csv` lacks the −PC arms | `experiments/results/*/summary.md`; `paper_assets.py` | GPT STAT-03, EXP-06; R077; R8 |
| A16 | Statistics in text: drop the "30/30, p = 2e-9" sign test (use the cluster-level test, p = 0.031) and the "KH 9/9, p = 0.004" claim. Add 95% CIs of paired differences. State the test family and apply Holm within each table. Do not claim "tie"/"matches" without an equivalence test | §5.11; K09 analysis; handout 5.6 | F9; R4; GPT STAT-01, STAT-04; R072, R073, R079 |
| A17 | Balanced-setting (IID) tables: put final-round accuracy beside the paper's best-round metric. Withdraw the CIFAR-10 "+1.6 / +1.8" claim (+0.14 at the final round). Label paper numbers as single published values | §5.10; abstract l.48–51 | F8; R4, R5; GPT STAT-02, EXT-01; R065, R074 |
| A18 | Related work: add DP-MERF (= our KME-Gen), FedRoD (= logit adjustment with client counts), FedVTC (a statistics-conditioned generator for heterogeneous FL), FedNPC (post-hoc calibration for federated long tails), Fed3R, FedNCM, cwFedAvg, FedProto's holder rule, f-CLSWGAN / Delta-encoder, FedFTG / FedCG. Remove "not used before" claims and KME-Gen as a contribution | §6; contributions l.69–90 | F2; R1, R10; GPT NOV-01…04; R011–R020 |
| A19 | Scope statements: say that at 10 clients each architecture has one client; tail classes are held by 6–8 of 10 clients; CIFAR-10's real imbalance is 48:1; the tail classes are always 6–9; and list the BatchNorm and frozen-extractor limits | §5.1; §7 | F14; R3, R5; GPT EXP-05, SYS-02; R041–R046, R057 |
| A20 | Report worst-class accuracy and tail recall in the main tables (MC-B FashionMNIST: worst class 33.2) | main tables | F19; GPT METRIC-01 |
| A21 | Update the handout and its PDF, the progress document, and `docs/diagnostics.md` (old K/2 bound) to match A1–A20 | docs | F16; GPT BR-01 |

---

## 3. Phase B: the two deciding experiments (Gate 1)

### B1 — Recalibration baseline job K16 (approved; built 15 Oct 2026, notebooks NB40–NB48)

- **Arms**, on all four datasets, seeds 0–2, full schedule, LT and IID:
  - GeFL-F;
  - GeFL-F + logit adjustment;
  - minimal variant;
  - Ours-A (T_s fixed per D4);
  - SVHN class-row branch.
- **Readouts computed on every run's trained heads, in the same pass, without retraining:**
  - (i) the heads as trained;
  - (ii) bias calibration;
  - (iii) ER from exact sums (per-architecture LDA, already in the code as `er_bal`);
  - (iv) **CCVR-style** refit restricted to the architecture's own clients (the published protocol);
  - (v) **Fed3R-style** ridge readout from the same sums;
  - (vi) the pooled-data oracle (diagnostic only).
- Optional: FedNPC, if its main paper confirms it fits heterogeneous heads.
- **Cost:** about 8–12 GPU hours on Kaggle (MNIST/FMNIST about 4 h; CIFAR-10/SVHN about 6 h). Models are saved (A12).
- **Ledger:** all arms except readouts (iv) and (v) duplicate existing runs. That is deliberate (D2); the new information is the readouts.

### B2 — K15, already built (`kaggle_notebooks/NB38`, `NB39`; 0 duplicates; about 7 GPU hours)

- Matched AdamW (decay 1e-3); no decay on the class rows only; lazy decay; each with logit adjustment.
- Compared against holder-weighted averaging + logit adjustment at 10 / 50 / 100 clients.
- Readout statistics are now logged automatically.

### Gate 1: what we write depending on the results (fixed now, before seeing them)

| Result | Then |
|---|---|
| Ours-A + ER beats GeFL-F + ER by ≥ 2 points on ≥ 3 of 4 datasets (paired, 95% CI excludes 0), **and** holder-weighted averaging beats the best optimiser fix by ≥ 1 point at 50 or 100 clients | **Method paper**, with honest margins over a recalibrated baseline. Target TMLR / AISTATS, or a main track if the Phase D generality results are strong |
| Ours-A + ER beats GeFL-F + ER by ≥ 2 on ≥ 3 datasets, but an optimiser fix ≈ holder-weighted averaging | Method paper without holder-weighted averaging as a contribution. Present the decay pitfall as a finding and "remove coupled decay" as the fix |
| Margin < 2 on 2 or more datasets | **Analysis paper:** "Where generator-based heterogeneous FL loses rare classes". Contents: the readout-vs-representation decomposition, the decay pitfall, exact readout as the practical fix, and the generator stack as a measured +2–4 representation gain. Target TMLR or an FL workshop, then IEEE TMC/TNNLS |

The CIFAR-10 result carries the most weight. Ours-A's own readout gap there is 6.2, the largest, so the generator may
matter most on the hardest data.

---

## 4. Phase C: evidence that strengthens either paper (after Gate 1)

| ID | Experiment | Answers | Cost | Ledger |
|---|---|---|---|---|
| C1 | **K14** (built; seeds 3–4 for the leave-one-out ablation and MC-B), plus within-job references | per-component significance; MC-B stability (F9; GPT EXP-03, EXP-07, EXP-08; R068) | about 4 GPU h | 0 duplicates (checked earlier) |
| C2 | Head-stage controls using saved generators (A12): kernel herding vs a random 1-of-4 subset and vs all 4n candidates; moment correction mean-only and spread-only; RFF dimension {1536, 6144, 12288} × pool multiplier {2, 4, 8} | whether the herding and moment-correction mechanisms are real (F11; R8; GPT R053–R055) | about 2 GPU h | new arms; check |
| C3 | Factorial: mean-anchored generator × moment correction × T_s ∈ {1, 10} | the interaction R8 found (the anchor helps only together with moment correction) | about 3 GPU h | partly duplicates E20/E25 arms; reuse those cells |
| C4 | **Final method frozen and run end to end:** gate in code (A13), T_s rule and thresholds chosen on the 2,000 held-out images as a validation split, then run on **new seeds 3–5** on all four datasets (this is K12 rebuilt) | "one pre-specified method" (F5; R3, R4; GPT ALG-01, R037, R099) | about 8 GPU h | new seeds; no duplicates |
| C5 | Class-row branch on CIFAR-10 (LT and IID) | whether the gate's choice matters near its threshold (F6) | about 4 GPU h | new arms |
| C6 | External baselines on the frozen extractor, per-architecture heads: FedLC- or FedRoD-style head loss (= our logit adjustment), FedRS, CCVR (from B1), FedProto, FedDF/FedMD-style distillation, FSG + LA in every table | "ahead of every other method" (F7; R5; GPT R014, R050) | about 4–6 GPU h (head stage only) | new arms |

---

## 5. Phase D: scope (only if Gate 1 gives a method paper aimed at a strong venue)

| ID | Experiment | Answers | Cost |
|---|---|---|---|
| D-1 | Several clients per architecture on CIFAR-10 and SVHN (K = 50) | the "heterogeneous FL" claim (F14; R3; GPT R042) | about 6 GPU h |
| D-2 | Partial participation (10–20% of clients per round) at K = 100; tail classes held by only 1–2 clients | realistic FL; the "rare at few sites" motivation (R3; GPT R043, R044, R087) | about 6 GPU h |
| D-3 | Imbalance 10 and 50; Dirichlet 0.1 and 1.0; random choice of which classes are rare, per seed | regime breadth (R5; GPT R063) | about 6 GPU h |
| D-4 | CIFAR-100-LT, or another dataset with many classes (needs D5 code work) | generality; the gate on an unseen dataset (R5, R10; GPT R006, R064) | about 2 days of code plus 10 GPU h |
| D-5 | DCGAN-F / DDPM-F under the long tail | whether "rare classes vanish from federated generators" holds beyond CVAE-F (R1; GPT NOV-02) | about 6 GPU h |

---

## 6. Phase E: privacy, systems, theory follow-ups (small)

| ID | Task | Source | Cost |
|---|---|---|---|
| E-1 | Membership attack on the real trained extractor (exact, ε = 8, ε = 2) | F12; R6 | CPU, < 1 h, needs saved models (A12) |
| E-2 | Batch kernel herding (blockwise) to remove the per-sample host sync; re-measure time and memory | F13; R7 | code, about half a day |
| E-3 | T01 extra seeds, plus logging of the angle between client rows (cancellation vs norm loss) and of λ ∈ {1e-6, 1e-7} | R2; GPT R024, R031 | CPU, about 1.5 h per seed |
| E-4 | A model of holder gain and non-holder loss that predicts the equilibrium row norm in advance | R2; GPT R007, R022 | theory, 1–2 days; optional |

---

## 7. Phase F: submission readiness

| ID | Task | Source |
|---|---|---|
| F-a | Tag one canonical commit. Merge the working branch into `main` (via a pull request, if you want one), or at least document which branch is current | GPT BR-01; R080 |
| F-b | Anonymised LaTeX manuscript in the venue's format, with the checklist and an anonymised supplement (strip repository and owner metadata) | GPT FMT-01; R096 |
| F-c | One command that regenerates every table and figure from `runs.jsonl` at the tagged commit; `CHECK.md` must be empty | GPT REPRO-01; R071, R080 |

---

## 8. Recommended order and total cost

1. **This week, no compute:** A1–A21 (text and code fixes) and decisions D1–D4.
2. **Gate 1:** B1 (about 10 GPU h) and B2/K15 (about 7 GPU h), run in parallel on two Kaggle sessions. Decide the paper type from the table in §3.
3. **Then:** C1, C2, C4, C6 (about 18 GPU h). C3 and C5 if a method paper.
4. **Only for a strong-venue method paper:** Phase D (about 30+ GPU h, plus code for many classes).
5. **Before submission:** Phase E-1/E-2 and Phase F.

**Minimum path to a defensible submission** (TMLR, AISTATS or a workshop): A1–A21, B1, B2, C1, C4, C6, F-a to F-c. That is about 35 GPU hours.

---

## 9. Requests considered and deliberately not planned

| Request | Why not |
|---|---|
| Cryptographic secure aggregation in the simulator (GPT PRIV-04, R049, R082) | Standard FL papers simulate the sum. Stating the threat model (A11) is enough |
| End-to-end DP for generator training | Out of scope for this paper. State it as a limitation (A11) |
| Asynchronous clients, concept drift, trainable extractor (GPT R041, R045, R057) | Different problem settings. List them as limitations (A19). D-2 covers partial participation |
| Responding to the 100 simulated scores as votes (GPT R095) | They are generated from one template, not independent reviews |
