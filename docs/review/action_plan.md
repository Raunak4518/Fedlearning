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

**Result (16 Oct 2026):** the readout half of Gate 1 passes on 4 of 5 datasets (ER margins +3.9 / +2.2 / +2.0 /
+2.6 / +3.4 on MNIST / FMNIST / SVHN / CIFAR-10 / CIFAR-100; SVHN misses the 2-point threshold by 0.02). The K15
half (NB38 / NB39) decides whether HWA stays a contribution. Details: `docs/PROGRESS.md` §1b.

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

**Status (17 Oct 2026).** C1: K14 in NB60. C2: K20 in NB63. C3: not planned to run (optional; partly covered by
E20 / E25). C4: K17 (MNIST done: validation-selected Ours beats every baseline by 3.6–4.6 points; the rest in NB60 /
NB61). C5: long tail in NB61. C6: K17 (FedLC, FedRS, balanced softmax, decay fix, CCVR / Fed3R / FedNCM readouts) +
K19 in NB63 (FSG + LA, FedDF, FedDF + LA); FedProto not run (needs a shared embedding dimension; its inference is
the FedNCM readout). D-1: SVHN K = 50 / 100 in NB62 (K18). D-4: done (CIFAR-100-LT in K16 / K17).

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

---

## 10. Every useful next experiment (18 Oct 2026, after the mathematical analysis)

**Where we are.** Phase A done; Gate 1 decided (K15, K16); C1 nearly done (K14 20 of 24), C4 / C5 / most of C6 done
(K17); D-4 done (CIFAR-100). The paper is now written as a stage-by-stage comparison with centralised training
(`docs/paper_draft.md`; theory in `docs/research/federated_vs_centralized_math.md`). That analysis changes what is
worth running: on MNIST / FashionMNIST / CIFAR-100 the federated pipeline is already within 0–2 points of the
centralised one, so the remaining questions are (i) whether that holds on SVHN / CIFAR-10, (ii) whether the two changes
the theory proposes add accuracy, and (iii) the breadth reviewers ask for.

**Rules applied.** Every new spec is checked with `python experiments/ledger.py --check <spec>` when it is built and
reuses seeds 3–5 so that K17 provides the paired references (no reference arm is re-run). Nothing from the ledger's
NOT RUN list is proposed: those notebooks are superseded (`docs/PROGRESS.md` §3). Times are wall-clock on Kaggle
T4 × 2, estimated from the logged `time_s` of K14–K17 (per seed and arm: MNIST ≈ 10–20 min, FashionMNIST ≈ 10–25,
SVHN ≈ 12–25, CIFAR-10 ≈ 50–70, CIFAR-100 ≈ 33–38; two GPUs run two groups at once).

### 10.1 Built and ready (run first)

| # | Notebook | What it decides | Wall time | Ledger |
|---|---|---|---|---|
| X1 | **NB63** = K19 + K20 + the 4 K14 runs NB60 lost | K19: FSG + LA, FedDF, FedDF + LA against ours on all five datasets (seeds 3–5, paired with K17) — the last missing published baselines. K20: are herding and moment correction real mechanisms (herding vs a random subset, mean-only vs spread-only, RFF dimension, pool size; predictions M1–M3). K14: completes the five-seed ablation on MNIST | ≈ 9.3 h | 0 duplicates |
| X2 | **NB62** = K18 | SVHN with 50 / 100 clients: HWA vs the decay fix where several clients share an architecture; **prospective test of the count-only collapse predictor** (every row with $\kappa_c<1$ collapses under GeFL-F, none under HWA; fixed in paper §3.5 before the run) | ≈ 10.5 h | 0 duplicates |

Both notebooks were rebuilt on 18 Oct (runs now log per-client class counts, which the $\kappa$ test needs). If you
already started the old versions, let them finish; only the $\kappa$ test on SVHN then needs the counts reconstructed.
NB60 is retired: its K17 part already ran (folder `K17_kaggle_fair_retest__nb60`).

### 10.2 Measurements that decide what else to build (cheap)

| # | Experiment | What it decides | Where / cost | Ledger |
|---|---|---|---|---|
| X3 | **P0: centralised comparator on SVHN and CIFAR-10** (the N7 protocol: federated vs centralised extractor; one-client vs pooled bodies; ER), seeds 0–2; plus CIFAR-100 with all 10 architectures and MNIST / FashionMNIST seed 2 (and FashionMNIST seed 1's lost centralised-extractor arm) | Paper §6.7 currently covers 3 of 5 datasets with 1–2 seeds. If the pooled body beats ours by ≥ 1.5 points on SVHN or CIFAR-10, a body-side method (P3) is worth building; if not, the paper can say federation costs ≤ 1 point everywhere | one Kaggle notebook running `docs/research/math_checks/n7*_*.py` on GPU, ≈ 5–6 h. **If you still have the K16 `__models.zip` files**, the federated-extractor arms need no training and it is shorter | not a spec (no run duplicates a ledger entry; Stage I is retrained because no K16 model archive is in the repository) |
| X4 | **P2 pilot: class-balanced Stage-I loss** (weights $N/(Cn_y)$ from the exact counts, clipped at 20), scored by raw-feature LDA and pooled-body ER on the 2,000-image validation split, MNIST / FashionMNIST / CIFAR-100, seeds 0–1 | Whether P2 is worth a full run. Prediction: +1 ± 1 on FashionMNIST, 0 ± 0.5 on MNIST, ≤ 0 on CIFAR-100 | CPU in this container, ≈ 3–4 h, no Kaggle time | not a spec |
| X5 | **P1 pilot: per-class signal-fraction estimate** $\hat s_c$ from the exact-sum LDA on generated features, against the referee's $s$, for GeFL-F + AdamW + LA and ours on CIFAR-10 / CIFAR-100 (`--quick`, seed 3) | Whether a label-free, server-side $\hat s_c$ tracks the referee's $s$ well enough to gate on (needed before X6) | Kaggle `--quick`, ≈ 1 h; can ride in the same session as X3 | new arms |
| X6 | **Instrumented run (M1, M2, M4, M7, M8 of the math report §7)**: per round, FE gradient heterogeneity; averaged vs pooled BatchNorm variance; holder gain / non-holder loss per row; retention of the synthetic phase through the real phase (absent vs held classes, with and without LA); logit spread on generated features. GeFL-F, decay fix and ours; MNIST and CIFAR-10, seed 3 | Turns Propositions 6–9 from toy-checked into measured on the real pipeline; gives the "what happens at each aggregation" figures | code ≈ 1 day; MNIST on CPU here, CIFAR-10 ≈ 2.5 h Kaggle | logging only; reuses K17 seeds |

### 10.3 New methods the theory proposes (build only after X3–X5)

| # | Experiment | Prediction (fixed now) | Wall time | Ledger |
|---|---|---|---|---|
| X7 | **P1: signal-fraction gate / per-class CSL weight** ($\beta_c=1-\hat s_c$, or synthetic labels only for classes with $\hat s_c\ge s_0$, $s_0$ chosen on validation). Arms: GeFL-F + AdamW + LA + P1 and ours + P1 on CIFAR-10, CIFAR-100 (where it should matter) and MNIST, FashionMNIST (no-harm check); seeds 3–5, paired with K17 | GeFL-F's synthetic gain on CIFAR-10 / CIFAR-100 moves from −0.8 / −2.2 to ≥ 0; ours ≥ 0 change on CIFAR-100, 0 ± 0.3 on MNIST / FashionMNIST. This is the most promising "new method": it follows from an exact identity (Proposition 9) and costs no communication | ≈ 6–7 h | new arms; references from K17 |
| X8 | **P2: class-balanced Stage-I loss in the full pipeline** (only if X4 is positive): ours + P2 on FashionMNIST, CIFAR-10, CIFAR-100 (+ MNIST no-harm), seeds 3–5 | as X4 | ≈ 4 h; can share a notebook with X7 | new arms |
| X9 | **P3: cross-client body refresh** (each client trains every architecture's body for a few rounds, FedAvg per architecture, then ER) — only if X3 shows a pooled-body gap ≥ 1.5 | ours + ER moves to within 0.5 of the pooled body. Departs from GeFL-F's one-architecture-per-client premise; must be presented as such | ≈ 4–6 h | new arms |

Not worth building (the analysis predicts ≤ 0.5 points): per-class step-normalised row aggregation (P4), a
class-balanced generator ELBO (P5, unless X5 shows low tail fidelity is the bottleneck), Stage-I drift correction,
exact BatchNorm recalibration (N7: ≤ 0.3), interleaving synthetic and real batches, shorter real phases, ERL.

### 10.4 Breadth reviewers ask for (Phase D; for a main-track or journal submission)

| # | Experiment | Answers | Wall time |
|---|---|---|---|
| X10 | **D-3 regime breadth:** IF ∈ {10, 50}, Dirichlet ∈ {0.1, 1.0}, and a random choice of which classes are rare (per seed); GeFL-F + AdamW + LA, ours and real-only + LA; FashionMNIST and SVHN (CIFAR-10 is 3× costlier), seeds 3–5. Needs the random-tail option in `make_partition` | "only one imbalance setting; the tail is always classes 6–9" (R5; GPT R063) | ≈ 7 h |
| X11 | **D-2 partial participation:** K = 100, 10% and 20% of clients per round, tail classes held by 1–2 clients; GeFL-F, decay fix, HWA, ours; MNIST and FashionMNIST, seeds 3–5. Needs client sampling in `core.py` | "realistic FL"; also stresses HWA, whose advantage grows when few holders are present | ≈ 6 h |
| X12 | **D-5 other generators under the long tail:** DCGAN-F and DDPM-F (trained without weight decay), GeFL-F vs HWA, MNIST / FashionMNIST, K = 10 / 50. Prediction from Lemma 4 / §3.6: no collapse, only dilution, so HWA's gain is ≈ 0 at K = 10 and grows with K | whether "rare classes vanish" is specific to CVAE-F's coupled decay — the paper's claim says it is | ≈ 6 h |
| X13 | **D-1 on CIFAR-10:** K = 50 (several clients per architecture), decay fix vs HWA vs ours, seeds 3–5 | the heterogeneous-FL claim on the hardest 10-class dataset (SVHN is covered by X2, MNIST / FashionMNIST by K15) | ≈ 8 h; lowest priority in this table |

### 10.5 Privacy and systems (small)

| # | Task | Cost |
|---|---|---|
| X14 | **E-1 membership attack** on the trained extractor and on the released statistics (exact, ε = 8, ε = 2), MNIST / FashionMNIST | CPU here, < 2 h (the extractor is retrained in minutes) |
| X15 | **E-2 batched kernel herding** (blockwise selection, no per-sample host sync), checked to select the same samples; re-measure time | code ≈ half a day; herding is 83–88% of our head stage, so this also shortens X7, X8, X10–X13 |

### 10.6 Recommended order (≈ 30 GPU-h per week)

1. **Week 1:** X1 (NB63) and X2 (NB62) on Kaggle; X4, X14 and the MNIST half of X6 on CPU here; X15 code.
2. **Week 2:** one notebook with X3 + X5 (≈ 7 h); then, if X5 shows $\hat s_c$ tracks $s$, build X7 (+ X8 if X4 is
   positive) as one ≈ 10–11 h notebook.
3. **Week 3:** X10, then X11 or X12 depending on the venue (main track: both).
4. **Before submission:** Phase F (tag, LaTeX, one-command regeneration).

**Minimum for a defensible TMLR / journal submission:** X1, X2, X3. **For a main-track attempt:** add X7 (a new
method with a margin on the hard datasets), X10 and X11.

### 10.7 Status (18 Oct 2026): everything in §10 is built; GPU parts packed into three ~11 h notebooks

| Notebook | Steps (most important first) | Est. wall time on T4 × 2 |
|---|---|---|
| **NB68_gate_and_balanced_fe** | K21 on CIFAR-10, CIFAR-100, MNIST (X5, X7, X8) | ≈ 11 h |
| **NB69_comparator_and_breadth** | X3 centralised comparator (SVHN, CIFAR-10, CIFAR-100; seeds 0-2; two GPUs as a queue); K25 instrumented MNIST + CIFAR-10 (X6); K21 FashionMNIST; K22 FashionMNIST (X10); K23 (X11) | ≈ 11 h |
| **NB70_more_clients_and_generators** | K26 CIFAR-10 with 50 clients (X13); K22 SVHN (X10); K24 DCGAN-F (X12) | ≈ 10.8 h |

Every notebook zips its results after each step, so a session that hits Kaggle's 12 h limit keeps every finished
step. NB64-NB67 (built earlier the same day) are retired: their steps are all in NB68-NB70. X1 / X2 remain NB63 /
NB62. Times are estimates from K14-K17 run times; the X3 and K25 steps have no earlier GPU timing.

| # | Where | State |
|---|---|---|
| X3 | NB69 (the CPU copy here was stopped: about 5 h per dataset-seed) | NB69 ready |
| X4 | CPU here, `docs/research/math_checks/x4_balanced_stage_i.py` | **done**: positive, prediction held (PROGRESS §3b; paper §6.7) |
| X5 | logged by every K21 run (`sfrac_hat` next to the referee's per-class fidelity) | NB68 / NB69 |
| X6 | K25 (`instrument=True`): M1, M2, M4, M7, M8 | NB69; every code path tested on CPU |
| X7, X8 | K21: SGATE / SCSL (`core.signal_fraction`) and BALFE, seeds 3-5, paired with K17 | NB68 / NB69; every new code path tested on CPU (tiny rounds) |
| X9 | not built: only if X3 shows a pooled-body gap ≥ 1.5 | waits for X3 |
| X10 | K22 (IF 10; Dirichlet 0.1 / 1.0; random rare classes via `permute_labels`) | **done** (NB69 FMNIST, NB70 SVHN): B1-B3 hold on both (paper §6.10) |
| X11 | K23 (K = 100, 10% / 20% participation via `Ctx.active`) | NB69 |
| X12 | K24 (DCGAN-F, flat vs HWA, K = 10 / 50) | NB70 ran out of time before this step (no results in the 22 Oct zip) |
| X13 | K26 (CIFAR-10, K = 50) | **done** (NB70): D1 holds at ≈ 0 (+0.1 with ER), D2 holds (+1.0 / +0.9) (paper §6.10) |
| X14 | CPU here, `x14_membership.py` | **done** (paper §6.9) |
| X15 | `HerdedGen.sample_batched` (same picks as the loop; `GEFL_KH_LOOP=1` restores it) | **done** |

Kaggle order: NB63, NB62 (X1, X2), then NB69, NB68, NB70.

## 11. Supervisor review, 22 Oct 2026: response and plan

Criticisms: no t-SNE / latent plots; no F1 / recall / precision; no class-imbalance baselines; plain PyTorch instead of an
FL framework; whole-generator aggregation. Full response: https://claude.ai/artifact/BHZxLZ2KeGiKR8XPjr69H1

| # | Criticism | Status | Action | Phase |
|---|---|---|---|---|
| R1 | t-SNE, latent scatter | missing (Kaggle models not returned) | `experiments/viz_latents.py`: real features, real vs generated per class, generator latents, classifier embeddings, confusion matrices. Models: upload saved Kaggle `models/`, else one notebook (5 datasets x GeFL-F / decay fix / ours, seed 3, `--save_gen`) | 1 |
| R2 | F1, recall, precision | recall, macro-F1, G-mean, worst / tail recall logged in every run; precision added to `core._class_metrics` on 22 Oct | T23 (`experiments/class_metrics_table.py`) into paper + handout; precision from the R1 runs | 0, 1 |
| R3 | class-imbalance baselines | 9 baselines (LA / balanced softmax, FedLC, FedRS, CCVR / Fed3R / FedNCM readouts, FSG + LA, FedDF +- LA) | heterogeneous track: CReFF, FedProto (projection to a shared size), FedGen; shared-architecture track: FedLoGe, Fed-GraB, BalanceFL, FedAvg, FedProx, ours. Seeds 3-5, ledger check first | 2 |
| R4 | FL framework | single-machine PyTorch simulation | Flower port of every stage, SecAgg+ for the sums, measured bytes, equivalence check on MNIST / CIFAR-10 | 3 |
| R5 | whole-generator aggregation | GeFL design: one shared generator architecture, FedAvg; 5.55 M params = 22 MB per round, ~4.4 GB per client over 100 rounds | explain; accuracy-vs-communication table: full generator, 25 rounds / 8-bit updates, no generator (FSG + LA) | 0, 3 |
| R6 | pre-empted | small frozen extractor; 10% data; no real-world data; 3 seeds | ResNet-18 frozen extractor on CIFAR; full-data check; Fed-ISIC2019 (FLamby); two more seeds | 4 |

Schedule: week 1 phases 0-1 (~12 GPU h); weeks 2-3 phase 2 runs (~48 GPU h) and phase 3; weeks 4-5 phase 4 (~42 GPU h)
and write-up.

## 12. Final architecture, decided 22 Oct 2026

| Component | Decision | Evidence (removing it; MNIST / FMNIST) | Cost | Privacy |
|---|---|---|---|---|
| Class-statistics release (n, sum h, sum ‖h‖²) | keep | required by the anchor, gate and MC | ~31 KB per client once | federation-level class means / spreads; rare-class membership AUC 0.63-0.73, DP option (eps = 8: 4.5-6.9 points) |
| Mean anchor + F gate + class rows with HWA when F < 0.05 | keep | anchor -3.9 / -0.5 as trained; SVHN rows 65.6 vs anchor 55.8 | generator 5.9 M vs 5.5 M params | no new release; class means inside the generator |
| Moment correction | keep | -1.1 / -0.6 as trained, -0.5 / -0.1 with ER | negligible | none new |
| Kernel herding | **dropped** | -0.7 / -0.7 as trained, -0.26 / -0.13 with ER | ~6x head stage | removes the RFF fingerprint release |
| Logit adjustment | keep | -3.1 / -5.8 as trained; ~0 with ER | none | prior stays on the client |
| Consensus soft labels | keep, **computed as a secure-aggregated sum over all clients** (each client sends w_k softmax(h_k(pool)), w_k = 1 / (G n_arch)); identical to the simulation | -1.4 / -1.5 as trained, -0.4 / -0.5 with ER | 24-583 MB per client over training at T_s = 10 (<= ~6% of generator traffic) | only the federation's averaged prediction on server-generated points is revealed |
| Exact readout | keep; **report with and without it** | GeFL-F +7 to +20; ours +9.2 CIFAR-10, +4.5 CIFAR-100, ~+1 elsewhere | one extra round, <= ~20 MB per client | bodies broadcast to all clients; class-wise embedding sums |
| T_s | **10 everywhere** | ER: 94.6 / 81.2 / 65.5 / 53.0 / 18.7 vs T_s = 1: 93.7 / 80.6 / 65.4 / 53.1 / 17.4 | cheap once KH is dropped (MNIST head stage ~106 s vs 42 s for GeFL-F) | none |

Missing for the final configuration (GATE + MC + LA + CSL, T_s = 10, no KH) on fresh seeds: MNIST s5, FMNIST s5 (s3-4 in
K14 / K20), CIFAR-10 s5 (s3-4 in K20), SVHN s3-5, CIFAR-100 s3-5. One notebook can run these with `--save_gen`, log
precision, and make the t-SNE / latent plots (R1). Ledger check before building.

## 13. Baselines, decided 22 Oct 2026 (framework: Flower; architecture §12; metrics §11 R2)

**Track 1, our setting (a different architecture per client; shared frozen extractor; heads per architecture).**
Kept from K17 / K19: GeFL-F; GeFL-F + decoupled decay + LA; real-data heads + LA / balanced softmax, FedLC, FedRS;
CCVR, Fed3R, FedNCM readouts; FedDF +- LA; Gaussian generator from exact statistics + LA.
New, model-heterogeneous (native): FedProto (AAAI 2022), FedTGP (AAAI 2024), FedGen (ICML 2021).
New, long-tail methods applied inside each architecture's head training: CReFF (IJCAI 2022), FedETF (ICCV 2023),
Fed-GraB (NeurIPS 2023), FedLoGe (ICLR 2024), FedYoYo (ICCV 2025).

**Track 2, the standard federated long-tail protocol (one shared architecture; FedLTLib, arXiv 2609.15625):**
CIFAR-10-LT / CIFAR-100-LT at IF 10 / 50 / 100, Dirichlet 0.5. FedAvg, FedProx, MOON, FedLC, CCVR (references) and
CReFF, FedETF, Fed-GraB, FedLoGe, FedYoYo, plus ours with a single architecture (G = 1).

Excluded, with the reason stated in the paper: CLIP2FL, CAPT (pretrained CLIP), FedKTL (pretrained generator), FedMD
(public labelled data). Second priority if time allows: RUCR (TIFS 2024), FEDIC (ICME 2022), FedGH (MM 2023), LG-FedAvg.
Implementation: authors' code (HtFLlib, LongTailLib, each method's repository) wrapped as Flower clients / strategies;
seeds 3-5, validation-selected settings, the full metric set. Ledger check before any spec.

## 14. Metrics, decided 22 Oct 2026

Main table (every method, heads as trained and with the readout, mean +- SD over seeds 3-5, paired 95% CI, Holm):
balanced accuracy, macro-F1, macro precision, MCC, G-mean, tail-class recall, worst-class recall, macro AUC-PR,
communication per client. Secondary / appendix: accuracy, Cohen's kappa, top-5 (CIFAR-100), macro AUC-ROC,
head / medium recall, recall SD, per-class precision / recall / F1, confusion matrices, ECE, class-wise ECE, NLL,
Brier, worst-architecture balanced accuracy, accuracy on each client's class mix (mean, SD, minimum), rounds to target
accuracy, wall-clock, peak GPU memory, FLOPs per round, membership-inference AUC, DP accuracy cost.

To add to the evaluation code (evaluation-time only): MCC, Cohen's kappa, macro AUC-PR and AUC-ROC (final round),
top-5, confusion matrix per readout, class-wise ECE, client-fairness spread / minimum, FLOPs, measured communication.
Cost: all free except rounds-to-target (evaluate every 5 rounds), membership inference (ours, once per dataset, no
training) and DP (ours only, eps = 8 and 2, two datasets: the only extra training).

## 15. Logging and plots, decided 22 Oct 2026

All metrics of §14 are kept (main table and appendix). The Flower implementation logs, per round and per client, to one
file per run (plus TensorBoard for live curves):
- Stage I: loss, test balanced accuracy, client drift, update / gradient norms, BatchNorm-statistic divergence.
- Stage II: reconstruction and KL loss, class-row norm per class, holder gain vs non-holder shrink, per-class fidelity,
  signal fraction, per-class MMD and moment mismatch, active latent units, Adam second moment of the rows.
- Stage III: synthetic- and real-phase loss, consensus entropy and agreement, balanced accuracy / macro-F1 / tail recall
  / ECE, per-class recall, logit spread before / after the synthetic phase, retention, classifier weight norm per class,
  predicted-class counts.
- Representation (every 5 rounds): NC1 / NC2 / NC3, minority-collapse angle, class separation F, CKA between architectures.
- System: bytes per message type, time per stage, GPU memory, participation.

Plot catalogue (one script makes all of them for every run): A training curves (1-8), B class-imbalance plots (9-18),
C representation and latent plots incl. t-SNE / UMAP / PCA, real vs generated overlays, class-mean and classifier
vector plots (19-27), D generator quality (28-32), E mechanism plots (33-37), F comparison plots incl. forest plot,
critical-difference diagram, radar chart, Pareto plot, ablation waterfall, sensitivity lines (38-46), G federated-system
plots (47-51), H privacy plots (52-54), I diagrams (55). Numbering as in the 22 Oct message.
**All 55 plots are required (user, 22 Oct)** for every dataset and track: the paper's main text shows 9 + 10, 11, 14,
22, 23, 33, 39, 44; every other plot goes in the appendix / supplement and the handout.

## 16. Collection phase (user, 22 Oct 2026): get everything first, decide the paper later

Scope: every method, every metric (§14), every log and plot (§15), in Flower, on a fresh consistent set of runs (old
results stay as the record; they lack the new metrics and per-round logs).
- Methods: §13 must-haves; then RUCR, FEDIC, FedGH, LG-FedAvg; then FedGELA, FedNH, FedVLS, FedLF, FML, FedKD, FD, FedMRL,
  SCAFFOLD. CLIP2FL, CAPT, FedKTL, FedMD in a separate "extra resources" table.
- Settings: MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100-LT, Tiny-ImageNet-LT, Fed-ISIC2019; IF 10 / 50 / 100;
  Dirichlet 0.1 / 0.5 / 1.0; K = 10 / 50 / 100; partial participation; full-data check; stronger frozen extractor; DP.
- Order: (1) main setting, every Track-1 method; (2) Track 2; (3) sweeps for ours + the five strongest baselines;
  (4) extra datasets and checks. Stage I / II models shared across methods wherever the method allows.
- Estimate: ~1,450 runs, ~500-650 T4 GPU-hours (~4-5 months at ~30 h / week); reuse of shared stages saves 30-40% of
  Track 1.

## 17. Flower implementation design (draft, 22 Oct 2026; Flower 1.42, Message API, Ray simulation)

- Principle: Flower orchestrates and carries messages; models and maths are imported from `core.py` (same networks,
  MC, readouts), and partitions come from the same seeds, so new runs pair with old ones.
- Layout: `fedflower/` with server_app, client_app (train / query / evaluate handlers), stages/ (extractor, stats
  release, generator, heads, readout), methods/ (ours + one plugin per baseline, common hooks), secure_sum, metrics
  (§14), logging (§15, JSONL per round per client + TensorBoard), plots (all 55), cache (shared Stage I / II), runner
  (run list split by helper number, resumable).
- Rounds: Stage I FedAvg (extractor over all clients, heads within architecture) -> statistics release (secure sum of
  n, sum h, sum ||h||^2; server computes m, V, F, gate) -> Stage II generator (plain FedAvg; class-row branch: secure sum
  of E(n) * row and E(n)) -> MC on the server -> Stage III rounds (pool seed; secure sum of weighted pool predictions =
  consensus; train heads; FedAvg within architecture) -> readouts (body broadcast; secure sum of per-class n, sum e,
  sum e e^T and the CCVR / Fed3R / FedNCM statistics). Evaluation every 5 rounds through a separate evaluation channel
  (test predictions + 1,000-image probe embeddings), excluded from communication accounting.
- Client state: generator Adam state in `context.state` (~44 MB per client).
- Secure aggregation: SecAgg+ (secaggplus_mod / SecAggPlusWorkflow) for every release of the method; extractor and
  generator weights plain FedAvg as in GeFL-F. Sums sent with weight 1 and scaled by public bounds into SecAgg+'s
  clipping range (+-8, 22-bit quantisation); acceptance test against a plain sum.
- Kaggle: two simulations per session (one per GPU), fractional client GPU shares after a benchmark, checkpoints at stage
  boundaries and every 10 rounds, zip after each run.
- Verification before large runs: unit tests (secure sum, readout, metrics vs scikit-learn), equivalence of GeFL-F and
  ours vs core.py on MNIST and CIFAR-10 seed 3, CPU smoke test of every plugin.
- Risks: SecAgg+ precision on second moments; CCVR per-class covariances on CIFAR-100 (low-rank if needed); Ray
  overhead; Kaggle CPU count.

### 17a. Design decisions (user, 23 Oct 2026)
- SecAgg+ on the method's releases only (statistics, class rows, consensus predictions, readout sums); extractor and
  generator weights plain FedAvg.
- Every run in Flower (sweeps included); parallelism set after a benchmark of Ray overhead.
- Evaluation: clients evaluate on a shared test copy, the server collects through the evaluation channel.
- Clean rewrite (no import of core.py). Acceptance before any large run: identical partitions (client counts equal to
  those logged by the old runs), identical class statistics, GeFL-F and ours on MNIST and CIFAR-10 within seed noise
  of the old results.

### 17b. Aggregation effect (pre- vs post-aggregation evaluation)
Every evaluation round (every 5 rounds) each averaged model is scored twice: each client's copy right after local
training (pre) and the aggregated model (post); effect = post - pre, per class.
- Generator (Stage II): per-class row norm, per-class fidelity (referee on 200 fakes per class), signal fraction,
  fake-vs-real MMD. Prediction (contraction theorem, T01 / K25): GeFL-F's rare classes lose after every aggregation;
  ours do not.
- Extractor (Stage I): balanced accuracy of local vs averaged extractor + heads.
- Heads (Stage III, K = 50 / 100): per-class recall before vs after within-architecture averaging.
- Local-only baseline (no federation): per-class benefit of collaboration.
New plots 56-60: aggregation effect per class over rounds (heatmap, GeFL-F vs ours); rare-class fidelity pre vs post;
row norm of holders' copies, non-holders' copies and the average; aggregation effect vs kappa; federated vs local-only
per class.

## 18. Datasets, decided 23 Oct 2026 (run in this order)

1. Already in results: MNIST, FashionMNIST, SVHN, CIFAR-10-LT, CIFAR-100-LT.
2. Light, natural imbalance: DermaMNIST, OrganAMNIST, BloodMNIST, PathMNIST (MedMNIST), GTSRB, EMNIST (byclass),
   FER-2013 (facial expressions; free, 7 classes, ~16:1).
3. Medium: Fed-ISIC2019 (FLamby, real hospitals), FEMNIST (writers as clients), Tiny-ImageNet-LT, CINIC-10,
   Flowers-102, HAM10000 (full resolution), APTOS-2019; AffectNet only with the authors' licence, at reduced resolution.
4. Off-topic, last (need other extractor / head families): Fed-Heart-Disease (tabular), Office-Caltech-10, AG News
   (text), Speech Commands (audio).
Dropped as too heavy for the week: ImageNet-LT, iNaturalist-2018, Places-LT, Fed-Camelyon16, ChestX-ray14 / CheXpert,
DomainNet. Estimate: tier 2 about +450 runs / 60-80 GPU-h; tier 3 about +500 runs / 200-300 GPU-h; tiers 1-2 and part
of tier 3 fit the week with 15 helpers.

## 19. Settings and run list (planned 23 Oct 2026; estimates, nothing run yet)

Generated by `python experiments/plan_runlist.py` -> `docs/review/run_list.csv` (one row per run: phase, priority,
dataset, method, seed, IF, alpha, K, participation, estimated minutes, Kaggle session and helper) and
`docs/review/run_list_summary.md`. Re-run the script whenever a setting changes; never edit the CSV by hand.

Settings
- Seeds 3, 4, 5 everywhere (same splits as the old runs, so old and Flower results can be paired).
- Track 1 main setting (phase A): IF = 100 where the imbalance is made, natural imbalance otherwise; Dirichlet 0.5;
  K = 10; full participation; the old architecture mix; 33 methods (ours, GeFL-F and the K17/K19 baselines, §13 and
  §16 lists, extra-resources methods marked); all 23 datasets of §18, tier order.
- Track 2 (phase B): FedLTLib datasets and IFs (CIFAR-10/100-LT at IF 10 / 50 / 100, Tiny-ImageNet-LT at IF 50),
  20 clients, Dirichlet 0.5, 8 per round; ResNet-8, 200 rounds, 10 local epochs, batch 32, SGD 0.1 taken from the
  CReFF protocol until checked against the LongTailLib code; 20 methods.
- Sweeps (phase C, tier-1 datasets, 8 placeholder methods until the five strongest baselines are known): IF 10 and 50;
  alpha 0.1 and 1.0; K = 50; K = 100 at participation 1.0, 0.2, 0.1; rare classes chosen at random.
- Checks (phase D): DP at eps 8 and 2 (ours, MNIST and FashionMNIST); ResNet-18 frozen extractor (CIFAR-10/100); full
  training set (MNIST, CIFAR-10).
- The tier-1 rows of phase A repeat old runs on purpose (§16: everything redone in Flower); `ledger.py --check` is
  still run on each spec to catch duplicates inside the new plan.

Totals: 3,885 runs, about 1,131 single-GPU hours (A 2,277 / 239 h; B 420 / 406 h; C 1,080 / 389 h; D 108 / 96 h).
With 15 helpers, 12 h sessions on T4 x 2 and 2 runs per GPU: 2 sessions (days) per helper, about 24 of the 30 weekly
Kaggle GPU hours per account, leaving days 3-7 for the Flower rewrite's acceptance tests, failed runs and analysis.
If packing 2 runs per GPU fails, 4 sessions per helper (still within the week, not within one weekly quota).

Assumptions to verify before the notebooks are written: stage times of new datasets (20k-image cap, scaled from K17
logs); cost factor 0.35 for methods reusing a cached extractor; overhead 1.7 for Flower / Ray and full evaluation;
Track-2 protocol details.

## 20. Method name, decided 23 Oct 2026

**FedECS-LT: Exact Class Statistics for Long-Tailed federated learning.** The name states the one principle behind
every part: per-class totals (counts, feature sums, squared lengths; class-wise output sums for the readout) obtained
exactly by secure aggregation, used instead of averaged models wherever an answer depends on the class distribution.
"-LT" follows the field's label for long-tailed data. Checked names: FedCSA is taken (IEEE TDSC 2026, class-conditional
anchor alignment with secure aggregation; close to ours, cite it), FedSAGE and FedTAIL taken, FedCSB too close to
Fed-CBS (ICML 2022); no method named FedECS was found.

Decided with the user (23 Oct 2026):
- "Holder-weighted averaging (HWA)" is renamed **class-count weighted averaging (CCWA)**: each class row of the generator
  is averaged over the clients that have that class, weighted by their capped class count E(n). The rule itself exists
  (FedProto, cwFedAvg, MLFCIL); applying it to generator rows is ours.
- The renaming in the documents ("Ours" -> FedECS-LT, HWA -> CCWA in paper_draft.md, paper.tex, the study guide and
  PROGRESS.md) is done **later**, when the paper is rewritten after the Flower results. New code, plots and tables
  (the Flower rewrite) use FedECS-LT and CCWA from the start.
- At that rewrite also: add FedCSA and MLFCIL (arXiv 2604.02356) to related work; update the study guide to T_s = 10 and
  no kernel herding (§12).

## 21. Figure standard: IEEE format for every plot (user, 23 Oct 2026)

All 60 plots of §15 / §17b, and every later figure, follow the IEEE graphics rules. One shared style module
(`experiments/plot_style.py`, written with the Flower code; nothing else may set fonts or sizes) applies them, so no plot
is styled by hand.
- **Size at final print size:** one column 3.5 in (88.9 mm) wide; two columns 7.16 in (181.9 mm); height at most
  8.5 in, normally 2.2-2.8 in for one column. Plots are drawn at this size, never scaled in LaTeX.
- **Fonts:** Times (serif, matching IEEEtran body text; maths in the same font), 8 pt for axis labels, ticks, legends
  and in-plot text, never below 7 pt after any scaling. Fonts embedded (Type 42 / TrueType), no Type 3.
- **Lines and marks:** lines at least 0.5 pt (data lines 1.0 pt), marker size 3-4 pt; each method keeps the same colour,
  marker and line style in every plot (one table of methods -> style in the module).
- **Colour:** RGB, colour-blind-safe palette; every plot also readable in greyscale print (methods told apart by
  marker and line style, not by colour alone); sequential maps viridis / cividis for heatmaps, diverging map centred at
  0 for effects (§17b).
- **Axes:** every axis labelled with quantity and unit in parentheses, e.g. "Balanced accuracy (%)", "Round",
  "Communication (MB)"; same y-range for plots that are compared; no chart titles inside the image (the caption says it).
- **Subfigures:** (a), (b), ... placed below each panel; one shared legend per figure where possible.
- **Files:** vector PDF for line art and charts; bitmaps (t-SNE with many points, confusion matrices with images)
  at 600 dpi; file names fig_<number>_<short name>.pdf.
- **Captions and tables (LaTeX side):** "Fig. 1." caption below the figure; tables numbered in Roman numerals
  ("TABLE I"), caption above, IEEE table style (horizontal rules only).
- Check before use: an automatic test in the style module renders each plot and checks width, font sizes and embedded
  fonts; IEEE's free graphics checker (Graphics Analyzer) on a sample before submission.

### 21a. IEEE only; resizing without breaking the rules (user, 23 Oct 2026)
Target: IEEE (IEEEtran template, journals and conferences). No other venue profiles.
- **Never resize a finished figure in LaTeX.** Shrinking a plot shrinks its text and lines below the minimum (8 pt
  text scaled to 80% is 6.4 pt: a violation). Every plot is redrawn at the width needed instead:
  `python experiments/make_figures.py --width column` (3.5 in), `--width page` (7.16 in) or a width in mm, always with
  8 pt Times text and 0.5 pt minimum lines at the printed size.
- **Every figure saves its data** (a CSV next to the PDF), so it can be redrawn at another size or for a reviewer
  request without re-running experiments.
- **Paper:** paper_draft.md stays the master text; the submission is written in the IEEEtran template (replacing the
  current article-class paper.tex when the paper is rewritten after the Flower results).

## 22. Implementation decisions (23 Oct 2026; folder fedecs_lt/, see its README)

- **Notebooks:** standalone (every helper pasted); each returns `results.zip`, `report.pdf` (every table and figure,
  IEEE captions) and `models.zip`. NB00 (user's account) runs first: GeFL-F and FedECS-LT on MNIST and CIFAR-10,
  seed 3, against the old runs; helpers' notebooks go out only after `ACCEPTANCE: PASS`.
- **Acceptance, honestly stated:** client splits identical to the old runs (checked on real data: MNIST 5,996 and
  CIFAR-10 18,485 images, every client's counts equal); every client training step equals core.py weight for weight
  (tests); whole runs cannot be bit-identical (Flower seeds each client action separately; core.py used one random
  stream), so accuracy must lie within max(2 points, 3 x the old seed SD) of the old seed-3 run.
- **Track 1** (different client models): FedECS-LT, GeFL-F, Balanced Softmax, FedLC, FedRS, FedDF, Local, and the
  model-heterogeneous methods FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL (HtFLlib, Apache-2.0, ported to
  Flower; their hyperparameters from the authors' scripts; same rounds and local epochs as our head stage).
  **Published names only (user, 25 Oct):** the ablations GeFL-F w/o generator, GeFL-F + LA (AdamW), GeFL-F w/
  Gaussian generator + LA and FedDF + LA are out of the main comparison and run in their own notebooks
  (NB30_ablation_cifar10 / _cifar100, seeds 3-5), paired with the main notebooks' FedECS-LT and GeFL-F runs (same
  splits and seeds). Reports print the method's name only (no "(LongTailLib)" suffix; the methods table names the
  code), and each method keeps one colour and marker, unique within every report.
- **Every §14 / §15 quantity is now measured (25 Oct; audit against this plan found gaps):** paired tests with Holm
  adjustment, rounds to a target accuracy, own-mix accuracy SD and minimum, forward FLOPs per image and training FLOPs
  per round, NC2 / NC3 and the minority-collapse angle (last layers saved with the embeddings), CKA between
  architectures, per-class MMD and mean mismatch of the generated features, active latent units, consensus
  agreement, logit spread before / after the synthetic phase and retention (figures 74-82 and four report tables).
  Embedding maps follow the t-SNE literature (Wattenberg et al. 2016; Kobak and Berens 2019): tail classes
  emphasised, GeFL-F vs FedECS-LT side by side (Fig. 83), a perplexity check (Fig. 84), 10-NN accuracy of the
  embeddings in captions and the representation table, and UMAP of the same embeddings (Fig. 85; umap-learn,
  BSD-3-Clause, installed at run time in the release that matches the image's scikit-learn, which is kept).
  Membership inference and DP stay in the privacy experiments (not built). A figure that fails to draw now fails
  the CPU test and is counted on the report's cover. The long-tail methods built for one shared model are
  compared in Track 2 only (user, 23 Oct).
- **Track 2** (one shared ResNet-8, FedLTLib protocol: 20 clients, 40% per round, 200 rounds, 10 local epochs, batch
  32, SGD 0.1 momentum 0.9 decay 0.99): FedAvg, FedProx, MOON, FedLC, CCVR, CReFF, FedETF, Fed-GraB, FedLoGe, FedYoYo,
  RUCR, CLIP2FL run with the authors' LongTailLib code unchanged (downloaded at run time at commit 4141678), on our
  splits (standard CIFAR-LT profile, Dirichlet 0.5); every metric is computed from the predictions LongTailLib's own
  evaluation ranks (recorded and checked against its accuracy). FedECS-LT in Track 2: ResNet-8 split after its second
  stage (frozen extractor 128 x 16 x 16), T_FE = T_TN = 100 rounds (the baselines' 200), generator rounds extra and
  reported.
- **Dropped:** FEDIC, FedGELA, FedLF, FML, FD, FedKD, FedKTL, FedMD, SCAFFOLD, CAPT (reasons in the plan of 23 Oct).
- **Full-model version dropped** (user, 23 Oct, after a CPU test): the six model-heterogeneous methods run only in
  the common protocol (shared Stage-I extractor, their authors' hyperparameters). Training each client's whole
  CNN (up to ResNet-152) would cost about 60-90 GPU-hours and ~12 GB of weights per notebook, and would not
  compare like with like (much larger client models, no shared extractor). The code is in git history
  (commit 24ccf7a) if a reviewer asks for it.
- **Per-class logging (user, 24 Oct):** every metric that has a per-class form is computed for every class at
  every evaluation (each round's pre- and post-aggregation evaluation, the final test and validation evaluation,
  the exact readout): recall, precision, F1, average precision, AUC-ROC and MCC of the class against the rest,
  calibration error of p(c|x), NLL, Brier, top-5 recall, share of predictions (`metrics.PER_CLASS`, checked
  against scikit-learn). The report has a per-class table for each (blocks of 20 classes) and figures 61-70.
- **Track-2 scope reduced** (user, 24 Oct, after NB00 measured 3.3 h for one LongTailLib FedAvg run; the full plan
  was ~1,250 run-hours): imbalance ratio 100 with seeds 3-5, ratios 10 and 50 with seed 3; CLIP2FL dropped (pretrained
  CLIP model = outside knowledge, slowest method); LongTailLib evaluates every 5th round (training unchanged).
  Diagnosis run in the first Track-2 notebook: FedECS-LT with its extractor trained for all 200 rounds (NB00:
  FedECS-LT 59.6 vs FedAvg 61.1 balanced accuracy, better tail 48.0 vs 44.2 and worst class 33.7 vs 19.4).
- **Notebook split** by `fedecs_lt/plan_time.py` (run times fitted to NB00 within 5%; each part estimated <= 9.5 h):
  47 notebooks, ~326 session-hours (Track 1: 23, ~142 h; Track 2: 23, ~179 h). FedYoYo at ratio 10 (CIFAR-10/100)
  runs alone on a GPU (NB29; the speed-up of not sharing a GPU is assumed, 25%, and is measured there).
- **First returns (27 Oct 2026), 17 notebooks** (`fedecs_lt/results/`, links in `KAGGLE_SOURCES.md`): three had failed
  runs, now fixed. GTSRB part 1: all 21 failed because the download from the GTSRB server timed out; GTSRB now comes from
  a Hugging Face copy of the official split (39,209 / 12,630 images), and parts 1-3 run again. EMNIST: 26 of 39 runs
  missing, because four parallel runs exhausted the 30 GB of host memory; they rerun one run per GPU. One shared
  model: FedECS-LT ran out of GPU memory three times (CIFAR-100 ratio 10; CIFAR-10 ratio 10; the 200-extractor-round
  diagnosis), because it holds about 13 GB of a T4's 14.6 GB (32,768-dimensional features of every image). Fixes:
  - such runs now take a GPU to themselves (`runner.GpuShare`, a per-GPU lock; scheduling only);
  - the previous round's synthetic slice is freed before the next is drawn (same values);
  - workers set `expandable_segments`.
  The builder puts every missing run of a returned notebook into folder 5 (`*_rerun_<id>`); the one-shared-model
  reruns are pooled into one notebook.
- **Not run:** FedNH and FedVLS (priority B). LongTailLib contains FedNH, so it can be added to Track 2 by one
  line in `fedecs_lt/build_notebooks.py` if the main results leave room.
- **Licences:** MIT / Apache code pasted with credit; unlicensed or GPL code downloaded at run time, never stored here.
- **Secure aggregation:** Flower SecAgg+ for every FedECS-LT release (statistics, CCWA rows, consensus, readout sums),
  30-bit quantisation, public bounds from the federation size and the server's validation features.
- **Sweeps and checks (phases C, D)** are built after the main results name the five strongest baselines (§19).
