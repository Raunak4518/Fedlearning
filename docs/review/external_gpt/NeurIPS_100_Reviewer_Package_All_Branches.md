# NeurIPS 2026-style 100-reviewer audit — all repository branches

**Audit basis:** `Raunak4518/Fedlearning`; reviewed all branches visible on 2026-10-15. The latest working snapshot is `claude/moved-session-c4mfhn` at `1064d67acf909ccffcea4f2578814b44eafc762b`; `main` and `revision/adam-collapse-drift` point to `f9f4b4437f12dc633f4f20ac086bcdb8e6120d57`. The latest branch is the technical review candidate, not the default branch. This is a simulated panel, not actual NeurIPS reviews or 100 independent model instances.

## Executive verdict

**Simulated overall: 2/6 — Reject as submitted; confidence 4/5.** The research direction and measured long-tail results are promising; the decision is driven by an unresolved optimizer-fix baseline, the missing direct classifier-recalibration baseline, the discrepancy between described and executable separation gating, and still-unrun K14/K15 jobs. This verdict is not a prediction of actual NeurIPS acceptance.

The most important correction from the previous main-only review is that **the latest branch has already corrected the old K/2 collapse theorem**. Do not repeat the old objection as if it still applies to the latest draft. The new bound is a valid conditional norm inequality, but its novelty/predictive force is modest; the mechanistic claim about Adam still needs broader empirical verification.

## 1. Branch inventory and version-of-record

| Branch | Tip | Blob files | Difference vs main | Review interpretation |
|---|---|---:|---|---|
| `main` | `f9f4b4437f` | 324 | +0 files / 0 modified paths | Audited; baseline reference |
| `revision/adam-collapse-drift` | `f9f4b4437f` | 324 | +0 files / 0 modified paths | Audited; identical to main; not counted independently |
| `claude/moved-session-c4mfhn` | `1064d67acf` | 525 | +201 files / 100 modified paths | Primary technical audit snapshot; not merged into default branch |

Repository links: [main](https://github.com/Raunak4518/Fedlearning/tree/main), [revision/adam-collapse-drift](https://github.com/Raunak4518/Fedlearning/tree/revision/adam-collapse-drift), [claude/moved-session-c4mfhn](https://github.com/Raunak4518/Fedlearning/tree/claude/moved-session-c4mfhn).

`revision/adam-collapse-drift` is not an independent code state: it resolves to the same commit as `main`. The current paper/evidence work sits in `claude/moved-session-c4mfhn` and is not merged. That branch has 201 unique files and 100 changed shared paths compared with main, including `docs/PROGRESS.md`, `experiments/LEDGER.md`, `docs/paper_assets/CHECK.md`, `docs/research/readout_report.md`, corrected `docs/paper_single_idea.md`, T01/T02/T03 evidence, E32 MC-B, and K14/K15 specs/notebooks.

### High-value branch deltas

- `docs/PROGRESS.md` — added on claude branch: Research-state ledger; makes pending jobs and draft/data discrepancies explicit.
- `experiments/LEDGER.md` — added on claude branch: Claims to index 1,177 completed runs; raw per-run audit improves reproducibility.
- `docs/paper_assets/CHECK.md` — added on claude branch: Reports paper-draft numbers that disagree with generated tables/raw runs.
- `docs/paper_assets/numbers.csv` — added on claude branch: Generated traceability for mean/sd/seeds/p-values/source arms.
- `docs/paper_single_idea.md` — modified on claude branch: Corrects obsolete K/2 contraction statement; adds calibrated results and pending data caveats.
- `docs/paper_draft.md` — modified on claude branch: Newer full draft; still has number-level mismatches listed in CHECK.md.
- `docs/research/readout_report.md` — added on claude branch: Readout gap, exact readout, failed mechanism hypothesis, recalibration baseline warning.
- `experiments/results/T01_collapse_threshold/summary.md` — added on claude branch: Corrected contraction inequality checked on one MNIST seed, 1,200 row-rounds.
- `experiments/results/T02_dilution_rates/summary.md` — added on claude branch: Shows dilution without weight decay and HWA scaling over K=10/50/100.
- `experiments/results/T03_exact_readout/PREREGISTRATION*.md + runs*.jsonl` — added on claude branch: Readout recalibration preregistered; original synthetic-mean mechanism prediction failed; follow-up only one reduced seed.
- `experiments/results/E32_moment_anchor_bayes/*` — added on claude branch: New MC-B results: promising on FMNIST; unstable MNIST LT seed.
- `experiments/K14_kaggle_ablation_seeds_mcb.py; experiments/K15_kaggle_collapse_fixes.py; kaggle_notebooks/NB32–NB39` — added on claude branch: Critical planned runs are specified but status says not run.
- `docs/paper_assets/tables/T17/T18_collapse_fixes*` — added on claude branch: Tables have many missing cells because K15 has not run.
- `experiments/results/K09_ablation_analysis.md` — added on claude branch: Calls out n=3 multiple-test weakness and CSV overwrite issue.
- `docs/professor_report/OursA_Backup_Handout.docx/pdf` — added/updated on claude branch: Latest research handout, should not override raw scripts and runs as source of truth.

**Source-of-truth rule used below:** latest draft and new experiments are evaluated from the Claude branch; where artifacts exist only in main, those are not counted as newer evidence. Main-only readers should treat their checkout as stale rather than assume it contains the latest corrections.

## 2. Reconstructed research state

**Central claim.** In GeFL-F, class-specific generator conditioning rows shrink under federated averaging when non-holders' coupled Adam weight-decay updates outweigh the holders' actual row-norm gains. ESA proposes anchoring class identity, moments and empirical kernel features to federated sums, with HWA/LA/CSL as complementary components and a mean-separation fallback for datasets such as SVHN.

**Current candidate method (per latest Claude branch).** PC-VAE/mean anchoring; scalar moment correction (MC) or a covariance variant (MC-S/MC-B); kernel herding (KH) against finite random Fourier features; logit adjustment (LA); cross-architecture consensus soft labels (CSL); and balanced-bias calibration (BBC). When mean separability F is low, the described algorithm uses original class rows with HWA plus moment/covariance correction and KH. The actual run configuration chooses generator type from the config, however, rather than dynamically from F—see ALG-01.

**Empirical status and scope.** On IF=100/Dirichlet(.5), K=10, reported raw final balanced accuracy rises about 75.3→94.8 on MNIST and 58.5→79.7 on FashionMNIST; BBC-calibrated numbers in some report tables are around 95.2 and 80.6. CIFAR-10 is approximately 46.1 (or ~49 with calibration depending on metric), and SVHN with the row/covariance fallback about 65.6 versus GeFL-F 50.5. MC-B reaches FMNIST LT 81.57±0.31 and IID best-mean-accuracy 85.75±0.45, but MNIST LT varies sharply across its three seeds (one run 92.50). The headline setting remains a fixed frozen feature extractor and four image benchmarks.

**Theorem state.** The Claude branch corrects the obsolete `p < 1/2` claim to `p < Dbar/(Gbar+Dbar)`, where Gbar/Dbar are observed holders' mean norm gain and non-holders' mean loss. The inequality is a valid conditional consequence of the triangle inequality. T01 reports no violations in 1,200 row-rounds on one MNIST split/seed. This validates consistency with the bound, not independently the Adam optimizer mechanism. T02 separately documents dilution without weight decay.

**Positive evidence in the newer branch.** It contains a comprehensive run ledger (claims 1,177 completed run records), raw JSONL results for many experiments, generated paper assets and a draft-vs-results checker; it candidly records failed hypotheses (T03’s synthetic-mean explanation; VCT vs MixUp; KME-Gen; Bayesian CSL) and explicitly marks K14/K15 as unrun. These are improvements over a main-only review and should not be ignored.

**Outstanding critical experiments.** K15 (matched AdamW/no-decay/lazy-decay against HWA at K=10/50/100) has specs/notebooks but no results; K14 (ablation seeds 3–4 and MC-B validation) is also pending. Direct matched CReFF/CCVR/Fed3R-style classifier calibration remains missing. Exact readout is promising but its direct T03 evidence is one seed and shortened schedule.

## 3. Cross-branch contradictions and source traceability

1. The latest branch’s `docs/paper_assets/CHECK.md` records multiple paper-vs-generated-data mismatches: MNIST IID 96.02 vs 96.01; K=100 MNIST +HWA+LA+CSL 82.7 vs 83.04; FSG+LA+CSL 85.5 vs 86.34; E16 long-tail values including GeFL-F 61.33 vs 59.13 and Ours Ts=10 79.33 vs 79.23; CIFAR-10 CSL 60.02 vs 59.85; DDPM-F 83.10 vs 82.82. These mostly look like stale numbers, not fabrication, but need one regenerated source of truth.
2. `docs/PROGRESS.md` says K=50 client entries 86.8/87.2/87.9 have no corresponding run and the K=100 entries include n=1–2. Remove or re-source those cells.
3. `experiments/results/E25_anchored_stack/summary.md` is inconsistent in its setting headings relative to `runs.jsonl`; regenerate the summary from merged raw records. K09 analysis says the worker process overwrote `runs.csv`, omitting the `-PC` arms, while `runs.jsonl` and `summary.md` contain them.
4. K01 CIFAR-10 per-seed `runs.jsonl` was not copied back. The updated progress file says the table’s test is Welch/unpaired there; most other referenced comparisons are paired. Keep this caveat visible.
5. The one-seed collapse and readout diagnostics, three-seed component ablations, MC-B instability and pending K14/K15 are not equivalent evidence strengths. The branch notes distinguish those levels but the final paper must preserve that distinction.

## 4. Master issue register

Severity key: **CRITICAL** blocks a defensible submission; **MAJOR** requires resolution or a narrow claim; **MODERATE** needs qualification/evidence but is not independently fatal; **LOW** is mainly housekeeping. `confidence` is confidence in the audit observation, not in the paper's ultimate outcome.

### BR-01 — CRITICAL — Branch/version control
**Location:** Repo refs; `main` vs `claude/moved-session-c4mfhn`
**Claim under review:** Current paper and supporting evidence are in a single unambiguous, reviewable snapshot.
**Evidence:** `main` and `revision/adam-collapse-drift` point to f9f4b443; the latest branch points to 1064d67 and contains 201 additional blobs plus 100 modified paths. The latest branch has a corrected theorem, progress ledger, paper-assets checker, T01/T02/T03, readout report, E32 and pending K14/K15 artifacts. These aren't on the default branch.
**Impact:** A reviewer cloning default `main` would read a different paper/evidence state and miss important fixes and limitations. There is no PR to define the canonical merge boundary.
**Best author counterargument:** The claude branch is the active working branch and main is simply behind.
**Required action:** Before submission, nominate one canonical commit and merge or tag it; archive older drafts; regenerate all tables and ledgers at that commit.
**Status:** Verified | Confidence 5/5 | Reviewers: R001;R008;R010;R020;R047;R076;R080;R091;R096;R100

### TH-01 — MAJOR — Theory / collapse
**Location:** Claude `docs/paper_single_idea.md` §3 + App A.1; `experiments/results/T01_collapse_threshold/summary.md`
**Claim under review:** The corrected bound provides a novel optimizer-independent result explaining collapse.
**Evidence:** The norm inequality follows from `||mean(w_k)|| ≤ mean||w_k||`; with observed holders' mean gain Gbar and non-holder loss Dbar, contraction follows if pGbar−(1−p)Dbar<0. Correct but conditional on observed updates. It is not a predictive optimization theorem by itself; T01's 0/1,200 violations check this inequality, which is a triangle-inequality bound.
**Impact:** The theorem can support a diagnostic framing, but novelty/significance must come from the empirically measured optimizer–aggregation mechanism and useful fix, not the generic inequality alone.
**Best author counterargument:** The threshold θ is informative when measured G/D are tied to optimizer and long-tail data; the work is explicit that the bound itself is optimizer-free.
**Required action:** Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
**Status:** Reframed; still needs careful claims | Confidence 5/5 | Reviewers: R001;R004;R019;R021;R022;R023;R025;R030;R031;R034;R094;R099

### TH-02 — MAJOR — Theory / Adam
**Location:** Claude `docs/paper_single_idea.md` App A.1; `docs/paper_draft.md` §3; `GenClient` and `gen_local_update` in `experiments/core.py`
**Claim under review:** Any nonzero coupled Adam weight-decay coefficient produces an approximately λ-independent full learning-rate-sized row shrink over all relevant non-holder updates.
**Evidence:** The draft now qualifies the argument by λ|w_j|≫ε and acknowledges sparse-gradient steps can exceed η. In code, each client has persistent Adam optimizer state across rounds while global parameters are reset to aggregated weights. T01 tests one MNIST split/seed (40 rounds) and sees λ=1e−4 and 1e−3 similarly collapse, and wd=0 does not; it is not an unrestricted theorem over persistent state, zero crossings, ε-dominated rows, or other architectures.
**Impact:** The empirical mechanism is plausible and supported, but any universal optimizer statement or transfer to DCGAN/DDPM/FedAvg variants is overstated. T02 shows dilution even without decay.
**Best author counterargument:** The paper narrows it to the regime actually measured, reports λ-invariance empirically, and separates dilution from decay.
**Required action:** State the per-coordinate assumptions and optimizer-state conditions; report T01 as one-seed mechanism evidence. Run K15's matched AdamW/no-decay/lazy-decay comparisons before claiming HWA is the best fix.
**Status:** Partially supported; K15 pending | Confidence 4/5 | Reviewers: R021;R022;R024;R025;R031;R032;R033;R034;R035;R038;R040;R094

### EXP-01 — CRITICAL — Baseline fairness / causal attribution
**Location:** Claude `docs/PROGRESS.md` §3; K15 specs and NB38/NB39; tables T17/T18
**Claim under review:** The proposed HWA/ESA treatment is superior to simpler optimizer fixes for conditioning-row collapse.
**Evidence:** K15 explicitly schedules matched AdamW (wd=1e−3), AdamW 1e−2 at K=10, no weight decay on class rows, lazy decay, compared against HWA. `docs/PROGRESS.md` says notebooks ready, not run; tables T17/T18 have 20 missing cells each.
**Impact:** This is the most direct alternative explanation for the optimizer-related contribution. Without it, the paper cannot claim HWA is superior to simply correcting the decay implementation or removing decay from the conditioning rows.
**Best author counterargument:** K15 is planned and can be added in a revision; HWA may still win because it fixes dilution as well as shrinkage.
**Required action:** Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
**Status:** Pending; critical | Confidence 5/5 | Reviewers: R002;R011;R022;R023;R032;R035;R040;R050;R061;R062;R068;R078;R091;R092

### ALG-01 — MAJOR — Algorithm-to-code fidelity
**Location:** Claude `docs/paper_single_idea.md` §4.2 and §4.6; `experiments/core.py` `run_one` and `stage_ii`
**Claim under review:** The algorithm automatically chooses mean anchoring vs class-row/HWA fallback from statistic F at runtime.
**Evidence:** The core computes/logs F from class statistics, but `stage_ii` chooses behavior based on configured `m['gen']['type']`; no `if F < 0.05` branch controls generator type there. The report describes an automatic separation gate; the experiment results are run using preselected variants/configs.
**Impact:** The claimed data-dependent switch is not demonstrated by the executable current pipeline as one adaptive algorithm. This is a reproducibility and algorithm-description issue, not proof that the reported per-dataset results are false.
**Best author counterargument:** Separate predeclared configurations were used for datasets, and F is a no-test-label statistic that explains which branch to run.
**Required action:** Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
**Status:** Verified code/description mismatch | Confidence 5/5 | Reviewers: R003;R007;R037;R038;R042;R047;R049;R051;R061;R089;R098

### MATH-01 — MAJOR — Moment correction
**Location:** Claude `experiments/core.py` `MomentCalibratedGen.sample`; paper Proposition 3 / implementation notes
**Claim under review:** The implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.
**Evidence:** The raw affine scalar map is optimal for unconstrained maps matching mean and total second central moment under its stated assumptions. Implementation clips scale to [0.1,10] and applies `ReLU` after the affine transform. ReLU changes means and covariance/spread; clipping can prevent reaching the target at all. Empirical spread approaches but is not exactly 1.
**Impact:** The mathematical optimum can hold for the ideal affine map, not the implemented clipped/ReLU transform. Claims should say approximate/statistics-targeted after projection, backed by residual measurements.
**Best author counterargument:** FE features are non-negative and projection is necessary; empirical spread ratios are close to real spread.
**Required action:** State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
**Status:** Verified qualification needed | Confidence 5/5 | Reviewers: R021;R026;R027;R028;R036;R046;R052;R053;R054

### MATH-02 — MAJOR — Covariance correction / optimal transport
**Location:** Claude `experiments/core.py` `anchor_cov`, `CovCalibratedGen.sample`; Proposition 6
**Claim under review:** MC-S/MC-B corrects exact class covariance by the minimum-displacement Gelbrich map.
**Evidence:** Covariance path uses only top-k (k≤256) subspace, shrinks small-sample class covariances toward pooled or generator covariance, clips eigenvalues of A to [0.1,10], scales orthogonal residual by a scalar, and then applies ReLU. Therefore it cannot generally match the full empirical class covariance exactly nor achieve unconstrained global minimum displacement after these operations.
**Impact:** The transport derivation may justify a core subspace map before stabilizers, but not all implemented approximations as one exact global optimum. This matters most for the new MC-S/MC-B claim and high-dimensional datasets.
**Best author counterargument:** The paper distinguishes subspace covariance from total spread and regularizes tails to reduce estimator variance; MC-B results are encouraging.
**Required action:** Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
**Status:** Verified qualification needed | Confidence 5/5 | Reviewers: R021;R026;R027;R028;R036;R053;R059;R066;R086

### MATH-03 — MAJOR — Kernel herding / generalization
**Location:** Claude `docs/paper_single_idea.md` §4.4 and Lemma 1; `RFFMap`, `HerdedGen` in core
**Claim under review:** Herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.
**Evidence:** Implementation uses 6,144 random Fourier features, so the target is a finite random-feature approximation, not the exact infinite-dimensional RKHS embedding. Greedy candidate selection reduces discrepancy for the finite feature map, conditional on the candidate pool. The displayed bound requires loss in a bounded RKHS ball; the manuscript does not establish that the deep heterogeneous heads' relevant losses satisfy this assumption.
**Impact:** The finite-RFF matching is an implementable heuristic with empirical support, but the general risk-bound interpretation is conditional and may not apply to these neural classifiers.
**Best author counterargument:** The authors use a standard MMD/RKHS result and explicitly treat the head loss as a condition of the lemma.
**Required action:** Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
**Status:** Needs scoped claim | Confidence 4/5 | Reviewers: R015;R016;R021;R029;R038;R052;R055;R056;R086

### THEORY-04 — MODERATE — Synthetic budget proposition
**Location:** Claude `docs/paper_single_idea.md` §4.5/App A.5; proposition concerning T_s
**Claim under review:** A scalar estimator result explains why 10 synthetic epochs should improve neural classifier accuracy after anchoring.
**Evidence:** Proposition uses an assumed MSE formula with unbiased real samples, equal per-sample variance, synthetic label bias b, and a scalar mixing-weight estimator. Neural SGD on correlated generated features, soft labels, repeated epochs, nonconvex heads, and client-specific distributions do not directly satisfy this model. The experiments do show extra fake epochs help MNIST/FashionMNIST more than low-fidelity CIFAR/SVHN and hurt final-round accuracy in hard settings.
**Impact:** Useful qualitative intuition but not a theorem about neural optimization or an optimal epoch count. Overstated causal/explanatory language weakens theory novelty.
**Best author counterargument:** The paper describes a stylized model and reports its converse on low-fidelity datasets.
**Required action:** Call it a toy bias-variance model; articulate mapping assumptions; present empirical interaction rather than claiming derivation of T_s=10 for neural training.
**Status:** Needs claim scope | Confidence 4/5 | Reviewers: R004;R015;R021;R026;R030;R040;R052;R056;R067;R078

### NOV-01 — MAJOR — Novelty / related work
**Location:** Claude `docs/related_work.md`; `docs/paper_single_idea.md` §§4,6
**Claim under review:** Exact class statistics, class prototype anchoring, feature generation, classifier recalibration and long-tail FL provide sufficient novelty over prior art.
**Evidence:** Building blocks are established: prototype transfer (FedProto), class-conditional data-free heterogeneous FL (FedGen/FedFTG/DENSE/FedCG/GeFL), class calibration/retraining (CCVR, CReFF, Fed3R), logit adjustment/FedLC, MMD/kernel herding, Gelbrich covariance mapping, and class-wise aggregation in FL/recommendation. The strongest novelty appears to be the coupled-Adam/FedAvg failure analysis in the GeFL-F generator and putting multiple aggregate statistics to work in a heterogeneous-generator pipeline; the review should not dismiss that combination as inherently non-novel.
**Impact:** Novelty hinges on specific evidence of incremental value over simpler HWA+LA+CSL and direct recalibration/statistics-based methods; broad claims of novel building blocks won't hold.
**Best author counterargument:** Paper openly attributes each component to prior sources and claims novelty in integration and setting.
**Required action:** Add method-by-method comparison with CReFF/CCVR/Fed3R and closest prototype/statistics transfer; claim novelty of measured failure mechanism + integrated anchoring only where directly supported.
**Status:** Major comparison needed | Confidence 4/5 | Reviewers: R001;R004;R011;R012;R013;R014;R015;R016;R017;R018;R019;R020;R093

### EXP-02 — CRITICAL — Baseline fairness / readout recalibration
**Location:** Claude `docs/research/readout_report.md` §§2,6,9; paper headline tables
**Claim under review:** The large Ours-A gain over GeFL-F reflects a large incremental contribution beyond existing classifier-side fixes.
**Evidence:** New readout report shows GeFL-F final heads gain from pooled-real last-layer refit/oracle by +16.5 MNIST, +20.5 FMNIST, +14.3 SVHN points. Approximate oracle baseline levels are 90.7/79.2/64.8 versus Ours-A 94.8/79.7/65.6, margins only +4.1/+0.5/+0.8. With readout/refit applied to both methods Ours-A oracle margins +4.2/+2.4/+3.6. This oracle has pooled real training access and is not a fair deployable federated baseline, but it quantifies how much of the apparent gap is readout error.
**Impact:** Headline gains over uncorrected GeFL-F are not incremental gains over the strongest relevant classifier recalibration family. FashionMNIST/SVHN margin may be within noise.
**Best author counterargument:** The oracle is centralized/upper-bound, the paper itself says recalibration baseline must be included, and a federated exact-readout path is now tested on a reduced schedule.
**Required action:** Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
**Status:** Critical missing comparative baseline | Confidence 5/5 | Reviewers: R001;R011;R017;R020;R050;R061;R062;R066;R068;R091;R092;R093;R094

### EXP-03 — MAJOR — Incomplete experiments
**Location:** Claude `docs/PROGRESS.md` §§1,3; K14/K15 specs and T17/T18 tables
**Claim under review:** The current evidence is complete enough to substantiate all major claims.
**Evidence:** K14 (ablation seeds 3–4, MC-B validation) and K15 (AdamW/no-decay/lazy decay comparisons at K=10/50/100) are marked ready but not run. The corresponding branch has notebooks and generated tables, but 20 missing cells in T17 and 20 in T18; K09 per-component significance is still weak at n=3.
**Impact:** The exact alternative most relevant to collapse and small-component effects remains untested. Not evidence of falsity, but it prevents a fully defensible submission.
**Best author counterargument:** K14/K15 are prepared jobs and can be completed later; the one-seed theory test is mathematically valid.
**Required action:** Run K15 first, then K14. Publish the paired per-seed metrics and regenerated tables; mark all remaining arms PENDING in manuscript until run.
**Status:** Pending; decisive | Confidence 5/5 | Reviewers: R008;R021;R022;R025;R035;R040;R061;R068;R071;R072;R073;R075;R076;R078;R080;R091

### STAT-01 — MAJOR — Statistical inference
**Location:** Claude `experiments/results/K09_ablation_analysis.md`; report/handout ablation claims
**Claim under review:** Thirty negative component-removal comparisons establish an aggregate sign-test p=2e−9.
**Evidence:** The 30 differences come from 5 component removals x 2 datasets x 3 seeds, and reuse the same full-model seed runs, partitions and generated artifacts. They are correlated comparisons, not 30 independent replicates. The current branch analysis correctly says no single component remains significant after Holm correction over 10 tests at n=3 (smallest adjusted p=.085); small MC/KH/FashionMNIST-PC effects are weak.
**Impact:** Aggregate p-value massively overstates evidence if treated as independent sample count. Per-component evidence must be reported with its correct unit and multiplicity.
**Best author counterargument:** All paired differences were directionally negative; the analysis itself qualifies per-component significance and says K14 is pending.
**Required action:** Remove the global 30-pair sign-test as confirmatory evidence; report paired effects per component/dataset with uncertainty and Holm-adjusted p-values; run seeds 3–4.
**Status:** Verified reporting risk | Confidence 5/5 | Reviewers: R061;R068;R071;R072;R073;R075;R079;R080;R091;R095

### STAT-02 — MAJOR — Evaluation leakage / model selection
**Location:** `Tracker.update` in `experiments/core.py`; `best_mean_acc` columns and paper IID tables
**Claim under review:** All headline metrics are equally fair because the same metrics are used for every method.
**Evidence:** The tracker updates the best score as max over each architecture's test accuracy every evaluated round; `best_mean_acc` therefore selects the best round using test labels/metrics. PROGRESS says the paper's balanced-setting comparisons retain this metric to match the original paper, while long-tail headlines use `final_bal`. Equal leakage across methods does not make a held-out test set valid for model selection.
**Impact:** IID best-round gains are optimistic and should not be the sole evidence for generalization. It makes cross-setting tables mix different estimands if not labeled prominently.
**Best author counterargument:** The comparison intentionally reproduces GeFL-F's original reported protocol; long-tail headline uses final checkpoints.
**Required action:** Keep protocol-reproduction table labeled as historical comparison; add validation-selected checkpoints or final-round results for all methods on the same seeds. Never tune on the test set for the paper's final main table.
**Status:** Verified | Confidence 5/5 | Reviewers: R061;R065;R066;R067;R071;R074;R075;R077;R079;R096

### STAT-03 — MAJOR — Result provenance / numerical consistency
**Location:** Claude `docs/paper_assets/CHECK.md`; `docs/paper_assets/INDEX.md`; results summaries
**Claim under review:** Every quoted paper number matches its generated table and raw per-seed results.
**Evidence:** CHECK.md enumerates draft-vs-data discrepancies, including IID MNIST 96.02 vs 96.01; K=100 MNIST client rows; E16 long-tail balanced/tail numbers; T_s=10 values; CIFAR-10 CSL 60.02 vs 59.85; DDPM-F 83.10 vs 82.82. PROGRESS calls out K=50 entries 86.8/87.2/87.9 with no source and K=100 sample counts only n=1–2. E25 summary.md also has a setting/header mismatch relative to runs.jsonl; K09 CSV omits -PC arms due workers overwriting runs.csv.
**Impact:** Individual differences are usually small, but unsourced client-scaling values and inconsistent source files undermine traceability; they are fixable rather than evidence of fabrication.
**Best author counterargument:** The new branch adds automated assets and `CHECK.md` to catch these; source runs/raw JSONL may be authoritative.
**Required action:** Regenerate paper assets from merged run JSONL, remove or re-source cells with no run, fix E25 header/data grouping, write CSV only from merged JSONL, and retain commit hashes for every table.
**Status:** Verified corrections required | Confidence 5/5 | Reviewers: R008;R061;R071;R075;R076;R077;R080;R096;R098

### STAT-04 — MAJOR — Seed uncertainty
**Location:** Main tables, T01, T03, E31/E32, K09/K14
**Claim under review:** The reported small improvements are robust across randomness.
**Evidence:** Many headline runs n=3; T01 is n=1; T03/T03b are n=1 on reduced schedule; K01 CIFAR-10 paired JSONL is missing so its p-values are Welch/unpaired; MC-B has one MNIST LT seed at 92.50 vs ~95.9 others and remains pending extra seeds. Some gains are only 0.3–1 points (FMNIST baseline recalibration and ablation components).
**Impact:** Large MNIST LT gains look consistent, but small increments and mechanism claims lack strong uncertainty resolution; overprecision/significance claims should be softened.
**Best author counterargument:** Three matched seeds are normal for preliminary work and the branch labels weak evidence/pending status transparently.
**Required action:** Run planned seeds and publish mean±SD plus paired differences and CIs; distinguish high-powered effects from descriptive direction-only evidence.
**Status:** Partially resolved; remaining small effects | Confidence 5/5 | Reviewers: R061;R071;R072;R073;R075;R077;R079;R080

### PRIV-01 — MAJOR — Differential privacy
**Location:** `experiments/core.py` clipping calibration in `run_one`; E26 spec/results; paper §4.7
**Claim under review:** The reported epsilon/delta-DP statistic release is independent of private-data calibration choices and applies as stated.
**Evidence:** E26 claims a public clipping bound, but the simulation estimates it as the 90th-percentile feature norm from `ctx.part['heldout']`, built by taking 200 training-corpus examples per class out of the benchmark training split and not assigning them to a simulated client. For the benchmark corpus these may be publicly known data, but that is an explicit dataset/public-pool assumption, not an automatic deployment property. DP guarantee is conditional on treating this bound/pool as public or outside protected client records.
**Impact:** Potential privacy-accounting circularity if the bound derives from data included in the privacy unit. This is not by itself proof the benchmark E26 result violates DP because the held-out subset is disjoint from client partitions and the datasets are public; applicability to real private client records is not established.
**Best author counterargument:** The split holds the calibration pool out from all clients and uses a public benchmark; nothing from those samples is uploaded by a client.
**Required action:** Precisely specify adjacency and protected records; provide a truly public/external calibration pool or fixed data-independent norm bound; document whether the held-out sample belongs to the DP domain.
**Status:** Conditional assumption needs formal statement | Confidence 4/5 | Reviewers: R081;R082;R083;R084;R097

### PRIV-02 — MAJOR — Differential privacy scope
**Location:** Claude `docs/paper_single_idea.md` §4.7, §5.7 and limitations; E26
**Claim under review:** ESA as a full FL protocol is differentially private.
**Evidence:** The manuscript correctly states the generator's own FedAvg is not DP and describes DP on the additional anchoring statistic releases only; the formal evaluation covers MNIST/FMNIST K=10 scalar MC route at ε∈{8,2}, δ=1e−5. Other model-training updates, frozen FE process, local client heads, and standard GeFL-F generator messages are not covered by this guarantee.
**Impact:** Acceptable as a scoped claim if strictly written as DP for additional aggregate statistics, but fatal to any reading of “the full training process is DP.” Privacy-utility generalization to covariance gate and SVHN not tested.
**Best author counterargument:** Limitations explicitly disclose generator FedAvg is not DP; paper says 'every released anchoring statistic' can be privatized, not full protocol.
**Required action:** Title/table/callout should consistently say “DP of the added anchoring-statistic releases”; map every release and budget. Do not claim end-to-end DP. Run/correct MC-S/MC-B privacy scope only if claimed.
**Status:** Scope partially documented | Confidence 5/5 | Reviewers: R081;R082;R083;R084;R089;R097

### PRIV-03 — MAJOR — Differential privacy / covariance path
**Location:** `experiments/core.py` conditional MC-S path; E26; E31/K11
**Claim under review:** The DP evaluation validates all ESA variants, including covariance correction.
**Evidence:** When `_STAT_DP` is active, `run_one` bypasses `CovCalibratedGen`/MC-S and falls back to `MomentCalibratedGen` for a requested covariance route. DP raw statistics for pooled dense covariance and class subspace covariance matrices are not covered by E26's reported MNIST/FMNIST scalar-anchor experiment. E26 is useful evidence only for its configured releases/variant.
**Impact:** Do not generalize scalar-MC DP results to MC-S/MC-B or covariance anchoring; covariance route needs its own clipping sensitivity, noise, and zCDP accounting.
**Best author counterargument:** MC-S privacy is not claimed in E26; the final DP table is the scalar MC configuration.
**Required action:** State exact DP-enabled configuration in table and add tests only if DP MC-S/MC-B is claimed. Ensure every released covariance, PCA basis, and gate input is included in sensitivity/compose accounting.
**Status:** Verified scope restriction | Confidence 5/5 | Reviewers: R027;R028;R081;R083;R084;R085;R086

### PRIV-04 — MODERATE — Secure aggregation / implementation fidelity
**Location:** `experiments/core.py` `anchor_stats`, `anchor_cov`, `stage_ii`, and `run_one`
**Claim under review:** The evaluated implementation operationally uses secure aggregation so that the server sees only sums.
**Evidence:** The simulator constructs `X = torch.cat(feats)`/`Y = torch.cat(ys_list)` and centrally forms counts/sums/covariances; the research protocol describes those sums as secure-aggregated messages, but the experiment code does not execute cryptographic secure aggregation. Secure aggregation is a system assumption/protocol design, not something tested by this simulator.
**Impact:** The algorithm could be implemented atop secure aggregation, but runtime/performance evidence does not validate the cryptographic protocol, fault/dropout behavior, or server-observation model.
**Best author counterargument:** The work is algorithm research; Bonawitz-style aggregation is specified as the communication primitive and not meant to implement crypto in the research simulator.
**Required action:** Say “simulated centrally; protocol sends secure-aggregated sums in deployment.” Explicitly state server trust, thresholds, collusion assumptions, client dropout and per-message protected fields.
**Status:** Needs clarity, not an algorithm fatality | Confidence 5/5 | Reviewers: R041;R043;R047;R048;R049;R082;R089;R090;R097

### COMM-01 — MAJOR — Communication/compute cost
**Location:** Claude `docs/paper_single_idea.md` §4.1/§4.7; `anchor_cov` in core
**Claim under review:** The added communication remains about 0.3 MB apart from the small class-level terms for the main method/covariance option.
**Evidence:** Mean/spread/RFF upload is about 69,000 numbers for D=768 and C=10. Covariance mode also computes a dense pooled D×D sum and class subspace moments C×k×k (k=256), plus class projected first/second moments. A D×D float matrix is about 4.7MB at D=768 or 26.2MB at D=2560 using float32 (about twice with float64), before per-class k×k matrices and protocol overhead. “0.3MB” applies to mean/spread/RFF path only.
**Impact:** Covariance variant has materially different client upload, server memory and latency. Headline cost is underestimation if read as applying to all ESA variants.
**Best author counterargument:** The paper separately says “plus Ck² numbers when covariance is used”; dense M can perhaps be streamed or represented in a lower precision in a real protocol.
**Required action:** Give per-variant bytes per client/server, dtypes, number of rounds and whether covariance sums are split/streamed; report measured peak memory and latency for SVHN/CIFAR dimensions.
**Status:** Requires cost table correction | Confidence 4/5 | Reviewers: R039;R085;R086;R087;R088;R090

### PRIV-05 — MODERATE — Information leakage
**Location:** Paper privacy section / statistics list; secure-aggregation assumptions
**Claim under review:** Secure aggregate totals reveal nothing except safe federation-wide facts.
**Evidence:** Secure aggregation hides individual client messages from the server under assumptions, but global class counts, mean/covariance/RFF totals can expose rare class support or sensitive population properties, and single-client classes can be inferred at federation level. The handout recognizes single-client class disclosure; blanket wording “reveal nothing beyond federation-level totals” is technically tautological not a privacy guarantee.
**Impact:** Privacy characterization must distinguish secure aggregation from differential privacy and describe what global statistics reveal.
**Best author counterargument:** The DP mechanism provides a utility/privacy trade-off for these stats and secure aggregation is not claimed to hide the released global sum.
**Required action:** Narrow language to “server sees only the aggregate under secure aggregation”; quantify membership/attribute risks for rare classes and make DP the guarantee for these releases.
**Status:** Needs precise language | Confidence 4/5 | Reviewers: R081;R082;R083;R084;R097

### THEORY-05 — MODERATE — Mean-anchor separation theory
**Location:** Claude `docs/paper_single_idea.md` Proposition 2 / separation gate
**Claim under review:** Between/within ratio F with threshold 0.05 indicates when class means do not carry class identity.
**Evidence:** Reported F≈0.01 for SVHN, ≈0.14–0.60 in other datasets. The bound limits generated-class W1 separation under Lipschitz assumptions; it does not prove F<0.05 is a universally valid cutoff. The threshold is chosen after examining four datasets and is not validated on an intermediate-F or unseen dataset.
**Impact:** The separation diagnostic is useful, but the fixed threshold and automatic-selection claim may overfit these datasets; it needs a sensitivity/generalization test.
**Best author counterargument:** The paper discloses that the rule is set from four datasets and explicitly calls for intermediate separation to be tested.
**Required action:** Add a preregistered threshold sweep or held-out dataset with intermediate F, compare mean, covariance, and HWA row routes, and describe threshold as heuristic until then.
**Status:** Known limitation acknowledged; key experiment missing | Confidence 4/5 | Reviewers: R006;R007;R037;R045;R058;R064;R070;R087

### EXP-04 — MAJOR — Metrics / calibrated inference
**Location:** Claude `docs/paper_single_idea.md` Tables 1–7; core `finish`, `balanced_bias_calibration`
**Claim under review:** Headline results represent one consistent deployed method and metric.
**Evidence:** The code returns `final_bal` on trained heads, then computes `bbc_bal` as an additional calibrated-head metric; `balanced_bias_calibration` builds bias wrappers for evaluation and `bbc_applied` is recorded separately. Some prose cites raw values, some cites calibrated values (e.g., CIFAR 46.1 vs ~49). Tables mix final-round balanced accuracy for LT with `best_mean_acc` for IID reproduction. The output model/calibration setting must be exactly stated.
**Impact:** Different metrics answer different questions; comparisons become ambiguous if GeFL-F and ESA or cross-setting values use different calibration/model-selection rules.
**Best author counterargument:** Paper reports calibration as a separate metric and explains the original-paper best-round protocol.
**Required action:** Choose one primary metric per table, label raw vs BBC outputs, apply the same inference procedure to all methods where applicable, and export reproducible calibration offsets/models. Keep IID historical replication table separate.
**Status:** Fixable but submission-critical | Confidence 5/5 | Reviewers: R061;R065;R066;R067;R074;R077;R079;R089;R096;R098

### EXP-05 — MAJOR — Scope/generalization
**Location:** Main result tables; `docs/PROGRESS.md`
**Claim under review:** ESA is broadly robust across heterogeneous federated learning and long-tail settings.
**Evidence:** Primary evaluations are four image benchmarks using a shared frozen feature extractor and GeFL-F's ten-head architecture framework, long-tail IF=100/Dirichlet(0.5), balanced IID settings and some client counts. Hard-dataset and new covariance variants have fewer runs in some conditions. No non-image, changing-client, temporal drift, partial participation or unseen-feature-space result is present.
**Impact:** Claim should be scoped to this protocol. It does not establish broad FL robustness or transfer to changing clients/feature spaces.
**Best author counterargument:** Paper explicitly keeps the frozen FE and reference GeFL-F protocol to isolate this contribution and discloses the limitation.
**Required action:** Use exact scope wording, add one held-out dataset/protocol only if making broad claims, and do not extrapolate from K=100 MNIST to every large federation.
**Status:** Scope should be narrowed | Confidence 5/5 | Reviewers: R006;R041;R042;R043;R045;R046;R049;R057;R064;R070;R087;R090

### EXP-06 — MAJOR — Missing raw artifacts / branch audit
**Location:** `docs/PROGRESS.md`; `experiments/LEDGER.md`; K01/K09 result directories
**Claim under review:** All reported experimental results are reproducible from checked-in raw per-seed results.
**Evidence:** Claude branch materially improves traceability and says LEDGER covers 1,177 runs, but PROGRESS says K01 CIFAR-10 raw `runs.jsonl` was never copied back (summary only), K09 `runs.csv` omits `-PC` arms due to worker overwrite while JSONL and summary are complete, and K15/K14 raw results do not exist because runs are pending. This is a partial artifact gap, not broad absence of results.
**Impact:** Reproduce certain summary claims/tests remains difficult. Claims must be tied only to logged source data; no unsupported claim should be presumed false.
**Best author counterargument:** Raw JSONL is primary for most runs, and some results are summary-only due Kaggle transfer not because experiment never ran.
**Required action:** Recover K01 CIFAR run zip/logs if possible; regenerate CSV from merged JSONL; tag output files with source notebook/commit/job/seed; maintain branch-specific ledger.
**Status:** Partially resolved on latest branch; still open | Confidence 5/5 | Reviewers: R076;R077;R080;R096;R100

### EXP-07 — MODERATE — Ablation interpretation
**Location:** Claude `experiments/results/K09_ablation_analysis.md`; E25 data
**Claim under review:** Every ESA component makes a statistically confirmed contribution to the final method.
**Evidence:** K09 runs n=3; effects small for MC/KH and PC on FMNIST; corrected analysis says no single arm survives Holm adjustment over 10 tests (minimum adjusted p .085). All paired changes are negative, which is directional evidence, not componentwise significance. K14 is pending.
**Impact:** Need avoid implying each component is independently established, especially for minor effects. The ablation is still valuable in showing consistent direction and effect heterogeneity.
**Best author counterargument:** Updated branch accurately marks weak significance and proposes K14 for the next pass.
**Required action:** Report unadjusted and Holm-adjusted values and effect sizes. Add seeds 3–4, and predefine the family of tests. Discuss LA/BBC redundancy.
**Status:** Acknowledged but pending resolution | Confidence 5/5 | Reviewers: R040;R068;R071;R072;R073;R075;R079;R080

### EXP-08 — MODERATE — New MC-B result
**Location:** Claude `experiments/results/E32_moment_anchor_bayes/summary.md`; paper asset tables T15
**Claim under review:** MC-B is a reliable across-dataset improvement over MC/MC-S.
**Evidence:** FMNIST LT 81.57±0.31 vs prior Ours-A ~79.7 across 3 seeds, FMNIST IID `best_mean_acc` 85.75±0.45 vs paper's best 84.28. MNIST LT 94.76±1.95 includes seed 2 at 92.50, a 2.7-point loss vs the other seeds and previous method. Progress marks it promising, pending K14 seeds 3–4.
**Impact:** MC-B may be an important better covariance strategy but is not yet uniformly robust; choose carefully whether the headline final method includes it.
**Best author counterargument:** The branch presents it as promising and exposes the unstable MNIST seed.
**Required action:** Run five-seed paired comparison, fix final-method selection without test-set tuning, report tail/worst-class accuracy as well as balanced accuracy, and state conditional benefits.
**Status:** Promising; pending | Confidence 5/5 | Reviewers: R026;R028;R053;R059;R061;R065;R067;R070;R071;R075

### EXP-09 — MODERATE — Exact readout validation
**Location:** Claude `docs/research/readout_report.md`; T03/T03b preregistrations and logs
**Claim under review:** The proposed exact readout and explanation of residual bias are established under full training.
**Evidence:** ER-LDA is within 0.8 points of a pooled-real oracle in 9/9 rows but evidence here is one seed and 0.4 schedule on CPU. Original synthetic-mean mismatch hypothesis P3 failed; revised scale normalization recovered only 16–34% of readout gap. Source-label confounding is labelled leading hypothesis, not tested. Full-schedule 3-seed oracle gaps exist but do not themselves validate ER full schedule.
**Impact:** Exact readout is a potentially strong known-style baseline, but claims about mechanism/close-to-oracle federated correction need robust full-schedule evidence.
**Best author counterargument:** The report explicitly labels preregistration failures and open hypotheses and claims only preliminary validation.
**Required action:** Run full schedule and seeds on MNIST/FMNIST/SVHN/CIFAR; compare ER-LDA against oracle, BBC and current heads; keep failed H-R hypothesis out of final causal narrative.
**Status:** Partial; mechanism correctly marked open | Confidence 5/5 | Reviewers: R017;R021;R026;R028;R040;R050;R061;R071;R075;R094;R095

### NOV-02 — MODERATE — Prior art on collapse
**Location:** Claude `docs/related_work.md` and collapse explainer
**Claim under review:** Conditional class collapse itself is new.
**Evidence:** The newer branch explicitly cites Collapse by Conditioning (ICLR 2022) and distinguishes that centralized limited-data mode-collapse mechanism from the federated non-holder coupled Adam decay + FedAvg mechanism; it also adds FedAwS/FedNCF context for class/item embeddings and sparse class updates. This resolves a much broader novelty objection from the earlier main-only audit.
**Impact:** The collapse phenomenon alone is not novel; the precise optimizer-plus-aggregation mechanism may be an original, useful diagnosis if fully supported. Literature framing on Claude is improved but still must compare mechanics, not simply say no one else observes collapse.
**Best author counterargument:** Latest branch explicitly says the novelty is the federated mechanism, not class-conditional collapse itself.
**Required action:** Keep nuanced claim, cite primary sources accurately, compare update recurrences and experimental regimes, avoid novelty assertion from omission alone.
**Status:** Improved on Claude branch | Confidence 4/5 | Reviewers: R011;R012;R019;R020;R022;R038;R093

### NOV-03 — MODERATE — Build blocks / originality
**Location:** Claude `docs/paper_single_idea.md` §§4,6; related work
**Claim under review:** Combining known blocks is itself a strong enough NeurIPS novelty argument.
**Evidence:** Mean prototypes, scalar moments, Gelbrich map, random Fourier features/MMD, kernel herding, HWA, LA, consensus distillation and bias calibration are mostly established tools. The latest manuscript makes an effort to name sources and distinguish contributions, but the principled unification and measured setting-specific benefit require direct tests. CReFF overlaps directly with heterogeneous/long-tailed FL classifier recalibration.
**Impact:** A combination can be novel, but it must show why the anchoring principle is more than an engineering stack and which features uniquely contribute beyond strong/simple alternatives.
**Best author counterargument:** Paper explicitly says components exist and claims the novelty is exact-statistics anchoring throughout a generator-based heterogeneous FL pipeline.
**Required action:** Emphasize single principle, identify what stats each stage uses and why the same secure aggregate serves each, and show an ablation/strong baseline comparison isolating anchor from HWA+LA+CSL/BBC/recalibration.
**Status:** Requires comparative evidence | Confidence 4/5 | Reviewers: R001;R004;R013;R014;R015;R016;R017;R018;R019;R093

### SYS-01 — MODERATE — Client partition / data use
**Location:** Claude `experiments/core.py` `make_partition`; runtime separation diagnostic
**Claim under review:** All evaluation choices use only federated training statistics and do not inspect test data.
**Evidence:** The code's `make_partition` constructs a held-out pool from the same public benchmark training corpus but excludes it from clients. The separation statistic F uses training class sums; `ncm_acc` is computed against `feats_te` and test labels but is diagnostic only, not part of the F threshold in current code. That diagnostic should not be described as server-computable or usable at inference.
**Impact:** The current metric is not automatically a leakage flaw if it is diagnostic only, but it can confuse readers about whether a deployment gate requires test labels. The calibration pool's public status also needs the assumption in PRIV-01.
**Best author counterargument:** Code comment says NCM is diagnostic only; the intended gate is F from train sums.
**Required action:** Document that F alone is the gate statistic; keep `ncm_acc` in offline diagnosis only, do not mention it as server input or use it for thresholds.
**Status:** Needs methodology clarity | Confidence 5/5 | Reviewers: R007;R041;R047;R065;R074;R080;R089;R098

### SYS-02 — MODERATE — FedBN / heterogeneous averaging
**Location:** BatchNorm state averaging; handout limitation and code `Averager`
**Claim under review:** FedAvg of model state dicts faithfully averages client distributional state for all buffers.
**Evidence:** The handout and progress note acknowledge averaging BatchNorm running variances omits between-client variance. This is an implementation limitation especially with highly non-IID client features; the same architecture/head matching avoids averaging different head types, but BN buffer semantics still matter.
**Impact:** Potentially biases feature distributions and variance, partially confounding the exact-spread and model aggregation analysis. Likely moderate, not enough to discard main result without a control.
**Best author counterargument:** The limitation is disclosed and inherited from reference GeFL-F protocol, not unique to ESA.
**Required action:** State this in limitations and add FedBN or pooled-moment running-stat control if the argument depends on accurate variance; use same protocol for baseline/final comparisons.
**Status:** Acknowledged; needs scope control | Confidence 4/5 | Reviewers: R035;R036;R041;R042;R046;R049;R061;R086;R089

### SYS-03 — MODERATE — Large-scale systems
**Location:** MC-S/MC-B communication, server covariance construction, all experiments
**Claim under review:** Method is scalable to many clients with acceptable communication/latency based on K=100 accuracy results.
**Evidence:** K=100 results are MNIST-centered; the high-dimensional covariance route computes a dense pooled covariance and class-subspace matrices and uses a server-side eigendecomposition. No wall-clock / client memory / communication evaluation across all branches or asynchronous/partial participation. Accuracy at K=100 is not a full scalability measurement.
**Impact:** The paper can claim accuracy scales to K=100 in the tested setting, not that the complete protocol is system-scalable under broader use.
**Best author counterargument:** Methods run in T4 environments and supplement provides rough notebook times; systems engineering isn't the central contribution.
**Required action:** Separate accuracy scaling from resource scaling; report measured time/memory/bytes for D=768 and D=2560 covariance cases, and limit deployment claims.
**Status:** Needs scoped claims | Confidence 4/5 | Reviewers: R039;R043;R048;R085;R086;R087;R088;R090

### REPRO-01 — MODERATE — Reproduction protocol
**Location:** `experiments/LEDGER.md`; `experiments/ledger.py`; paper_assets/check
**Claim under review:** The full set of listed runs can be reliably recreated from the repository by rerunning scripts as is.
**Evidence:** Newest branch improves reproducibility with a run ledger and a generated paper-asset script. But notebooks require Kaggle GPU T4×2/internet in some cases, some source raw runs have not been copied back, some result artifacts are generated after the source scripts were modified, and model/dataset acquisition paths require setup. This is improved traceability, not turnkey full replication.
**Impact:** Reproducibility has improved significantly vs main but is not turnkey; reproducibility claims should identify hardware/data/checkpoints and exact commit.
**Best author counterargument:** The branch instructs to reconstruct assets and provides Kaggle notebooks/artifacts; infrastructure limitations are disclosed.
**Required action:** Add environment lockfile, one-command local smoke test, immutable run config per JSONL, exact source commit per artifact, and a clean generation check in CI.
**Status:** Improved but not complete | Confidence 4/5 | Reviewers: R076;R077;R080;R089;R096;R100

### FMT-01 — MAJOR — Submission readiness
**Location:** Current full/short paper drafts, handout and report in different branch refs
**Claim under review:** The repository contains a submission-ready, anonymized NeurIPS manuscript.
**Evidence:** The branch includes markdown working drafts, HTML reports and Word/PDF handouts; it does not in the reviewed tree provide a final anonymized NeurIPS-formatted 9-content-page PDF+checklist as prescribed in the official handbook. Main branch is stale and `claude` carries noncanonical handout/editorial artifacts; code/figures have repository owner/branch metadata that must be anonymized if linked under double-blind review.
**Impact:** Even a correct research result is not submission-ready until converted into official format, kept anonymous and supported by a single consistent supplemental ZIP. Nine-content-page rule and other submission conditions matter.
**Best author counterargument:** The repository is an in-progress research project, not a final submission package.
**Required action:** Create an anonymized submission branch/tag with final LaTeX/PDF, appendix, checklist and separate anonymized supplement; ensure citations/links and repo metadata do not reveal authors if required by review policy.
**Status:** Not ready as a submission artifact | Confidence 5/5 | Reviewers: R008;R010;R091;R092;R096;R098;R100

### METRIC-01 — MODERATE — Tail metrics
**Location:** Tables T02/T11/T13; balanced accuracy/final_tail/final_worst
**Claim under review:** Balanced accuracy alone demonstrates rare-class robustness.
**Evidence:** Some MC-B/branch reports show improved mean/balanced accuracy while worst-class accuracy can be low: E32 FMNIST long-tail worst class 33.19±4.02 vs balanced accuracy81.57; seed 2 instability on MNIST LT. The notebook/progress mentions per-class tail metrics, but headlines often emphasize balanced accuracy and omit worst-class/recall variability.
**Impact:** A high average can hide catastrophic rare-class performance; this matters directly to the motivation that “rare classes come back.”
**Best author counterargument:** Tail and worst metrics are logged in raw summary artifacts; main results include tail recall.
**Required action:** Report worst-class, mean tail recall and per-class recall distribution for final candidate, especially MC-B and SVHN, in same table and seed pairing.
**Status:** Needs stronger reporting | Confidence 5/5 | Reviewers: R044;R058;R059;R063;R065;R070;R071;R079

### GATE-02 — MODERATE — Robustness / threshold
**Location:** F threshold 0.05 and MC/MC-S selection
**Claim under review:** The gate covers intermediate mean separability and chooses the best source of class identity in unseen settings.
**Evidence:** F values across the four datasets are well separated (SVHN≈0.01, others≈0.14–0.60); this gives no empirical evidence at F near the 0.05 cutoff. Covariance on FashionMNIST may improve IID but harms MNIST LT, MC-B varies by dataset/seed. The gate criterion does not compare measured class covariance informativeness online, only mean-separation F.
**Impact:** F could be a useful flag for a bad mean anchor, but low mean-separability does not by itself prove that covariance correction or HWA is the best route.
**Best author counterargument:** The authors acknowledge an intermediate-F dataset is future work and use F solely to detect an anchor limitation.
**Required action:** Add held-out intermediate regime and evaluate candidate branches on non-test validation data; treat covariance route as heuristic and report selection error/cost.
**Status:** Acknowledged but unresolved | Confidence 4/5 | Reviewers: R006;R037;R045;R053;R058;R064;R067;R070

### CONS-01 — MODERATE — Consensus soft labels
**Location:** Claude `docs/paper_single_idea.md` §4.5 and CSL experiments F09/F10/E18
**Claim under review:** Ensemble agreement is an unbiased or generally more reliable target for generated features.
**Evidence:** The authors test Bayesian CSL and report circularity: heads trained in related settings can agree due to shared generator errors; half-one-hot/half-ensemble is a useful empirical blend but not generally unbiased. E18 rejected Bayesian CSL; E16/F10 show CSL small effects and in IID sometimes gains sub-point. Claims should remain empirical and conditional on different head errors.
**Impact:** The label method is a known distillation technique in a new feature-transfer location, not a correctness guarantee about consensus labels.
**Best author counterargument:** Paper openly reports failed Bayesian variant and uses beta=.5 due to experiments.
**Required action:** Avoid unbiased/“better label” universal claims; include error correlation/teacher diversity; report its modest effect with multiple comparisons.
**Status:** Mostly resolved by transparent negative results | Confidence 4/5 | Reviewers: R014;R017;R038;R040;R051;R056;R068;R069;R073

### HWA-01 — MODERATE — Weighting rule rationale
**Location:** Claude `docs/paper_single_idea.md` §4.1; HWA aggregator and effective number weights
**Claim under review:** Effective-number weighting is the statistically optimal holder aggregation weighting in this setting.
**Evidence:** HWA weights holders by E(n)=(1−β^n)/(1−β), motivated by a saturating effective-sample-number/variance model; branch docs say counts and uniform-holder averages were ablated, but direct proof that this weight is inverse variance for learned row estimates is not established by current loss model. In main code, weights are computed from client counts and row state dict aggregate; client-specific data/optimizer histories may create correlated, biased row estimates.
**Impact:** HWA's empirical performance can stand without an optimality claim. Stronger claims about inverse-variance optimality require a model and variance measurement.
**Best author counterargument:** It is a known weighting heuristic adapted from effective number, and weighting is compared empirically.
**Required action:** Present as motivated heuristic; compare uniform-holder and raw-count HWA under current full protocol; measure row estimation error by n_kc before claiming optimality.
**Status:** Needs modest scope correction | Confidence 4/5 | Reviewers: R013;R026;R035;R036;R039;R062

### KME-01 — LOW — Negative-result reporting
**Location:** Claude Appendix B; E24 KME-Gen; K13 VCT
**Claim under review:** Discarded methods and negative experiments are adequately represented.
**Evidence:** Latest branch reports failures: KME-Gen beats GeFL-F but trails PC+MC; VCT is worse than MixUp on long-tail and slightly above feature MixUp IID but neither reaches image-space MixUp; Bayesian CSL, lazy decay with HWA and server distillation are discarded. This is a strength, although branch reports/tables still need to be synchronized.
**Impact:** This is positive evidence of research completeness and helps narrow the best contribution to a coherent anchored stack. It should not be scored as a weakness by itself.
**Best author counterargument:** The manuscript already lists these in Appendix B and PROGRESS marks VCT negative.
**Required action:** Keep this candor; ensure paper assets include the negative-result source and don't put failed VCT in the final claimed contribution.
**Status:** Strength; no adverse correction beyond synchronization | Confidence 4/5 | Reviewers: R004;R008;R069;R091;R093;R095

### EXT-01 — MODERATE — Balanced-setting comparisons
**Location:** Claude Tables T01/T04/T11; `best_mean_acc` protocol
**Claim under review:** ESA exceeds the original paper's best of ten methods in the balanced IID setting.
**Evidence:** The numbers are promising: MNIST 97.62 vs 96.44; CIFAR10 60.93 vs 59.36; SVHN 77.07 vs 76.26; FashionMNIST Ours-A 83.96 slightly below 84.28, MC-B 85.75 above. However `best_mean_acc` selects best round by test accuracy and the comparisons span published paper results vs own runs with some reproduction deltas. Different variants (Ours-A vs Ours-A-B) must not be blended into the same claim.
**Impact:** The balanced comparison is useful but weak as a confirmatory result, particularly small margins and variant selection.
**Best author counterargument:** The paper reproduces original protocol and acknowledges small balanced gains, frozen FE differences and the best-of-ten baseline.
**Required action:** Use validation checkpoint selection for new results, clearly mark historical replication metric, compare same version on same splits, and label MC-B separately with its seed uncertainty.
**Status:** Scoped claim; improvement desirable | Confidence 4/5 | Reviewers: R001;R006;R061;R062;R064;R065;R067;R071;R074;R077

### NOV-04 — MAJOR — Recent prior art / classifier calibration
**Location:** Claude `docs/related_work.md`; `docs/paper_single_idea.md` §§4.5–6; BBC evaluation
**Claim under review:** The paper's classifier-side balanced-bias calibration is adequately distinguished from the current prior art.
**Evidence:** The current related-work draft (identified as searched 2026-10-09) discusses CCVR, CReFF and Fed3R but does not mention FedNPC, published in CVPR Findings 2026. FedNPC explicitly studies post-hoc classifier calibration for federated long-tailed learning using a virtual balanced dataset. ESA's BBC calibrates predictions using balanced anchored synthetic features, so this is a close method to compare. Primary source: https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.html.
**Impact:** A current and close calibration paper is missing from novelty framing; depending on protocol/assumptions, it may reduce the apparent originality of BBC and may be a necessary baseline. This does not negate the separate generator-anchoring contribution.
**Best author counterargument:** FedNPC is not necessarily designed for generator-based heterogeneous heads or uses the same feature-space/communication assumptions; methodological differences may remain substantial.
**Required action:** Read and cite FedNPC's primary paper; compare objective, calibration data/statistics, assumptions, privacy, and compatibility with heterogeneous private heads. Include a matched baseline if it fits the setting, or clearly explain the incompatibility.
**Status:** New likely omission; must verify fit and compare | Confidence 4/5 | Reviewers: R017;R020;R065;R066;R093;R094;R100

## 5. Mathematical audit summary

| Claim | Status | Consequence |
|---|---|---|
| Corrected conditioning-row contraction inequality | Valid as a conditional triangle-inequality bound; T01 has 0 violations in 1,200 row-rounds on one seed | Use as diagnostic theorem; do not claim the empirical check proves the Adam mechanism |
| Adam coupled-decay λ-invariance | Approximate under λ|w_j|≫ε and assumed gradient/moment behavior; observed on one seed | Specify assumptions and complete matched optimizer alternatives |
| No class row in mean-anchored generator | True for PC branch; class identity then limited by mean separability | Use F threshold as a heuristic and test an intermediate regime |
| MC scalar minimum displacement | True for ideal affine mapping under specified moments; implementation clips scale and applies ReLU | Report residual moments and narrow “exact/optimal” language |
| MC-S/MC-B Gelbrich map | Ideal PSD subspace map has a transport result; implementation shrinks covariance, clips eigenvalues, uses only top-256 subspace and ReLU | Call regularized subspace approximation; measure post-transform covariance residual |
| Kernel-herding risk bound | Valid only when loss is in a bounded RKHS ball; code uses finite RFF approximation/candidate pool | Do not claim a general neural classification-risk guarantee without assumptions |
| Synthetic-budget proposition | Valid under stylized scalar estimator assumptions, not a theorem for nonlinear SGD training | Present as intuition supported by observed interaction |

## 6. Experimental evidence ledger (highest priority)

| Claim/test | Evidence present in latest branch | Evidence strength / outstanding caveat |
|---|---|---|
| Main long-tail results | F01/E25/K08/K10/K11 raw JSONL and summaries for several settings | Generally 3 seeds on key MNIST/FMNIST settings; hard-dataset and metric/correction variants must be kept distinct |
| Collapse bound/mechanism | T01 summary and logs, 4 arms, 40 rounds, MNIST LT, seed 0 | 0 bound violations checks the inequality; one seed cannot establish robustness; K15 alternatives missing |
| Dilution without decay | T02 summary at K=10/50/100, MNIST/FMNIST, seed-0 splits | Mechanistic descriptive evidence; no multi-seed uncertainty |
| Component ablations | K09 raw JSONL/summary and K09 analysis; 3 seeds, 2 datasets | No single component passes Holm with n=3; K14 seeds 3–4 pending; 30 comparisons correlated |
| MC-B | E32 raw JSONL summary; FMNIST 3 seeds, MNIST 3 seeds | Promising FMNIST, unstable MNIST LT; K14 follow-up not run |
| Exact readout (ER-LDA) | T03/T03b preregistration/logs; one seed, schedule×0.4 CPU | Within ~0.8 of oracle in 9/9 settings; original mechanism P3 failed; full-schedule matched calibration is missing |
| DP anchoring statistics | E26 MNIST/FMNIST, K=10, ε=8/2, δ=1e−5 | Useful statistics-only DP evidence; not full protocol DP, and not MC-S/MC-B DP evaluation |
| Client scale | E22 MNIST LT K=100; K03/K08 FMNIST and related artifacts | Some main rows have missing/no raw seeds; hard dataset K=50/100 pending if claimed |
| CIFAR K01 pairing | K01 summary exists; raw run JSONL not copied back per PROGRESS | Use Welch/unpaired test only for affected CIFAR entry |
| Paper assets | `paper_assets/numbers.csv`, `INDEX.md`, `CHECK.md`; generated tables/figures | Good audit scaffolding, but `CHECK.md` lists unresolved values and summaries need regeneration |

## 7. Novelty map

| Element | Prior-work status | Defensible claim | Missing evidence/comparison |
|---|---|---|---|
| Class-conditional collapse | Class-conditioning collapse known in centralized generators (e.g. Collapse by Conditioning); related class/item embedding aggregation exists | Specific interaction of coupled Adam decay on non-holder class rows + flat FedAvg in GeFL-F appears more distinctive | K15 matched AdamW/no-decay/LCD vs HWA is the decisive missing experiment |
| HWA | Related to sparse/item-wise and class-wise aggregation approaches; effective number is established | A tailored holder-only row aggregator applied to generator conditioning parameters | Match against uniform-over-holders/raw-count weights and test same setup/compute |
| PC mean-anchor | Class prototypes/statistics transfer and conditional generation are established | Using exact federated class means as the generator’s class identity within this heterogeneous generator pipeline | Compare against closest statistics-conditioned generation/prototype methods |
| MC/MC-S/MC-B | Scalar moment matching and Gelbrich/Wasserstein covariance maps are established math | Adapted sampling-time correction anchored to aggregate statistics with data-limited covariance shrinkage | Correct exactness claims after clipping/ReLU; MC-B multi-seed stability and privacy scope |
| Kernel herding/RFF | Kernel embeddings, MMD, random Fourier features and herding are known | A practical use of aggregated RFF feature sums to calibrate synthetic pool in this pipeline | Bound only applies under RKHS-loss assumptions; ablate RFF dimension/pool multiplier |
| LA/CSL/BBC | Known logit adjustment and model distillation/prior calibration | Integration in the staged generator-head protocol, not individually new | Show net gain with same calibrated metric; LA marginal effect overlaps BBC |
| Exact readout / ER-LDA | Classifier recalibration from federated feature stats is established (CReFF/CCVR/Fed3R-related family) | Cross-architecture embedding-statistics readout and readout-gap analysis may be useful extensions | Full-schedule matched CReFF/CCVR/Fed3R and both-sides recalibration baseline missing |

### Primary literature links used for the novelty audit

- [CReFF — Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features, IJCAI 2022](https://www.ijcai.org/proceedings/2022/308).
- [CCVR — No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data, NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html).
- [Fed3R — Accelerating Heterogeneous Federated Learning with Closed-form Classifiers, ICML 2024](https://proceedings.mlr.press/v235/fani-24a.html).
- [FedNPC — Stochastic Noise-driven Post-hoc Classifier Calibration Method for Federated Long-tailed Learning, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_FedNPC_Stochastic_Noise-driven_Post-hoc_Classifier_Calibration_Method_for_Federated_Long-tailed_CVPRF_2026_paper.html).
- [Collapse by Conditioning, ICLR 2022](https://openreview.net/pdf?id=7TZeCsNOUB_).

This search identified FedNPC as a new likely missing citation/baseline compared with the October 9 related-work search documented in the branch. It is directly relevant to BBC-style post-hoc calibration, though compatibility with the generator-based heterogeneous-head protocol must be checked before calling it a fair drop-in baseline.

## 8. Cross-examination: resolved vs surviving criticisms

**Withdraw / narrow older objection:** “The current paper still uses the old K/2 theorem.” This is true of `main`'s old `paper_single_idea.md`, but false of the latest Claude branch, which replaced it with the gain/loss threshold. The all-branch audit therefore treats stale main as a branch governance problem, not as the current branch's theorem.

**Narrow rather than reject:** the row norm inequality itself is valid. The unsupported leap is treating its empirical verification as a proof of Adam’s full mechanism or of a universal λ-invariant step. Mean/spread/covariance map theorems are correct for ideal transforms under assumptions; their implemented post-ReLU/clipped variants need residual evidence.

**Survives cross-examination:** (1) K15 optimizer alternatives not run; (2) missing matched classifier-recalibration baseline despite an oracle showing readout dominates GeFL-F’s gap; (3) paper's automatic F gate does not select generator type in core runtime; (4) raw/generated tables conflict; (5) DP scope does not encompass the full FL protocol; (6) covariance resource cost/guarantees differ from the simple mean/RFF path.

**Potential author response:** Authors can legitimately state that (i) DP claim is only for added statistics and they disclose non-DP generator FedAvg; (ii) centralized secure-aggregation simulation is a methodological prototype; (iii) the oracle is not a deployable baseline but motivates a federated readout baseline; and (iv) only one seed tests T01. Those caveats mitigate overclaiming but do not substitute for the K15/ER experiments or one canonical branch.

## 9. Review dashboard

Scores use the actual NeurIPS 2026 scale (Quality, Clarity, Significance, Originality: 1–4; Overall: 1–6; Confidence: 1–5). Contribution type most plausibly **General** (empirical method with theory and application motivation), while reviewers should interpret criteria under official guidelines. The numerical panel is simulated and correlated; do not interpret fractions as a probability of acceptance.

| Criterion | N | Mean | Median | Mode | Range | Q1–Q3 | IQR | Count distribution |
|---|---:|---:|---:|---:|---|---|---:|---|
| quality | 100 | 1.9 | 2.0 | 2 | 1–3 | 2.0–2.0 | 0.0 | `{"1": 20, "2": 70, "3": 10}` |
| clarity | 100 | 2.1 | 2.0 | 2 | 1–3 | 2.0–2.0 | 0.0 | `{"1": 10, "2": 70, "3": 20}` |
| significance | 100 | 2.6 | 3.0 | 3 | 1–4 | 2.0–3.0 | 1.0 | `{"1": 5, "2": 40, "3": 45, "4": 10}` |
| originality | 100 | 2 | 2.0 | 2 | 1–3 | 2.0–2.0 | 0.0 | `{"1": 10, "2": 80, "3": 10}` |
| overall | 100 | 2.04 | 2.0 | 2 | 1–3 | 2.0–2.0 | 0.0 | `{"1": 6, "2": 84, "3": 10}` |
| confidence | 100 | 4.11 | 4.0 | 4 | 3–5 | 4.0–4.0 | 0.0 | `{"3": 10, "4": 69, "5": 21}` |

### By panel

| Panel | Quality mean | Clarity mean | Significance mean | Originality mean | Overall mean | Overall distribution |
|---|---:|---:|---:|---:|---:|---|
| A — General NeurIPS reviewers and contribution assessment | 1.9 | 2.1 | 3.1 | 2 | 2.1 | `{"2": 9, "3": 1}` |
| B — Originality and literature review | 1.9 | 2.1 | 2.1 | 2 | 2 | `{"2": 10}` |
| C — Mathematical proof and theoretical foundations | 1.9 | 2.1 | 2.1 | 2 | 1.7 | `{"1": 3, "2": 7}` |
| D — Optimization and algorithmic behavior | 1.9 | 2.1 | 3.1 | 2 | 2 | `{"2": 10}` |
| E — Federated system design and heterogeneity | 1.9 | 2.1 | 3.1 | 2 | 2.3 | `{"2": 7, "3": 3}` |
| F — Generative models, distributions and statistics | 1.9 | 2.1 | 3.1 | 2 | 2.3 | `{"2": 7, "3": 3}` |
| G — Experimental methodology and benchmark quality | 1.9 | 2.1 | 3.1 | 2 | 2.3 | `{"2": 7, "3": 3}` |
| H — Statistical reliability and reproducibility | 1.9 | 2.1 | 2.1 | 2 | 2 | `{"2": 10}` |
| I — Privacy, communication, complexity and systems | 1.9 | 2.1 | 2.1 | 2 | 1.7 | `{"1": 3, "2": 7}` |
| J — Meta-review, ethics, communication and final decision | 1.9 | 2.1 | 2.1 | 2 | 2 | `{"2": 10}` |

### Overall recommendation distribution

`{"1": 6, "2": 84, "3": 10}`. The scores are assigned simulated assessments, not actual conference votes.

## 10. One hundred individual simulated reviews

Each reviewer below has a unique role/focus, strengths, evidence-tied concern, score set, confidence, and actionable question. Reviewers may share the same evidence issue; the issue database deduplicates it. The use of shared base evidence means these are **not statistically independent human reviews**.

### R001 — A — General NeurIPS reviewers and contribution assessment — Senior general ML reviewer
**Specialty focus:** Overall technical quality, originality, significance and completeness.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the senior general ml reviewer perspective, the key question is overall technical quality, originality, significance and completeness. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** Main contribution is promising but paper state is not canonical across refs. Primary issue(s): BR-01, EXP-02, EXP-03. Evidence status: Verified, Critical missing comparative baseline, Pending; decisive. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** BR-01, EXP-02, EXP-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What commit/PDF is the intended submission; can all claims be regenerated from it?
2. Can the authors provide a traceable result or proof resolving current paper and supporting evidence are in a single unambiguous, reviewable snapshot.? Minimum evidence requested: Before submission, nominate one canonical commit and merge or tag it; archive older drafts; regenerate all tables and ledgers at that commit.
3. What preregistered test on the actual implementation would falsify the concern about overall technical quality, originality, significance and completeness, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about branch/version control, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) BR-01, EXP-02, EXP-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R002 — A — General NeurIPS reviewers and contribution assessment — Federated-learning generalist
**Specialty focus:** Meaningful FL problem and progress beyond standard approaches.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the federated-learning generalist perspective, the key question is meaningful fl problem and progress beyond standard approaches. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** This is a relevant long-tail problem for heterogeneous heads; direct calibration baselines are decisive. Primary issue(s): EXP-02, EXP-03, ALG-01. Evidence status: Critical missing comparative baseline, Pending; decisive, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, EXP-03, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What happens against matched CReFF/CCVR/Fed3R under the same feature-freezing budget?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about meaningful fl problem and progress beyond standard approaches, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, EXP-03, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R003 — A — General NeurIPS reviewers and contribution assessment — Machine-learning methodology reviewer
**Specialty focus:** Principled method and adequate evidence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the machine-learning methodology reviewer perspective, the key question is principled method and adequate evidence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** Exact statistics align the pipeline, but each correction needs its own operational definition. Primary issue(s): EXP-03, ALG-01, TH-01. Evidence status: Pending; decisive, Verified code/description mismatch, Reframed; still needs careful claims. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-03, ALG-01, TH-01.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Which accuracy number is the declared primary metric: raw final or BBC calibrated?
2. Can the authors provide a traceable result or proof resolving the current evidence is complete enough to substantiate all major claims.? Minimum evidence requested: Run K15 first, then K14. Publish the paired per-seed metrics and regenerated tables; mark all remaining arms PENDING in manuscript until run.
3. What preregistered test on the actual implementation would falsify the concern about principled method and adequate evidence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about incomplete experiments, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-03, ALG-01, TH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R004 — A — General NeurIPS reviewers and contribution assessment — Scientific contribution reviewer
**Specialty focus:** Scientific discovery vs engineering complexity.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the scientific contribution reviewer perspective, the key question is scientific discovery vs engineering complexity. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** The strongest discovery may be the optimizer/aggregation diagnosis, not the component count. Primary issue(s): ALG-01, TH-01, BR-01. Evidence status: Verified code/description mismatch, Reframed; still needs careful claims, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, TH-01, BR-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What remains new once HWA+LA+CSL and known recalibration are included?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about scientific discovery vs engineering complexity, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, TH-01, BR-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R005 — A — General NeurIPS reviewers and contribution assessment — Significance reviewer
**Specialty focus:** Problem importance and field impact.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the significance reviewer perspective, the key question is problem importance and field impact. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** Long-tail gains are large on MNIST/FashionMNIST, smaller after readout calibration on some sets. Primary issue(s): TH-01, BR-01, EXP-02. Evidence status: Reframed; still needs careful claims, Verified, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, BR-01, EXP-02.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 3/5.
**E. Author questions:**
1. Are +0.5 FMNIST and +0.8 SVHN gains distinguishable from paired-seed noise?
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about problem importance and field impact, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) TH-01, BR-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R006 — A — General NeurIPS reviewers and contribution assessment — Scope and generalization reviewer
**Specialty focus:** Breadth of claims vs tested settings.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the scope and generalization reviewer perspective, the key question is breadth of claims vs tested settings. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** Four image datasets and a frozen FE constrain generality. Primary issue(s): BR-01, EXP-02, EXP-03. Evidence status: Verified, Critical missing comparative baseline, Pending; decisive. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** BR-01, EXP-02, EXP-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can the rule be validated on an unseen/intermediate-F dataset?
2. Can the authors provide a traceable result or proof resolving current paper and supporting evidence are in a single unambiguous, reviewable snapshot.? Minimum evidence requested: Before submission, nominate one canonical commit and merge or tag it; archive older drafts; regenerate all tables and ledgers at that commit.
3. What preregistered test on the actual implementation would falsify the concern about breadth of claims vs tested settings, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about branch/version control, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) BR-01, EXP-02, EXP-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R007 — A — General NeurIPS reviewers and contribution assessment — Problem-formulation reviewer
**Specialty focus:** Problem assumptions and formalization.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the problem-formulation reviewer perspective, the key question is problem assumptions and formalization. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** The revised row threshold is a valid conditional norm bound, not a future predictor on its own. Primary issue(s): EXP-02, EXP-03, ALG-01. Evidence status: Critical missing comparative baseline, Pending; decisive, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, EXP-03, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. How are Gbar/Dbar estimated before training, rather than read after each round?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about problem assumptions and formalization, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, EXP-03, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R008 — A — General NeurIPS reviewers and contribution assessment — Research completeness reviewer
**Specialty focus:** Missing elements that block defensible contribution.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the research completeness reviewer perspective, the key question is missing elements that block defensible contribution. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** The new branch improves reporting but leaves K14/K15 pending. Primary issue(s): EXP-03, ALG-01, TH-01. Evidence status: Pending; decisive, Verified code/description mismatch, Reframed; still needs careful claims. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-03, ALG-01, TH-01.
**D. Ratings:** Quality 1/4; Clarity 1/4; Significance 2/4; Originality 1/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Will the critical alternatives and missing cells be run before main-track submission?
2. Can the authors provide a traceable result or proof resolving the current evidence is complete enough to substantiate all major claims.? Minimum evidence requested: Run K15 first, then K14. Publish the paired per-seed metrics and regenerated tables; mark all remaining arms PENDING in manuscript until run.
3. What preregistered test on the actual implementation would falsify the concern about missing elements that block defensible contribution, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about incomplete experiments, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-03, ALG-01, TH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R009 — A — General NeurIPS reviewers and contribution assessment — Skeptical but fair reviewer
**Specialty focus:** Strongest valid objection while checking fixes.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the skeptical but fair reviewer perspective, the key question is strongest valid objection while checking fixes. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** The strongest rejection case must not misstate the revised theorem as still the old K/2 theorem. Primary issue(s): ALG-01, TH-01, BR-01. Evidence status: Verified code/description mismatch, Reframed; still needs careful claims, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, TH-01, BR-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can the team present the bound as conditional and prove the separate Adam mechanism?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about strongest valid objection while checking fixes, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, TH-01, BR-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R010 — A — General NeurIPS reviewers and contribution assessment — Senior area-chair perspective
**Specialty focus:** Synthesis of four criteria into decision.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the senior area-chair perspective perspective, the key question is synthesis of four criteria into decision. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The empirical problem is concrete, and latest branch reports strong long-tail recovery with informative negative results.
**C. Weaknesses:** An overall reject is based on still-pending decisive comparisons and branch ambiguity, not on the research idea being worthless. Primary issue(s): TH-01, BR-01, EXP-02. Evidence status: Reframed; still needs careful claims, Verified, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, BR-01, EXP-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What minimum evidence can resolve branch state, gate implementation and recalibration baseline?
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about synthesis of four criteria into decision, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-01, BR-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R011 — B — Originality and literature review — Federated long-tail learning specialist
**Specialty focus:** Class imbalance / label skew prior art.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the federated long-tail learning specialist perspective, the key question is class imbalance / label skew prior art. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** The newer related-work section correctly distinguishes general conditional collapse from the FL-specific mechanism. Primary issue(s): NOV-01, NOV-02, NOV-03. Evidence status: Major comparison needed, Improved on Claude branch, Requires comparative evidence. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-01, NOV-02, NOV-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What exact update recurrence is novel compared with Collapse by Conditioning/FedAwS/FedNCF?
2. Can the authors provide a traceable result or proof resolving exact class statistics, class prototype anchoring, feature generation, classifier recalibration and long-tail fl provide sufficient novelty over prior art.? Minimum evidence requested: Add method-by-method comparison with CReFF/CCVR/Fed3R and closest prototype/statistics transfer; claim novelty of measured failure mechanism + integrated anchoring only where directly supported.
3. What preregistered test on the actual implementation would falsify the concern about class imbalance / label skew prior art, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about novelty / related work, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-01, NOV-02, NOV-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R012 — B — Originality and literature review — Heterogeneous FL specialist
**Specialty focus:** Cross-architecture knowledge transfer.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the heterogeneous fl specialist perspective, the key question is cross-architecture knowledge transfer. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** CReFF and Fed3R make cross-architecture readout comparisons highly relevant. Primary issue(s): NOV-02, NOV-03, EXP-02. Evidence status: Improved on Claude branch, Requires comparative evidence, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-02, NOV-03, EXP-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 3/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can the same hidden representation protocol support a matched readout baseline?
2. Can the authors provide a traceable result or proof resolving conditional class collapse itself is new.? Minimum evidence requested: Keep nuanced claim, cite primary sources accurately, compare update recurrences and experimental regimes, avoid novelty assertion from omission alone.
3. What preregistered test on the actual implementation would falsify the concern about cross-architecture knowledge transfer, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about prior art on collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-02, NOV-03, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R013 — B — Originality and literature review — Class-aware aggregation specialist
**Specialty focus:** Per-class aggregation, prototypes and priors.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the class-aware aggregation specialist perspective, the key question is per-class aggregation, prototypes and priors. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Holder-only parameter averaging resembles sparse per-item/label aggregation in other areas. Primary issue(s): NOV-03, EXP-02, MATH-03. Evidence status: Requires comparative evidence, Critical missing comparative baseline, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-03, EXP-02, MATH-03.
**D. Ratings:** Quality 3/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What distinguishes HWA mathematically from classwise aggregation methods?
2. Can the authors provide a traceable result or proof resolving combining known blocks is itself a strong enough neurips novelty argument.? Minimum evidence requested: Emphasize single principle, identify what stats each stage uses and why the same secure aggregate serves each, and show an ablation/strong baseline comparison isolating anchor from HWA+LA+CSL/BBC/recalibration.
3. What preregistered test on the actual implementation would falsify the concern about per-class aggregation, prototypes and priors, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects moderate concerns about build blocks / originality, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-03, EXP-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R014 — B — Originality and literature review — Synthetic-data FL specialist
**Specialty focus:** Generator and feature-transfer prior art.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the synthetic-data fl specialist perspective, the key question is generator and feature-transfer prior art. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Generator-based FL already includes FedGen/FedFTG/DENSE/FedCG/GeFL-family approaches. Primary issue(s): EXP-02, MATH-03, NOV-01. Evidence status: Critical missing comparative baseline, Needs scoped claim, Major comparison needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, MATH-03, NOV-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Which methods were implemented under identical feature-level protocol and which are literature-only?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about generator and feature-transfer prior art, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, MATH-03, NOV-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R015 — B — Originality and literature review — Distribution matching specialist
**Specialty focus:** Moment matching, optimal transport and alignment.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the distribution matching specialist perspective, the key question is moment matching, optimal transport and alignment. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Moment maps and Gelbrich geometry are known; the contribution is context and integration. Primary issue(s): MATH-03, NOV-01, NOV-02. Evidence status: Needs scoped claim, Major comparison needed, Improved on Claude branch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-03, NOV-01, NOV-02.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Is the exact map theorem about unconstrained affine transforms or post-ReLU samples?
2. Can the authors provide a traceable result or proof resolving herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.? Minimum evidence requested: Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
3. What preregistered test on the actual implementation would falsify the concern about moment matching, optimal transport and alignment, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about kernel herding / generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-03, NOV-01, NOV-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R016 — B — Originality and literature review — Kernel-methods specialist
**Specialty focus:** MMD/RKHS/kernel herding novelty.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the kernel-methods specialist perspective, the key question is mmd/rkhs/kernel herding novelty. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Kernel herding and MMD are standard tools; RFF makes the computed target approximate. Primary issue(s): NOV-01, NOV-02, NOV-03. Evidence status: Major comparison needed, Improved on Claude branch, Requires comparative evidence. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-01, NOV-02, NOV-03.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 1/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you state the risk guarantee without asserting the network loss is in a bounded RKHS ball?
2. Can the authors provide a traceable result or proof resolving exact class statistics, class prototype anchoring, feature generation, classifier recalibration and long-tail fl provide sufficient novelty over prior art.? Minimum evidence requested: Add method-by-method comparison with CReFF/CCVR/Fed3R and closest prototype/statistics transfer; claim novelty of measured failure mechanism + integrated anchoring only where directly supported.
3. What preregistered test on the actual implementation would falsify the concern about mmd/rkhs/kernel herding novelty, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about novelty / related work, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-01, NOV-02, NOV-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R017 — B — Originality and literature review — Calibration specialist
**Specialty focus:** LA, BBC, CSL and classifier calibration.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the calibration specialist perspective, the key question is la, bbc, csl and classifier calibration. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Recent FedNPC (CVPR Findings 2026) is directly relevant to post-hoc classifier calibration in federated long-tail learning, but is absent from the related-work draft. Primary issue(s): NOV-02, NOV-03, EXP-02, NOV-04. Evidence status: Improved on Claude branch, Requires comparative evidence, Critical missing comparative baseline, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-02, NOV-03, EXP-02, NOV-04.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. How does BBC compare to FedNPC’s virtual-balanced-set calibration and to CCVR/CReFF/Fed3R under the same heterogeneous-head protocol?
2. Can the authors provide a traceable result or proof resolving conditional class collapse itself is new.? Minimum evidence requested: Keep nuanced claim, cite primary sources accurately, compare update recurrences and experimental regimes, avoid novelty assertion from omission alone.
3. What preregistered test on the actual implementation would falsify the concern about la, bbc, csl and classifier calibration, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about prior art on collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-02, NOV-03, EXP-02, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R018 — B — Originality and literature review — Statistics-conditioned generation specialist
**Specialty focus:** Means, moments and sufficient statistics as conditions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the statistics-conditioned generation specialist perspective, the key question is means, moments and sufficient statistics as conditions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** PC-VAE conditioning on prototypes has close conceptually related work. Primary issue(s): NOV-03, EXP-02, MATH-03. Evidence status: Requires comparative evidence, Critical missing comparative baseline, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** NOV-03, EXP-02, MATH-03.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 1/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which primary statistics-conditioned generator is the closest competitor, and why does it fail?
2. Can the authors provide a traceable result or proof resolving combining known blocks is itself a strong enough neurips novelty argument.? Minimum evidence requested: Emphasize single principle, identify what stats each stage uses and why the same secure aggregate serves each, and show an ablation/strong baseline comparison isolating anchor from HWA+LA+CSL/BBC/recalibration.
3. What preregistered test on the actual implementation would falsify the concern about means, moments and sufficient statistics as conditions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects moderate concerns about build blocks / originality, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 1/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) NOV-03, EXP-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R019 — B — Originality and literature review — Theoretical novelty reviewer
**Specialty focus:** What propositions are genuinely new.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the theoretical novelty reviewer perspective, the key question is what propositions are genuinely new. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** The contraction inequality is an application of triangle inequality; empirical mechanism may carry originality. Primary issue(s): EXP-02, MATH-03, NOV-01. Evidence status: Critical missing comparative baseline, Needs scoped claim, Major comparison needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, MATH-03, NOV-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which theoretical claim is nontrivial enough to be a main contribution?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about what propositions are genuinely new, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, MATH-03, NOV-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R020 — B — Originality and literature review — Citation and priority auditor
**Specialty focus:** Primary sources, dates and completeness.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the citation and priority auditor perspective, the key question is primary sources, dates and completeness. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The newer branch is notably more candid about known building blocks and distinguishes the federated failure mechanism from general conditioning collapse.
**C. Weaknesses:** Related work improved on Claude, but many references need exact, mechanism-level comparison. Primary issue(s): MATH-03, NOV-01, NOV-02, NOV-04. Evidence status: Needs scoped claim, Major comparison needed, Improved on Claude branch, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-03, NOV-01, NOV-02, NOV-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can citations support every ‘first’ or ‘not used before’ assertion, especially stats-based transfer?
2. Can the authors provide a traceable result or proof resolving herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.? Minimum evidence requested: Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
3. What preregistered test on the actual implementation would falsify the concern about primary sources, dates and completeness, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about kernel herding / generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-03, NOV-01, NOV-02, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R021 — C — Mathematical proof and theoretical foundations — Mathematical proof auditor
**Specialty focus:** Independently reconstruct central proposition proofs.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the mathematical proof auditor perspective, the key question is independently reconstruct central proposition proofs. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** The draft now has a corrected conditional bound, but its novelty is modest absent the measured recurrence. Primary issue(s): TH-01, TH-02, MATH-01. Evidence status: Reframed; still needs careful claims, Partially supported; K15 pending, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, TH-02, MATH-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Separate a general norm inequality from the empirical update model and specify which part is theorem.
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about independently reconstruct central proposition proofs, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-01, TH-02, MATH-01 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R022 — C — Mathematical proof and theoretical foundations — Adaptive optimization theorist
**Specialty focus:** Adam/coupled decay/local federated state.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the adaptive optimization theorist perspective, the key question is adam/coupled decay/local federated state. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Adam's approximate scale invariance depends on epsilon and history; persistent moments deserve analysis. Primary issue(s): TH-02, MATH-01, MATH-02. Evidence status: Partially supported; K15 pending, Verified qualification needed, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-02, MATH-01, MATH-02.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 1/6; Confidence 4/5.
**E. Author questions:**
1. Can you derive a bound with nonzero epsilon and cross-round moments?
2. Can the authors provide a traceable result or proof resolving any nonzero coupled adam weight-decay coefficient produces an approximately λ-independent full learning-rate-sized row shrink over all relevant non-holder updates.? Minimum evidence requested: State the per-coordinate assumptions and optimizer-state conditions; report T01 as one-seed mechanism evidence. Run K15's matched AdamW/no-decay/lazy-decay comparisons before claiming HWA is the best fix.
3. What preregistered test on the actual implementation would falsify the concern about adam/coupled decay/local federated state, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / adam, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) TH-02, MATH-01, MATH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R023 — C — Mathematical proof and theoretical foundations — FedAvg theorist
**Specialty focus:** Local updates, aggregation, rare-class recurrence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the fedavg theorist perspective, the key question is local updates, aggregation, rare-class recurrence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** FedAvg averaging inequality is sound under definitions; the contraction condition is data-dependent. Primary issue(s): MATH-01, MATH-02, MATH-03. Evidence status: Verified qualification needed, Verified qualification needed, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, MATH-03.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 1/6; Confidence 3/5.
**E. Author questions:**
1. Does the sign of contraction persist if holders/nonholders have non-collinear row directions?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about local updates, aggregation, rare-class recurrence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R024 — C — Mathematical proof and theoretical foundations — Stochastic optimization specialist
**Specialty focus:** Gradient noise, optimizer state and convergence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the stochastic optimization specialist perspective, the key question is gradient noise, optimizer state and convergence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Sparse gradients/optimizer state can violate simplistic η-per-step narrative. Primary issue(s): MATH-02, MATH-03, THEORY-04. Evidence status: Verified qualification needed, Needs scoped claim, Needs claim scope. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-02, MATH-03, THEORY-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 1/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can T01 measure per-coordinate steps and moment-state norms across more seeds?
2. Can the authors provide a traceable result or proof resolving mc-s/mc-b corrects exact class covariance by the minimum-displacement gelbrich map.? Minimum evidence requested: Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
3. What preregistered test on the actual implementation would falsify the concern about gradient noise, optimizer state and convergence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about covariance correction / optimal transport, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-02, MATH-03, THEORY-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R025 — C — Mathematical proof and theoretical foundations — Counterexample specialist
**Specialty focus:** Minimal counterexamples and boundary cases.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the counterexample specialist perspective, the key question is minimal counterexamples and boundary cases. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** A counterexample at epsilon-dominated coordinates undermines a universal Adam claim, not the observed result. Primary issue(s): MATH-03, THEORY-04, TH-01. Evidence status: Needs scoped claim, Needs claim scope, Reframed; still needs careful claims. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-03, THEORY-04, TH-01.
**D. Ratings:** Quality 1/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 1/6; Confidence 4/5.
**E. Author questions:**
1. How do results change for rows with λ|w| comparable to optimizer epsilon?
2. Can the authors provide a traceable result or proof resolving herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.? Minimum evidence requested: Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
3. What preregistered test on the actual implementation would falsify the concern about minimal counterexamples and boundary cases, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about kernel herding / generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) MATH-03, THEORY-04, TH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R026 — C — Mathematical proof and theoretical foundations — Statistical estimation theorist
**Specialty focus:** Finite-sample bias/variance of class statistics.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the statistical estimation theorist perspective, the key question is finite-sample bias/variance of class statistics. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Noisy and small-class estimates introduce finite-sample error in covariance/fingerprint releases. Primary issue(s): THEORY-04, TH-01, TH-02. Evidence status: Needs claim scope, Reframed; still needs careful claims, Partially supported; K15 pending. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** THEORY-04, TH-01, TH-02.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. What uncertainty bounds hold when n_c is 24 or the private count becomes near zero?
2. Can the authors provide a traceable result or proof resolving a scalar estimator result explains why 10 synthetic epochs should improve neural classifier accuracy after anchoring.? Minimum evidence requested: Call it a toy bias-variance model; articulate mapping assumptions; present empirical interaction rather than claiming derivation of T_s=10 for neural training.
3. What preregistered test on the actual implementation would falsify the concern about finite-sample bias/variance of class statistics, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about synthetic budget proposition, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) THEORY-04, TH-01, TH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R027 — C — Mathematical proof and theoretical foundations — Optimal transport theorist
**Specialty focus:** Gelbrich/minimum displacement assumptions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the optimal transport theorist perspective, the key question is gelbrich/minimum displacement assumptions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Gelbrich minimum displacement is a sound ideal theorem; implementation clipping/projection alters it. Primary issue(s): TH-01, TH-02, MATH-01. Evidence status: Reframed; still needs careful claims, Partially supported; K15 pending, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, TH-02, MATH-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you quantify post-transform mean/covariance residual and state a narrower optimality scope?
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about gelbrich/minimum displacement assumptions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-01, TH-02, MATH-01 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R028 — C — Mathematical proof and theoretical foundations — Covariance mathematics specialist
**Specialty focus:** PSD, singular covariance, shrinkage and dimensions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the covariance mathematics specialist perspective, the key question is psd, singular covariance, shrinkage and dimensions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Covariance maps are stabilized in a low-dimensional subspace and with eigen clipping. Primary issue(s): TH-02, MATH-01, MATH-02. Evidence status: Partially supported; K15 pending, Verified qualification needed, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-02, MATH-01, MATH-02.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 1/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What fraction of true covariance is preserved after ReLU and subspace approximation?
2. Can the authors provide a traceable result or proof resolving any nonzero coupled adam weight-decay coefficient produces an approximately λ-independent full learning-rate-sized row shrink over all relevant non-holder updates.? Minimum evidence requested: State the per-coordinate assumptions and optimizer-state conditions; report T01 as one-seed mechanism evidence. Run K15's matched AdamW/no-decay/lazy-decay comparisons before claiming HWA is the best fix.
3. What preregistered test on the actual implementation would falsify the concern about psd, singular covariance, shrinkage and dimensions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about theory / adam, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 1/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-02, MATH-01, MATH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R029 — C — Mathematical proof and theoretical foundations — RKHS/generalization theorist
**Specialty focus:** Herding bound and classifier-risk assumptions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the rkhs/generalization theorist perspective, the key question is herding bound and classifier-risk assumptions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** The RKHS loss ball assumption is missing for heterogeneous deep heads. Primary issue(s): MATH-01, MATH-02, MATH-03. Evidence status: Verified qualification needed, Verified qualification needed, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, MATH-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you bound the loss norm for the actual head architecture or present the lemma as conditional intuition?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about herding bound and classifier-risk assumptions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R030 — C — Mathematical proof and theoretical foundations — Assumption-strength auditor
**Specialty focus:** Whether assumptions overrestrict or fail intended scope.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the assumption-strength auditor perspective, the key question is whether assumptions overrestrict or fail intended scope. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The revised conditional norm bound is mathematically defensible under its definitions; several transport results are valid for idealized maps when stated narrowly.
**C. Weaknesses:** Theoretical conditions may be almost tautological or not applicable to code. Primary issue(s): MATH-02, MATH-03, THEORY-04. Evidence status: Verified qualification needed, Needs scoped claim, Needs claim scope. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-02, MATH-03, THEORY-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which claims are fully proved for the implemented pipeline rather than its unconstrained idealized operator?
2. Can the authors provide a traceable result or proof resolving mc-s/mc-b corrects exact class covariance by the minimum-displacement gelbrich map.? Minimum evidence requested: Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
3. What preregistered test on the actual implementation would falsify the concern about whether assumptions overrestrict or fail intended scope, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about covariance correction / optimal transport, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-02, MATH-03, THEORY-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R031 — D — Optimization and algorithmic behavior — Optimizer dynamics specialist
**Specialty focus:** Evolution of rare conditioning rows.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the optimizer dynamics specialist perspective, the key question is evolution of rare conditioning rows. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** T01 is informative, but row norms do not fully capture orientation/cosine cancellation. Primary issue(s): TH-02, EXP-01, ALG-01. Evidence status: Partially supported; K15 pending, Pending; critical, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-02, EXP-01, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 4/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Are the observed contractions caused by norm loss, angular disagreement, or both?
2. Can the authors provide a traceable result or proof resolving any nonzero coupled adam weight-decay coefficient produces an approximately λ-independent full learning-rate-sized row shrink over all relevant non-holder updates.? Minimum evidence requested: State the per-coordinate assumptions and optimizer-state conditions; report T01 as one-seed mechanism evidence. Run K15's matched AdamW/no-decay/lazy-decay comparisons before claiming HWA is the best fix.
3. What preregistered test on the actual implementation would falsify the concern about evolution of rare conditioning rows, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / adam, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-02, EXP-01, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R032 — D — Optimization and algorithmic behavior — Weight-decay specialist
**Specialty focus:** Coupled vs decoupled decay and code match.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the weight-decay specialist perspective, the key question is coupled vs decoupled decay and code match. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Coupled vs decoupled decay is central and needs matched baseline evidence. Primary issue(s): EXP-01, ALG-01, MATH-01. Evidence status: Pending; critical, Verified code/description mismatch, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, ALG-01, MATH-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 1/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Where are K15 results for AdamW at matched decay and no decay on the row only?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about coupled vs decoupled decay and code match, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, ALG-01, MATH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R033 — D — Optimization and algorithmic behavior — Local training dynamics specialist
**Specialty focus:** Missing classes, local steps and client data.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the local training dynamics specialist perspective, the key question is missing classes, local steps and client data. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Non-holder batches have no class gradient but optimizer state persists. Primary issue(s): ALG-01, MATH-01, MATH-02. Evidence status: Verified code/description mismatch, Verified qualification needed, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, MATH-01, MATH-02.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Are per-client optimizer slots preserved/reset exactly as the reference GeFL-F implementation?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about missing classes, local steps and client data, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, MATH-01, MATH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R034 — D — Optimization and algorithmic behavior — Optimizer-state specialist
**Specialty focus:** Moment estimates, resets and stabilization.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the optimizer-state specialist perspective, the key question is moment estimates, resets and stabilization. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Optimizer moments are local client state while parameters are overwritten each round. Primary issue(s): MATH-01, MATH-02, HWA-01. Evidence status: Verified qualification needed, Verified qualification needed, Needs modest scope correction. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, HWA-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you test state resets and per-coordinate epsilon sensitivity as ablations?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about moment estimates, resets and stabilization, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, HWA-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R035 — D — Optimization and algorithmic behavior — Aggregation-weighting specialist
**Specialty focus:** HWA derivation and alternatives.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the aggregation-weighting specialist perspective, the key question is hwa derivation and alternatives. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** HWA removes nonholder-only row shrink and seems strongest as K grows. Primary issue(s): MATH-02, HWA-01, TH-02. Evidence status: Verified qualification needed, Needs modest scope correction, Partially supported; K15 pending. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-02, HWA-01, TH-02.
**D. Ratings:** Quality 1/4; Clarity 1/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. What is its matched effect versus uniform holder averaging and raw-count weighting?
2. Can the authors provide a traceable result or proof resolving mc-s/mc-b corrects exact class covariance by the minimum-displacement gelbrich map.? Minimum evidence requested: Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
3. What preregistered test on the actual implementation would falsify the concern about hwa derivation and alternatives, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about covariance correction / optimal transport, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-02, HWA-01, TH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R036 — D — Optimization and algorithmic behavior — Stability specialist
**Specialty focus:** Numerical stability, scale and degeneracy.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the stability specialist perspective, the key question is numerical stability, scale and degeneracy. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** The implemented MC maps use clipping, eigenspectrum floors and ReLU. Primary issue(s): HWA-01, TH-02, EXP-01. Evidence status: Needs modest scope correction, Partially supported; K15 pending, Pending; critical. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** HWA-01, TH-02, EXP-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What is the exact target-moment error under numerically difficult classes?
2. Can the authors provide a traceable result or proof resolving effective-number weighting is the statistically optimal holder aggregation weighting in this setting.? Minimum evidence requested: Present as motivated heuristic; compare uniform-holder and raw-count HWA under current full protocol; measure row estimation error by n_kc before claiming optimality.
3. What preregistered test on the actual implementation would falsify the concern about numerical stability, scale and degeneracy, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about weighting rule rationale, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) HWA-01, TH-02, EXP-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R037 — D — Optimization and algorithmic behavior — Adaptive-gating specialist
**Specialty focus:** F gate thresholds, robustness and data dependence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the adaptive-gating specialist perspective, the key question is f gate thresholds, robustness and data dependence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** F is computed but does not dynamically choose the generator in current run path. Primary issue(s): TH-02, EXP-01, ALG-01. Evidence status: Partially supported; K15 pending, Pending; critical, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-02, EXP-01, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can an unseen dataset be run with a single config where F selects the actual model?
2. Can the authors provide a traceable result or proof resolving any nonzero coupled adam weight-decay coefficient produces an approximately λ-independent full learning-rate-sized row shrink over all relevant non-holder updates.? Minimum evidence requested: State the per-coordinate assumptions and optimizer-state conditions; report T01 as one-seed mechanism evidence. Run K15's matched AdamW/no-decay/lazy-decay comparisons before claiming HWA is the best fix.
3. What preregistered test on the actual implementation would falsify the concern about f gate thresholds, robustness and data dependence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / adam, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-02, EXP-01, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R038 — D — Optimization and algorithmic behavior — Algorithmic equivalence specialist
**Specialty focus:** Known methods/special-case equivalence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the algorithmic equivalence specialist perspective, the key question is known methods/special-case equivalence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Some pieces reduce to known classwise aggregation or prior calibration. Primary issue(s): EXP-01, ALG-01, MATH-01. Evidence status: Pending; critical, Verified code/description mismatch, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, ALG-01, MATH-01.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which implementation path is uniquely ESA rather than config-composed baselines?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about known methods/special-case equivalence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, ALG-01, MATH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R039 — D — Optimization and algorithmic behavior — Complexity reviewer
**Specialty focus:** Operation counts, parameters and overhead.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the complexity reviewer perspective, the key question is operation counts, parameters and overhead. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Dense covariance adds substantial server memory and communication. Primary issue(s): ALG-01, MATH-01, MATH-02. Evidence status: Verified code/description mismatch, Verified qualification needed, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, MATH-01, MATH-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What are measured bytes and wall-clock for D=2560, k=256?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about operation counts, parameters and overhead, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, MATH-01, MATH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R040 — D — Optimization and algorithmic behavior — Mechanism-isolation reviewer
**Specialty focus:** Causal mechanism vs simultaneous changes.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the mechanism-isolation reviewer perspective, the key question is causal mechanism vs simultaneous changes. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The code and per-round logs make important update dynamics inspectable, and the wd=0 versus wd>0 contrast is useful.
**C. Weaknesses:** Ablations have correlated seed reuse and some effects overlap in test-time calibration. Primary issue(s): MATH-01, MATH-02, HWA-01. Evidence status: Verified qualification needed, Verified qualification needed, Needs modest scope correction. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, HWA-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Do mechanisms survive on one consistent deployed metric after LA/BBC is applied?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about causal mechanism vs simultaneous changes, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, HWA-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R041 — E — Federated system design and heterogeneity — Non-IID FL specialist
**Specialty focus:** Realism/completeness of client heterogeneity.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the non-iid fl specialist perspective, the key question is realism/completeness of client heterogeneity. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** Tests label skew through fixed Dirichlet partitions; not general non-IID distributions. Primary issue(s): SYS-01, PRIV-04, EXP-05. Evidence status: Needs methodology clarity, Needs clarity, not an algorithm fatality, Scope should be narrowed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** SYS-01, PRIV-04, EXP-05.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 3/6; Confidence 3/5.
**E. Author questions:**
1. How does the method handle client concept shift or class-conditional feature shift?
2. Can the authors provide a traceable result or proof resolving all evaluation choices use only federated training statistics and do not inspect test data.? Minimum evidence requested: Document that F alone is the gate statistic; keep `ncm_acc` in offline diagnosis only, do not mention it as server input or use it for thresholds.
3. What preregistered test on the actual implementation would falsify the concern about realism/completeness of client heterogeneity, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about client partition / data use, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) SYS-01, PRIV-04, EXP-05 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R042 — E — Federated system design and heterogeneity — Model heterogeneity specialist
**Specialty focus:** Which architectural differences tested.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the model heterogeneity specialist perspective, the key question is which architectural differences tested. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** K=10 maps one client per architecture, higher K repeats architectural groups. Primary issue(s): PRIV-04, EXP-05, ALG-01. Evidence status: Needs clarity, not an algorithm fatality, Scope should be narrowed, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-04, EXP-05, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What heterogeneity exactly is supported: architectures, data size, or only repeated fixed heads?
2. Can the authors provide a traceable result or proof resolving the evaluated implementation operationally uses secure aggregation so that the server sees only sums.? Minimum evidence requested: Say “simulated centrally; protocol sends secure-aggregated sums in deployment.” Explicitly state server trust, thresholds, collusion assumptions, client dropout and per-message protected fields.
3. What preregistered test on the actual implementation would falsify the concern about which architectural differences tested, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about secure aggregation / implementation fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-04, EXP-05, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R043 — E — Federated system design and heterogeneity — Client participation specialist
**Specialty focus:** Dropout, partial/asynchronous participation.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the client participation specialist perspective, the key question is dropout, partial/asynchronous participation. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** All clients participate synchronously; no dropouts/asynchrony are tested. Primary issue(s): EXP-05, ALG-01, EXP-02. Evidence status: Scope should be narrowed, Verified code/description mismatch, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-05, ALG-01, EXP-02.
**D. Ratings:** Quality 3/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Do aggregate stats survive client churn and secure aggregation thresholds?
2. Can the authors provide a traceable result or proof resolving esa is broadly robust across heterogeneous federated learning and long-tail settings.? Minimum evidence requested: Use exact scope wording, add one held-out dataset/protocol only if making broad claims, and do not extrapolate from K=100 MNIST to every large federation.
3. What preregistered test on the actual implementation would falsify the concern about dropout, partial/asynchronous participation, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about scope/generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-05, ALG-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R044 — E — Federated system design and heterogeneity — Label-skew specialist
**Specialty focus:** Rare classes and client holder counts.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the label-skew specialist perspective, the key question is rare classes and client holder counts. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** Classes are held by many of 10 clients in the most cited collapse result. Primary issue(s): ALG-01, EXP-02, SYS-01. Evidence status: Verified code/description mismatch, Critical missing comparative baseline, Needs methodology clarity. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, EXP-02, SYS-01.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. What if rare classes occur on one or two clients only?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about rare classes and client holder counts, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, EXP-02, SYS-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R045 — E — Federated system design and heterogeneity — Distribution-shift specialist
**Specialty focus:** Temporal drift and changed populations.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the distribution-shift specialist perspective, the key question is temporal drift and changed populations. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** The frozen feature extractor prevents evaluation of representation drift. Primary issue(s): EXP-02, SYS-01, PRIV-04. Evidence status: Critical missing comparative baseline, Needs methodology clarity, Needs clarity, not an algorithm fatality. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, SYS-01, PRIV-04.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. Does exact anchoring still help if the FE changes during FL or client domains differ?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about temporal drift and changed populations, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, SYS-01, PRIV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R046 — E — Federated system design and heterogeneity — Feature-space assumptions specialist
**Specialty focus:** Frozen FE and representational limit.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the feature-space assumptions specialist perspective, the key question is frozen fe and representational limit. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** If features have lost class identity upstream, generator-level anchoring cannot restore it. Primary issue(s): SYS-01, PRIV-04, EXP-05. Evidence status: Needs methodology clarity, Needs clarity, not an algorithm fatality, Scope should be narrowed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** SYS-01, PRIV-04, EXP-05.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What upper-bound is implied by the frozen FE oracle and how does it constrain claims?
2. Can the authors provide a traceable result or proof resolving all evaluation choices use only federated training statistics and do not inspect test data.? Minimum evidence requested: Document that F alone is the gate statistic; keep `ncm_acc` in offline diagnosis only, do not mention it as server input or use it for thresholds.
3. What preregistered test on the actual implementation would falsify the concern about frozen fe and representational limit, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about client partition / data use, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) SYS-01, PRIV-04, EXP-05 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R047 — E — Federated system design and heterogeneity — Federated protocol auditor
**Specialty focus:** Client-server message lifecycle.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the federated protocol auditor perspective, the key question is client-server message lifecycle. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** The algorithm uploads statistics once, then continues generator/head rounds. Primary issue(s): PRIV-04, EXP-05, ALG-01. Evidence status: Needs clarity, not an algorithm fatality, Scope should be narrowed, Verified code/description mismatch. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-04, EXP-05, ALG-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. List every client/server message and show which raw local values the server can see in the implementation versus protocol.
2. Can the authors provide a traceable result or proof resolving the evaluated implementation operationally uses secure aggregation so that the server sees only sums.? Minimum evidence requested: Say “simulated centrally; protocol sends secure-aggregated sums in deployment.” Explicitly state server trust, thresholds, collusion assumptions, client dropout and per-message protected fields.
3. What preregistered test on the actual implementation would falsify the concern about client-server message lifecycle, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about secure aggregation / implementation fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-04, EXP-05, ALG-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R048 — E — Federated system design and heterogeneity — Cross-client transfer specialist
**Specialty focus:** Information transferred vs shared representation limit.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the cross-client transfer specialist perspective, the key question is information transferred vs shared representation limit. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** Exact sums transfer global moments but cannot transfer missing higher-order modes by themselves. Primary issue(s): EXP-05, ALG-01, EXP-02. Evidence status: Scope should be narrowed, Verified code/description mismatch, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-05, ALG-01, EXP-02.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What evidence shows KH repairs multimodality, beyond lowering finite RFF discrepancy?
2. Can the authors provide a traceable result or proof resolving esa is broadly robust across heterogeneous federated learning and long-tail settings.? Minimum evidence requested: Use exact scope wording, add one held-out dataset/protocol only if making broad claims, and do not extrapolate from K=100 MNIST to every large federation.
3. What preregistered test on the actual implementation would falsify the concern about information transferred vs shared representation limit, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about scope/generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-05, ALG-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R049 — E — Federated system design and heterogeneity — Realistic deployment reviewer
**Specialty focus:** Credibility outside benchmark simulation.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the realistic deployment reviewer perspective, the key question is credibility outside benchmark simulation. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** Secure aggregation is specified but not cryptographically run in the simulator. Primary issue(s): ALG-01, EXP-02, SYS-01. Evidence status: Verified code/description mismatch, Critical missing comparative baseline, Needs methodology clarity. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** ALG-01, EXP-02, SYS-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 5/5.
**E. Author questions:**
1. What threat model and dropout threshold would a deployable implementation need?
2. Can the authors provide a traceable result or proof resolving the algorithm automatically chooses mean anchoring vs class-row/hwa fallback from statistic f at runtime.? Minimum evidence requested: Implement a genuine runtime switch driven only by released training statistics, test it on unseen/intermediate-F data, or describe the reported numbers as a manually configured per-dataset variant rather than an automatic method.
3. What preregistered test on the actual implementation would falsify the concern about credibility outside benchmark simulation, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about algorithm-to-code fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) ALG-01, EXP-02, SYS-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R050 — E — Federated system design and heterogeneity — Federated baseline specialist
**Specialty focus:** Strong competitive protocol-matched baselines.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the federated baseline specialist perspective, the key question is strong competitive protocol-matched baselines. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The client/server lifecycle and feature-space boundary are explicit; several assumptions are acknowledged rather than concealed.
**C. Weaknesses:** CReFF/CCVR/Fed3R-style readout is the main baseline gap. Primary issue(s): EXP-02, SYS-01, PRIV-04. Evidence status: Critical missing comparative baseline, Needs methodology clarity, Needs clarity, not an algorithm fatality. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, SYS-01, PRIV-04.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 1/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Will the final benchmark include protocol-matched recalibration, not just its centralized oracle?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about strong competitive protocol-matched baselines, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, SYS-01, PRIV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R051 — F — Generative models, distributions and statistics — Conditional generation specialist
**Specialty focus:** Generator conditioning/training consistency.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the conditional generation specialist perspective, the key question is generator conditioning/training consistency. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Generator is anchored by class mean and then shared residual decoder; identity fails if means overlap. Primary issue(s): MATH-01, MATH-02, MATH-03. Evidence status: Verified qualification needed, Verified qualification needed, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, MATH-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. What is the failure boundary for an intermediate-F class distribution?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about generator conditioning/training consistency, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R052 — F — Generative models, distributions and statistics — Feature-fidelity specialist
**Specialty focus:** Distributional properties for downstream heads.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the feature-fidelity specialist perspective, the key question is distributional properties for downstream heads. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Referee fidelity improves substantially on MNIST/FMNIST, but not uniformly on SVHN/CIFAR. Primary issue(s): MATH-02, MATH-03, THEORY-04. Evidence status: Verified qualification needed, Needs scoped claim, Needs claim scope. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-02, MATH-03, THEORY-04.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Do tail fidelity and synthetic-to-real discrepancy predict all per-dataset accuracy changes?
2. Can the authors provide a traceable result or proof resolving mc-s/mc-b corrects exact class covariance by the minimum-displacement gelbrich map.? Minimum evidence requested: Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
3. What preregistered test on the actual implementation would falsify the concern about distributional properties for downstream heads, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about covariance correction / optimal transport, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-02, MATH-03, THEORY-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R053 — F — Generative models, distributions and statistics — Moment-matching specialist
**Specialty focus:** Sufficiency of mean/second moment.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the moment-matching specialist perspective, the key question is sufficiency of mean/second moment. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Mean and scalar spread do not determine multimodal class distribution. Primary issue(s): MATH-03, THEORY-04, EXP-08. Evidence status: Needs scoped claim, Needs claim scope, Promising; pending. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-03, THEORY-04, EXP-08.
**D. Ratings:** Quality 3/4; Clarity 1/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. What empirical test shows moment matching improves modes instead of only spread?
2. Can the authors provide a traceable result or proof resolving herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.? Minimum evidence requested: Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
3. What preregistered test on the actual implementation would falsify the concern about sufficiency of mean/second moment, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about kernel herding / generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-03, THEORY-04, EXP-08 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R054 — F — Generative models, distributions and statistics — Multimodal distribution specialist
**Specialty focus:** Same moments, wrong modes/boundaries.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the multimodal distribution specialist perspective, the key question is same moments, wrong modes/boundaries. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Matching first/second moments can still produce a wrong mode structure or support. Primary issue(s): THEORY-04, EXP-08, GATE-02. Evidence status: Needs claim scope, Promising; pending, Acknowledged but unresolved. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** THEORY-04, EXP-08, GATE-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can a multimodal counterexample be tested with the proposed correction?
2. Can the authors provide a traceable result or proof resolving a scalar estimator result explains why 10 synthetic epochs should improve neural classifier accuracy after anchoring.? Minimum evidence requested: Call it a toy bias-variance model; articulate mapping assumptions; present empirical interaction rather than claiming derivation of T_s=10 for neural training.
3. What preregistered test on the actual implementation would falsify the concern about same moments, wrong modes/boundaries, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about synthetic budget proposition, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) THEORY-04, EXP-08, GATE-02 are fixed, not by score averaging.
**Cross-review disposition:** Moderate/conditional concern.

### R055 — F — Generative models, distributions and statistics — Sample selection specialist
**Specialty focus:** Finite pool and herding utility.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the sample selection specialist perspective, the key question is finite pool and herding utility. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Finite candidate pool and 6,144 RFFs make herding approximate. Primary issue(s): EXP-08, GATE-02, MATH-01. Evidence status: Promising; pending, Acknowledged but unresolved, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-08, GATE-02, MATH-01.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. How robust are gains to pool multiplier and RFF dimension?
2. Can the authors provide a traceable result or proof resolving mc-b is a reliable across-dataset improvement over mc/mc-s.? Minimum evidence requested: Run five-seed paired comparison, fix final-method selection without test-set tuning, report tail/worst-class accuracy as well as balanced accuracy, and state conditional benefits.
3. What preregistered test on the actual implementation would falsify the concern about finite pool and herding utility, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects moderate concerns about new mc-b result, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) EXP-08, GATE-02, MATH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R056 — F — Generative models, distributions and statistics — Synthetic-data evaluation specialist
**Specialty focus:** Faithful/useful samples vs task gains.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the synthetic-data evaluation specialist perspective, the key question is faithful/useful samples vs task gains. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** TSTR referee fidelity is a useful diagnostic but not definitive distributional evidence. Primary issue(s): GATE-02, MATH-01, MATH-02. Evidence status: Acknowledged but unresolved, Verified qualification needed, Verified qualification needed. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** GATE-02, MATH-01, MATH-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you report downstream transfer with confidence intervals and a distributional test independent of the referee?
2. Can the authors provide a traceable result or proof resolving the gate covers intermediate mean separability and chooses the best source of class identity in unseen settings.? Minimum evidence requested: Add held-out intermediate regime and evaluate candidate branches on non-test validation data; treat covariance route as heuristic and report selection error/cost.
3. What preregistered test on the actual implementation would falsify the concern about faithful/useful samples vs task gains, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about robustness / threshold, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) GATE-02, MATH-01, MATH-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R057 — F — Generative models, distributions and statistics — Representation learning specialist
**Specialty focus:** Frozen representation and upstream errors.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the representation learning specialist perspective, the key question is frozen representation and upstream errors. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Frozen feature extractor places an upper bound on all downstream strategies. Primary issue(s): MATH-01, MATH-02, MATH-03. Evidence status: Verified qualification needed, Verified qualification needed, Needs scoped claim. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-01, MATH-02, MATH-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Would the result survive a different or trainable representation?
2. Can the authors provide a traceable result or proof resolving the implemented transform gives exactly the real class mean and spread and is minimum-displacement optimal.? Minimum evidence requested: State theorem for the unconstrained affine map, then characterize the projected map separately. Log pre/post-ReLU moment errors and failure cases for all classes/seeds.
3. What preregistered test on the actual implementation would falsify the concern about frozen representation and upstream errors, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about moment correction, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-01, MATH-02, MATH-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R058 — F — Generative models, distributions and statistics — Class-overlap specialist
**Specialty focus:** Overlap and separability failure conditions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the class-overlap specialist perspective, the key question is overlap and separability failure conditions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** SVHN F≈.01 suggests mean anchor carries little class info, but not that covariance is always sufficient. Primary issue(s): MATH-02, MATH-03, THEORY-04. Evidence status: Verified qualification needed, Needs scoped claim, Needs claim scope. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-02, MATH-03, THEORY-04.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 1/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can F itself distinguish which fallback—HWA, MC-S or MC-B—is best?
2. Can the authors provide a traceable result or proof resolving mc-s/mc-b corrects exact class covariance by the minimum-displacement gelbrich map.? Minimum evidence requested: Rename guarantee to regularized/subspace approximation; report covariance residuals after clipping/ReLU; prove any narrower guarantee for the composed map; clearly mark MC-B as provisional pending five-seed comparison.
3. What preregistered test on the actual implementation would falsify the concern about overlap and separability failure conditions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about covariance correction / optimal transport, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) MATH-02, MATH-03, THEORY-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R059 — F — Generative models, distributions and statistics — Small-sample statistician
**Specialty focus:** Rare-class covariance noise/singularity.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the small-sample statistician perspective, the key question is rare-class covariance noise/singularity. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** Covariance of tail classes is unstable; E32 shows MNIST long-tail variance across seeds. Primary issue(s): MATH-03, THEORY-04, EXP-08. Evidence status: Needs scoped claim, Needs claim scope, Promising; pending. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** MATH-03, THEORY-04, EXP-08.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. What shrinkage prior selection is fixed independently of the test outcomes?
2. Can the authors provide a traceable result or proof resolving herding to the exact kernel embedding lowers the classification-risk term for arbitrary downstream heads.? Minimum evidence requested: Make the bound explicitly conditional, do not claim the neural risk is guaranteed to improve without an RKHS-norm assumption, and measure approximation/convergence sensitivity to RFF dimension and pool factor.
3. What preregistered test on the actual implementation would falsify the concern about rare-class covariance noise/singularity, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about kernel herding / generalization, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) MATH-03, THEORY-04, EXP-08 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R060 — F — Generative models, distributions and statistics — Generative alternatives specialist
**Specialty focus:** Simpler/stronger generator comparisons.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the generative alternatives specialist perspective, the key question is simpler/stronger generator comparisons. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper probes real vs synthetic spread/fidelity and reports failed variants instead of cherry-picking only successful generators.
**C. Weaknesses:** KME-Gen and Gaussian alternatives were tested and rejected—a valuable negative finding. Primary issue(s): THEORY-04, EXP-08, GATE-02. Evidence status: Needs claim scope, Promising; pending, Acknowledged but unresolved. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** THEORY-04, EXP-08, GATE-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Are their budgets matched closely enough for a fair alternative generator conclusion?
2. Can the authors provide a traceable result or proof resolving a scalar estimator result explains why 10 synthetic epochs should improve neural classifier accuracy after anchoring.? Minimum evidence requested: Call it a toy bias-variance model; articulate mapping assumptions; present empirical interaction rather than claiming derivation of T_s=10 for neural training.
3. What preregistered test on the actual implementation would falsify the concern about simpler/stronger generator comparisons, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about synthetic budget proposition, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) THEORY-04, EXP-08, GATE-02 are fixed, not by score averaging.
**Cross-review disposition:** Moderate/conditional concern.

### R061 — G — Experimental methodology and benchmark quality — Experimental design specialist
**Specialty focus:** Hypotheses isolated by the experiment plan.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the experimental design specialist perspective, the key question is hypotheses isolated by the experiment plan. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** Ablation plan is broad, but latest important K14 and K15 runs remain pending. Primary issue(s): EXP-01, EXP-02, EXP-03. Evidence status: Pending; critical, Critical missing comparative baseline, Pending; decisive. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, EXP-02, EXP-03.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 4/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. Can the missing runs be completed and the table assets regenerated before submission?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about hypotheses isolated by the experiment plan, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, EXP-02, EXP-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R062 — G — Experimental methodology and benchmark quality — Baseline fairness specialist
**Specialty focus:** Data, budgets, tuning and model access.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the baseline fairness specialist perspective, the key question is data, budgets, tuning and model access. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** Raw JSONLs support many comparisons; missing K01 CIFAR records weaken one test. Primary issue(s): EXP-02, EXP-03, STAT-02. Evidence status: Critical missing comparative baseline, Pending; decisive, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, EXP-03, STAT-02.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 3/4; Originality 3/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can all reported p-values be reproduced directly from paired run IDs?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about data, budgets, tuning and model access, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, EXP-03, STAT-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R063 — G — Experimental methodology and benchmark quality — Long-tail benchmark specialist
**Specialty focus:** Imbalance/profile/client partition construction.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the long-tail benchmark specialist perspective, the key question is imbalance/profile/client partition construction. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** IF=100/Dirichlet(.5) probes a hard regime but only a narrow imbalance grid. Primary issue(s): EXP-03, STAT-02, STAT-03. Evidence status: Pending; decisive, Verified, Verified corrections required. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-03, STAT-02, STAT-03.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What does performance do across intermediate IF and α?
2. Can the authors provide a traceable result or proof resolving the current evidence is complete enough to substantiate all major claims.? Minimum evidence requested: Run K15 first, then K14. Publish the paired per-seed metrics and regenerated tables; mark all remaining arms PENDING in manuscript until run.
3. What preregistered test on the actual implementation would falsify the concern about imbalance/profile/client partition construction, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about incomplete experiments, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-03, STAT-02, STAT-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R064 — G — Experimental methodology and benchmark quality — Dataset diversity specialist
**Specialty focus:** Breadth and dataset dependence.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the dataset diversity specialist perspective, the key question is breadth and dataset dependence. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** MNIST, FashionMNIST, CIFAR10 and SVHN are all small image benchmarks. Primary issue(s): STAT-02, STAT-03, EXT-01. Evidence status: Verified, Verified corrections required, Scoped claim; improvement desirable. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-02, STAT-03, EXT-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Is there an independent additional dataset or task that tests the statistic-driven gate?
2. Can the authors provide a traceable result or proof resolving all headline metrics are equally fair because the same metrics are used for every method.? Minimum evidence requested: Keep protocol-reproduction table labeled as historical comparison; add validation-selected checkpoints or final-round results for all methods on the same seeds. Never tune on the test set for the paper's final main table.
3. What preregistered test on the actual implementation would falsify the concern about breadth and dataset dependence, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about evaluation leakage / model selection, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-02, STAT-03, EXT-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R065 — G — Experimental methodology and benchmark quality — Metric specialist
**Specialty focus:** Balanced accuracy, tail metrics and checkpoint choice.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the metric specialist perspective, the key question is balanced accuracy, tail metrics and checkpoint choice. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** Best_mean_acc is test-selected, while LT uses final_bal; these need separation. Primary issue(s): STAT-03, EXT-01, EXP-01, NOV-04. Evidence status: Verified corrections required, Scoped claim; improvement desirable, Pending; critical, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-03, EXT-01, EXP-01, NOV-04.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 4/5.
**E. Author questions:**
1. Can primary tables use validation-selected or final-round metrics consistently?
2. Can the authors provide a traceable result or proof resolving every quoted paper number matches its generated table and raw per-seed results.? Minimum evidence requested: Regenerate paper assets from merged run JSONL, remove or re-source cells with no run, fix E25 header/data grouping, write CSV only from merged JSONL, and retain commit hashes for every table.
3. What preregistered test on the actual implementation would falsify the concern about balanced accuracy, tail metrics and checkpoint choice, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about result provenance / numerical consistency, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) STAT-03, EXT-01, EXP-01, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R066 — G — Experimental methodology and benchmark quality — Oracle specialist
**Specialty focus:** Interpretation/fairness of privileged upper bounds.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the oracle specialist perspective, the key question is interpretation/fairness of privileged upper bounds. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** Pooled-real oracle quantifies potential, but is privileged and not a deployable baseline. Primary issue(s): EXT-01, EXP-01, EXP-02, NOV-04. Evidence status: Scoped claim; improvement desirable, Pending; critical, Critical missing comparative baseline, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXT-01, EXP-01, EXP-02, NOV-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 4/4; Originality 1/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What matched federated classifier-retraining baseline approaches the oracle?
2. Can the authors provide a traceable result or proof resolving esa exceeds the original paper's best of ten methods in the balanced iid setting.? Minimum evidence requested: Use validation checkpoint selection for new results, clearly mark historical replication metric, compare same version on same splits, and label MC-B separately with its seed uncertainty.
3. What preregistered test on the actual implementation would falsify the concern about interpretation/fairness of privileged upper bounds, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about balanced-setting comparisons, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 4/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXT-01, EXP-01, EXP-02, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R067 — G — Experimental methodology and benchmark quality — Hyperparameter selection specialist
**Specialty focus:** Tuning/model selection and search bias.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the hyperparameter selection specialist perspective, the key question is tuning/model selection and search bias. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** F threshold and beta/epoch budget choices follow exploratory testing over same datasets. Primary issue(s): EXP-01, EXP-02, EXP-03. Evidence status: Pending; critical, Critical missing comparative baseline, Pending; decisive. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, EXP-02, EXP-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Which values were prespecified and which selected after seeing benchmark outcomes?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about tuning/model selection and search bias, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, EXP-02, EXP-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R068 — G — Experimental methodology and benchmark quality — Ablation specialist
**Specialty focus:** Removal/replacement/interactions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the ablation specialist perspective, the key question is removal/replacement/interactions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** LA/BBC overlap and minor MC/KH gains complicate attribution. Primary issue(s): EXP-02, EXP-03, STAT-02. Evidence status: Critical missing comparative baseline, Pending; decisive, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, EXP-03, STAT-02.
**D. Ratings:** Quality 1/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Do five seeds and Holm-adjusted component tests preserve every component?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about removal/replacement/interactions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, EXP-03, STAT-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R069 — G — Experimental methodology and benchmark quality — Negative-results specialist
**Specialty focus:** Omitted/failed methods and interpretation.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the negative-results specialist perspective, the key question is omitted/failed methods and interpretation. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** Negative results are documented: VCT, KME, Bayesian CSL etc. This is a strength. Primary issue(s): EXP-03, STAT-02, STAT-03. Evidence status: Pending; decisive, Verified, Verified corrections required. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-03, STAT-02, STAT-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 3/6; Confidence 3/5.
**E. Author questions:**
1. Are discarded methods and failure reasons included in the main paper/supplement consistently?
2. Can the authors provide a traceable result or proof resolving the current evidence is complete enough to substantiate all major claims.? Minimum evidence requested: Run K15 first, then K14. Publish the paired per-seed metrics and regenerated tables; mark all remaining arms PENDING in manuscript until run.
3. What preregistered test on the actual implementation would falsify the concern about omitted/failed methods and interpretation, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about incomplete experiments, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 3/6 is therefore driven by whether the primary unresolved issue(s) EXP-03, STAT-02, STAT-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R070 — G — Experimental methodology and benchmark quality — Generalization-test specialist
**Specialty focus:** Unseen partitions, client counts and shifts.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the generalization-test specialist perspective, the key question is unseen partitions, client counts and shifts. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The latest branch has extensive per-seed JSONL files, a ledger, generated tables, and a mismatch checker.
**C. Weaknesses:** K=100 accuracy result is not a broad systems scale benchmark. Primary issue(s): STAT-02, STAT-03, EXT-01. Evidence status: Verified, Verified corrections required, Scoped claim; improvement desirable. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-02, STAT-03, EXT-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can the claims be narrowed to measured accuracy scaling and report actual compute costs?
2. Can the authors provide a traceable result or proof resolving all headline metrics are equally fair because the same metrics are used for every method.? Minimum evidence requested: Keep protocol-reproduction table labeled as historical comparison; add validation-selected checkpoints or final-round results for all methods on the same seeds. Never tune on the test set for the paper's final main table.
3. What preregistered test on the actual implementation would falsify the concern about unseen partitions, client counts and shifts, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about evaluation leakage / model selection, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-02, STAT-03, EXT-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R071 — H — Statistical reliability and reproducibility — Statistical inference specialist
**Specialty focus:** Recompute means, uncertainty and tests.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the statistical inference specialist perspective, the key question is recompute means, uncertainty and tests. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** Branch asset generator now exposes numerical inconsistencies rather than hiding them. Primary issue(s): STAT-01, STAT-02, STAT-03. Evidence status: Verified reporting risk, Verified, Verified corrections required. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-01, STAT-02, STAT-03.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can the final paper be rebuilt from one fixed commit with no CHECK.md differences?
2. Can the authors provide a traceable result or proof resolving thirty negative component-removal comparisons establish an aggregate sign-test p=2e−9.? Minimum evidence requested: Remove the global 30-pair sign-test as confirmatory evidence; report paired effects per component/dataset with uncertainty and Holm-adjusted p-values; run seeds 3–4.
3. What preregistered test on the actual implementation would falsify the concern about recompute means, uncertainty and tests, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about statistical inference, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-01, STAT-02, STAT-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R072 — H — Statistical reliability and reproducibility — Small-sample testing specialist
**Specialty focus:** Power and assumptions at n=3.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the small-sample testing specialist perspective, the key question is power and assumptions at n=3. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** With n=3, t-tests are unstable and t assumptions hard to assess. Primary issue(s): STAT-02, STAT-03, STAT-04. Evidence status: Verified, Verified corrections required, Partially resolved; remaining small effects. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-02, STAT-03, STAT-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you give paired differences and CIs, not only p-values?
2. Can the authors provide a traceable result or proof resolving all headline metrics are equally fair because the same metrics are used for every method.? Minimum evidence requested: Keep protocol-reproduction table labeled as historical comparison; add validation-selected checkpoints or final-round results for all methods on the same seeds. Never tune on the test set for the paper's final main table.
3. What preregistered test on the actual implementation would falsify the concern about power and assumptions at n=3, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about evaluation leakage / model selection, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-02, STAT-03, STAT-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R073 — H — Statistical reliability and reproducibility — Multiple-comparisons specialist
**Specialty focus:** Many components/metrics/configs.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the multiple-comparisons specialist perspective, the key question is many components/metrics/configs. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** The 30 negative ablation differences share seeds/full method and are not 30 independent trials. Primary issue(s): STAT-03, STAT-04, EXP-06. Evidence status: Verified corrections required, Partially resolved; remaining small effects, Partially resolved on latest branch; still open. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-03, STAT-04, EXP-06.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Will the global 30-pair sign-test claim be removed or correctly clustered?
2. Can the authors provide a traceable result or proof resolving every quoted paper number matches its generated table and raw per-seed results.? Minimum evidence requested: Regenerate paper assets from merged run JSONL, remove or re-source cells with no run, fix E25 header/data grouping, write CSV only from merged JSONL, and retain commit hashes for every table.
3. What preregistered test on the actual implementation would falsify the concern about many components/metrics/configs, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about result provenance / numerical consistency, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-03, STAT-04, EXP-06 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R074 — H — Statistical reliability and reproducibility — Selection-bias specialist
**Specialty focus:** Test set for model/hyperparameter selection.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the selection-bias specialist perspective, the key question is test set for model/hyperparameter selection. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** best_mean_acc selection on test set is a model-selection leak, even if inherited. Primary issue(s): STAT-04, EXP-06, REPRO-01. Evidence status: Partially resolved; remaining small effects, Partially resolved on latest branch; still open, Improved but not complete. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-04, EXP-06, REPRO-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 1/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Will the final evidence use validation checkpoint selection?
2. Can the authors provide a traceable result or proof resolving the reported small improvements are robust across randomness.? Minimum evidence requested: Run planned seeds and publish mean±SD plus paired differences and CIs; distinguish high-powered effects from descriptive direction-only evidence.
3. What preregistered test on the actual implementation would falsify the concern about test set for model/hyperparameter selection, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about seed uncertainty, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-04, EXP-06, REPRO-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R075 — H — Statistical reliability and reproducibility — Randomness/seed specialist
**Specialty focus:** Paired split/init controls.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the randomness/seed specialist perspective, the key question is paired split/init controls. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** Pairing is valid for most comparisons but K01 CIFAR is summary-only/unpaired. Primary issue(s): EXP-06, REPRO-01, STAT-01. Evidence status: Partially resolved on latest branch; still open, Improved but not complete, Verified reporting risk. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-06, REPRO-01, STAT-01.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which claims rely on Welch tests and which are paired by identical split/seed?
2. Can the authors provide a traceable result or proof resolving all reported experimental results are reproducible from checked-in raw per-seed results.? Minimum evidence requested: Recover K01 CIFAR run zip/logs if possible; regenerate CSV from merged JSONL; tag output files with source notebook/commit/job/seed; maintain branch-specific ledger.
3. What preregistered test on the actual implementation would falsify the concern about paired split/init controls, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about missing raw artifacts / branch audit, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-06, REPRO-01, STAT-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R076 — H — Statistical reliability and reproducibility — Reproducibility specialist
**Specialty focus:** Regeneration from repository artifacts.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the reproducibility specialist perspective, the key question is regeneration from repository artifacts. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** LEDGER, paper_assets/numbers.csv and CHECK.md are positive additions on Claude branch. Primary issue(s): REPRO-01, STAT-01, STAT-02. Evidence status: Improved but not complete, Verified reporting risk, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** REPRO-01, STAT-01, STAT-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can raw data, scripts and commit SHA be bundled with submitted supplement?
2. Can the authors provide a traceable result or proof resolving the full set of listed runs can be reliably recreated from the repository by rerunning scripts as is.? Minimum evidence requested: Add environment lockfile, one-command local smoke test, immutable run config per JSONL, exact source commit per artifact, and a clean generation check in CI.
3. What preregistered test on the actual implementation would falsify the concern about regeneration from repository artifacts, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about reproduction protocol, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) REPRO-01, STAT-01, STAT-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R077 — H — Statistical reliability and reproducibility — Numerical consistency specialist
**Specialty focus:** Tables, logs, plots and raw values.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the numerical consistency specialist perspective, the key question is tables, logs, plots and raw values. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** E25 summary appears mislabeled and CHECK shows explicit number mismatch. Primary issue(s): STAT-01, STAT-02, STAT-03. Evidence status: Verified reporting risk, Verified, Verified corrections required. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-01, STAT-02, STAT-03.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can summary markdown be regenerated from JSONL, eliminating manual setting labels?
2. Can the authors provide a traceable result or proof resolving thirty negative component-removal comparisons establish an aggregate sign-test p=2e−9.? Minimum evidence requested: Remove the global 30-pair sign-test as confirmatory evidence; report paired effects per component/dataset with uncertainty and Holm-adjusted p-values; run seeds 3–4.
3. What preregistered test on the actual implementation would falsify the concern about tables, logs, plots and raw values, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about statistical inference, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-01, STAT-02, STAT-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R078 — H — Statistical reliability and reproducibility — Training-budget specialist
**Specialty focus:** Rounds, epochs, sample counts and fairness.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the training-budget specialist perspective, the key question is rounds, epochs, sample counts and fairness. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** Some methods use 10 synthetic epochs; hard datasets peak higher then end lower. Primary issue(s): STAT-02, STAT-03, STAT-04. Evidence status: Verified, Verified corrections required, Partially resolved; remaining small effects. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-02, STAT-03, STAT-04.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 1/4; Originality 2/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Are the chosen evaluation rounds and number of training updates matched exactly?
2. Can the authors provide a traceable result or proof resolving all headline metrics are equally fair because the same metrics are used for every method.? Minimum evidence requested: Keep protocol-reproduction table labeled as historical comparison; add validation-selected checkpoints or final-round results for all methods on the same seeds. Never tune on the test set for the paper's final main table.
3. What preregistered test on the actual implementation would falsify the concern about rounds, epochs, sample counts and fairness, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about evaluation leakage / model selection, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 1/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-02, STAT-03, STAT-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R079 — H — Statistical reliability and reproducibility — Statistical reporting specialist
**Specialty focus:** Significance/uncertainty prose.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the statistical reporting specialist perspective, the key question is significance/uncertainty prose. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** Significance wording sometimes outstrips n=3 evidence; MC-B MNIST seed is unstable. Primary issue(s): STAT-03, STAT-04, EXP-06. Evidence status: Verified corrections required, Partially resolved; remaining small effects, Partially resolved on latest branch; still open. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-03, STAT-04, EXP-06.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can all reported effect sizes carry variability and multiplicity-adjusted interpretation?
2. Can the authors provide a traceable result or proof resolving every quoted paper number matches its generated table and raw per-seed results.? Minimum evidence requested: Regenerate paper assets from merged run JSONL, remove or re-source cells with no run, fix E25 header/data grouping, write CSV only from merged JSONL, and retain commit hashes for every table.
3. What preregistered test on the actual implementation would falsify the concern about significance/uncertainty prose, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about result provenance / numerical consistency, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-03, STAT-04, EXP-06 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R080 — H — Statistical reliability and reproducibility — Research-process auditor
**Specialty focus:** Config changes, incomplete jobs, traceability.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the research-process auditor perspective, the key question is config changes, incomplete jobs, traceability. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The branch's generated CHECK.md and K09 analysis expose statistical/reproducibility weaknesses explicitly.
**C. Weaknesses:** Version/branch ambiguity is a reproducibility flaw distinct from scientific results. Primary issue(s): STAT-04, EXP-06, REPRO-01. Evidence status: Partially resolved; remaining small effects, Partially resolved on latest branch; still open, Improved but not complete. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-04, EXP-06, REPRO-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 2/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can reviewers clone one commit and recreate every table via a documented command?
2. Can the authors provide a traceable result or proof resolving the reported small improvements are robust across randomness.? Minimum evidence requested: Run planned seeds and publish mean±SD plus paired differences and CIs; distinguish high-powered effects from descriptive direction-only evidence.
3. What preregistered test on the actual implementation would falsify the concern about config changes, incomplete jobs, traceability, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about seed uncertainty, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-04, EXP-06, REPRO-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R081 — I — Privacy, communication, complexity and systems — Differential privacy specialist
**Specialty focus:** Sensitivity, noise, accounting and privacy unit.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the differential privacy specialist perspective, the key question is sensitivity, noise, accounting and privacy unit. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** E26 is DP for additional statistic releases, not end-to-end protocol; the paper admits this. Primary issue(s): PRIV-01, PRIV-02, PRIV-03. Evidence status: Conditional assumption needs formal statement, Scope partially documented, Verified scope restriction. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-01, PRIV-02, PRIV-03.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What is the privacy unit/adjacency and how are the non-DP generator/head messages described?
2. Can the authors provide a traceable result or proof resolving the reported epsilon/delta-dp statistic release is independent of private-data calibration choices and applies as stated.? Minimum evidence requested: Precisely specify adjacency and protected records; provide a truly public/external calibration pool or fixed data-independent norm bound; document whether the held-out sample belongs to the DP domain.
3. What preregistered test on the actual implementation would falsify the concern about sensitivity, noise, accounting and privacy unit, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about differential privacy, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-01, PRIV-02, PRIV-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R082 — I — Privacy, communication, complexity and systems — Secure aggregation specialist
**Specialty focus:** Server observations and protocol assumptions.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the secure aggregation specialist perspective, the key question is server observations and protocol assumptions. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** The stats aggregator is centrally simulated; secure aggregation is specified, not cryptographically executed. Primary issue(s): PRIV-02, PRIV-03, PRIV-04. Evidence status: Scope partially documented, Verified scope restriction, Needs clarity, not an algorithm fatality. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-02, PRIV-03, PRIV-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 1/4; Overall 1/6; Confidence 4/5.
**E. Author questions:**
1. What threat model and minimum cohort assumptions are required?
2. Can the authors provide a traceable result or proof resolving esa as a full fl protocol is differentially private.? Minimum evidence requested: Title/table/callout should consistently say “DP of the added anchoring-statistic releases”; map every release and budget. Do not claim end-to-end DP. Run/correct MC-S/MC-B privacy scope only if claimed.
3. What preregistered test on the actual implementation would falsify the concern about server observations and protocol assumptions, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about differential privacy scope, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) PRIV-02, PRIV-03, PRIV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R083 — I — Privacy, communication, complexity and systems — Information-leakage specialist
**Specialty focus:** Counts, means, covariances and repeated releases.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the information-leakage specialist perspective, the key question is counts, means, covariances and repeated releases. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Global moments/counts reveal class support and distributions even if clients are hidden. Primary issue(s): PRIV-03, PRIV-04, PRIV-05. Evidence status: Verified scope restriction, Needs clarity, not an algorithm fatality, Needs precise language. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-03, PRIV-04, PRIV-05.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 1/6; Confidence 4/5.
**E. Author questions:**
1. How does DP prevent rare-class attribute inference under a small class holder count?
2. Can the authors provide a traceable result or proof resolving the dp evaluation validates all esa variants, including covariance correction.? Minimum evidence requested: State exact DP-enabled configuration in table and add tests only if DP MC-S/MC-B is claimed. Ensure every released covariance, PCA basis, and gate input is included in sensitivity/compose accounting.
3. What preregistered test on the actual implementation would falsify the concern about counts, means, covariances and repeated releases, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects major concerns about differential privacy / covariance path, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) PRIV-03, PRIV-04, PRIV-05 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R084 — I — Privacy, communication, complexity and systems — Privacy-utility specialist
**Specialty focus:** Utility costs at epsilon levels.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the privacy-utility specialist perspective, the key question is utility costs at epsilon levels. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Privacy-utility costs are 4.5–6.9 points at ε=8 and 9.8–14.1 at ε=2 on MNIST/FashionMNIST. Primary issue(s): PRIV-04, PRIV-05, COMM-01. Evidence status: Needs clarity, not an algorithm fatality, Needs precise language, Requires cost table correction. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-04, PRIV-05, COMM-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Does the same trade-off hold for SVHN covariance/MC-B where the main gate is most complex?
2. Can the authors provide a traceable result or proof resolving the evaluated implementation operationally uses secure aggregation so that the server sees only sums.? Minimum evidence requested: Say “simulated centrally; protocol sends secure-aggregated sums in deployment.” Explicitly state server trust, thresholds, collusion assumptions, client dropout and per-message protected fields.
3. What preregistered test on the actual implementation would falsify the concern about utility costs at epsilon levels, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about secure aggregation / implementation fidelity, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-04, PRIV-05, COMM-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R085 — I — Privacy, communication, complexity and systems — Communication-cost specialist
**Specialty focus:** Bytes/dimensions/rounds for every release.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the communication-cost specialist perspective, the key question is bytes/dimensions/rounds for every release. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Covariance variant’s dense pooled second moment can dominate 0.3MB mean/RFF messages. Primary issue(s): PRIV-05, COMM-01, SYS-03. Evidence status: Needs precise language, Requires cost table correction, Needs scoped claims. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-05, COMM-01, SYS-03.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 1/6; Confidence 5/5.
**E. Author questions:**
1. Can the paper provide bytes per class/client/dataset for MC-S and baseline generator?
2. Can the authors provide a traceable result or proof resolving secure aggregate totals reveal nothing except safe federation-wide facts.? Minimum evidence requested: Narrow language to “server sees only the aggregate under secure aggregation”; quantify membership/attribute risks for rare classes and make DP the guarantee for these releases.
3. What preregistered test on the actual implementation would falsify the concern about bytes/dimensions/rounds for every release, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects moderate concerns about information leakage, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 1/6 is therefore driven by whether the primary unresolved issue(s) PRIV-05, COMM-01, SYS-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R086 — I — Privacy, communication, complexity and systems — Computational-complexity specialist
**Specialty focus:** Server compute, memory and local scaling.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the computational-complexity specialist perspective, the key question is server compute, memory and local scaling. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Server eigensolver and candidate sampling are nontrivial; there is no full runtime/memory table. Primary issue(s): COMM-01, SYS-03, PRIV-01. Evidence status: Requires cost table correction, Needs scoped claims, Conditional assumption needs formal statement. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** COMM-01, SYS-03, PRIV-01.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What are latency/peak memory figures at CIFAR’s D=2560?
2. Can the authors provide a traceable result or proof resolving the added communication remains about 0.3 mb apart from the small class-level terms for the main method/covariance option.? Minimum evidence requested: Give per-variant bytes per client/server, dtypes, number of rounds and whether covariance sums are split/streamed; report measured peak memory and latency for SVHN/CIFAR dimensions.
3. What preregistered test on the actual implementation would falsify the concern about server compute, memory and local scaling, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about communication/compute cost, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) COMM-01, SYS-03, PRIV-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R087 — I — Privacy, communication, complexity and systems — Large-federation specialist
**Specialty focus:** Sparse holders as K grows.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the large-federation specialist perspective, the key question is sparse holders as k grows. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Class holder counts shrink as K grows; HWA dilution data shows advantage growth. Primary issue(s): SYS-03, PRIV-01, PRIV-02. Evidence status: Needs scoped claims, Conditional assumption needs formal statement, Scope partially documented. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** SYS-03, PRIV-01, PRIV-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Are there client-count results on the hard datasets, or is K=100 robustness only MNIST/FMNIST?
2. Can the authors provide a traceable result or proof resolving method is scalable to many clients with acceptable communication/latency based on k=100 accuracy results.? Minimum evidence requested: Separate accuracy scaling from resource scaling; report measured time/memory/bytes for D=768 and D=2560 covariance cases, and limit deployment claims.
3. What preregistered test on the actual implementation would falsify the concern about sparse holders as k grows, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects moderate concerns about large-scale systems, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) SYS-03, PRIV-01, PRIV-02 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R088 — I — Privacy, communication, complexity and systems — Resource-efficiency specialist
**Specialty focus:** Accuracy vs simple baselines and compute.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the resource-efficiency specialist perspective, the key question is accuracy vs simple baselines and compute. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** No matched compute/accuracy frontier compares method to simple HWA+LA or ER. Primary issue(s): PRIV-01, PRIV-02, PRIV-03. Evidence status: Conditional assumption needs formal statement, Scope partially documented, Verified scope restriction. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-01, PRIV-02, PRIV-03.
**D. Ratings:** Quality 1/4; Clarity 2/4; Significance 1/4; Originality 3/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can you compare wall clock, transfer cost, and accuracy against minimal baseline?
2. Can the authors provide a traceable result or proof resolving the reported epsilon/delta-dp statistic release is independent of private-data calibration choices and applies as stated.? Minimum evidence requested: Precisely specify adjacency and protected records; provide a truly public/external calibration pool or fixed data-independent norm bound; document whether the held-out sample belongs to the DP domain.
3. What preregistered test on the actual implementation would falsify the concern about accuracy vs simple baselines and compute, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about differential privacy, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 1/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-01, PRIV-02, PRIV-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R089 — I — Privacy, communication, complexity and systems — Engineering/implementation specialist
**Specialty focus:** Synchronization/aggregation implementation.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the engineering/implementation specialist perspective, the key question is synchronization/aggregation implementation. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Core simulator concatenates clients’ features, unlike secure aggregation deployment. Primary issue(s): PRIV-02, PRIV-03, PRIV-04. Evidence status: Scope partially documented, Verified scope restriction, Needs clarity, not an algorithm fatality. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-02, PRIV-03, PRIV-04.
**D. Ratings:** Quality 2/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which code paths model server messages and what needs replacement in deployment?
2. Can the authors provide a traceable result or proof resolving esa as a full fl protocol is differentially private.? Minimum evidence requested: Title/table/callout should consistently say “DP of the added anchoring-statistic releases”; map every release and budget. Do not claim end-to-end DP. Run/correct MC-S/MC-B privacy scope only if claimed.
3. What preregistered test on the actual implementation would falsify the concern about synchronization/aggregation implementation, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about differential privacy scope, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-02, PRIV-03, PRIV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R090 — I — Privacy, communication, complexity and systems — Practical deployment specialist
**Specialty focus:** Trust assumptions and coordination needs.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the practical deployment specialist perspective, the key question is trust assumptions and coordination needs. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The paper distinguishes DP on extra statistics from non-DP generator FL training; this is an important scope clarification.
**C. Weaknesses:** Communication assumptions and fallback logic are part of the algorithm; code currently preselects variants. Primary issue(s): PRIV-03, PRIV-04, PRIV-05. Evidence status: Verified scope restriction, Needs clarity, not an algorithm fatality, Needs precise language. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** PRIV-03, PRIV-04, PRIV-05.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can one config select branch and calculate protocol costs end-to-end?
2. Can the authors provide a traceable result or proof resolving the dp evaluation validates all esa variants, including covariance correction.? Minimum evidence requested: State exact DP-enabled configuration in table and add tests only if DP MC-S/MC-B is claimed. Ensure every released covariance, PCA basis, and gate input is included in sensitivity/compose accounting.
3. What preregistered test on the actual implementation would falsify the concern about trust assumptions and coordination needs, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about differential privacy / covariance path, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) PRIV-03, PRIV-04, PRIV-05 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R091 — J — Meta-review, ethics, communication and final decision — Area-chair meta-reviewer
**Specialty focus:** Whether whole contribution meets contribution standard.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the area-chair meta-reviewer perspective, the key question is whether whole contribution meets contribution standard. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** Main candidate is promising but the incremental contribution is weaker against readout calibration. Primary issue(s): BR-01, TH-01, EXP-01. Evidence status: Verified, Reframed; still needs careful claims, Pending; critical. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** BR-01, TH-01, EXP-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What fair, matched baseline most changes your verdict?
2. Can the authors provide a traceable result or proof resolving current paper and supporting evidence are in a single unambiguous, reviewable snapshot.? Minimum evidence requested: Before submission, nominate one canonical commit and merge or tag it; archive older drafts; regenerate all tables and ledgers at that commit.
3. What preregistered test on the actual implementation would falsify the concern about whether whole contribution meets contribution standard, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about branch/version control, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) BR-01, TH-01, EXP-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R092 — J — Meta-review, ethics, communication and final decision — Senior area-chair reviewer
**Specialty focus:** Strongest accept/reject arguments and weights.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the senior area-chair reviewer perspective, the key question is strongest accept/reject arguments and weights. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** A correct conditional bound plus unresolved empirical optimizer scope is better than calling theorem false. Primary issue(s): TH-01, EXP-01, EXP-02. Evidence status: Reframed; still needs careful claims, Pending; critical, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, EXP-01, EXP-02.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can the manuscript consistently state the corrected theorem across abstract, proof, figures and main branch?
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about strongest accept/reject arguments and weights, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-01, EXP-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R093 — J — Meta-review, ethics, communication and final decision — Novelty-significance arbiter
**Specialty focus:** Importance of residual original contribution.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the novelty-significance arbiter perspective, the key question is importance of residual original contribution. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** Integration novelty is plausible, but close methods limit broad claims. Primary issue(s): EXP-01, EXP-02, STAT-03, NOV-04. Evidence status: Pending; critical, Critical missing comparative baseline, Verified corrections required, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, EXP-02, STAT-03, NOV-04.
**D. Ratings:** Quality 3/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. What does ESA uniquely offer after HWA+LA+CSL and readout baselines?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about importance of residual original contribution, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 3/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, EXP-02, STAT-03, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R094 — J — Meta-review, ethics, communication and final decision — Theory-vs-experiment arbiter
**Specialty focus:** Whether theorem issue undermines empirical results.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the theory-vs-experiment arbiter perspective, the key question is whether theorem issue undermines empirical results. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** Empirical claims can survive even if the strongest optimizer narrative is narrowed. Primary issue(s): EXP-02, STAT-03, FMT-01, NOV-04. Evidence status: Critical missing comparative baseline, Verified corrections required, Not ready as a submission artifact, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, STAT-03, FMT-01, NOV-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Would the measured gains remain after removing the causal theorem and adding matched optimizer fixes?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about whether theorem issue undermines empirical results, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, STAT-03, FMT-01, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R095 — J — Meta-review, ethics, communication and final decision — Strongest-rebuttal reviewer
**Specialty focus:** Best author response and surviving objections.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the strongest-rebuttal reviewer perspective, the key question is best author response and surviving objections. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** Best author response should concede low-confidence claims, not argue from the 100 simulated score votes. Primary issue(s): STAT-03, FMT-01, BR-01. Evidence status: Verified corrections required, Not ready as a submission artifact, Verified. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** STAT-03, FMT-01, BR-01.
**D. Ratings:** Quality 1/4; Clarity 3/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Which exact issue and artifact would change the reject recommendation?
2. Can the authors provide a traceable result or proof resolving every quoted paper number matches its generated table and raw per-seed results.? Minimum evidence requested: Regenerate paper assets from merged run JSONL, remove or re-source cells with no run, fix E25 header/data grouping, write CSV only from merged JSONL, and retain commit hashes for every table.
3. What preregistered test on the actual implementation would falsify the concern about best author response and surviving objections, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about result provenance / numerical consistency, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) STAT-03, FMT-01, BR-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R096 — J — Meta-review, ethics, communication and final decision — Reproducibility/checklist reviewer
**Specialty focus:** Transparency, docs, code and checklist.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the reproducibility/checklist reviewer perspective, the key question is transparency, docs, code and checklist. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** The latest branch adds substantial reproducibility artifacts but has unresolved mismatches. Primary issue(s): FMT-01, BR-01, TH-01. Evidence status: Not ready as a submission artifact, Verified, Reframed; still needs careful claims. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** FMT-01, BR-01, TH-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 3/4; Originality 3/4; Overall 2/6; Confidence 3/5.
**E. Author questions:**
1. Can you build a clean 9-page anonymous PDF and supplement from the tagged snapshot?
2. Can the authors provide a traceable result or proof resolving the repository contains a submission-ready, anonymized neurips manuscript.? Minimum evidence requested: Create an anonymized submission branch/tag with final LaTeX/PDF, appendix, checklist and separate anonymized supplement; ensure citations/links and repo metadata do not reveal authors if required by review policy.
3. What preregistered test on the actual implementation would falsify the concern about transparency, docs, code and checklist, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects major concerns about submission readiness, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 3/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 3/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) FMT-01, BR-01, TH-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R097 — J — Meta-review, ethics, communication and final decision — Ethical/privacy reviewer
**Specialty focus:** Data/privacy claims and risks.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the ethical/privacy reviewer perspective, the key question is data/privacy claims and risks. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** DP and secure aggregation need accurate boundaries rather than sweeping privacy language. Primary issue(s): BR-01, TH-01, EXP-01. Evidence status: Verified, Reframed; still needs careful claims, Pending; critical. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** BR-01, TH-01, EXP-01.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Will the abstract state statistics-only DP and release-level assumptions?
2. Can the authors provide a traceable result or proof resolving current paper and supporting evidence are in a single unambiguous, reviewable snapshot.? Minimum evidence requested: Before submission, nominate one canonical commit and merge or tag it; archive older drafts; regenerate all tables and ledgers at that commit.
3. What preregistered test on the actual implementation would falsify the concern about data/privacy claims and risks, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about branch/version control, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) BR-01, TH-01, EXP-01 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R098 — J — Meta-review, ethics, communication and final decision — Clarity/communication reviewer
**Specialty focus:** Notation, claims, figures and reproducibility.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the clarity/communication reviewer perspective, the key question is notation, claims, figures and reproducibility. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** Plain-language handout and generated figures help readability, but are not the final paper. Primary issue(s): TH-01, EXP-01, EXP-02. Evidence status: Reframed; still needs careful claims, Pending; critical, Critical missing comparative baseline. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** TH-01, EXP-01, EXP-02.
**D. Ratings:** Quality 1/4; Clarity 3/4; Significance 1/4; Originality 2/4; Overall 2/6; Confidence 4/5.
**E. Author questions:**
1. Can the core paper define the algorithm and metrics without relying on handout/HTML?
2. Can the authors provide a traceable result or proof resolving the corrected bound provides a novel optimizer-independent result explaining collapse.? Minimum evidence requested: Call it a conditional contraction bound; separate (i) tautological bound validity, (ii) measured G/D mechanism, and (iii) empirical collapse. Don't use zero bound violations as independent evidence for Adam behavior.
3. What preregistered test on the actual implementation would falsify the concern about notation, claims, figures and reproducibility, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 1/4 reflects major concerns about theory / collapse, while preserving credit for the reported long-tail gains and corrected research state. Clarity 3/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 1/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) TH-01, EXP-01, EXP-02 are fixed, not by score averaging.
**Cross-review disposition:** Retain with scope change.

### R099 — J — Meta-review, ethics, communication and final decision — Adversarial final reviewer
**Specialty focus:** Missed consequential objection, carefully verified.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the adversarial final reviewer perspective, the key question is missed consequential objection, carefully verified. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** The main missed objection is that the ‘automatic’ gate isn't runtime-selected in code. Primary issue(s): EXP-01, EXP-02, STAT-03. Evidence status: Pending; critical, Critical missing comparative baseline, Verified corrections required. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-01, EXP-02, STAT-03.
**D. Ratings:** Quality 2/4; Clarity 1/4; Significance 2/4; Originality 2/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can the actual method’s config/code and reported selected branch be reconciled?
2. Can the authors provide a traceable result or proof resolving the proposed hwa/esa treatment is superior to simpler optimizer fixes for conditioning-row collapse.? Minimum evidence requested: Run K15 at K=10/50/100 with same splits/seeds and matched generator/optimizer budgets. At minimum compare row norms, tail fidelity, balanced accuracy, and runtime. Until then, narrow claims to demonstrated alternatives.
3. What preregistered test on the actual implementation would falsify the concern about missed consequential objection, carefully verified, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / causal attribution, while preserving credit for the reported long-tail gains and corrected research state. Clarity 1/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 2/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-01, EXP-02, STAT-03 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

### R100 — J — Meta-review, ethics, communication and final decision — Independent final decision-maker
**Specialty focus:** Evidence-led recommendation, not score average.
**A. Paper summary:** The paper proposes Exact-Statistics Anchoring (ESA) for generator-based heterogeneous federated learning: use class-level sums/statistics to anchor class identity and synthetic distributions, while combining known classifier-side adjustments. From the independent final decision-maker perspective, the key question is evidence-led recommendation, not score average. The current candidate is the latest Claude branch; the default main branch is stale.
**B. Strengths:** The project has an identifiable useful idea and a more mature evidence record on the current working branch than on main.
**C. Weaknesses:** The recommendation is about current submission readiness, not inherent value of the research idea. Primary issue(s): EXP-02, STAT-03, FMT-01, NOV-04. Evidence status: Critical missing comparative baseline, Verified corrections required, Not ready as a submission artifact, New likely omission; must verify fit and compare. I would not infer misconduct from missing or mismatched artifacts; the branch ledger indicates some are pending or not copied back.
**Issue IDs:** EXP-02, STAT-03, FMT-01, NOV-04.
**D. Ratings:** Quality 2/4; Clarity 2/4; Significance 2/4; Originality 1/4; Overall 2/6; Confidence 5/5.
**E. Author questions:**
1. Can branch, method, results, recalibration baseline, and DP scope all be aligned in one commit?
2. Can the authors provide a traceable result or proof resolving the large ours-a gain over gefl-f reflects a large incremental contribution beyond existing classifier-side fixes.? Minimum evidence requested: Implement matched CReFF/CCVR/Fed3R-style baselines under the same frozen FE, head architecture, long-tail partitions and training budgets. If using ER-LDA, run full schedule/multi-seed and compare both methods with ER.
3. What preregistered test on the actual implementation would falsify the concern about evidence-led recommendation, not score average, and would the authors change the claim if that test failed?
**F. Overall reasoning:** Quality 2/4 reflects critical concerns about baseline fairness / readout recalibration, while preserving credit for the reported long-tail gains and corrected research state. Clarity 2/4 reflects the separate branch snapshots and remaining metric/code alignment work. Significance 2/4 reflects the practical long-tail problem and large gains against uncorrected GeFL-F. Originality 1/4 reflects known component families and a potentially useful new integration/diagnosis. Overall 2/6 is therefore driven by whether the primary unresolved issue(s) EXP-02, STAT-03, FMT-01, NOV-04 are fixed, not by score averaging.
**Cross-review disposition:** Major concern survives; correction/evidence required.

## 11. Simulated area-chair meta-review

**Recommendation: 2/6 — Reject as submitted. Confidence: 4/5.**

The submission addresses a real limitation in generator-based heterogeneous FL, and the Claude branch documents substantial long-tail gains, careful negative experiments, a useful conditional contraction bound, and a credible principle: class-level knowledge may be more stable when taken from secure-aggregated statistics than from a learned class-conditioning parameter averaged across heterogeneous clients. The newer branch is materially more reviewable than main because it includes a ledger, per-seed files, generated paper assets, a mismatch checker, revised theorem, and a detailed readout analysis.

The present manuscript is not ready for acceptance because the central algorithmic comparison is unfinished. First, the strongest simple competing explanation for row collapse—matched AdamW, disabling decay for conditioning rows, or lazy decay—is specified but not run; without K15, it is impossible to decide whether HWA is the right solution or whether standard optimizer correction removes most of the issue. Second, GeFL-F's long-tail error is dominated by a classifier/readout gap recoverable by known recalibration-style methods. The pooled-real oracle is privileged, not fair; it makes the missing matched federated CReFF/CCVR/Fed3R/ER baseline more important, not less. Third, the paper describes F as an automatic gate, while the run code's generator branch is configured beforehand and F only logged, so the single adaptive algorithm is not what is executed. Fourth, generated tables and source summaries are inconsistent and must be re-built at a canonical commit.

Theory should be reframed rather than discarded: the current conditional contraction inequality is correct, but it is a triangle-inequality bound conditioned on measured gain/loss, and T01 checks it on one seed. The “Adam makes decay coefficient irrelevant” statement has a restricted epsilon/optimizer-state regime; measured evidence is promising but not a universal proof. Similarly, the MC/MC-S ideal affine/Gelbrich results do not automatically cover the implementation's clipping, subspace approximation and ReLU. DP is a useful statistics-release guarantee, not full-protocol DP; the current paper largely admits this but must state it consistently.

The strongest reasons to accept are (i) substantial measured gains over uncorrected GeFL-F in long-tail MNIST/FashionMNIST and useful SVHN fallback results, (ii) a coherent statistics-anchoring principle, (iii) evidence of self-correction and negative results in the latest branch, and (iv) possible heterogenous exact-readout extension. Strongest reasons to reject are (i) missing optimizer-fix control, (ii) missing matched recalibration baseline, (iii) gate/code mismatch, and (iv) outstanding raw-to-table inconsistency / branch split.

A path to a materially stronger resubmission is: merge/tag a canonical branch; finish K15; finish K14 seeds 3–4 (including MC-B); run matched readout recalibration baselines across the full schedules/datasets; implement or correctly relabel F as a preconfigured branch choice; regenerate all assets from raw JSONL; scope theorem and optimality/DP claims to what is actually proved and evaluated; and report final raw/calibrated metrics separately with consistent checkpoint selection.

## 12. Prioritized action plan

### P0 — must resolve before submission
1. **One version of record:** merge/pin `claude/moved-session-c4mfhn` (or equivalent final state) into a submission branch. Do not base a submission on stale `main`; keep `revision/adam-collapse-drift` marked as duplicate unless it diverges.
2. **K15:** run matched AdamW (wd=1e−3), row-no-decay, lazy decay, and HWA on MNIST/FashionMNIST long-tail for K=10/50/100; report row norms, tail fidelity, final balanced accuracy and runtime. The research's causal story currently depends on this missing baseline.
3. **Recalibration baseline:** compare against matched CReFF/CCVR/Fed3R-style or ER-LDA stats-based readout, under same FE/head/splits and full schedule. Report both methods with the same inference/calibration procedure.
4. **Gate fidelity:** implement F-driven generator selection within the actual runtime and test it without test labels, or rewrite algorithm claims/results as separately configured variants.
5. **Reproducibility gate:** regenerate `docs/paper_assets/*` from raw results; resolve `CHECK.md`, remove unsourced K=50 numbers; regenerate E25/K09 summaries/CSVs from merged JSONL; recover K01 CIFAR raw data if possible and accurately flag Welch/unpaired testing.

### P1 — correctness and uncertainty
6. **Theory scope:** retain the conditional contraction bound; distinguish inequality validity from Adam λ-invariance and empirical causal mechanism. Explicitly state epsilon, optimizer-state and row-norm regimes.
7. **Moment theorem scope:** report raw, pre-ReLU/post-ReLU residuals; do not call clipped/ReLU transforms exact or globally displacement-optimal without a new proof.
8. **Kernel guarantee:** present Lemma 1 as conditional on a bounded RKHS loss; clarify 6,144-RFF finite approximation and pool factor. Run an RFF/pool sensitivity analysis.
9. **DP scope:** define adjacency/privacy unit, calibration pool assumption, all released messages, composition and exactly which variants are DP. Do not call full training DP. If MC-S/MC-B privacy is claimed, build and audit that mechanism separately.
10. **Ablation + MC-B:** run K14 seeds 3–4; report Holm-adjusted tests and per-seed effects. If five-seed findings remain mixed, make MC-B a separate variant instead of silently using it as the universal main method.
11. **Selection protocol:** report `final_bal` as primary LT endpoint; if `best_mean_acc` is retained to replicate the old paper, mark it historical and add final/validation-selected comparisons. Report raw versus `bbc_bal` separately.

### P2 — strengthen the contribution
12. Validate the F gate on a preregistered intermediate-F dataset or regime; otherwise state the four-dataset heuristic limitation.
13. Add measured bytes, rounds, peak server memory and wall-clock for mean/RFF vs MC-S/MC-B covariance paths, especially CIFAR feature dimension.
14. Keep negative methods (KME-Gen, VCT, Bayesian CSL, lazy decay, etc.) in supplement with matched budget and sources; this strengthens credibility when synchronized.
15. Prepare a 9-content-page anonymized NeurIPS PDF, appendix, checklist and anonymous code/data supplement. The checked-in Markdown/HTML/Word handout is not that final artifact.

## 13. Limitations of this audit

This is an evidence-based model-assisted simulation, not an official NeurIPS review and not 100 independent human reviewers. Scores and repeated issue counts are correlated by construction and are for triage, not acceptance probability. Repository review is snapshot-based; not all Kaggle reruns were executed during this audit. The latest Claude branch was treated as the current research candidate because it includes the most recent evidence, but it has not been merged into GitHub's default branch as of the audited snapshot.

## Official framework

The scales and contribution-type framing follow the [NeurIPS 2026 Reviewing Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) and [NeurIPS Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook). Both emphasize choosing the appropriate contribution type, technical soundness, clear reproduction details, and evidence-backed rather than vague novelty objections. The 2026 handbook also specifies a nine-content-page limit for the main paper and requires an anonymized submission plus a paper checklist. [Guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) · [Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook).
