# Adversarial review dossier: GeFL-F under long-tailed clients / Exact-Statistics Anchoring (ESA)

*Prepared 15 October 2026. Object of review: `docs/paper_draft.md` (current manuscript), with the implementation in
`experiments/core.py`, every `experiments/results/*/runs.jsonl`, and the supporting documents (`docs/PROGRESS.md`,
`docs/research/readout_report.md`, `docs/paper_single_idea.md`, the professor handout). Ten reviewers worked
independently and read-only. Their full reports are in `docs/review/reviewers/R1…R10_*.md`, and the scripts behind
every recomputed number are in `docs/review/work/r*_work/`. No research file was changed.*

**Names used.**
- GeFL-F: the original method.
- Ours-A: the final anchored method.
- HWA: holder-weighted averaging of the generator's class rows.
- PC-VAE: the generator conditioned on the exact class mean.
- MC / MC-S / MC-B: moment correction (spread / covariance / blended covariance).
- KH: kernel-herding sample selection.
- CSL: consensus soft labels.
- LA: logit adjustment with each client's own class shares.
- BBC: post-hoc balanced bias calibration.
- ER: exact readout, a last layer computed from secure-aggregated embedding sums.
- Oracle: each head's last layer refit on pooled real data, which is not federated.
- `final_bal`: balanced accuracy at the last round.
- `best_mean_acc`: the paper's metric. For each architecture it takes the best test accuracy over rounds, then averages over architectures.
- "LT" means the long-tailed setting: imbalance 100:1, Dirichlet(0.5), K = 10 clients, unless stated otherwise.

---

## 1. Executive rejection assessment

**Verdict.** The paper should be **rejected at a top-tier ML venue** (NeurIPS / ICML / ICLR level; the manuscript
names no venue). Confidence: **high (4/5)**. No single fatal mathematical error was found. The rejection rests on a
combination of three independently sufficient grounds:
- insufficient significance against the right baseline;
- insufficient novelty and theoretical depth;
- an evaluation protocol that cannot certify the claimed final method.

What survives is real. The large gains over the authors' own GeFL-F reproduction recompute exactly from the per-seed
logs and are robust to the choice of GeFL-F replicate:
- MNIST LT: +19.6, p = 0.0025;
- FashionMNIST LT: +21.2, p = 0.011;
- SVHN LT with the class-row branch: +15.1, p = 0.018.

The problem is what those gains are evidence *for*.

**The five most serious surviving objections:**

1. **Most of the gain is a last-layer (readout) correction, and the baseline that tests this is missing.**
   - Refitting only the last layer of *plain GeFL-F* gives 90.7 / 79.2 / 64.8 on MNIST / FashionMNIST / SVHN, against Ours-A's 94.8 / 79.7 / 65.6.
   - Ours-A's margin over that refit is +3.6 (p = 0.03) on MNIST, +0.7 (p = 0.23) on FashionMNIST and +0.8 (p = 0.29) on SVHN.
   - The authors' own federated version (ER) matches the refit within 0.8 points, but only on one seed at a shortened schedule.
   - The manuscript never reports this comparison. (R1, R3, R4, R5, R8, R10.)
2. **Novelty and depth.**
   - The organising principle (take class information from exactly aggregated sums) is the premise of CCVR, Fed3R, FedNCM and FedProto.
   - Every component has a direct precedent. The closest precedents not cited are DP-MERF (for KME-Gen), FedRoD (for LA) and FedVTC (for the generator conditioned on class statistics).
   - The collapse theorem is the triangle inequality applied to measured norms, and its threshold is not predictive.
   - The diagnosed mechanism disappears when weight decay is removed, and the standard optimiser fixes were never compared with HWA. (R1, R2, R10.)
3. **The final method is assembled per dataset on the test set.**
   - There is no validation split.
   - The number of synthetic epochs differs by headline: 10 for MNIST/FashionMNIST, 1 for CIFAR-10 LT and for all SVHN numbers.
   - The generator gate was introduced after the SVHN failure. It is never executed in code, and the gated method (job K12) was never run.
   - The gate's robustness claim is contradicted by the logs: CIFAR-10's separation statistic is 0.055–0.078, not "at least 0.14". (R3, R4, R5, R8, R9.)
4. **Small effects are not established.**
   - Three seeds; non-deterministic training (`cudnn.benchmark = True`). For Ours-A on MNIST, re-running the same seed varies more than changing the seed (0.44 vs 0.26 sd).
   - No multiplicity control: of 204 logged p-values, 6 survive Holm correction.
   - The K09 "30/30 negative, p = 2·10⁻⁹" pools dependent comparisons. The correct cluster-level test gives p = 0.031, and swapping in the other Ours-A replicate gives 24/30.
   - Component-level claims for MC, KH and CSL, and every "matches the paper's best" claim, are within noise. (R4, R8, R9.)
5. **Several stated facts are wrong.**
   - The 50/100-client table quotes runs that were invalidated by a bug fix.
   - "A class held by every client cannot be contracted" is false.
   - The +1.9% cost figure measures the minimal variant; Ours-A costs 2.2–2.8×.
   - "Formal privacy is cheap" covers class counts of the minimal variant only.
   - The CIFAR-10 balanced-setting win exists only under a test-selected best round: 56.82 vs 56.68 at the final round. (R2, R3, R6, R7, R9.)

**The rejection rests primarily on** insufficient significance against a simple baseline, insufficient novelty and theoretical depth, and an evaluation protocol open to selection on the test set. It does **not** rest on mathematical invalidity: the proofs are correct where they are stated precisely.

---

## 2. The ten independent reviews

Each summary below is condensed. The full report, with evidence ledger and file:line references, is in
`docs/review/reviewers/`.

### Reviewer 1: Originality and novelty (`R1_novelty.md`)
**Recommendation:** reject (top tier), confidence 4/5. Could be borderline at a specialised FL venue if repositioned.
**Actual contribution:**
- the first long-tail evaluation of GeFL-F;
- a measured collapse mechanism for one generator under one optimiser setting;
- an engineering integration of known statistic-based components;
- a readout/representation decomposition.

**Strongest arguments:**
- *Component-by-component precedent* (verified primary sources):

  | Component | Precedent |
  |---|---|
  | HWA | FedProto's holder-only, count-weighted rule; cwFedAvg class weights |
  | PC-VAE | FedProto prototype plus class-vector feature generators (f-CLSWGAN, Delta-encoder); FedVTC (ICLR 2026) conditions a generator on shared class means and covariances for model-heterogeneous FL |
  | MC / MC-S | CCVR's exact federated moments plus the classical Gelbrich / whitening-colouring map |
  | KH | kernel herding; distribution matching (DM, FedDM) |
  | KME-Gen | DP-MERF (AISTATS 2021); not cited |
  | CSL | FedMD consensus plus FedDF generator inputs |
  | LA | FedRoD balanced softmax with client counts (not cited); FedLC |
  | BBC | Saerens et al. 2002 |
  | ER | Fed3R / CCVR / FedNCM |

- *The principle is the premise of an existing line of work* (CCVR, Fed3R, FedNCM, FedProto). Ours-A also breaks it: the decoder is FedAvg-trained, and on SVHN class identity comes from FedAvg rows.
- *The genuinely new diagnosis is narrow:* coupled L2 decay applied through Adam to a one-hot embedding. It vanishes at λ = 0. The paper's other generators (DCGAN-F, DDPM-F) use no decay and were never measured under the long tail.

**Narrowest defensible claim.** In GeFL-F's released configuration the CVAE's class rows collapse under label skew, removing the non-holder decay term fixes it, and the mean-anchored stack adds +2 to +4 points over a recalibrated GeFL-F.

**Questions:**
- K15 results (AdamW / no row decay vs HWA)?
- GeFL-F + LA + {CCVR, Fed3R, ER} on all datasets?
- How does KME-Gen differ from DP-MERF, LA from FedRoD, and CSL from FedMD?

### Reviewer 2: Mathematical proofs (`R2_proofs.md`)
**Recommendation:** major revision, confidence 4/5 on demonstrated items.

**Correct:**
- Theorem 1 (triangle inequality).
- Lemma 2 (Adam scale invariance), for λ ≳ 1e-5 at these norms.
- Proposition 4 (MC minimum displacement over *all* maps, no Gaussian assumption needed).
- Proposition 6 (Gelbrich map).
- zCDP accounting.
- Lemma R1.
- ER equation (2) under its model.

**False:**
- "A class held by every client cannot be contracted by the aggregation" (`paper_draft.md:125-126`). Counterexample: two holders each grow the row from 1.0 to 1.30, and the average has norm 0.50. The authors' own T01 log shows classes 3–4 (held by 10/10 clients) contracting.
- The old K/2 bound (still quoted in `docs/diagnostics.md:19`). Counterexample with K = 1, d = 2, S = 1: change 1.41η against a bound of η.

**Stronger than the derivation:**
- *θ_c is not a threshold.* It is computed from the same round's measured gains. In a toy federation, holder gain is 4× larger when non-holders are present, and at equilibrium θ_c equals the holder share by identity.
- *HWA does not "prevent" contraction;* it removes one term.
- *MC is not exact.* The deployed MC (ReLU, plug-in moments from 500 samples, clamping, KH re-selection) does not meet the constraints of Proposition 4: after the ReLU the mean is off by 0.10√V and the spread ratio is 0.83. So "MC fixes exactly the two moments" is false for the implemented map.
- *Lemma 1 (KH risk bound) is vacuous for these heads.* ReLU-network losses, and even constants, are not in a Gaussian RKHS. KH is also not Frank–Wolfe (selection without replacement), and its MMD rises in 130 of 199 steps.
- *Proposition 3's variance term models subsampling, not the stated weighting.* The closed form w* changes, though the qualitative conclusion survives.

**Questions:**
- A predictive model of Ḡ(r) and D̄(r)?
- Post-ReLU, post-KH moments?
- Which neighbouring relation and trust model does the DP accounting assume?

### Reviewer 3: Federated system model (`R3_fl_system.md`)
**Recommendation:** reject in current form, confidence 4/5.

**What fails as an attack.** The training pipeline does not use test data or data the server should not have. MC, KH, PC-VAE, BBC and the BBC gate use only sums or synthetic samples.

**Strongest arguments:**
- *At K = 10 the heads are not federated.* There is exactly one client per architecture, so heads are never averaged, and "FedAvg per architecture" is local training (the authors' own K07 spec says so). All CIFAR-10, SVHN and ablation results are at K = 10.
- *A readout correction on plain GeFL-F recovers most of the gain:* 80% / 98% / 95% on MNIST / FashionMNIST / SVHN.
- *The "final method" is chosen per dataset.* The gate is never computed in code, and the SVHN number uses 1 synthetic epoch.
- *The system model is narrow:*
  - full participation every round;
  - optimiser state kept per client across rounds;
  - tail classes held by 6–8 of 10 clients, never the "few sites" the introduction motivates.
- *Running every architecture undercuts heterogeneity.* CSL and ER both require running every architecture's head, which weakens the case for heterogeneous models.

**Questions:**
- Results with several clients per architecture on CIFAR-10 and SVHN?
- Partial participation?
- Tail classes held by 1–2 clients?

### Reviewer 4: Statistics and inference (`R4_statistics.md`)
**Recommendation:** major revision; weak reject on secondary claims. Confidence 4/5.

**Robust:** the large gains over GeFL-F. They hold against each of five independent GeFL-F replicates.

**Demonstrated:**
- *Training is not deterministic.* Same-seed replicate sd is 0.3–0.96, comparable to the component effects.
- *K09 sign test is pseudo-replicated.* Five arms share one reference per (dataset, seed). The cluster sign test gives p = 0.031; the correct intersection–union test gives p = 0.106; with the other Ours-A replicate as reference, 24/30 differences are negative.
- *KH "9/9 positive, p = 0.004"* is a cross-job comparison mixing metrics. Within one job: p = 0.059 / 0.097.
- *Multiplicity:* 204 p-values, 6 survive Holm.
- *Power at n = 3:* 0.22–0.56 for the observed small effects. Every "tie" or "within noise" claim is uninformative without an equivalence test.
- *Test-set selection with no validation split,* shown in git history. The BBC gate, the generator gate, MC-B and T_s were all chosen after seeing test results.
- *The gate statistic contradicts the text:* CIFAR-10 F = 0.055–0.078.
- *`best_mean_acc` is a per-head test-set early stop over 9–17 evaluations.* Its optimism is +4.1 on CIFAR-10 balanced data and +3.7 on CIFAR-10 LT.
- *Against the refit GeFL-F,* Ours-A's margins on FashionMNIST and SVHN are not significant (p = 0.23, 0.29).
- *T03 pre-registration was honest but not independent:* one seed, and the follow-up re-tests the same seed.

**Questions:**
- Which data chose each design value?
- Final-round metrics for every balanced-setting claim?
- Which family of tests does each p-value belong to?

### Reviewer 5: Benchmarks and baselines (`R5_benchmarks.md`)
**Recommendation:** major revision, confidence 4/5.

**Demonstrated:**
- *No published FL long-tail method is run* (CCVR, CReFF, FedLC, FedRS, Fed3R, BalanceFL, FEDIC, FedLoGe, Fed-GraB).
  - SVHN and CIFAR-10 have no baseline outside the GeFL-F family: the K07 notebook was never run.
  - `baselines/creff.py` actually implements CCVR-style re-training and is never called.
- *The closest statistic-based competitor is left out of the main table.* FSG + LA ties the minimal method on MNIST (87.0 vs 87.9) and Ours-A on CIFAR-10 LT (44.17 vs 44.38).
- *BBC is reported only for Ours-A,* although GeFL-F + BBC is logged (79.24 on MNIST).
- *The CIFAR-10 setting is misreported.* The head classes hit the available images, so the effective imbalance is 48:1, not 100:1, and the total is 18,485, not 25,000.
- *Tail classes are the same in every seed* (always classes 6–9; FashionMNIST's hardest class, *shirt*, is always a tail class).
- *"Best of ten" margins compare our runs with numbers copied from the GeFL paper,* while our own GeFL-F differs from the paper's by −0.5 to +3.2.
- *Draft cells for K = 50 (86.8 / 87.2 / 87.9) have no valid source.*

**Narrowest supported claim.** HWA + LA repairs GeFL-F by 13–17 points (MNIST/FashionMNIST), and by 12 on SVHN with the row branch. The anchored stack adds 4–5 (MNIST) and 2–4 (FashionMNIST).

### Reviewer 6: Privacy and security (`R6_privacy.md`)
**Recommendation:** major revision. Confidence: high on accounting, medium on attacks.

**What holds:** the DP accounting is correct for add/remove neighbours with central noise, and the reported DP numbers recompute.

**Demonstrated:**
- *The guarantee covers one release of four statistics, per example, with a trusted noiser.* It does not cover:
  - generator and feature-extractor training;
  - the heads;
  - the CSL prediction sums sent every round;
  - the second moments (switched off in DP mode);
  - the BBC gate's exact counts.

  So "every statistic made differentially private" and "formal privacy is cheap" overstate the guarantee.
- *Under DP the rarest classes lose their correction steps.* They fall back to the unprotected generator.
- *Secure aggregation is assumed, not implemented.* If noise were added by the clients instead, collusion would degrade ε = 8 to 9.8 (3 of 10 clients colluding) or 32 (9 of 10).
- *The neighbour relation and the per-example unit are unstated.* A client holding 9 rare samples has an effective ε of about 148.
- *The clipping bound is computed on "public" data drawn from the training distribution.*
- *Membership inference on the exact class kernel mean of a 24-sample class:* AUC 0.78–0.93 (stand-in features, so medium confidence). At ε = 2 it drops to chance.
- *The manuscripts contradict each other* on which releases are noised.

**What fails as an attack:** no class has a single holder in the actual splits.

### Reviewer 7: Systems and communication cost (`R7_systems.md`)
**Recommendation:** major revision. Confidence: high on bytes, medium-high on wall-clock.

**What survives:**
- the statistics releases are small: 0.28 MB (MNIST), 0.35 MB (CIFAR-10);
- whole-training traffic grows +8% on MNIST and +24–28% on CIFAR-10, mostly from the larger PC-VAE (6.93 M vs 5.61 M parameters on CIFAR-10, not reported).

**What does not survive:**
- *Compute.*
  - The draft's "+1.9%" is the HWA + LA variant.
  - Ours-A at 10 synthetic epochs is 2.2–2.8× GeFL-F end to end and 5–14× in Stage III.
  - KH accounts for 83–88% of Stage III, because of a per-sample host synchronisation.
  - GPU memory: 8–14 GB vs 2.5 GB on SVHN.
- *The CSL protocol contradicts itself.* "No client message beyond GeFL-F's" vs per-round uploads. At K = 10, GeFL-F's Stage III has no traffic at all.
  - The pool must either be shipped (0.9 GB on MNIST, 18.9 GB on CIFAR-10) or regenerated bit-exactly on every client.
- *KME-Gen's "1/3000"* is arithmetically inconsistent and not like-with-like: about 1/190 for whole training.
- *ER is 2.08 M numbers per client, not "1.1 M per architecture".* It also broadcasts every single-client head to every other client.
- *MC-S's D×D release (5–29 MB) is missing from the accounting.*

### Reviewer 8: Ablation and causal attribution (`R8_ablation.md`)
**Recommendation:** major revision, confidence 4/5.

**Demonstrated:**
- *81–88% of Ours-A's gain over GeFL-F disappears under a last-layer refit* (MNIST / FashionMNIST).
- *LA's effect is entirely readout-level,* and BBC substitutes for it: LA's marginal value given BBC is 0.26 / 0.53.
- *CSL is the only component whose effect survives the refit fully,* and it is not an exact-statistics component.
- *PC-VAE replacing HWA changes accuracy by +0.34 (p = 0.53) on MNIST and −0.65 on FashionMNIST,* despite tail fidelity jumping 34 → 94. Its value appears only together with MC and 10 synthetic epochs, so the stand-alone "key fix" claim is contradicted.
- *KH's rationale is contradicted by the authors' own data.* MMD falls, but training only on fakes does not improve (−0.2 / −1.1), and there is no random-subset control.
- *MC's "through the spread" explanation is not isolated:* the mean also improves.
- *The step-by-step table chains three different jobs.* Same-seed drift between jobs (0.5–1.1) is as large as the PC, MC and KH steps.
- *The principle's own purest instances underperform* (FSG, KME-Gen, MC-S on MNIST), and on SVHN the method falls back to the opposite of the principle.

**Cross-examination correction (by the integrator):** R8's "minimal + oracle beats Ours-A by 1.06 (p = 0.016)" compares a refitted model with an unrefitted one. Oracle-vs-oracle, Ours-A leads (81.55 vs 80.73). The point is downgraded.

### Reviewer 9: Reproducibility and traceability (`R9_reproducibility.md`)
**Recommendation:** major revision. High confidence in the specific findings.

**What holds:** about 90% of checked cells recompute exactly or to rounding.

**Demonstrated:**
- *"Ours-A" names two configurations* (1 vs 10 synthetic epochs). CIFAR-10 LT is 44.38 or 46.1, and SVHN appears as 54.41, 55.76 or 55.89 depending on the document.
- *The 50/100-client table rows come from runs the authors set aside after a bug fix.* `F04_clients/runs_before_cslfix.jsonl` gives 86.78 / 87.17 / 87.86 and 82.71 / 85.45 / 85.22, exactly the draft's 86.8 / 87.2 / 87.9 and 82.7 / 85.5 / 85.2. The integrator verified this.
- *The code behind the Kaggle results cannot be recovered.* Runs record no commit hash, notebooks NB1–NB31 were never committed, and `requirements.txt` is unpinned and incomplete.
- *The abstract says HWA gives "+12 to +20" on CIFAR-10;* the data say +7.0.
- *The oracle is 300 SGD steps,* so it is not a true upper bound.
- *The K01 CIFAR-10 per-seed file is missing.* This is unavailable evidence, not misreporting.

### Reviewer 10: Area chair (`R10_area_chair.md`)
**Recommendation:** reject (top tier), confidence 4/5. The proofs it checked are correct.

**Three independently sufficient grounds:**
1. The diagnosis does not account for the failure, and the correct (recalibration) baseline is missing.
2. Originality and depth are insufficient: a triangle-inequality theorem, an informal Adam lemma, an untested trivial fix, and known components.
3. Generality and selection: four 10-class datasets, a one-block frozen feature extractor, one client per architecture, a test-selected composition, and a false gate claim.

**Cascades:**
- If K15 shows that removing decay matches HWA, the minimal-fix contribution shrinks to LA plus CSL.
- The readout finding undermines the motivation for the generator design, not just one number.

**What survives:** an empirical analysis paper on readout vs representation gaps, plus the coupled-decay pitfall. That fits an FL workshop, TMLR, or IEEE TMC/TNNLS.

---

## 3. Master flaw inventory (after cross-examination)

Severity: **1** = potentially fatal, **2** = major, **3** = moderate, **4** = minor.
"Indep." means the issue alone could sink acceptance at a top venue.
Status: D = demonstrated, U = unresolved, P = partly demonstrated.

| ID | Title | Sev | Manuscript claim / location | Evidence | Consequence | Conf | Status | Indep. |
|---|---|---|---|---|---|---|---|---|
| F1 | Gains are mostly a readout correction; recalibrated-GeFL-F baseline missing | **1** | Abstract l.30–37; §5.11 | GeFL-F oracle 90.7/79.2/64.8 vs Ours-A 94.8/79.7/65.6. Margins +3.6 (p=.03), +0.7 (p=.23), +0.8 (p=.29) (R4). 81–88% of the gain disappears under refit (R8). ER ≈ oracle within 0.8 (T03, 1 seed, 0.4× schedule) | Headline +15 to +21 overstates the generator-side contribution; true margin over a simple readout fix is 0.7–3.6 points (2.4–4.2 oracle-vs-oracle) | High (oracle) / Med (federated feasibility) | P | Yes |
| F2 | Novelty: principle and every component have precedents; key precedents uncited | 2 | Contributions l.69–90; single_idea:409 "prototypes have not been used to anchor a generator" | CCVR, Fed3R, FedNCM, FedProto, FedVTC (ICLR 2026), DP-MERF, FedRoD, FedMD/FedDF, Saerens (R1, R10; primary sources) | Contribution reduces to integration plus a narrow diagnosis | High | D | Yes (with F3) |
| F3 | Collapse diagnosis is a configuration artefact whose standard fix is untested | 2 | "structural failure, not a tuning issue" l.65 | Vanishes at λ = 0 (T01). AdamW / no-row-decay / lazy-decay arms never run at full schedule (K15). Quick 1-seed: no decay recovers 66% (MNIST) / 84% (FMNIST) of HWA's gain | HWA's necessity unestablished; the diagnosis may reduce to "do not L2-decay embeddings under Adam" | High | D (row level) / U (accuracy) | With F2 |
| F4 | Theory: tautological bound, non-predictive threshold, one false statement, guarantees not about the deployed method | 2 | §3.1 l.120–139; §4.5 MC l.331–372; KH l.400–404; Prop 3 l.257–274 | R2: "class held by every client cannot be contracted" false (counterexample + T01 log). θ_c endogenous. MC post-ReLU mean error 0.10√V, spread 0.83. Lemma 1 premise false for ReLU heads (B = ∞). Prop 3 variance term wrong for the stated model | "Each step has a guarantee" and "theory for each anchor" cannot be accepted; theory pillar thin | High | D | With F2 |
| F5 | Final method is selected per dataset on the test set and never run end to end | 2 | §4.6 "threshold is not tuned"; abstract CIFAR 46.1 | No validation split (`core.py:209`). T_s = 1 for CIFAR-10 LT and SVHN headlines, 10 elsewhere. Gate added after SVHN failure (git `6f9e1a3`), never executed in code. K12 (gated method) NOT RUN | No single pre-specified algorithm behind the headline table; optimistic selection bias | High | D | Yes |
| F6 | Gate robustness claim contradicted by the logs | 2 | Abstract l.42; §4.6 l.465–466 "≥ 0.14 elsewhere … any threshold 0.02–0.1 same choice" | Logged `class_sep_F`: CIFAR-10 0.055–0.078; SVHN 0.005–0.013 (integrator re-checked all runs). Row branch never run on CIFAR-10 | A threshold of 0.06–0.08 flips CIFAR-10; the gate is a post-hoc dataset switch | High | D | Part of F5 |
| F7 | Missing external baselines | 2 | "ahead of every other method" l.37 | Only FedAvg/LG-FedAvg (= local at K=10) on MNIST/FMNIST; none on SVHN/CIFAR-10 (K07 NOT RUN); FSG+LA (ties on CIFAR-10) excluded from main table | Comparative claims unsupported | High | D | With F1 |
| F8 | Balanced-setting (IID) superiority depends on test-selected best round and copied numbers | 2 (for IID claims) | Abstract l.48–51; §5.10 | CIFAR-10 IID: 60.93 vs 59.10 best-round, 56.82 vs 56.68 final (+0.14, p = 0.93) (integrator re-checked). Paper numbers single-valued; our GeFL-F offset −0.5…+3.2 | IID claims unsupported; FMNIST "match" and "+0.8 SVHN" within implementation offset | High | D | No |
| F9 | Small-effect and component claims not statistically established | 2 | §5.11 KH "9/9, p=0.004"; K09 "30/30, p=2e-9"; "tie"/"no cost" claims | Pseudo-replication; non-deterministic training (replicate sd ≥ seed sd for Ours-A MNIST); 204 p-values, 6 Holm; power 0.22–0.56 | MC, KH, CSL, MC-B and "matches" claims are exploratory | High | D | No |
| F10 | Reporting errors | 2 | §5.3 table l.520–532; §5.6 "within 0.4"; abstract HWA "+12 to +20"; CHECK.md | K=50/100 rows from `runs_before_cslfix.jsonl` (verified); HWA weighting rules differ 0.99 on MNIST (n = 2); CIFAR HWA gain +7.0; 13 CHECK mismatches; "Ours-A" means two configs | Must be fixed before any submission; erodes trust; not misconduct (authors' own CHECK flags several) | High | D | No |
| F11 | Causal story partly contradicted | 2 | "mean anchor is the key fix"; KH via MMD bound; MC "through the spread" | PC replacing HWA: +0.34 / −0.65 (n.s.) despite +60 pts fidelity; KH TSTR −0.2/−1.1; no mean-only/spread-only MC arm | Mechanism claims must be withdrawn or tested | High | D | No |
| F12 | Privacy claims broader than the guarantee | 3 | "Formal privacy is cheap" l.51–52; "every statistic made DP" | Guarantee: one release, per example, trusted noiser; generator/heads/CSL/second moments/gate not covered; DP on Ours-A stats costs 4.5–6.9 (ε=8) and 9.8–14.1 (ε=2); MIA AUC 0.78–0.93 on exact class kernel mean (stand-in features) | Must narrow; no end-to-end DP | High (scope) / Med (MIA) | D | No |
| F13 | Cost claims describe the minimal variant | 3 | §5.12 "+1.9%", "C numbers per round" l.876–878 | Ours-A 2.2–2.8× end to end, Stage III 5–14×, KH 83–88%; CSL protocol contradictory; PC-VAE +23% bytes on CIFAR-10; MC-S D×D omitted | Practicality overstated | High | D | No |
| F14 | Scope: heterogeneity, sites, datasets, regimes | 3 | Title, intro "rare at only a few sites" l.60 | K = 10 ⇒ one client per architecture; tail held by 6–8/10 clients; 4 × 10-class 32×32 datasets; one-block frozen extractor; fixed tail classes; Ours-A only at IF 100 and IID; α = 0.5 only; CIFAR effective IF 48 | Generalisation claims exceed evidence | High | D | Contributes to rejection |
| F15 | Reproducibility | 3 | §5.1 "each seed fixes … initialisation" | `cudnn.benchmark=True`; no commit hashes; notebooks NB1–31 not committed; unpinned requirements; K01 CIFAR per-seed file missing | Exact reproduction impossible; cross-job pairing pairs splits, not models | High | D | No |
| F16 | Old false K/2 bound still in a project document | 4 | `docs/diagnostics.md:19` | Counterexample (R2) | Internal only | High | D | No |
| F17 | Pixel vs feature F mixed in Prop 5 discussion | 4 | §4.5 l.315–320 | R2 | Inconsistent numbers | High | D | No |

**Attacks that failed or were withdrawn after cross-examination:**
- *The server secretly uses test or private data in training.* **Fails.** BBC is fitted on synthetic data, the gate uses training counts, and MC/KH/PC use sums (R3).
- *The DP accounting is wrong.* **Fails.** It is correct for add/remove neighbours with central noise (R2, R6).
- *Propositions 4 and 6 are false.* **Fail.** Both are correct over all maps with no Gaussian assumption; only the deployed map differs (R2).
- *The headline gains over GeFL-F are noise or a lucky replicate.* **Fails.** They are robust to five independent GeFL-F replicates (R4).
- *The statistics releases are costly.* **Fails.** They are 0.28–0.35 MB once; Stage II dominates (R7).
- *A single-holder rare class is exposed by secure aggregation.* **Does not apply** to the actual splits (R6).
- *The T03 pre-registration was written after the results.* **Fails.** It was committed before the first result; the scoring is honest (R4).
- R8's "minimal + oracle beats Ours-A" is **downgraded**: it compares an oracle with a head (see §2).
- R1/R8's "removing decay ≈ HWA" is **downgraded** to "recovers 66% / 84% in a one-seed quick pass".
- R3's "GeFL-F + ER (79.1) beats Ours-A (75.7)" is **downgraded**: that Ours-A ran at 1 synthetic epoch and a 0.4× schedule. At full schedule Ours-A is 79.7.
- **Counterargument to F1 that survives in part.** At K = 10, a *standard* per-architecture CCVR or Fed3R would see only one client's data. The 90.7 / 79.2 / 64.8 figures need statistics pooled through every architecture's body, which is the authors' own ER extension and requires broadcasting each single-client body (R7). So "a published method already recovers it" is not demonstrated. What is demonstrated is that "a simple readout fix of plain GeFL-F recovers most of it". That still suffices for the significance objection, and the paper must report it.

---

## 4. Mathematical attack report

| Result | Original claim | Reconstructed result | Status | Implication |
|---|---|---|---|---|
| Theorem 1 | ‖r⁺‖ ≤ ‖r‖ + pḠ − (1−p)D̄; contraction iff p < θ | Triangle inequality; correct. Ḡ and D̄ are measured, so the bound restates the data. θ is endogenous (= p at equilibrium) | Correct; tautological; "threshold" stronger than the derivation | No predictive diagnosis |
| "Held by every client ⇒ cannot contract" | l.125–126 | Counterexample: two holders 1.0 → 1.30, average 0.50; T01 classes 3–4 contract | **False** | Delete |
| Lemma 2 (λ-invariance) | D̄ independent of λ > 0 | Exact when ε = 0; numerically invariant for λ ≳ 1e-5, D̄ = 0.48 at 1e-7 and 0.10 at 1e-8 | Correct under an extra assumption (λ‖r‖ ≫ ε) | Restrict prediction (ii) |
| Old K/2 bound | Δ\|w\| ≤ (2m/K − 1)Sη | Wrong units (norm step up to √d·η), overshoot, zero rows | **False** (already retracted in the draft; survives in `diagnostics.md`) | Correct the reason given for retraction |
| HWA prevents contraction | prediction (iv) | Removes the D̄ term only; holder cancellation remains; T01 HWA rows shrink in 1–6/40 rounds | Stronger than the derivation | Reword |
| Prop 3 (synthetic budget) | E = σ²(1−w)/n_r + w²b² | Correct variance for weighting: σ²[(1−w)²/n_r + w²/n_s]; w* = (σ²/n_r)/(σ²/n_r + σ²/n_s + b²) | Proof correct for an unstated model; formula wrong for the stated one; qualitative conclusion survives | Fix model statement |
| Prop 4 (MC) | Minimum displacement subject to exact mean and spread | Correct over all couplings, no Gaussianity needed. Deployed map (ReLU, 500-sample moments, clamp, KH) violates the constraints | Correct (idealised); "fixes exactly" false as implemented | Report post-ReLU moments |
| Prop 5 (separation) | W₁ ≤ (1 + L‖A‖)‖m_c − m_c′‖ | Correct for class-independent latents; L may be huge; MC extension unproved | Correct; extension unverified | Upper bound only; no class identity guarantee |
| Prop 6 (Gelbrich / MC-S) | Minimum displacement to exact covariance | Correct (numerically exact on t₅ data); deployed map clips, shrinks and applies ReLU | Correct (idealised); "exact covariance" false for tail classes | Same fix |
| MC-B "empirical Bayes" | Bayes blend | Fixed-strength shrinkage, no fitted prior | Heuristic | Wording |
| Lemma 1 (KH risk bound) | Gap ≤ B·MMD | Premise ℓ ∈ RKHS false for ReLU heads (constants not in a Gaussian RKHS); RFF space finite | Correct inequality, inapplicable | KH motivation is heuristic only |
| KH = Frank–Wolfe | l.401 | Herding without replacement from 4n; MMD non-monotone (rises 130/199 steps) | Description inaccurate | Rename |
| Prop 7 (MixUp) | Second-order expansion | Formal expansion correct; remainder unbounded | Correct formally | Minor |
| zCDP accounting | σ = √(2/ρ); conversion | Correct (Bun–Steinke) for add/remove, central noise | Correct; scope unstated | Narrow claims |
| Lemma R1, ER eq. (2) | Readout report | Correct under the LDA / Gaussian model | Correct (model-restricted) | Label as models |

No theorem whose failure would invalidate the *empirical* results was found. The failures remove the "theory for each
step" pillar and the predictive status of the collapse diagnosis.

---

## 5. Novelty and prior-art report

- **Already known:**
  - holder-only count-weighted aggregation (FedProto, cwFedAvg);
  - exact federated class moments and virtual features (CCVR);
  - closed-form classifiers from federated sums (Fed3R, FedNCM);
  - prototype- or statistics-conditioned generators for heterogeneous FL (FedVTC, ICLR 2026);
  - class-vector-conditioned feature generators (f-CLSWGAN, Delta-encoder);
  - kernel-mean-embedding generators (DP-MERF) and herding;
  - balanced softmax with client counts (FedRoD), logit adjustment (Menon et al.), FedLC;
  - consensus distillation on generated inputs (FedMD, FedDF);
  - EM prior correction (Saerens et al.);
  - Gaussian optimal-transport maps (Gelbrich; Olkin–Pukelsheim);
  - the fact that L2 under Adam is not weight decay (AdamW);
  - class-specific weights degrading under missing classes (FedAwS, FedRS, FedLC);
  - mode collapse under class conditioning (Shahbazi et al.; CBDM).
- **Genuinely different:**
  - the first long-tail evaluation of GeFL-F;
  - the identification of coupled-decay-through-Adam plus flat FedAvg as the collapse mechanism for a *generator's* conditioning rows (narrow; a configuration pitfall);
  - the readout-vs-representation decomposition of GeFL-F's long-tail deficit;
  - the per-architecture extension of exact readout, which needs cross-architecture statistics.
- **Adaptation or combination:** PC-VAE, MC on generator samples, KH toward federated targets, the separation gate, and the overall ESA stack.
- **Unsupported novelty claims:**
  - "prototypes, second moments, kernel embeddings have not been used to anchor a generator" (`paper_single_idea.md:409`);
  - "a principle";
  - KME-Gen as a contribution;
  - "none of these methods trains a class-conditional generator by FedAvg" (FedCG, FedFTG);
  - "ahead of every other method".
- **Is the remainder significant enough?** Not for a top-tier main track. It is plausible as an analysis paper (workshop / TMLR / journal) centred on the decomposition and the decay pitfall, with honest margins against a recalibrated baseline.

---

## 6. Experimental and statistical integrity report

**Demonstrated defects:**
- no validation split; per-dataset configuration chosen on the test set (F5);
- false gate statistic statement (F6);
- best-round test selection drives the CIFAR-10 balanced-setting claim (F8);
- pseudo-replicated sign tests (F9);
- no multiplicity control (F9);
- non-deterministic training (F15);
- table rows from invalidated runs (F10);
- "within 0.4" contradicted (F10);
- BBC applied asymmetrically (R5);
- FSG excluded from the main table (R5);
- cross-job step table (R8);
- the CIFAR-10 effective imbalance is 48:1, not 100:1 (R5);
- fixed tail class identities (R5);
- KH's and MC's mechanisms untested or contradicted (F11).

**Questions that need evidence rather than being shown false:**
- federated ER at full schedule with 3 seeds (F1);
- K15, removing decay vs HWA (F3);
- rows branch on CIFAR-10 (F6);
- K14 extra ablation seeds (F9);
- the K01 CIFAR-10 per-seed file (F15; unavailable, not misreported);
- cross-device reproducibility of the KH pool (R7, suspected).

**Verified as sound:**
- pairing of splits by seed;
- headline recomputations, about 90% of cells exact (R9);
- no test leakage in training (R3);
- T03 pre-registration scoring (R4).

---

## 7. Privacy, security and systems report

| Property | Status |
|---|---|
| Exact sums are mathematically what secure aggregation returns | Established (simulation; protocol assumed, not implemented) |
| (ε, δ)-DP of the four statistics | **Conditional:** add/remove neighbours, per example, trusted central noiser, clipping on in-distribution "public" data |
| DP of the full pipeline (generator, heads, CSL sums, gate, ER second moments, MC-S) | **Not established**; several releases are not privatised at all |
| "Formal privacy is cheap" | **Overstated:** true for class counts of the minimal variant at ε = 1 (≤ 1.3 points); Ours-A statistics cost 4.5–6.9 (ε = 8) and 9.8–14.1 (ε = 2); FMNIST ε = 2 vs GeFL-F +7.1, p = 0.084 |
| Leakage of exact rare-class statistics | **Suspected, medium confidence:** membership AUC 0.78–0.93 on a 24-sample class using the kernel mean (stand-in features) |
| ER exposes single-client heads (K = 10) | Demonstrated by protocol (every body broadcast) |
| Statistics communication | Established: 0.28–0.35 MB once (two sequential SA rounds) |
| Whole-training communication | +8% (MNIST), +24–28% (CIFAR-10), not reported |
| Compute | **Contradicts the draft:** 2.2–2.8× end to end vs "+1.9%"; KH 83–88% of Stage III |
| CSL per-round protocol | Under-specified and contradictory; pool shipping 0.9–18.9 GB per client or bit-exact regeneration |

---

## 8. The rejection case as an actual review

> **Summary.** The paper studies GeFL-F, a generator-based model-heterogeneous FL method, under long-tailed,
> label-skewed clients. It argues that Adam's coupled weight decay and flat FedAvg contract the conditioning rows of rare
> classes in the CVAE feature generator, and proposes Exact-Statistics Anchoring: a mean-conditioned generator, moment
> correction, kernel herding, consensus soft labels, client logit adjustment, calibration and a separation gate. On
> MNIST, FashionMNIST, SVHN and CIFAR-10 (IF = 100, Dir 0.5, mostly K = 10) it reports +12 to +21 balanced-accuracy
> points over GeFL-F.
>
> **Strengths.** Careful paired experiments; a faithful reproduction of GeFL-F; candid reporting of negative results;
> an instrumented demonstration that tail-class rows shrink under coupled decay and recover with holder-only averaging.
>
> **Weaknesses.**
>
> *1. The gain is not attributed to the proposed mechanism, and the decisive baseline is absent.* The authors' own
> diagnostics refit only the last layer of each GeFL-F head on real features. This recovers 90.7 / 79.2 / 64.8 on
> MNIST / FashionMNIST / SVHN, i.e. 80–98% of the method's improvement. Against this, the full method leads by 3.6 (MNIST),
> 0.7 (FashionMNIST, p = 0.23) and 0.8 (SVHN, p = 0.29). A federated closed-form readout from aggregated statistics
> reaches the same accuracy in a pilot. Classifier recalibration from federated statistics is established (CCVR, Fed3R,
> FedNCM). Without a recalibrated GeFL-F baseline, and without any published long-tailed FL method (FedLC, FedRoD, CReFF,
> CCVR, Fed-GraB) or heterogeneous-FL method under the long tail, the significance of the generator-side contribution
> cannot be assessed. On two of three datasets it appears to be under one point.
>
> *2. Limited originality and theoretical depth.* Theorem 1 is the triangle inequality over measured norm changes; its
> threshold is computed from the same measurements and is not predictive. The accompanying statement that a class held
> by all clients cannot contract is false (counterexample; it also contradicts the paper's own log). The causal content
> lies in Lemma 2, a direct consequence of Adam's scale invariance. The paper shows the effect vanishes at λ = 0 but never
> compares HWA with the obvious remedies (AdamW, no decay on embeddings), so "a structural failure, not a tuning issue"
> is unsupported. The "anchor to exact statistics" principle is the premise of CCVR, Fed3R and FedProto. A
> statistics-conditioned generator for heterogeneous FL appears in FedVTC, and the server-side kernel-embedding generator
> is DP-MERF (not cited). Logit adjustment with client counts is FedRoD (not cited), and consensus labels are
> FedMD/FedDF. The optimality results (Gelbrich, Cauchy–Schwarz) are classical and do not describe the deployed maps. The
> kernel-herding risk bound requires the loss to lie in a Gaussian RKHS, which a ReLU network's loss does not.
>
> *3. The evaluated method is not a single, pre-specified algorithm.* There is no validation set. The synthetic budget
> is 10 epochs for some headline numbers and 1 for others (CIFAR-10 LT, all SVHN results). The separation gate was added
> after the method failed on SVHN, is not executed in the code, and its stated robustness ("F ≥ 0.14 on every other
> dataset; any threshold in 0.02–0.1 makes the same choice") is contradicted by the logged values (CIFAR-10: 0.055–0.078).
> The gated pipeline has never been run end to end. The balanced-setting comparisons use the best test round per head;
> under final-round accuracy the CIFAR-10 advantage is +0.14.
>
> *4. Statistical support for component claims is weak.* With three seeds, non-deterministic training (same-seed
> replicates differ as much as seeds), 200+ uncorrected p-values and pseudo-replicated sign tests, the contributions of
> MC, KH and CSL (0.3–1.6 points) and every "matches the best of ten methods" claim are not established.
>
> *5. Several statements must be corrected.* The 50/100-client table quotes runs invalidated by a bug fix. The cost
> section measures a different variant (the proposed method is 2–3× slower, not +1.9%). "Formal privacy is cheap" refers
> to class counts of the minimal variant; the generator and per-round label sums carry no guarantee.
>
> **Assessment.** The engineering is sound and the analysis of where GeFL-F's error lies is valuable. As submitted, the
> contribution over a simple recalibration is small and unproven outside MNIST, and the novelty is an integration of
> known statistic-based components. The theory does not describe the implemented method, and the evaluation protocol
> does not certify a single method. **Reject.** I would welcome a resubmission framed as an analysis of
> readout-vs-representation failure in generator-based FL, with the recalibration baselines, the optimiser controls and a
> validation protocol.
>
> **Score:** 3/10 (reject). **Confidence:** 4/5.

---

## 9. Author rebuttal stress test

| Objection | Strongest plausible rebuttal | Evaluation |
|---|---|---|
| F1 readout baseline | "The oracle uses pooled data and is not federated. The per-architecture CCVR at K = 10 sees one client. ER needs cross-architecture bodies, a new protocol we can claim. With ER on both sides Ours-A still leads by +4.2 / +2.4 / +3.6, so the generator improves representations." | **Resolves partly under narrower claims.** The oracle-vs-oracle margin is real (3 seeds). But the abstract's +15 to +21 must become +2.4 to +4.2 over a recalibrated baseline, and ER must be run at full schedule, so **new experiments are required**. The rebuttal also turns ER into a contribution, which moves the paper toward the analysis framing. |
| F2 novelty | "The integration inside a heterogeneous generator pipeline, the diagnosis and the decomposition are new; FedVTC is concurrent (ICLR 2026)." | **Partly convincing.** FedVTC timing may be concurrent and should be argued with dates. DP-MERF, FedRoD, CCVR and FedProto predate and must be cited. Novelty remains incremental for a top venue. **Fails at top tier, resolves at a lower tier.** |
| F3 untested optimiser fix | "Lemma 2 itself says λ = 0 works. HWA also handles the dilution that remains without decay, which grows with K." | **Requires the experiment.** Dilution is real (Theorem 1 at λ = 0), but whether it costs accuracy at K = 50/100 is exactly what K15 measures. Until then "structural, not a tuning issue" must go. |
| F4 theory | "Theorem 1 is presented as a consistency check and the λ contrast is the test; Props 4/6 are cited as classical; the RKHS bound is motivation." | **Resolves only by narrowing.** Delete the false sentence, call θ a decomposition, state that guarantees concern idealised maps, and drop "each step has a guarantee". A **corrected proof is not possible** for "held by all ⇒ no contraction"; it must be deleted. |
| F5/F6 selection and gate | "Choices follow mechanisms (Prop 5 for the gate; Prop 3 for T_s), not test scores; the large gains dwarf selection effects." | **Fails for the specific claims.** The git history shows ordering, and the F values contradict the stated margin. The large GeFL-F gaps survive selection, but small margins and the gate do not. A **validation protocol and an end-to-end K12 run are required.** |
| F7 baselines | "FL-LT methods target a single global model; adapting them to heterogeneous heads is non-trivial." | **Weak.** GeFL-F already averages per architecture, and LA/FedLC/FedRoD/CCVR apply per head on the same frozen extractor. **Experiments required.** |
| F8 IID metric | "best_mean_acc is the GeFL paper's metric; we must compare on it." | **Resolves only for comparison with the paper,** with final-round numbers alongside. The CIFAR-10 gain does not survive. **Narrow the claim.** |
| F9 statistics | "Effects are consistent in sign across seeds and datasets." | **Fails.** Dependence and replicate variance are demonstrated. Needs ≥ 5 seeds, within-job references and Holm correction. |
| F10 reporting | "Draft not yet regenerated from the asset pipeline." | **Resolves by correction.** It does not affect the method, but it must be fixed before submission. |
| F12 privacy | "DP is an optional add-on; secure aggregation protects the sums." | **Resolves by narrowing.** Secure aggregation does not imply DP, and the abstract sentence must change. The membership result needs the real extractor to confirm (suspected). |
| F13 cost | "Time is dominated by an unoptimised KH loop; it can be batched." | **Plausible.** It still requires reporting the actual cost and an optimised measurement. |

---

## 10. Decisive failure tests (ranked by information gain per cost)

Compute rules from `CLAUDE.md` apply: reuse seeds 0–2 and existing references. ER is now logged automatically (`er_bal`)
in every run, so new jobs can include it at no extra cost.

| Rank | Test | Resolves | Cost | Impact on verdict |
|---|---|---|---|---|
| 1 | **GeFL-F, GeFL-F + LA, Ours-A and the minimal variant with `er_bal` logged**, full schedule, seeds 0–2, all four datasets (the CIFAR-10/SVHN job already proposed, plus MNIST/FMNIST) | F1 (and F7 partly) | ≈ 6–10 GPU-h | Decisive. If Ours-A + ER − GeFL-F + ER ≥ 3 on ≥ 3 datasets, significance survives in narrowed form; if ≤ 1, the generator stack is not justified |
| 2 | **K15** (AdamW matched / no decay on rows / lazy decay vs HWA, K = 10/50/100; notebooks ready, 0 duplicates) | F3 | ≈ 7 GPU-h | Decides whether HWA is a contribution or one of several equivalent fixes |
| 3 | **Text corrections** (no compute): gate F values; delete "held by every client"; replace the K = 50/100 rows; fix the cost section, privacy sentence, HWA CIFAR gain and "within 0.4"; final-round metrics beside best_mean_acc | F4 (part), F6, F8, F10, F12, F13 | 0 | Removes all demonstrated false statements |
| 4 | **Validation protocol:** freeze one configuration (single T_s rule, gate in code) chosen on the 2,000 held-out images; run K12 end to end on new seeds (3–5) | F5, F6 | ≈ 6–8 GPU-h | Converts the "final method" into a single algorithm |
| 5 | **Two external FL-LT baselines on the frozen extractor** (FedLC or FedRoD-style head loss; CCVR per architecture) and FSG + LA in all main tables | F7 | ≈ 2–4 GPU-h (head stage only, cached generators) | Required for any comparative claim |
| 6 | **K14** (ablation seeds 3–4) + within-job references + Holm; mean-only / spread-only MC; KH vs random 1-of-4 subset | F9, F11 | ≈ 4 GPU-h (head stage only) | Establishes or removes component claims |
| 7 | **Row branch on CIFAR-10** (LT and IID) | F6 | ≈ 4 GPU-h | Tests whether the gate's choice matters at CIFAR-10's F |
| 8 | Membership test with the real trained extractor at ε = ∞ / 8 / 2 | F12 | CPU, < 1 h | Confirms or removes the leakage concern |

Until tests 1 and 2 are run, the conclusions about significance (F1) and HWA's necessity (F3) remain **provisional**.

---

## 11. Final publication decision

- **Recommendation:** reject at a top-tier ML venue (NeurIPS / ICML / ICLR main track).
- **Confidence:** 4/5.
- **Smallest set of independently sufficient rejection grounds:**
  1. **Significance (F1 + F7).** No recalibrated or external baseline; the margin over a simple readout fix is 0.7–3.6 points and not significant on two of three datasets.
  2. **Novelty and depth (F2 + F3 + F4).** The principle and components are known; the theorem is non-predictive and contains a false statement; the diagnosed mechanism's standard fix is untested.
  3. **Evaluation validity (F5 + F6).** The final method is selected per dataset on the test set, never run end to end, and its gate claim is contradicted by the logs.
- **Most significant unresolved uncertainties:**
  - federated ER at full schedule;
  - K15;
  - the CIFAR-10 recalibration margin (Ours-A's own readout gap there is 6.2, the largest, so the generator may matter most on the hardest dataset);
  - whether gains hold with many classes.
- **Does a meaningful contribution remain if the weakest claims are removed?** Yes. The remainder:
  - a documented pitfall (coupled decay under Adam with flat FedAvg erases rare-class conditioning);
  - a readout-vs-representation decomposition showing that GeFL-F's long-tail deficit is mostly last-layer error on classes a client does not hold, which offsets cannot fix and exact sums can;
  - a measured +2.4 to +4.2 representation gain from faithful synthetic features.
- **Current manuscript:** reject outright at a top-tier venue. Reconsider after major revision as an analysis paper at a workshop, TMLR, or an IEEE journal (TMC/TNNLS). Ranks 1–5 of §10 would be required for that, and the abstract's numbers would have to be restated against the recalibrated baseline.

---

*Full reviewer reports: `docs/review/reviewers/R1_novelty.md` … `R10_area_chair.md`. Recomputation scripts:
`docs/review/work/r*_work/`. The scripts were written to run from the session scratchpad; their data paths point at
`/home/user/Fedlearning/experiments/results/`.*

---

## Addendum (15 Oct 2026): issues added after comparison with the GPT review package

See `docs/review/gpt_review_comparison.md`. The verdict is unchanged. Added to the inventory:

| ID | Title | Sev | Evidence | Status |
|---|---|---|---|---|
| F18 | FedNPC (CVPR 2026 Findings), post-hoc classifier calibration for federated long-tailed learning, is not cited and is a candidate baseline next to CCVR / Fed3R / ER | 3 | CVF open-access supplement exists; main paper not yet read | D (existence) / U (fit to per-architecture heads) |
| F19 | Worst-class accuracy missing from headline tables (e.g. MC-B FMNIST LT 81.57 balanced vs 33.19 ± 4.02 worst class) | 3 | `E32_moment_anchor_bayes/summary.md` | D |
| F20 | No anonymised NeurIPS-format manuscript or checklist; the latest work is only on an unmerged branch (`main` still has the old K/2 theorem) | 4 | branch tips; repository tree | D |

GPT's statement that the gate statistic is "≈ 0.14–0.60" outside SVHN repeats the draft's error and is contradicted by the logs (F6).
