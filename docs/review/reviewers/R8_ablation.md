# Reviewer 8: Component analysis and causal interpretation

**Specialty.** Experimental science: ablation design, mechanism tests, causal attribution of ML results.

**Recommendation.** Major revision. As written, the evidence does not support the causal and unifying claims. The empirical gains over GeFL-F are real and reproducible: K09's Ours-A matches E25 within 0.4 points across hardware, and every seed is positive. The problem is the explanation. The data show that most of the long-tail gain is a correction to the classifier's readout. They do not show that "anchoring the generator to exact statistics" is the cause, and the principle has no demonstrated predictive power.
**Confidence.** 4/5. I recomputed every number in this report from `runs.jsonl` and the D01 JSON files; the scripts are in `r8_work/`.

---

## 1. Headline causal finding (recomputed)

Every run logs `oracle_bal`: the same heads with only their last layer re-fit, class-balanced, on pooled real features. The difference between `final_bal` and `oracle_bal` separates two things:
* what a component does to the **representation** (the head bodies);
* what it does to the **readout bias** (the last layer).

| Contrast (paired by seed, n = 3 unless stated) | Δ final_bal (MNIST / FMNIST) | Δ oracle_bal (MNIST / FMNIST) | Share of the gain that survives a readout re-fit |
|---|---|---|---|
| Ours-A (T_s=10, E25) vs GeFL-F (E20) | +19.57 / +21.21 | +3.73 / +2.62 | 19% / 12% |
| HWA removed from the minimal variant (K02, n=2) | −6.93 / −7.80 | −1.10 / −0.67 | 16% / 9% |
| LA removed (K09) | −2.98 / −6.18 | +0.06 / +0.32 | ~0 |
| PC removed (K09, becomes CVAE+HWA) | −4.72 / −0.67 | −2.46 / **+0.21** | 52% / none |
| MC removed (K09) | −0.96 / −0.49 | −0.76 / +0.16 | 79% / none |
| KH removed (K09) | −0.71 / −0.30 | −0.39 / −0.11 | 55% / 37% |
| CSL removed (K09) | −1.63 / −1.38 | −1.40 / −1.43 | **~100% / ~100%** |
| SVHN rows branch vs GeFL-F (K10 vs K01) | +15.11 | +3.55 | 23% |

**What this shows about simpler baselines.**
* GeFL-F with an oracle readout scores 91.21 / 78.94 (E20). Ours-A beats it by +3.61 (p = 0.028) on MNIST and by only **+0.74 (p = 0.23)** on FMNIST. On SVHN the rows branch beats GeFL-F's oracle by +0.78.
* The minimal variant (HWA+LA+CSL, E20) with an oracle readout **beats Ours-A on FMNIST by 1.06 (p = 0.016)**. At T_s = 10 (E16) the margin is 2.08.
* The oracle uses pooled data, but the federated exact readout (ER) is within 0.8 points of it (T03: 1 seed, schedule × 0.4). With ER, GeFL-F reaches 90.8 / 79.1, against 92.8 / 80.3 for Ours-A.

**Conclusion.** Most of the accuracy gain is classifier-side prior and readout correction. "Anchoring improves generator fidelity" turns into representation-level accuracy only on MNIST, and there by about 2–4 points. CSL is the one component whose effect is fully representational, and it is **not** an exact-statistics component.

---

## 2. Component-to-claim map

Design codes:
* **LOO**: leave one out from the full Ours-A (K09: same FE and generator per seed, T_s = 10, n = 3).
* **ADD**: added on top within one experiment.
* **X-EXP**: cumulative step across different experiments (different generator trainings on the same split seeds).

| Component | Claim (location) | Evidence file | Design | n | Effect ± sd of paired diffs (MNIST / FMNIST) | Mechanism tested? | Simpler alternative tested? | Verdict |
|---|---|---|---|---|---|---|---|---|
| **HWA** | Removes the non-holder decay term; +9.0 / +11.2 (§4.1, §5.3) | T01, F01, K02, D01_kaggle | ADD (F01); LOO from minimal (K02) | T01: 1 seed; K02: 2 | LOO −6.93±0.48 / −7.80±1.20; oracle −1.10 / −0.67 | **Yes, well.** Row norms, λ-invariance and HWA arms (T01); fidelity 10→31 / 20→71 (D01) | Weighting rule yes (irrelevant). **No-decay / AdamW only in a quick pass, 1 seed** (E02: NOWD 62.55 vs HWA 63.43 FMNIST; 69.87 vs 70.70 MNIST). K15 not run | Mechanism well shown for row collapse, but the accuracy channel is mostly readout. It is unknown whether removing weight decay is as good |
| **PC-VAE** | "Key fix"; class identity from the exact mean; "why the anchor beats holder averaging" (§4.5, handout 7.1) | E20, K09, D01, K08/K10 | ADD replacing HWA (E20); LOO (K09) | 3 | ADD: **+0.34±0.79 (p=0.53) / −0.65±0.46 (p=0.14)**; oracle −0.63 / −0.28. LOO: −4.72±1.09 / −0.67±0.41 (p=0.11). SVHN: −9.71±0.74 vs rows | Fidelity yes (34→94 / 68→90). TSTR +11 / +1. But fidelity gains do **not** turn into accuracy on their own | CVAE+HWA is the comparison (it is the K09 −PC arm). No frozen-row or mean-initialised-row control | **Not supported as a stand-alone cause.** Its value appears only with MC and T_s = 10 (an interaction). It loses on SVHN |
| **MC** | Minimum-displacement correction of mean and spread; "works through the spread" (§4.5, K09 note 6, handout 7.2) | E20, K09, D01 | ADD on PC; ADD on CVAE; LOO | 3 | On PC +1.56±0.39 / +1.60±0.15. On CVAE +0.71±1.12 (n.s.) / +0.59±0.28. LOO −0.96±0.28 / −0.49±0.29 (p=0.10) | Partly. Spread 0.44→0.85, MMD 0.196→0.088. Raw→MC tail fidelity *falls* (K09: 0.92→0.87 MNIST, 0.88→0.80 FMNIST). **Mean and spread corrections are never separated** (mean error also falls, 0.22→0.09 std) | No mean-only or spread-only arm | Small, real effect that interacts with PC. "Through the spread" is asserted, not isolated |
| **MC-B** | "MC (or MC-B)" in the final method (readout report §1) | E32 | X-EXP vs E25 | 3 | −0.06±1.47 / **+1.89±0.43 (p=0.016)** | Not separated | — | It is undefined which variant is "the method". Unstable on MNIST (seed 2 −1.73). K14 not run |
| **KH** | Lowers the MMD term of the head's risk bound (§4.5); "+ in all 9 paired tests" (§5.11) | K09, E20→E25, E22, D01 | LOO; X-EXP (the "9/9"); ADD at K = 100 (E22, same generator) | 3 | LOO −0.71±0.31 (p=0.059) / −0.30±0.17 (p=0.097). X-EXP +0.80 / +0.66. K = 100: +1.21±0.18 (p=0.007) | **Tested and contradicted.** MMD 0.088→0.071 / 0.093→0.078, but TSTR **−0.2 / −1.1** (D01); oracle effect −0.39 / −0.11 | **No** random 1-of-4 subset control and no "use all 4n candidates" control | Small, consistent end-to-end effect. The MMD rationale is not what drives it. Mechanism unknown |
| **CSL** | Consensus labels lower the bias of synthetic labels; Prop. 3 interaction (§4.3–4.4) | K09, K02, E16, D01 | LOO; ADD | 3 (K02: 2) | LOO −1.63±0.98 / −1.38±0.22; K02 minimal −2.37 / +0.28; oracle −1.40 / −1.43 | Weakly. TV to referee 0.100→0.092 / 0.212→0.202; Bayesian CSL fails | β ∈ {0.25, 0.5, 0.75, 1}; **no uniform label-smoothing control** | Robust, representational effect. Not an exact-statistics component |
| **LA** | Client-prior logit adjustment; the largest component (§4.2) | K09, K02, F01, D01 | LOO; ADD | 3 | −2.98±0.66 / −6.18±1.37. **With BBC: −0.26±0.14 / −0.53±0.04.** Oracle +0.06 / +0.32 | Yes: prediction-share correlation 0.64→0.44 (D01, 1 seed) | **BBC tested and recovers ~90–95% of it.** −LA+BBC vs Ours-A without BBC: −0.03 / +0.37 | Real, but a substitute for BBC. Both are in the final method, which double-counts the same correction |
| **BBC** | Gated post-hoc prior correction; fixes the bias LA leaves (§4.5) | K09, E25, K08, K10, K11 | ADD (logged on every run) | 3 | +0.22 / +0.90 (K09); CIFAR +2.65; SVHN rows +1.36 | Lemma R1: offsets cannot fix weight errors. BBC < oracle in every run | LA (substitute); BCR (weaker) | Small given LA. The gate (max/min > 2) was only ever evaluated at imbalance 1.0 or 48–102, so "any threshold 1–10 gives the same decisions" is untested |
| **Generator gate (F < 0.05)** | Rows when F < 0.05; "F is 0.01 on SVHN and ≥ 0.14 elsewhere; any threshold 0.02–0.1 gives the same choice" (§4.6, abstract) | K08, K10, K11, K13 | Chosen after the SVHN failure, on the same 4 datasets | 3 | SVHN rows vs anchor: +9.71±0.74 | Prop. 5 is consistent with SVHN | **Rows branch never run on CIFAR-10** (NB20/21 NOT RUN, marked "superseded: the gate selects the anchor there") | **Not validated.** CIFAR-10 feature-space F is **0.055–0.078**, not ≥ 0.14, so a threshold between 0.06 and 0.1 flips the decision |
| **T_s = 10** | Pays off once the generator is faithful (Prop. 3) | E16, E25, E22, K08 | ADD | 3 | PC: +1.70±0.62 / +1.46±0.16. **CVAE+HWA: −0.05 / +2.28±0.14.** CIFAR −1.67 (p=0.028); SVHN −1.47; K = 100 +3.89 | Prop. 3 only (stylised). On FMNIST the *less* faithful CVAE gains *more* | T_s ∈ {1, 3, 5, 10} | Dataset-dependent and chosen after the fact. No server rule selects it. The SVHN "final" number uses T_s = 1 |
| **ER** | Federated exact readout closes the readout gap (readout report, handout Part 6) | T03 | Single runs | **1 seed, schedule × 0.4** | Within 0.8 of the oracle in 9/9 runs; GeFL-F+ER 90.8 / 79.1 | Pre-registered P3 failed (Δ_c does not explain the gap) | **Not compared against Ours-A at full schedule** | Probably the strongest single "exact statistics" tool, yet not in Ours-A and not tested at 3 seeds |

---

## 3. Weak causal attributions

1. **"PC replaces HWA" contradicts "the mean anchor is the key fix."** In E20, replacing HWA with PC changes final accuracy by **+0.34 (MNIST, p = 0.53) and −0.65 (FMNIST, p = 0.14)**, while tail fidelity jumps 34→94 and 68→90. Oracle accuracy *drops* (−0.63 / −0.28).
   * PC is worth +1.20 / +0.37 over HWA only once MC is added.
   * The K09 LOO value (−4.72) is measured with MC, KH and T_s = 10 present.
   * So PC's effect is an interaction (PC × MC × T_s). Handout 7.1 ("why the anchor beats holder averaging: the class average is exact from the first round") is contradicted by the add-one data.
2. **Fidelity is not the operative variable.**
   * MC *lowers* referee tail fidelity yet raises accuracy.
   * PC raises fidelity by 60 points (MNIST) yet adds nothing on its own.
   * MMD (KH's target) falls, yet TSTR does not move.
   * The across-stage correlation between MMD and accuracy (r = −0.69 / −0.47, handout ev04) is confounded by stage. It is not evidence for the bound.
3. **KH's justification (the MMD bound) is falsified by the authors' own TSTR** (−0.2 / −1.1). The end-to-end effect (LOO −0.71 / −0.30, neither significant) has no tested mechanism and no random-selection control.
4. **MC "works through the spread"** (K09 analysis note 6, handout 7.2) is not isolated. MC also cuts the class-mean error 0.22→0.09 std (D01). A spread-only and a mean-only arm are needed.
5. **Prop. 3 "interaction" holds on MNIST only.**
   * On FMNIST, T_s = 10 helps the CVAE+HWA (+2.28, p = 0.001) more than the anchored generator (+1.46).
   * On CIFAR-10 and SVHN it hurts.
   * The paper reports the MNIST interaction (+1.75) as confirmation.
6. **The step table (handout 4.7) is a cross-experiment, order-dependent chain:**

   | Row | Source |
   |---|---|
   | GeFL-F | E20 |
   | + LA, + HWA | F01 |
   | + CSL, PC replaces HWA, + MC | E20 |
   | + KH, T_s = 10 | E25 |

   * The same seeds give GeFL-F 74.24–75.31 and HWA+LA+CSL 89.74–90.40 across experiments (F10, F01, E16, E18, E20, K02). That cross-run drift (~0.5–0.7) is as large as the PC, KH and MC steps.
   * The "+LA" step reads +3.67 in the table but +4.30 within F01.
   * The KH step pairs two separate generator trainings: raw tail fidelity differs per seed between E20 and E25.
   * "PC replaces HWA" is a swap, not an add-one step.
   * Only K09 is a consistent design, and it omits HWA, BBC, T_s, the gate and any 2-way interaction.
7. **The HWA mechanism is causal for row collapse, not for accuracy.** It is well shown at the row level (T01, but 1 seed). Its accuracy effect is ~85–90% readout (K02 oracle). Whether simply removing coupled decay (AdamW or λ = 0) gives the same accuracy was tested only in a quick pass on 1 seed, where it gave most of HWA's gain. K15 was not run.

---

## 4. Does the integrated method beat a simpler baseline?

| Simpler baseline | MNIST LT | FMNIST LT | Other | Evidence |
|---|---|---|---|---|
| GeFL-F | +19.6 (p=0.002) | +21.2 (p=0.011) | CIFAR +12.3; SVHN (rows) +15.1 | E20/E25, K08/K10 |
| Minimal variant (HWA+LA+CSL), same T_s = 10 | +4.66 (p=0.052) | **+0.45 (p=0.065)** | CIFAR +4.0 (T_s=1, Welch); SVHN rows+MC+KH vs HWA+LA +3.15 | E16 vs E25 |
| GeFL-F + readout re-fit (oracle, CCVR/CReFF/Fed3R upper bound) | +3.61 (p=0.028) | **+0.74 (p=0.23)** | SVHN +0.78 | E20 oracle, K01 |
| Minimal variant + readout re-fit (oracle) | +1.18 (p=0.11) | **−1.06 (p=0.016)** | — | E20 oracle |
| Federated ER on GeFL-F (T03, 1 seed, short schedule) | −2.0 vs Ours-A+ER | −1.2 vs Ours-A+ER | — | T03 |

* **MNIST:** yes, the integrated method beats every simpler baseline.
* **FMNIST:** no, not demonstrably. It ties the minimal variant at matched T_s, ties GeFL-F + recalibration, and loses to minimal + recalibration.
* **SVHN:** the "integrated method" is the old CVAE+HWA plus MC and KH, so the anchoring part is switched off.
* **CIFAR-10:** the anchor wins over minimal, but the rows + MC + KH alternative was never run there.

---

## 5. Is the unifying principle doing work?

The principle's predictions, tested against the repository's own data:

* **Purest exact-statistics instantiations underperform:**
  * FSG (exact Gaussian sufficient statistics) is 87.72 / 73.64, against 90.40 / 76.60 for the minimal variant (E20).
  * KME-Gen (no federated training) is 89.88 / 75.11 (1 seed), below the minimal variant.
  * MC-S is 93.25 against 94.82 on MNIST (E31).
* **On SVHN** the exact-mean anchor loses 9.7 points to FedAvg-learned (HWA-protected) rows. The final method's fallback is the *opposite* of the principle.
* **The two most robust components are not exact-statistics components:**
  * LA uses a client-local prior and is the largest component.
  * CSL uses consensus of FedAvg-trained heads. It is the only component whose gain survives the readout re-fit.
* **The one exact-statistics tool that clearly works on the residual (ER) is not part of Ours-A.** It would make most of the generator stack redundant on FMNIST.

**Verdict.** "Exact-Statistics Anchoring" is a post-hoc label for a set of tricks: some use exact sums (PC, MC, KH, gate), some don't (LA, CSL, T_s), and it is switched off on 1 of 4 datasets. The defensible framing is narrower: *a readout-bias diagnosis plus generator repairs that mostly act through readout bias.*

---

## 6. Evidence ledger

Severity: 1 fatal, 2 major, 3 moderate, 4 minor. Status: D = demonstrated, S = suspected, U = unresolved.

| ID | Claim + location | Evidence | Reasoning | Consequence | Sev | Conf | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| R8-1 | ESA anchoring of the generator explains the +20-point long-tail gain (abstract, §4.5, handout Summary) | `oracle_bal` in E20/E25/K02/K09: 81–88% of the gain disappears under a last-layer re-fit. GeFL-F oracle 78.94 vs Ours-A 79.67 FMNIST (p=0.23). Minimal oracle beats Ours-A by 1.06 (p=0.016). ER ≈ oracle (T03) | If generator fidelity drove accuracy, the representational (oracle) gain would be large | The causal headline is unsupported. Most of the gain is classifier-prior/readout correction | 1 (for the causal claim; the empirical gain stands) | High | D | Report final, oracle and ER side by side for every arm. Add GeFL-F+ER and minimal+ER, 3 seeds, full schedule. Reframe the contribution |
| R8-2 | Mean anchor is the key fix (§4.5, handout 7.1) | E20: PC replaces HWA +0.34 (p=0.53) / −0.65; oracle −0.63 / −0.28; fidelity +60 / +22 | A large fidelity gain with no accuracy gain on its own | PC's value is an interaction with MC and T_s; the stand-alone claim is contradicted | 2 | High | D | Factorial PC × MC × T_s on the K09 seeds; report the interaction terms |
| R8-3 | Step table "how each step adds up, same seeds" (handout 4.7, §5.11) | Rows come from E20, F01, E25. Same-seed baselines drift 0.5–1.1 across experiments. KH step pairs different generator trainings | Order-dependent cumulative chain across runs; steps under ~0.7 are not attributable | The step attribution is unreliable for PC, MC and KH | 2 | High | D | Regenerate the table from one experiment, or replace it with LOO plus Shapley-style averaging over orders |
| R8-4 | KH lowers the MMD term of the risk bound, hence helps (§4.5, handout 7.3) | D01: MMD falls 19%, TSTR −0.2 / −1.1. K09 LOO −0.71 / −0.30 (p > 0.05). "9/9" is cross-experiment | The proposed mechanism predicts a TSTR gain that does not occur | The mechanism is falsified; the effect is small and unexplained | 2 | High | D | Random 1-of-4 subset and all-4n controls; drop the MMD-bound justification or test it properly |
| R8-5 | MC works through the spread, not fidelity (K09 analysis, handout 7.2) | MC also corrects the mean (0.22→0.09 std). No mean-only or spread-only arm. FMNIST oracle effect +0.16 | Two simultaneous changes, one attribution | "Through the spread" is unproven | 3 | Med | U | Mean-only and scale-only MC arms (head-stage only, cheap: reuse the cached generators) |
| R8-6 | LA is the largest component (−3 / −6) and BBC adds on top (§4.2, §4.5) | K09: −LA with BBC −0.26 / −0.53; −LA+BBC vs Ours-A without BBC −0.03 / +0.37; LA oracle effect ~0 | LA and BBC are substitutes | The final method double-counts one correction; LA's marginal value is ~0.3–0.5 | 2 | High | D | Pick one headline metric (final or BBC); present LA and BBC as alternatives |
| R8-7 | F-gate: "≥ 0.14 on every other dataset; any threshold 0.02–0.1 gives the same choice" (abstract, §4.6) | CIFAR-10 feature-space F = 0.055–0.078 (K13). Rows branch never run on CIFAR (NB20/21 NOT RUN; PROGRESS calls them superseded because "the gate selects the anchor", which is circular). Threshold chosen after the SVHN result | False statement; decision unvalidated near the threshold; no held-out dataset | The gate is a post-hoc dataset switch | 2 | High | D | Run rows+MC+KH on CIFAR-10 (LT and IID); validate on a 5th dataset; correct the F values in the abstract and §4.6 |
| R8-8 | BBC gate: "any threshold between 1 and 10 gives the same decisions" (§4.5) | BBC gate observed only at global imbalance 1.0 or 48–102 | Vacuous; it equals "IF = 100 experiment or not" | Gate untested | 3 | High | D | Evaluate at IF = 10 and at label skew only (IF = 1, Dir 0.5) with Ours-A |
| R8-9 | Prop. 3: the budget helps once the generator is faithful (§4.4, §5.11) | FMNIST: CVAE+HWA gains +2.28 (p=0.001) from T_s = 10, more than PC (+1.46). CIFAR −1.67, SVHN −1.47 | The predicted interaction holds on MNIST only and is reversed on FMNIST | T_s is chosen per dataset after the fact; the "final method" config differs by dataset (SVHN rows at T_s = 1) | 2 | High | D | Report the T_s × generator interaction on all four datasets; state one T_s rule chosen in advance, or report both |
| R8-10 | Exact-statistics anchoring is the unifying principle (title, §4.5) | FSG, KME-Gen and MC-S (MNIST) underperform. Anchor loses 9.7 on SVHN. LA and CSL are not exact statistics. ER (exact statistics) is not in the method | The principle does not predict which components work | Post-hoc label | 2 | Med-High | D | Reframe, or add a test where the principle makes a prediction in advance (e.g., ER replacing BBC/LA) |
| R8-11 | HWA is needed rather than an optimiser fix (§4.1) | K15 not run. E02 quick, 1 seed: no-decay recovers ~85–90% of HWA's gain | Simplest alternative untested at full schedule | A cheaper fix may suffice | 3 | Med | U | Run K15 (already prepared) |
| R8-12 | Per-component necessity (§5.11, handout 5.6) | K09 n = 3: no component survives Holm; K14 not run. E20 `summary.md` says n = 2 but `runs.jsonl` has 3 (stale). E20 `bbc_bal` n = 2 | Underpowered; inconsistent records | MC and KH "necessary" is not established | 3 | High | D | Run K14 (seeds 3–4); regenerate the E20 summary |
| R8-13 | Which moment correction is the method (readout report §1: "MC (or MC-B)") | E32 MC-B +1.89 FMNIST (p=0.016, cross-experiment), unstable MNIST (seed 2 −1.73); not in K09 | Undefined method | Headline numbers may mix variants | 3 | High | D | Fix one variant; ablate it in K09/K14 |
| R8-14 | Draft lists K09 as "pending" (§5.11) and omits it; abstract F range | K09 is complete (LEDGER); CIFAR pixel-space F 0.067 appears in §4.5 itself | Internal inconsistency | Reader cannot audit the ablation | 4 | High | D | Add K09 (final + BBC + oracle columns) to the paper; fix the abstract |

**Positive.** K09's Ours-A (Kaggle) reproduces E25 (local) within 0.36 / 0.12 points. The head-arm design (same FE and generator per seed; identical raw fidelity across arms) is sound. HWA's row-level mechanism is tested with the right interventions (λ, aggregation).

---

## 7. Questions to the authors

1. Which metric is the headline, `final_bal` or `bbc_bal`? Please give the K09 ablation on that metric, and LA's marginal value given BBC.
2. What are GeFL-F + ER and minimal + ER at 3 seeds and full schedule, on all four datasets? If minimal + ER ≥ Ours-A on FMNIST and SVHN, what does the generator stack contribute beyond MNIST?
3. Why does PC replacing HWA leave accuracy unchanged and lower the oracle, despite a 60-point fidelity gain? Please run a factorial PC × MC × T_s.
4. What is KH's effect against a random 1-of-4 subset and against using all 4n candidates? If TSTR does not improve, by what route does KH help?
5. What do mean-only and spread-only MC give?
6. Why is CIFAR-10 F (0.055–0.078) reported as ≥ 0.14? What does the rows branch give on CIFAR-10, and how was the 0.05 threshold chosen?
7. Which T_s is "the method" on each dataset, and by what server-side rule?
8. Does removing coupled decay (AdamW or λ = 0) match HWA in accuracy (K15)?
9. MC or MC-B: which one is the method, and has MC-B been ablated?
10. How does each component change the per-class readout gap on *unheld* classes, where the readout report says the error sits?

*Recomputation scripts: `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r8_work/` (`k09.py`, `pair.py`, `d01.py`, `e16.py`, `summ.py`).*
