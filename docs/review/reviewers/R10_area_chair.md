# Reviewer 10 — Senior Area Chair report

**Specialty.** Federated learning (heterogeneous / long-tailed), generative-model-based knowledge transfer, and what a top-tier ML venue counts as a contribution (significance, originality, rigour, generality).

**Manuscript.** *Rare Classes Vanish from Federated Feature Generators — and How Exact Statistics Bring Them Back* (`docs/paper_draft.md`). Supporting material: `docs/PROGRESS.md`, `docs/research/readout_report.md`, `docs/paper_assets/{CHECK.md,tables/*}`, `experiments/results/K09_ablation_analysis.md`, `experiments/LEDGER.md`, and spot checks of `experiments/core.py` and `experiments/results/*/runs.jsonl`.

**Target venue.** The manuscript names no venue. It uses the format and claim style of NeurIPS/ICML/ICLR (theorem plus principle plus SOTA framing), so I judge it against that standard. The baseline (GeFL) was published in IEEE TMC, which is the natural fallback tier.

**Recommendation: Reject (for a top-tier ML venue). Confidence: 4/5.**
The empirical gains over the authors' GeFL-F are real and carefully paired. But the paper's central story, "rare classes vanish from the generator, and exact statistics bring them back", fails three independent tests:
1. **Diagnosis.** The authors' own measurements put most of GeFL-F's long-tail deficit in the classifier readout, not in the generator.
2. **Significance.** Against a known classifier-recalibration baseline, the generator-side advantage shrinks to under 1 point on two of the three datasets measured.
3. **Novelty and depth.** The principle and every component are known, the theorem is a triangle inequality, and the diagnosed mechanism is a hyperparameter artifact whose standard remedies are not tested.

---

## 1. What the paper actually contributes (stripped of framing)

1. **An empirical observation, on one generator of one method.** Under GeFL-F's CVAE-F with Adam and *coupled* L2 decay (λ = 1e-3, from GeFL Table XV), conditioning rows of tail classes shrink.
   - Measured on 1 seed in T01 (40 rounds, MNIST), with row-norm ratios over 3 seeds in F01.
   - Why it happens: non-holders' gradient on row c is λ·r_c, and Adam normalises this into a step of about η per coordinate.
   - At λ = 0 the contraction disappears (tail/head ratio 0.85; only 2 of 400 row-rounds contract).
2. **A minimal fix, HWA + LA + CSL.** Holder-weighted averaging of the class rows, logit adjustment by the client prior (FedLC-like, conceded as not new), and consensus soft labels on synthetic features. Gains are +13 to +17 balanced-accuracy points under IF = 100 (T02).
3. **A large final pipeline, Ours-A**, with a dataset-level gate:
   - PC-VAE (decoder conditioned on the exact class mean) when F ≥ 0.05, and CVAE + HWA rows otherwise;
   - MC moment correction (MC-S / MC-B variants), kernel herding toward random-feature (RFF) class embeddings, LA, CSL, gated BBC, and T_s = 10.
   - Gains of +19.6 / +21.2 under the long tail (T11), +12.3 on CIFAR-10 LT (T13), and +15.1 on SVHN LT through the rows branch (T14).
4. **Six small propositions:**
   - Prop. 3: a bias–variance parabola.
   - Prop. 4: Cauchy–Schwarz.
   - Prop. 5: a Lipschitz coupling bound.
   - Prop. 6: the Gelbrich bound, which is classical and conceded as such.
   - Prop. 7: a second-order MixUp expansion. The method built on it (VCT) failed (PROGRESS claim 17).
5. **Not yet in the draft:** exact readout (ER). This is a per-architecture uniform-prior LDA last layer fitted from secure-aggregated embedding sums. It sits within 0.8 points of a pooled-real oracle (T03: 1 seed, reduced schedule). The authors' own report calls it "the known recalibration principle extended to heterogeneous heads".

## 2. Dependency analysis (claim → depends on → status)

| # | Claim | Depends on | Status |
|---|---|---|---|
| C1 | "GeFL-F fails under long tails *because* rare classes vanish from the generator; every head is then taught from class-agnostic features" (abstract ¶1) | Collapse explains most of the accuracy loss | **Contradicted by own data.** GeFL-F's readout gap is 16.5 / 20.5 / 14.3 points (readout_report §2). A balanced last-layer refit on GeFL-F's *own* bodies reaches 90.7 / 79.2 / 64.8. The collapsed-generator bodies are therefore largely adequate. |
| C2 | Collapse is "structural, not a tuning issue" (§1 Gap) | Collapse persists under standard optimiser hygiene | **Contradicted / untested.** It disappears at λ = 0 (§3.2). AdamW, no-decay-on-embeddings and lazy decay are not run: T17/T18 rows 3–6 are empty and K15 is "ready, not run". DCGAN-F and DDPM-F use no decay, and their long-tail loss is "not yet measured" (§3.2). |
| C3 | Theorem 1 is a diagnostic or predictive result | Non-trivial content | **Near-tautology.** It is ‖mean‖ ≤ mean‖·‖, with θ_c defined from the measured Ḡ_c and D̄_c. The "0 violations in 1,200 row-rounds" check is conceded to only verify instrumentation. The predictive content is Lemma 2, which is an informal "of order η" argument and a known consequence of coupled L2 under Adam (Loshchilov & Hutter 2019). |
| C4 | Principle: "anchor every stage to exactly aggregated statistics" is new | Absence of prior anchoring methods | **Not new.** CCVR (NeurIPS'21): class mean/covariance → virtual features. Fed3R (ICML'24): closed-form classifier from sums. FedVTC (ICLR'26): mean/covariance-conditioned generator for heterogeneous clients. DP-MERF (AISTATS'21): generator trained only on RFF kernel mean embeddings, which is the same design as KME-Gen and is uncited. Also kernel herding (Welling 2009; Chen et al. 2010), Saerens EM prior correction (BBC), and Gelbrich maps (MC-S). |
| C5 | Ours-A is a significant improvement | Fair, strong baselines | **Insufficient.** The only baselines are GeFL-F, FedAvg/LG-FedAvg (+LA) and in-house variants. No FL long-tail method (CCVR, CReFF, FedLC, Fed-GraB, FedLoGe, CLIP2FL) and no other heterogeneous-FL method (FedGen, FedProto, FedDF, FedVTC) is run under the long tail. Against the GeFL-F + recalibration upper bound, the margin is +4.1 / +0.5 / +0.8 (MNIST / FMNIST / SVHN). CIFAR-10 is unknown. |
| C6 | Statistically reliable gains | Adequate n, multiplicity control, untouched test set | **Weak.** See the list below the table. |
| C7 | Theory applies to the implemented method | Theorem about class rows | **Mostly does not.** Ours-A's generator has no class rows on MNIST, FMNIST and CIFAR-10, so Theorem 1 only motivates removing them. The residual error is in the readout, which no proposition covers. The readout mechanism hypothesis H-R was falsified by the authors' pre-registration (P3 fail). |
| C8 | "Formal privacy is cheap" (abstract) | DP covering the method | **Overstated.** See the list below the table. |
| C9 | Communication is "C numbers per client per round for HWA, plus soft labels" (§5.12) | Ours-A's uploads | **Understated.** See the list below the table. |
| C10 | Generality; the F-gate is "not tuned" and any threshold 0.02–0.1 "makes the same choice everywhere" | Gate values on the datasets | **Demonstrably false as stated.** See E7. Generality otherwise rests on four 10-class small-image datasets, a frozen one-conv-block feature extractor (3×16×16), and K = 10 with G = 10, i.e. one client per architecture. |

**C6 detail (statistics):**
- n = 3 seeds with paired t-tests at df = 2.
- No multiplicity control across about 139 method labels in the ledger.
- The final composition was chosen per dataset on test-set results: T_s = 1 vs 10, the F-gate added after SVHN failed, and MC vs MC-S vs MC-B vs PCR vs rows.
- `best_mean_acc` is best-on-test (the authors concede this in PROGRESS §4).
- The CIFAR-10 LT tests are Welch tests computed from summaries.
- K09 per-component effects do not survive Holm correction.

**C8 detail (privacy):**
- The ε = 1 figure covers class counts of the *minimal* variant only.
- DP on Ours-A's statistics costs 4.5–6.9 points at ε = 8 and 9.8–14.1 at ε = 2 (T08).
- The generator's FedAvg training is non-DP in every method, so the system has no end-to-end guarantee.
- DP clipping uses an in-distribution, class-balanced "public" pool carved from the training set (`core.py` heldout, 200 per class).

**C9 detail (communication):**
- Ours-A adds a one-off upload per client of C × 6144 RFF sums (`federated_kme`, D = 6144) plus h-sums and h²-sums.
- MC-S adds C·k² ≈ 655k numbers.
- ER adds Σ_g(C·d_g + d_g²), up to 1.1M per architecture, and requires broadcasting every architecture's body to every client.

## 3. Independent vs cascading failures

**Independent grounds.** Each would stand even if the others were repaired.

- **(I-1) Significance against the right baseline.** This does not depend on whether the theory is valid. Suppose the theorem were deep and every component novel. A recalibrated GeFL-F would still nearly match the final head on FMNIST (+0.5) and SVHN (+0.8). That leaves a paper whose gain is largely a readout fix, which is published (CCVR/CReFF/Fed3R).
- **(I-2) Novelty and depth.** This does not depend on the numbers. The principle and components are known, and the theory is elementary or classical. MNIST +4 over recalibration would not by itself create a top-tier-level conceptual contribution.
- **(I-3) Generality and selection.** This does not depend on the other two. The design choices (gate, T_s, variant) were selected on the test sets of four 10-class datasets, and one of the gate's stated invariances is false. Even a novel and significant method would need held-out selection and at least one larger-class or realistic benchmark (CIFAR-100-LT, ImageNet-LT subsets, FEMNIST/iNat-fed).

**Cascades.**

- *C1 and C2 cascade into the HWA contribution.* If λ = 0, AdamW or no-decay-on-rows recovers HWA's accuracy, HWA reduces to "do not L2-regularise embeddings under Adam", which is standard practice. The minimal-fix contribution then shrinks to LA (conceded as not new) plus CSL (+0.4 to +2 points). This is unresolved because K15 has not been run.
- *C1 cascades into C7 and the narrative.* The paper's causal chain is collapse → class-agnostic features → bad heads. The readout analysis replaces it with "biased last layer on unheld classes, plus some body improvement". Generator anchoring then becomes one of several ways to shrink a readout gap that a closed-form readout closes directly. This undermines the motivation for the anchored-generator design, not just one number.
- *C3 does not cascade into the empirical results.* A vacuous theorem does not invalidate the measured gains. It removes the "theory" pillar, which at a top venue is half of the claimed contribution.
- *C10 (gate) cascades only into the "single method for all datasets" claim.* SVHN still benefits from the rows branch, and CIFAR-10 is near the threshold. With a threshold of 0.08 to 0.1, CIFAR-10 would switch branch, and that combination was never measured.

**Technical invalidity vs insufficiency.**

Statements that are false or unsupported:
- "structural, not a tuning issue";
- "any threshold between 0.02 and 0.1 makes the same choice everywhere" and "F … 0.14–0.60 on the other datasets";
- "every head is then taught from class-agnostic features" as the explanation of the failure;
- "Formal privacy is cheap", as phrased;
- §5.3 K = 50 numbers (86.8 / 87.2 / 87.9) quoted with no existing run (CHECK.md, T03).

None of these is a fatal mathematical error. The proofs I checked (Thm 1, Props 3–6) are correct. The rejection rests on **insufficient novelty, significance and evidence**, not on incorrect mathematics.

## 4. Meta-review (AC recommendation)

> The submission studies GeFL-F, a generator-based model-heterogeneous FL method, under long-tailed non-IID clients. It argues that (i) Adam's coupled weight decay combined with flat FedAvg contracts the conditioning rows of tail classes, and (ii) a pipeline that anchors the generator, its samples and the classifier bias to secure-aggregated class statistics repairs this. Paired 3-seed experiments on MNIST, FashionMNIST, SVHN and CIFAR-10 show large gains over GeFL-F (+12 to +21 balanced accuracy under IF = 100). The engineering is careful, the baseline reproduces the published GeFL-F, and the negative results are reported candidly.
>
> I recommend rejection on three grounds, each sufficient on its own.
>
> **1. The diagnosis does not account for the failure it is meant to explain, and the correct baseline is missing.** The authors' own diagnostics show that refitting only the last layer of GeFL-F's heads, class-balanced on pooled real features, recovers 14–20 of GeFL-F's 16–21-point long-tail deficit (90.7 / 79.2 / 64.8 on MNIST / FMNIST / SVHN). A federated closed-form readout from exact sums matches that oracle within 0.8 points (T03; one seed, reduced schedule). Classifier recalibration from federated feature statistics is established (CCVR, CReFF, Fed3R). Against GeFL-F + recalibration, the full method's advantage is +4.1 on MNIST and only +0.5 on FMNIST and +0.8 on SVHN. It is unknown on CIFAR-10, where Ours-A's own readout gap is 6.2. No external long-tailed-FL or heterogeneous-FL baseline is run under the long tail. The headline gains are therefore measured against an un-recalibrated baseline and overstate the contribution of the generator-side principle.
>
> **2. Insufficient originality and technical depth for this venue.** Theorem 1 is the triangle inequality applied to measured norm changes, with a threshold defined from the same measurements. The predictive content (Lemma 2) is an informal restatement of the known interaction between coupled L2 and Adam. The paper shows the effect vanishes at λ = 0, yet never evaluates the standard remedies (AdamW, no decay on embeddings). This contradicts its claim that the failure is structural rather than a tuning issue. The "exact-statistics anchoring" principle is already in prior work:
> - CCVR and Fed3R (classifier from federated sums);
> - FedVTC, ICLR 2026 (a generator driven by shared class means and covariances for heterogeneous clients);
> - DP-MERF, AISTATS 2021 (a generator trained only on RFF kernel mean embeddings, i.e. KME-Gen; uncited).
>
> The components are logit adjustment (conceded), ensemble soft labels, affine or Gelbrich moment matching (conceded classical), kernel herding, and Saerens-style prior correction. The remaining propositions are elementary. The paper is a well-executed composition of known parts tuned to one protocol.
>
> **3. Generality and selection.** All evidence comes from four 10-class small-image datasets, a frozen one-block feature extractor, and K = 10 with one client per architecture. The final pipeline was assembled per dataset from about 140 evaluated configurations using test-set results: T_s, the F-gate introduced after the SVHN failure, and the choice among MC, MC-S, MC-B, PCR and rows. Statistics use n = 3 (df = 2) with no multiplicity control, and the per-component ablation does not survive Holm correction. The gate's claimed robustness ("any threshold in 0.02–0.1 makes the same choice"; "F ≥ 0.14 on every other dataset") is contradicted by the logged values: CIFAR-10's F is 0.055–0.078 in the run files.
>
> The privacy and cost statements also overreach. "Formal privacy is cheap" refers to DP on class counts for the minimal variant. DP on the final method's statistics costs 10–14 points at ε = 2, and the generator's training carries no guarantee. The stated communication omits the RFF, second-moment and readout uploads.
>
> **What would change my assessment:**
> - (a) GeFL-F + CCVR/Fed3R/ER and at least two FL long-tail baselines adapted to heterogeneous heads, with the generator-side margin reported against them on all four datasets;
> - (b) K15 (AdamW / no-decay / lazy decay vs HWA);
> - (c) a held-out validation protocol for every design choice, with ≥ 5 seeds;
> - (d) one benchmark with many classes and a realistic tail;
> - (e) a reframing around the measured readout-vs-representation decomposition.

## 5. What survives if the weakest claims are removed

- **An empirical analysis paper on long-tailed generator-based heterogeneous FL.** Its parts:
  - the coupled-decay row-contraction observation, presented as a pitfall with its trivial remedy;
  - a measured decomposition showing that GeFL-F's long-tail deficit is mainly a readout gap on unheld classes, with a K-invariant readout gap and a representation gap that shrinks with clients per architecture;
  - evidence that offsets (LA/BBC) cannot reach that gap (Lemma R1 plus measurements) while a closed-form readout from sums can;
  - a smaller, real body-quality gain from faithful synthetic features: +2.4 to +4.2 oracle-vs-oracle over recalibrated GeFL-F.
- **Practical recipe:** GeFL-F + HWA (or no row decay) + LA + CSL + ER, with honest margins.
- **Suitable tier:**
  - a solid workshop paper (e.g. a NeurIPS/ICML FL workshop);
  - a mid-tier journal or conference (IEEE TMC/TNNLS, ECML-PKDD, AISTATS only with substantially stronger baselines);
  - a TMLR submission framed as an analysis or negative-results paper, where claims are judged on correctness rather than novelty.
- **Not** NeurIPS/ICML/ICLR main track in its current form.

## 6. Evidence ledger

Severity scale: 1 = fatal, 2 = major, 3 = moderate, 4 = minor.

| ID | Claim + location | Evidence | Reasoning | Consequence | Sev. | Conf. | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| E1 | Collapse explains GeFL-F's long-tail failure (Abstract ¶1; §1) | readout_report §2: GeFL-F head 74.2/58.7/50.5 vs oracle 90.7/79.2/64.8 (readout gap 16.5/20.5/14.3); repr. gap negative | Most of the deficit is fixed without touching the generator or bodies | Central causal narrative misattributed | 1 | High | Demonstrated (oracle, 3 seeds) | Reframe; report readout/representation decomposition as main finding |
| E2 | Ours-A significant over the state of the art (§5.11) | Ours-A head 94.8/79.7/65.6 vs GeFL-F + recal. bound 90.7/79.2/64.8; ER within 0.8 of oracle (T03, 1 seed) | Margin over a published-class baseline is +0.5 to +0.8 on two of three datasets | Significance collapses outside MNIST | 1 | High (bound) / Med (ER feasibility at full schedule) | Demonstrated (bound); unresolved for CIFAR-10 | Run GeFL-F + CCVR/Fed3R/ER, 3+ seeds, all 4 datasets |
| E3 | No external baselines under long tail (§5.3, §5.11) | Baselines: FedAvg, LG-FedAvg(+LA), GeFL-F, in-house FSG/hybrids | Top-venue SOTA claims need FL-LT and HtFL comparators | Comparative claims unsupported | 2 | High | Demonstrated | Add CCVR, CReFF, FedLC/Fed-GraB, FedGen/FedProto adapted |
| E4 | "Structural, not a tuning issue" (§1) | §3.2: λ = 0 → ratio 0.85, 2/400 contractions; T17/T18 AdamW/no-decay/lazy rows empty; K15 not run | Mechanism is coupled L2 under Adam; standard hygiene removes it | HWA's necessity unproven; diagnosis is a config pitfall | 2 | High | Demonstrated (row norms); accuracy unresolved | Run K15; report HWA vs AdamW / no-decay |
| E5 | Theorem 1 as diagnosis (§3.1) | Proof = triangle inequality; θ_c from measured Ḡ, D̄; "consistency check" conceded | Not predictive; Lemma 2 is informal ("of order η") | Theory pillar thin | 2 | High | Demonstrated | Formalise Lemma 2 (bounds on D̄_c via Adam state, S_k, η); predict θ_c ex ante |
| E6 | Principle and components novel (§1 contrib. 3; §4.5) | CCVR, Fed3R, FedVTC (ICLR'26), DP-MERF (AISTATS'21), kernel herding, Saerens 2002, Gelbrich | KME-Gen ≈ DP-MERF; PC-VAE + MC-S ≈ FedVTC-style; ER ≈ CCVR/Fed3R (conceded) | Originality limited to combination and setting | 2 | High | Demonstrated | Related work for every anchored component; precise delta |
| E7 | F-gate "not tuned"; "F ≥ 0.14 on every other dataset; any threshold 0.02–0.1 same choice" (Abstract; §4.6) | runs.jsonl `sep_F`: CIFAR-10 LT 0.066–0.078, IID 0.055–0.063; SVHN 0.005–0.007; §4.5 pixel CIFAR 0.067 | CIFAR-10 is within ~10–50% of the 0.05 threshold; thresholds ≥0.08 flip it | Gate robustness claim false; post-hoc gate | 2 | High | Demonstrated | Correct the claim; evaluate both branches on CIFAR-10; justify threshold a priori |
| E8 | Design choices made on test data (§4.6, §5.11) | ~139 method labels (LEDGER); T_s per dataset (CIFAR uses T_s = 1, MNIST/FMNIST T_s = 10); gate introduced after SVHN failure; best_mean_acc is best-on-test | Selection-induced optimism; no validation split | Inflated headline numbers; weak external validity | 2 | Med-High | Demonstrated (process); size of bias unresolved | Fix one configuration a priori; validation split; report final-round only |
| E9 | Statistics (§5.1; T15) | n = 3, df = 2; K09: no component survives Holm; CIFAR LT Welch from summaries | Small effects (MC 0.5–1.0, KH 0.3–0.7) not established individually | Component claims weak | 3 | High | Demonstrated | ≥5 seeds; Holm or hierarchical tests; restore K01 runs.jsonl |
| E10 | "Formal privacy is cheap: ε = 1 costs ≤1.3" (Abstract) | T07 (counts, minimal variant) vs T08 (Ours-A stats: −4.5–6.9 at ε=8, −9.8–14.1 at ε=2); generator FedAvg non-DP; clipping from in-distribution "public" pool (`core.py` heldout) | Headline applies to a different method and a subset of releases | Privacy claim misleading | 2 | High | Demonstrated | Rephrase; state the end-to-end guarantee (none); justify public pool |
| E11 | Communication cost (§5.12) | `federated_kme` D = 6144 per class; MC-S C·k² (k = 256); ER Σ_g(C d_g + d_g²), broadcast of all bodies | Stated overhead covers the minimal variant only | Cost understated for Ours-A/ER | 3 | High | Demonstrated | Table of uploads/disclosures per variant |
| E12 | Generality (§7) | 4 × 10-class datasets; FE 1 conv block; K=10, G=10 (1 client/arch); tail = 4 classes; CIFAR-10 accuracy 33–49 | Behaviour with many classes, larger FEs, multiple clients per architecture untested | Conclusions may not transfer | 2 | High | Demonstrated (scope) | CIFAR-100-LT or similar; FE ablation |
| E13 | Theory covers the method (§3, §4.5) | Ours-A has no class rows on 3/4 datasets; readout H-R falsified (P3 fail); Prop. 3 "confirmed" but T_s=10 hurts CIFAR/SVHN (explained post hoc) | Propositions are not guarantees for the implemented pipeline | No guarantee applies to headline results | 3 | High | Demonstrated | Restrict theory claims; state Prop. 3 predictions before runs |
| E14 | Draft numbers sourced (§5.3, §5.9, §5.8) | CHECK.md: 13 mismatches; K = 50 entries 86.8/87.2/87.9 have no run (T03 "Missing") | Reporting integrity | Must fix before any submission | 3 | High | Demonstrated | Regenerate all tables from runs.jsonl; delete unsourced rows |
| E15 | IID "above the best of all ten methods" (Abstract; §5.10) | Comparing own implementation with paper-reported numbers (own GeFL-F CIFAR is +3.2 above paper's); best-on-test metric; margins +0.04 to +1.2 | Cross-implementation, selection-biased comparison | IID superiority claim weak | 3 | Med-High | Demonstrated | Compare only within own implementation; final-round metric |
| E16 | Collapse evidence breadth (§3.2) | T01 = 1 seed, MNIST, 40 rounds; DCGAN-F/DDPM-F loss under LT unmeasured | Mechanism shown for one generator / one config | Narrow diagnosis | 3 | High | Demonstrated | More seeds; DDPM-F LT runs |

## 7. Questions to the authors

1. What is the balanced accuracy of GeFL-F + ER (or + CCVR / + Fed3R) at the full schedule on all four datasets, and Ours-A's margin over it? On CIFAR-10 especially, Ours-A's own readout gap is 6.2.
2. With AdamW, or simply no weight decay on the conditioning rows, does flat-FedAvg CVAE-F + LA match HWA + LA in accuracy at K = 10/50/100? If so, what remains of HWA's contribution?
3. How was each design choice selected without a validation set: T_s per dataset, F-threshold 0.05, MC vs MC-S vs MC-B, PC vs rows? What happens on CIFAR-10 with the rows branch, given F = 0.055–0.078?
4. How does PC-VAE + MC-S differ from FedVTC, which conditions a generator on shared class means and covariances for heterogeneous clients? How does KME-Gen differ from DP-MERF?
5. Can Lemma 2 be made a theorem that predicts D̄_c, and hence θ_c, from (η, S_k, Adam state reset policy, batch composition) *before* training?
6. Which end-to-end privacy guarantee does the deployed method (including generator FedAvg and the ER second moments) provide? Is an in-distribution balanced public pool realistic in the long-tailed setting it targets?
7. Do the conclusions hold with many classes (≥ 100), several clients per architecture, and a larger or non-frozen feature extractor?

*Sources consulted for the literature checks:* DP-MERF (https://proceedings.mlr.press/v130/harder21a.html), Fed3R (https://icml.cc/virtual/2024/poster/33597), FedVTC (https://iclr.cc/virtual/2026/poster/10006626), CLIP2FL (https://arxiv.org/pdf/2312.08648), GeFL (https://arxiv.org/abs/2412.18460).
