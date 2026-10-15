# Reviewer 1: Originality, Novelty and Contribution Audit

**Submission:** "Rare Classes Vanish from Federated Feature Generators — and How Exact Statistics Bring Them Back"
(GeFL-F under long-tailed clients; ESA / Ours-A). I read the manuscript in full (`docs/paper_draft.md`). I also read
`docs/paper_single_idea.md`, `docs/related_work.md`, `docs/research/readout_report.md`, `docs/PROGRESS.md`,
the K09 ablation analysis, tables T15/T17/T18 and the raw T03, F01 and E02 result files.

## 1. Specialty

My background is federated learning under heterogeneity: label skew, long tails, classifier calibration,
model-heterogeneous FL by distillation, prototypes and generators. I also work on data-free and generative knowledge
transfer and know the long-tail recognition literature. I can assess the optimisation argument (Adam, weight decay)
and the OT/kernel parts (Gelbrich map, herding, MMD) at a working level. I did not re-run any experiment. Every number
I quote comes from the repository files named below.

## 2. Overall recommendation

**Reject** at a NeurIPS/ICML/ICLR-level venue on originality and significance grounds. It could become a
**borderline/weak accept** at a specialised FL venue or workshop if it is repositioned as described in §7.
**Confidence: 4/5.**

The paper is careful and unusually honest. It records its negative results, pre-registers some tests and checks
its baseline against the authors' code. The problem is that almost every building block is an existing technique,
and in several places the manuscript's own positioning says so. The two elements that are truly new are both weak.

* **The collapse mechanism.** It is specific to one hyperparameter choice: L2 decay applied through Adam to a one-hot
  label embedding. The obvious one-line fix (no decay on that embedding, or AdamW) has never been compared with HWA.
* **The integrated ESA stack.** The authors' own exact-readout analysis shows that plain GeFL-F plus a known
  classifier recalibration recovers most of the headline gain.

## 3. Summary of the actual contribution

The contributions I find genuinely new:

1. **An empirical finding.** GeFL-F, evaluated only on IID clients in its source paper, degrades badly under a
   100:1 long tail with Dirichlet(0.5) clients. I checked that the GeFL arXiv v2 text contains no non-IID or Dirichlet
   experiments.
2. **A measured mechanism for one generator (CVAE-F).** GeFL's configuration uses coupled Adam weight decay. Under it,
   non-holders shrink class-conditioning rows at Adam's step size, and flat FedAvg then contracts tail rows. This is
   shown with instrumented row norms (one seed, T01) and a λ ∈ {0, 1e-4, 1e-3} contrast.
3. **An engineering integration.** Federated exact class statistics are plugged into the generator stage of a
   model-heterogeneous pipeline: a mean-conditioned VAE, moment/Gelbrich correction of samples, and RFF kernel herding.
   Consensus soft labels, logit adjustment and post-hoc prior correction sit on the heads.
4. **A diagnostic decomposition** (the readout report). The residual error splits into a readout gap and a
   representation gap.

Everything else consists of known techniques recombined, as the novelty matrix shows: HWA, LA, CSL, BBC, MC/MC-S,
KH, KME-Gen, ER and the gate.

## 4. Novelty matrix

Legend: **V** = I verified the primary source in this review (abstract or paper text). **A** = cited by the authors;
I did not re-verify it. Assessments run from "Not novel" through "Incremental" to "Novel".

| # | Component | Closest prior work (verified?) | Exact overlap | Genuine difference | Assessment |
|---|---|---|---|---|---|
| 1 | Rare-class conditioning collapse (Thm 1, Lemma 2) | **FedAwS**, Yu, Rawat, Menon, Kumar, ICML 2020 (V): with only positive labels, FedAvg class embeddings "can all converge to a single point". **FedRS**, Li & Zhan, KDD 2021 (A; method confirmed via MAP, arXiv 2404.09232, V): missing-class weights are pushed wrongly in local training. **FedLC**, ICML 2022 (V). **Collapse by Conditioning**, Shahbazi et al., ICLR 2022 (V): class conditioning causes mode collapse in limited-data cGANs. **CBDM**, Qin et al., CVPR 2023 (V): tail-class mode collapse in diffusion models. **Adam**, Kingma & Ba (V): the abstract states Adam "is invariant to diagonal rescaling of the gradients". **AdamW**, Loshchilov & Hutter, ICLR 2019 (A): L2 under Adam is not weight decay. Lazy sparse-embedding Adam variants (TF `lazy_adam`, MindSpore LazyAdam; V) skip updates to rows absent from the batch. | That class-specific parameters of under-represented or missing classes degrade under FedAvg is known for classifiers (FedAwS, FedRS, FedLC) and for centralised conditional generators (Shahbazi, CBDM). Lemma 2's λ-invariance follows in one line from Adam's documented scale invariance. Thm 1 is a triangle inequality; the authors call its verification a "consistency check" (draft:158–160). | I found no paper that identifies coupled L2 decay through Adam, combined with flat FedAvg, as the cause of collapse of a *generator's* conditioning rows. The measured θ_c > 1/2 refutes the naive K/2 rule in this setting. | **Incremental and narrow.** The mechanism is a configuration artefact. It disappears at λ = 0 (draft:167–169), does not apply to DCGAN-F/DDPM-F (draft:175–178), and the standard fix (no decay on embeddings, or AdamW) is untested (K15 not run; T17/T18 rows empty). |
| 2 | HWA: holder-weighted aggregation of class rows | **FedProto**, AAAI 2022 (V): "Given a class j, the server receives prototypes from a set of clients that have class j", weighted by \|D_{i,j}\|/N_j. **cwFedAvg**, Lee & Choi, ICCV 2025 (V): class-j model = Σ_i q_{i,j} w_i with q_{i,j} = n_{i,j}/Σ_i n_{i,j}. **FedNCF** (A): item embeddings averaged over the clients that updated them. Effective number from Cui et al. CVPR 2019 (A). | Holder-only, count-weighted averaging of class-specific quantities is FedProto's prototype rule and cwFedAvg's class weighting. HWA with raw counts (β→1) *is* the cwFedAvg weight applied to row c. The authors' own K02 ablation finds effective number, raw counts and uniform-over-holders within 0.4 points (draft:602–603), so the E(n) weighting adds nothing. | It is applied to a generator's conditioning rows rather than to prototypes or whole models. | **Not novel as a technique.** The application to a generator is a minor variation. |
| 3 | PC-VAE: generator conditioned on federated class means | **FedProto** (V): the count-weighted global prototype over holders *equals* the exact pooled class mean on a frozen FE. **f-CLSWGAN**, Xian et al., CVPR 2018 (V): generates CNN features from class-level vectors, so classes share one generator. **Delta-encoder**, Schwartz et al., NeurIPS 2018 (V): transferable intra-class "deltas" applied to class exemplars. **Feature transfer**, Yin et al., CVPR 2019 (V): transfers intra-class variance from regular to under-represented classes. **FedVTC**, Niu et al., ICLR 2026 (V): clients share feature mean and covariance statistics and train a generator from them for model-heterogeneous FL. **ProtoFedGAN** (V, abstract only): federated GAN aligned to class prototypes. | Conditioning a feature generator on a class-level vector instead of a learned one-hot embedding, and modelling the class as anchor plus shared residual, are standard in zero- and few-shot feature generation. Federated prototypes are FedProto. | A FedAvg-trained CVAE conditioned on the FedProto prototype inside GeFL-F. Prop. 5's separation bound for mean-anchored generators is a simple Lipschitz coupling argument. | **Incremental.** It combines two known ideas. The single-idea draft's claim "Prototypes have not been used to anchor a generator" (single_idea:409) is contradicted at least in spirit by FedVTC and ProtoFedGAN, and in mechanism by f-CLSWGAN and the Delta-encoder. |
| 4 | MC / MC-S / MC-B: moment correction of synthetic features | **CCVR**, Luo et al., NeurIPS 2021 (V, paper text): clients upload per-class µ_{c,k}, Σ_{c,k} and counts; the server forms the exact global µ_c, Σ_c and samples virtual features from N(µ_c, Σ_c). **Free Lunch / Distribution Calibration**, Yang, Liu, Xu, ICLR 2021 (V): calibrates few-sample class statistics by "transferring statistics from the classes with sufficient examples". **WCT**, Li et al., NeurIPS 2017 (V): whitening/colouring to match target feature covariance. **Gelbrich 1990; Olkin & Pukelsheim 1982** (A; cited by the authors). **DFL-FS**, Chen, Wu, Ji, arXiv 2403.08364 (V): long-tail FL regenerating features from global feature statistics. | The exact federated class moments are CCVR's. MC-S's tail-class shrinkage toward the pooled covariance is a distribution-calibration step in the spirit of Yang et al. The map is the classical Gaussian OT/Gelbrich map; the authors acknowledge this (draft:371–372). Prop. 4 (scalar case) is Cauchy–Schwarz. | The correction is applied to *a learned generator's samples* rather than replacing them with Gaussian samples. That is a reasonable but small step, and its value over CCVR-style Gaussian virtual features (FSG) is dataset-dependent (FSG fails on SVHN but wins at K = 100 on MNIST against CVAE + HWA). | **Not novel in components; marginal as a combination.** The effect is small in the ablation: −MC costs 0.96 (MNIST) and 0.49 (FMNIST) points, and FMNIST p = 0.10 (K09). |
| 5 | KH: kernel herding toward the federated class kernel mean embedding (RFF) | **Kernel herding**, Chen, Welling, Smola, UAI 2010 (V). **DM**, Zhao & Bilen, WACV 2023 (V): synthesises data by matching feature distributions in sampled embedding spaces. **FedDM**, Xiong et al., CVPR 2023 (V): distribution matching inside FL. **DP-MERF**, Harder, Adamczewski, Park, AISTATS 2021 (V, paper text): a label-conditioned generator G_θ(z, y) trained to match random-feature mean embeddings φ̂(x)yᵀ (per-class RFF embeddings via a one-hot label kernel), with the embedding privatised once and reused. | Herding to a mean embedding is textbook. "KME-Gen" (a generator trained only by MMD to exact class RFF embeddings, DP by post-processing, draft:448–455) **is DP-MERF** with the embedding summed across clients. DP-MERF is not cited anywhere in the manuscript. | Selection is done among generator candidates, per class, with federated targets. | **Not novel.** KME-Gen is a re-derivation of DP-MERF, and the readout report itself lists it as "rejected on evidence". KH's effect is small: −0.71 (p = 0.059) and −0.30 (p = 0.097) in K09. |
| 6 | CSL: consensus soft labels across heterogeneous heads | **FedMD**, Li & Wang 2019 (V): parties compute scores on public data, the server forms a consensus (average), parties "digest" it and then revisit their private data. **FedDF**, Lin et al., NeurIPS 2020 (V): ensemble distillation over heterogeneous models on unlabeled data or samples "generated from a pre-trained generator". **DS-FL**, Itahara et al., TMC (V): averaged outputs, entropy reduction. **FedGKD** (V): ensemble of historical global models. Hinton et al. CE + KD mixing; Menon et al. ICML 2021 bias–variance (A). | CSL is FedMD's consensus-then-digest protocol, with the public set replaced by generator samples (already used as distillation inputs by FedDF), plus the standard KD/one-hot mixing. The single-idea draft itself calls it "the cross-architecture distillation of [3, 4] applied to generated features" (single_idea:205–206), while paper_draft contribution 2 claims it. | Labels are per-sample on a shared synthetic pool and consumed client-side in GeFL-F's synthetic phase. | **Not novel.** The draft's Prop. 3 (bias–variance of the synthetic share) is a two-line stylised calculation. |
| 7 | LA: logit adjustment with each client's private prior | **FedRoD**, Chen & Chao, ICLR 2022 (V, paper text): the generic head is trained with balanced softmax using each client's own counts N_{m,y}^γ, which is exactly LA with the client prior. **FedLC**, ICML 2022 (V): per-client calibrated logits with pairwise label margins. **Menon et al.**, ICLR 2021 (A). FEDIC (A). | Identical in form to FedRoD's BSM loss; the same family as FedLC. | None, apart from the pipeline it is applied in. | **Not novel.** The authors partly concede this (draft:930–932). FedRoD, the exact precedent, is **not cited**. |
| 8 | BBC: balanced bias calibration (gated) | **Saerens, Latinne, Decaestecker**, Neural Computation 2002 (V): EM prior adjustment; the draft states BBC "is" its fixed point (draft:409). **CCVR** (V), **CReFF**, IJCAI 2022 (A), **Fed3R**, ICML 2024 (V): closed-form ridge classifier from federated statistics, "immune to statistical heterogeneity". **FedNCM**, Legate et al., NeurIPS 2023 (V): last layer from federated class means. | Post-hoc prior correction is Saerens. Recalibrating on synthetic or virtual features is CCVR/CReFF. The gate (max/min > 2) is a heuristic. | Calibration is done on anchored generator samples. | **Not novel.** K09 also shows that BBC makes LA largely redundant: without LA, bbc_bal falls only 0.26 (MNIST) and 0.53 (FMNIST). |
| 9 | Separation gate (F < 0.05 → CVAE rows + HWA) | The Fisher/LDA between/within ratio is classical. Prop. 5 is a Lipschitz coupling bound. | The gate is decided by one dataset. SVHN is the only F < 0.05 case among four, and the authors pre-registered the *opposite* prediction for CIFAR-10 and were wrong (draft:829–831). | Choosing the generator branch from released statistics is a sensible engineering rule. | **Novel but unvalidated.** It is a post-hoc rule fitted to one dataset. The claim "the threshold is not tuned" (draft:465) is not credible: the threshold was set after the SVHN failure was observed. |
| 10 | Integrated ESA stack and the "anchor every stage to exact statistics" principle | Exact federated sufficient statistics are the core idea of **CCVR** (V), **Fed3R** (V), **FedNCM** (V), **FedProto** (V), **FedVTC** (V), **DFL-FS** (V) and **DP-MERF** (V, centralised). Classifier recalibration from federated statistics covers the readout. | The "principle" is the shared premise of a whole line of FL work. The ESA instantiation also breaks it: the generator decoder and, on SVHN, the class rows are still FedAvg-estimated. | Combining these statistics at the generator, sample and readout stages of a model-heterogeneous generator pipeline. | **Incremental integration.** Its added value over a known recalibration of plain GeFL-F is +0.5 to +4 points (§6), not the +15 to +22 headlined. |
| 11 | ER: exact readout (per-architecture LDA last layer) | **Fed3R** (V), **CCVR** (V), **FedNCM** (V). | The authors concede "the principle … is known" (readout_report:140–145). | Per-architecture statistics: every client embeds with every architecture's body. | **Not novel.** It is an important *negative* finding for the paper's significance (§6). |

### Is the long-tailed GeFL-F setting itself new?

**Narrowly yes; broadly no.**

* **GeFL/GeFL-F.** The paper (Kang, Cha, Kang; arXiv 2412.18460; IEEE TMC 2025, verified) reports IID experiments only.
* **Generator-based FL under non-IID class imbalance.** This has been studied before:
  * FedFTG, CVPR 2022 (V), explicitly treats biased generators under long-tailed local data, with class-aware label
    sampling and teacher weighting.
  * FedGen, ICML 2021 (A), uses label-count-weighted teachers.
  * PFLDF (Pattern Recognition Letters 2024, found via search; abstract only, not fully verified) trains a server
    generator that produces balanced features for long-tailed personalised FL.
* **FL-LT methods that would be expected baselines** (each applicable per architecture group, since GeFL-F averages
  same-architecture heads, draft:102):
  * CCVR (NeurIPS 2021, V), CReFF (IJCAI 2022), FedLC (ICML 2022, V), FedRoD (ICLR 2022, V), FedETF (ICCV 2023, V),
    Fed-GraB (NeurIPS 2023), BalanceFL (IPSN 2022), FedLoGe (ICLR 2024), Fed3R (ICML 2024, V);
  * the model-heterogeneous baselines FedProto (V), FedTGP (AAAI 2024, V), FedMD/FedDF (V) and FedGen;
  * CLIP2FL (AAAI 2024, V) relies on CLIP, so it is reasonably excluded on a small FE.
* **What the main long-tail table (F01) actually compares.** Only FedAvg per architecture, LG-FedAvg (+LA), GeFL-F
  (+LA) and the authors' own variants (`experiments/results/F01_main_longtail/runs.jsonl`).
* **A new benchmark exists.** FedLTLib (arXiv 2609.15625, Sept 2026, V) now provides 9 FL-LT baselines. Reviewers
  will expect them.

## 5. Evidence ledger

Severity is graded 1 = fatal, 2 = major, 3 = moderate, 4 = minor.

| ID | Exact manuscript claim (file:line) | Independent evidence | Reasoning | Consequence | Sev. | Conf. | Status | Minimum action to resolve |
|---|---|---|---|---|---|---|---|---|
| E1 | "Classifier recalibration… must" be compared (readout_report:54–57); GeFL-F + recalibration "reaches 90.7 / 79.2 / 64.8 … within +4.1 / +0.5 / +0.8 of the final method's head" (readout_report:233–237). The paper_draft headline is "+19.6 / +21.2 over GeFL-F" (draft:759). | F01 runs.jsonl: GeFL-F `oracle_bal` 0.91 (MNIST), 0.79 (FMNIST), 3 seeds. T03 runs.jsonl (1 seed, reduced schedule): GeFL-F `er_lda_bal` = 0.9075 / 0.7912, a *federated* closed-form readout, while Ours-A (T_s = 1) `head_bal` = 0.9107 / 0.7566. Known methods: Fed3R, CCVR, FedNCM (V). | A one-round, known recalibration of *unmodified* GeFL-F recovers 80–97% of ESA's long-tail gain. At matched (reduced) schedule, GeFL-F + ER **beats** Ours-A (T_s = 1) on FMNIST by 3.5 points and ties on MNIST. The full Ours-A (T_s = 10 + BBC) keeps only +4.5 / +1.3 / ≈+2 over recalibrated GeFL-F. | The headline significance (+15 to +22) is measured against a baseline the authors themselves call incomplete. The incremental value of the generator-side novelty is small on FMNIST and SVHN. | **1–2** | High | Demonstrated (oracle 3 seeds; ER 1 seed) | Add GeFL-F + CCVR/Fed3R/ER (and + LA + ER) as baselines, 3+ seeds, all four datasets including CIFAR-10. Restate every gain against it. |
| E2 | "GeFL-F fails… for a reason that can be stated exactly… whatever the value of λ > 0" (draft:12–17, 133–139); "A minimal fix… HWA… already gives +12 to +20" (draft:47–48). | Draft:167–169: at λ = 0 the ratio stays 0.85 and only 2/400 rows contract. E02 quick run (`E02_conditioning_collapse_quick/summary.md`): **NOWD alone** restores the tail/head row ratio to 0.90 (FMNIST) and 0.83 (MNIST), against 1.00 and 0.71 for HWA, with final_bal 62.55 vs 63.43 and 69.87 vs 70.70. T17/T18: AdamW, no-decay-on-rows and lazy-decay rows are all "–" (K15 never run; PROGRESS:67 calls it "the question a reviewer asks first"). | The collapse is caused by one optimiser setting. Practitioners routinely exclude embeddings from decay or use AdamW. Lemma 2 is Adam's documented scale invariance (Kingma & Ba abstract, V). | The diagnosis contribution reduces to "do not apply L2 decay through Adam to a sparse one-hot embedding". The case for HWA as a *method* is unproven against the trivial fix. | **2** | High | Demonstrated (quick pass, 1 seed) / unresolved at full schedule | Run K15. If no-decay-on-rows ≈ HWA, reframe HWA as one of several equivalent fixes and drop "structural failure, not a tuning issue" (draft:65). |
| E3 | "This is a structural failure, not a tuning issue: it worsens as the number of clients grows" (draft:65–66). | Draft:175–186: DCGAN-F and DDPM-F have no weight decay, so there is no contraction, only dilution, and their "accuracy loss… under the long tail is not yet measured". | The title ("Rare Classes Vanish from Federated Feature Generators") generalises from one generator and one hyperparameter. | The claim does not extend to the paper's other generators. | **2** | High | Demonstrated (from the manuscript) | Narrow the title and abstract to CVAE-F with coupled decay, or measure DCGAN-F/DDPM-F under the long tail. |
| E4 | "The diagnosis suggests a principle: anchor every stage… to exactly aggregated statistics" (draft:19–21); "Our single idea… class-level information should… be anchored to statistics that secure aggregation computes exactly" (single_idea:16–18). | CCVR (exact global µ_c, Σ_c from client stats, V), Fed3R ("immune to statistical heterogeneity", V), FedNCM (V), FedProto (V), FedVTC (V), DFL-FS (V), DP-MERF (V). | Partition-invariant sufficient statistics are the founding premise of an established line of FL work. ESA also does not follow the principle where it matters: the decoder is FedAvg-trained, and on SVHN class identity comes from FedAvg rows. | The "principle" cannot be the paper's novelty claim. | **2** | High | Demonstrated | Present ESA as applying the CCVR/Fed3R/FedProto principle to the generator stage, and cite them in the abstract-level positioning. |
| E5 | "Prototypes have not been used to *anchor a generator*, nor have second moments, kernel embeddings or the displacement-optimal correction" (single_idea:409–410). | DP-MERF: G_θ(z, y) trained to label-conditioned RFF mean embeddings, DP by post-processing (V, paper text). FedVTC: mean + covariance → generator, model-heterogeneous FL (V). f-CLSWGAN (V) and Delta-encoder (V) for class-vector conditioning and residual synthesis. WCT (V) for the covariance-matching transform. | The claim is false for kernel embeddings (DP-MERF) and arguably for second moments (FedVTC). | It is an over-claim, and DP-MERF is an uncited direct precedent for KME-Gen (draft:448–455, listed as contribution 3 at draft:82). | **2** | High | Demonstrated | Cite DP-MERF, FedVTC, f-CLSWGAN, Delta-encoder and WCT. Remove KME-Gen from the contributions; the readout report already marks it rejected. |
| E6 | Contribution 2: "A minimal fix… HWA, LA and CSL. Each fixes one measured failure" (draft:75–77); CSL with Prop. 3 presented as a method (draft:238–283). | FedRoD BSM with client counts N_{m,y}^γ (V) = LA. FedMD consensus/digest (V) and FedDF on generator samples (V) = CSL. FedProto holder-only count-weighted aggregation (V) and cwFedAvg q_{i,j} = n_{i,j}/Σn_{i,j} (V) = HWA with raw counts. The handout says "Logit adjustment, consensus labels and holder-weighted averaging are adopted, not claimed" (build_lean.js:433), and single_idea:202 calls the head components "standard". | All three are known. The documents contradict each other on whether they are claimed. | The "minimal fix" contribution is a recombination of known tricks. | **2** | High | Demonstrated | Move HWA/LA/CSL to "adopted components" and cite FedRoD, FedMD, cwFedAvg and FedProto explicitly. |
| E7 | "On CIFAR-10 it rises from 33.7 to 46.1, or about 49 with the calibration, ahead of every other method" (draft:37). | F01/K-series baselines contain no FL-LT method (CCVR, CReFF, FedLC, FedRoD, FedETF, Fed-GraB, BalanceFL, Fed3R) and no model-heterogeneous method (FedProto, FedTGP, FedGen, FedMD/FedDF). The CIFAR-10 tests are Welch from printed means (draft:821–823; PROGRESS:40). The ER/oracle readout gap on CIFAR-10 LT is 6.2, the largest (readout_report:40). | "Every other method" means the authors' internal variants plus FedAvg and LG-FedAvg. | It is an over-claim, and the expected baselines are missing. | **2** | High | Demonstrated | Add the FL-LT and model-heterogeneous baselines above, or replace "every other method" with "every variant we tested". |
| E8 | Gate: "The threshold is not tuned: F is 0.01 on SVHN and at least 0.14 on every other dataset" (draft:465–466). | One dataset per branch. The registered prediction for CIFAR-10 failed (draft:829–831). The authors concede that "a dataset with intermediate separation would need both branches evaluated" (draft:998–999). | A one-dataset decision rule is post-hoc model selection. | The SVHN result (65.6) is the CVAE + HWA branch, i.e. the "minimal fix" plus MC + KH, not the anchored generator. | **3** | High | Demonstrated | Validate on held-out datasets (e.g. CIFAR-100-LT, Tiny-ImageNet-LT, a non-image dataset) chosen before running. |
| E9 | "Each step has a guarantee: the collapse cannot occur, the corrections are displacement-optimal, and the herding lowers the term of each head's risk bound" (single_idea:27–28); KH "lowers exactly the term of the head's risk bound" (draft:402–404). | Lemma 1 needs the loss in an RKHS ball, which a CE loss on neural heads does not satisfy. KH targets an RFF approximation. Props 4 and 6 are classical (authors' own citation). K09: −KH −0.71 / −0.30 points, p = 0.06 / 0.10, not Holm-significant. | The theory is decorative rather than predictive, and the effects are small. | The claim of "theory for each anchor" is overstated. | **3** | Med-High | Demonstrated | Present the propositions as motivation and remove "guarantee" language. |
| E10 | "None of these methods trains a class-conditional generator by FedAvg, so none encounters the row collapse we analyse" (draft:933–934, about the FL-LT literature); the question "the distillation and partial-sharing literature has not asked of a shared generator" (draft:896). | FedCG (V): "shares clients' generators with the server for aggregating" (conditional GANs). FedFTG (V) explicitly studies generator degradation under long-tailed client data. | The literature has asked how class imbalance affects shared generators, though not about the decay mechanism. | Positioning is too strong. | **3** | Med | Demonstrated (FedFTG, FedCG abstracts) | Soften the claim and discuss FedFTG's class-imbalance treatment as the closest generator-side precedent. |
| E11 | Abstract IID claims: "+1.2, +1.6 and +0.8 over the best of all ten methods in the paper" (draft:48–50). | The comparison is against *published* numbers on different seeds. `best_mean_acc` is a best-over-rounds-on-test metric; PROGRESS:96 lists "best-on-test selection biases best_mean_acc" as a known audit point. Our GeFL-F on CIFAR-10 is 3.2 above the paper's own number (draft:50). | The margins are within the reproduction gap and the selection bias of the metric. | It is not evidence of a contribution in the IID regime. | **3** | Med-High | Demonstrated | Report final-round accuracy, re-run the strongest paper methods under the same seeds, or drop the claim. |
| E12 | "Formal privacy is cheap: ε = 1 on every client's class counts costs at most 1.3 points" (draft:51–52). | Draft:614–632: DP on *all* ESA statistics costs 4.5–6.9 points at ε = 8 and 9.8–14.1 at ε = 2. The generator's FedAvg is not DP. | The abstract quotes the cheapest sub-case. | It is misleading, though not a novelty issue as such. | **4** | High | Demonstrated | Quote the full-statistics DP cost in the abstract. |
| E13 | Ablation-based significance of the novel components (draft:772, "KH… helps in all 9 paired comparisons"). | K09: per-component p-values do not survive Holm (smallest adjusted p = 0.085); MC and KH are worth 0.3–1.0 points; PC is −0.67 on FMNIST (p = 0.106). Ours-A-B (MC-B) is unstable on MNIST (94.76 ± 1.95). | The novel generator-side components each contribute about a point. Most of the gain comes from LA (FedRoD/FedLC family) and from removing the collapse (fixable by optimiser configuration). | The contribution is concentrated in non-novel parts. | **2** | High | Demonstrated | Run K14 (5 seeds). Report the share of the gain attributable to the novel components against GeFL-F + LA + ER. |
| E14 | Related work omits direct precedents. | Not cited anywhere in paper_draft.md: FedRoD, DP-MERF, FedVTC as a method competitor (it is cited only descriptively), Fed3R, FedNCM, FedETF, CLIP2FL, FedTGP, cwFedAvg-as-HWA precedent (cited for classifier layers only), Free-Lunch distribution calibration, f-CLSWGAN/Delta-encoder, kernel herding/DM/FedDM (in paper_draft; single_idea cites Chen et al. only). | These are standard citations a reviewer will check. | Any of them could trigger a novelty rejection on its own. | **3** | High | Demonstrated (by grep of draft references) | Add them with explicit differences. |

## 6. The "exact readout" result and significance (Task 4)

The result substantially undermines the significance claim, though it does not erase the contribution.

**Numbers** (all from the repository):

| | MNIST LT | FMNIST LT | SVHN LT |
|---|---|---|---|
| GeFL-F head | 74.2 | 58.7 | 50.5 |
| GeFL-F, last layer re-fit on pooled real data (oracle, 3 seeds) | 90.7 | 79.2 | 64.8 |
| GeFL-F + federated ER-LDA (T03, 1 seed, reduced schedule) | 90.8 | 79.1 | – |
| Ours-A head (T_s = 10; SVHN = rows branch) | 94.8 | 79.7 | 65.6 |
| Ours-A + gated BBC | 95.2 | 80.6 | ≈ 67 |
| Ours-A oracle | 94.9 | 81.6 | 68.4 |

**What follows.**

1. **The headline is mostly a readout gain.** About 16.5 of GeFL-F's MNIST deficit and 20.5 of its FMNIST deficit
   sit in the last layer. A known, one-round, closed-form recalibration (Fed3R/CCVR/FedNCM principle; T03 shows the
   federated version matches the oracle within 0.8 in 9/9 runs) recovers them. ESA's margin over this baseline is
   +4.1 / +0.5 / +0.8 on the head, or +4.5 / +1.4 / ≈ +2 with BBC. On FMNIST and SVHN that margin is within about
   1–2 sd of seed noise (FMNIST Ours-A sd 0.31; GeFL-F sd 3.4).
2. **At matched schedule the baseline wins on FMNIST.** In T03 (same seed, same reduced schedule), GeFL-F + ER = 79.1
   while Ours-A (T_s = 1) = 75.7 on FMNIST, and 90.8 vs 91.1 on MNIST. Part of ESA's full-schedule advantage comes
   from T_s = 10 and BBC, which are not themselves novel.
3. **The fair comparison favours ESA only modestly.** With ER on both sides, ESA leads by +4.2 / +2.4 / +3.6. This is
   the defensible number for the generator-side contribution: a few points, with MNIST the only clear case.
4. **Caveats in the authors' favour.**
   * ER needs every client to run every architecture's body and to upload d_g² second moments per architecture
     (readout_report:111–113). That is more communication and disclosure than ESA's statistics.
   * The oracle is non-federated. Only T03 (1 seed, reduced schedule) shows that ER reaches it.
   * CIFAR-10 has the largest readout gap (6.2), and the recalibrated-GeFL-F number there is unknown.

   These caveats weaken the attack somewhat. They do not remove it, because CCVR-style per-architecture
   recalibration with per-class mean and covariance is standard and cheap at these head sizes.
5. **The draft does not report this.** paper_draft.md never mentions it. The headline abstract numbers compare
   against un-recalibrated GeFL-F. A top-tier reviewer who knows CCVR or Fed3R will ask for exactly this baseline, and
   the authors already hold evidence that it closes most of the gap.

## 7. Strongest rejection arguments

1. **Recombination of known parts.** Every component except the coupled-decay diagnosis has a direct published
   precedent:

   | Component | Precedent |
   |---|---|
   | HWA | FedProto / cwFedAvg |
   | LA | FedRoD / FedLC |
   | CSL | FedMD / FedDF |
   | BBC | Saerens / CCVR |
   | MC / MC-S | CCVR + Gelbrich / WCT + distribution calibration |
   | KH | kernel herding / DM |
   | KME-Gen | DP-MERF |
   | ER | Fed3R / CCVR / FedNCM |
   | PC-VAE | FedProto prototype + class-vector-conditioned feature generators |

   The overarching "exact statistics" principle is the premise of CCVR, Fed3R and FedProto. The paper's internal
   documents concede most of this (build_lean.js:433; readout_report:140–145; single_idea:202, 411–413), but the
   manuscript's contribution list still claims it.
2. **The novel diagnosis is narrow, and its fix is trivial.** It is a configuration artefact (coupled L2 through Adam
   on a one-hot embedding). It is absent at λ = 0 and absent in the paper's other two generators. The obvious fixes
   (AdamW or no decay on the embedding; E02 quick pass suggests NOWD ≈ HWA) were never compared.
3. **Significance collapses against the right baseline.** Known last-layer recalibration of plain GeFL-F recovers
   80–97% of the headline gain (E1). The novel generator-side components each add about 0.3–1.0 points with
   non-significant per-component p-values (K09).
4. **Thin evaluation for a top venue.** Four small datasets on a one-block frozen FE. No FL-LT or
   model-heterogeneous baselines under the long tail. A one-dataset gate. Per-seed CIFAR-10 data are missing.

**What would strengthen these arguments:**
* the K15 result showing no-decay-on-rows ≈ HWA;
* GeFL-F + LA + ER within about 1 point of Ours-A + BBC on CIFAR-10;
* a CCVR-per-architecture baseline matching ER.

**What would weaken them:**
* K15 showing HWA ≫ AdamW/no-decay at K = 50/100. This is plausible, because dilution still slows rare rows
  (draft:177–181).
* Ours-A + ER beating GeFL-F + ER by more than 3 points on CIFAR-10 and SVHN over 5 seeds.
* Gains holding at scale (CIFAR-100-LT, more architectures, a stronger FE), where generator fidelity matters more
  and readout recalibration less.

### Narrowest defensible claim

"In GeFL-F's released configuration, the CVAE feature generator's class-conditioning rows collapse under label skew,
because non-holders apply coupled Adam decay that flat FedAvg does not cancel. Removing the non-holder term fixes it.
Replacing the learned class rows with federated class means (FedProto-style), moment-corrected and herded samples, adds
a further +2 to +4 points (MNIST/FMNIST/SVHN, long tail) over GeFL-F with a federated last-layer recalibration. That
recalibration alone recovers most of GeFL-F's long-tail loss."

### Broadest claim the evidence does not justify

* "Rare classes vanish from federated feature generators" in general; "a structural failure, not a tuning issue".
* A new *principle* of "anchoring every stage to exact statistics".
* "+15 to +22 points over GeFL-F" as the value of the method; "ahead of every other method" on CIFAR-10.
* "Each step has a guarantee".
* "Prototypes, second moments, kernel embeddings have not been used to anchor a generator".
* Superiority over all ten GeFL variants in IID (best-on-test metric, published numbers, different seeds).

## 8. Questions for the authors

1. What do AdamW (λ = 1e-3 and 1e-2), no decay on the conditioning embedding, and lazy decay give against HWA at
   K = 10/50/100, with 3+ seeds? (K15 is prepared but has not been run.)
2. Please report GeFL-F + LA + {CCVR per architecture, Fed3R, ER} on all four datasets, with 3+ seeds and the same
   splits. What is Ours-A's margin over the best of these?
3. How does KME-Gen differ from DP-MERF (Harder et al., AISTATS 2021) with a federated sum of the embedding? How does
   LA differ from FedRoD's balanced-softmax generic head with client counts? How does CSL differ from FedMD's consensus
   step with FedDF's generator inputs?
4. HWA with raw counts equals cwFedAvg's class weights applied to one row, and FedProto's holder-only rule applied to
   parameters. What, then, is claimed as new in HWA?
5. Why should the F-gate threshold be trusted when only one dataset lies on one side of it? Can you add held-out
   datasets chosen before running?
6. Do DCGAN-F and DDPM-F (no weight decay) lose rare classes under the long tail? By how much, and does HWA help them?
7. Why are FedLC, FedRoD, CCVR, CReFF, FedETF, Fed-GraB, FedProto/FedTGP and FedGen/FedFTG missing from the long-tail
   baselines? Several apply directly per architecture group.
8. Can the IID comparison use final-round accuracy and re-run the competing GeFL variants on your seeds?

## Sources consulted (primary pages read in this review)

* GeFL: https://arxiv.org/abs/2412.18460 (PDF text checked for non-IID)
* FedAwS: https://icml.cc/virtual/2020/poster/6592
* Collapse by Conditioning: https://arxiv.org/abs/2201.06578
* CBDM: https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Class-Balancing_Diffusion_Models_CVPR_2023_paper.pdf
* Adam: https://arxiv.org/abs/1412.6980
* MAP (FedRS description): https://arxiv.org/abs/2404.09232
* FedLC: https://proceedings.mlr.press/v162/zhang22p.html (PDF text)
* FedRoD: https://arxiv.org/abs/2107.00778 (PDF text, Eq. 6 BSM with N_{m,y}^γ)
* FedProto: https://arxiv.org/abs/2105.00243 (PDF text, holder-only aggregation)
* cwFedAvg: https://arxiv.org/html/2406.07800 (Eqs. 5–7)
* CCVR: https://arxiv.org/abs/2106.05001 (PDF text, global µ_c, Σ_c from client statistics)
* Free Lunch / Distribution Calibration: https://arxiv.org/abs/2101.06395
* WCT: https://proceedings.neurips.cc/paper_files/paper/2017/file/49182f81e6a13cf5eaa496d51fea6406-Paper.pdf
* Kernel herding: https://proceedings.mlr.press/r8/chen10a.html
* DM: https://openaccess.thecvf.com/content/WACV2023/html/Zhao_Dataset_Condensation_With_Distribution_Matching_WACV_2023_paper
* FedDM: https://arxiv.org/abs/2207.09653
* DP-MERF: https://proceedings.mlr.press/v130/harder21a.html (PDF text: G_θ(z, y), label kernel)
* FedMD: https://arxiv.org/abs/1910.03581
* FedDF: https://arxiv.org/abs/2006.07242 (PDF text: generator samples, heterogeneous models)
* DS-FL: https://arxiv.org/abs/2008.06180
* FedGKD: https://arxiv.org/abs/2107.00051
* Saerens et al.: https://mlanthology.org/neco/2002/saerens2002neco-adjusting
* Fed3R: https://proceedings.mlr.press/v235/fani-24a.html
* FedNCM: https://proceedings.neurips.cc/paper_files/paper/2023/hash/dcc0ac74ac8b95dc1939804acce0317d-Abstract.html
* FedVTC: https://arxiv.org/abs/2508.01669
* DFL-FS: https://arxiv.org/abs/2403.08364
* FedTGP: https://ojs.aaai.org/index.php/AAAI/article/view/29617
* FedPCL: https://proceedings.neurips.cc/paper_files/paper/2022/hash/7aa320d2b4b8f6400b18f6f77b6c1535-Abstract.html
* CLIP2FL: https://ojs.aaai.org/index.php/AAAI/article/view/29416
* FedETF: https://openaccess.thecvf.com/content/ICCV2023/papers/Li_No_Fear_of_Classifier_Biases_Neural_Collapse_Inspired_Federated_Learning_ICCV_2023_paper.pdf
* FedLTLib: https://arxiv.org/abs/2609.15625
* f-CLSWGAN: https://openaccess.thecvf.com/content_cvpr_2018/html/Xian_Feature_Generating_Networks_CVPR_2018_paper.html
* Delta-encoder: https://papers.nips.cc/paper/2018/hash/1714726c817af50457d810aae9d27a2e-Abstract.html
* Yin et al. (feature transfer): https://openaccess.thecvf.com/content_CVPR_2019/html/Yin_Feature_Transfer_Learning_for_Face_Recognition_With_Under-Represented_Data_CVPR_2019_paper.html
* FedCG: https://arxiv.org/abs/2111.08211
* FedFTG: https://arxiv.org/abs/2203.09249
* ProtoFedGAN (abstract only): https://sciencedirect.com/science/article/abs/pii/S0167739X25006478
* LazyAdam docs: https://tensorflow.google.cn/versions/r2.3/api_docs/python/tf/tpu/experimental/embedding/Adam

**Not verified by me** (cited by the authors; metadata taken on trust): AdamW, Menon et al. logit adjustment, CReFF,
Fed-GraB, BalanceFL, FedLoGe, FedGen, Gelbrich 1990, Olkin & Pukelsheim 1982, FedNCF. PFLDF was seen only as a search
snippet.
