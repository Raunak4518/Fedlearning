# Referee report (reviewer G, cycle 2): "FedECS-LT: Exact Class Statistics for Long-Tailed Federated Learning with Heterogeneous Client Models"

Submission build reviewed: `docs/papers/elsevier/main_final.pdf` (54 pages), plus `highlights.tex`. I checked a few doubtful numbers against `docs/papers/generated/tables/*.csv` and `fedecs_lt/results/*/runs.jsonl`. Page numbers below are the PDF's printed page numbers.

## 1. Summary

The paper studies generator-based model-heterogeneous FL in the form of GeFL-F. Clients share a frozen extractor trained by FedAvg and a FedAvg-trained conditional feature generator, and each architecture trains its own head. The setting is long-tailed, Dirichlet-skewed label distributions. The authors name three places where rare classes are lost. First, Adam with coupled L2 decay shrinks the generator's class rows on clients that lack a class (Lemmas 1-2, Fig. 1). Second, each head's last layer fits one client's prior, and per-client logit corrections only reparametrise it (Lemma 3). Third, label-free synthetic features penalise logit spread (Lemma 5, Fig. 2).

FedECS-LT replaces the failing estimates with statistics summed by secure aggregation: a generator anchored to the exact class means (or CCWA class rows when a Fisher-type gate says the classes are poorly separated), moment correction of generated features, logit adjustment, consensus soft labels, and finally a class-balanced shrinkage LDA readout for each architecture. With ten architectures (K = G = 10), FedECS-LT beats eleven methods by +15.1 / +8.6 points on CIFAR-10/100-LT. When the same readout is given to every GeFL-F-pipeline method, the lead drops to +1.4 / +1.5 and is not significant. With one shared ResNet-8 the method is below several baselines.

The claimed contributions are (i) a long-tail evaluation and stage-wise diagnosis of GeFL-F, (ii) the method, and (iii) a comparison with 21 methods in two protocols, with the readout added to the baselines so that its share of the gain can be separated from the rest.

## 2. Scores (1 = poor, 5 = excellent)

| Criterion | Score | Justification |
|---|---|---|
| Originality | 2 | The readout is a known closed-form classifier (Fed3R/FedNCM/FedCOF family). What is new is the coupled-decay diagnosis and applying the readout per architecture. |
| Technical soundness | 3 | The lemmas are correct but elementary. The diagnosis is descriptive, not causal, as Sec. 4.1 itself says, and no component is ablated. |
| Experimental rigour | 2 | Many baselines, paired seeds, per-class logging and honest caveats. Against that: only three seeds, settings fixed on the same splits and test sets, no ablation, several announced experiments absent, and a single K = G configuration. |
| Clarity | 3 | Precise and self-critical, but dense and heavy with hedges. Terminology is confusing in places ("as published", "loss", "exact") and symbols are overloaded. |
| Significance for Neural Networks | 2 | The lead that survives the shared readout is ≤1.5 points and unresolved. It costs about 185× the communication of a no-generator baseline. |

**Recommendation: major revision.** The paper is unusually honest about what its evidence shows, but that honesty also shows that the main empirical contribution reduces to applying a known readout. The method's own components are not yet shown to matter.

## 3. Major concerns

**M1. No component ablation, so contribution 2 is unsupported beyond the readout.** (Needs experiments.)
The conclusion (p. 34) says: "Which of these components carry the effect beyond the readout is not yet measured." FedECS-LT changes five things plus the readout (Sec. 6.2). With the readout, its margin over the best other method is +1.4 [-0.7, +3.6] on CIFAR-10-LT and +1.5 on CIFAR-100-LT (Holm p = 0.37 / 0.058). On the six datasets never used in development the margin is "+0.0 to +0.9" (Sec. 7.3).

On BloodMNIST, FedDF+ER actually has the higher mean in the released runs: 86.38 against 86.35 (`nat_er.csv`; per-seed `er_bal`). The text's "+0.0" hides this sign.

Requested runs:
- leave-one-out ablation of anchor/CCWA, moment correction, logit adjustment, consensus labels and T_s=10, all with ER;
- GeFL-F + AdamW (decoupled decay), the direct test of diagnosis 1 that Sec. 4.1 says is missing;
- GeFL-F and FedDF with T_s = 10, the compute-matched comparison that Sec. 6.9 says is missing.

Until these exist, contribution 2 should be reduced to "the exact readout plus a set of modifications whose joint effect is ≤1.5 points".

**M2. The readout's dominance may be an artefact of K = G = 10.** (Needs experiments.)
With one client per architecture, heads are never averaged (Sec. 3, Table 5: "none at K = 10"). So Balanced Softmax, FedLC and FedRS amount to purely local head training on the shared extractor, and the readout is the only channel through which an architecture sees other clients' classes. In the one-shared-model protocol, where the last layer is trained by all clients, the readout *hurts*: -3.0 points on CIFAR-10-LT (Sec. 7.4). FedECS-LT with readout (58.4) is then below plain FedAvg (61.0) and below its own heads (61.3).

Requested: at least one setting with several clients per architecture (e.g. K = 50, G = 10), which the paper itself lists as a planned run. Also state plainly in the abstract and the limitations that the main result is for one client per architecture.

**M3. Settings tuned on the reported splits and test sets.** (Partly text, partly experiments.)
Sec. 6.5 says about 36 configurations on CIFAR-10-LT and 17 on CIFAR-100-LT were compared "with seeds 3 to 5 with the same client splits as the reported runs, and ... on the same test sets". Sec. 6.9 adds: "There are no client splits that the development never saw." The baselines were not tuned.

This makes the CIFAR comparisons optimistically biased for FedECS-LT, and the small ER margins are exactly the size such selection could produce. The clean evidence, the natural-imbalance datasets, shows ≤0.9 points.

Requested: fresh seeds/splits (e.g. 6-8) for the CIFAR settings. Better still, select settings on the held-out validation block the paper already has (Sec. 6.3), and report all settings selected that way.

**M4. Diagnostic claims are stronger in the intro and conclusion than in Sec. 4.** (Text, plus cheap analyses.)
- Sec. 4.1 says the row shrinkage is "one measured property of GeFL-F's generator, not a demonstrated cause of its rare-class loss". The generator carries little class information for any class: fidelity 12.9 % on CIFAR-10 and 1.1 % (chance) on CIFAR-100. Yet the conclusion states that the method "loses rare classes in at least three places: coupled weight decay under Adam ...".
- The intro says holders "barely grow" the row, but Ḡ_c, D̄_c and κ_c (Eq. 1) are never reported. Please report them and check κ_c against Fig. 1. Fig. 1 is not monotone: class 9 has a longer row than classes 7-8.
- Fig. 2 does not support diagnosis 3. FedDF uses GeFL-F's label-free generator (fidelity 12.8 %, Table C.15), yet its synthetic phase reduces logit spread *less* than FedECS-LT's (≈-1.6 vs ≈-2.0). The text mentions only the comparison with GeFL-F.
- The moment correction has no measured benefit. Fidelity goes from 25.9 ± 2.1 to 28.9 ± 4.6 on CIFAR-10 and from 8.8 to 8.8 on CIFAR-100, and tail fidelity *drops* (44.1→40.3, 11.5→9.9; Table C.15).

The conclusion and highlights should be rewritten to match the hedged Sec. 4.

**M5. Promised content is missing from the submitted build.** (Text, or the runs.)
- Sec. 6.5 lists "GeFL's baselines" (grouped FedAvg, FedProx, FedALA, AvgKD, grouped CCVR and FedETF) as one of six method groups. They also appear in Table 1 ("Methods in this study"), in Table 5 and in the limitations ("CCVR and FedETF enter only GeFL's grouped protocol"), but no table reports them.
- Sec. 6.8 mentions a "pooled-data reference" that is never shown.
- Sec. 6.4 defines the shared-model protocol at ρ ∈ {100, 50, 10}, but only ρ = 100 is reported.
- The natural-imbalance numbers with the readout, quoted in the abstract and in Sec. 7.3, are in no table. The same holds for the tail recall with readout ("+0.0 ... -0.1", Sec. 7.2).

Either add the tables (a natural-imbalance ER table and a paired-difference table for Table 8) or remove every mention.

**M6. The trade-offs are absent from the abstract and highlights.** (Text.)
- On the clients' own class mixes, FedECS-LT with readout is below all 12 other methods on CIFAR-10-LT: 59.2 vs 73.0 for GeFL-F (Table C.17). The readout lowers this metric by 9.4 points.
- On CIFAR-100-LT its ECE (43.8) is worse than GeFL-F (42.4) and FedDF (31.8) (Table C.18). Sec. 7.6 reports only the favourable CIFAR-10 ECE.
- Cost: 11.3 GB per client against 60.8 MB for Balanced Softmax, which comes within 1.4 points once both have the readout (Table 12).

The phrase "With one shared model, the strongest long-tail methods are better" understates Table 11. Even FedAvg and MOON beat FedECS-LT on CIFAR-10-LT, and the CIFAR-100 entry rests on one seed.

**M7. Privacy of the released statistics is under-analysed.** (Text, ideally a small experiment.)
Class means of d = 10×16×16 = 2560-dimensional features from a two-convolution extractor are close to down-sampled mean images. The server sends μ_c to every client (Sec. 5.2), the smallest CIFAR-100 classes have a handful of images, and singly held classes reveal one client's statistic exactly. Sec. 5.5 acknowledges this qualitatively.

Requested: an inversion test (reconstruct images from μ_c for small classes) or a DP variant of the statistics release, together with the accuracy cost.

## 4. Minor concerns and presentation

- **"As published"** is misleading. Readers will take it to mean numbers copied from the original papers, but it means "without the readout, run by us". Use "native" or "without ER".
- **"The loss has three sources"** in the abstract mixes up accuracy loss and training loss. Write "the accuracy is lost in three places".
- **"Exact"** is used for quantised, clipped sums, and for a plug-in LDA that is "exact" only in the sense of being computed from aggregates. Define the term once, early, and avoid "exact readout" implying optimality.
- **Imbalance labels:** settings are named "ρ = 100", but the realised ratio is 48 on CIFAR-10-LT (Table 3). Captions should give the realised ratio.
- **Symbol overloading:**
  - G is the generator, the number of architectures and Ḡ_c;
  - ρ is the imbalance ratio and the contraction factors ρ_{r,i} in Lemma 6;
  - α is the Dirichlet concentration and the FedRS scale;
  - ω is the consensus weight and the synthetic weight ω_i in Lemma 6;
  - u is the uniform distribution and the body output u(h).
  LSE is never defined. The batch size B and the dimensions d and d_e are never given (Table 2 lacks B). ψ_h is undefined in the proof of Eq. (1).
- **Statistics:**
  - The "unresolved" rule (|Δ| < 0.7 with ER, Sec. 6.8) would class the +1.4/+1.5 ER leads as *resolved*, while the abstract calls them unresolved. State that the floor is a necessary condition, not a decision rule.
  - Four-digit p-values (0.0052, 0.0042, 0.338) from three seeds are over-precise.
  - Table 8 has no intervals or significance marks.
- **Table 10:** FedECS-LT (heads only) is bold on OrganAMNIST (83.7 vs 83.3 with readout), so the readout lowers accuracy there. Comment on this. The table font is very small.
- **Wording, Sec. 7.1:** "below 11 of the 11 other federated methods ... GeFL-F, ..., Local, ... are above it" counts Local, which is not federated, among them. "0 classes on average" is awkward.
- **Fig. 1:** the FedDF curve is identical to GeFL-F's and adds nothing. Add the FedECS-LT CCWA rows where they are used.
- **Fig. 3:** text is unreadable at print size.
- **Fig. 4:** t-SNE of one unnamed architecture adds little beyond the k-NN numbers. Consider moving it to the appendix.
- **Fig. C.7:** labels are rotated and crowded.
- **Fonts:** the PDF embeds Type 3 bitmap fonts (pdffonts: body fonts F37-F73), so text extraction loses ligatures ("dierent", "classier"). Install cm-super or use `lmodern` with T1.
- **Citations:**
  - [53] (Menon et al., statistical view of distillation) is cited for "one secure sum", which it does not address.
  - [52] (Soudry et al.) does not concern logit adjustment.
  - The citation of a repository file ("experiments/LEDGER.md", Sec. 6.5) is inappropriate in a journal paper. Give the numbers in a supplementary table instead.
  - [28] FedLTLib is an arXiv preprint from September 2026. If it is the protocol authority, cite the code version or commit.
- **Repetition:** "most of the margin is the readout's ... not resolved with three seeds" appears in the abstract, the contributions, Sec. 7.1, Sec. 7.3, the conclusion and the highlights. The tuning caveat appears in Sec. 6.5 and again in Sec. 6.9. The "every client runs every body" requirement appears in Sec. 3, 5.4 and 5.5. Consolidate.
- **Sec. 6.2** says "changes five things ... and adds a last step" and then gives a list of six numbered items. Number the readout separately.
- **Sec. 1** says "federation cannot bias class-conditional quantities by itself", while Sec. 4.1 shows that federated averaging under Adam biases the class rows. Rephrase as "the target is partition-invariant; the estimators are not".
- **Highlight 5** ("Compared with 21 published methods in two protocols") is fine. Highlight 1 should say "in GeFL-F's generator" and drop any causal reading.

## 5. Checks of specific risks

- **Claims stronger than evidence:**
  - the conclusion's three "places" where rare classes are lost (M4);
  - "best on all six naturally imbalanced datasets, by at most +0.9 points over the same-readout methods" (abstract): with ER, BloodMNIST is a tie or slightly negative (M1);
  - "the strongest long-tail methods are better" understates Table 11 (M6);
  - "the clients that hold the class barely grow it" (Sec. 1) is not measured.
- **Numbers vs tables:** spot checks agree.
  - Sec. 7.1/7.2 margins: +15.1, +8.6, +18.9, +36.4, +39.7.
  - Sec. 7.3: +6.4 to +18.4.
  - Sec. 7.4: 6 above and 5 below; 3 above and 8 below.
  - Sec. 7.6: ×1.13, ×1.30, ×185, ×15; ECE ranks 3 and 5 of 13.
  - Sec. 4.2: +16.6 and +7.2 match Tables 6/8.

  The exceptions are numbers cited without any table: the natural-dataset ER margins, the tail recall with ER, the zero-recall class counts, the k-NN accuracies, and the ER p-values (0.338, 0.0095, 0.0143).
- **Promotional or vague sentences:** very few. The text is, if anything, over-hedged. "The embeddings of FedECS-LT separate the classes better" (Sec. 7.5) rests on 40.9 vs 38.9 % k-NN accuracy, a modest difference.
- **Repeated content:** see Sec. 4 above.
- **Undefined symbols and terms:** LSE, B, d, d_e, ψ_h, and the overloaded G/ρ/α/ω/u; "referee" is defined only in the appendix.
- **Figure and table quality:**
  - Fig. 3 and Table 10 are too small to read.
  - Fig. 1 has a redundant curve.
  - Tables 6-11 lack a paired-difference companion with ER.
  - The Type 3 fonts need fixing before production.

## 6. What the authors do well

- **Honesty about the readout.** Evaluating every baseline in the pipeline with the same readout is exactly the control most papers omit. Reporting that it removes most of the margin deserves credit.
- **Lemma 3 is confirmed by the data.** It predicts that Balanced Softmax, FedLC and FedRS differ only through the last layer. Table 8 bears this out: 51.2 / 51.2 / 51.1 and 17.2 / 17.2 / 17.2 once the last layer is replaced.
- **Experimental hygiene.** Paired seeds, identical splits verified across methods, last-round evaluation, per-class logging, a measured GPU non-determinism floor, a Holm correction, and an explicit threats-to-validity section listing the tuning leakage.
- **Breadth of comparison.** 21 methods are run in the authors' or library code on common splits, with cost reported for each.
- **Correct theory, scoped modestly.** The lemmas are correct and clearly scoped, and the moment-correction and anchor bounds are clean.
- **Reproducibility.** The released code and per-run logs let a reviewer verify the printed numbers, which I did for several of them.
