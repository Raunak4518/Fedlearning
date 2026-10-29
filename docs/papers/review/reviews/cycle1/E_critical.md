# Agent E: critical external review (Neural Networks), manuscript v03 (git cddbac7)

Reviewer role: a skeptical referee looking for the strongest reasons to reject. I worked read-only. Sources: the built
PDF `docs/papers/elsevier/main.pdf` (its text is identical to `git show cddbac7:docs/papers/elsevier/main.pdf`); the
v03 sources (`git show cddbac7:docs/papers/content/*.tex`; HEAD moved to v04/0c7b15e while I was reviewing, so line
numbers below refer to cddbac7); `fedecs_lt/src`; `fedecs_lt/results/*/runs.jsonl` and `gefl_measure.jsonl`;
`docs/papers/generated/` (numbers, tables, STATUS.md); `experiments/LEDGER.md`. I do not repeat earlier findings (C-*,
D-*, N-*, B N1-N8) unless I add new evidence or a new consequence.

**Precondition (desk-reject risk).** As built, v03 is not a submittable manuscript. Tables 13, 14 and 15 are empty
placeholders ("Placeholder: table slot abl_bal (GUIDE.md)"). Section 7.5 contains "[Write once the component ablation
has been run (GUIDE.md, section 4, experiment E7) ...]". Unfilled slots are printed in the running text:
"[abl/cifar10/GeFL-F + LA/gain]", "[t2/cifar10-r100/fe200_di]" and four "[main/.../Centralised (pooled data)/...]".
Because the draft build prints every `\whentab`/`\whennum` branch, the paper also states evidence that does not exist
(E-1). An editor would return the paper on sight. The rest of this review treats the paper as if the slots were
removed, so that the science can be judged.

---

## Strongest reasons to reject

### R1. Once the readout is applied to every method, the distinctive parts of FedECS-LT show no demonstrated effect, and nothing isolates them

- **Evidence.**
  - Table 9 gives the balanced accuracy with the exact readout (ER). FedECS-LT reaches 52.6 / 18.7. Balanced
    Softmax reaches 51.2 / 17.2. In this setting Balanced Softmax is local logit-adjusted training of each client's
    head on the shared extractor: no generator, and no Stage-III exchange at K = 10 (runner.py:27; sec 6.5).
    - The margins are +1.4 [-0.7, +3.6] and +1.5 [+0.6, +2.4], not significant after Holm.
    - The per-seed differences on CIFAR-10-LT are 0.6, 1.3 and 2.4 (runs.jsonl). Two of the three are at or below
      the paper's own ER noise floor of 0.7 to 1.4.
  - On the six naturally imbalanced datasets, which are the only ones not used during development (E-7), plain
    GeFL-F with the readout is within 0.0 to 1.3 points of FedECS-LT with the readout (`generated/tables/nat_er.csv`):
    - DermaMNIST 44.6 vs 45.4;
    - OrganAMNIST 82.5 vs 83.3;
    - BloodMNIST 86.3 vs 86.3;
    - PathMNIST 72.3 vs 73.6;
    - EMNIST 66.5 vs 67.3;
    - FER-2013 38.2 vs 39.2.
  - The component ablation (Table 14), the GeFL-F variants (Table 13) and the sweeps (Table 15) are empty
    (STATUS.md: NB30 and NB31 not returned). So nothing shows that the mean anchor, CCWA, moment correction or
    consensus labels contribute anything.
- **Why it matters.** The paper's own measurements say that the readout, a known LDA-type closed-form head
  (Fed3R, FedNCM, FedCOF; credited as known in v04), carries the result. The generator-side contributions are
  indistinguishable from noise once every method has that readout. RQ4 is unanswered. RQ2 as trained is confounded
  (R4).
- **Fix.**
  - Run NB30 and NB31 (already specified). Report every arm both as trained and with ER.
  - Add one arm the plan does not have: the FedECS-LT heads (T_s = 10, consensus labels, LA) trained on features
    from a Gaussian sampler with the exact class means and spreads, with no trained generator. This is CCVR's
    sampler. If this arm matches FedECS-LT, Stage II is unnecessary.
  - Until then, state in the abstract and conclusion that the generator-side components have no measured effect
    beyond the readout.
- **Verification.** Table 14 is filled. Each component's ER effect is reported with its CI and compared with the
  0.7-point floor. The Gaussian-sampler arm exists.

### R2. The analysis (Section 4) and the headline method (Sections 5 and 7) contradict each other on the last layer, and the "exact class statistics" framing does not fit three of the five changes

- **Evidence.**
  - Section 4.2 and Proposition 3 argue that the last layer is where most of the loss sits, and that logit
    corrections "add no information about the classes the client lacks". Yet the headline method keeps per-client
    last layers trained with logit adjustment using the client's own prior π̂_k (sec5 §5.3). The one repair of loss
    (ii) that does use pooled information, the readout, is excluded from the headline: "the headline numbers of
    FedECS-LT are its heads as trained" (sec5 §5.4).
  - The consequence is stark. The paper's evaluation device applied to the unmodified base method (GeFL-F + ER:
    50.3 / 14.9) beats the proposed method as trained (42.8 / 14.3) by 7.5 points on CIFAR-10-LT. FedECS-LT's own
    heads are 9.8 points below its own readout.
  - The abstract (abstract.tex:8-11) says FedECS-LT "replaces each of these estimates by an exact class statistic
    computed under secure aggregation: ... and logit-adjusted heads trained on consensus soft labels". In fact:
    - LA uses a local, non-aggregated prior;
    - the consensus is averaged predictions, not a class statistic;
    - T_s = 10 is compute.
    Only the anchor, CCWA and the moment correction use the exact statistics.
- **Why it matters.** Either the readout is part of the method or it is not.
  - If it is, the honest headline is +1.4 / +1.5 points, not significant, and the method's best part is known.
  - If it is not, a trivial combination of known parts (GeFL-F or local LA heads plus a federated LDA head) beats
    the method by about 8 points, and the paper has to explain why anyone would deploy FedECS-LT's trained heads.
  - Either way the title's claim, that exact class statistics are what make the method work, is not supported.
- **Fix (text).**
  - Make the decision explicit.
  - Either (a) define FedECS-LT as including the readout and lead with the ER comparison, or (b) keep "as trained"
    and add GeFL-F + ER and Balanced Softmax + ER as baselines in Table 7, with the 7.5-point gap discussed.
  - Rewrite the abstract's component list so that it says which components use secure-aggregated statistics.
- **Verification.** The abstract and Table 7 contain the GeFL-F + ER comparison or the redefinition, and the
  component list matches sec 6.2's five changes.

### R3. Diagnosis (i), that coupled decay erases rare classes, is not tested causally, and the paper's own logged generator diagnostics contradict its rare-class selectivity

- **Evidence.**
  - The runs log a referee classifier's per-class accuracy on generated features (`fidelity`, `fidelity_mc` in
    runs.jsonl; server.py:370-377). The paper trains this referee (sec 6.3) but never reports it.
  - CIFAR-10-LT (chance 0.10):
    - GeFL-F's fidelity is 0.113 / 0.156 / 0.118 (seeds 3/4/5) averaged over classes.
    - On the four tail classes alone it is 0.172 / 0.168 / 0.166, which is not lower.
    - The head classes are just as unrecognisable. Seed 3, classes 0-2: 0.16, 0.04, 0.09. The rows of these classes
      are the longest in Fig. 1.
  - CIFAR-100-LT: GeFL-F's fidelity is 0.010 to 0.012, which is chance.
  - So GeFL-F's generator fails for all classes. The row-norm ordering of Fig. 1 does not predict which classes
    fail. That undercuts the causal story in which row shrinkage loses the rare classes.
  - No run tests decoupled decay alone, or λ = 0. The planned arm "GeFL-F + LA" (runner.py:23) changes AdamW and
    LA together.
  - Highlight 1 still says "Coupled weight decay under Adam erases rare classes from federated generators."
- **Caveat.** The referee's accuracy on real held-out features is not logged. Absolute fidelity values need that
  reference, but the head-versus-tail comparison does not.
- **Fix.**
  - Report the referee fidelity per class (head, medium, tail), together with the referee's accuracy on real
    features.
  - Add an arm "GeFL-F with AdamW" (cross-entropy) and Fig. 1 row norms with λ = 0.
  - Soften highlight 1 and the abstract to what Fig. 1 shows (shorter rows), unless the AdamW-only arm shows a
    rare-class accuracy gain.
- **Verification.** A fidelity table exists. The AdamW-only arm's tail recall is reported. Highlight 1 matches it.

### R4. The confirmatory evidence is not independent of the development, and the as-trained gain is confounded with more compute and ensemble distillation

- **Evidence.**
  - Settings were fixed on "seeds 3 to 5 with the same client splits as the reported runs, and every earlier run was
    evaluated on the same test sets" (sec6:210-212). `experiments/LEDGER.md` lists about 40 FedECS-LT variants run on
    CIFAR-10-LT, K = 10, α = 0.5 (K01-K26), many on seeds 3-5. The reported CIFAR results therefore come after
    selection on the test set.
  - The gate statistic on CIFAR-10-LT is F = 0.080, 0.066 and 0.071 against F0 = 0.05, chosen in that development.
  - The as-trained margin over FedDF (+5.2 [+1.2, +9.3], p = 0.066) compares the following:
    - FedECS-LT: 10 synthetic epochs per round, consensus soft labels (ensemble distillation from all ten heads, the
      AvgKD/FedDF target form) and LA;
    - GeFL-F and FedDF: 1 synthetic epoch, hard labels, cross-entropy;
    - training GFLOPs per round: 1,424 against 579 (Table 16).
  - The T_s = 1 arm is not returned. The development record does contain a T_s = 1 vs 10 comparison (K17: "Ours
    T_s = 1" 44.8 vs "T_s = 10" 43.8 on CIFAR-10-LT, earlier implementation and configuration). That points the
    other way, but it is part of the selection and cannot be used as confirmation.
- **Why it matters.**
  - With an ER margin of about 1.5 points and n = 3, a small selection effect on the test set is enough to produce
    the result.
  - The only datasets outside development (natural imbalance) show ER margins of 1.3 points or less.
  - As trained, the gains over GeFL-F there are large (+3.4 to +9.4) and count in the paper's favour. But the
    method's three non-statistics components (T_s, consensus, LA) are not separated from its statistics components.
- **Fix.**
  - Rerun FedECS-LT and the three strongest comparators on fresh seeds (for example 6-8, new splits) for both CIFAR
    benchmarks, or report results on a held-out half of the test set that development never touched.
  - Add GeFL-F and FedDF with T_s = 10, and FedDF + LA.
  - Report a sensitivity check for F0 (both branches on CIFAR-10-LT).
- **Verification.** A fresh-seed table exists, and the margins as trained and with ER are restated from it.

### R5. The setting undermines the premise of model heterogeneity, and the cost-benefit ratio is poor

- **Evidence.**
  - In the main setting each architecture has exactly one client (K = 10), so "heads of the same architecture are
    averaged" does nothing. Stage III is local training plus the generator and the consensus.
  - The K = 50 setting is announced in sec 6.4 and Table 2 but has no results (STATUS.md: folder 7 not returned).
  - The consensus requires the server to send "every architecture's current head to every client" (sec5:82-83). The
    readout requires every client to run "a forward pass ... of every architecture's body" (sec5 §5.4). If clients
    can run every architecture, the motivating constraint ("participants run different networks on different
    hardware") no longer applies. Architecture confidentiality is also lost.
  - Cost (Table 16), FedECS-LT against Balanced Softmax:
    - communication 11,278 MB per client against 60.8 MB, which is 185 times more;
    - wall-clock 7,910 s against 542 s, which is 15 times more;
    - with the same readout, the gain is +1.4 points, not significant.
  - The extractor has 3 or 10 channels. Absolute balanced accuracy is 42.8 on CIFAR-10-LT and 14.3 on CIFAR-100-LT.
  - With one shared model, FedECS-LT (61.3) equals FedAvg (61.0) and is 16.8 points below FedYoYo.
- **Why it matters.** For Neural Networks, a method that is near-useless in absolute terms, matched by a local
  baseline plus a known readout at 1/185 of the communication, and evaluated in a degenerate one-client-per-
  architecture regime, has weak significance.
- **Fix.**
  - Return the K = 50 runs, where averaging within an architecture actually happens.
  - State the per-client compute and download requirement (all G heads and bodies) as an assumption in sec 3, and
    discuss whether it is compatible with heterogeneous hardware.
  - Add accuracy-versus-communication to the cost discussion, with Balanced Softmax + ER as the reference.
  - Optionally add one experiment with a stronger, pretrained extractor.
- **Verification.** K = 50 results are in Section 7.1. Sec 3 states the assumption. The cost paragraph compares
  against Balanced Softmax + ER.

---

## Findings table

| id | location (cddbac7) | problem | evidence (quote / data) | why it matters | sev. | conf. | proposed action | verification |
|---|---|---|---|---|---|---|---|---|
| E-1 | PDF §6.4 (sec6:146-147), Tab. 2 (sec6:38), sec4:80, sec6:384, sec8:17, sec8:25, sec8:39-42, sec7:178 | The draft build prints conditional statements about evidence that does not exist | "GeFL's second setting has K = 50 clients ... every method of the main comparison is run there too"; "its effect on the body is measured by removing logit adjustment ... (Table 14)"; "The ablation with one synthetic epoch shows how much ..."; "covered only by Table 15"; pooled-reference sentence with unfilled slots | Statements of performed experiments that were not performed, in the version circulated for review | P0 | high | Build with `\finalslotstrue`, or remove every sentence whose slot is empty; never circulate a draft-mode PDF | `pdftotext main.pdf - \| grep -c "\["` finds no slot names; no sentence refers to Tables 13-15 or K = 50 while they are empty |
| E-2 | Tab. 9, Tab. C (nat_er), sec7 §7.1, abstract | The generator-free local LA baseline plus the readout, or GeFL-F plus the readout, is within 0-1.5 points of FedECS-LT with the readout | CIFAR-10/100 ER: 52.6 vs 51.2, 18.7 vs 17.2 (Balanced Softmax); natural: GeFL-F + ER 44.6/82.5/86.3/72.3/66.5/38.2 vs 45.4/83.3/86.3/73.6/67.3/39.2 | The distinctive components have no measured effect beyond a known readout (R1) | P1 | high | Component ablation, plus a Gaussian-sampler arm with FedECS-LT heads; abstract and conclusion say the generator-side gain is not resolved | Tab. 14 filled; Gaussian arm reported |
| E-3 | sec4 §4.2, sec5 §5.3-5.4, abstract:8-11 | Logic gap: the dominant loss (last layer) is repaired only by the readout, which is excluded from the method; LA is kept although the paper shows it adds no information | "the correction adds no information about the classes the client lacks"; "the headline numbers of FedECS-LT are its heads as trained"; GeFL-F + ER 50.3 > FedECS-LT 42.8 | The method's own evaluation device beats it by 7.5 points (R2) | P1 | high | Decide whether the readout is part of the method; if it is not, add GeFL-F + ER and Balanced Softmax + ER rows to Tab. 7 and discuss | Tab. 7 or the abstract reflects the decision |
| E-4 | title, abstract:8-11, highlights:9, sec1 "Our method" | The "exact class statistic" framing covers 3 of the 5 changes | LA uses π̂_k of the client; consensus = averaged softmax; T_s = 10 is compute (sec6 §6.2 list) | The title and abstract attribute the effect to the statistics without evidence | P1 | high | List which components use secure-aggregated statistics; remove "replaces each of these estimates by an exact class statistic" or qualify it | Abstract component list matches §6.2 |
| E-5 | sec4 §4.1, Fig. 1, highlights:8, abstract:5-6 | The coupled-decay diagnosis has no causal test; only correlational row norms | No AdamW-only or λ = 0 arm; "GeFL-F + LA" switches AdamW and LA together (runner.py:23-24) | RQ1's first claim is unverified; highlight 1 ("erases") overstates | P1 | high | Add a GeFL-F + AdamW arm (CE) and row norms with λ = 0; soften highlight 1 now | Arm reported; highlight matches it |
| E-6 | sec4 §4.3, sec5 §5.2, sec6:122 (referee) | The logged generator fidelity is unreported and contradicts rare-class selectivity and the "teaching the missing classes" mechanism | GeFL-F CIFAR-10 fidelity 0.11-0.16 overall, tail 0.17 (not lower); CIFAR-100 0.01 (chance). FedECS-LT after moment correction: CIFAR-10 0.24/0.34/0.29, CIFAR-100 0.083-0.092 with 51-54 of 100 classes below 5%; moment correction changes CIFAR-100 fidelity by -0.009 to +0.015; FER-2013 (rows branch) 0.24 vs GeFL-F 0.21 | The mechanism by which synthetic data help is not shown, and the moment correction does not make generated features more recognisable | P1 | high (data) / medium (interpretation: referee real accuracy not logged) | Report fidelity by head/medium/tail with the referee's real-feature accuracy; temper sec4 §4.3 and sec5 claims; log the referee's real accuracy in future runs | Fidelity table in the appendix; text consistent with it |
| E-7 | sec6:210-212, sec6 §6.10, LEDGER.md | Development on the same seeds, splits and test sets; about 40 CIFAR-10-LT variants | "These included seeds 3 to 5 with the same client splits ... every earlier run was evaluated on the same test sets"; LEDGER cifar10 IF=0.01 K=10 (K01-K26) | Disclosure does not remove selection bias; the ER margin of about 1.5 is within plausible selection effects. The natural datasets are absent from the ledger, so they are the only clean evidence | P1 | high | Fresh seeds or splits (or a held-out test half) for FedECS-LT and the top 3 comparators; state which datasets development never touched; give the number of configurations tried | Fresh-seed table; sec 6.10 names the untouched datasets |
| E-8 | sec7 §7.1, Tab. 6, Tab. 16, sec6 §6.10 | The as-trained margin is confounded with T_s = 10 (2.5x training GFLOPs) and ensemble distillation | 1,424 vs 579 GFLOPs/round; consensus = mean softmax of all ten heads, ω = 0.5 | The +5.2 / +4.3 headline could come from compute or distillation rather than the statistics (D-6/D-26 known; the new part is the specific missing arms) | P1 | medium | Arms: GeFL-F T_s = 10, FedDF T_s = 10, FedDF + LA, FedECS-LT w/o consensus (planned) | Arms reported with paired CIs |
| E-9 | sec5:82-83, sec5 §5.4, sec1:2-4 | Every client must download and run all G heads each round and all G bodies once, which contradicts the heterogeneous-hardware premise; "hides nothing beyond what FedAvg already reveals" is false relative to FedAvg within an architecture | "sends every architecture's current head to every client"; "every client must run a forward pass ... of every architecture's body" | The method needs what model-heterogeneous FL assumes clients lack; architecture confidentiality is lost | P1 | high | State the assumption in sec 3 and the limitations; correct the privacy sentence ("reveals every architecture's head to every client, which FedAvg within an architecture does not") | Sec 3 and sec 5.5 changed |
| E-10 | sec6 §6.4, sec8:17 | The main regime is degenerate (one client per architecture); K = 50 is promised but absent | Tab. 6: "none at K = 10"; STATUS.md folder 7 not returned | Head federation, the core of GeFL-F's design, is never exercised | P1 | high | Return the K = 50 runs or restrict the claims to "one client per architecture" | K = 50 table, or the claims restricted |
| E-11 | highlights.tex:8-11 | Highlights overclaim | H1 "erases" (untested, E-5/E-6); H2 "fix generator, heads and readout" (no ablation; the heads use a local prior); H3 "recovers most of the long-tail loss" (relative to a pooled reference that was never run, and the readout is not part of the method); H4 "Compared in Flower with 21 published methods" (11 one-shared-model baselines run in LongTailLib, sec6 §6.7) | Highlights are what editors read | P1 | high | e.g. H1 "Rare-class conditioning rows of a federated generator shrink under coupled Adam decay"; H2 "Exact secure-aggregation statistics anchor the generator and correct its features"; H3 "With the same exact readout, gains over the best baseline are 1.4-1.5 points"; H4 "Compared with 21 published methods in two protocols" | `fill.py --check-highlights`; each bullet maps to a table |
| E-12 | abstract:16-17, sec1:70 | The natural-imbalance sentence omits the ER result and counts a one-seed dataset | "It also has the highest mean balanced accuracy on all six datasets" (EMNIST: FedECS-LT seed 3 only, STATUS.md); ER margins ≤ 1.3, BloodMNIST tied | Asymmetric with the CIFAR sentence, which does give the ER caveat | P1 | high | "... on all six (one with a single seed; significant on two); with the exact readout the margins are below 1.3 points" | Abstract slot text |
| E-13 | sec8:48-53 (Conclusion) | "Each can be described precisely, and each can be repaired with class statistics" | No component ablation; the last-layer "repair" is the readout outside the method | The conclusion claims what RQ4 has not answered | P1 | high | "... each is addressed by a component of FedECS-LT; which components carry the effect is not yet measured" until Tab. 14 exists | Conclusion text |
| E-14 | sec4:65-81 (Prop. 3 and text) | "a logit shift ... is therefore equivalent to a different initialisation of the last-layer bias" omits the test-time shift | Training with ℓ(f+a_k) ends at b_T, while plain training from b_0+a_k ends at b_T+a_k: the LA head is the shifted-init plain head minus a_k, i.e. post-hoc logit adjustment (Menon et al. 2021 §4) | The stated equivalence is wrong for the evaluated predictor, and the correct statement is a known one | P2 | high | "equivalent to plain training from the bias b_0 + a_k followed by subtracting a_k from the logits at test time (post-hoc logit adjustment [15])" | Text changed |
| E-15 | sec1 contribution 1, sec4 | Elementary facts presented as propositions | Prop. 1 = triangle inequality; Prop. 4 = associativity of addition (known, v04 credits it); Prop. 5 = one-line identity; Prop. 3 = translation of a free bias | Inflates the theoretical contribution; NN reviewers will discount sec 4 | P2 | high | Demote them to remarks or lemmas; claim the contribution as "a measured diagnosis", not "exact results" | Contribution 1 reworded |
| E-16 | sec3:20-28, sec1:20-22 | "it can make it biased only through (i) the optimiser, (ii) ..., (iii) ..." is unproved and, with "the optimiser" this broad, unfalsifiable | No statement or proof; client drift of the shared extractor (Stage I FedAvg on skewed data) is not covered | The analysis's completeness claim is rhetorical | P2 | medium | Reword as "we examine three sources" and drop "only" | Text changed |
| E-17 | Tab. C.19 vs Tab. 7 | FedDF's "best round" (35.6) is below its last-round value (37.5), which is impossible for the same model | `gefl_measure.jsonl` FedDF gefl_last_acc 0.339/0.360/0.350 vs final_bal 0.365/0.389/0.373; gefl_measure.py:44-46 falls back to `pre` (client heads before server distillation) when `post_arch` is absent | C.19 scores FedDF without its distillation step, so FedECS-LT's lead under GeFL's measure is inflated (10.5 vs 5.2) | P2 | high | Footnote FedDF in C.19 as pre-distillation, or drop the row until `post_arch` logs exist | Caption or row changed |
| E-18 | sec6 §6.5, Tab. 7 | No competitive heterogeneous long-tail baseline | The strongest baseline, FedDF, distils on GeFL-F features at chance fidelity (E-6); HtFLlib ports degrade from round 1 (D-7); Balanced Softmax/FedLC/FedRS are local; grouped CCVR/FedETF have no results | The +5.2 / +4.3 margins are over weak opponents | P2 | high | Return the grouped CCVR/FedETF runs; add CCVR per architecture with exact-statistic Gaussian features (cheap, same statistics) | Rows in Tab. 7 |
| E-19 | sec1 contribution 3, highlights:11 | "twenty-one published methods on eight datasets in two protocols" | 11 methods on the eight datasets of the main protocol; 11 others on two datasets only; the one-shared-model baselines are not run in Flower | Overstatement of scope | P2 | high | "eleven methods on eight datasets with different client models, and eleven on two benchmarks with one shared model" | Text changed |
| E-20 | sec6 §6.2, sec8 limitations | The extractor has 3 or 10 channels; balanced accuracy is 42.8 / 14.3 | Tab. 7; sec6:702-707 | Relevance of the results to neural-network practice is doubtful; a 1.5-point difference at 14% has little practical meaning | P2 | high | One experiment with a stronger or pretrained extractor (e.g. ResNet-8 Stage I from the one-shared protocol, reused) | Table or limitation explicit |
| E-21 | sec7 §7.7 | Calibration framed selectively | "37.8 ± 4.0 ... against 48.0 for GeFL-F and 35.9 for FedDF, the strongest other method"; Tab. C.21: FedProto 17.1, FedTGP 8.8 | FedECS-LT is worse calibrated than three baselines on CIFAR-10-LT | P3 | high | Say it ranks 4th of 13 on ECE (CIFAR-10-LT) | Text |
| E-22 | sec7 §7.1, Fig. 2 | The logit-spread comparison is confounded | FedECS-LT has 10x synthetic steps, soft targets and LA in the real phase; GeFL-F has 1 synthetic epoch, hard labels and CE | "reduces it less for FedECS-LT" cannot be attributed to label-carrying features | P3 | medium | Add the caveat; compare at matched T_s once NB31 returns | Caption or text |
| E-23 | Tab. 12, Tab. 16, Tab. C.20, C.21 | Broken ± typesetting | "78.1 0.9 42.6 0.5 ±"; "485 34±"; "417 1.3e+02 ±"; "501 ± 1.2e+02" | A reviewer stops trusting the tables | P3 | high | Fix fill.py formatting (no scientific notation, ± inside cells) | Rebuilt PDF |
| E-24 | refs [8], Data availability, front matter | Placeholders | "[repository URL ...]", "[First Author]", "IEEE Transactions on Mobile ComputingArXiv" | The reproducibility claim ("are public") cannot be checked | P3 | high | Fill them in, or say "will be released on acceptance" | PDF |
| E-25 | sec5 §5.2 | The gate was chosen in development and CIFAR-10-LT sits at 1.3-1.6 times F0 | F = 0.080/0.066/0.071 vs 0.05 (D-8) | New consequence: with R1, the branch could decide the ER margin, and the other branch is untested | P2 | high | NB31 "w/ class rows (CCWA)" on CIFAR-10-LT; report both branches | Tab. 14 row |
| E-26 | sec6 §6.5, §6.10 | The number of development configurations evaluated on the test set is not disclosed | LEDGER.md: about 40 labelled variants (K01-K26) on CIFAR-10-LT K = 10 | Readers cannot judge the selection pressure | P2 | high | Give the count and the selection criterion | Sentence in §6.5 |

## Fixable by text now (no new runs)

E-1 (build mode / remove sentences), E-3 (decide the status of the readout; add GeFL-F + ER and Balanced Softmax + ER to
the headline discussion: the data exist in `er_bal`), E-4, E-9, E-11, E-12, E-13, E-14, E-15, E-16, E-17 (caption),
E-19, E-21, E-22 (caveat), E-23, E-24, E-26, and the reporting half of E-6. The fidelity data already exist in
runs.jsonl. Reporting them is a text and table change, but the interpretation must be tempered.

## Needs new experiments

- **E-2 / R1:** NB30 and NB31 (planned), plus a FedECS-LT-heads-on-Gaussian-sampler arm (new).
- **E-5 / R3:** GeFL-F + AdamW (CE) alone, and row norms with λ = 0 (new; small).
- **E-7 / R4:** fresh seeds or splits, or a held-out test half, for FedECS-LT, FedDF, Balanced Softmax and GeFL-F on
  both CIFAR benchmarks (new; about 24 runs).
- **E-8:** GeFL-F and FedDF with T_s = 10, and FedDF + LA (FedDF + LA is planned in NB30; the T_s = 10 arms are new).
- **E-10:** K = 50 (planned, folder 7).
- **E-18:** grouped CCVR/FedETF (planned); CCVR per architecture with exact statistics (new).
- **E-20:** one stronger-extractor experiment (new, optional).
- **E-25:** the CCWA branch on CIFAR-10-LT (planned, NB31).
- **E-6:** log the referee's real-feature accuracy (code change; rerun not needed if the referee can be re-evaluated
  offline, which it cannot be from the stored logs).

Is the paper acceptable without the pending experiments? No. Without Table 14, RQ4 is unanswered. Without fresh seeds
or a held-out test half, the only significant evidence of an effect beyond the known readout (CIFAR-100-LT ER CI
[+0.6, +2.4], uncorrected) comes from seeds on which the method was developed. The sweeps and the three further
benchmarks are less important. The ablation, the fresh-seed check and K = 50 are the minimum.

## What the paper does well (for balance)

- Paired design on identical splits.
- Every per-class metric is logged.
- The readout is reported for every pipeline method, not only for the proposed one.
- Negative results are stated plainly (one shared model; ER margins not significant).
- There is a measured run-to-run noise floor.
- Balanced Softmax, FedLC and FedRS are identical under the readout (51.2/51.2/51.1), which confirms Proposition 3's
  prediction.
- The as-trained gains on the natural datasets, which development did not touch, are large and consistent
  (+3.4 to +9.4 over GeFL-F).

These are the strengths of a good benchmark study. They do not support the method claims as currently written.

---

## Overall recommendation: **Major revision, bordering on reject.**

In its current form (v03 as built) it is a reject: empty result tables, statements of experiments not run, and the
central component claims untested. If the planned ablation and K = 50 runs return, and the paper is reframed around
what they show, it could become a solid empirical paper. That reframing would be: the readout carries most of the
gain, plus whichever generator components survive a fresh-seed check.

## Scores (0-5)

| criterion | score | justification |
|---|---|---|
| Research question | 3 | Clear and relevant (model heterogeneity × long tail), but narrowed to one method family (GeFL-F). |
| Gap | 3 | A genuine, under-studied intersection; the K = 10 one-client-per-architecture regime does not exercise it. |
| Literature | 3 | Broad and current; in v03 the readout's closest precedents (FedCOF, AFL) were missing (fixed in v04). |
| Contribution | 2 | The largest measured effect is a known readout; the novel components have no isolated effect. |
| Methodology | 2 | Paired and careful, but tuned on the same seeds and test sets, confounded by compute, and with weak heterogeneous long-tail baselines. |
| Evidence | 2 | n = 3, with the key ER margins non-significant and at 1-2x the noise floor; the ablations are missing. |
| Logic | 2 | Sec 4 says the last layer is the main loss and LA adds nothing; the headline method keeps LA heads and excludes the readout. |
| Conclusions | 2 | The conclusion and highlights claim repairs ("each can be repaired", "erases", "fix") that are not demonstrated. |
| Reproducibility | 3 | Code, logs and per-run JSON exist internally and the splits are checked, but the repository URL is a placeholder and GPU non-determinism adds 1.2 points. |
| Clarity | 2 | Dense but precise prose; the draft-mode placeholders, broken tables and conditional sentences make the PDF unreadable as a submission. |
| Limitations | 4 | Unusually candid threats section (tuning, compute, missing runs, privacy); it still omits the all-heads-to-all-clients assumption. |
| Suitability for Neural Networks | 2 | Within scope, but tiny frozen extractors at 14-43% accuracy and a marginal methodological advance weaken its significance for the journal. |
