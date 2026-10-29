# Cycle 1, Agent F (academic writing editor): FedECS-LT manuscript v03

Base: git `cddbac7`, `docs/papers/content/*.tex` and `elsevier/highlights.tex`. Built text: `elsevier/main.pdf`
(53 pages at this commit; IEEE 21). Line numbers refer to the files **as committed in cddbac7**. Other agents are
editing `appendix.tex`, `sec1`, `sec3`, `sec4` and `sec5` in the working tree right now, so lines in those files may
have moved by 1 or 2. I did not edit the paper. Every proposed text keeps the slots it needs (`\R{...}`) and adds no
value. Where a proposal needs a value that has no slot, it says so instead of typing one.

`style_check.py`: 0 em dashes, 10 flags. Eight are false positives: the word `robust` in table and slot names
(`sec6_setup.tex:41`, `sec7_results.tex:216, 258, 263, 264`, `sec8_discussion.tex:16, 17`). The heading
`\paragraph{Robustness.}` (`sec7:255`) is fine, but "Other settings" would be more exact. Two flags are real:
`sec7_results.tex:185` "clearly" (P3-22 below) and `declarations.tex:20` "in order to", which is placeholder text
in the AI statement and should become "to".

---

## P1: structural

### P1-1. Paragraph titles print a double full stop in both builds
Every `\paragraph{Title.}` renders as "Title.." in the Elsevier build (elsarticle adds a period) and as "a) Title.:"
in the IEEE build. The PDF has about 45 cases, including "Where the rare classes are lost..", "Limitations.." and
"Proposition 1..". **Fix:** drop the final period from every `\paragraph{...}` title in `content/*.tex`, which lets
each class add its own punctuation.

### P1-2. Threats to validity (§6.10) and Limitations (§8) say the same things twice
Together they hold 430 + 490 words. Seven items appear in both, and some appear a third time elsewhere:

| Idea | Occurrences | Keep |
|---|---|---|
| Settings tuned on the same seeds, splits and test sets; baselines not tuned | sec5:44 (parenthesis), sec6:207-210, sec6:379-381, sec8:27-29 | one item in the merged list; sec5:44 becomes "(fixed before the reported runs; \cref{sec:setup-threats})"; sec6:207-210 becomes one sentence with a pointer |
| Ten synthetic epochs vs one; more communication; the T_s=1 ablation does not show what others reach with ten | sec6:382-385, sec8:22-26, sec6:354-355 | one item |
| Small frozen extractor, results may not carry over | sec6:408-409, sec8:11-12 | one item |
| Only label shift, no feature shift | sec6:411, sec8:13 | one item |
| One-shared-model runs of FedECS-LT missing on some seeds | sec6:386-387 **and** sec6:412 (twice within the same subsection) | one item |
| Three seeds, little power | sec6:369-373, sec8:14-15 | §6.9 only; drop the sec8 bullet |
| With one shared model the long-tail methods are better | abstract, sec1:73-74, sec7:167-176, sec8:30-32, conclusion | sec7.4 and the conclusion; drop the sec8 bullet |
| Released sums are not differentially private | sec5:132-136, sec8:18-20 | sec5.5; sec8 keeps one sentence |
| Timing measured with a second run on the GPU | tab:cost caption, sec6:310, sec6:388-389 | §6.6 metric definition only |

**Proposal:** delete §6.10 and turn §8 into "Limitations and threats to validity", with one bullet per item grouped
as internal / construct / external. Most items (FedProto's decline, missing runs, readout scope) refer to results,
so they read better after §7. *Saves about 400 words.* The §8 paragraph "What the gains consist of" (sec8:3-7)
repeats §7.1's last paragraph (sec7:59-71) and adds one new sentence. Move that sentence into the construct
bullets: "The readout is the same linear classifier for every body, so accuracy with it does not measure every
property of a representation." Then delete the paragraph (*about 60 words*) and rename §8 "Limitations".

### P1-3. The study design describes experiments that the final build may not report
- `sec6_setup.tex:19` (RQ4) and `:20-21` (RQ5 "with other client numbers, label skews and imbalance ratios") are
  unconditional, but their table rows (`:30-31`, `:41`) and results (§7.5 `\whentab`) are conditional.
- §6.8 "Ablations" (`sec6:340-358`) is unconditional, yet `tab:components` has no results (the built PDF prints
  "[Write once the component ablation has been run ...]" and empty slots `[abl/cifar10/rest]`).
- Four sentences rely on that missing ablation, all unconditionally:
  - sec4:79-81, "its effect on the body is measured by removing logit adjustment";
  - sec6:354-355, the T_s=1 run "separates the effect of the longer synthetic phase";
  - sec6:384, "The ablation with one synthetic epoch shows ...";
  - sec8:25-26, the same sentence again.

**Fix:** wrap RQ4, the sweep clause of RQ5, the second bullet of §6.8, and these four sentences in
`\whentab{abl_components}` / `\whentab{robust}`, or rewrite them in the future tense with "not run" until the
results exist. Otherwise the final build promises results that it does not contain.

### P1-4. The three loss routes differ between the introduction and §3
- `sec1_intro.tex:21-22` says federation can lose rare classes "through the optimiser, through parameters tied to a
  client's label prior, and through **sample size**", then lists three losses, the third being synthetic data.
- `sec3_setting.tex:27` (v03) lists the optimiser, the prior and "**what the synthetic data teach**". The working
  tree now adds "(iv) sample size".

The two lists must match, and the introduction must say which routes the three losses come from. Proposed
`sec1:20-23`:

> \paragraph{Where the rare classes are lost} A client split is a pure label shift, so federation alone does not bias
> class-conditional quantities (\cref{sec:setting}); bias can enter only through the routes listed there. We find a
> loss through three of them (\cref{sec:analysis}).

### P1-5. Each of four ideas is stated four to seven times
1. **The GeFL-F pipeline** is described in sec1:13-16, sec3:78-87, sec5:3 and sec6:49-53. The last of these
   (§6.2, first paragraph, 65 words) is a full copy of §3. *Delete it* and open §6.2 with "\ours keeps the pipeline
   of \cref{sec:setting} and changes five things".
2. **The components of FedECS-LT** appear in the abstract, sec1:37-45 (165 words), sec5 (definitive), Fig. 1's
   caption (7 lines), Algorithm 1, the sec6:55-63 list and tab:fairness. Keep §5, Algorithm 1 and the §6.2 list,
   because the list defines what the ablations remove. Shorten the introduction (P2-1) and the Fig. 1 caption
   (P2-12). *About 120 words.*
3. **"The readout is an evaluation device; the main numbers are the heads as trained"** appears in sec5:103-106,
   sec6:64-65, sec6:283-284, sec6:400, the Fig. 1 caption and the abstract. Keep sec5:103-106 and the metric
   definition in sec6:283-284; delete sec6:64-65 and sec6:400. *About 50 words.*
4. **The novelty statement "readout per architecture needs statistics pooled across architectures"** appears in
   sec1:48-50 and sec2:29-30. Keep the introduction; end §2's second paragraph after "(FedAwS, FedRS)" and cut
   "These methods assume ... which \ours does." *About 25 words.*

### P1-6. Material in §6 that belongs in the appendix (the main text keeps a pointer)
| Passage | Words | Destination |
|---|---|---|
| Extractor layers per dataset, sec6:68-73 | 75 | merge into app. "Models" / "Head architectures" (app:69-85), which already say the models follow GeFL-F |
| Secondary metrics list (macro-F1, precision, MCC, G-mean, AUC-PR, class-wise ECE), sec6:291-295 | 45 | app:more, next to the tables that use them; keep tail recall, worst class, zero-recall classes, ECE, weakest architecture and own-class-mix accuracy in §6.6 |
| Cost measurement bullets, sec6:305-312 | 70 | tab:cost caption or appendix |
| Flower version, Ray, Kaggle and GPUs, sec6:320-323 | 60 | app:protocol |
| Reproduction check, sec6:330-332 | 50 | already in app:75-80 (duplicate); delete from §6 |
| Last-round evaluation, every five rounds, sec6:317-318 | 25 | already in sec6:302-303 and app:99-100; delete |
| "A seed fixes the split ... same clients", sec6:125-126 | 30 | duplicate of sec6:316-317; keep §6.7 |
| HtFLlib settings | 3 times | sec6:198-199, tab:baselines caption, app "HtFLlib ports" (working tree); keep the appendix and the caption |
| "exchange nothing in Stage III at K=10" | 3 times | sec6:178-179, sec6:183-184, tab:fairness caption and rows; keep the table, cut the two list sentences |
| Opening roadmap sentence, sec6:3-6 ("This section states ...") | 35 | delete; the subsection titles carry it |

*About 450 words moved or cut from §6.*

### P1-7. Floats that repeat other floats (the paper is 53 pages, mostly floats)
- **tab:rq** repeats the RQ list directly above it. Keep the list and add "(answered in \cref{...})" to each item;
  delete the table. This also removes the conditional-row machinery (sec6:30-31).
- **fig:forest** plots exactly the CIFAR-10 column of tab:paired, and the text only says "shows the paired
  differences" (sec7:57). Move it to the appendix or drop it.
- **tab:positioning** (§2) and **tab:baselines** (§6) list the same methods. Add the four check-mark columns to
  tab:baselines and delete tab:positioning, or keep tab:positioning and drop the Venue column of tab:baselines.
- **fig:hmt** overlaps tab:tail and the per-group numbers in sec7:99-108. **fig:realgen** and
  **fig:shared-rounds** are cited but never discussed (P1-8). Move all three to the appendix unless the text gains
  an observation.

*Estimate: 3 to 4 Elsevier pages.*

### P1-8. Figures cited without anything said about them
- sec7:98-99 (fig:recall-size), sec7:108-109 (fig:hmt), sec7:189 (fig:shared-rounds), sec7:279 (fig:realgen) and
  sec7:57 (fig:forest) only say "shows" or "overlays".
- tab:app-calib is never referenced: sec7:309 cites `app:more`. tab:app-calib and tab:app-worstarch also share one
  float with two captions (app:149-165); split them.

**Fix:** for each figure, either add one sentence that states what it shows (as a `\claim{}` with slots) or move
the figure to the appendix. Cite `\cref{tab:app-calib}` at sec7:306.

### P1-9. The paragraph on the main comparison has six purposes
`sec7_results.tex:7-29` is one paragraph covering:
1. the margin and its significance;
2. classes with zero recall;
3. the difference to GeFL-F;
4. why the HtFLlib methods are weak;
5. last-round vs best-round accuracy;
6. accuracy on the clients' own class mixes.

Split it into three paragraphs: (1)-(3) "Against the strongest method and GeFL-F", (4)-(5) "Other model-heterogeneous
methods", and (6) "Accuracy on the clients' own class mixes". No words change.

### P1-10. Results text names datasets that the results do not contain
- sec6:68-70 (extractor per dataset), sec6:105 (fractions for MNIST, Fashion-MNIST, SVHN), sec6:280-281 (balanced
  test sets) and app:70-73 name MNIST, Fashion-MNIST, SVHN and GTSRB. The built tables have only CIFAR-10/100 and
  six natural datasets.
- tab:main, tab:paired, tab:readout and tab:tail have hard-coded MNIST to SVHN headers.
- The term "long-tailed benchmarks" means five datasets in §6.3 but two in the built results.

**Fix:** make these names conditional (`\whennum`), or generate them from the same slot as `main/datasets_cite`.

---

## P2: paragraph

**P2-1. sec1:37-50, "Our method" (165 words; repeats §5).** Proposed (about 110 words, the novelty sentences
unchanged):
> \paragraph{Our method} \ours (Exact Class Statistics for Long-Tailed FL) replaces each estimate that fails by a
> class statistic that secure aggregation~\cite{bonawitz2017secagg,bell2020secaggplus} computes exactly: the per-class
> counts, feature sums and sums of squared norms over all clients. The generator is anchored to the class means and
> has no class row for the decay to shrink; when the classes are poorly separated, a gate computed from the same
> statistics falls back to class rows averaged over their holders only (class-count weighted averaging, CCWA).
> Generated features are rescaled to the exact class means and spreads, and heads are trained with logit adjustment
> and with consensus soft labels, the averaged prediction of all architectures. To compare the bodies of the heads,
> every architecture's last layer can also be replaced by the exact readout, a class-balanced linear discriminant
> computed from aggregated embedding sums. None of these statistics is new on its own: [unchanged to line 50].

**P2-2. sec1:28-32, item 2.** The FedRS step-size detail belongs to Prop. 2 only. Proposed:
> \item \emph{The last layer.} Each head is trained on one client's real data, so its last layer fits that client's
> label prior. Per-client logit corrections (logit adjustment~\cite{menon2021la}, FedLC~\cite{zhang2022fedlc},
> Balanced Softmax~\cite{ren2020bsm} and, up to a step size, FedRS~\cite{li2021fedrs}) change only the initialisation
> of this layer and bring in no information about the classes the client lacks.

**P2-3. sec1:67-74, contribution 3 (70 words of results; repeats the abstract almost word for word).** A
contribution names what was done; the results summary belongs in the abstract and §7. Proposed:
> \item A comparison against \R{count/methods_w} published methods on \R{count/datasets_w} datasets in two
> protocols, on the same client splits, with paired statistics, per-class metrics and the cost of every method
> (\cref{sec:results}). \claim{With different client models \ours has the highest balanced accuracy as trained, and
> most of its margin is in the last layer; with one shared model the strongest long-tail methods are better.}

**P2-4. sec4:75-94, the paragraph after Prop. 2.** Its order is hard to follow: the reparametrisation, a caveat, a
pointer to the ablation, then the readout, then minority collapse at the end. Proposed order and text:
> For a head trained on one client's real data, a logit shift is therefore a different initialisation of the
> last-layer bias. The body can still learn a different representation, because the shifted logits change the
> gradients that reach it, but the correction brings in no information about the classes the client lacks. Such a
> head is an extreme case of minority collapse~\cite{fang2021minority}: for the classes the client lacks, the
> imbalance ratio is infinite. The last layer can instead be computed from aggregated sums.
>
> [Prop. 3]
>
> The exact readout of \ours (\cref{sec:method-readout}) is such an estimator, and so are Fed3R, FedNCM and CCVR when
> computed over all clients. With it, every architecture's last layer equals the one computed centrally on the same
> body, so differences in accuracy with this readout come from the bodies. \claim{[readout gain sentence unchanged]}

The pointer to tab:components then becomes conditional (P1-3).

**P2-5. sec5:86-106, "Exact readout" (one paragraph with four purposes).** Split it into three paragraphs:
definition (to line 96), interpretation (97-101) and cost and role (101-106). Introduce the name at first use.
Proposed opening: "After Stage III, every architecture's last layer can be replaced by the *exact readout* (ER), a
class-balanced linear discriminant computed from securely aggregated sums." Proposed interpretation sentence (lines
99-101):
> Computed from the sums with shrinkage and a pseudo-inverse, it is a plug-in estimate of that classifier; when the
> embeddings are not Gaussian with a shared covariance, it is a linear classifier without an optimality guarantee.

Proposed last sentence:
> We report every method of the GeFL-F pipeline both as trained and with this readout; the results of \ours that
> answer RQ2 are those of its heads as trained.

**P2-6. sec7:165-179, one shared model.** The paragraph mixes results, an explanation that the text itself calls
untested, the readout and the extractor run. Keep the results and the untested explanation. Delete "The generator,
which is what lets clients with different models exchange knowledge, is not needed when the models can simply be
averaged." (line 175-176): it is speculative, uses "simply", and repeats sec8:31-32. The setup sentence "trains on
frozen features of images without augmentation" is already in sec6:162-163; replace it with
"(\cref{sec:setup-protocols})".

**P2-7. sec7:181-189, published numbers.** "which trains 300 rounds" (182-183) and "FedYoYo's evaluation trains 300
rounds and ours 200" (188) say the same thing; keep the second. The published values typed at line 184 (58.2, 68.1,
...) come from another paper, so they are not results of this paper. They are acceptable as citations, but they
should carry "(\cite{yan2025fedyoyo}, their Table N)" so that a reader can tell them apart from the slot values.

**P2-8. sec8:33-43, the GeFL's-own-setting bullet (140 words; repeats app:92-97).** It also contains an unsupported
attribution: "so on CIFAR-100 the level is set by the design and the data budget, and on CIFAR-10 the long tail,
together with the smaller training set it leaves, costs most of the difference". Proposed:
> \item The absolute accuracy of the main setting is low. GeFL's authors attribute GeFL-F's lower accuracy on
> CIFAR-10 to its feature extractor and target networks (their term for the heads)~\cite{kang2025gefl}; in GeFL's
> balanced setting our earlier implementation of GeFL-F reaches \R{repro/cifar10/GeFL-F/final}\% on CIFAR-10 and
> \R{repro/cifar100/GeFL-F/final}\% on CIFAR-100 (\cref{app:protocol}).\whennum{main/cifar100/Centralised (pooled
> data)/bal}{ The pooled-data reference reaches \R{main/cifar10/Centralised (pooled data)/bal} on CIFAR-10-LT and
> \R{main/cifar100/Centralised (pooled data)/bal} on CIFAR-100-LT.} These numbers are not comparable with centralised
> long-tail results for ResNet-32 and larger networks.

**P2-9. sec8:46-54, Conclusion.** The conclusion is weaker on the readout caveat than the abstract, which reads as a
stronger claim. It also contains filler ("Each can be described precisely") and repeats the code-release line of
sec1:76. Proposed:
> Generator-based model-heterogeneous federated learning loses rare classes in three places: coupled weight decay
> under Adam on clients that lack a class, a last layer fitted to one client's label prior, and generated features
> that do not carry their label. Each can be repaired with class statistics that secure aggregation computes up to
> its quantisation. \claim{With different client models, \ours has the highest balanced accuracy as trained, on
> CIFAR-10-LT mostly through the recall of the tail classes. Most of its margin is in the last layer: with the same
> exact readout for every method of the GeFL-F pipeline, the remaining margins are not significant with three seeds.
> With one model shared by all clients, the strongest long-tail methods remain better.}

**P2-10. sec6:8-23, RQ1.** The item restates the analysis ("The analysis names three places ..."). Proposed:
> \item \textbf{RQ1 (diagnosis).} Do the three losses of \cref{sec:analysis} occur in GeFL-F's own runs?

**P2-11. sec2:17-30, the second related-work paragraph.** One 69-word sentence (lines 18-23, four clauses joined by
semicolons) carries the readout positioning in the middle. Proposed split:
> Classifier calibration re-trains it on virtual features drawn from aggregated class statistics
> (CCVR~\cite{luo2021ccvr}) or on federated features learned on the server (CReFF~\cite{shang2022creff}). Closed-form
> classifiers are computed from aggregated sums (Fed3R~\cite{fani2024fed3r}, a ridge-regression classifier, and
> FedNCM~\cite{legate2023fedncm}, nearest class means); the exact readout of \ours is a class-balanced linear
> discriminant of the same kind, computed for each architecture. FedETF fixes the classifier to a simplex
> equiangular tight frame~\cite{li2023fedetf}.

**P2-12. Fig. 1 caption (sec5:11-17).** It repeats §5 step by step. Proposed:
> Overview of \ours. C: computed by each client; S: computed by the server; SA: summed over clients by secure
> aggregation. The class statistics released after Stage I give the class means and spreads and the gate $F$, which
> chooses the generator of Stage II. The moment correction is computed once after Stage II. In Stage III heads train
> on corrected generated features with consensus soft labels and on real features with logit adjustment. The exact
> readout (dashed) is an evaluation step.

The word "(dashed)" applies only if the figure draws it that way; otherwise drop it.

**P2-13. Caption consistency.**
- **Setting string.** Captions give it inconsistently: fig:recall-size, fig:hmt, fig:spread and tab:components omit
  α; tab:cost omits α. Use one string everywhere, "imbalance ratio 100, $K=10$, Dirichlet $\alpha=0.5$", or "Same
  setting as \cref{tab:main}".
- **Spread wording.** "mean ± standard deviation" (most), "mean ± one standard deviation" (fig:hmt) and "band one
  standard deviation" (fig:spread). Use "mean ± standard deviation over seeds 3-5"; for figures, "band: ± one
  standard deviation".
- **Dataset names.** Column headers say "CIFAR-10" while the text says "CIFAR-10-LT"; tab:datasets says
  "MNIST-LT". Use the -LT names in headers.
- **t-SNE caveat.** "Cluster sizes and distances ... not interpretable" appears in fig:tsne, fig:realgen and
  sec7:277-278. Keep it in the fig:tsne caption, and let fig:realgen say "same settings and caveats as
  \cref{fig:tsne}".
- **tab:tail** "(the 40% least frequent; %)" becomes "Recall (%) of the tail classes (the 40% least frequent)".
- **tab:fairness.** The caption "What each method of the main setting uses (CIFAR rounds; per-round epochs of Stage
  III)" is cryptic, and the Local row uses undefined $T_{\mathrm{FE}}, T_{\mathrm{TN}}$. Proposed:
  > Training budget and exchange of each method in the main setting: generator and its rounds (CIFAR), synthetic /
  > real epochs per Stage-III round, what is exchanged in Stage III beyond the averaging of heads of the same
  > architecture (which exchanges nothing at $K=10$), whether the exact readout is evaluated, and the loss of the real
  > phase. Stage I is shared by all federated methods.

  Define $T_{\mathrm{I}}, T_{\mathrm{III}}$ (rounds of Stages I and III) in §3 and use them in Algorithm 1 and in this
  table.
- **fig:hmt.** "Medium" is not defined. Write "head (30% most frequent), medium (the next 30%) and tail (40% least
  frequent) classes", or rename the groups (T-12).

---

## P3: sentence level

Each item gives location, current text, then the proposed text.

1. **abstract:8-11.** Current: "We propose FedECS-LT, which replaces each of these estimates by an exact class
   statistic computed under secure aggregation: ...; a class-balanced readout computed from aggregated sums serves to
   compare the bodies." "These estimates" has no antecedent, "bodies" is undefined, and "serves to" is vague. See
   the abstract proposal below.
2. **sec1:10-11.** Current: "Little is known about the two together, although most deployments that have one also
   have the other." This is unsupported. Proposed: "The two are rarely studied together: model-heterogeneous methods
   are evaluated without a global long tail~\cite{zhang2025htfllib}, and federated long-tail methods assume one
   shared model."
3. **sec1:25-27.** Current: "moves the generator's class-specific conditioning parameters toward zero at Adam's full
   step size". Proposed: "moves the generator's class row toward zero by about one learning rate per step"
   (matches Lemma 1).
4. **sec1:63-64.** Current: "with short exact results checked against measurements". Proposed: "with propositions
   (proofs in \cref{app:proofs}) checked against measurements". The same applies to **sec4:3**, "a short exact
   statement", which becomes "a proposition". "Exact" should mean only "equal to the pooled-data value" (T-1).
5. **sec1:56-57.** The built text reads "(CIFAR-10 and CIFAR-100-LT at imbalance ratio 100)", because
   `\R{t2/datasets}-LT` attaches "-LT" to the last name only. Proposed: "(long-tailed \R{t2/datasets} at imbalance
   ratio \R{t2/ratios}; ...)".
6. **sec3:25-26.** Current: "Every client's data therefore point at the centralised target of any class-conditional
   quantity". Proposed: "A class-conditional quantity, such as a class mean, a class row or a readout, therefore has
   the same target on every client as on the pooled data."
7. **sec3:16-17.** Current: "a linear projection whose $c$-th column, the \emph{conditioning row} $r_c$". This calls
   a column a row. Proposed: "a linear projection; its weights for class $c$, which we call the \emph{class row}
   $r_c$ (a row of the label-embedding table), are the only class-specific parameter."
8. **sec3:20-21 (Stage III).** "Body" is used from the abstract onwards but never defined. Add: "Each head consists
   of a \emph{body} of convolutional blocks, which maps $h$ to an embedding $e$, and a linear last layer."
9. **sec4:17.** Current: "The bound only describes the measured changes; the optimiser decides how large $\bar D_c$
   is." Proposed: "The bound only accounts for the measured norm changes; how large $\bar D_c$ is depends on the
   optimiser."
10. **sec4:35-37.** Current: "as long as the class appears more often than the estimate forgets." This is vague.
    Proposed, if this is the intended meaning: "as long as batches containing the class occur within the memory of
    that estimate, about $1/(1-\beta_2)$ steps."
11. **sec4:105-110.** One 56-word sentence. Proposed: "Alternating the synthetic phase with the real phase decides
    what survives. In an illustrative model with quadratic losses (the heads are nonlinear and trained with
    cross-entropy), the rounds converge to a point where the synthetic data decide the directions that the client's
    real loss leaves free (\cref{app:proofs})." The rest stays as in the working tree.
12. **sec5:3-5.** Current: "Wherever the data decide a class's result, it uses a statistic ...". Proposed: "For
    every quantity specific to one class, it uses a statistic that secure aggregation computes exactly instead of a
    parameter estimated by FedAvg."
13. **sec5:82-84.** Current: "this hides nothing beyond what FedAvg already reveals; it only makes the consensus one
    secure sum." "Hides" is the wrong verb. Proposed: "The server holds every head, so this exchange reveals nothing
    beyond what the averaging of heads already reveals; secure aggregation only turns the consensus into one sum."
14. **sec5:133-134.** Current: "and they still describe rare classes held by few samples: a class mean over a handful
    of images reveals more about them than a mean over thousands." Proposed: "for a class with few images, they
    reveal more about each image than a mean over thousands would."
15. **sec6:121-124.** "the referee classifier of the generator diagnostics" uses a term that is undefined at its
    first use. Proposed: "it trains the classifier that scores generated features in the diagnostics
    (\cref{app:protocol})". Add a definition there if none exists.
16. **sec6:171.** Current: "The methods fall into six groups, each chosen for one question." "References" answers no
    question. Proposed: "The methods fall into six groups."
17. **sec6:282-283.** Current: "GeFL~\cite{kang2025gefl} reports the mean over the ten models in the same way." GeFL
    reports accuracy, not balanced accuracy. Proposed: "GeFL~\cite{kang2025gefl} also averages its metric over the
    ten models."
18. **sec6:288-290.** Current: "which is the lowest class recall of the recall averaged over the models; the number
    of classes that no model recognises (pooled recall zero)". Proposed: "the lowest class recall after averaging
    recall over the models; the number of classes with zero recall for every model".
19. **sec6:316, 387; tab:shared caption.** Current: "came back", which is informal. Proposed: "settings with fewer
    completed seeds say so"; "the seeds that completed".
20. **sec6:367.** Straight quotes `"Best"` render as typewriter quotes. Use ``` ``Best'' ``` and
    ``` ``strongest other method'' ```.
21. **sec6:369-371.** Current: "they rest on the $t$ model ... A verdict near $p=0.05$ ...". Proposed: "they assume
    normally distributed paired differences ... A test result near $p=0.05$ ...".
22. **sec7:185.** Current: "Three methods are clearly below the published values". Proposed: "Three methods are
    further below the published values".
23. **sec7:16-18.** Current: "..., because nothing in them gives an architecture information about the classes its
    clients lack." This causal claim is untested. Proposed: "; none of them gives an architecture information about
    the classes its clients lack."
24. **sec7:25.** Current: "Balanced accuracy has a price on the clients' own class mixes." This is figurative.
    Proposed: "On the clients' own class mixes, \ours is below most other federated methods."
25. **sec7:59.** Current: "Most of GeFL-F's long-tail loss is in its last layer". "Long-tail loss" is not defined
    or measured. Proposed: "The exact readout raises GeFL-F's balanced accuracy by
    \R{main/cifar10/GeFL-F/gain_er} points on CIFAR-10-LT (\cref{sec:analysis-head})."
26. **sec7:147-149.** "the margins are below one point ... on BloodMNIST FedDF is level with \ours" are typed
    results with no slot. Add slots (for example `nat/er_margin_max`, `nat/er_level_list`) in `fill.py`, or drop the
    sentence apart from the noise-floor reference.
27. **sec7:171.** The slot output reads "(3 and 4 on CIFAR-10-LT, 4 on CIFAR-100-LT; ...)". Proposed text before the
    slot: "(seeds \R{t2/cifar10-r100/margin_seeds} on CIFAR-10-LT and \R{t2/cifar100-r100/margin_seeds} on
    CIFAR-100-LT; ...)".
28. **sec7:278.** Current: "carry no meaning~\cite{wattenberg2016tsne}; only the grouping does." Delete the sentence
    (it is in the fig:tsne caption; see P2-13).
29. **sec7:306-308.** Current: "\R{main/cifar10/strongest}, the strongest other method". In a sentence about
    calibration this reads as "best calibrated". Proposed: "the method with the next-highest balanced accuracy".
30. **sec8:14.** Current: "Three seeds per setting give fragile intervals". If the bullet is kept, proposed: "Three
    seeds per setting give wide intervals and tests with little power (\cref{sec:setup-stats})."
31. **sec8:13.** Current: "pure label skew". §3 calls this "pure label shift". Use "label shift" (T-9).
32. **abstract:12-14.** Current: "42.8 ± 2.0 balanced accuracy". The unit is missing. Proposed: "a balanced accuracy
    of \R{...}\,\%".
33. **declarations.tex:20.** Current: "in order to". Proposed: "to".

---

## Terminology: one name per concept

| # | Variants in v03 (locations) | Proposal |
|---|---|---|
| T-1 | "exact" for SA statistics, for the readout's name, for propositions ("short exact results" sec1:64, "exact statement" sec4:3, "exactly the trajectory" sec4:68) | "exact" only for "equal to the pooled-data value up to quantisation" (defined sec5:32-33); "proposition" for the mathematics |
| T-2 | exact readout / class-balanced readout (abstract:11, tab:readout) / class-balanced linear readout (sec1:44) / class-balanced linear discriminant (sec2:21, sec4:88, sec5:92) / ER / statistics-based linear classifier (sec5:100); code: LDA | **exact readout (ER)** for the procedure, "class-balanced linear discriminant" once as its definition (as in C-19) |
| T-3 | conditioning row (sec3:17, sec4:6, 39, fig:rows) / class rows (fig:rows caption, sec5, sec6, app:55) / class-specific conditioning parameters (sec1:26) / class-specific parameters (abstract:6, sec1:41) / "class-row generator" (sec7:150) | **class row $r_c$**; "class-specific parameters" only in the abstract, where the term is not yet defined |
| T-4 | Stage I/II/III / head stage (app:131, sec6:199) / head round (app:87) / head-training round (fig:spread) / $T_{\mathrm{FE}}$, $T_{\mathrm{TN}}$ (undefined; alg:fedecs, tab:fairness) | **Stage I, II, III**; "Stage-III round"; define $T_{\mathrm{I}}, T_{\mathrm{II}}, T_{\mathrm{III}}$ in §3 |
| T-5 | head / body / last layer / client model (sec1:69, sec6:16, 282, sec7:13) / head architecture / architecture / target networks (sec8:34) / personal model | **architecture** $g$ = design; **head** = body + last layer (define in §3); **model of architecture $g$** = extractor + head of $g$; avoid "client model" except where $K=10$ makes it the same thing |
| T-6 | GeFL-F pipeline (7 times) / GeFL-F family (sec7:60, sec8:3) | **GeFL-F pipeline** |
| T-7 | strongest other method (6) / strongest baseline (sec7:261, 267, sec8:5) | **strongest other method** (defined sec6:367) |
| T-8 | generated features (12) / synthetic features (abstract:7, sec5:15, 70, 80, 119, sec8:49) / synthetic data | **generated features**; keep "synthetic phase" for the training phase |
| T-9 | pure label shift (sec1:20, sec3:22) / pure label skew (sec8:13) / label skew / long-tailed label skew | **label shift** for the $P_k(x\mid y)=P(x\mid y)$ property, **label skew** for the Dirichlet split |
| T-10 | consensus soft labels / consensus labels (tab:hyper, tab:fairness) / consensus predictions (alg:118) / soft targets (alg:119) | **consensus soft labels** |
| T-11 | label prior / client's prior (sec8:49) / label frequencies (sec2:23) / class frequencies (sec5:73) | **label prior $\pi_k$** (estimated by smoothed class frequencies $\hat\pi_k$) |
| T-12 | "head classes" (sec7:101, 105, fig:hmt) vs "head" = network | rename the groups **frequent / medium / tail classes**, or say "head-class" each time; "head" already means the network |
| T-13 | main setting (9) / main comparison (4) / model-heterogeneous setting (sec1:54) / "different client models" protocol | protocols: **different client models** and **one shared model**; **main setting** = the different-models protocol at $\rho=100$, $K=10$, $\alpha=0.5$ (define once in §6.5) |
| T-14 | long-tailed benchmarks (5 datasets in §6.3, 2 in the results) / CIFAR benchmarks (sec1:72) | **long-tailed benchmarks**, generated from the same slot as the results (P1-10) |
| T-15 | pooled-data reference / Centralised (pooled data) (slot key, row label) / centralised | **pooled-data reference** in text and table rows |
| T-16 | spread $V_c$ (a variance) / logit spread (a standard deviation, fig:spread caption in the working tree) | keep "spread" for $V_c$; define "logit spread" in §4.3 at its first use, not only in a caption |
| T-17 | FedLTLib / LongTailLib / "LongTailLib/FedLTLib" | **FedLTLib** for the paper and protocol, **LongTailLib** for the code; say so once (sec1:54) |

Symbol clashes still open from C-19 in v03:
- $G$: number of architectures, the generator $G(h\mid y)$, $\bar G_c$, and $G_c$ in Prop. 7.
- $\alpha$: Dirichlet concentration and the FedRS factor; both are 0.5 in sec6:116 and sec6:204.
- $\rho$: imbalance ratio and $\rho_{r,i}$.
- $\omega$: consensus weight and $\omega_i$.
- $e$: one-hot $e_y$ and embedding $e$.
- $T$: statistic in Prop. 3, map in Prop. 8, and $T_s, T_r$.
- $\lambda$: weight decay and eigenvalue (working tree).
- $\tau$: local steps and the logit-adjustment strength. The working tree has already renamed local steps to $q_k$.

---

## Abstract (Elsevier ≤ 250; v03 is 258 words in the built PDF)

Problems with the current abstract:
- It is over the limit.
- "these estimates" has no antecedent, and "bodies" is undefined.
- "serves to compare" is vague, and the % unit is missing.
- "It also has" has an ambiguous antecedent.
- The last sentence is a method detail.

Proposed (243 words with slot values expanded; same slots):

> Federated learning across clients that run different model architectures and hold long-tailed, skewed label
> distributions loses most of its accuracy on rare classes. We study generator-based model-heterogeneous federated
> learning, in which clients share a frozen feature extractor and a federated conditional feature generator, and each
> architecture trains its own head. We trace the loss to three places: under Adam, coupled weight decay shrinks the
> generator's class-specific parameters on clients that lack the class; each head's last layer fits one client's label
> prior; and generated features that do not carry their label penalise the spread of the logits. We propose FedECS-LT,
> which replaces the failing quantities by class statistics that secure aggregation computes exactly: a generator
> anchored to the class means (class-count weighted averaging when classes are poorly separated), moment correction
> of generated features, and logit-adjusted heads trained on consensus soft labels. With ten client architectures, the
> heads of FedECS-LT as trained reach a balanced accuracy of \R{main/cifar10/FedECS-LT/bal}\,\% on CIFAR-10-LT and
> \R{main/cifar100/FedECS-LT/bal}\,\% on CIFAR-100-LT, \R{main/cifar10/margin} and \R{main/cifar100/margin} points
> above the strongest of \R{main/nbaselines_w} other methods (significant on CIFAR-100-LT only). When every method of
> the GeFL-F pipeline uses the same exact readout, a class-balanced linear classifier computed from aggregated sums,
> the margins shrink to \R{main/cifar10/er_margin} and \R{main/cifar100/er_margin} points and are not significant.
> FedECS-LT also has the highest mean balanced accuracy on \R{nat/nbest_of} datasets with natural class imbalance
> (significant on \R{nat/n_sig}). With one shared model, the strongest long-tail methods are better than FedECS-LT.

If space allows, keep the significance wording paired: "(paired tests over three seeds: significant on CIFAR-100-LT,
not on CIFAR-10-LT)" costs 9 more words and still fits.

## Highlights (3-5, ≤ 85 characters)

Problems with the current highlights:
- H1 "erases" overstates the evidence: the rows are several times shorter, not zero.
- H2 "fix generator, heads and readout" overstates the effect, and the readout is an evaluation step, not a repair.
- H3 "recovers most of the long-tail loss" refers to a quantity that is neither defined nor measured.
- H4's "21" is typed and must match `count/methods_w`.

Proposed (character counts checked):

1. `Coupled weight decay in Adam shrinks rare-class parameters of federated generators.` (83)
2. `FedECS-LT uses class statistics from secure aggregation in the generator and heads.` (83)
3. `With ten client architectures, FedECS-LT has the best balanced accuracy as trained.` (83)
4. `With one exact readout for every method, the margins shrink and are not significant.` (84)
5. `Compared with 21 published methods in Flower, in two federated learning protocols.` (82); alternatively,
   `With one shared model, the strongest long-tail methods are better than FedECS-LT.` (81)

H3 is true only while `main/nbest_of` and `nat/nbest_of` are "both" or "all"; re-check it whenever the results
change.

---

## Words saved (main text)

| Item | Approximate words |
|---|---|
| P1-2 merge Threats and Limitations; delete "What the gains consist of" | 460 |
| P1-5 repeated pipeline, component, readout-role and novelty statements | 260 |
| P1-6 §6 material to the appendix or deleted as duplicate | 450 |
| P2-1/2/3 introduction | 130 |
| P2-6/7/8/9 results, Limitations, Conclusion | 170 |
| P1-7 floats (tab:rq, tab:positioning or its columns, fig:forest; fig:hmt, fig:realgen, fig:shared-rounds to the appendix) | 3-4 pages |
| **Total** | **about 1,450 words of text (about 4 Elsevier pages) plus 3-4 pages of floats** |

No result, slot or piece of information is lost. Everything cut is either a duplicate or moved to the appendix.
