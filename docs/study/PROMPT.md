# Prompt for the FedECS-LT study text

### A1. Role, reader and goal
You are writing a study text for one reader: the first author of a research paper on federated learning, who must
submit and defend the paper but did not do the research.
- **What the reader knows:** machine-learning basics (neural networks and their layers, SGD and Adam, weight decay,
  cross-entropy, softmax and logits, train/validation/test splits, overfitting). They have heard of federated
  learning.
- **What the reader does not know:** this project; generator-based model-heterogeneous FL; long-tail methods;
  secure aggregation; linear discriminant analysis.
- **The goal.** After one careful reading the reader can:
  1. state the problem and why it matters;
  2. describe the base method GeFL-F and where it fails, with the evidence;
  3. explain every component of FedECS-LT, why it exists, how it works, what it costs, and its formula;
  4. explain the study design and why each choice was made;
  5. read every table and figure and state what it shows and what it does not;
  6. state the limitations honestly;
  7. answer the hardest reviewer questions from the evidence.

### A2. Sources (use only these, never invent anything)
- **The paper:** `docs/papers/content/*.tex` (sections 1–8, appendix, `refs.bib`) and the built PDF
  `docs/papers/elsevier/main_final.pdf`.
  - Every number comes from `docs/papers/generated/numbers.tex` (the `\R{key}` slots) and
    `docs/papers/generated/tables/*.tex`.
  - Figures are `docs/papers/generated/figures/*.pdf`.
- **The method's beginner guide:** `docs/guide/how_our_method_works.html`. Use its intuitions and tiny examples,
  but **only for components the paper still has**: kernel herding, for example, is not in the paper; leave it out.
  Where the guide and the paper differ, the paper wins.
- **The code**, `fedecs_lt/src/*.py` (`server.py`, `client.py`, `fedecs.py`, `runner.py`, `settings.py`): only to
  answer "how exactly is this done" when the paper is silent. Cite the function when you do.
- **The reviews and decisions:** `docs/papers/review/final_review.md`, `decision_log.md`, `issue_ledger.md`,
  `reviews/cycle1/E_critical.md`, `reviews/cycle2/G_final_review.md`, `reviews/cycle2/I_verification.md`.
- **The run plan:** `docs/review/action_plan.md` (sections 24–26), for which experiments are pending and why.

Rules for sources:
- If two sources disagree, follow the paper; if the paper is silent, follow the code; say so in one sentence.
- Never add a number, citation, dataset, method or result that is not in these sources.
- Separate three kinds of statement and mark them in the wording:
  - **measured:** "the runs show …";
  - **proved:** "Lemma 3 states …";
  - **interpreted:** "this suggests …".

### A3. The line of argument (chapter by chapter, in this order)
The text is one argument. Each chapter exists because the previous one raised a question. Every chapter ends with
**one** sentence that states the question the next chapter answers. That is the only "bridge" allowed: no
summaries.

Running example, used from Chapter 2 onward and returned to in every chapter:
- **The setting:** CIFAR-10-LT, imbalance ratio 100, 10 clients, one per head architecture (CNN-1 … CNN-10),
  Dirichlet α = 0.5.
- **Its two classes:** one frequent class ("class 0") and one rare class ("class 9"). Use the realised class sizes
  from `tab:datasets` and the client counts from the paper/code where they exist; otherwise say "for illustration"
  and keep the numbers simple.

**Chapter 1. The problem**
- **Goal:** why rare classes are lost when clients have different models and long-tailed, skewed labels, and how
  the loss is measured.
- **Content:**
  - federated learning in one paragraph (clients keep data, a server combines what they send);
  - why different architectures stop FedAvg from working;
  - what label skew and a global long tail are (the Dirichlet split and the imbalance ratio, with the realised
    ratio caveat);
  - why the rarest classes are held by few clients in few copies;
  - why accuracy hides this and balanced accuracy does not (worked example: a model that ignores the rare class);
  - the two protocols the paper uses (different client models; one shared model) and why both.
- **Paper artefacts:** sec1 (first two paragraphs), sec3 "Clients and data", sec6 "Metrics", `tab:datasets`.
- **Ends with:** how do clients with different models share knowledge at all?

**Chapter 2. The base system: GeFL-F**
- **Goal:** the reader can draw GeFL-F's three stages and say what every client and the server compute in each.
- **Content:**
  - **Stage I:** a small shared extractor trained by FedAvg, then frozen; why freeze it; what a feature h = φ(x)
    is.
  - **Stage II:** the conditional VAE generator G(h | y) trained by FedAvg with Adam and coupled weight decay; the
    conditioning row r_c (the generator's only class-specific parameter); why a generator lets an architecture
    learn about classes its clients lack.
  - **Stage III:** each client trains its own head T_s epochs on generated features with uniform labels, then T_r
    epochs on real features; heads of one architecture are averaged.
  - With K = 10 and one client per architecture, nothing is averaged in Stage III; say this now, because it matters
    in Chapters 7 and 9.
- **Paper artefacts:** sec3 "The pipeline of GeFL-F", sec2 first paragraph (where GeFL-F sits among
  model-heterogeneous methods), appendix "Models".
- **Diagram:** a simple three-stage diagram (TikZ) with what flows between clients and server.
- **Ends with:** under a long tail GeFL-F loses most rare-class accuracy; where exactly?

**Chapter 3. Why federation should not bias class-level quantities, and how it still can**
- **Goal:** the key fact behind the whole project, and the four routes by which bias still enters.
- **Content:**
  - A client split is a pure label shift. P_k(x | y) = P(x | y), so every client's class-c samples point at the
    same target.
  - A sum over clients is exact and partition-invariant (Lemma "partition invariance"), so any estimator built from
    such sums equals the centralised one. Worked example: class means from per-client counts and sums.
  - Secure aggregation: what it reveals (the sum) and hides (each client's vector); quantisation and clipping
    bounds; the three-client minimum.
  - The four routes of bias: the optimiser, parameters tied to a client's prior, what the synthetic data teach, and
    sample size. The list is examined, not claimed complete.
- **Paper artefacts:** sec3 "A client split is a pure label shift", "Secure aggregation"; Lemma invariance (sec4);
  sec5 "Statistics release" (the bounds).
- **Ends with:** which of these routes actually hurt GeFL-F?

**Chapter 4. Where GeFL-F loses the rare classes (the diagnosis)**
One subsection per cause, each in the same order: intuition with the running example → the precise statement →
what was measured → how strong the evidence is.

- **4.1 The generator: coupled weight decay under Adam.**
  - **This is a hard point and needs full care.** With coupled L2, a client without class c sees gradient λ r_c
    on that class's row.
  - Adam normalises gradient scale, so each such step moves the row about one learning rate η toward zero
    whatever λ is. Show this with a two-step numeric example of Adam's update.
  - The holders' data gradient need not grow the norm. Flat averaging then contracts the row (Lemma
    "contraction", the condition p_c < D̄/(Ḡ+D̄)).
  - The step-count heuristic κ_c (Eq. kappa) is a heuristic, not checked against data.
  - Decoupled decay (AdamW) and CCWA remove the effect.
  - **Evidence:** Fig. rows (rare rows shorter), and the fidelity table, in which GeFL-F's generated features are
    near chance for every class.
  - **State plainly:** the shrinkage is measured, its causal role is not. The pending "GeFL-F + AdamW" run tests
    it.
- **4.2 The last layer: logit corrections are reparametrisations.**
  - **Hard point.** Each head's last layer fits its one client's prior.
  - Lemma "reparametrisation": training with ℓ(f + a_k, y) equals plain training from bias b_0 + a_k followed by
    subtracting a_k at test time, which is post-hoc logit adjustment.
  - So logit adjustment, FedLC and Balanced Softmax add no information about absent classes. FedRS is the same up
    to a smaller step on absent rows.
  - Worked numeric example with 3 classes.
  - **Consequence:** the last layer needs information from other clients. This motivates the exact readout.
  - **Evidence:** the readout alone raises GeFL-F by the amounts in sec4 (slots `main/<ds>/GeFL-F/gain_er`).
  - **Evidence:** the three loss corrections become equal once the last layer is replaced (`tab:readout`).
- **4.3 The synthetic data: label-free features penalise the logit spread.**
  - Lemma "label-free synthetic data": with uniform labels independent of the feature, the expected loss is
    log C plus a term in the spread of the logits. So training on such features pulls the logits together instead
    of teaching classes.
  - The alternation argument (appendix lemma) in words.
  - **Evidence:** Fig. spread, including the caveat that FedDF, on GeFL-F's generator, loses less spread than
    FedECS-LT. So the label explanation is not the whole story.
- **Paper artefacts:** sec4 entirely, appendix proofs, `fig:rows`, `fig:spread`, `tab:fidelity`, sec1 item list.
- **Ends with:** what would a method look like that uses exact statistics exactly where these losses occur?

**Chapter 5. FedECS-LT, component by component**
- **Goal:** every component as an answer to one cause, in pipeline order.
- **For each component give:**
  - the problem it answers (pointing back to Chapter 4);
  - how it works, in words;
  - the formula, with every symbol defined at first use;
  - a tiny example with numbers;
  - what is sent and what it costs;
  - what the paper's evidence says about it, including "not yet measured" where true.
- **Components:**
  1. **Statistics release:** n_kc, ΣH, Σ‖h‖² → μ_c, V_c (Eq. moments); C(d+2) numbers once.
  2. **Separation gate:** the Fisher-type ratio F (Eq. gate), F_0 = 0.05 fixed in development; the measured range
     of F and where class rows were chosen (slots `gate/*`). Why an anchor fails when means are close (appendix
     lemma "what a mean anchor can express").
  3. **Mean-anchored generator:** Eq. anchor; no class-specific parameter, so nothing for the decay to shrink.
     **Class rows with CCWA**, the fallback: Eq. ccwa with effective numbers; zero weight for non-holders.
  4. **Moment correction (hard point):** Eq. mc; least-displacement map (appendix lemma); ReLU and clipping make it
     approximate; measured spread ratio (`main/mc_spread_*`). **Its fidelity result:** it raises head-class
     fidelity and lowers tail fidelity on CIFAR-10-LT, so its value for accuracy rests on the pending component
     ablation.
  5. **Heads:**
     - logit adjustment on real data (each client's own prior; not a secure statistic);
     - consensus soft labels (Eq. csl; ω = 0.5; one secure sum; what it reveals);
     - T_s = 10 synthetic epochs (compute, not statistics).
  6. **Exact readout (the hardest and most important; give it the most space).**
     - After Stage III every client sends, per architecture, counts, Σe and Σee^T of its embeddings.
     - The server forms class means ν_c and the pooled covariance Σ, shrinks it (γ = 0.1), and sets the last layer
       to the class-balanced LDA (Eq. er).
     - Explain LDA from first principles: Gaussian classes with a shared covariance, equal priors, why the decision
       rule is linear, and what shrinkage does. Build it up for 2 classes in 2 dimensions with numbers.
     - It equals the centralised estimator on the same bodies (partition invariance). It is a plug-in estimate,
       not optimal in general.
     - Cost: C(d_e+1)+d_e² numbers, one extra round; every client runs every body.
     - Where it helps and where it hurts: one shared model (−3.0 on CIFAR-10-LT); OrganAMNIST; own-class-mix
       accuracy.
     - It is a known family (Fed3R, FedNCM, FedCOF). What is new is applying it per architecture with statistics
       pooled across architectures.
- **The whole recipe:** Algorithm 1 walked line by line, and what is sent when (overview figure `fig:overview`).
- **Cost and privacy:** what the server and the clients learn; the shallow-feature caveat; no differential privacy.
- **Paper artefacts:** sec5 entirely, `alg:fedecs`, `fig:overview`, `tab:hyper`, sec3 "What every client must
  run", appendix lemmas w1 and mc.
- **Ends with:** how do we test whether this works, and against what?

**Chapter 6. The study design**
- **Goal:** the reader can justify every experimental choice.
- **Content:**
  - **The research questions RQ1–RQ5,** each mapped to the experiment and the section that answers it.
  - **Datasets:** the long-tailed benchmarks (target ratio 100 vs the realised ratio) and the six naturally
    imbalanced datasets (never used in development).
  - **Splits:** Dirichlet α = 0.5, seeds 3–5, identical for every method.
  - **Protocols:** the main setting with its rounds T_I/T_II/T_III, and the one-shared-model FedLTLib/CReFF protocol
    with how FedECS-LT is fitted into it.
  - **The method groups and why each is there:** GeFL-F; FedDF; per-client long-tail losses; model-heterogeneous
    methods from HtFLlib; GeFL's grouped baselines; references.
  - **Fairness of the comparison** (`tab:fairness`): what each method exchanges, its synthetic epochs, and who gets
    the readout.
  - **Metrics:** balanced accuracy as the mean over 10 models of mean recall; worst-class; tail; own-mix; ECE;
    GeFL's best-round measure.
  - **Statistics (hard point):**
    - paired differences over seeds and t intervals;
    - Holm's correction and the test family;
    - why three seeds give little power, including the permutation-floor fact;
    - the GPU noise floor from the repeated BloodMNIST notebook, and why exceeding it is necessary, not
      sufficient.
  - **Development disclosure:** about 36 / 17 configurations on CIFAR-10-LT / CIFAR-100-LT, the same test sets, and
    why the fresh-seed run matters.
  - **Threats to validity.**
- **Paper artefacts:** sec6 entirely, `tab:hyper`, `tab:datasets`, `tab:baselines`, `tab:fairness`, appendix
  "Experimental details".
- **Ends with:** what do the results show?

**Chapter 7. The results, read table by table**
For every table and figure, in the paper's order, give:
1. the question it answers;
2. how to read it (rows, columns, the meaning of bold and ±, the seeds);
3. the key numbers, through slots;
4. what it shows;
5. what it does not show.

- **Main comparison** (`tab:main`, `tab:paired`):
  - FedECS-LT with its readout vs every method without it: +15.1 / +8.6 over FedDF, Holm p.
  - The heads alone: +5.2 / +4.3.
  - Why most of the margin is the readout's.
- **The same readout for every method** (`tab:readout`, `tab:app-er-paired`): +1.4 / +1.5, not significant; +2.3 /
  +3.8 over GeFL-F. Explain why this is the fair comparison of the bodies, and why the authors report both.
- **Other model-heterogeneous methods:** close to Local; best in the first rounds and then declining; the port
  caveat.
- **Own class mixes** (`tab:app-local`): the cost of balanced accuracy and of the readout.
- **Rare classes** (`tab:tail`, `fig:recall-size`, `fig:hmt`): where the gain sits; with the readout the tail gain
  over the loss-only methods vanishes.
- **Natural imbalance** (`tab:natural`, `tab:app-nat-er`): margins without and with the readout (−0.03 to +0.9);
  EMNIST has one seed.
- **One shared model** (`tab:shared`, `fig:shared-rounds`): six methods better on CIFAR-10-LT, FedAvg among them;
  the readout lowers FedECS-LT; why the generator is not needed when models can be averaged; the comparison with
  published numbers.
- **Representations** (`fig:tsne`, 10-NN): modest differences; what t-SNE can and cannot show.
- **Calibration, fairness across architectures, cost** (`tab:cost`): 185× the communication of Balanced Softmax.
- **Pending experiments,** whose text is already written:
  - several clients per architecture (`tab:k50`, `fig:clients`);
  - fresh seeds 6–8;
  - GeFL-F variants (`tab:ablation`, `fig:variants`);
  - components (`tab:components`, `fig:components`);
  - other settings (`tab:robust`).

  For each: the question it answers, what result would support or weaken the paper, and how the text changes.
- **Ends with:** what follows, and what does not?

**Chapter 8. What the evidence supports, and its limits**
- One paragraph on what is established, with its scope (one client per architecture, three seeds, a small frozen
  extractor).
- The limitations of sec8, each with its consequence.
- The trade-offs: own-mix accuracy, cost, the condition that every client can evaluate every head, privacy of
  class means.

**Chapter 9. Defending the paper**
- The 12 hardest questions a reviewer will ask, taken from the reviews (E: R1–R5; G: M1–M7; verifier I's FAIL and
  PARTIAL items).
- For each: the question in the reviewer's words; the honest answer; the evidence (section, table, number); and
  what experiment would settle it, with its status (pending notebook or not planned).
- **Mandatory questions:**
  - "Isn't the gain just the readout?"
  - "Is the decay diagnosis causal?"
  - "Were the settings tuned on the test sets?"
  - "Does it hold with several clients per architecture?"
  - "Why is it worse than FedAvg with one shared model?"
  - "Is it worth 185× the communication?"
  - "What do the released statistics leak?"
  - "Why are the absolute accuracies so low?"
  - "How is this different from FedCOF / Fed3R / CCVR?"
  - "Why only three seeds?"

### A4. Depth rules
- **Be brief on what the reader knows:** softmax, SGD, what a split or FedAvg is. One sentence each, at most.
- **Give full treatment** (intuition → formal statement → a small worked example with numbers → what it implies) to
  these points:
  - Adam with coupled decay on non-holders;
  - the contraction condition;
  - logit corrections as reparametrisations and post-hoc logit adjustment;
  - the label-free loss expansion;
  - partition invariance and secure aggregation;
  - the separation gate;
  - the moment correction;
  - class-balanced LDA with shrinkage (the readout);
  - consensus soft labels;
  - paired tests with Holm's correction and the noise floor.
- **Proofs:** give the idea of each lemma's proof in two to five lines. Point to the appendix of the paper for the
  full proof only when it is longer.
- **Never explain the same thing twice.** Refer back instead ("the readout of Section 5.6").
- Every symbol is defined at first use and keeps one meaning. Where the paper reuses a symbol (G for the generator
  and for the number of architectures; ρ; α; ω; u), say so once and keep them apart.

### A5. Style rules
- Plain, exact English. Short sentences. Active voice. One idea per paragraph.
- **One term per concept, used identically everywhere:** "class row", "exact readout", "heads as trained",
  "without the readout", "secure aggregation", "balanced accuracy".
- **Forbidden:**
  - filler: "it is important to note", "in this section we", "as mentioned", "interestingly", "notably",
    "furthermore", "moreover", "in order to";
  - promotional words: "novel", "crucial", "pivotal", "robust" (about results), "comprehensive", "leverage",
    "delve", "seamless", "significant" outside statistics;
  - em dashes;
  - rhetorical questions inside chapters (the one closing question per chapter is the exception);
  - summaries that repeat the section;
  - motivational remarks.
- **Every sentence explains, defines, argues or reports.** Delete any that does none of these.
- **Mathematics:**
  - LaTeX, numbered only when referred to later;
  - after each displayed equation, one sentence in words on what it says;
  - a worked example right after each hard formula.
- **Numbers:**
  - exactly as in the paper, with ± and the number of seeds;
  - never rounded differently, never extrapolated;
  - a difference is called significant only if the paper says so after Holm's correction;
  - a difference below the noise floor is called unresolved.
- **Figures:**
  - reuse the paper's figure files;
  - add a TikZ diagram only for Chapter 2 (pipeline), Chapter 4.1 (row shrinkage over rounds, illustrative) and
    Chapter 5.6 (the LDA decision boundary in 2-D, illustrative);
  - mark illustrative diagrams as such;
  - refer to every figure in the text before it appears;
  - every caption says what to look at.

### A6. Format
- **LaTeX compiled to PDF:** single column, 11 pt, book-like margins, numbered chapters and sections.
- **Leave out:** the title page (one title line instead), the table of contents, the abstract, the list of figures,
  the acknowledgements, any preface.
- **Cross-references:** tables and figures numbered and referred to by number, plus "Table 3 of the paper" where a
  paper table is meant.
- **Length:** whatever the content needs, likely 45–70 pages. Nothing is padded; nothing in A3 is skipped.

### A7. Self-check before finishing
1. **Read the whole text once as the reader.**
   - No chapter uses a term or result introduced later.
   - Every chapter ends with exactly one forward question.
   - No paragraph repeats an earlier one.
2. **Every number is checked against `numbers.tex` or the tables.** Every "measured", "proved" and "interpreted"
   statement is the right kind.
3. **Every forbidden word and every em dash is removed.**
4. **Every pending experiment is described as pending,** never as a result.
5. **Chapter 9 answers every mandatory question.**
