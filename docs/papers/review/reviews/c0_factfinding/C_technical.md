# Agent C: technical review (fact finding, cycle 0)

Scope: `docs/papers/content/{abstract,sec1_intro,sec3_setting,sec4_analysis,sec5_method,sec6_setup,appendix}.tex`
(plus the sentences of `sec7_results.tex` / `sec8_discussion.tex` that describe procedure), checked against the
code that produced the results, `fedecs_lt/src/` (code version 79ef95114493 in the returned results). Read-only review;
nothing in the paper or code was changed. Line numbers are those of the files as of 28 Oct 2026.

Severity: P0 = a statement in the paper is false about what was done; P1 = major technical error or missing
essential detail; P2 = significant clarity problem or missing detail; P3 = minor.
Results are referred to only as slots; where returned runs were inspected to confirm a code path, no number is
proposed for the text.

---------------------------------------------------------------------------------------------------------------------

## Summary table

| id | sev | where (tex) | one line |
|----|-----|-------------|----------|
| C-1 | P0 | abstract.tex:14-15, sec1_intro.tex:69-71, sec7_results.tex:133, sec5_method.tex:99 | "every method is given the same exact readout" is false: only the GeFL-F-pipeline methods get it |
| C-2 | P0 | sec6_setup.tex:98-100 | worst-class recall is NOT computed per model and averaged; it is the min of the architecture-averaged recall |
| C-3 | P0 | sec6_setup.tex:54 vs 58-61, sec7_results.tex:166-169 | "Hyperparameters are those of each method's paper" is false for HtFLlib, LongTailLib and FedDF |
| C-4 | P0 | sec6_setup.tex:117-118, appendix.tex:74-77 | the Flower code was checked against the authors' earlier implementation (experiments/core.py), not "the original GeFL-F implementation" |
| C-5 | P1 | sec6_setup.tex:69-71 (tab:baselines) | Balanced Softmax, FedLC, FedRS run without any generator (GeFL-F w/o generator + loss); not disclosed |
| C-6 | P1 | sec3_setting.tex:18-19, sec6_setup.tex:135, tab:hyper | GeFL-F (and FedDF) use T_s = 1, FedECS-LT T_s = 10; the paper never states GeFL-F's T_s |
| C-7 | P1 | abstract.tex:9-12, alg:fedecs line 116, sec5_method.tex:99 | method is defined to end with the exact readout, but the headline FedECS-LT number is the as-trained heads (no readout) |
| C-8 | P1 | sec4_analysis.tex:69-71, sec1_intro.tex:29-31 | FedRS: "larger learning rate" is wrong (alpha = 0.5 gives alpha^2 = 0.25 times the step); FedRS is not a shifted-bias reparametrisation |
| C-9 | P2 | appendix.tex:14-20, sec4_analysis.tex:29-36 | step-ratio derivation assumes |m_hat/sqrt(v_hat)| <= 1, false for Adam with sparse gradients (bound (1-b1)/sqrt(1-b2) = 3.16) |
| C-10 | P2 | sec5_method.tex:38-39, appendix.tex:47-54 | "can separate classes only as far as their means are apart" is stronger than Prop. w1 (factor 1+L||A|| unbounded); also before moment correction only |
| C-11 | P2 | sec4_analysis.tex:104-108, appendix.tex:35-45 | alternation result: needs 0 <= eta*lambda <= 1; "directions the real loss leaves free ... are the directions of the absent classes" unproved and false for CE |
| C-12 | P2 | sec6_setup.tex:13-15 | validation set also sets the SecAgg+ public bounds; sizes are min(200, 2000//C), not "200, 20 on CIFAR-100" |
| C-13 | P2 | sec5_method.tex:78-80, 124-125 | "no client's prediction is revealed": the server already holds every head and the generator and can recompute every prediction; clients receive all exact class means |
| C-14 | P2 | sec6_setup.tex:103-104, sec7 tab:cost caption | wall-clock time measured with two runs sharing one T4; secure-sum bytes counted as 8 B/element, not protocol traffic; FLOPs are computed, not measured |
| C-15 | P2 | sec6_setup.tex:38-48 | Track 2 details missing: T_KA = 100, Stage I 10 local epochs, LR decay continues across stages, CSL degenerates to self-distillation, heads on non-augmented frozen features |
| C-16 | P2 | sec6_setup.tex:72, 58-61 | FedDF and HtFLlib ports deviate from the original methods in ways a reader must know |
| C-17 | P2 | sec5_method.tex:69-73 | "T_s = 10 epochs on synthetic features" undefined: it is 10 x floor(n_k/B) batches of fresh samples from one shared pool |
| C-18 | P2 | sec1_intro.tex:57-58 | "Every result is the mean over three seeds" contradicts sec6_setup.tex:109 and the Track-2 table |
| C-19 | P2 | notation (several files) | beta, m_c, G, tau, alpha, rho, H, e, S reused for different objects; W_1, G_c, LSE, body, T_FE/T_KA/T_TN undefined |
| C-20 | P2 | appendix.tex:79-84, sec8_discussion.tex:29-31 vs sec1_intro.tex:52 | GeFL's-own-setting numbers come from the earlier implementation, not the Flower code that "re-implemented every comparison" |
| C-21 | P2 | sec8_discussion.tex:33-36 | pooled reference is not "the same extractor" and not a "ceiling" (plain CE on long-tailed pooled data) |
| C-22 | P3 | sec6_setup.tex:33-35 | "ten CNN head architectures of increasing depth and width" is false (CNN-4, CNN-7, CNN-10 are narrower than their predecessors) |
| C-23 | P3 | sec6_setup.tex:8-10 | CIFAR-100-LT is capped too (largest class 480 < 504 asked), so its ratio is also below 100 |
| C-24 | P3 | sec6_setup.tex:5-6, 35, appendix.tex:70 | CIFAR-100 fraction (0.22) and rounds are not GeFL-F's (GeFL has no CIFAR-100) |
| C-25 | P3 | sec6_setup.tex:53 | "Local ... is a lower bound" contradicted by sec7_results.tex:17-19 |
| C-26 | P3 | sec4_analysis.tex:13-14, 23-24 | "equality iff parallel" (should be non-negative multiples); decoupled decay does not make shrinkage vanish, it makes it negligible |
| C-27 | P3 | sec4_analysis.tex:27 | "so G_c << D_c" is an empirical observation, not a consequence |
| C-28 | P3 | sec5_method.tex:94-96 | "It is the Bayes classifier": plug-in estimate with shrinkage, pinv and jitter |
| C-29 | P3 | sec5_method.tex:31-34, sec6_setup.tex:116-117 | exactness statement omits clipping at the bound and the plain-sum fallback for < 3 clients; SecAgg+ runs with a complete sharing graph |
| C-30 | P3 | appendix.tex:89-90 | FedProto/FedTGP scores are negative mean squared distances |
| C-31 | P3 | sec6_setup.tex:94-95 | "tail = 40% least frequent" is round(0.6C)..C-1 (36-44% on the natural datasets) |
| C-32 | P3 | sec3_setting.tex:27-28 vs sec1_intro.tex:21-22 | list of bias sources omits "sample size" in sec3 but includes it in sec1 |
| C-33 | P3 | sec6_setup.tex:43, tab:shared | LongTailLib "unchanged": training unchanged, but evaluation every 5th round and Fed-GraB rescored with its classifier |
| C-34 | P3 | appendix.tex:53 | sqrt(2F) is the RMS distance over all ordered pairs including c = c' |
| C-35 | P3 | sec6_setup.tex:13-15 | "Training images that no client holds ... form a validation set": only the held-out block is; the rest is discarded |
| C-36 | P3 | sec4_analysis.tex:96-101 vs fig:spread | "spread" in Prop. variance is the variance (1/C) sum (f_c - f_bar)^2; the figure plots the mean per-image unbiased standard deviation |

Section 2 lists the implementation details a reproducer needs that the paper does not state (R-1 to R-30).
Section 3 is the "Facts for the Study Design section".

---------------------------------------------------------------------------------------------------------------------

## 1. Findings

### C-1 (P0, confidence high): "the same exact readout for every method" is false

- Location: abstract.tex:14-15 ("when every method is given the same exact readout, the margins shrink to
  \R{main/cifar10/er_margin} ..."); sec1_intro.tex:69-71 ("with the same exact readout for every method it remains
  ... ahead"); sec7_results.tex:133 ("With the exact readout for every method the margins are below one point");
  sec5_method.tex:99 ("We report every method both as trained and with this readout"); sec7_results.tex:71-72
  (tab:readout caption "Methods whose heads end in a linear layer").
- Code: the readout is computed only in `server.run_pipeline` (fedecs_lt/src/server.py:660-664), i.e. for the
  methods of `runner.PIPELINE` (runner.py:18-37). `htfl_server.run_htfl` (htfl_server.py:79-280),
  `grouped_server.run_grouped` and `track2.run` never compute it (no `er_*` key; grep finds no readout in these
  files). The returned runs confirm it: in `results/NB10_main_cifar10_*/runs.jsonl` FedProto, FedTGP, FedGen, FedGH,
  LG-FedAvg, FedMRL and Local have `er_bal = None`. The slot `er_margin` is the minimum over methods that have
  `er_bal` (docs/papers/fill.py:341-345, `matched(E, S, "er_bal", exclude=("Local",))`), i.e. over GeFL-F, Balanced
  Softmax, FedLC, FedRS, FedDF only.
- Also: LG-FedAvg, FedGH, FedGen and FedMRL do end in a linear layer (`HtModel.head = nn.Linear(512, C)`,
  htfl_client.py:30), so the caption's criterion "Methods whose heads end in a linear layer" does not describe the set.
- Why it matters: the abstract's key qualification of the main result names a comparison that was not run for 6 of
  the 11 other methods; a reader concludes FedECS-LT's representation was compared against all methods.
- Fix (abstract.tex:14-15): "when the methods of the GeFL-F family (GeFL-F, Balanced Softmax, FedLC, FedRS, FedDF)
  are given the same exact readout, the margins shrink to \R{main/cifar10/er_margin} and \R{main/cifar100/er_margin}
  points and are not significant." Same change in sec1_intro.tex:69-70 ("with the same exact readout for every
  method of the GeFL-F family") and sec7_results.tex:133 ("With the exact readout for every method of the GeFL-F
  family"). sec5_method.tex:99: "We report every method of the GeFL-F pipeline (\cref{tab:readout}) both as trained
  and with this readout". Caption of tab:readout: "Methods trained in the GeFL-F pipeline (same extractor, heads with
  a CNN body and a linear last layer); the HtFLlib methods were not evaluated with the readout." Alternatively, run
  the readout for the HtFLlib methods (their bodies are saved in the models zip) and keep the sentence.
- Verification: every method named in the "every method" sentences has an `er_bal` in runs.jsonl, or the sentences
  name only those that do.

### C-2 (P0, confidence high): worst-class recall definition

- Location: sec6_setup.tex:98-100: "the geometric mean and the worst-class recall are computed for each model and
  then averaged, which makes them lower than the same metrics on the models' pooled predictions."
- Code: metrics.py:125-134. `summarize` first averages every scalar over models (line 125-128, which includes
  `worst`), then overwrites it: `out.update(bal=..., worst=float(rec.min()), ...)` with
  `rec = np.mean([v["recall_c"] for v in vals], 0)` (line 129, 132-134). So "worst" = min over classes of the
  architecture-averaged recall. G-mean is indeed the per-model value averaged (not overwritten). The same
  `summarize` is used for HtFLlib (htfl_server.py:260) and Local (server.py:710).
- Why it matters: the stated definition is false; min_c mean_g r_gc >= mean_g min_c r_gc, so the reported
  worst-class recall is the higher of the two quantities, and the explanation "makes them lower" is wrong for it.
  Affects tab:app-cifar10/100 "Worst rec." and sec7 rare-class sentences.
- Fix: "the geometric mean is computed for each model and then averaged; the worst-class recall is the lowest
  class recall of the recall averaged over the ten models (balanced accuracy and tail recall are the same either way)."
- Verification: recompute `worst` from `final_recall_c` of one run: equals min of that vector.

### C-3 (P0, confidence high): "Hyperparameters are those of each method's paper"

- Location: sec6_setup.tex:54: "Hyperparameters are those of each method's paper; where the paper and the authors'
  code disagree we use the code."
- Contradicted by the paper itself and the code:
  - sec7_results.tex:166-169: CReFF and Fed-GraB "run in the benchmark's code with its common settings (learning
    rate 0.1, batch size 32, ten local epochs), which differ from those in the methods' own papers".
  - HtFLlib methods use the HtFLlib run-script values (htfl_server.py:3-6, 24-32: SGD lr 0.01, batch 10, feature
    dim 512, lambda 10, ...), with 5 local epochs instead of HtFLlib's (htfl_server.py:88 `epochs=5`) and T_TN
    rounds from the Stage-I extractor/bodies.
  - FedDF (runner.py:32-33): 20 server steps of batch 128, SGD lr 0.05, beta = 1, on GeFL-F's generator (C-16).
  - FedLC tau = 1 with (n+1)^(-1/4) (fedecs.py:119-122, runner.py:28); FedRS alpha = 0.5 (fedecs.py:128).
- Fix: "Hyperparameters: the HtFLlib methods use the values of HtFLlib's run scripts (\cref{tab:baselines}) with 5
  local epochs; the LongTailLib methods use the benchmark's common settings; the losses of Balanced Softmax, FedLC
  (tau = 1) and FedRS (alpha = 0.5) and the FedDF distillation use the settings in \cref{app:protocol}. No method
  was tuned on our data."  (and add those settings to app:protocol, see R-21/R-22).
- Verification: each method's hyperparameters in the appendix equal those in htfl_server.py:24-32, runner.py:18-37,
  fedecs.py:119-135.

### C-4 (P0, confidence high): what the reproduction check compared against

- Location: sec6_setup.tex:117-118 "The Flower implementation reproduces the client splits of the original GeFL-F
  implementation exactly and its accuracy within seed noise (\cref{app:protocol})"; appendix.tex:74-77 "checked
  against the original implementation ... the balanced accuracy of GeFL-F and \ours was within seed noise of the
  original runs".
- Code: acceptance.py:1-14, 23-36: the reference is "the earlier experiments" (experiments/results K14/K17/K21,
  produced by experiments/core.py, the authors' own pre-Flower pipeline). Tolerance is max(2 points, 3 x old seed
  SD), seed 3 only, and the CIFAR-10 FedECS-LT reference "still had kernel herding" (tolerance 3 points). The
  long-tailed Dirichlet split cannot be GeFL's (GeFL is balanced).
- Why it matters: a reader reads "original GeFL-F implementation" as Kang et al.'s code; the check validates the
  port against the authors' previous code, not against GeFL's.
- Fix (sec6): "The Flower implementation reproduces the client splits of our earlier implementation (which follows
  GeFL's code) exactly and its balanced accuracy within max(2 points, three seed standard deviations)
  (\cref{app:protocol})." Appendix: "checked against our earlier, single-process implementation on MNIST-LT and
  CIFAR-10-LT with seed 3: identical client class counts, client training steps equal weight for weight in unit
  tests (tests/test_local_steps.py), and balanced accuracy within the tolerance above; the CIFAR-10 FedECS-LT
  reference was an earlier configuration (with kernel herding)."
- Verification: wording names experiments/core.py-based runs; tolerance matches acceptance.py.

### C-5 (P1, confidence high): Balanced Softmax, FedLC, FedRS have no generator

- Location: sec6_setup.tex:69-71 (tab:baselines: "loss on each client's head", "loss re-implemented").
- Code: runner.py:27-29: `"Balanced Softmax": dict(pipeline="gefl", gen=None, head=dict(ts=0, tr=5, loss="la",
  tau=1.0))`, `"FedLC": ... gen=None, ts=0, loss="lc", tau=1.0`, `"FedRS": ... gen=None, ts=0, loss="rs"`.
  Balanced Softmax is implemented as logit adjustment with tau = 1 and the add-one smoothed client prior
  (fedecs.py:113-116, 133-134), FedLC as `logits - tau (n_kc+1)^(-1/4)` (fedecs.py:119-122, 130), FedRS as absent-class
  logits x 0.5 (fedecs.py:123-124, 131-132).
- Why it matters: these are the strongest baselines in several tables; a reader assumes they are GeFL-F + a loss.
  They are "GeFL-F w/o generator" + a loss. Combined with prop:reparam, Balanced Softmax and FedLC train exactly the
  same bodies as GeFL-F w/o generator (and their heads differ only by a constant bias shift), which the paper could
  state as a check of the proposition (the returned ER values of BSM, FedLC and FedRS are near-identical per seed).
- Fix (tab:baselines rows): "Balanced Softmax & ... & different models & heads on real features only (no generator),
  loss logits + log of the client's add-one smoothed prior"; "FedLC & ... & both & no generator, loss re-implemented
  (tau = 1, counts + 1) / LongTailLib"; "FedRS & ... & different models & no generator, loss re-implemented
  (alpha = 0.5)". Add in sec6.3: "Balanced Softmax, FedLC and FedRS are applied to GeFL-F without its generator:
  each head trains on its client's real features only."
- Verification: table text matches runner.py:27-29.

### C-6 (P1, confidence high): GeFL-F's T_s is not stated

- Location: sec3_setting.tex:18-19 ("trains its head for T_s epochs on generated features ... then for T_r epochs");
  sec6_setup.tex:135 / tab:hyper (T_s/T_r = 10/5 for FedECS-LT only); sec7 tab:components row "T_s = 1 instead of 10".
- Code: runner.py:22-23 `"GeFL-F": ... head=dict(ts=1, tr=5, loss="ce")`; FedDF ts=1 (runner.py:32-33);
  FedECS-LT ts=10 (runner.py:21).
- Why it matters: T_s = 10 is a sixth difference between FedECS-LT and GeFL-F that the abstract and sec5 never list
  as a component, although the ablation shows it is one; the reader cannot reproduce GeFL-F.
- Fix: sec3_setting.tex:18-19 "... for T_s epochs on generated features ... and then for T_r epochs on its real
  features (GeFL-F: T_s = 1, T_r = 5);" and sec5_method.tex:69: "Each round, client k trains its head for T_s = 10
  synthetic epochs (GeFL-F: 1; \cref{tab:components} shows the effect) ...". Consider adding the change to the
  abstract's list of components.
- Verification: text values equal runner.py:21-23.

### C-7 (P1, confidence high): is the readout part of FedECS-LT?

- Location: abstract.tex:9-12 (the method "... and a class-balanced readout computed from aggregated sums. With ten
  client architectures, FedECS-LT reaches \R{main/cifar10/FedECS-LT/bal}"); alg:fedecs (sec5_method.tex:116, last
  line "server sets every last layer by \cref{eq:er}"); sec5_method.tex:99 ("We report every method both as trained
  and with this readout").
- Code/fill: `main/<ds>/FedECS-LT/bal` is `final_bal` (fill.py:48 `("bal", "final_bal")`), the heads as trained
  before the readout (server.py:654-656); the readout result is `er_bal` (server.py:660-664).
- Why it matters: the method as defined (Algorithm 1) ends with the readout, but every headline number is the model
  without it. Either the readout is a component (then the headline is not the method) or an evaluation device applied
  to all methods (then it should not be in the method's list in the abstract and Algorithm 1).
- Fix (preferred, consistent with how results are used): Algorithm 1 last line "\textbf{Optional evaluation:}
  clients release embedding sums per architecture; server computes the readout \cref{eq:er} (reported as \ER)";
  abstract: "... logit-adjusted heads trained on consensus soft labels; a class-balanced readout computed from
  aggregated sums can replace every architecture's last layer. With ten client architectures, FedECS-LT's heads as
  trained reach ...".
- Verification: the abstract's number and the method description refer to the same model.

### C-8 (P1, confidence high): FedRS in prop:reparam

- Location: sec4_analysis.tex:69-71 "FedRS ... is cross-entropy in rescaled last-layer variables with a larger
  learning rate on those rows; the body follows the same trajectory."; sec1_intro.tex:29-31 lists FedRS among
  corrections that "are exact reparametrisations of such a head: training with them is plain training from a shifted
  initial bias".
- Appendix proof (appendix.tex:23-26) is correct: "a step of size eta on w_c is a step of eta alpha^2 on w~_c".
  With FedRS alpha in (0,1] (code: alpha = 0.5, fedecs.py:128), alpha^2 = 0.25 < 1: a SMALLER learning rate.
- Also: FedRS is not "plain training from a shifted initial bias" (sec1): it is plain training from a rescaled
  initialisation (w~ = alpha w, b~ = alpha b on absent rows) with a per-row learning rate alpha^2 eta, so its body
  trajectory differs from plain CE. "the body follows the same trajectory" is true only relative to that rescaled,
  slower-row training, which the sentence does not say.
- Fix (sec4:69-71): "FedRS, which scales the logits of absent classes by alpha in (0,1]~\cite{li2021fedrs}, is
  cross-entropy in the rescaled last-layer variables alpha w_c, alpha b_c of those classes, trained with the smaller
  learning rate alpha^2 eta; it changes the initialisation and the step size of the absent classes' rows, not the
  information the client has about them." sec1_intro.tex:29-31: "(logit adjustment, FedLC, Balanced Softmax) are
  exact reparametrisations of such a head: training with them is plain training from a shifted initial bias; FedRS
  is one up to a smaller learning rate on the absent classes' rows. None brings in information ..."
- Verification: statement direction agrees with appendix.tex:25.

### C-9 (P2, confidence high): Adam step ratio bound in the kappa derivation

- Location: appendix.tex:14-20 "Per coordinate an Adam step is eta psi with psi = m_hat/sqrt(v_hat) in [-1,1]";
  sec4_analysis.tex:31-36 ("the row drifts toward zero in every coordinate when kappa_c < 1").
- Problem: |m_hat/sqrt(v_hat)| <= 1 does not hold for Adam in general. Kingma and Ba (2015, Sec. 2.1) give
  |step| <= eta (1-beta1)/sqrt(1-beta2) when (1-beta1) > sqrt(1-beta2), which is the case for the defaults used here
  (beta1 = 0.9, beta2 = 0.999: bound 3.16 eta; client.py:164-170 uses `torch.optim.Adam` defaults). The bound is
  approached exactly for sparse gradients, i.e. for a holder whose batches contain class c rarely - the case the
  derivation is about. The derivation also counts holder motion only on the tau_k f_kc batches with class c, but the
  holder also takes decay-only steps (gradient lambda r_c) on its other batches with non-zero momentum. The text
  calls kappa "a heuristic, not a bound" (sec4:34), but the appendix asserts "points toward zero for every coordinate
  and every holder direction when kappa_c < 1" as if derived.
- Fix (appendix): "Per coordinate an Adam step is eta psi with psi = m_hat/sqrt(v_hat); for a gradient of constant
  sign psi -> sign, and in general |psi| <= (1-beta1)/sqrt(1-beta2) (about 3.2 for the defaults) \cite{kingma2015adam}.
  If |psi| <= 1 on the holders' class-c steps and the holders' other steps are neglected, the drift per round is ...,
  which points toward zero when kappa_c < 1." State "heuristic" in the proof header too.
- Verification: inequality matches Kingma and Ba; the sentence makes the |psi| <= 1 assumption explicit.

### C-10 (P2, confidence high): the gate's justification is stronger than prop:w1

- Location: sec5_method.tex:38-39 "A generator anchored to the class means can separate classes only as far as their
  means are apart (\cref{prop:w1})"; appendix.tex:53-54 "when F is small the anchored generator cannot separate the
  classes".
- Problem: prop:w1 gives W_1 <= (1 + L||A||_2) ||mu_c - mu_c'||; L (decoder Lipschitz constant) and ||A|| are not
  bounded or measured, so the bound does not say "only as far as their means are apart" and does not imply "cannot"
  for small F. Also, the deployed sampler is the moment-corrected one, ReLU(mu_c + s_c (X - m_c)) (fedecs.py:91-96),
  for which the coupling gives an extra term |s_c - s_c'| E||X - m||; prop:w1 covers the raw generator only.
  Implementation check: g(z, A mu) = dec(ReLU(dec_z(z) + proto_proj(mu))) with BatchNorm in eval mode (models.py:180-183,
  client.py:236), so g is Lipschitz as assumed.
- Fix: "A generator anchored to the class means separates two classes by at most a constant times the distance of
  their means (\cref{prop:w1}); when the means are close relative to the within-class spread (small F), the anchor
  alone carries little class information." Appendix:53-54 "... when F is small, the anchored generator can separate
  the classes only through the factor (1 + L||A||) ...". Mention that the bound is for the generator before moment
  correction.
- Verification: no sentence claims more than the inequality.

### C-11 (P2, confidence high): alternation proposition: conditions and interpretation

- Location: appendix.tex:35-45; sec4_analysis.tex:104-108.
- Checked: the round map, fixed point and Omega are correct (I verified theta* = theta_r + Omega (theta_s - theta_r)).
  The claim omega_i <= rho_{r,i} needs rho_{s,i}, rho_{r,i} in [0,1], i.e. 0 <= eta lambda <= 1 for every eigenvalue
  (for eta lambda in (1,2), (1 - eta lambda)^S can be negative). "equals one along directions the real loss leaves
  free" needs rho_{s,i} < 1 there (otherwise 0/0 and the contraction fails).
- Main text: "the synthetic data decide exactly the directions that the client's real loss leaves free, which are
  the directions of the absent classes". The second half is not proved and is false for cross-entropy: the real
  loss's gradient on an absent class's row is sum_i p_c(x_i) u_i != 0 (plain CE pushes absent logits down; the next
  sentence of the paper says so). With logit adjustment the push is weaker but not zero.
- Fix (prop): add "with eta lambda_max(H_s), eta lambda_max(H_r) <= 1"; "equals one along directions that the real
  loss leaves free and the synthetic loss does not". Main text: "... converges to a point where the synthetic data
  alone decide the directions that the client's real loss leaves free (\cref{app:proofs}); for a head trained on a
  client's real data the directions of the absent classes are nearly free in this sense, which logit adjustment
  makes more nearly true by weakening the push on their logits~\cite{soudry2018implicit}."
- Verification: proposition states its step-size condition; no unproved identification of free directions.

### C-12 (P2, confidence high): uses and sizes of the validation set

- Location: sec6_setup.tex:13-15 "Training images that no client holds (200 per class, 20 per class on CIFAR-100,
  and at most a quarter of a class with natural imbalance) form a validation set. It trains the referee classifier of
  the generator diagnostics and is never used to select a model or a hyperparameter".
- Code: size `nh = min(200, 2000 // C)` (data.py:213) and, for natural imbalance, `min(nh, len(class)//4)`
  (data.py:214-217): 181 per class on OrganAMNIST (C = 11), 46 on GTSRB (C = 43), 32 on EMNIST (C = 62). The
  validation features also set the public SecAgg+ bounds: `fmax = 4 max|h_val|`, `smax = 4 max ||h_val||^2`
  (server.py:281-289), and per-architecture embedding bounds `emax = 4 max|e_val|` for the readout
  (server.py:506-520); they feed the active-unit diagnostic (server.py:628) and the logged `val_*` metrics
  (server.py:656, 664). No selection is made on them (confirmed: final = last round, server.py:654).
- Fix: "... form a validation set (min(200, 2000/C) images per class, at most a quarter of a class with natural
  imbalance). It trains the referee classifier of the generator diagnostics and sets the public bounds by which
  secure aggregation scales the released vectors (\cref{sec:setup-impl}); it is never used to select a model or a
  hyperparameter."
- Verification: numbers equal data.py:213-217; bounds sentence matches server.py:284-286, 519-520.

### C-13 (P2, confidence high): privacy statements of the consensus and statistics release

- Location: sec5_method.tex:78-80 "the secure sum over clients is the average in \cref{eq:csl}, so no client's
  prediction is revealed"; sec5_method.tex:124-125 "The server learns only federation-level sums".
- Code: the server holds every architecture's head after each round (it averages them in clear, server.py:434-441,
  and sends them, server.py:414) and the generator, the moment-correction parameters and the pool seed
  (server.py:411, 632-643), so it can recompute every client's prediction on the pool itself; the secure sum of the
  consensus adds no protection over what FedAvg already reveals (in the main setting each architecture has exactly one
  client, so its head is that client's model). Every client receives the exact class means of all classes: as the
  `proto` buffer of the anchored generator (models.py:176, server.py:619, 642-643) and in the moment-correction
  arrays `mu` (server.py:634). For a class held by one client, the "federation-level sum" equals that client's sum.
- Fix: sec5:78-80 "... the secure sum over clients is the average in \cref{eq:csl}; since the server holds every
  head, this protects nothing beyond FedAvg itself, but it lets the clients compute the consensus without sending
  their heads to each other." sec5:124-125: "The server, and through the generator and the moment correction every
  client, learns the federation-level class means and spreads. For a class that a single client holds, these equal
  that client's own statistics."
- Verification: privacy text names everything sent to the server and to the clients.

### C-14 (P2, confidence high): what the cost numbers measure

- Location: sec6_setup.tex:103-104 "Cost is the measured communication per client, the wall-clock time on one GPU and
  the floating-point operations"; tab:cost caption (sec7_results.tex) "wall-clock time of one run on one T4 GPU".
- Code: notebooks run two runs per T4 in parallel (`--gpus 2 --per_gpu 2`, build_notebooks.py:293-299; per_gpu = 1
  only for the long Track-2 notebooks and runs that need the whole GPU, runner.py:160-164), so times are measured
  under contention. Secure-sum traffic is counted as 8 bytes per summed element per client upload, downlink 0
  (server.py:94), not the SecAgg+ messages (key exchange, shares, 40-bit masked vectors). Evaluation messages are not
  counted (`account=False`). FLOPs are computed from a forward pass, training as 3 x forward
  (server.py:556-571), not measured.
- Fix: "Cost: bytes of the model and statistics messages per client (secure sums counted as 8 bytes per summed value
  and client, without the protocol's key and share messages; evaluation traffic excluded), the wall-clock time of a
  run on a T4 that ran a second run at the same time, and forward FLOPs per image with training FLOPs estimated as
  three forward passes."
- Verification: caption and text match server.py:94 and build_notebooks.py:299.

### C-15 (P2, confidence high): one-shared-model protocol details of FedECS-LT

- Location: sec6_setup.tex:38-48.
- Missing or imprecise (settings.py:51-58, runner/server/client):
  1. Generator rounds T_KA = 100 (settings.py:52) - the text says only "plus those of its generator".
  2. Stage I uses the protocol too: 10 local epochs, momentum 0.9, lr 0.1 x 0.99^r, batch 32, augmentation
     (client.py:115-117, local_train.py:111-117).
  3. Head-stage learning rate continues the decay: 0.1 x 0.99^(T_FE + r), i.e. it starts at 0.1 x 0.99^100
     (server.py:650, client.py:397).
  4. With one architecture, the consensus soft label is the current global head's own prediction on the pool
     (every client evaluates the same global head; w_k = 1/(1 x 8), server.py:415-418), so CSL is
     self-distillation; the synthetic pool is split into disjoint client slices (fedecs.py:139-146).
  5. The head is trained on frozen features of non-augmented images (client.py:129-144, 369); the baselines train
     end to end with augmentation. This matters for sec7's explanation (sec7_results.tex:155-158).
  6. The 8 clients per round are a seeded subset fixed per (seed, stage, round) (data.py:272-278); LongTailLib picks
     its own subsets, so participation is not paired.
  7. Generator: CIFAR-10 used the class-row branch (gate) in this protocol (manuscript_v00.txt:500).
- Fix: append to sec6:44-46: "... the extractor is trained for 100 rounds with the protocol's local updates, the
  generator for 100 rounds, and the head for 100 with 10 real epochs per round, the learning-rate decay continuing
  from the extractor's rounds. With one architecture the consensus soft label is the current global model's own
  prediction. The head trains on frozen features of unaugmented images. Participating clients are a seeded subset
  per round, not the subsets the library draws."
- Verification: every number in the paragraph is in settings.py:51-58 or the code paths cited.

### C-16 (P2, confidence high): deviations of re-implemented and ported baselines

- Location: sec6_setup.tex:58-61 (caption), 72 (FedDF row), appendix.tex:86-90.
- FedDF (runner.py:32-33, server.py:445-446, 464-502): GeFL-F pipeline (flat-averaged CVAE with coupled decay,
  T_s = 1, CE real phase); after each head round the server distils every architecture for 20 SGD steps (lr 0.05,
  batch 128, no momentum) on fresh generated features with uniform labels, targets = mean softmax over the ten
  architectures (beta = 1), BN running statistics kept. The original FedDF distils on unlabeled public data with Adam
  and early stopping on a validation set; the paper should say what is kept (ensemble distillation on the server)
  and what differs.
- HtFLlib ports (htfl_server.py:1-6, 24-32, 88; htfl_client.py:22-36, 54-68): the frozen Stage-I extractor is shared
  (HtFLlib trains whole models); each client's model is the Stage-I head body + a parameter-free
  AdaptiveAvgPool1d(512) + Linear(512, C); for most architectures the embedding is shorter than 512 (CNN-8: 128,
  CNN-9: 100, CNN-10: 80, CNN-5: 256), so the pooling replicates coordinates; T_TN rounds (e.g. 100 on CIFAR) of 5
  local epochs, SGD lr 0.01, batch 10 with drop_last; personal models persist; FedGen's qualified labels are the
  federation's class counts.
- Fix: add to app:protocol a paragraph "Baselines" with the items above (exact values from the files cited), and in
  tab:baselines FedDF row: "re-implementation: server-side ensemble distillation on GeFL-F's generated features
  (\cref{app:protocol})".
- Verification: appendix values equal the code.

### C-17 (P2, confidence high): definition of a "synthetic epoch"

- Location: sec5_method.tex:69-73; sec3_setting.tex:18-19.
- Code: local_train.py:76, 81-92, 120-122; client.py:385-396; server.py:399-400, 424-431. The synthetic phase is one
  pass over n_syn = T_s x floor(n_k/B) x B freshly generated, moment-corrected samples (batches of B, the real batch
  size), with a fresh SGD optimiser (lr 0.1, no momentum, no decay); then a new SGD optimiser for T_r real epochs. With
  CSL in the main setting, all clients train on the first n_syn samples of one shared pool (offset 0); without CSL
  (GeFL-F) each client draws its own samples.
- Fix (sec5:69): "Each round, client k first takes T_s floor(n_k/B) SGD steps on batches of B freshly generated,
  corrected features (T_s = 10 synthetic epochs of the size of its real data) and then T_r = 5 epochs on its real
  features, each phase with a fresh SGD optimiser."
- Verification: matches `n_syn` (local_train.py:120-122).

### C-18 (P2, confidence high): "Every result is the mean over three seeds"

- Location: sec1_intro.tex:57-58. Contradicted by sec6_setup.tex:109 ("Settings that were run with one seed say so")
  and the Track-2 table caption ("over the seeds that came back"); fill.py:303 counts seeds per setting.
- Fix: "Results are means over seeds 3-5 (fewer where stated) with paired tests and a Holm correction, and every
  metric is logged per class."

### C-19 (P2, confidence high): notation and terminology

- beta: CCWA effective-number parameter (sec4_analysis.tex:45-46, 0.999) and consensus weight (sec5_method.tex:75,
  0.5); tab:hyper lists both as "beta" (sec6_setup.tex:132, 137). Use beta for CCWA and e.g. lambda_cs or omega for
  the consensus weight (code: `csl_beta`, `ccwa_beta`, settings.py:75, 78).
- m_c: number of holders |H_c| (sec3_setting.tex:7) and generator sample mean (sec5_method.tex:57, eq:mc). Rename the
  latter (e.g. \tilde\mu_c, matching \tilde V_c).
- G: number of architectures (sec3:8, eq:csl), the CVAE G(h|y) (sec3:15), \bar G_c (sec4:9) and the distributions
  G_c in prop:w1 (appendix:49, undefined). Use P_c for the generated distribution and a different letter for the
  generator.
- tau: logit-adjustment strength (sec5:71) and local steps tau_k (sec4:30-32). alpha: Dirichlet concentration and
  FedRS factor (sec4:70). rho: imbalance ratio (sec3:5) and eigenvalues rho_{r,i} (appendix:40). H: holder set H_c and
  Hessians H_s, H_r. e: one-hot e_y (sec5:77) and embeddings e (sec5:86). S: sums S^{(1)}, S^{(2)}, steps S_s, S_r,
  sum S (prop:invariance).
- Undefined: LSE (sec4:98), W_1 (appendix:49), "body" (sec5:85, 94; = head without its last layer), T_FE/T_KA/T_TN
  (alg:fedecs uses T_TN; sec6 never names them), "Stage II" also called "generator stage"; "class rows" /
  "conditioning row" / "class-specific conditioning parameters" used interchangeably (sec1:25-27, sec3:16-17,
  sec4:6) - define once. sec3:16-17 says "the c-th column ... the conditioning row r_c": in the code it is row c of
  `label_proj` (C x 512, models.py:114, 127); say "the c-th column of the projection matrix (a row of the
  embedding table), the conditioning row r_c".
- "\ER" macro text "ER" and "exact readout", "class-balanced readout", "class-balanced linear discriminant", "LDA"
  (code) - pick two names (exact readout = the procedure, class-balanced LDA = the classifier).

### C-20 (P2, confidence medium): results from the earlier implementation

- Location: appendix.tex:79-84 ("our earlier implementation"), sec8_discussion.tex:29-31; vs sec1_intro.tex:52-54
  ("We re-implemented the pipeline and every comparison in the Flower framework") and the README claim that the
  Flower code produced every result.
- Fix: sec1: "... and every comparison of \cref{sec:results} in the Flower framework; the check of GeFL's own
  balanced setting (\cref{app:protocol}) uses our earlier implementation." Or re-run GeFL's setting in Flower.

### C-21 (P2, confidence high): the pooled reference

- Location: sec8_discussion.tex:33-36 "the same extractor and heads trained on all the training data pooled reach ...,
  the ceiling of this design".
- Code: runner.py:41-44, data.py:339-341, server.py:194-195: every client holds the whole federated training set,
  Stage I is retrained on it (the cache key includes `pooled`), no generator, plain CE heads. It is the same
  architectures, not the same extractor, and plain CE on the long-tailed pooled data is not an upper bound on
  balanced accuracy (it has no long-tail correction).
- Fix: "the same architectures trained with cross-entropy on all the training data pooled (extractor and heads, no
  generator) reach ... (... with the exact readout), a reference for this design without a long-tail correction".

### C-22 (P3, high): head architectures

- sec6_setup.tex:33-35 "ten CNN head architectures of increasing depth and width (CNN-1 to CNN-10)". Code
  settings.py:61-62: widths (16), (16,32), (20,40), (10,20), (16,32,64), (20,40,80), (10,20,40), (16,32,64,128),
  (20,40,80,100), (10,20,40,80); each block conv3x3-[BN]-ReLU-maxpool2, then a linear layer (models.py:32-52). Not
  monotone. Fix: "ten CNN head architectures of one to four convolutional blocks with different widths (CNN-1 to
  CNN-10, \cref{app:protocol})" and list the widths in the appendix.

### C-23 (P3, high): CIFAR-100-LT is capped as well

- sec6_setup.tex:8-10. The profile asks 504 and 481 images for the two largest CIFAR-100 classes, above the 480 left
  after validation (data.py:210-219 with frac 0.22, settings.py:28-30); CIFAR-10 asks 10073 and 6038 against 4800.
  Fix: "On CIFAR-10-LT and CIFAR-100-LT the profile asks for more than that for the largest classes ...".

### C-24 (P3, high): "GeFL-F's fraction" and rounds on CIFAR-100

- sec6_setup.tex:5-6, 35; appendix.tex:70. GeFL does not report CIFAR-100; the fraction 0.22 is the authors' choice
  "to give the standard CIFAR-100-LT shape" (experiments/core.py:66). Fix: "GeFL-F's fraction of the training set
  (MNIST, Fashion-MNIST, SVHN: 0.1; CIFAR-10: 0.5; CIFAR-100, which GeFL does not use: 0.22) ..." and "the rounds of
  GeFL-F (CIFAR-100: those of CIFAR-10)".

### C-25 (P3, high): Local is not a lower bound

- sec6_setup.tex:53 vs sec7_results.tex:17-19 ("close to or below that of Local"). Fix: "Local, in which each client
  trains its extractor and head alone for the epochs of Stages I and III (5 T_FE + 5 T_TN epochs,
  server.py:704), is the reference without federation." (It also does not share the Stage-I extractor,
  sec6_setup.tex:36: say "All federated methods of this setting share ...".)

### C-26 (P3, high): wording of prop:contraction and lem:adam

- sec4_analysis.tex:13-14 "with equality if and only if all r_{k,c} are parallel" - the appendix (line 5-6) says
  correctly "non-negative multiples of one another"; use that.
- sec4_analysis.tex:23-24 "With lambda = 0, or with decoupled weight decay, this shrinkage vanishes": with AdamW the
  decay is a factor (1 - eta lambda) per step (1 - 1e-6 here); "becomes negligible (a factor 1 - eta lambda per
  step)". (Code GeFL-F + LA uses AdamW wd 1e-3, runner.py:24-25.)

### C-27 (P3, medium): "so G_c << D_c"

- sec4_analysis.tex:27 "A holder's data gradient fits the row and need not increase its norm, so G_c << D_c": the
  premise allows G_c <= D_c, not <<; the size is what fig:rows measures. Fix: "... need not increase its norm; in
  GeFL-F's runs G_c is much smaller than D_c (\cref{fig:rows}), so ...".

### C-28 (P3, high): the readout is a plug-in estimate

- sec5_method.tex:94-96. Code (fedecs.py:163-176): pooled covariance with denominator N - C_present, shrinkage
  gamma = 0.1 plus 1e-6 I, Moore-Penrose pseudo-inverse (`torch.linalg.pinv(..., hermitian=True)`, CPU).
  Fix: "With the population means and covariance and gamma = 0, it would be the Bayes classifier for equally likely
  classes when ...; computed from the sums (denominator N minus the number of held classes, pseudo-inverse) it is a
  plug-in estimate of it." Add the 1e-6 jitter and pinv to app:protocol.

### C-29 (P3, high): exactness, clipping, SecAgg+ parameters

- sec5_method.tex:31-34; sec6_setup.tex:116-117. Code: values outside the bound are clipped and counted
  (secure_sum.py:82-87; bound = factor 4 over the validation maximum, server.py:284-286); with fewer than three
  participants the sum is plain (server.py:91); SecAgg+ runs with `num_shares = n` (complete graph) and
  `reconstruction_threshold = n//2+1`, modulus 2^40, clipping +-8, stochastic rounding (secure_sum.py:66-68,
  settings.py:87; Flower 1.42 `quantize` uses stochastic rounding). Fix: add "Values beyond the bound would be
  clipped; the logs count them" (and report the count), and the SecAgg+ parameters in sec6.5.

### C-30 (P3, high): prototype scores

- appendix.tex:89-90 "softmax of the negative distances". Code htfl_client.py:180-189: score = minus the mean (over
  the 512 coordinates) squared distance; absent global prototypes get -inf (then -1e9). Fix: "a softmax of the
  negative mean squared distances".

### C-31 (P3, high): tail classes

- sec6_setup.tex:94-95 "the 40% least frequent". Code settings.py:66-69: classes round(0.6C) to C-1 by frequency rank:
  4/10, 40/100, but 3/7 (DermaMNIST, FER-2013), 4/11, 3/8, 4/9, 17/43, 25/62. Fix: "(the last 40% by frequency rank,
  rounded: classes round(0.6C) to C)". Natural datasets are relabelled by training-set frequency (data.py:245-252).

### C-32 (P3, medium): sources of bias

- sec3_setting.tex:27-28 lists (i)-(iii); sec1_intro.tex:21-22 adds "through sample size". Make them agree (sample
  size makes nonlinear estimates such as V_c or Sigma^{-1} biased in small classes).

### C-33 (P3, high): "LongTailLib code, unchanged"

- sec6_setup.tex:43; track2.py:101-245, 273-276. Training is unchanged; the driver sets `-eg 5` (evaluation every
  fifth round), records the library's own scores, and scores Fed-GraB with its trained classifier copied into the
  head (track2.py:189-193). Fix: "... training code unchanged; we evaluate every fifth round and score Fed-GraB with
  its trained classifier".

### C-34 (P3, high): sqrt(2F)

- appendix.tex:53: (1/C^2) sum_{c,c'} ||mu_c - mu_c'||^2 = 2 (1/C) sum ||mu_c - mu_bar||^2, so sqrt(2F) is the root
  mean square distance over all ordered pairs including c = c' (over distinct pairs sqrt(2FC/(C-1))), in units of
  sqrt(mean V_c). Fix: "the root-mean-square distance between two class means, in units of the root-mean-square
  within-class spread, is sqrt(2F C/(C-1))".

### C-35 (P3, high): which unused images are the validation set

- sec6_setup.tex:13-15. Only the held-out block (first nh images of each permuted class, data.py:218) is the
  validation set; the images dropped by the long-tail subsampling or the 20,000 cap are discarded. Fix: "A held-out
  block of training images that no client holds (...) forms a validation set; the other unused images are discarded."

### C-36 (P3, medium): logit spread definitions

- sec4_analysis.tex:96-101 (variance (1/C) sum (f_c - f_bar)^2) vs fig:spread caption "logit spread" computed as the
  mean over images of the unbiased standard deviation of the logits over classes (local_train.py:56-67). Define the
  plotted quantity in the caption.

---------------------------------------------------------------------------------------------------------------------

## 1b. Statements checked and found consistent with the code (no change needed)

- Gate: F = mean_c ||mu_c - mu_bar||^2 / mean_c V_c, rows if F < 0.05 (fedecs.py:39-43, settings.py:74,
  server.py:610-616). Counts floored at 1 (fedecs.py:33).
- Statistics release: n, S1 = sum h, S2 = sum ||h||^2 per class, float64, C(d+2) numbers, once, secure
  (fedecs.py:21-27, server.py:281-289, client.py:454-457).
- CCWA: E(n) = (1 - beta^n)/(1 - beta), beta = 0.999, rows = sum E r / sum E over the participating clients by secure
  sum, other generator parameters flat-averaged (fedecs.py:47-59, server.py:339-343, client.py:464-468); a class no
  participant holds keeps its previous rows (undisclosed, R-12).
- Mean anchor: x = ReLU(mu_y + dec(ReLU(dec_z(z) + A mu_y))), z ~ N(0, I), A shared (affine) (models.py:149-198).
- Moment correction: 500 samples per class, s_c = clip(sqrt(V_c / V~_c), 0.1, 10), ReLU(mu_c + s_c (x - m_c)),
  measured on the server, (mu, m, s) sent to clients (fedecs.py:63-109, server.py:631-643).
- Logit adjustment: logits + tau log pi_hat_k, tau = 1, add-one smoothing, real phase only; soft CE without adjustment
  in the synthetic phase (fedecs.py:113-116, 133-135; local_train.py:84-100).
- CSL: (1 - beta) e_y + beta mean_g softmax(f_g), beta = 0.5, w_k = 1/(G n_arch), secure sum, heads at the start
  of the round, eval mode (fedecs.py:149-150, server.py:410-422, client.py:469-477).
- Readout: per architecture counts, sum e, sum ee^T of real training embeddings, secure, gamma = 0.1, b = -1e4 for
  absent classes; C(d_e+1)+d_e^2 numbers per architecture and client; one extra round (fedecs.py:155-176,
  server.py:506-528, client.py:478-489).
- Quantisation: 30 bits over [-8, 8] in units of the bound: step 16/2^30 = 1.49e-8 of the bound (settings.py:87,
  Flower 1.42 stochastic rounding).
- Rounds: MNIST/F-MNIST 20/100/50, SVHN 50/100/100, CIFAR-10/100 50/200/100; OrganAMNIST, EMNIST 20/100/50;
  DermaMNIST, BloodMNIST, PathMNIST, GTSRB, FER-2013 50/100/100 (settings.py:17-46) - appendix.tex:69-72 correct.
- Track-2 profile n_c = floor(n_max rho^{-c/(C-1)}) with n_max = 4800 / 480, 20 clients, alpha 0.5, 40% per round,
  batch 32, SGD 0.1, momentum 0.9, decay 0.99, 10 local epochs, smallest CIFAR-100 class 4 at rho = 100
  (track2.py:45-70, settings.py:51-58).
- Evaluation every 5 rounds (and round 1, last), before and after aggregation; final = after the last round
  (server.py:216, 328, 405, 654); HtFLlib personal heads scored after local training (htfl_client.py:212-243).
- Statistics: sample SD (plots.py:117-121), paired t interval and test, Holm over the compared methods
  (fill.py:193-211, representation.py:120-129).
- prop:contraction, lem:adam (gradient lambda r_c on a non-holder: label_proj is used only in `decode`,
  models.py:126-128; coupled decay, client.py:167-169), prop:reparam part (1), prop:invariance, prop:variance
  (verified algebraically), prop:alternation algebra, prop:w1 inequality, prop:mc proof (cross terms vanish;
  Cauchy-Schwarz) are correct. Consistent with Prop. 2(1): Balanced Softmax and FedLC (no generator, constant per-client
  shifts) train the same bodies as GeFL-F w/o generator, so their \ER values should coincide up to GPU
  non-determinism - a check the paper could state.

---------------------------------------------------------------------------------------------------------------------

## 2. Implementation details a reproducer needs that the paper does not state

| id | detail (code location) |
|----|------------------------|
| R-1 | Extractor: MNIST type conv3x3(in,3)-BN-ReLU-maxpool2 -> 3x16x16; CIFAR type conv3x3(in,3)-BN-ReLU-conv3x3(3,10)-BN-ReLU-maxpool2 -> 10x16x16 (models.py:14-29). MNIST-type for MNIST, F-MNIST, SVHN, OrganAMNIST, EMNIST, FER-2013; CIFAR-type for CIFAR-10/100, DermaMNIST, BloodMNIST, PathMNIST, GTSRB (settings.py:17-46). |
| R-2 | Head widths (settings.py:61-62) and BN in the heads only for F-MNIST, SVHN and all natural datasets (`header_bn`). |
| R-3 | CVAE-F: encoder 4 conv4x4/2 (64,128,256,512)+BN+ReLU on [h, label as a constant channel]; latent 16 (MNIST-type, natural) or 50 (CIFAR); decoder ReLU(dec_z(z) + r_y) then 4 transposed convs; loss = summed MSE + KL (models.py:96-146). PC-VAE encoder takes [h, mu_y] (models.py:160-187). |
| R-4 | Stage I: SGD lr 0.1, no momentum, no weight decay, 5 local epochs, batch B, random crop (pad 4) + flip on CIFAR only, batches < 2 skipped; extractor averaged over all clients, heads within architecture, unweighted; BN statistics averaged (local_train.py:11-30, fl_common.py:61-80, server.py:214-237). |
| R-5 | Stage III heads start from the Stage-I heads (server.py:651 `s1["hd_sd"]`); HtFLlib bodies too (htfl_server.py:132-133). |
| R-6 | Stage II: 5 local epochs, batch 64 for every dataset (local_train.py:33, client.py:185), Adam lr 1e-3, betas (0.9, 0.999), eps 1e-8, coupled weight decay 1e-3 on every generator parameter, fused; one persistent optimiser per client across rounds while the weights are reset to the global ones (client.py:164-170). |
| R-7 | Frozen features are computed once, in eval mode, from unaugmented images (client.py:129-144); Stages II and III see no augmentation. |
| R-8 | Batch sizes B: 64 (MNIST, F-MNIST, SVHN, natural), 128 (CIFAR-10/100), 32 (one shared model) (settings.py). |
| R-9 | Stage III optimiser: fresh SGD per phase per round, lr 0.1, momentum 0 (Track 1); synthetic phase n_syn = T_s floor(n_k/B) B samples (local_train.py:70-122). |
| R-10 | CSL pool: one pool of max_k n_syn samples shared by all clients when each architecture has one client, disjoint slices otherwise; labels uniform (torch.randint) from a per-round seed; generated in 1 GiB chunks (fedecs.py:139-146, client.py:268-337, server.py:411). |
| R-11 | Moment correction measured with the run seed, on 500 samples per class, from the generator in eval mode (server.py:632-633, fedecs.py:67-89). |
| R-12 | CCWA over the participating clients only; a class none of them holds keeps its previous row (fedecs.py:54-59). |
| R-13 | Long tail: n_total = round(frac N); n_c = floor(n_total rho^{-c/(C-1)} / sum); classes = label index (0 most frequent); fractions 0.1/0.1/0.1/0.5/0.22 (settings.py:17-30, data.py:197-219). Natural: frac = min(1, 20000/N), each class floor(frac n_c), classes relabelled by training frequency (data.py:192-194, 245-252). |
| R-14 | Dirichlet split: per class, proportions ~ Dir(0.5 1_K), cumulative cuts; redraw up to 100 times until every client has >= 10 images, then move random images from the largest to the smallest client (data.py:224-239). Same rule for Track 2 (track2.py:56-68). |
| R-15 | Architecture assignment: contiguous blocks of K/10 clients per architecture (data.py:266-269). |
| R-16 | Preprocessing: /255; 28x28 and 48x48 images resized to 32x32 bilinear with antialias, GTSRB resized by PIL to 32; normalisation MNIST none, F-MNIST and natural (x-0.5)/0.5, SVHN/CIFAR channel mean/std (data.py:174-186, settings.py). EMNIST transposed (data.py:139-141). |
| R-17 | Data sources: Hugging Face copies of SVHN, CIFAR-10/100, GTSRB, EMNIST ByClass, FER-2013 (train / privateTest as test; publicTest unused), MedMNIST from Zenodo record 10519652 (data.py:93-150). |
| R-18 | SecAgg+ parameters: quantisation 2^30, modulus 2^40, clipping +-8, max_weight 1, num_shares = n, threshold n//2+1; bounds: N/4 (counts), N x 4max|h_val|/4, N x 4max||h_val||^2/4, readout N, N emax, N emax^2 /4; CCWA 1000 x (20 max|row| + 1)/4 and 1000/4; consensus 1/4 (secure_sum.py:53-72, server.py:284-286, 340-342, 418, 519-520). |
| R-19 | Readout details: denominator N - (number of held classes), + 1e-6 I, pinv on CPU, BN in eval mode, embeddings of the clients' training features (fedecs.py:163-176, client.py:478-489). |
| R-20 | Referee: CNN-5 with BN on validation features, Adam 1e-3, 40 epochs, batch 64 (server.py:255-277); fidelity on 300 samples per class (server.py:369-377). |
| R-21 | FedDF settings (runner.py:32-33, server.py:464-502), FedLC tau = 1 with counts + 1, FedRS alpha = 0.5, Balanced Softmax = LA tau = 1 with counts + 1 (fedecs.py:113-135). |
| R-22 | HtFLlib settings and model (htfl_server.py:24-32, htfl_client.py:22-36; drop_last batches of 10 over the client's shuffled data, htfl_client.py:119-122). |
| R-23 | Local: SGD 0.1, 5 T_FE + 5 T_TN epochs on own data, augmentation for CIFAR, each client's own model scored (server.py:698-717, client.py:412-444). |
| R-24 | Track 2 FedECS-LT: the items of C-15; ResNet-8 widths 64/128/256 (scaling 4), split after stage 2 (128x16x16 features), head = stage 3 + GAP + linear (models.py:292-321). |
| R-25 | Seeding: split RandomState(seed); Stage-I init torch/np seed; generator init seed+1000; every client action seeded by a hash of (seed, stage, round, client) (fl_common.py:19-28); pool seed per round; LongTailLib `-seed seed`. |
| R-26 | Participation: all clients every round in the main setting (pfrac 1); seeded 40% subsets in Track 2 (data.py:272-278, build_notebooks.py:343). |
| R-27 | Evaluation of the GeFL-family: post-aggregation heads evaluated by client 0's process on the full test features; per-architecture metrics averaged; an averaged-softmax ensemble is also logged (metrics.py:145-163). |
| R-28 | Determinism: cudnn.benchmark = True and CUDA graphs for full-size batches (client.py:34-38, models.py:237-252): runs are not bitwise reproducible; same seed = same split and initialisation. |
| R-29 | Stage caching: Stage I and II results are cached by configuration key and shared between methods of a notebook (server.py:192-203, 293-316); this is what makes "the same Stage-I extractor" literal. |
| R-30 | Logged diagnostics that the analysis uses: row norms and Adam second moments of rows per client (client.py:199-204), logit spread probes (local_train.py:56-67). |

---------------------------------------------------------------------------------------------------------------------

## 3. Facts for the Study Design section (code-verified)

System and software
- Framework: Flower 1.42.0 (`flwr[simulation]==1.42.0`, build_notebooks.py:25), simulation engine with the Ray
  backend (Ray 2.55.1, pinned by Flower 1.42 for Python 3.13), one simulation per run, `num_supernodes = K`, client
  resources 1 CPU and 1 GPU (one client process per simulation holding all clients' data and state), server app in
  the worker process (runner.py:74-98). Python 3.13.15, PyTorch 2.11.0+cu128 (environment table of the returned
  report PDFs). Hardware: Kaggle notebooks with two NVIDIA Tesla T4 GPUs; four worker processes (two runs per GPU in
  parallel) except notebooks/runs that need a whole GPU (FedECS-LT with one shared model; long Track-2 runs).
- Secure aggregation: Flower SecAgg+ workflow for every FedECS-LT release (class statistics, CCWA rows, consensus
  predictions, readout sums) whenever >= 3 clients take part; 30-bit stochastic quantisation over [-8, 8] after
  division by a public bound (from N and the validation features), modulus 2^40, complete sharing graph.

Data
- Long-tailed benchmarks: MNIST, Fashion-MNIST, SVHN (fraction 0.1 of the training set), CIFAR-10 (0.5), CIFAR-100
  (0.22) made long-tailed with rho = 100 by n_c proportional to rho^{-c/(C-1)}, capped at the images left after the
  validation block (CIFAR-10 and CIFAR-100 largest classes capped).
- Natural imbalance: DermaMNIST, OrganAMNIST, BloodMNIST, PathMNIST, GTSRB, EMNIST ByClass, FER-2013; each class keeps
  floor(min(1, 20000/N) n_c) images; classes relabelled by frequency.
- Validation block: min(200, floor(2000/C)) images per class (natural: at most a quarter of the class), held by no
  client; used for the referee, the SecAgg+ bounds and logged validation metrics; never for selection.
- Split: Dirichlet(alpha = 0.5) per class over K = 10 clients (main) or 20 (one shared model); minimum 10 images per
  client (redraw up to 100 times, then move images). One client per architecture in the main setting.
- Test: the official test sets (FER-2013: privateTest), full size.

Models
- Main setting: shared extractor (R-1), ten heads CNN-1..CNN-10 (R-2), CVAE-F or mean-anchored PC-VAE generator
  (R-3). One shared model: ResNet-8 (64/128/256), split after stage 2.

Stages and rounds (T_FE / T_KA / T_TN)
- MNIST, F-MNIST, OrganAMNIST, EMNIST 20/100/50; SVHN, DermaMNIST, BloodMNIST, PathMNIST, GTSRB, FER-2013 50/100/100;
  CIFAR-10, CIFAR-100 50/200/100; one shared model 100/100/100 (plus an extra run with T_FE = 200 on CIFAR-10-LT
  rho = 100 seed 3).
- Stage I: FedAvg (unweighted), SGD 0.1, 5 local epochs (one shared model: 10, momentum 0.9, lr 0.1 x 0.99^round,
  batch 32), augmentation on CIFAR. Extractor frozen afterwards; features computed once without augmentation.
- Statistics release (FedECS-LT): n_c, sum h, sum ||h||^2 once; gate F < 0.05 -> class rows with CCWA (beta 0.999),
  else mean anchor.
- Stage II: FedAvg (unweighted) of the generator, 5 local epochs, batch 64, Adam 1e-3 with coupled L2 1e-3, client
  optimiser state persistent across rounds.
- Moment correction: 500 samples per class, scale clipped to [0.1, 10].
- Stage III: per round T_s synthetic epochs (FedECS-LT 10, GeFL-F and FedDF 1, Balanced Softmax/FedLC/FedRS 0) then
  T_r = 5 real epochs (one shared model: 10), fresh SGD lr 0.1 each phase (one shared model: momentum 0.9, decay
  continued); FedECS-LT real phase with logit adjustment tau = 1 (add-one prior), synthetic phase with CSL weight 0.5;
  heads averaged within architecture.
- Readout (evaluation for the GeFL-F family): class-balanced LDA from secure sums, shrinkage 0.1, absent classes
  b = -1e4.

Baselines
- GeFL-F family in the same pipeline (runner.PIPELINE): GeFL-F, Balanced Softmax / FedLC / FedRS (no generator),
  FedDF (server distillation), controls and ablations.
- HtFLlib ports (FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL): frozen Stage-I extractor, Stage-I head bodies,
  T_TN rounds x 5 local epochs, SGD 0.01, batch 10, feature dim 512.
- LongTailLib (FedAvg, FedProx, MOON, FedLC, CCVR, CReFF, FedETF, Fed-GraB, FedLoGe, FedYoYo, RUCR) at commit
  414167807aa2, ResNet-8, on the suite's splits, 200 rounds.

Seeds and evaluation
- Seeds 3, 4, 5 (fewer for some one-shared-model settings); a seed fixes split and initialisations; runs with the
  same seed are paired.
- Evaluation every 5 rounds (and the first and last), before and after aggregation; reported model = after the last
  round; metrics averaged over the ten architecture models, each on the whole test set (worst-class recall: of the
  averaged recall, C-2); as trained and, for the GeFL-F family, with the exact readout.
- Statistics: mean +- sample SD over seeds; FedECS-LT minus each method paired over common seeds, 95% t interval,
  paired t-test, Holm over the methods of the setting.
