# Agent C: technical re-review of the Study design (cycle 1)

Manuscript v02, git 248170e. Between v01 (87b55d6) and v02, sec6_setup.tex, sec5_method.tex, sec4_analysis.tex and
appendix.tex are unchanged. abstract.tex, sec1_intro.tex, sec7_results.tex and sec8_discussion.tex changed. I checked
everything against `fedecs_lt/src` (code version 79ef95114493), `experiments/LEDGER.md` and the returned
`fedecs_lt/results/*/runs.jsonl`. This review is read-only: nothing else was edited. Severities: P0 = a statement is
false about what was done; P1 = major error or missing essential detail; P2 = significant; P3 = minor.

## Part A: status of the earlier findings

| id | status | before (v00) | after (v02) / reason |
|----|--------|--------------|----------------------|
| C-1 | PARTIAL | "when every method is given the same exact readout" (abstract, sec1, sec7:133, sec5:99) | abstract:14, sec1:70-71, sec7:146 and sec5:105 now say "every method of the GeFL-F pipeline". This matches server.py:660-664 (only `run_pipeline` computes `er_*`). **Remaining:** the tab:readout caption still says "Methods whose heads end in a linear layer" (sec7_results.tex:73). LG-FedAvg, FedGH, FedGen and FedMRL also end in `nn.Linear(512, C)` (htfl_client.py:30) but have no readout. Fix: "Methods of the GeFL-F pipeline (heads with a CNN body and a linear last layer, trained on the shared extractor)". |
| C-2 | PASS | "geometric mean and worst-class recall computed per model and averaged" | sec6:279-283: worst = min_c (1/G) sum_g r_gc and G-mean per model then averaged. This matches metrics.py:125-134. |
| C-3 | PASS (one P3 left) | "Hyperparameters are those of each method's paper" | sec6:192-202 lists the sources per group, consistent with htfl_server.py:24-32, grouped_server.py:23-24, grouped_client.py:159-162 and runner.py:27-33. Minor: FedRS does not use smoothed counts; it masks classes with n_kc = 0 (fedecs.py:123-124). See N-12. |
| C-4 | PARTIAL | sec6 and appendix: "checked against the original implementation" | sec6:319-321 is now correct (acceptance.py:1-14). **Remaining:** appendix.tex:74-77 still says "checked against the original implementation ... within seed noise of the original runs". Replace it with the sec6 wording: our earlier implementation; tolerance max(2 points, 3 SD); seed 3. |
| C-5 | PASS | tab:baselines "loss on each client's head" | sec6:173-176 and table rows 221-223: "applied to GeFL-F without its generator", "no generator". This matches runner.py:27-29. |
| C-6 | PASS | GeFL-F's T_s never stated | sec6:60, tab:hyper caption (sec6:76), sec5:71-72 and tab:fairness all give T_s = 1 for GeFL-F. This matches runner.py:22-23. |
| C-7 | PARTIAL | Algorithm 1 ended with the readout; the headline number was as trained | Algorithm 1 now has "Evaluation: ... (reported separately as \ER)", and sec5:106-107 says the headline numbers are the heads as trained. **Remaining:** abstract.tex:9-11 still lists "and a class-balanced readout computed from aggregated sums" among the components of FedECS-LT, right before the as-trained headline number. Fix: "... logit-adjusted heads trained on consensus soft labels; a class-balanced readout computed from aggregated sums serves to compare the bodies. With ten client architectures, the heads of FedECS-LT as trained reach ...". |
| C-8 | PASS (one P3 left) | "larger learning rate"; FedRS listed as a shifted-bias reparametrisation | sec4:70-72 "smaller learning rate alpha^2 eta", consistent with appendix.tex:25 and fedecs.py:128 (alpha = 0.5). sec1:29-32 is correct. Minor: sec4:75-76 still says "such a correction is therefore equivalent to a different initialisation of the last-layer bias". That sentence does not cover FedRS. Suggest "a logit shift (logit adjustment, FedLC, Balanced Softmax) is therefore equivalent to ...". |
| C-10 | PARTIAL | "can separate classes only as far as their means are apart" | sec5:38-41 now matches prop:w1. **Remaining:** appendix.tex:53-54 still says "when F is small the anchored generator cannot separate the classes". Fix: "... carries little class information through the anchor; the factor 1 + L\|A\| is not controlled". |
| C-12 | PASS | validation sizes; uses of the validation set | sec6:118-121: min(200, floor(2000/C)), at most a quarter; referee and SecAgg+ bounds; no selection. This matches data.py:213-217 and server.py:281-289, 506-520. |
| C-13 | PARTIAL | "no client's prediction is revealed"; "the server learns only federation-level sums" | sec5:83-84 and 133-135 are improved and correct about the server. **New inaccuracy:** "the clients obtain the consensus without sending their heads to each other" (sec5:84) implies that the heads stay private between clients. In fact the server sends every architecture's head to every participating client in every round (server.py:414 `query.heads` with `heads_rec(hd_sd, S.groups)`; this is 145 MB per client on CIFAR-10-LT, in the `bytes` record of runs.jsonl), and again for the readout (server.py:510). Fix: "... Since the server holds every head, this hides nothing beyond what FedAvg already reveals. The server sends every architecture's current head to every client, which needs only its own architecture's head for the consensus and all of them for the readout." |
| C-14 | PARTIAL | cost "measured ... on one GPU" | sec6:296-302 is correct (server.py:94, 556-571; build_notebooks.py:293-299). **Remaining:** (a) the tab:cost caption (sec7_results.tex:304) still says "wall-clock time of one run on one T4 GPU". (b) sec6:312-313 "All runs used ... two runs per GPU at a time" contradicts sec6:300 ("in most notebooks") and the code: per_gpu = 1 for the long one-shared-model notebooks (build_notebooks.py:416, 471, 482), and FedECS-LT with one shared model takes the whole GPU (runner.py:160-164). (c) "without evaluation traffic": the exact-readout sums are counted in `comm_mb_per_client` (`sums.readout` for GeFL-F and `secsum.readout` for FedECS-LT in runs.jsonl), although sec5 calls the readout an evaluation. State that the readout's messages are included, or exclude them. |
| C-15 | PASS (see N-5) | one-shared-model details | sec6:156-163: T_KA = 100, Stage I with the protocol's updates, decay continued, no augmentation, CSL = own prediction, seeded subsets not paired. All match settings.py:51-58, server.py:650, client.py:115-117 and 397, server.py:415-418, data.py:272-278. But tab:hyper contradicts this paragraph for the head optimiser (N-5). |
| C-17 | PASS | "T_s = 10 epochs" undefined | sec5:70-73: T_s floor(n_k/B) steps on batches of B, with a fresh SGD optimiser in each phase. This matches local_train.py:70-122. |
| C-19 | PARTIAL | beta reused; m_c reused; G reused; ... | The consensus weight is now omega (sec5:77, tab:hyper). **Remaining:** m_c is both the number of holders |H_c| (sec3:7, sec4:10, sec4:28) and the generator's sample mean (sec5:59-66, Alg. 1 line 117). G is the number of architectures, the generator G(h\|y), \bar G_c and G_c in prop:w1. tau is the LA strength and tau_k the local steps. W_1, LSE and "body" are still undefined. |
| C-21 | PARTIAL | pooled reference: "same extractor", "ceiling" | sec6:184-187 is now correct (runner.py:41-44, data.py:339-341). **Remaining:** sec8_discussion.tex:33-36 still says "the same extractor and heads trained on all the training data pooled ... the ceiling of this design". Use the sec6 wording and drop "ceiling". |
| C-22 | PASS | "increasing depth and width" | sec6:69-71: "one to four convolutional blocks of different widths ... do not increase monotonically". This matches settings.py:61-62. (The widths are promised in app:protocol but not given there; see N-9.) |
| C-23 | PASS | only CIFAR-10 capped | sec6:105-107: CIFAR-10-LT and CIFAR-100-LT capped. This matches data.py:219 (504/481 asked against 480). |
| C-24 | PASS | "GeFL-F's fraction" on CIFAR-100 | sec6:103-104 gives the fractions 0.1 / 0.5 / 0.22 and says GeFL does not use CIFAR-100. This matches settings.py:18-30. sec6:141-142 "the same on CIFAR-100" is acceptable. |
| C-25 | PASS | "Local is a lower bound" | sec6:184-185 "the reference without federation". |
| C-28 | PASS | "It is the Bayes classifier" | sec5:98-101 "plug-in estimate", consistent with fedecs.py:163-176. |
| C-35 | PASS | "images that no client holds form a validation set" | sec6:118, 121-122: "a held-out block ...; the other unused training images are discarded". This matches data.py:218-219. |

Totals: 13 PASS, 8 PARTIAL, 0 FAIL. Every PARTIAL is a passage outside the rewritten section that kept the old
wording (appendix, sec7 captions, sec8, abstract), except C-13, where the new sentence introduces a new inaccuracy.

## Part B: new findings in the Study design and the changed passages

### N-1 (P0, confidence high): the settings were NOT chosen on other seeds, and the splits are not new
- Location: sec6_setup.tex:203-205 ("fixed during its development, on earlier runs of the same benchmarks with other
  seeds (0--2)"); sec6_setup.tex:364-366 ("chosen on earlier runs ... with other seeds and the same test sets ... The
  test splits of the reported runs are new"); sec8_discussion.tex:26-27 ("chosen on earlier runs of the same
  benchmarks with other seeds").
- Evidence: experiments/LEDGER.md lists development runs of the method with seeds 3, 4 and 5 on the same benchmarks,
  for example "Ours Ts=10 | K17_kaggle_fair_retest__nb60 | 3,4,5" (line 58), "Ours Ts=10 ... | K17_..._rows | 3,4,5"
  (103), "Ours Ts=10 + SGATE/SCSL/BALFE | K21_kaggle_signal_gate__nb68 | 3,4,5" (104-106), "Ours (reference) |
  K20_kaggle_mechanism_controls | 3,4" (52), "Ours (gated) | K26_kaggle_cifar10_more_clients | 3,4,5" (76). The
  component ablations behind the final configuration (-CSL, -KH, -LA, -MC, -PC) used seeds 3,4 (K14, lines 292-300).
  fedecs_lt/src/acceptance.py:1-14 and 23-36 uses K14/K17 seed 3-5 runs as the reference, including "mnist|-KH"
  (the removal of kernel herding), and requires the client splits to be identical (`COUNTS`). So the reported runs
  use the same seeds, the same client splits and the same official test sets as the development runs. The
  development also used seeds 0-2 (K01, K08, K13, K16), but not only those.
- Why it matters: this is the main internal-validity disclosure, and it currently states the opposite of what was done.
  Tuning on the reported splits and test sets is the larger risk.
- Fix (sec6:203-205): "The settings of \ours (\cref{tab:hyper}) were fixed during its development, on earlier runs
  of the same benchmarks with seeds 0-5, including seeds 3-5 and their client splits, evaluated on the same test
  sets, and were not changed for the runs reported here." Threats (sec6:364-366): "The settings of \ours were chosen
  on earlier runs of the same benchmarks, partly with the same seeds and client splits and always the same test
  sets; the baselines were not tuned. The reported numbers may therefore favour \ours." Make sec8:26-27 match.
- Verification: the seeds named in the text equal those in LEDGER.md for every "Ours ..." row used to choose a
  setting.

### N-2 (P1, confidence high): "the test set, which is balanced on every long-tailed benchmark"
- Location: sec6_setup.tex:271-272.
- Evidence: the official SVHN test set is strongly imbalanced (from 1,595 to 5,099 images per class), and MNIST's
  test set is mildly imbalanced (892 to 1,135). data.py:93-99 and 89-92 use the full official test sets. Only
  Fashion-MNIST, CIFAR-10 and CIFAR-100 are balanced (runs.jsonl `test_class_counts`: 1000 x 10 and 100 x 100).
- Why it matters: it suggests that balanced accuracy equals accuracy on these benchmarks, and it is false for SVHN-LT.
- Fix: "... on the official test set, which is balanced on Fashion-MNIST, CIFAR-10 and CIFAR-100 and imbalanced on
  MNIST, SVHN and the naturally imbalanced datasets; balanced accuracy weights every class equally in either case."

### N-3 (P2, confidence high): "twelve methods" and "five methods" are typed numbers that will change
- Location: sec6_setup.tex:352-353.
- Evidence: the Holm family is every method of the experiment group with at least two common seeds, except
  FedECS-LT, the diagnostic and the references (fill.py:232-250). The v01 commit adds GeFL's grouped baselines
  (FedAvg, FedProx, FedALA, AvgKD, CCVR, FedETF; build_notebooks.py:131-146) to the main group. Once they return,
  the family on CIFAR-10/100-LT has 18 methods, not twelve. These are result-dependent counts.
- Fix: replace them with slots (for example \R{main/holm_n} and \R{main/holm_n_er}) filled by fill.py from
  `len(rows)` in `paired`.

### N-4 (P2, confidence high): the grouped baselines exchange nothing at K = 10
- Location: sec6_setup.tex:177-180 (list item), the tab:baselines rows (224-227), and tab:fairness, where they are
  missing.
- Evidence: in grouped_server.py:1-6 the server averages heads "within each architecture" (unweighted). At K = 10
  every architecture has one client (data.py:266-269), so FedAvg, FedProx and FedALA reduce to each client training
  its Stage-I head on its own real features (5 local epochs, SGD 0.1; grouped_client.py:162-166), which is the same
  as "GeFL-F w/o generator". AvgKD (other architectures' predictions) and CCVR (server statistics) still exchange
  information. The paper states this for the per-client losses (sec6:175-176) but not for these methods. They are
  also run at K = 10 (build_notebooks.py:141, "10clients_grouped_baselines"), while the table rows exist only when
  `k50_bal` exists.
- Fix: add "At K = 10 each group has one client, so FedAvg, FedProx and FedALA exchange nothing in Stage III; AvgKD
  and CCVR still use the other clients' predictions or statistics." Add a tab:fairness row for them, and show the
  tab:baselines rows whenever their K = 10 results are in.

### N-5 (P2, confidence high): tab:hyper contradicts the one-shared-model protocol
- Location: sec6_setup.tex:75-76 (caption) and 93 ("Head optimiser (each phase, each round) & SGD, learning rate
  0.1, no momentum").
- Evidence: with one shared model, the heads use momentum 0.9 and learning rate 0.1 x 0.99^(T_FE + r)
  (local_train.py:111-117, client.py:397; settings.py:54, 58). The caption mentions only the 10 real epochs.
- Fix (caption): "with one shared model: 10 real epochs per round, SGD with momentum 0.9 and the protocol's learning
  rate decay".

### N-6 (P2, confidence high): description of the SecAgg+ bound
- Location: sec6_setup.tex:315-317 ("after division by a public bound, four times the largest value on the
  validation features").
- Evidence: a client divides each array by b and clips to +-8 (secure_sum.py:82-87). For the class sums,
  b = N x (4 max|h_val|)/4 = N max|h_val| (server.py:284-286). For the squared norms, b = N max||h_val||^2. For counts,
  b = N/4. The readout uses N, N emax and N emax^2 with emax = 4 max|e_val| (server.py:519-520). The CCWA rows use
  1000 (20 max|row| + 1)/4 (server.py:340-342), and the consensus uses 1/4 (server.py:418). So the bound is not "four
  times the largest value", and for the rows and the consensus it does not come from the validation features.
- Fix: "after division by a public bound, a multiple of the federation size N and of the largest value on the
  validation features (for class sums and readout sums; fixed bounds for counts, CCWA rows and consensus); values
  beyond eight times the bound would be clipped and are counted in the logs".

### N-7 (P2, confidence high): Dirichlet redraw rule
- Location: sec6_setup.tex:114-115 ("The draw is repeated until every client holds at least ten images").
- Evidence: data.py:224-238 redraws at most 100 times. If that fails, it moves random images from the largest to the
  smallest client until the minimum is met. track2.py:56-67 does the same.
- Fix: "The draw is repeated, at most 100 times, until every client holds at least ten images; if it still fails,
  random images move from the largest to the smallest client."

### N-8 (P2, confidence medium): unsupported sentences about FedDF and the HtFLlib methods
- sec6_setup.tex:171-172: "As in GeFL, the distillation runs on generated features instead of public data. It is the
  strongest way to use the same generator without changing it." The first sentence needs a citation to the place in
  GeFL where FedDF is run on generated features. Nothing in the code or the results shows that it is "the strongest
  way". Fix: "It uses the same generator without changing it, through server-side ensemble distillation."
- sec6_setup.tex:371-372: "FedProto and FedTGP are best in their first round and then lose accuracy" states a
  result without a slot. sec7 (lines 16-18) words it "best in the first rounds" and puts the drops in slots. Fix:
  "\claim{FedProto and FedTGP lose \R{main/cifar10/FedProto/gefl_drop} and \R{main/cifar10/FedTGP/gefl_drop} points
  from their best evaluated round to the last}" or refer to \cref{sec:res-main} without restating it.

### N-9 (P2, confidence high): references to app:protocol that the appendix does not fulfil
- sec6_setup.tex:69-70 refers to the appendix for the head widths. sec6_setup.tex:201 says "FedDF uses 20 server
  distillation steps per round (\cref{app:protocol})". appendix.tex:68-90 gives neither the widths nor the FedDF
  settings (grep finds no "FedDF" and no widths in appendix.tex).
- Fix: add to app:protocol: the widths (16), (16,32), (20,40), (10,20), (16,32,64), (20,40,80), (10,20,40),
  (16,32,64,128), (20,40,80,100), (10,20,40,80), each block conv3x3-[BN]-ReLU-maxpool (settings.py:61-62,
  models.py:32-52); FedDF: 20 SGD steps per round, batch 128, lr 0.05, no momentum, targets = mean softmax over the
  ten architectures on fresh generated features with uniform labels, BN running statistics kept (runner.py:32-33,
  server.py:464-502). The other details a reproducer needs are R-1 to R-30 of the cycle-0 report.

### N-10 (P3, confidence high): tab:fairness entries
- sec6_setup.tex:262: "HtFLlib methods & FedGen only & 0 / 5": FedGen adds a CE term on generated embeddings in
  every batch (htfl_client.py:140-143), so "0 synthetic epochs" is misleading for it. Write "FedGen: in every batch".
- sec6_setup.tex:263: "Local & none & 0 / 5": Local trains its extractor and head end to end for 5 T_FE + 5 T_TN
  epochs (server.py:704) with no rounds. Write "-- (5 T_FE + 5 T_TN epochs alone)".
- sec6_setup.tex:258-260: the column is called "Exchange in Stage III", but the caption explains "Head exchange".
  The entry "heads" for FedECS-LT, GeFL-F and the losses is a no-op at K = 10, as the caption says. Write "heads
  (none at K = 10)" in the cells, or drop "heads".

### N-11 (P3, confidence medium): sources of non-determinism
- sec6_setup.tex:323: "GPU computation is not bit-reproducible (cuDNN autotuning, CUDA graphs)". cudnn.benchmark =
  True (client.py:34) selects possibly non-deterministic algorithms. CUDA graphs replay the same kernels and are not
  a source of non-determinism in themselves. Suggest "(cuDNN autotuning and non-deterministic GPU kernels)". The
  repeat check is consistent with the code: within a notebook, Stage I is shared (identical Stage-I accuracy for
  every method of a seed in runs.jsonl); between the original and the repeated BloodMNIST notebook it differs.

### N-12 (P3, confidence high): FedRS and smoothed counts
- sec6_setup.tex:199-200: "Balanced Softmax, FedLC (tau = 1) and FedRS (alpha = 0.5) are re-implemented losses with
  the client's add-one smoothed class counts". FedRS uses the mask n_kc = 0 (fedecs.py:123-124, 131-132), not
  smoothed counts. Write "Balanced Softmax and FedLC (tau = 1) with the client's add-one smoothed class counts, FedRS
  (alpha = 0.5) with the classes the client lacks".

### N-13 (P3, confidence high): sec8 communication sentence (v02)
- sec8_discussion.tex:22-23: "its extra communication mostly in the training of the generator". This is supported:
  for CIFAR-10-LT seed 3, train.gen is 10,627 MB per client for FedECS-LT against 8,577 MB for GeFL-F, out of a total
  difference of about 2,600 MB (runs.jsonl `bytes`). The cause is not stated: the anchored generator is larger
  (proto_proj Linear(d, 512) and an encoder with 2 x in_ch input channels, models.py:160-176), and it carries the
  class-mean buffer in every message. A reader would expect the consensus to dominate. Suggest adding "because the
  anchored generator, with its projection of the class means, is larger than GeFL-F's". (I quote these numbers only
  as evidence; the text should use a slot if it gives a figure.)

### Checked and consistent (no change needed)
- sec6:47-51, 66-69: the stages and the extractor match models.py:14-29 and server.py:592-651.
- sec6:115-116, 143-145: client k = architecture k at K = 10, and 5 per architecture at K = 50 (data.py:266-269).
- sec6:123-124: "the class counts of every client are identical across methods for every dataset and seed". I
  verified this over all 30 settings in the returned runs.jsonl: no difference.
- sec6:142-143: the Stage-I extractor is shared. Within each notebook it is identical for every method and seed,
  including across the CIFAR part notebooks.
- sec6:111-112: the "three of them" with a small realised ratio match the returned client counts (PathMNIST about
  1.6, BloodMNIST about 3.3, OrganAMNIST about 4.5). Better through the tab:datasets slot.
- sec6:147-163: the one-shared-model protocol matches settings.py:51-58, track2.py:24-29, 45-70, 263-266 and
  189-193.
- sec6:192-198: HtFLlib values (htfl_server.py:24-32, 88); FedProx mu = 0.01 (grouped_server.py:24).
- sec6:296-302: cost definitions (server.py:94, 556-571), apart from N-6 and C-14(c).
- sec6:306-308: evaluation every 5 rounds, last-round model (server.py:405, 654).
- sec6:319-321: reproduction check (acceptance.py). sec6:350-355: statistics (fill.py:232-266; plots.py:117-121).
- sec6:335-345: the ablation list matches runner.py:30-58.
- v02 sec7/sec8 changes: the new slots (tail_diff, n_sig, above_list, gain_er for one shared model) are used only
  through \R{}. "FedYoYo's evaluation trains 300 rounds and ours 200" matches settings.py:52 and track2.py:262. The
  one-shared-model readout sentence is correct: FedECS-LT is a pipeline method there, so `er_bal` exists.

## Summary of required changes, by priority
1. N-1 (P0): rewrite the tuning disclosure in sec6:203-205, sec6:364-366 and sec8:26-27.
2. N-2 (P1): the test sets are not balanced on MNIST and SVHN.
3. The PARTIALs C-1, C-4, C-7, C-10, C-13, C-14, C-19, C-21: the old wording survives in the sec7 captions
   (lines 73, 304), appendix.tex:53-54 and 74-77, abstract.tex:9-11 and sec8:33-36; the new C-13 sentence is wrong.
4. N-3 to N-9 (P2), N-10 to N-13 (P3).
