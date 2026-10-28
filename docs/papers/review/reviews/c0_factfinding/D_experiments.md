# Agent D (Experimental Researcher): fact-finding on the FedECS-LT experiments

Date: 28 Oct 2026. I only read the repository. Every number below was recomputed from
`fedecs_lt/results/*/runs.jsonl`, the per-round logs and the prediction files in
`fedecs_lt/results/_downloads/*.zip`, unless it is marked as quoted. All scripts and outputs are in
`/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/agentD/` (abbreviated
`$S` below). Untrusted archives were extracted to `$S/data` and `$S/pred`, and every script was run with `python3 -I`.

**Overall.** `docs/papers/generated/numbers.tex` recomputes exactly: every mean, SD, paired difference, CI and
Holm p-value I checked matches to the printed digit (`$S/recompute_out.txt`). Client splits are identical across all
methods for every dataset and seed, Track 2 included. The problems are of three kinds:

- scope: 2 of the 5 long-tailed benchmarks, one-seed cells, and Track-2 FedECS-LT failures;
- claims stated more broadly than the data support: tail concentration, "end-to-end LT methods are stronger", the
  published-number paragraph, and the cost wording;
- baseline construction and statistics that a Neural Networks reviewer will question.

---

## 1. Inventory: experiments in the paper and the runs behind them

Seeds planned: 3, 4, 5. "Runs" counts runs that are in `runs.jsonl`. A dash means nothing came back.

| Paper element | Notebook(s) | Returned (methods × seeds) | Failed / missing | Status in paper |
|---|---|---|---|---|
| Main comparison, CIFAR-10-LT (Tab. 5, 6, 8; abstract) | NB10_main_cifar10_part1-3 | 13 methods × {3,4,5} = 39 | none | filled |
| Main comparison, CIFAR-100-LT | NB10_main_cifar100_part1-2 | 13 × {3,4,5} = 39 | none | filled |
| Main comparison, MNIST-LT / FMNIST-LT / SVHN-LT (columns of Tab. 5–8) | NB10_main_mnist, _fmnist, _svhn | – | 3 × 39 runs not returned | columns absent; text auto-says "both" |
| Exact-readout comparison (Tab. 7) | same runs (`er_bal`) | 6 GeFL-family methods only | – | filled (CIFAR-10/100) |
| Natural imbalance: DermaMNIST, OrganAMNIST, BloodMNIST | NB10_main_* | 13 × {3,4,5} each | none | filled |
| PathMNIST, FER-2013 | NB10_main_*_part1-3 | 13 × {3,4,5} each | none | filled |
| EMNIST (ByClass) | NB10_main_emnist | 13 runs: 11 methods seed 3 only; FedMRL and LG-FedAvg seed 5 only | 26 runs (24 host-RAM OOM); rerun NB10_main_emnist_rerun_7bfd7d not returned | 1 seed, unpaired cells mixed in one column |
| GTSRB | NB10_main_gtsrb_part1-3 | 0 | part 1: 21/21 failed (download time-out); parts 2–3 not returned | absent (silently dropped from Tab. 9) |
| BloodMNIST repeat (same splits and seeds, second account) | `_repeats/NB10_main_bloodmnist__wizard345` | 39 | – | not used (useful for nondeterminism, §6) |
| One shared model, CIFAR-10-LT ρ=100 (Tab. 10) | NB20_track2_cifar10_IF100_s3/s4/s5 | FedAvg, FedProx, MOON, CReFF, FedETF, FedLoGe {3,4,5}; **FedECS-LT, CCVR, Fed-GraB, FedLC, FedYoYo, RUCR {3,4}** | s5_part1of2 not returned; FedECS-LT 200-extractor diagnosis failed (CUDA OOM, cuSOLVER) | filled; caption says seeds 3–5 |
| One shared model, CIFAR-100-LT ρ=100 | NB20_track2_cifar100_IF100_s3/s4 | 10 methods {3,4}; **FedECS-LT {4} only**; FedLoGe {4} only | FedECS-LT s3 failed twice (`train.head` ClientAppException); FedLoGe s3 missing; s5 parts 1–2 not returned; reruns rerun_2a90fa and rerun_f2bb89 not returned | filled with 1 seed for FedECS-LT |
| One shared model, ρ=10 | NB20_..._IF010_s3_part1-3, NB29 | CIFAR-10: 8 baselines s3; CIFAR-100: 4 baselines s3 | **FedECS-LT failed on both** (train.head); part 3 of each not returned | dropped by fill.py (no FedECS-LT) |
| One shared model, ρ=50 | NB20_..._IF050_s3_part1-2 | – | 4 notebooks not returned | columns absent |
| 200-extractor-round diagnosis (`t2/.../fe200_diff`) | NB20 s3 part1 + rerun_2a90fa | – | OOM; rerun not returned | slot empty (`\whennum` hides it) |
| Variants of GeFL-F (Tab. 11) | NB30_ablation_cifar10/100 | – | not returned | table hidden; text slots `abl/...` empty |
| Components of FedECS-LT (Tab. 12) | NB31_components_cifar10/100 | – | not returned | placeholder text |
| Robustness: α 0.1/1.0, K 50/100, ρ 50/10 (Tab. 13) | NB40_sweep_cifar10_* (6) | – | not returned | placeholder text |
| GeFL protocol: grouped baselines at 10 clients; all methods at 50 clients | folder 7 (7 notebooks) | – | not returned | **no slot or section in the paper**; fill.py would drop these runs (D-14) |
| Pooled-data reference (sec 8) | cifar10_cifar100_pooled_data_reference | – | not returned | slot can never fill (D-14) |
| Cost (Tab. 14) | main CIFAR-10 runs (`time_s`, `bytes`) | 13 × 3 | – | filled; measurement confounded (D-11) |
| Calibration, worst architecture, own-mix accuracy (sec 7.7, App. C) | main runs | 13 × 3 | – | filled |
| GeFL best-round measure (Tab. C.17) | `gefl_measure.jsonl` from per-round logs | all main and natural runs | – | filled |
| t-SNE / 10-NN (Fig. 8, 9) | CIFAR-10 `__results.zip` predictions | FedECS-LT and GeFL-F × 3 seeds | – | filled (re-verified, §6) |
| Reproduction in GeFL's IID setting (sec 8, App.) | `experiments/results/K16_kaggle_recalibration__*_iid`, earlier code, seeds 0–2 | GeFL-F, "Ours (gated)" × 3 | – | filled; FedECS-LT label is a different configuration (D-15) |

Totals: 350 runs returned and used, about 522 run-hours of wall-clock time (two runs per GPU on Kaggle 2×T4).
START_HERE lists 65 notebooks and about 410 session-hours in all. Not returned: 3 main-benchmark notebooks,
GTSRB parts 2–3 plus a rerun of part 1, the EMNIST rerun, 3 Track-2 s5 parts, 7 Track-2 ρ=10/50 parts, 2 Track-2
reruns, NB30 ×2, NB31 ×2, NB40 ×6 and folder 7 ×8.

---

## 2. Baseline fairness

### 2a. Table

FedECS-LT for reference: shared frozen Stage-I extractor; Stage I/II/III rounds 50/200/100 on CIFAR; anchored
generator; T_s = 10 synthetic and T_r = 5 real epochs per round; LA τ = 1; CSL; exact readout; Adam.

| Method | Code | Data / split | Rounds (I/II/III) | Local epochs | Optimiser / HP | Tuning | Eval point | Seeds (CIFAR) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| GeFL-F | re-implementation (core.py → Flower, acceptance-checked) | same split | 50/200/100 | T_s=1, T_r=5 | Adam + coupled L2 1e-3 (GeFL-F's own) | none | last round | 3 | fair; disclosed |
| Balanced Softmax | loss on head (`la`, τ=1) | same | 50/–/100 | T_r=5, **no generator** | as GeFL-F's head | none | last | 3 | **at K=10 no cross-client exchange in Stage III**: one client per architecture, so averaging within an architecture does nothing. Same loss as FedECS-LT's real phase. Not disclosed in Tab. 3 (D-6) |
| FedLC | loss `lc`, τ=1 (Track 1); LongTailLib (Track 2) | same | 50/–/100 | 5 | – | none | last | 3 | same as BSM; Track 2: 58.3 vs 69.0 published (D-4) |
| FedRS | loss `rs` | same | 50/–/100 | 5 | – | none | last | 3 | same as BSM |
| FedDF | re-implementation: distillation on GeFL-F's generated features (reuses GeFL-F's cached generator) | same | 50/200 (cached)/100 + 20 server distillation epochs | T_s=1 | – | none | last (after distillation) | 3 | original FedDF uses unlabeled public data; disclosed as "distillation on generated features" |
| FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL | HtFLlib (Apache-2.0) ported to Flower | same; heads on the **shared frozen Stage-I extractor** | 50/–/100 | **5** (HtFLlib default 1) | SGD lr 0.01, batch 10, feature dim 512, λ=10 (HtFLlib scripts) | none | last; personal head after the last local step (HtFLlib scores LG/FedGH/FedGen after the server update; disclosed in App.) | 3 | FedProto and FedTGP peak at **round 1 on every seed** and decline (D-7); the full-model protocol was dropped (action plan §22) |
| Local | each client alone | same | – | – | – | none | final | 3 | lower bound |
| Track-2 baselines (11) | LongTailLib at a pinned commit, unchanged | our splits, checked identical | 200 rounds, 40 % participation | 10 | SGD 0.1, momentum 0.9, decay 0.99, batch 32 (benchmark defaults) | none | last round (FedLTLib reports the best) | 2–3 | CReFF −13.7, Fed-GraB −21.2 and **FedLC −10.7** points below FedYoYo's published values (D-4) |
| FedECS-LT, Track 2 | ours | same | FE 100 + generator 100 + head 100 (300 in all) | 10 real | equal-weight averaging (the benchmark weights by sample size) | settings from development | last | **2 (CIFAR-10), 1 (CIFAR-100)** | failed on CIFAR-100 s3 and on ρ=10 (D-2) |

### 2b. Findings

**D-6 (P1, high confidence). Three of the "strongest" baselines exchange nothing in Stage III, and the compute is
not matched.**
- Location: Tab. 3 ("loss on each client's head"); sec 7.1; sec 8 limitations.
- Evidence: `fedecs_lt/src/runner.py` lines 22–29 set `gen=None, ts=0` for Balanced Softmax, FedLC and FedRS.
  `server.stage_three` averages heads within an architecture (`hd_avg = {g: Averager()}`), and with K=10 each
  architecture has exactly one client. These methods are therefore local head training on the shared Stage-I
  extractor. FedECS-LT's head loss is the same as Balanced Softmax's (`loss="la", tau=1.0`).
- Compute per round differs:
  - synthetic epochs: FedECS-LT 10, GeFL-F and FedDF 1, BSM/FedLC/FedRS/HtFLlib 0;
  - communication rounds: 350 for the generator methods, 150 for the others;
  - communication per client on CIFAR-10: FedECS-LT 11,268 MB, GeFL-F and FedDF 8,659 MB, BSM/FedLC/FedRS 60.8 MB,
    HtFLlib 17–128 MB.
- Why it matters: a reviewer will ask whether the margin over GeFL-F and FedDF comes from 10× synthetic epochs. The
  BSM ≈ FedLC ≈ FedRS identity with the exact readout (51.16 / 51.16 / 51.14 on CIFAR-10) is the expected result for
  reparametrised local heads. The text should say this.
- Action:
  - state in Tab. 3 / sec 6.3 that BSM, FedLC and FedRS use no generator and, at K=10, no head exchange;
  - add GeFL-F (T_s=10) and FedDF (T_s=10) arms, or a "GeFL-F + LA, T_s=10" arm (NB30 has GeFL-F+LA only at T_s=1);
  - report communication relative to the cheapest competitive method, not only GeFL-F.
- Verification: Tab. 3 and sec 6.3 contain the statement, and a matched-T_s arm exists in NB30/NB31.

**D-7 (P1, medium confidence). HtFLlib ports degrade from round 1.**
- Location: sec 7.1 ("best in the first rounds"); Tab. 5.
- Evidence (`$S/curves_out.txt`): the round of the best mean-over-models accuracy, per seed, is
  - FedProto [1,1,1] on CIFAR-10 and CIFAR-100;
  - FedTGP [1,1,1] on CIFAR-10 and CIFAR-100;
  - FedGen [1,1,20] on CIFAR-10 and [1,1,1] on CIFAR-100.
- The CIFAR-10 means fall from 32.3 to 27.1 (FedProto), 32.2 to 25.7 (FedTGP) and 31.7 to 26.9 (FedGen). Round 1
  is one round after the shared Stage-I heads.
- Why it matters: a method whose accuracy falls monotonically from its initialisation usually signals a learning
  rate, λ or loss-scale mismatch for frozen low-dimensional features (HtFLlib's defaults are for end-to-end
  training). A reviewer will treat "close to or below Local" as possibly a port artefact.
- Action:
  - sanity-check one port (FedProto) on HtFLlib's own setting against its published number;
  - or run a small learning-rate / λ grid selected on the held-out validation set, for all baselines, FedECS-LT
    included.
- Verification: the port reproduces HtFLlib within seed noise, or the tuned curve no longer peaks at round 1.

**D-8 (P1, high confidence). FedECS-LT was developed on these datasets and the baselines were not tuned; the gate
sits near its threshold.**
- Location: sec 6.3 ("Hyperparameters are those of each method's paper"); Tab. 4; sec 5 ("threshold fixed during
  development").
- Evidence:
  - The method's settings came out of the E*/K* series in `experiments/results/` (more than 60 job folders on
    MNIST/FMNIST/SVHN/CIFAR, seeds 0–2, selected on the test set according to dossier F5). Seeds 3–5 give new
    splits but the test sets are the same.
  - Logged `class_sep_F` for FedECS-LT against F0 = 0.05:
    - CIFAR-10 main: 0.080, 0.066, 0.071;
    - DermaMNIST: 0.099, 0.099, 0.073;
    - CIFAR-10 IID reproduction (K16): 0.054;
    - CIFAR-10 Track 2: 0.042, 0.041, so the rows branch is used;
    - FER-2013: 0.014–0.016, rows branch.
  - A threshold of 0.08 would flip CIFAR-10 main to the other branch (dossier F6 still applies).
- Why it matters: undisclosed selection asymmetry. The disclosure "fixed before these runs and not swept" is
  accurate but incomplete.
- Action:
  - disclose the development history and the datasets it used;
  - report the gate's alternative branch on CIFAR-10 (NB31 "w/ class rows (CCWA)" provides it);
  - sweep F0 or show that both branches give the same result.
- Verification: NB31 returns, and the text gives the gap between the two branches on CIFAR-10.

**D-4 (P1, high confidence). The comparison with published numbers is selective.**
- Location: sec 7.4, "Comparison with published numbers".
- Evidence: FedYoYo's paper (arXiv 2503.06916, Tab. CIFAR-10-LT, α=0.5, ρ=100; fetched) compared with our Tab. 10:

  | Method | Ours | FedYoYo's paper | Difference |
  |---|---|---|---|
  | FedAvg | 61.0 | 58.22 | +2.8 |
  | FedProx | 54.4 | 55.68 | −1.3 |
  | CCVR | 68.0 | 68.13 | −0.1 |
  | **FedLC** | 58.3 | 69.02 | **−10.7** |
  | FedETF | 65.3 | 68.25 | −3.0 |
  | CReFF | 56.8 | 70.51 | −13.7 |
  | Fed-GraB | 49.4 | 70.63 | −21.2 |
  | RUCR | 54.0 | 55.32 | −1.3 |
  | FedLoGe | 66.7 | 70.54 | −3.8 |
  | FedYoYo | 78.1 | 81.45 | −3.4 |

- The paragraph names only CReFF and Fed-GraB as "clearly below" and omits FedLC.
- CIFAR-100-LT is not discussed. Differences there: CReFF −4.8, Fed-GraB −5.7, RUCR +5.3, FedLoGe −3.2,
  FedYoYo −3.5.
- FedLTLib (arXiv 2609.15625, Tab. VI, fetched) reports higher values for every method, as the paper says. It
  appears to report the best round. Our CIFAR-100 values are 5–10 points below its values (FedAvg 33.2 vs 40.6,
  FedYoYo 42.6 vs 52.8).
- Why it matters: the paragraph leaves out the third method that is clearly below its published value.
- Action: name FedLC; give the CIFAR-100 comparison; state the round counts (300 in FedYoYo, 200 here) and the
  realised ρ of CIFAR-100 (120 here, from rounding down).
- Verification: the paragraph lists every method that differs by more than 5 points.

---

## 3. Statistics audit

**What fill.py does (verified by re-implementation in `$S/recompute.py`).**
- `paired()`: per setting and metric, FedECS-LT minus each method over common seeds. Mean, 95 % t interval
  (df = n − 1) and a one-sample t-test. Holm correction over all methods present: 12 as trained (Local included),
  5 for the exact readout.
- `matched()`: mean difference over common seeds. It selects "best", "strongest" and "margin" (Local excluded).
  `strongest_p` is the Holm-adjusted p of that method.
- `paired_table` bolds Holm p < 0.05.
- Table cells are means over all of each method's seeds, which can be different seed sets (Track 2, EMNIST).
- All checked values reproduce, for example:
  - CIFAR-10 vs FedDF: +5.22, CI ±4.04, raw p 0.0309, Holm 0.0663;
  - CIFAR-100 vs FedDF: +4.26, ±1.61, raw p 0.0076, Holm 0.0208;
  - exact readout vs BSM (CIFAR-10): +1.43, ±2.13, Holm 0.375;
  - exact readout vs FedRS (CIFAR-100): +1.47, ±0.89, raw p 0.0195, Holm 0.058.
- "Every seed favours FedECS-LT" on CIFAR-10 is true (per seed +4.09, +4.49, +7.08).

**D-9 (P2, high confidence). With n = 3 the inference is fragile and depends on the family.**
- Location: sec 6.4, sec 7.1, abstract, sec 8.
- Evidence (`$S/tail_out.txt`, inline checks):
  - With 2 degrees of freedom, normality cannot be checked. The exact two-sided sign-flip permutation p for 3/3
    differences of one sign is 0.25, so no distribution-free test can ever reject at n = 3. All significance rests
    on the t-model.
  - The Holm family is chosen, not given. On CIFAR-10, FedECS-LT vs FedDF gives Holm p = 0.066 over 12 methods
    (Local and the weak HtFLlib ports included), 0.050 over the 5 GeFL-family methods, and Bonferroni × 12 = 0.37.
    The paper's "not significant on CIFAR-10-LT" is a borderline call that depends on this choice.
  - Nothing corrects across the two datasets, the as-trained / exact-readout pair or the metrics.
  - Exact readout on CIFAR-100: the paper says "not significant (Holm p = 0.058)", yet the uncorrected 95 % CI
    [+0.6, +2.4] excludes zero and all three seeds are positive. Against GeFL-F (Holm p 0.0095) and FedDF (0.014)
    the exact-readout differences are significant. "These remaining differences are not significant" is true only
    for the strongest method.
  - Power of the paired t-test at α = 0.05/12: about 0.12 for the CIFAR-10 as-trained margin, 0.04 for the CIFAR-10
    exact-readout margin and 0.19 for the CIFAR-100 one. At n = 5 these become 0.78, 0.22 and 0.94.
- Why it matters: "not significant" here means "underpowered". A reviewer will ask for more seeds, or for intervals
  without verdicts.
- Action:
  - run 5–10 seeds on the CIFAR headline and exact-readout comparisons;
  - pre-specify the family (for example the GeFL-family methods only, or Holm across datasets);
  - say "not significant after Holm; uncorrected CI excludes 0" where that is the case;
  - report the per-seed differences.
- Verification: sec 6.4 states the family, and the exact-readout sentence distinguishes "strongest" from "every
  method".

**D-10 (P2, high confidence). Same seed does not mean the same run.**
- Location: sec 6.4 ("a seed fixes the split and the initialisations of every method, so runs with the same seed
  are paired").
- Evidence (`$S/repeat_out.txt`): `client.py:34` sets `torch.backends.cudnn.benchmark = True`. In the repeated
  BloodMNIST notebook (same splits and seeds, 39 runs):
  - per-run SD of the rerun difference is 0.87 points as trained and 0.41 with the exact readout;
  - the largest change is 3.6 points (FedDF);
  - the FedDF mean changes 70.6 → 68.1;
  - FedECS-LT minus FedDF with the exact readout changes −0.03 → +0.59, and minus GeFL-F +0.08 → +0.63.
- Why it matters: the exact-readout margins on the natural datasets (0.0–0.9) are inside run-to-run noise.
  Pairing removes split variance, not training noise.
- Action: say so in sec 6.4; report the run-to-run SD from the repeat; avoid claims about differences below about
  1 point.
- Verification: the sentence in sec 6.4 is changed and the repeat is cited.

**D-13 (P2, high confidence). "Three seeds" is stated universally and is false for some settings.**
- Location: intro ("Every result is the mean over three seeds with paired tests and a Holm correction"); sec 6.4
  ("Every configuration is run with seeds 3, 4 and 5"); Tab. 10 caption ("seeds 3–5").
- Evidence: EMNIST has 1 seed, with FedMRL and LG-FedAvg on a different seed from the rest. Track 2 has FedECS-LT on
  CIFAR-10 with 2 seeds and on CIFAR-100 with 1 seed (FedLoGe also 1 seed). Six Track-2 baselines have 2 seeds.
- Action: rephrase until the reruns return, or give the seeds per cell.
- Verification: the text matches STATUS.md.

**Stated correctly.**
- "Settings that were run with one seed say so and carry no interval": fill prints no ± for n = 1 and appends
  ", one seed".
- Tables mark single-seed cells by the missing ±.
- The margins use matched seeds.

---

## 4. Claim-by-claim verification

S = Supported, PS = Partially supported, U = Unsupported, UR = Unresolved (pending experiments).

| # | Claim (abridged) | Location | Evidence / recomputation | Status |
|---|---|---|---|---|
| 1 | FedECS-LT reaches 42.8 (CIFAR-10-LT) and 14.3 (CIFAR-100-LT), +5.2 and +4.3 over the strongest of 11 methods | abstract; 7.1 | 42.76 ± 1.97 and 14.31 ± 0.56; strongest FedDF on both; +5.22 and +4.26 | S (2 of 5 planned benchmarks) |
| 2 | Paired tests: significant on CIFAR-100-LT (p = 0.021), not on CIFAR-10-LT (p = 0.066), although every seed favours FedECS-LT | abstract; 7.1 | Holm 0.0208 and 0.0663; per seed +4.09, +4.49, +7.08 | S as computed; depends on the family (D-9) |
| 3 | "highest balanced accuracy on both benchmarks" | 7.1; intro | true for CIFAR-10 and CIFAR-100; MNIST, FMNIST and SVHN not returned | S; UR for the planned 5 |
| 4 | On CIFAR-100-LT, 0.3 classes on average are recognised by no model of FedECS-LT, against 12.3 for FedDF | 7.1 | pooled recall-zero classes [0,1,0] vs [16,11,10] | S |
| 5 | vs GeFL-F: +9.1 [+4.8, +13.3] (CIFAR-10); +6.6 [+5.6, +7.6] (CIFAR-100) | 7.1 | recomputed exactly | S |
| 6 | prototype and shared sub-network methods are close to or below Local under the long tail | 7.1 | CIFAR-10: FedProto 27.1, FedTGP 25.3, FedGH 31.9, LG 31.3 < Local 32.1; FedMRL 33.7 (+1.6). CIFAR-100: FedMRL 8.3 vs Local 7.9, others below | S, but see D-7 (possible port artefact) |
| 7 | FedProto/FedTGP/FedGen best in the first rounds; lose 5.7/6.7/5.3 (CIFAR-10) and 2.8/3.2/3.1 (CIFAR-100) | 7.1 | best round = 1 on nearly every seed; drops match `gefl_drop` | S |
| 8 | they are also the weakest on their clients' own class mixes | 7.1 | own-mix accuracy (CIFAR-10): FedProto 61.3, FedTGP 62.7, FedGen 67.0, lowest three; also on CIFAR-100 | S |
| 9 | FedECS-LT is past its best round by 3.4 (CIFAR-10) and 0.9 (CIFAR-100) | 7.1 | `gefl_drop`; the mean curve peaks at round 15 on CIFAR-10 (45.6 → 42.8) | S |
| 10 | Own-mix price: 68.6 / 31.5, below BSM, FedLC, FedRS, FedDF, Local and FedMRL on both | 7.1 | true on both. On CIFAR-10 FedECS-LT is below 9 of 12 methods (GeFL-F 73.0, FedGH 72.6, LG 71.0 as well); the list is the intersection over both datasets | S (understates the price on CIFAR-10; D-20) |
| 11 | With the same readout the gap to every method shrinks; +2.3 / +3.8 vs GeFL-F; +1.4 / +1.5 vs the strongest (BSM; FedRS) | 7.1 | the gap shrinks for all 5 methods on both datasets; values exact | S |
| 12 | these remaining differences are not significant (Holm p 0.37 and 0.058; CI [−0.7, +3.6], [+0.6, +2.4]) | 7.1; abstract; intro | true for the strongest. The CIFAR-100 CI excludes 0; vs GeFL-F and FedDF significant on CIFAR-100 | PS (wording; D-9) |
| 13 | most of the margin as trained is in the last layer | 7.1; 8 | margin 5.2 → 1.4 and 4.3 → 1.5; gain from the readout: GeFL-F +16.6, FedECS-LT +9.8 | S |
| 14 | The gain of FedECS-LT is concentrated in the tail (CIFAR-100: tail 6.4 vs 1.7, worst 0.2 vs 0.0) | 7.2 | numbers correct. In points vs GeFL-F on CIFAR-100: head +7.0, medium +8.7, tail +4.7; vs FedDF: head −2.3, medium +9.1, tail +5.6. On CIFAR-10 the tail does dominate (+17.5 vs head +3.0). With the exact readout the tail gain vs BSM/FedLC is −0.0 on both datasets | PS (U for CIFAR-100 in absolute points; D-3) |
| 15 | Highest as trained on all six natural datasets; margins +3.4 (FER) to +12.3 (EMNIST, one seed) | 7.3; abstract; intro | true as point estimates. Holm p vs strongest: Derma 0.036, Organ 0.023, Blood 0.10, Path 0.15, FER 0.20; EMNIST n = 1, and FedMRL and LG-FedAvg have no common seed with FedECS-LT | PS (D-12) |
| 16 | With the exact readout the natural-imbalance margins are below one point; on BloodMNIST FedDF is level | 7.3 | +0.8, +0.9, −0.03 (FedDF ahead), +0.8, +0.8 (1 seed), +0.9 | S (inside run-to-run noise; D-10) |
| 17 | The gate chose the class-row generator on FER-2013 | 7.3; 5 | F 0.014–0.016 on all 3 seeds; on Track-2 CIFAR-10 too (0.041) | S |
| 18 | F ranges 0.014–0.449; class rows on Track-2 CIFAR-10 and FER-2013 | 5 | recomputed | S |
| 19 | The generated spread after the moment correction is 0.73–0.98 of the exact one (median 0.86) | 5 | 330 class×run values: 0.730 / 0.864 / 0.985 | S |
| 20 | One shared model: FedECS-LT 61.3 / 34.1, about FedAvg (61.0 / 33.2); FedYoYo ahead by 16.8 and 8.9 on common seeds | 7.4 | exact; CIFAR-100 FedECS-LT is 1 seed (seed 3 failed) | S (n = 1–2; D-2) |
| 21 | "Here end-to-end long-tail methods are stronger" / "remain stronger" | 7.4; conclusion; intro | CIFAR-10: only FedYoYo, CCVR, FedLoGe and FedETF are above; FedLC 58.3, CReFF 56.8, RUCR 54.0 and Fed-GraB 49.4 are below FedECS-LT. CIFAR-100 (1 seed): above are FedYoYo, FedLoGe, CCVR; 5 LT methods below. CCVR is post-hoc calibration, not end-to-end | PS / overgeneralised (D-5). The limitations bullet ("FedYoYo, FedLoGe and CCVR") is correct |
| 22 | LT methods "learn a better representation than a frozen extractor trained for half of the rounds" | 7.4 | no representation measurement in Track 2; the 200-extractor diagnosis failed (pending rerun_2a90fa) | U / UR |
| 23 | Comparison with published numbers (within about 3 points; CReFF and Fed-GraB clearly below; FedLTLib higher for every method) | 7.4 | FedAvg, CCVR, FedETF, FedYoYo match the quoted values; FedLC −10.7 omitted; FedLTLib higher for all (S) | PS (D-4) |
| 24 | 10-NN accuracy 30.9 % vs 26.0 % (first architecture) | 7.6 | reproduced exactly. Architecture 0 has the largest gap of the ten (+4.9); mean over 10 architectures 40.9 vs 37.7; FedECS-LT ahead in 29/30 architecture×seed cells | S (cherry-pick risk; D-16) |
| 25 | ECE 37.8 vs 48.0 (GeFL-F) and 35.9 (FedDF); weakest architecture 38.8 | 7.7 | exact. FedECS-LT is less calibrated than FedDF, FedProto (17.1) and FedTGP (8.8) | S (accurate as stated) |
| 26 | FedECS-LT takes 1.13× the wall-clock time and 1.30× the communication of GeFL-F | 7.7 | `time_s` ratio 1.13; the per-stage sum gives 1.10; time is confounded (D-11); communication 11,268 / 8,659 = 1.30 | PS |
| 27 | "costs more than GeFL-F mainly through the longer synthetic phase" | 8 | time: the extra is mainly Stage III (about +2,600 s). Communication: +2.15 GB of the +2.6 GB extra is the generator's training (train.gen 11.1 GB vs 9.0 GB) | PS |
| 28 | Some settings were run with fewer than three seeds; their tables say so | 8 | true (± missing, ", one seed") | S |
| 29 | Every result is the mean over three seeds | intro; 6.4 | false for EMNIST and Track 2 | U (D-13) |
| 30 | All methods share the same client splits | abstract | `client_counts` identical across all methods per dataset and seed, Track 2 included | S |
| 31 | Comparison against 21 published methods on 8 datasets | intro | 11 Track-1 + 10 Track-2 (FedLC shared) = 21; 8 datasets | S |
| 32 | GeFL's setting: GeFL-F 56.8 % last / 59.3 % best vs 55.9 % published; CIFAR-100 12.5 % | 8; App. | values exact; the "FedECS-LT" 57.8 / 60.0 in the App. is K16 "Ours (gated)" (T_s = 1, kernel herding, different head) | S for GeFL-F; PS for FedECS-LT (D-15) |
| 33 | On CIFAR-10 the long tail costs most of the difference | 8 | IID run uses 25,000 balanced images vs 18,485 LT, so data size and imbalance are confounded | PS (interpretive) |
| 34 | Pooled-data ceiling | 8 | pending, and fill.py cannot fill it (D-14) | UR |
| 35 | Ablation: removing coupled decay + LA recovers X points; FedECS-LT adds Y | 7.5 | NB30 not returned | UR |
| 36 | Components, robustness | 7.5 | placeholders | UR |
| 37 | Conclusion: FedECS-LT improves balanced accuracy and in particular the recall of the rarest classes | 9 | balanced accuracy S on the returned data; "in particular the rarest classes": S on CIFAR-10, not in absolute points on CIFAR-100; ≈0 vs BSM with the exact readout | PS |
| 38 | Readout raises GeFL-F's balanced accuracy by 16.6 / 7.2 | 4 | `gain_er` exact | S |
| 39 | The exact-readout comparison measures the representation | 8 | interpretive; the per-architecture exact-readout margin vs BSM is mixed (archs 0 and 3 negative, 19/30 cells) | PS |

---

## 5. Missing experiments and analyses

### 5a. Pending (built, results not returned)

| Experiment | Notebooks | What it establishes | Cost (planned) |
|---|---|---|---|
| MNIST/FMNIST/SVHN main | NB10_main_mnist/fmnist/svhn | breadth of the main claim (5 benchmarks, as Tab. 5 is laid out) | 117 runs, about 13 h |
| GTSRB | gtsrb part 1 (rerun) + 2 + 3 | 43-class natural imbalance | 39 runs, about 18.5 h |
| EMNIST seeds 4–5 | NB10_main_emnist_rerun_7bfd7d | paired EMNIST; removes the one-seed cell | 26 runs, 7.8 h |
| Track-2 seed 5 and failed runs | NB20 s5 parts (3), rerun_2a90fa, rerun_f2bb89 | 3 seeds for Tab. 10; FedECS-LT CIFAR-100 seed 3; 200-extractor diagnosis | about 30 h |
| Track 2 at ρ = 50 / 10 | NB20_IF050 (4), IF010 part 3 (2) + FedECS-LT reruns | whether "LT methods stronger" holds at lower imbalance | about 60 h |
| Variants of GeFL-F | NB30_ablation_cifar10/100 | decay-only fix vs the full method (dossier F3); GeFL-F without generator; Gaussian generator | 24 runs |
| Components | NB31_components_cifar10/100 | contribution of each component with and without the readout; the gate's other branch on CIFAR-10 (D-8) | 36 runs |
| Robustness | NB40_sweep_* (6) | α, K = 50/100 (several clients per architecture), ρ = 50/10 | 72 runs, about 28 h |
| GeFL protocol | folder 7 (8) | GeFL's grouped baselines (FedAvg/FedProx/FedALA/AvgKD, CCVR/FedETF), 50 clients, pooled ceiling | about 33 h. **The paper has no section for it, and fill.py drops these runs** |

### 5b. Not planned; a Neural Networks reviewer is likely to ask

1. **More seeds** (5–10) for the CIFAR headline and the exact-readout comparison. This is the cheapest way to settle
   D-9. About 2 × 13 × 2–7 extra runs; one hour per FedECS-LT run.
2. **Matched-compute baselines**: GeFL-F and FedDF with T_s = 10, and BSM with the GeFL-F generator (D-6).
3. **Baseline sanity / tuning**: reproduce one HtFLlib method in its own setting; tune the learning rate for every
   method on the validation split (D-7, D-8).
4. **Gate threshold F0 sweep**, or both branches on every dataset. NB31 covers CIFAR-10/100 only.
5. **Hyperparameter sensitivity** of τ, the CSL weight β, T_s and the readout shrinkage γ, at least on CIFAR-10.
6. **Tail-class permutation**: `tail_perm` exists in the code but was never run; the tail classes are always the
   highest-index classes (dossier F14).
7. **Partial participation** in the main setting. Earlier K23 exists in the old code only.
8. **Privacy of the released statistics**: membership inference on class means for small classes, or the
   accuracy cost of DP (Gaussian mechanism). Earlier experiments (K02, E26) used the old code; the paper states
   no DP.
9. **Communication-matched comparison**: accuracy per transmitted MB (FedECS-LT 11.3 GB vs BSM 61 MB per client).
10. **Larger or pretrained extractor, and feature shift.** These are listed as limitations; at least one
    experiment would help.

### 5c. Analyses that can be computed now on CPU

Done (§6): per-round curves and leader per round; head / medium / tail decomposition; per-architecture comparison
as trained and with the exact readout; run-to-run variance from the repeat; 10-NN on all architectures; sensitivity
of Holm to the family; power.

Still possible and not done:
- stratified bootstrap over test images (negligible with 10,000 images, low priority);
- per-class recall against the number of holders;
- calibration with the exact readout (`er_ece` is logged).

---

## 6. CPU analyses I ran

| Script | Output | Result |
|---|---|---|
| `$S/recompute.py` | `$S/recompute_out.txt` | every mean, SD, paired difference, CI, raw and Holm p for 10 settings × {as trained, exact readout}, with per-seed values. **Identical to numbers.tex.** |
| `$S/curves.py` | `$S/curves_out.txt` | test balanced accuracy per round (mean over the 10 models, `pre` evaluation, 21 evaluations). **FedECS-LT leads at every evaluated round on both CIFAR datasets**: CIFAR-10 margin over the best other method 3.5 (round 1), 8.6 (round 15), 6.4 (round 100); CIFAR-100 0.8 → 5.3 → 4.8. The ranking does not depend on the last-round choice. FedProto, FedTGP and FedGen peak at round 1 (D-7). FedECS-LT CIFAR-10 peaks at round 15 (45.6) and ends at 42.8. |
| `$S/tail.py` | `$S/tail_out.txt` | head/medium/tail gains (D-3); zero-recall classes; Spearman correlation of class recall with log class size, which is lower for FedECS-LT (CIFAR-100 0.72–0.75 vs 0.88–0.91), so its accuracy depends less on class size; exact permutation p = 0.25; power. |
| `$S/perarch.py` | `$S/perarch_out.txt` | as trained, FedECS-LT beats GeFL-F, FedDF and BSM on 10/10 architectures and 29–30/30 architecture×seed cells on CIFAR-10/100, Derma, Organ, Blood and FER. PathMNIST: architecture 3 is −4.1 vs GeFL-F. |
| `$S/er_perarch.py` | `$S/er_perarch_out.txt` | exact readout per architecture (CIFAR-10, from `er_probs`; reproduces 52.59 / 51.16 / 51.00 / 50.30). The margin vs BSM is concentrated in small heads (archs 7–9: +4.2, +3.3, +1.8) and is negative on archs 0 and 3; 19/30 cells. vs FedDF: 28/30 cells. |
| `$S/repeat.py` | `$S/repeat_out.txt` | run-to-run SD 0.87 (as trained) / 0.41 (exact readout); conclusions under the repeat (D-10). |
| `$S/knn.py` | `$S/knn_out.txt` | 10-NN for all 10 architectures × 3 seeds (D-16). |
| inline | above | Holm over 5 vs 12 methods; communication breakdown; split identity; dataset statistics; gate values; spread-ratio range. |

---

## 7. Remaining findings (not covered in §2–3)

**D-1 (P1, high confidence). The main comparison covers 2 of the 5 benchmarks the paper is laid out for.**
- Location: Tab. 5–8 headers; abstract; intro.
- Evidence: see §1. The headers list five datasets, and STATUS.md shows that MNIST, FMNIST and SVHN have not
  returned.
- Action: run folder 1, or reduce the layout to the CIFAR pair.
- Verification: STATUS.md shows 5 main settings.

**D-2 (P1, high confidence). FedECS-LT fails selectively in Track 2, and the gaps are reported as if complete.**
- Location: Tab. 10; sec 7.4.
- Evidence: FedECS-LT failed with `train.head` errors on CIFAR-100 ρ=100 seed 3 and on ρ=10 for both datasets. It
  is the only method with such failures.
- Why it matters: the missing runs are FedECS-LT's own, so this is not random missingness, and the CIFAR-100 cell
  rests on one seed. The ρ=10/50 columns are empty.
- Action: return rerun_2a90fa / f2bb89 before submission; in the caption, give the seeds per method.
- Verification: STATUS.md shows 3 seeds for FedECS-LT in Track 2.

**D-3 (P1, high confidence). "Concentrated in the tail" is shown with the dataset where it holds least.**
- Location: sec 7.2.
- Evidence: see claim 14.
- Action: give the head/medium/tail decomposition, for example "on CIFAR-10-LT the tail gains +17.5 against +3.0
  for the head; on CIFAR-100-LT the gain is spread across all groups and is largest in relative terms in the
  tail". Note that with the exact readout the tail gain over BSM vanishes.
- Verification: the sentence matches `final_head`, `final_medium` and `final_tail`.

**D-5 (P1, high confidence). The Track-2 conclusion is overgeneralised.**
- Location: sec 7.4; conclusion; intro.
- Evidence: see claim 21.
- Action: name the methods that are stronger, or define "end-to-end long-tail methods". Drop or test the
  representation explanation (fe200 diagnosis; a linear probe on the extractor).
- Verification: the text lists FedYoYo, FedLoGe, CCVR (and FedETF on CIFAR-10).

**D-11 (P2, high confidence). The cost table's time is not one run on one GPU.**
- Location: Tab. 14 caption ("wall-clock time of one run on one T4 GPU"); sec 7.7.
- Evidence:
  - `time_s` is the whole-process time, measured with two runs per GPU.
  - Stages loaded from the cache are not timed. FedDF loads GeFL-F's generator and BSM loads the Stage-I
    extractor, yet they still log `fe_time_s`.
  - Waiting for another run's claim is timed.
  - FedDF ranges from 1,893 to 6,735 s, and FedECS-LT's Stage II takes 6,147 s vs 3,584 s across seeds.
- Action: report the summed stage times (`fe_time_s + gen_time_s + stage3`) or GPU-seconds, and say that runs
  shared a GPU. Report communication relative to BSM / FedProto as well.
- Verification: the caption matches the measurement.

**D-12 (P2, high confidence). The natural-imbalance claims need caveats.**
- Location: abstract; sec 7.3; Tab. 2.
- Evidence:
  - Realised largest/smallest class ratios: PathMNIST 1.6, BloodMNIST 3.3, OrganAMNIST 4.5, FER-2013 16.6,
    EMNIST 20.4, DermaMNIST 74.9. Three of the six datasets are barely imbalanced, so their results speak to label
    skew rather than a long tail.
  - Three of the five three-seed datasets are not significant after Holm.
  - EMNIST is one seed, with two methods on a different seed.
- Action: qualify the abstract ("highest point estimate"), and give the imbalance ratio next to each result.
- Verification: the abstract and sec 7.3 contain the qualifiers.

**D-14 (P2, high confidence). fill.py cannot use the pending folder-7 and pooled runs.**
- Location: `docs/papers/fill.py`, `group_of` (lines about 63–77).
- Evidence:
  - The function returns None for any K=10 label outside TRACK1_MAIN, the ablations and the "FedECS-LT w…"
    components. So "Centralised (pooled data)" and the grouped FedAvg, FedProx, FedALA, AvgKD, CCVR and FedETF
    are silently dropped.
  - Methods at K=50 other than the 4 SWEEP_METHODS are dropped too.
  - The sec 8 `\whennum{main/cifar100/Centralised (pooled data)/bal}` slot can never fill.
  - `paired()` does not exclude REFS or DIAG, so these would enter the Holm family once they exist.
- Action: add `grouped` and `pooled` groups and a paper section or appendix table for the GeFL protocol (action
  plan §24); exclude REFS and DIAG from `paired`.
- Verification: STATUS.md lists the folder-7 runs under a group after they return.

**D-15 (P2, high confidence). The App. reproduction attributes another configuration to FedECS-LT.**
- Location: App. "GeFL's own setting".
- Evidence: the K16 "Ours (gated)" method has `ts=1`, `kh=true` and the old head config. FedECS-LT uses `ts=10`,
  no kh and the readout.
- Action: relabel it, or rerun FedECS-LT IID in Flower.
- Verification: the label matches the configuration.

**D-16 (P3, high confidence). The 10-NN figure uses the architecture with the largest gap.**
- Location: sec 7.6.
- Evidence: see claim 24.
- Action: report the mean over all ten architectures (40.9 vs 37.7) and the strongest baseline.

**D-17 (P3, high confidence). Captions say "imbalance ratio 100".**
- Evidence: the realised ratios are 48 (CIFAR-10-LT), 96 (CIFAR-100-LT) and 120 (Track-2 CIFAR-100). The setup
  text discloses this; the table captions do not. The tail identities are fixed.
- Action: add "(realised 48)" in the captions.

**D-18 (P3, high confidence). The exact-readout margin is architecture-dependent.**
- Evidence: §6 er_perarch.
- Action: one sentence in sec 7.1.

**D-19 (P3, high confidence). In Track 2 the exact readout lowers FedECS-LT.**
- Evidence: 58.4 vs 61.3 as trained on CIFAR-10 (not reported in the text).
- Action: report it and give a possible reason (rows branch, F = 0.041).

**D-20 (P3, high confidence). The own-mix sentence understates the price on CIFAR-10.**
- Evidence: FedECS-LT is below 9 of 12 methods there.
- Action: give the count.

Two positive findings to keep:
1. The as-trained advantage is robust to the evaluation round and to the architecture (§6).
2. Splits are identical across methods.

---

## 8. Facts for the Study Design section

**Datasets after the long tail or cap**, from `client_counts` of the returned runs. The figures are training images
held by the clients, with the largest/smallest class:
- CIFAR-10-LT: 18,485 images (4,800 / 100, ratio 48); 10 classes; test set 10,000.
- CIFAR-100-LT: 10,926 (480 / 5, ratio 96); 100 classes; test 10,000.
- DermaMNIST: 6,231 (4,493 / 60, ratio 75); test 2,005.
- OrganAMNIST: 19,994 (3,567 / 785, ratio 4.5); test 17,778.
- BloodMNIST: 10,359 (2,130 / 649, ratio 3.3); test 3,421.
- PathMNIST: 19,997 (2,863 / 1,752, ratio 1.6); test 7,180.
- EMNIST (ByClass): 19,970 (1,099 / 54, ratio 20); 62 classes; test 116,323.
- FER-2013: 19,997 (5,026 / 303, ratio 17); test 3,589.
- Track 2, CIFAR-10-LT ρ=100: 11,909 (4,800 / 48); CIFAR-10-LT ρ=10: 19,613; CIFAR-100-LT ρ=100: 10,410
  (480 / 4, ratio 120); CIFAR-100-LT ρ=10: 18,787.
- MNIST, FMNIST, SVHN and GTSRB: no returned run.

**Splits.**
- Dirichlet α = 0.5 per class; K = 10 clients, one per head architecture (CNN-1…CNN-10).
- Held-out validation: 200 per class (20 for CIFAR-100; at most a quarter of a class for natural datasets). It is
  used only to train the referee classifier.
- Client sizes on CIFAR-10-LT range from 286 to 6,178.
- The rarest class is held by 8–10 of 10 clients on CIFAR-10-LT and by 4 on CIFAR-100-LT.
- Track 2: 20 clients, 40 % per round, standard profile n_c = ⌊n_max ρ^{−c/(C−1)}⌋.
- Client counts are identical across methods for every dataset and seed (verified).

**Seeds and runs per setting.**
- Seeds 3, 4, 5. Main and natural settings: 13 methods × 3 = 39 runs per dataset (EMNIST: 13 of 39).
- Track 2 ρ=100: 12 methods × 3 planned; returned 30 (CIFAR-10) and 22 (CIFAR-100).
- 350 runs used in all (NB00 excluded). One full repeat of BloodMNIST (39 runs) measures run-to-run variation.

**Rounds.**
- CIFAR: 50 / 200 / 100 (extractor / generator / heads).
- Derma, Blood, Path, FER and GTSRB: 50 / 100 / 100. OrganA and EMNIST: 20 / 100 / 50.
- Heads per round: T_s = 10 synthetic and T_r = 5 real epochs for FedECS-LT; T_s = 1 for GeFL-F and FedDF; 0 for
  the others. HtFLlib methods: 5 local epochs.
- Track 2: baselines 200 rounds × 10 local epochs; FedECS-LT 100 + 100 + 100.

**Evaluation timing.**
- Test set every 5 rounds, before and after aggregation.
- The reported model is the one after the last round (no selection).
- GeFL's measure is the mean over models of each model's best accuracy at rounds 1, 10, …, 100 (App. C.17).

**Statistical procedure.**
- Mean ± sample SD over seeds.
- Paired FedECS-LT-minus-method differences over common seeds: 95 % t interval with df = n − 1, one-sample t-test,
  and Holm over the methods of the table (12 as trained, 5 with the exact readout), per dataset.
- Best, strongest and margin are computed on common seeds.
- No correction across datasets or metrics; n = 1 cells carry no interval.

**Compute.**
- Kaggle, 2 × T4 (14.6 GB) per notebook, two runs per GPU; Flower 1.42 with the Ray backend.
- About 522 run-hours of wall-clock time over the 350 runs.
- Typical FedECS-LT run: 0.7–2.9 h in the main setting and 2.1–3.4 h in Track 2.
- Median run per dataset: 21 min (Derma) to 68 min (Path); Track-2 baselines median 3.7 h.
- 65 notebooks and about 410 session-hours planned in all.
- `gpu_peak_mb` values of 13–19 GB exceed a single T4, so they must be summed over processes or GPUs; do not quote
  them as single-GPU memory.

**Sources consulted.**
- FedYoYo, arXiv 2503.06916 (CIFAR-10/100-LT table, 300 rounds, ResNet-8, 20 clients):
  https://arxiv.org/html/2503.06916v1
- FedLTLib, arXiv 2609.15625 (Tab. VI, apparently best round): https://arxiv.org/html/2609.15625
