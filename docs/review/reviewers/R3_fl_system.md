# Reviewer 3: federated systems and algorithms

**Specialty.** Federated optimisation under non-IID and long-tailed data, model-heterogeneous FL (distillation, partial sharing, generator-based transfer), aggregation rules, system models (participation, statefulness, communication).

**Recommendation.** Reject in the current form. With the baselines and protocol fixes listed below, this could become a borderline-accept workshop or journal paper.
**Confidence.** 4/5. I read the full draft, all of `experiments/core.py` that the K/E/F runs use, the specs, and every relevant `runs.jsonl`. The aggregate numbers below were recomputed from the raw runs with `r3_work/agg.py`.

**Bottom line.** The training pipeline itself mostly respects its stated disclosure model. Server-side MC, KH, PC, BBC and the BBC gate use only class sums and synthetic data, and nothing is fitted on test data. So the attack "the server secretly uses data it should not" **largely fails** for training. What does hold up:
1. At K = 10 the setting is ten *local* heads plus a shared FE and generator. It is not heterogeneous model aggregation.
2. The paper's own diagnostics show that a classifier-readout correction on plain GeFL-F closes 80–98% of the gap the full pipeline closes. No recalibration or long-tail FL baseline is compared.
3. The "final method" is assembled per dataset after looking at test results: branch choice, T_s and metric choice.
4. Headline IID gains exist only under a test-selected best-round metric.
5. The cost claims omit a measured ~11× increase in head-stage compute and the extra statistics, pool and cross-architecture traffic.

---

## 1. Reconstructed client–server lifecycle (as implemented)

The simulation is a single process. Every "server" quantity is computed from the concatenated client tensors. For the sum-type statistics this is mathematically what secure aggregation (SecAgg) would return, so it is not by itself a violation.

| Stage | Client does | Exchanged | Server does | Code |
|---|---|---|---|---|
| Partition | — | — | LT profile n_c ∝ IF^{c/9} with a fixed total (≈ 6 000 images on MNIST/FMNIST), then Dirichlet(0.5) per class. 2 000 balanced held-out images are kept out of all clients. | core.py:195-235 |
| Architecture assignment | — | — | contiguous blocks, `blk = max(K//10, 1)`, so **K = 10 gives one client per architecture** | core.py:238-241 |
| I. FE warm-up (T_FE rounds) | 5 local SGD epochs (lr 0.1, fresh optimiser each round) on FE + own head | FE and head state dicts | FE: **flat, unweighted** mean over all K clients. Heads: flat mean within each architecture (at K = 10 this is the identity). | core.py:796-834; Averager 527-569 |
| Freeze | the FE computes features once | — | all methods reuse the cached FE per seed (paired) | core.py:2235-2250 |
| Class separation F | — | (class sums) | F computed from exact sums. **Logged only; never used to select anything in code.** | core.py:2253-2267 |
| II. Generator (T_KA = 100–200 rounds) | 5 local epochs of Adam, lr 1e-3, coupled wd 1e-3. **Each client keeps a persistent Adam state across rounds** (GenClient). | full generator each round | flat mean. HWA: per-class rows weighted by E(n_kc), taken from the **true per-client counts** passed to the server (`ctx.part["counts"]`); fallback to the previous row if no holder | core.py:1012-1033, 1284-1312, 1100-1111 |
| PC-VAE prototypes | — | class sums of h and counts (1 upload) | m_c = S_c / n_c, stored as a buffer in the generator | core.py:1179-1183, 353-437 |
| MC | — | sums of h, ‖h‖², counts | samples the generator, rescales to the exact mean and spread | core.py:1522-1560 |
| KH | — | class sums of RFF φ(h), D = 6144 (61k numbers per client) | herding over 4× candidates per class | core.py:1465-1519 |
| III. Heads (T_TN rounds) | T_s epochs on synthetic data (soft CSL targets) then T_r = 5 real epochs with LA by the client's own prior; fresh SGD each round | heads up; pool seed and soft labels down | builds the pool with G + MC + KH, **runs every architecture's head on the pool**, and forms β-mixed consensus targets. At K = 10 all clients share one pool slice. Then a flat per-architecture mean (identity at K = 10). | core.py:1749-1973 (pool 1779-1823; shared slice 1784-1787; head aggregation 1824, 1971-1973) |
| Evaluation | — | — | **test set every 10 rounds.** `best_mean_acc` is the mean over architectures of each head's best test accuracy over rounds, stage I included. `final_bal` is the last evaluation. | core.py:780-792, 2029-2032, 2276, 2428-2436 |
| BBC | — | — | Saerens fixed point on class-balanced synthetic features. Gate: global train counts, max/min > 2. **No test data in the fit or the gate.** | core.py:2049-2070, 2411-2415 |
| Oracle | — | — | refits each fc on **pooled real** features (non-federated diagnostic) | core.py:2133-2154 |
| Exact readout (ER) | **every client embeds its data with every architecture's body** | per architecture: C·d_g + d_g² sums | uniform-prior LDA readout | core.py:2100-2130 (loop 2113-2120) |
| DP clipping (E26) | — | — | clip norm = 90th percentile of the 2 000 held-out *in-distribution, labelled* images, treated as "public" | core.py:2268-2275 |

**System model, as implemented.** Synchronous rounds, **full participation every round** (`for k in range(ctx.K)` at core.py:805, 1299, 1825), no dropout, no stragglers. Clients are stateful: persistent Adam moments at K = 100 (core.py:1293), with CPU offloading only for memory. The FE is homogeneous. "Heterogeneity" is ten tiny CNN heads (widths 10–128, core.py:66-67) on a shared 3×16×16 map. Federations are small: 5 996 MNIST/FMNIST images in total; at K = 100 the smallest client has 10–13.

---

## 2. Strongest algorithmic objection

**The improvement is mostly a classifier-readout and prior effect, and the paper never compares against the simple mechanisms that target it. At K = 10 the heads are purely local models.**

* At K = 10 there is no head aggregation (assign_arch gives one client per architecture; the Averager over one state dict is the identity). The authors' own K07 spec admits that "FedProx / FedALA coincide with grouped FedAvg at K = 10 (one client per architecture)" (specs/K07…py:8-9). So the "FedAvg (per architecture)" baseline (62.9 / 52.8) is local training.
* The paper's own diagnostics (`oracle_bal`, `final_bal`; 3 seeds, full schedule) show how much a last-layer correction on plain GeFL-F recovers:

| | GeFL-F | GeFL-F + fc refit on its own frozen bodies | Final method | Share of the gain recovered by the refit |
|---|---|---|---|---|
| MNIST LT (F10 / E25) | 74.24 | 90.73 | 94.82 | 80% |
| FMNIST LT (F10 / E25) | 58.74 | 79.17 | 79.67 | **98%** |
| SVHN LT (K01 / K10) | 50.49 | 64.82 | 65.60 | **95%** |

* The refit here uses pooled real data, which is an oracle. But the authors' own *federated* version, ER (exact sums), matched the oracle within 0.8 points in 9/9 runs (readout_report.md:187-197). On the same reduced schedule, **GeFL-F + ER (FMNIST 79.1) beat the Ours-A head (75.7)** (readout_report.md:189-191).
* The authors' research report states that the paper "must add the GeFL-F + recalibration baseline" (readout_report.md:54-57, 233-237). The draft shows the oracle row only for its own method (paper_draft.md:753) and never shows GeFL-F's oracle.
* Prior correction alone also does much of the work:
  * GeFL-F + LA: +4.3 / +6.2.
  * In K09, removing LA costs 3–6 points, but BBC then recovers almost all of it: bbc_bal 95.14 vs 95.40 on MNIST and 80.16 vs 80.69 on FMNIST (K09_ablation_analysis.md finding 4). LA and BBC are substitutes, not additive.
* Taken together, a reviewer can reasonably argue that most of the measured gain comes from:
  1. un-collapsing the generator, which removing coupled weight decay may also achieve (see §5);
  2. a prior correction (LA or BBC);
  3. a readout correction.

  None of these is novel, and none is benchmarked against CCVR / CReFF / Fed3R / FedLC / FedRS / FedProto / FedGen on the same frozen FE.

---

## 3. Restricting assumptions

1. **One client per architecture at K = 10.** All long-tail headline numbers, the K09 ablation and every CIFAR-10/SVHN result use K = 10. Head heterogeneity is never exercised together with head aggregation, except in MNIST/FMNIST at K = 50 / 100.
2. **Full, synchronous participation with stateful clients.** HWA's no-holder fallback (core.py:567) and the persistent Adam state are never tested under client sampling. In cross-device FL, persistent per-client optimiser state across 100 rounds is unrealistic.
3. **Rare classes are not rare across sites.** Minimum tail-class holders are 6–8 of 10 at K = 10 and 16–24 of 50 / 100 (`tail_holders` in runs.jsonl). The introduction motivates "rare conditions appear at only a few sites" (paper_draft.md:60), but that regime (1–2 holders) is never tested. It is also exactly where SecAgg does not hide per-class sums (paper_draft.md:995-996).
4. **Cross-architecture execution.** CSL needs the server to run every architecture on the pool (core.py:1800-1804). ER needs every client to run every architecture's body (core.py:2113-2120). If each client can run all ten heads, the hardware motivation for model heterogeneity largely disappears.
5. **Homogeneous, frozen, tiny FE.** All transfer happens in one 768-d (2 560-d on CIFAR-10) feature space produced by one conv layer.
6. **Tiny federations.** About 6 000 training images per federation on MNIST/FMNIST, about 7 300 on SVHN and about 18 500 on CIFAR-10.
7. **An in-distribution labelled "public" set** is assumed for DP clipping (core.py:2269-2272). If such a set exists, a balanced readout could be trained on it directly.
8. **Class counts.**
   * The simulation gives the server the per-client histograms for HWA (core.py:1295). The SecAgg variant is described but not implemented.
   * The BBC gate and HWA reveal federation-level counts.
   * At K = 10 the "aggregated head" the server uses for CSL and BBC is the single client's own head, in the clear (forwarded to Reviewer 6).

---

## 4. Most important missing FL setting

The regime that would test the paper's claims has K ≥ 50, **partial participation** (for example 10% per round), several clients per architecture, and **tail classes held by 1–3 sites**. A run there should report final-round balanced accuracy against:
* GeFL-F + LA + federated readout recalibration (ER or CCVR-style);
* FedLC / FedRS heads on the same frozen FE;
* FSG + LA.

That is the setting the introduction describes, and HWA, the statistics releases and CSL behave differently in it (sampled holders, stale Adam state, a pool regenerated per round).

---

## 5. Simpler competing algorithms that could explain the results

1. **GeFL-F + LA (or BBC) + readout recalibration** (ER, Fed3R, CCVR). The diagnostic upper bound is 90.7 / 79.2 / 64.8 against 94.8 / 79.7 / 65.6. It is not run as a baseline.
2. **GeFL-F with weight decay removed or decoupled.** Lemma 2 itself predicts that λ = 0 removes the contraction.
   * One-seed quick passes: NOWD gives 69.9 / 62.6 against HWA+LCD 70.3 / 62.2 (E02_conditioning_collapse_quick).
   * The full-schedule comparison (K15: AdamW, no decay on rows, LCD) is **NOT RUN** (LEDGER.md:619-620, PROGRESS.md §3).
   * HWA's necessity relative to a one-line optimiser change is therefore unestablished.
3. **CCVR at the FE level.** FSG + LA is a Gaussian on exact sums. It ties the final method on CIFAR-10 LT: 44.17 against Ours-A T_s=10 at 44.38. It is 7–8 points behind on MNIST/FMNIST and 21 behind on SVHN, so it is a real but partial competitor.
4. **Compute budget.**
   * T_s = 10 multiplies head-stage compute by about 11.5×: 59 s → 680 s per MNIST run (`head_time_s`, F10 vs E25).
   * No baseline gets the same budget. GeFL-F with hard labels at T_s = 10 under the long tail does not exist.
5. **Ensembling.** The 10-head softmax ensemble is above the mean head in every setting (CIFAR-10 IID 67.5 vs 56.8). Much of what CSL transfers is plain ensemble distillation (FedDF / FedMD family), which is not benchmarked.

---

## 6. Evidence ledger

Severity: 1 = fatal, 2 = major, 3 = moderate, 4 = minor.

| ID | Manuscript claim (location) | Evidence (file:line) | Reasoning | Consequence | Sev. | Conf. | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| R3-1 | Gains over GeFL-F of +19.6 / +21.2, plus the CIFAR and SVHN gains, are attributed to anchoring the generator (Abstract l.30-37; §5.11 l.751-760) | GeFL-F oracle 90.73 / 79.17 / 64.82 vs final 94.82 / 79.67 / 65.60 (F10, E25, K01, K10 runs.jsonl; readout_report.md:30-41, 54-57); ER ≈ oracle 9/9 (readout_report.md:187-197); draft shows only its own oracle (paper_draft.md:753) | A readout correction on the unmodified GeFL-F recovers 80–98% of the gain | Contribution over a known technique (CCVR / Fed3R) is +0.5 to +4 points, not +15 to +21 | 1 | High | Demonstrated (oracle); suspected (federated ER at full schedule) | Add GeFL-F(+LA) + ER / CCVR / Fed3R baselines with 3+ seeds on all 4 datasets and report the margin against them |
| R3-2 | "Heterogeneous FL": heads of each architecture are averaged (§2 l.101-102) | assign_arch core.py:238-241; head averaging core.py:1824, 1971-1973; K07 spec l.8-9 | At K = 10 each head is one client's local model. Cross-client transfer is only FE + generator + CSL. Every CIFAR, SVHN and K09 result is K = 10. | The paper studies local classifiers with global synthetic data, not heterogeneous aggregation. "FedAvg" baselines are local training. | 2 | High | Demonstrated | State this explicitly. Add K ≥ 50 with several clients per architecture on CIFAR and SVHN, and report heterogeneity-aware baselines (FedProto, FedGen, FedDF/FedMD) |
| R3-3 | The final method is one rule: F-gate + T_s = 10 + gated BBC (§4.6 l.459-473) | F is computed but unused in code (core.py:2253-2267). The SVHN "final" 65.60 is T_s = 1 (specs/K10 ARMS). The abstract's CIFAR "46.1 / ≈49" is T_s = 1 (K08 cifar10_lt); T_s = 10 gives 44.38 / 47.26. The gate was introduced after the SVHN failure (specs/K10 docstring). | The branch and T_s were picked per dataset from test outcomes. With the method as stated (T_s = 10), CIFAR-10 LT ties FSG + LA (44.38 vs 44.17). | Headline numbers are not from one pre-specified algorithm | 2 | High | Demonstrated | Implement the gate in code. Run the single specified method (T_s fixed) on all datasets. Report its numbers in the abstract. |
| R3-4 | "F is 0.01 on SVHN and 0.14–0.60 elsewhere; any threshold in 0.02–0.1 gives the same choice" (Abstract l.42; §4.6 l.465-466; K12 spec "CIFAR-10 ~0.14") | Measured feature-space F on CIFAR-10 is 0.063 (D01_kaggle/cifar10_s0.json) and 0.06–0.07 (K13 class_sep_F). The rows branch on CIFAR-10 was never run (LEDGER: NB20/21 NOT RUN; PROGRESS.md §3 "superseded"). | The threshold claim is false for 0.07–0.1. CIFAR-10 sits 0.01 above the cutoff. "Best choice in every regime measured" (l.863-864) was not measured on CIFAR-10. | The gate's justification is fragile | 2 | High | Demonstrated | Correct the F values. Run the rows branch on CIFAR-10. Choose the threshold on a validation protocol. |
| R3-5 | IID: 97.62 / 60.93 / 77.07, above "best of all ten" (Abstract l.48-51; §5.10-5.11) | `best_mean_acc` = per-architecture maximum test accuracy over rounds, including stage I (core.py:780-792, 2276). CIFAR-10 IID final_bal: Ours-A T_s=10 56.82 vs GeFL-F 56.68 (K08 cifar10_iid; K01 summary); best_mean_acc 60.93 vs 59.10. | The CIFAR-10 IID gain (+1.8) vanishes (+0.14) at the final round. The metric selects on test. (It is the paper's metric, but PROGRESS.md says final balanced accuracy is the headline.) | IID superiority claims are not supported by a non-test-selected metric | 2 | High | Demonstrated | Report final-round, or validation-selected (held-out 2 000 images), accuracy for every IID claim |
| R3-6 | Cost: +1.9% time; HWA adds C numbers per round; CSL ≈ 24 KB per round (§5.12 l.876-878; §4.3 l.252-253) | head_time_s 59 s (GeFL-F) vs 680 s (Ours-A T_s=10), MNIST LT. CSL pool P = max_k T_s·n_k (core.py:1783-1787), so labels scale 10× at T_s = 10, and the pool must be shipped (P × 768 floats) or regenerated bit-identically on clients with G + MC + KH. Uncounted one-shot releases: RFF sums C × 6144; MC-S D² + C·k²; ER Σ(C·d_g + d_g²). | The cost section describes the minimal variant, not Ours-A | Efficiency claims are inaccurate. The comparison is not compute-matched. | 2 | High | Demonstrated (compute); suspected (bytes) | Give a full per-client communication and computation table for Ours-A. Add compute-matched baselines (T_s = 10). |
| R3-7 | Every component is necessary (K09; §5.11) | K09_ablation_analysis.md: MC −0.5 to −1.0, KH −0.3 to −0.7, PC on FMNIST −0.67 (p = 0.11); none Holm-significant at n = 3; LA ≈ BBC (finding 4). No BBC, T_s or gate ablation; no CIFAR/SVHN ablation. E32's MC-B variant scores higher than Ours-A. | MC + KH + RFF fingerprints add machinery for about 1 point on easy datasets. Several "final" candidates were tried on the test set. | The pipeline is over-engineered relative to the evidence | 3 | High | Demonstrated | Run ablations on CIFAR and SVHN with ≥ 5 seeds. Collapse LA/BBC into one component. Drop or justify MC/KH. |
| R3-8 | HWA is needed to stop the collapse (§3, §4.1) | Lemma 2 says λ = 0 removes D̄_c. Quick pass NOWD ≈ HWA+LCD (E02_conditioning_collapse_quick). K15 (AdamW / no row decay) NOT RUN (LEDGER.md:619-620). | The simplest fix is a one-line optimiser change, and it has no full-schedule test | HWA's practical novelty and necessity are unestablished | 2 | Medium | Unresolved | Run K15 (3+ seeds, K = 10/50/100) |
| R3-9 | Baselines cover the field (§5.3 table) | Long-tail baselines are only FedAvg (= local), LG-FedAvg and LG-FedAvg + LA, on MNIST/FMNIST (F01). K07 (CIFAR/SVHN baselines) NOT RUN (LEDGER.md:591). No FedLC, FedRS, CCVR, CReFF, Fed3R, FedProto, FedGen, FedETF or FedLoGe, although all are cited (§6). | Long-tail FL methods are discussed but not evaluated | The SOTA claim ("ahead of every other method", Abstract l.37) is unsupported | 2 | High | Demonstrated | Add at least FedLC, FedRS, CCVR/CReFF and Fed3R-style heads on the same FE, plus one heterogeneous-KD method |
| R3-10 | Rare classes held by few sites motivate the work (§1 l.60-61) | tail_holders 6–8 / 10 and 16–24 / 50–100 (runs.jsonl); D01_kaggle holders lists | The tested long tail is global scarcity, not site scarcity | The external validity of the HWA/SecAgg story is untested | 3 | High | Demonstrated | Add a pathological split with tail classes held by 1–2 clients |
| R3-11 | System model (implicit) | Full participation at core.py:805, 1299, 1825; persistent Adam at 1012-1033; HWA fallback at 567 never exercised | No sampling, dropout or asynchrony | Practical FL relevance is unknown | 3 | High | Demonstrated | Partial-participation experiments (10–20% per round) at K = 100 |
| R3-12 | CSL "uses no client message beyond GeFL-F's"; the server already holds the heads (§4.3 l.243, 252) | Code comment: "clients would upload only these probabilities" (core.py:1780-1782) vs server-side evaluation (1800-1804). At K = 10 the aggregated head is one client's head. | The two descriptions differ. The server evaluates individual client models in the clear. | Protocol ambiguity and a privacy implication (forwarded to Reviewer 6) | 3 | Medium | Demonstrated | Specify the CSL protocol: who samples, who evaluates, what is sent |
| R3-13 | ER: "without real data leaving any client" (readout_report.md:104-111; core.py:2100-2105) | Every client runs every architecture's body (core.py:2113-2120) | This contradicts the hardware-heterogeneity premise | If ER is adopted, the heterogeneity motivation weakens | 3 | High | Demonstrated | Discuss it, or give an ER variant computed only by clients of architecture g (and show it still works at K = 10) |
| R3-14 | Server steps use no held-out or test data (§4.6 l.472-473) | BBC fit uses synthetic data only (core.py:2049-2069). Gate uses train counts (2411-2415). MC/KH/PC use sums (1138-1183). Test set only in evaluation (2029-2032). | **Attack fails** for the training pipeline. Exceptions: (a) model, branch and hyperparameter selection was done on the test set across ~1 177 runs with no validation split, although 2 000 held-out images exist (core.py:209); (b) DP clipping uses held-out "public" data (core.py:2269-2272). | The data path is sound; selection is test-contaminated | 3 | High | Demonstrated | Use the held-out split as validation for all design choices. Justify the public-data assumption. |
| R3-15 | Aggregation follows FedAvg | Stage I FE and heads are flat **unweighted** means (core.py:532-559), with client sizes from 85 to over 1 000 | This matches the authors' code (engine_f.py:10-23), but it is not size-weighted FedAvg. Under Dirichlet skew this affects the FE that every method shares. | Minor, but it should be stated | 4 | High | Demonstrated | State it, or ablate weighted FedAvg |

---

## 7. Questions for the authors

1. At K = 10 each architecture has one client. In what sense are the heads "federated"? What do the long-tail results look like when several clients share an architecture on CIFAR-10 and SVHN?
2. Please report GeFL-F(+LA) + federated readout recalibration (ER, or CCVR/Fed3R on the head embeddings) with 3+ seeds and the full schedule. What is the margin of the final method over it?
3. Which single, pre-specified configuration is "the final method" (generator branch, T_s, BBC on or off)? Please give its numbers on all four datasets, with the F-gate executed in code.
4. CIFAR-10 has feature-space F ≈ 0.063. Why does the paper say 0.14? How does the rows branch perform on CIFAR-10?
5. Can all IID claims be reported with final-round or validation-selected accuracy instead of `best_mean_acc`?
6. How do HWA and CSL behave under 10% client sampling at K = 100, and with tail classes held by 1–2 clients?
7. Does removing or decoupling the coupled weight decay (K15), plus LA, match HWA + LA at full schedule?
8. What is the full per-round and per-client communication and computation of Ours-A, including the pool, soft labels, RFF/MC sums and the T_s = 10 synthetic epochs? What do the baselines achieve with the same compute?
9. Who generates the CSL pool, and how do clients obtain the exact pool features the soft labels refer to?
10. Were T_s, β, τ, the KH multiplier, the MC clamps and the F threshold tuned on the test set? Why was the 2 000-image held-out split not used for validation?
