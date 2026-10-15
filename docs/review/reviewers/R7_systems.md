# Reviewer 7: systems review (distributed ML, communication-efficient FL)

**Specialty.** Communication and compute accounting for FL protocols, secure-aggregation (SA) deployment, client and server cost, and scaling.

**Object of review.** GeFL-F with exact-statistics anchoring (ESA, "Ours-A": PC-VAE + MC + KH + LA + CSL, T_s = 10, gated BBC). Also reviewed: the MC-S/MC-B variants, KME-Gen and exact readout (ER).

Sources:
- `docs/paper_draft.md` (all of it)
- `docs/professor_report/build_lean.js` (§1.3–1.4, 4.2–4.9, 6.1)
- `docs/research/readout_report.md`
- `docs/kme_gen.md`
- `experiments/core.py`
- all `experiments/results/*/runs.jsonl`

Every parameter count below comes from instantiating the models in `core.py` on CPU (scripts in `r7_work/count.py`, `pool.py`, `calc.py`, `timing.py`). Client sizes come from re-running `make_partition` with each dataset's class counts. Because the RNG stream depends only on array lengths, this reproduces `n_train` = 5,996 for MNIST LT exactly.

## Recommendation

**Major revision. Confidence: high** for the communication numbers (counted from code), **medium-high** for wall-clock (measured on mixed hardware).

- **The communication story mostly survives.** The exact-statistics releases are tiny compared with GeFL-F's generator traffic. The "≈0.3 MB once" figure is arithmetically correct for the four basic releases. Whole-training traffic grows by about 8% on MNIST and 24–28% on CIFAR-10. The CIFAR-10 growth comes almost entirely from the larger PC-VAE, which the draft never reports.
- **The compute story does not survive.** The only cost section (§5.12, "+1.9%") measures the minimal variant HWA + LA, not Ours-A. Ours-A at T_s = 10 is 2.2–2.8× GeFL-F end to end and 5–14× in Stage III. Kernel herding causes 83–88% of that.
- **Several protocol pieces are under-specified or contradict each other:**
  - who computes the CSL consensus, and how the synthetic pool reaches clients;
  - extra sequential rounds;
  - exposure of single-client models under ER.
- **Some claimed costs are not like-with-like:** KME-Gen's "1/3000", the per-architecture ER figure, and MC-S, whose D×D release is missing from the accounting.

None of this is fatal to the accuracy claims. Taken together, it is fatal to any claim that the method is "practically free".

---

## 1. Transparent cost table (per client, fp32 = 4 B)

Unless noted, the setting is K = 10, IF = 100, Dir(0.5), C = 10.

- Feature dimension D: 768 for MNIST, FMNIST and SVHN (3×16×16); 2,560 for CIFAR-10 (10×16×16).
- Rounds T_FE / T_KA / T_TN: 20 / 100 / 50 for MNIST; 50 / 200 / 100 for CIFAR-10 (`core.py:47–63`).

| # | Item | Formula | MNIST (D = 768) | CIFAR-10 (D = 2,560) | When | Source |
|---|---|---|---|---|---|---|
| 1 | Stage I: FE + own head, up + down | 2·T_FE·(\|head\| + \|φ\|) | avg head 36.5k → **5.8 MB** | avg head 37.7k → **15.1 MB** | per round | counted; `build_lean.js:188` |
| 2 | Stage II: GeFL-F CVAE-F | \|G\| per round, each direction | 5,546,659 = 22.2 MB/rd; up + down 100 rds = **4.44 GB** | 5,613,294 = 22.5 MB/rd; 200 rds = **8.98 GB** | per round | `core.py:303–360`; claim "22 MB" ✓ (`build_lean.js:198,208`) |
| 3 | Stage II: PC-VAE (Ours-A) | \|G\| per round | **5,937,315** = 23.7 MB/rd; **4.75 GB** (+0.31 GB) | **6,928,622** = 27.7 MB/rd; **11.09 GB (+2.10 GB, +23%)** | per round | `core.py:362–440`. Claim "5.94 M vs 5.55 M" is MNIST-only (`build_lean.js:350,439`) |
| 4 | Basic statistics: n, Σh, Σ‖h‖², Σφ(h) | C(1 + D + 1 + 6144) | 69,140 = **0.277 MB** ✓ | 87,060 = **0.348 MB** ✓ | once, but **two sequential SA rounds**: the RFF bandwidth s0 is computed from the first release | `core.py:1138–1168, 1465–1474` (s0 at `:1470`); claim `build_lean.js:340` |
| 5 | MC/KH parameters needed by whoever samples: μ_c, m̃_c, scale, Φ-means, φ seed | C(2D + 1 + 6144) | 76.8k = 0.31 MB | 112k = 0.45 MB | once, down | `core.py:1530–1561, 1477–1527` |
| 6 | CSL consensus, P×C per round (option B: clients upload prediction sums, server returns labels) | P = T_s·max_k⌊n_k/B⌋B (K = G); P = ΣT_s·n_k (K > G) | P/fake epoch: IID 576 (**23 KB** ✓), **LT 1,173 (47 KB)**. T_s = 10: **469 KB/rd → 23.5 MB per direction over Stage III** | LT P = 4,309 (**172 KB/epoch**). T_s = 1: 17 MB/stage; **T_s = 10: 1.72 MB/rd → 172 MB per direction** | per round, synchronous | `core.py:1780–1823`; claims `paper_draft.md:252–253`, `build_lean.js:393,438` |
| 6a | CSL option A (paper text: server holds heads) | head up per round + P×C down | +7.3 MB up (50 × 146 KB) | +15 MB up | per round | `paper_draft.md:243–253` vs `build_lean.js:203` |
| 6b | Synthetic pool, if it is shipped rather than regenerated from a seed | need_k × D per round | avg client, T_s = 10: **18.4 MB/rd → 0.92 GB** | LT, T_s = 10: **189 MB/rd → 18.9 GB** (2× GeFL-F's whole Stage II) | per round | computed |
| 7 | BBC | heads up once + C offsets down | ≈0.15 MB | ≈0.15 MB | once | `core.py:2049–2071` |
| 8 | ER: download all G bodies, upload Σ_g (C·d_g + d_g² + C) | d_g = 1024, 512, 640, 320, 256, 320, 160, 128, 100, 80 | bodies 342k (1.4 MB) down; **2,084,540 numbers = 8.3 MB up** (4.2 MB packing the symmetric part); largest architecture 1,058,826 ✓ | same; d_g does not depend on D | once (extra round) | `core.py:2100–2131`, `HEADER_WIDTHS` `core.py:66`; claim `build_lean.js:556` |
| 9 | MC-S / MC-B release | pooled ΣhhᵀD² + broadcast B (D·k) + C·k + C·k², k = 256 | 589,824 + 657,920 = **1.25 M = 5.0 MB**; B down 0.79 MB; **+2 sequential rounds** | 6,553,600 + 657,920 = **7.21 M = 28.8 MB**; B 2.6 MB | once | `core.py:1570–1633`; paper mentions only "C k²" (`paper_draft.md:345–346`) |
| 9′ | (If MC-B used per-class D×D covariances, as the review prompt hypothesised) | C·D² | 5.9 M (23.6 MB) | 65.5 M (262 MB) | – | **not what the code does**; it uses per-class k×k in a shared subspace |
| 10 | KME-Gen generator, broadcast to clients | \|G_KME\| | 3,159,363 = **12.6 MB** down | 4,101,450 = 16.4 MB | once | `core.py:1209–1246` |

**Whole-training totals per client, up + down.**

| Pipeline | MNIST LT, K = 10 | CIFAR-10 LT, K = 10 |
|---|---|---|
| GeFL-F | 5.8 MB + 4,437 MB ≈ **4.44 GB** | 15 MB + 8,981 MB ≈ **9.0 GB** |
| Ours-A (CSL option B, pool regenerated from a seed) | 5.8 + 0.28 + 4,750 + 0.31 + 47 + 0.15 ≈ **4.80 GB (+8%)** | T_s = 1: ≈ **11.14 GB (+24%)**; T_s = 10: ≈ **11.45 GB (+27%)** |
| + ER | +9.7 MB | +9.7 MB |
| Ours-A if the pool is shipped | ≈ 5.7 GB (+29%) | T_s = 10: ≈ **30 GB (3.3×)** |
| KME-Gen (E24 config, T_s = 1) | 5.8 + 0.28 + 12.6 + 4.7 ≈ **23 MB** (≈1/190 of GeFL-F) | – |

Stage II is 97–99% of every total, so the statistics releases really are negligible. The real increments are PC-VAE's size (CIFAR-10) and CSL's per-round traffic and synchronisation.

---

## 2. Measured runtimes (mean of 3 seeds; s)

`time_s` cannot be compared across methods. The runner caches stages I and II per experiment file and resets the clock at the start of each run (`core.py:2223, 2421`), so a "+CSL" row shows 65 s. I therefore sum `fe_time_s + gen_time_s + head_time_s`. `head_time_s` includes MC/KH construction and Stage III but not the post-hoc diagnostics (`core.py:2360–2395`).

| Setting | GeFL-F fe / gen / head = total | Ours-A T_s = 10 fe / gen / head = total | Stage III ratio | End-to-end ratio |
|---|---|---|---|---|
| MNIST LT K = 10 (local; E20 vs E25) | 58 / 261 / 64 = 383 | 54 / 287 / **680** = 1021 | **10.6×** | **2.67×** |
| FMNIST LT K = 10 (E20 vs E25) | 74 / 288 / 70 = 432 | 72 / 280 / **610** = 962 | 8.7× | 2.23× |
| MNIST IID K = 10 (F10 vs E25) | 58 / 259 / 66 = 383 | 58 / 268 / 328 = 654 | 5.0× | 1.71× |
| SVHN LT K = 10 (Kaggle K01 vs K08) | 111 / 579 / 110 = 800 | 100 / 592 / **1565** = 2257 | 14.2× | 2.82× |
| MNIST LT K = 100 (F04 vs E22) | 92 / 798 / 150 = 1040 | 87 / 827 / **2011** = 2925 | 13.4× | 2.81× |
| CIFAR-10 LT K = 10 (K08) | not available (K01 cifar10 has only `summary.md`) | 123 / 2966 / 2032 | – | unresolved |

- **Ours-A at T_s = 1** on MNIST LT: 54 / 287 / 109 = 450 s, or 1.17× GeFL-F.
- **Share of KH** (K09 ablation, same Kaggle job): Stage III takes 878 s with KH and 106 s without it on MNIST (KH = **88%**), and 647 s vs 113 s on FMNIST (**83%**). The cause is code-level: `HerdedGen.sample` (`core.py:1508–1514`) runs one GEMV, one argmax and one `int()` host sync per selected sample, which gives P × T_TN sequential steps (MNIST LT T_s = 10: about 586k; CIFAR-10 LT T_s = 10: about 4.3M).
- **KH FLOPs** (RFF of 4n candidates, 6144-dim, plus herding): about 0.68 TFLOP per round, or 34 TFLOP per Stage III (MNIST LT T_s = 10), and about 6.3 TFLOP per round, or 630 TFLOP per Stage III (CIFAR-10 LT T_s = 10).
- **Client-side head training per round:** (T_s + T_r)·n_k sample passes, which is 15 n_k against 6 n_k for GeFL-F (**2.5×**), before any pool generation.
- **PC-VAE in Stage II:** wall-clock +2% to +16% (287 vs 247–261 s on MNIST LT; 592 vs 579 s on SVHN).
- **GPU peak memory** (`gpu_peak_mb`, indicative only): GeFL-F 1.5 GB against Ours-A 2.0–2.9 GB locally. On Kaggle: 2.5 GB against 8–14 GB (SVHN). One KH chunk alone holds a 16,384 × 6,144 fp32 RFF matrix (≈0.4 GB). With T_s = 10, the CIFAR-10 LT pool on a client is 43,090 × 2,560 × 4 B ≈ 0.44 GB.
- **§5.12 is stale.** "373 s vs 366 s (+1.9%)" (`paper_draft.md:876`) reproduces exactly as F01 "+HWA+LA" (55 + 257 + 62 = 374) against GeFL-F (55 + 252 + 60 = 367). That is the minimal variant without CSL, not Ours-A.

---

## 3. Scaling analysis

| Quantity | K | C | D | Comment |
|---|---|---|---|---|
| Basic stats C(D + 6146) | flat per client; SA server cost O(mK²) | linear: C = 100 → 2.8 MB, C = 1000 → 27.6 MB (MNIST D) | linear | Stays small next to the generator |
| PC-VAE | – | **independent of C** (no C×512 rows; CVAE-F has C×512) | proto_proj D×512 plus 2× encoder input | Favourable in C, unfavourable in D (+23% at D = 2,560) |
| CSL labels P×C | for K > G, P = T_s·N_total, so per architecture P grows with total data | **linear** | – | C = 1000 at T_s = 10, MNIST N: 48 MB per round → 2.4 GB per Stage III, **comparable to all of Stage II** |
| KH compute | ∝ P ∝ T_s·N_total (head time 680 s → 2011 s from K = 10 to 100) | herding term ∝ P²/C per chunk; RFF term flat | RFF ∝ D·6144 | At C = 1000, chunks of 4,096 give about 4 samples per class per chunk, so herding degenerates. Not tested |
| ER Σ_g(C·d_g + d_g²) | per client flat | C·Σd_g = C·3,540 | – | C = 1000 → 5.6 M (22 MB) per client. Every client needs G bodies |
| MC-S D² + C·k² | – | linear in C·k² | **quadratic** | CIFAR-10 already 28.8 MB; D = 8,192 would be 268 M (1.07 GB) |
| Rounds and barriers | CSL adds a synchronous barrier in every Stage III round; the stats add 2 SA instances; MC-S 2 more; ER 1 | – | – | Each Bonawitz SA instance is about 4 message rounds with dropout handling |

---

## 4. Claims against full requirements: contradiction or acceptable trade-off

| Claim (location) | Finding | Verdict |
|---|---|---|
| "≈0.3 MB per client, once" (`build_lean.js:340, 437`; `paper_draft.md:290` "one upload per client") | Size ✓ (0.277 / 0.348 MB). But it takes two sequential releases (s0 depends on the first, `core.py:1470`), each under SA. MC-S, ER and CSL are excluded. | Acceptable trade-off; wording inaccurate ("once" means 2 SA rounds) |
| "GeFL-F already sends 22 MB per round" (`build_lean.js:198, 340`) | ✓ 22.19 MB uplink | Correct |
| "5.94 M params ≈ 23.7 MB (GeFL-F: 22 MB)" (`build_lean.js:350, 439`) | ✓ MNIST/FMNIST/SVHN. **CIFAR-10: 6.93 M vs 5.61 M, +2.1 GB per client over Stage II.** | Omission; CIFAR-10 cost understated |
| "≈24 KB per fake epoch" (`build_lean.js:393, 438`); "about 24 KB per round here" (`paper_draft.md:253`) | True only for MNIST/FMNIST IID at K = 10. LT: 47–51 KB; CIFAR-10 LT: 172 KB; K = 100 MNIST LT: 206 KB per fake epoch; ×10 at T_s = 10. | Minor understatement |
| "CSL uses … no client message beyond GeFL-F's. The server broadcasts the pool seed and the P×C soft labels" (`paper_draft.md:252–253`) vs "At 10 clients … nothing is exchanged in this stage" (`build_lean.js:203`) and "Clients upload only sums of predictions" (`build_lean.js:393`) | **Contradiction.** At K = 10, GeFL-F's Stage III has no traffic. CSL therefore needs either per-round head uploads (option A, which exposes each single-client head to the server) or per-round P×C prediction uploads under SA (option B). Both add a per-round synchronous barrier to a stage that was communication-free. The code (`core.py:1797–1805`) is a centralised simulation that does not settle which. | Contradiction (2) |
| Pool "drawn from a public seed" (`paper_draft.md:244`) | The pool is MC + KH output. Regenerating it from a seed requires every client to reproduce the KH argmax chain (`core.py:1508–1514`) bit-exactly across hardware, and under option B the architecture's representative must generate the full P-sample pool with KH (34–630 TFLOP per Stage III). Shipping features instead costs 0.9 GB (MNIST) to 18.9 GB (CIFAR-10) per client. | Unaccounted; suspected non-determinism |
| "Summed over all stages … 373 s vs 366 s (+1.9%)" (`paper_draft.md:876–878`) | That measurement is HWA + LA (F01). Ours-A at T_s = 10 is 2.2–2.8× end to end and 5–14× in Stage III. | **Contradiction** for the method actually proposed (2) |
| "Communication adds C numbers per client per round for HWA, and one broadcast of soft labels" (`paper_draft.md:877–878`) | Ours-A has no HWA (except on the SVHN branch). CSL is a per-round exchange, not "one broadcast". PC-VAE size and stats are omitted. | Stale |
| KME-Gen "at 1/3000 of the communication" (`build_lean.js:296`; "of the generator communication", `diagnostics.md:192`); "one upload of about 0.3 MB replaces 100 rounds" (`paper_draft.md:454`, `kme_gen.md:71`) | Arithmetic does not give 3000 (uplink 2.22 GB / 0.28 MB ≈ 8,000; up + down ≈ 16,000). It is also not like-with-like: it omits the 12.6 MB generator broadcast (generator channel ≈ **1/340**), Stage I and CSL (whole training ≈ **1/190**), and the second SA round for s0. It also compares a dropped, less accurate method (89.9 vs 94.8) against GeFL-F's training channel. | Not like-with-like (3) |
| ER: "one extra round; under 1.1 M numbers per architecture for the largest head" (`build_lean.js:556`; `readout_report.md:111–112, 179`) | Per architecture ✓ (1,058,826). **Per client 2.08 M (8.3 MB)**, because every client embeds with all 10 bodies (`core.py:2113–2120`) and must first download them. At K = 10 each body is trained on one client's data, so broadcasting it exposes that client's model to all others. This is not discussed; the "disclosure" sentence covers only embedding statistics. ER exists only as a post-hoc diagnostic (`core.py:2401–2405`). | Acceptable cost but misleading framing; privacy omission (3) |
| MC-S "C k² numbers; k = 256" (`paper_draft.md:345–346`) | It also needs the pooled D×D Σhhᵀ (589,824 / 6.55 M) and a broadcast basis, giving 5.0 MB / 28.8 MB and 2 extra sequential rounds. Under DP the code silently falls back to MC (`core.py:2362`), so the DP table never covers MC-S. | Omission (3) |
| T_s = 10 synthetic epochs | 2.5× client head-training passes plus pool generation. On CIFAR-10 and SVHN it **lowers** final balanced accuracy (`paper_draft.md:835–838`), so there the cost buys nothing. | Acceptable trade-off if reported per dataset |

---

## 5. Evidence ledger

Severity runs from 1 (fatal) to 4 (minor). Status is demonstrated (Dem.), suspected (Susp.) or unresolved (Unres.).

| ID | Claim and location | Evidence | Reasoning | Consequence | Sev. | Conf. | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| R7-1 | "+1.9% cost" (`paper_draft.md:876`) | §2 table; F01 sums reproduce 367/374 | Measured on HWA + LA, not Ours-A | Practical-cost claim for the proposed method is unsupported | 2 | High | Dem. | Report per-stage time and FLOPs for Ours-A (T_s = 1 and 10) against GeFL-F on matched hardware |
| R7-2 | CSL adds no client message (`paper_draft.md:252`) vs `build_lean.js:203, 393` | Code centralised (`core.py:1797–1805`) | K = 10 GeFL-F Stage III has zero traffic | New per-round barrier and messages; protocol undefined | 2 | High | Dem. | Specify the CSL protocol (who holds heads, what is uploaded) and count it |
| R7-3 | Pool from a public seed (`paper_draft.md:244`) | `core.py:1508–1514` argmax chain; calc §1 row 6b | Bit-exact KH replay across devices is fragile; shipping costs GBs | Either heavy client compute or 0.9–19 GB per client | 2 | Med | Susp. (determinism) / Dem. (sizes) | State who samples; test cross-device reproducibility or budget the pool download |
| R7-4 | KH is "no training; one matrix product" (`build_lean.js:375`) | K09: KH = 83–88% of Stage III; 586k–4.3M sequential host syncs | O(P) sequential selection with 4× oversampling | Dominant compute cost; memory 0.4 GB per chunk | 2 | High | Dem. | Report KH cost; batch the herding; ablate KH against cost |
| R7-5 | PC-VAE 5.94 M vs 5.55 M (`build_lean.js:350`) | count.py: CIFAR-10 6.93 M vs 5.61 M | proto_proj D×512 and 2D encoder input scale with D | +2.1 GB per client on CIFAR-10 (+23%) | 3 | High | Dem. | Give per-dataset sizes and whole-training bytes |
| R7-6 | "≈0.3 MB once / one upload" (`build_lean.js:340`, `paper_draft.md:290`) | `core.py:1470` s0 from first release | Two dependent SA instances | Extra coordination; SA cost not paid by the baseline | 4 | High | Dem. | Say "two SA rounds"; account for SA overhead |
| R7-7 | MC-S "C k²" (`paper_draft.md:346`) | `core.py:1585–1605` | Pooled D×D omitted | 5.0 / 28.8 MB; quadratic in D; DP not covered (`core.py:2362`) | 3 | High | Dem. | Add the full release size, rounds and DP status of MC-S |
| R7-8 | KME-Gen "1/3000" (`build_lean.js:296`) | calc: 8,000 / 16,000 / 340 / 190 | Ratio inconsistent and excludes generator download and other stages | Overstated efficiency of a dropped variant | 3 | High | Dem. | Compare whole-training bytes, both directions |
| R7-9 | ER "1.1 M per architecture" (`build_lean.js:556`) | count.py: Σ = 2.08 M per client; bodies broadcast | Per-architecture framing hides a 10× sum; single-client models exposed | Model-privacy issue at K = G | 3 | High | Dem. | State per-client totals; discuss body exposure; consider an SA-friendly alternative |
| R7-10 | T_s = 10 is a free gain | E25/K08 times; `paper_draft.md:835` | 2.5× head passes plus 10× pool; lower final accuracy on CIFAR-10 and SVHN | Cost-benefit negative on hard data | 3 | High | Dem. | Report the accuracy–cost Pareto per dataset |
| R7-11 | Scalability in C (implicit) | §3 | CSL P×C and ER C·Σd_g are linear; MC-S D²; KH degenerates per chunk | At C = 1000 CSL ≈ Stage II size | 3 | Med | Unres. | One C = 100 experiment (CIFAR-100) with byte counts |
| R7-12 | Timing methodology | `core.py:2223, 2421` caching | `time_s` excludes cached stages | Naive `time_s` comparisons wrong (the draft's 366/373 used sums correctly) | 4 | High | Dem. | Always report stage sums; include diagnostics/ER/BBC time |
| R7-13 | CIFAR-10 cost vs GeFL-F | no GeFL-F CIFAR-10 `runs.jsonl` | Cannot compute the matched ratio | Gap in evidence | 4 | High | Unres. | Copy K01 cifar10 `runs.jsonl` back (no rerun needed) |
| R7-14 | Server sees heads (CSL option A, BBC) | `core.py:1797–1805, 2049` | At K = 10 a head is one client's model | Weaker privacy than GeFL-F's Stage III | 3 | Med | Susp. | Specify whether heads leave clients; if so, add to the disclosure table |

---

## 6. Questions to the authors

1. In deployment, who generates the CSL pool, the server or each client? If clients, how is bit-exact reproduction of MC + KH (argmax ties, GPU non-determinism) ensured? If the server, how are P×D features delivered, and at what cost?
2. At K = 10, does any head leave its client during Stage III? Reconcile `paper_draft.md:243–253` with `build_lean.js:203, 393`.
3. Please give a whole-training byte count (up and down) for GeFL-F, Ours-A, Ours-A + ER and KME-Gen on MNIST and CIFAR-10. How was "1/3000" computed?
4. Please give the wall-clock and FLOPs of Ours-A at T_s = 10 against GeFL-F on the same hardware, with KH separated out. Can herding be batched (for example, a blockwise Frank–Wolfe) to remove the per-sample host sync?
5. For ER, all ten bodies are broadcast to every client, and at K = 10 each is trained on one client's data. How does this fit the privacy model? Is ER part of the method or only a diagnostic?
6. Why is the MC-S pooled D×D release absent from the communication table, and why is MC-S disabled under DP (`core.py:2362`)?
7. How do CSL label traffic and KH behave at C = 100 or 1000, where 4,096-sample chunks give only a few samples per class?
8. SA overhead: every "exact sum" assumes Bonawitz-style SA, which the GeFL-F baseline does not pay. Please quantify the number of SA instances and their rounds.

*Scratch scripts and their outputs are in `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r7_work/`.*
