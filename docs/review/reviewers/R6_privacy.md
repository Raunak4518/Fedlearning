# Reviewer 6: Privacy, secure aggregation, differential privacy, accounting

**Specialty:** federated security (threat models, secure aggregation), differential privacy (Gaussian/Laplace mechanisms, zCDP accounting), and privacy attacks (membership inference, reconstruction).

**Recommendation:** *Major revision.* The privacy claims must be narrowed before the paper goes out. The work itself is not invalidated.

**Confidence:** high on the accounting and the code findings. Medium on the attack results, which use surrogate features rather than the trained feature extractor.

**Bottom line.**
- The zCDP arithmetic in E26 is correct and the code matches it: σ, ρ split, conversion, sensitivities and gate counts all check out.
- The accuracy numbers (90.36/85.06 and 72.77/65.59) reproduce exactly from `runs`/`summary.md`.
- But the guarantee covers only one example-level, central-model release of four statistics. These parts of the pipeline have no DP:
  - the feature extractor (Stage I FedAvg);
  - the generator (FedAvg every Stage-II round);
  - the heads;
  - the per-round CSL prediction sums;
  - the second-moment releases (M, Z_c, Σee^T), which are explicitly switched *off* in DP mode;
  - the BBC gate, which reads exact counts.
- Under DP, MC and KH are gated off for the rarest classes. Those classes then reach the heads only through the non-DP FedAvg generator.
- Several sentences ("all ESA releases made DP", "every statistic made differentially private", "formal privacy is cheap", "GeFL-F, which carries no formal guarantee") imply an end-to-end guarantee that does not exist.
- Secure aggregation is assumed, not implemented. DP noise is added centrally in plaintext. No distributed-noise, dropout or collusion analysis is given.
- On the exact-statistics method (the headline numbers), a simple tracing attack on the kernel-mean release identifies members of the 24-sample rarest class with AUC 0.77–0.93 (surrogate features).

---

## 1. Threat model: stated vs implemented

**Stated (scattered; there is no explicit threat-model section in either manuscript).**
- "Secure aggregation [44] lets a server learn only the sum of client vectors" (paper_draft.md:959).
- "The server sees the totals and never a client's term" (paper_single_idea.md:135).
- "they reveal nothing beyond federation-level totals" (paper_single_idea.md:19).
- "Our method keeps GeFL-F's FE and privacy model" (paper_draft.md:737).
- The adversary type (honest-but-curious server? malicious? colluding clients?), the dropout model, and the DP neighbouring relation are never stated.

**Implemented.**

| Aspect | What the code does | Evidence |
|---|---|---|
| Secure aggregation | **Not implemented.** Sums are computed in plaintext over all clients (`torch.cat(feats)` then `index_add_`). "secure aggregation" appears only in docstrings. | `E26_private_anchoring.py:1162-1168`, `:1180-1183`; `grep -i "secure\|bonawitz\|mask" experiments/core.py` → docstrings only |
| Where DP noise is added | Centrally, on the exact aggregate, by the simulator (trusted-curator model). | `E26:1169-1174`, `:1184-1186` |
| Noise randomness | Deterministic, guessable seed `9001 + 97*seed + int(eps*1000)`, using torch's non-cryptographic PRNG and floating-point Gaussians (Mironov 2012). Acceptable for simulation, but not as deployed DP. | `E26:2294` |
| Clip bound | 90th percentile of feature norms on a 2,000-image "public" pool. The pool is carved from the *same MNIST/FMNIST training set* (200 per class). It is passed through the FE, which was trained non-privately on client data. | `E26:219-229`, `:2289-2292` |
| Heads | Averaged within architecture groups. With K = 10 and M = 10 architectures, each group has **one client**, so "aggregated head" = that client's head. | `E26:258-261`, `:1844-1848`, `:1993` |

**Collusion and dropout.**
- Bonawitz et al. protects against an honest-but-curious server colluding with fewer than t clients, and tolerates dropouts. Nothing here depends on that protocol, because it is not run.
- If the noise were distributed (each client adds a σ/√K share, the only way to combine continuous Gaussian DP with SecAgg without a trusted party), then t colluding or dropped clients remove their shares. ρ then scales by K/(K − t).
- At ε = 8 (δ = 1e-5) the realised ε becomes 8.49 / 9.81 / 11.93 / 32.5 for t = 1 / 3 / 5 / 9 (`r6_work/`, computed below). At ε = 2: 2.11 / 2.41 / 2.88 / 6.87.
- SecAgg also works over a finite group, so continuous Gaussian noise must be replaced by a discrete Gaussian or Skellam mechanism with modular-wraparound analysis (Kairouz–Liu–Steinke 2021; Agarwal et al. 2021). Not discussed.

**Single-holder classes (task A).**
- I re-ran the exact `make_partition` logic on the real MNIST/FMNIST labels (`r6_work/part.py`; IF = 0.01, Dir(0.5), seeds 0–2).
- **No tail class has a single holder in any evaluated partition.** The rarest class (24 samples) has 6–8 holders at K = 10 and 21–24 at K = 100.
- The largest single-client share of a tail class is 0.24–0.72 at K = 10 (FMNIST seed 2, class 7: one client holds 48 of 67).
- So the single-holder failure the authors admit (paper_draft.md:995-996, paper_single_idea.md:423) does **not** occur in their experiments. It is a deployment risk, not a demonstrated leak.
- Several holders contribute only one sample: e.g. MNIST seed 0, class 9 holders hold [9, 8, 2, 1, 1, 1, 1, 1].
  - A server that colludes with the other holders, or that sees a SecAgg round with all but one holder dropped, learns that one sample's feature vector exactly.

**Head exposure is inherited, but it contradicts the SecAgg framing.**
- With one client per architecture, any per-architecture head average the server holds is an individual client's head. SecAgg over a group of one gives no protection.
- build_lean.js:210 says heads are uploaded "only if an architecture has several clients".
- But paper_draft.md:242-243 says "The server already holds every architecture's aggregated head" (for CSL).
- And readout_report.md:170 (ER) says "server broadcasts the bodies e_1 .. e_G of all architectures". With K = 10 that sends each client's private head body to every other client: a new disclosure beyond GeFL-F.

---

## 2. Table of every release

| # | Release | When | Size (MNIST, K = 10) | Protected by (stated) | DP in E26? |
|---|---|---|---|---|---|
| R1 | FE φ (+ heads) FedAvg | every Stage-I round | 36–390 + 5k–110k numbers/client | "as GeFL-F" | **No** |
| R2 | Per-class counts n_kc | once after Stage I | C = 10 | SecAgg; optional DP | Yes (Gaussian, Δ = 1) |
| R3 | Per-class feature sums S_kc | once | C × 768 | SecAgg; optional DP | Yes (Δ = clip) |
| R4 | Per-class squared-norm sums Q_kc | once | C | SecAgg; optional DP | Yes (Δ = clip²) |
| R5 | Per-class RFF sums Φ_kc | once | C × 6144 | SecAgg; optional DP | Yes (Δ = √2) |
| R6 | Pooled second moment M = Σhh^T, class Z_c = Σzz^T (MC-S, rows branch) | once | 768² + C·256² ≈ 1.25 M | SecAgg | **No: MC-S disabled under DP** (`E26:2382 ... and not _STAT_DP`) |
| R7 | HWA weighted rows Σ E(n_kc) w_c^(k), Σ E(n_kc) (rows branch / minimal variant) | every Stage-II round | generator rows | SecAgg; Laplace on counts (K02) | Counts only (Laplace ε, drawn once, `E26:1314-1315`). Rows: no |
| R8 | Generator weights (PC-VAE or CVAE-F) FedAvg | every Stage-II round (100 rounds) | ≈ 5.5–5.9 M numbers/client | "same as GeFL-F" | **No** (paper_draft.md:632) |
| R9 | CSL prediction sums on shared pool | every Stage-III round (100 rounds) | P × C | "sums only" | **No** |
| R10 | Head weights | Stage III, per group | 5k–110k | "unchanged" | **No** |
| R11 | Exact readout (ER): per-architecture Σe, n_c, Σee^T; requires broadcasting all head bodies | once (optional) | Σ_g (C d_g + d_g²) ≤ 1.1 M | SecAgg | **No** (diagnostic in code, runs with exact stats `E26:2422-2424`) |
| R12 | BBC gate: global max/min count > 2 | end | 1 bit | "post-processing" | **No: reads exact `ctx.part["counts"]`** (`E26:2433`) |
| R13 | Generator gate F | once | 1 bit | "post-processing" | Computed from **exact** stats (`E26:2272-2282`). In E26 the branch is forced to PC, so this is unused there |

---

## 3. Accounting verification (task B)

**Mechanism** (`E26:1143-1196`, `:2288-2295`; identical in `core.py:1129-1176`, `:2251-2275`).

**Privacy unit.**
- Single example, add/remove neighbouring (implied by the sensitivities; never stated).
- Model: central DP over the *union* of all clients' data.

**Sensitivities.** Each example lies in exactly one class, so one example changes one row of each C × d release matrix. The L2 sensitivity of the whole matrix therefore equals the per-row bound, and no parallel-composition argument is needed. Per release:
- counts: 1 ✓
- S (after clipping to C): C ✓
- Q (after clipping): C² ✓
- RFF: ‖φ‖² = (2/D) Σcos² ≤ 2, so √2 ✓ (computed on unclipped h, which is fine because φ is bounded regardless).

**Noise.**
- Each release gets σ = sig1·Δ with sig1 = √(2/ρ).
- Gaussian zCDP gives Δ²/(2σ²) = 1/(2 sig1²) = ρ/4 per release, so ρ in total over the four ✓.

**Conversion.**
- Code: `zcdp_rho(eps, delta) = (sqrt(L+eps) - sqrt(L))**2`, L = log(1/δ).
- This inverts Bun & Steinke 2016 Prop. 1.3, ε = ρ + 2√(ρ log(1/δ)): set u = √ρ; then u² + 2u√L − ε = 0, so u = √(L+ε) − √L ✓.
- It is slightly conservative; the tighter conversion of Canonne–Kamath–Steinke 2020 would give a smaller ε.

**Numbers** (`r6_work/acct.py`).

| target | ρ | sig1 | per-release ρ | gate n_c ≥ (MC, d = 768) | gate n_c ≥ (KH, D = 6144) | classes with MC / KH | reported |
|---|---|---|---|---|---|---|---|
| (8, 1e-5) | 1.0491 | 1.381 | 0.2623 | 77 | 216 | 7 / 5 | 7 / 5 ✓ |
| (2, 1e-5) | 0.0800 | 4.999 | 0.0200 | 277 | 784 | 5 / 3 | 5 / 3 ✓ |

- Class sizes are [2417, 1449, 868, 520, 312, 187, 112, 67, 40, 24].
- The relative noise norm on the rarest class mean is 1.59 × clip at ε = 8 and 5.77 × clip at ε = 2. On the RFF mean it is 4.5 and 16.3.
- So the tail-class prototypes that condition the PC-VAE are dominated by noise (`class_prototypes` has no gate).
- δ = 1e-5 < 1/N = 1/5996 ✓.
- Composition: anchor_stats is called by several consumers, but with a fixed seed it returns the identical release, so there is no hidden re-release ✓. RFF is released once with seed + 1 ✓.

**Caveats that change the number.**
1. *Neighbouring relation.*
   - Under replace-one, the sensitivities double for S and RFF (2C, 2√2) and become √2 for counts.
   - The worst case is ρ × 4, i.e. the claimed (8, 1e-5) becomes about (18.1, 1e-5) and (2, 1e-5) becomes about (4.2, 1e-5).
   - The paper must state add/remove.
2. *Unit is the example, not the client.*
   - Group privacy for a client with n_k samples gives ρ·n_k².
   - The top holder of MNIST class 9 (9 samples) gets ε ≈ 148 at nominal ε = 8. A 600-sample client's guarantee is vacuous.
   - Acceptable for cross-silo FL if stated. The FL-DP literature the paper cites ([47, 48]) is *client*-level.
3. *Conditioning on non-private objects.*
   - The release is DP **conditional on** the FE, which is a non-DP function of the same data (R1).
   - The clip bound is derived from that FE on an in-distribution pool.
   - Under adaptive composition, a non-DP stage makes the overall pipeline non-DP.
4. *Distributed or SecAgg deployment.* See §1 (ρ·K/(K − t); discrete noise).

**Laplace-on-counts (K02).**
- `class_weights` adds Laplace(1/ε) to every entry of each client's histogram. That gives L1 sensitivity 1 under add/remove, so ε-DP per client histogram ✓.
- The noise is drawn once per run (`E26:1314-1315`) ✓. Note: the K = 100 runs never applied it.
- But that is the *only* DP in the K02 minimal variant. The rows the counts weight are trained and uploaded with no DP every round, and a non-holder's row update is pure decay, so holdership leaks through the update itself to any party that sees it.

---

## 4. Pipeline-level coverage (task C)

**What is and is not DP in "Ours-A, ε = 8/2".** Only R2–R5 are DP. R1, R8, R9 and R10 are not. R12 is not post-processing of the DP release. R6 and R11 are not covered, which is why MC-S is disabled in DP mode.

**Consequence for the tail classes.**
- At ε = 2, MC is off for classes 5–9 and KH for classes 3–9.
- The rarest classes therefore reach the heads through the PC-VAE alone. Its decoder and encoder are trained by FedAvg on these classes' exact, unclipped features, with no noise.
- So the "privacy" for exactly the samples that need it most is provided by nothing formal.

**Comparison with GeFL-F.**
- Ours-A (DP) releases a **superset** of GeFL-F's messages: GeFL-F + DP statistics R2–R5 + non-DP CSL sums R9.
- Its formal privacy is therefore no better than GeFL-F's (neither has an end-to-end guarantee), and its disclosure is strictly larger.
- Labelling the baseline row "GeFL-F (no formal privacy)" next to "DP" rows implies the opposite.

**Label-distribution leakage.**
- Counts (R2) reveal the federation's global label histogram exactly in non-DP mode. That is fine under SecAgg.
- But HWA weighting needs per-client E(n_kc) multiplied into rows: client-side, then summed. That is fine only if SecAgg is real.
- Under GeFL-F's plain FedAvg (server sees individual models), the row-decay signature reveals which classes each client lacks, with or without Laplace counts.
- LA uses the client's own prior locally ✓ ("The prior never leaves the client", paper_draft.md:231).

**Contradictions inside the manuscript.**
- paper_draft.md:962-963: "Our DP experiment adds Laplace noise to class counts only." This contradicts §5.6 (E26 Gaussian on four releases) in the same file.
- paper_draft.md:252: "CSL uses no real data and no client message beyond GeFL-F's" vs paper_single_idea.md:206 "uses no client message beyond prediction sums" vs build_lean.js:438, which lists a per-round "prediction sums on a shared fake pool" release.
  - With K = 10 (one client per architecture), GeFL-F uploads no heads in Stage III (build_lean.js:210). CSL therefore *is* a new per-round client message, computed on client heads and not privatised.

---

## 5. Utility–privacy numbers (task D)

**Verified against `experiments/results/E26_private_anchoring/summary.md` (3 seeds each).**
- MNIST: ε = 8 gives 90.36 ± 0.30 and ε = 2 gives 85.06 ± 0.54 ✓.
- FMNIST: ε = 8 gives 72.77 ± 1.67 and ε = 2 gives 65.59 ± 0.58 ✓.
- Exact reference 94.82 / 79.67 (E25, same seeds, no BBC; DP `final_bal` is also without BBC, so the comparison is consistent).
- Costs 4.46 / 6.90 (ε = 8) and 9.76 / 14.08 (ε = 2) ✓.
- Margins over GeFL-F at ε = 2: 9.81 / 7.12 ✓.
- T08_dp.md matches.

**Verified against K02.**
- ε = 1 costs 0.18 on FMNIST (76.51 → 76.33) and 1.28 on MNIST (90.15 → 88.87) ✓.
- At ε = 0.1 the method is 11.4 / 14.1 above GeFL-F ✓.

**Framing problems.**
- DP is evaluated only on MNIST/FMNIST with the PC branch forced. The SVHN/CIFAR configurations that headline the abstract use the rows branch and MC-S (R6/R7), which were **never** run with DP.
- paper_single_idea.md:33-34 couples "Every released statistic can be made formally DP" with "+7–10 points" as if the claim held across the four-dataset results.
- "Formal privacy is cheap: ε = 1 on every client's class counts costs at most 1.3 points" (paper_draft.md:51-52) prices a 10-number release whose noise is negligible against hundreds of samples. It says nothing about the cost of privatising what actually leaks (R1, R8–R10).

---

## 6. Membership and reconstruction risk (task E)

**Setup** (`r6_work/mia.py`, `mia_rff.py`, `inv.py`).
- Real MNIST, the authors' exact partition code, seeds 0–2.
- Members: the 24 (class 9) or 40 (class 8) federation samples.
- Non-members: 2,000 unused same-class training images.
- Reference: the 200-image held-out pool the authors themselves call public.
- Features: raw pixels (784-d) and a random 1-conv-block ReLU FE giving 3 × 16 × 16 = 768-d. **These are surrogates; the trained FE was not available.**

**Exact class mean (R3), tracing score ⟨x − m_ref, m_rel − m_ref⟩.**

| class (n) | exact | ε = 8 | ε = 2 |
|---|---|---|---|
| 9 (24), fe768, seeds 0/1/2 | 0.66 / 0.81 / 0.81 | 0.56 / 0.55 / 0.60 | 0.50 / 0.48 / 0.50 |
| 8 (40), fe768 | 0.70 / 0.76 / 0.68 | 0.64 / 0.58 / 0.57 | 0.57 / 0.50 / 0.51 |

**Exact kernel mean embedding (R5, D = 6144, the authors' bandwidth rule), score ⟨φ(x) − μ_ref, μ_rel − μ_ref⟩.**
- Class 9 (n = 24): AUC 0.78 / 0.93 / 0.89 with TPR at 1% FPR of 0.29 / 0.25 / 0.21.
- Class 8 (n = 40): AUC 0.80 / 0.93 / 0.83 with TPR at 1% FPR of 0.25 / 0.45 / 0.30.
- This is a one-shot attack using only the single release the paper calls "federation-level".
- **Demonstrated on surrogates. Suspected for the real FE**, since a 1-conv-block FE is close to injective.
- The DP version removes most of the signal at ε = 2 (AUC about 0.5), consistent with the accounting.

**Reconstruction.**
- I attempted naive inversion: 24 points optimised to match Φ, S and Q exactly.
- It **failed**: no member was closer to a recovered point than to another real image.
- This is unresolved, not evidence of safety. Stronger priors or the exact second moments (R6/R11: 768² and up to 1024² entries over 24-sample classes, d ≫ n) were not tried.

---

## 7. Claims to narrow or withdraw (exact quotes)

| # | Location | Quote | Problem | Action |
|---|---|---|---|---|
| Q1 | paper_draft.md:51-52 | "Formal privacy is cheap: ε = 1 on every client's class counts costs at most 1.3 points." | Prices only the count release. Pipeline not private. | Withdraw from the abstract, or rewrite as "DP on the added count release (ε = 1) costs ≤ 1.3 points; the generator, FE and heads are not DP." |
| Q2 | paper_single_idea.md:33-34 | "Every released statistic can be made formally (ε, δ)-differentially private. At ε = 2 the method stays 7–10 points above GeFL-F, which offers no formal guarantee." | M, Z_c, Σee^T and the CSL sums were never privatised (MC-S disabled under DP). Generator, FE and heads are releases too. "which offers no formal guarantee" implies ESA does. | "The four anchoring statistics (counts, sums, squared norms, RFF sums) were made (ε, 1e-5)-DP at the example level on MNIST/FMNIST; the end-to-end pipeline, like GeFL-F, has no formal guarantee." |
| Q3 | paper_single_idea.md:376 | "All ESA releases made (ε, 10⁻⁵)-DP" | False per §2 table | Rename: "Anchoring statistics made DP" |
| Q4 | build_lean.js:488 | "Ours-A with every statistic made differentially private (ε, δ = 10⁻⁵)" | Same | Same |
| Q5 | paper_single_idea.md:17-19 | "they reveal nothing beyond federation-level totals" | Totals over a 24-sample class are member-identifiable (§6). A single holder, or collusion with the other holders, gives exact disclosure. | Withdraw "nothing" |
| Q6 | paper_single_idea.md:239; paper_draft.md:625 | "prototypes, corrections, herding targets and both gates are post-processing" / "Prototypes, MC and KH targets, and the BBC gate are post-processing." | BBC gate reads exact counts (`E26:2433`). F gate computed from exact stats (`E26:2279`). MC/KH also depend on the non-DP generator. | Fix the code to use noisy counts, or drop the claim. Say "post-processing of the DP statistics *and the non-DP generator*". |
| Q7 | paper_draft.md:452-453 | "Gaussian-mechanism DP on the single release covers the whole synthetic channel by post-processing." | True for KME-Gen only if the decoder's prototype buffer also comes from the DP release, which it does. But it is conditional on the non-DP FE. KME-Gen has no DP result. | Add "conditional on the FE, which is not DP"; report a DP KME-Gen run (code exists: `kme_dp_eps`, `E26:1281`, using the classical (ε, δ) σ, not zCDP) |
| Q8 | paper_draft.md:962-963 | "Our DP experiment adds Laplace noise to class counts only." | Contradicts §5.6/E26 | Update |
| Q9 | paper_draft.md:252 | "CSL uses no real data and no client message beyond GeFL-F's." | With one client per architecture, CSL needs per-round prediction sums from client heads (build_lean.js:438; single_idea.md:206) | List CSL as a new, non-DP, per-round release |
| Q10 | paper_draft.md:222-223, 959-961; build_lean.js:330, 426 | "Under secure aggregation it learns neither any client's row nor any client's histogram." | SecAgg is assumed, not implemented. No collusion or dropout threshold. | Say "would be compatible with SecAgg (not implemented)". Give the threat model. |
| Q11 | paper_draft.md:737 | "Our method keeps GeFL-F's FE and privacy model." | Adds R2–R6, R9 and R11 (head-body broadcast) | "keeps GeFL-F's messages and adds …" |
| Q12 | build_lean.js:211 | "images, features and labels never leave a client; only model weights move" | Feature sums, class counts, RFF sums and prediction sums move | Correct |
| Q13 | paper_draft.md:630; single_idea.md:385 | "whose 24 samples cannot be both private and accurate" | Under DP these classes fall back to the non-DP generator. Their privacy is not formal either way. | Rephrase |

---

## 8. Evidence ledger

| ID | Claim + location | Evidence | Reasoning | Consequence | Sev (1 fatal–4 minor) | Conf. | Status | Minimum action |
|---|---|---|---|---|---|---|---|---|
| P1 | "All ESA releases/every statistic DP" (single_idea:33, :376; build_lean:488) | MC-S disabled in DP mode `E26:2382`; generator FedAvg not DP (draft:632); CSL R9 not DP | Several releases have no noise | Headline privacy claim false as worded | 2 | high | demonstrated | Narrow per Q2–Q4 |
| P2 | Tail classes protected by DP (implied, draft:629-631) | Gates: MC n ≥ 77 / 277, KH n ≥ 216 / 784 (`acct.py`), matching the reported 7/5 and 5/3. Tail falls back to the non-DP PC-VAE trained on exact tail features. | DP noise is "spent" but the tail's information still flows via the non-DP generator | The privacy–utility table measures the cost of noising statistics, not of privacy | 2 | high | demonstrated | State explicitly. Add a DP-SGD/DP-FedAvg generator arm, or label the rows "DP statistics, non-DP generator" |
| P3 | SecAgg protects sums (draft:222, 959; single_idea:135) | No SecAgg code; plaintext sums `E26:1162-1168` | Assumed protocol; no threat model, collusion threshold or dropout handling | Claims are conditional on an unspecified protocol | 3 | high | demonstrated | Threat-model section; "compatible with", not "protected by" |
| P4 | DP + SecAgg compose (implied) | Noise added centrally `E26:1169-1186` | A trusted curator contradicts SecAgg. Distributed noise needs discrete mechanisms and loses ρ·K/(K − t) under t colluders or drops (ε = 8 becomes 9.8 at t = 3) | Deployment ε unspecified | 2 | high | demonstrated (analysis) | Specify the noise placement; discrete Gaussian/Skellam; account for dropout |
| P5 | zCDP accounting correct | `zcdp_rho`, sig1, sensitivities checked; gate counts reproduced | Correct for add/remove, example-level | Positive | – | high | verified | Keep; state the neighbouring relation and unit |
| P6 | Unit = example (unstated) | Sensitivities 1, C, C², √2 | Client-level ε for a 9-sample holder ≈ 148; replace-one gives ε ≈ 18.1 for nominal 8 | Guarantee weaker than readers in FL assume | 3 | high | demonstrated | State "example-level, add/remove, central model" |
| P7 | Gates are post-processing (single_idea:239; draft:625) | BBC gate exact counts `E26:2433`; F exact `E26:2279` | Not a function of the DP release | Small leak (1 bit each) but a false statement | 4 | high | demonstrated | Use noisy counts / noisy F |
| P8 | Clip bound from "public" pool (draft:623) | Pool = 200/class from the training split `E26:219-229`; through the non-DP FE | In-distribution public data assumed; DP is conditional on the FE | Needs explicit assumption | 3 | high | demonstrated | State; test the sensitivity of the clip to an out-of-distribution public set |
| P9 | Exact stats "reveal nothing beyond totals" (single_idea:19) | Tracing AUC 0.78–0.93, TPR@1%FPR 0.21–0.45 from the RFF sum on n = 24/40 classes (surrogate features) | d ≫ n tracing (Dwork et al. 2015 style) | The non-DP headline configuration leaks membership of rare-class examples | 2 | medium (surrogate FE) | demonstrated on surrogate; suspected on the real FE | Withdraw the claim; run the attack with the trained FE; default to DP or a minimum class-size threshold |
| P10 | Single-holder leak (draft:995) | Partitions: rarest class has 6–8 holders (K = 10), 21–24 (K = 100); several holders have exactly 1 sample | No single-holder case is evaluated; a 1-sample holder is exposed under collusion with the other holders | Risk acknowledged but not quantified | 3 | high | demonstrated (partition facts) / suspected (attack) | Report the holders-per-class distribution; require a minimum number of holders |
| P11 | CSL adds no client message (draft:252) | build_lean:438 lists it as a per-round release; one client per architecture | New per-round non-DP release from individual heads (summed over all K by SecAgg, if run) | Disclosure understated | 3 | high | demonstrated (document inconsistency) | List it; account for it |
| P12 | ER disclosure "same kind as MC-S" (readout_report:112-113) | "server broadcasts the bodies e_1 .. e_G" (:170) with one client per architecture | Each client's head body goes to all others; Σee^T up to 1024² is not DP | New model-level disclosure | 3 | high | demonstrated (design) | Do not present ER as disclosure-neutral |
| P13 | K02 "ε = 1 costs ≤ 1.3" (draft:51-52, 610) | 76.51→76.33; 90.15→88.87 ✓ | Correct numbers, but only 10 numbers per client are noised; rows not DP | Misleading in the abstract | 3 | high | demonstrated | Rephrase per Q1 |
| P14 | DP numbers (draft:620-621) | summary.md matches exactly | – | – | – | high | verified | – |
| P15 | DP generalises to SVHN/CIFAR (implied, single_idea:30-34) | No DP run with the rows/MC-S branch | Covariance releases have huge relative noise for small classes | Unsupported | 3 | high | demonstrated (absence) | Restrict the claim to MNIST/FMNIST PC branch, or run it |
| P16 | Internal contradiction (draft:962-963 vs §5.6) | Text | – | Confusing | 4 | high | demonstrated | Edit |
| P17 | Noise RNG (`E26:2294`) | Deterministic seed, float Gaussian | Fine for simulation; not deployable (Mironov 2012) | – | 4 | high | demonstrated | Note it |
| P18 | Reconstruction from Φ/S/Q | Naive inversion failed (`inv.py`) | Not evidence of safety | – | – | low | unresolved | Authors should run a stronger inversion, including the second moments |

---

## 9. Questions to the authors

1. What is the adversary? Is the server honest-but-curious or malicious, what collusion threshold applies, and how are dropouts handled? Is SecAgg run over *all* K clients for every release, including CSL and generator rounds?
2. Who adds the DP noise in a SecAgg deployment? If clients add it, which discrete mechanism, and what is ε under t dropped or colluding clients?
3. Is the neighbouring relation add/remove of one example? Will you report client-level ε, or justify example-level DP for your target (cross-silo) setting?
4. Why is MC-S disabled under DP? Can the M, Z_c and Σee^T releases be privatised at useful accuracy for tail classes (n = 24) in d = 768 or more?
5. Will you report a run where the generator (and ideally the FE) is trained with DP-FedAvg/DP-SGD, so that "Ours-A, ε" is an end-to-end number?
6. The clip pool comes from the training distribution. What happens with a genuinely public, shifted pool, or with a fixed a-priori clip?
7. Please run membership inference with the *trained* FE against the exact-statistics method (the headline configuration) and against generator weights, for the rarest class.
8. Fix the BBC gate to use the noisy counts and confirm the E26 numbers are unchanged.
9. With one client per architecture, does the server ever see individual heads (CSL as described in draft:242-243)? Does ER broadcast private head bodies?
10. Will you add a minimum-holders / minimum-count rule (k-anonymity-style suppression) to the exact-statistics mode?

---

*Scripts (read-only use of the repository data):*
- `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r6_work/part.py`: holders per class
- `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r6_work/acct.py`: ρ, σ, gates
- `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r6_work/mia.py`: mean tracing
- `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r6_work/mia_rff.py`: KME tracing
- `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/review/r6_work/inv.py`: inversion attempt
