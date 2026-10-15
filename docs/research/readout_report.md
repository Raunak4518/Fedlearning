# Beyond ESA: where GeFL-F's remaining error lives, and the readout-bias mechanism

*Technical report, 2026-10-14. Status of every claim is marked: **[proved]**, **[measured]** (existing runs),
**[T03]** (new isolated test, `experiments/T03_exact_readout.py`, pre-registered in
`experiments/results/T03_exact_readout/PREREGISTRATION.md`), **[conjecture]**, **[known]** (prior work).*

## 1. Current research state (verified against the repository)

* **Implemented final method (Ours-A).** PC-VAE generator conditioned on exact class means; MC (or MC-B) moment
  correction; KH kernel herding; LA heads; CSL consensus labels; gated BBC; T_s = 10 on MNIST/FMNIST. On SVHN-like
  data (F < 0.05) the gate selects CVAE class rows + HWA + MC + KH. Ablation K09 (3 seeds): every component helps
  on 30/30 seed-level comparisons; per-component significance pending K14.
* **Proved.** Theorem 1 / Lemma 2 (row contraction under coupled Adam decay), Prop. 3 (synthetic budget), Prop. 4
  (MC is minimum-displacement), Prop. 5 (mean anchor separates only as far as means), Prop. 6 (Gelbrich map),
  Prop. 7 (MixUp to second order).
* **Rejected on evidence.** Prior-completing batches, lazy decay as a fix with HWA, FSG / hybrid generators, BCR and
  BBC-only recalibration *on synthetic features*, SED (server distillation on synthetic features), Bayesian CSL,
  held-class alignment, KME-Gen, VCT (K13: VCT 60.29 vs MixUp 59.98, neither reaches 62.67).
* **Pending.** K14 (ablation seeds 3-4, MC-B), K15 (collapse fixes vs HWA).

## 2. Where the remaining error is [measured]

Two diagnostics the pipeline already logs for every run:
* **oracle**: each head's last layer (`fc`) re-fit on the *pooled real* training embeddings with class-balanced
  sampling (body frozen);
* **ensemble**: the average of the G heads' softmax outputs.

Define, in balanced accuracy, the **readout gap** = oracle − head and the **representation gap** = ensemble − oracle.

| run (3 seeds) | head | oracle | ensemble | readout gap | repr. gap |
|---|---|---|---|---|---|
| GeFL-F, MNIST LT | 74.2 | 90.7 | 84.9 | **16.5** | −5.8 |
| GeFL-F, FMNIST LT | 58.7 | 79.2 | 71.7 | **20.5** | −7.5 |
| GeFL-F, SVHN LT | 50.5 | 64.8 | 60.0 | **14.3** | −4.8 |
| + HWA + LA, MNIST LT | 87.8 | 92.4 | 94.1 | 4.6 | 1.7 |
| + HWA + LA, FMNIST LT | 76.0 | 80.1 | 81.8 | 4.1 | 1.7 |
| Ours-A, MNIST LT | 94.8 | 94.9 | 96.1 | 0.1 | 1.2 |
| Ours-A, FMNIST LT (K = 10 / 50 / 100) | 79.7 / 79.5 / 79.0 | 81.6 / 81.2 / 81.0 | 83.1 / 82.1 / 81.1 | 1.9 / 1.7 / 2.0 | 1.5 / 0.9 / 0.1 |
| Ours-A (rows), SVHN LT | 65.6 | 68.4 | 72.2 | 2.8 | 3.8 |
| Ours-A, CIFAR-10 LT | 46.1 | 52.3 | 57.1 | **6.2** | 4.8 |
| Ours-A, CIFAR-10 IID | 56.8 | 57.9 | 67.5 | 1.1 | 9.6 |

Reading:
1. **GeFL-F's long-tail failure is mostly a readout failure.** Its bodies are good enough that a balanced
   last-layer refit on pooled real data adds 14-20 points; the readout gap is larger than the ensemble gap.
2. **ESA's components shrink the readout gap** (16.5 → 4.6 → 0.1 on MNIST; 20.5 → 4.1 → 1.9 on FMNIST) and, beyond
   that, improve the bodies: Ours-A's head beats GeFL-F's *oracle* by +4.1 (MNIST), +0.5 (FMNIST), +0.8 (SVHN,
   rows), and its oracle beats GeFL-F's oracle by +4.2 / +2.4 / +3.6.
3. **The representation gap is a single-client-head effect.** With K = 10 and G = 10 architectures every head is
   trained on one client's real data. As clients per architecture grow (FMNIST K = 50, 100) the representation gap
   falls to 0.1 while the **readout gap stays at 1.7-2.0**: the readout gap is the K-invariant residual of the
   final method under the long tail, and it is large on CIFAR-10 (6.2).

Consequence for the paper [known + measured]: classifier re-calibration from federated feature statistics (CCVR,
NeurIPS 2021; CReFF, IJCAI 2022; Fed3R, ICML 2024) attacks exactly the readout gap. Applied to GeFL-F its *upper
bound* (the oracle) is 90.7 / 79.2 / 64.8, which the final method beats by only +4.1 / +0.5 / +0.8. The paper does
not yet compare against this family; it must.

## 3. Competing hypotheses for the K-invariant readout gap

| | Hypothesis | Intervention | Predicts | Status |
|---|---|---|---|---|
| H-R | **Readout bias from synthetic dependence × synthetic mismatch.** A head learns tail classes mostly from synthetic features; in the head's embedding space their mean is off by Δ_c; prior corrections cannot remove a mean error. | Exact-statistics readout (ER) from real embedding sums, per architecture | ER ≈ oracle; gap ∝ (1 − α)·‖Δ‖; BBC < oracle | selected; T03 |
| H-P | Pure class-prior bias (the LA/BBC target) | better prior correction | BBC = oracle | already contradicted: BBC < oracle in every LA run (e.g. FMNIST 80.6 vs 81.6, CIFAR-10 48.7 vs 52.3) |
| H-V | Single-client head variance (heads see one client's data) | cross-architecture codistillation on local real data | ensemble gap shrinks; readout gap unchanged | explains the *representation* gap (falls with K); known technique (federated codistillation, FedGKD, FedDF) |

## 4. Derivation (H-R)

**Setting.** FE output $h \in \mathbb R^{D}$. Head $g$: body $e_g:\mathbb R^D\to\mathbb R^{d_g}$, readout
$(W_g, b_g) \in \mathbb R^{C\times d_g}\times\mathbb R^C$. Client $k$ (architecture $g = a(k)$) has $n_{k,c}$ real
samples of class $c$, $n_k = \sum_c n_{k,c}$. Each round: $T_s$ synthetic epochs (about $T_s n_k / C$ samples per
class, uniform labels) then $T_r = 5$ real epochs. Target: balanced risk
$R_{\mathrm{bal}}(f) = \frac1C\sum_c \mathbb E_{h\sim P_c}\,\ell(f(h), c)$.

**Readout model.** Fix the body. In embedding space let real class $c$ have mean $\mu_c$ and common covariance
$\Sigma$; let generated class-$c$ samples have mean $\mu_c + \Delta_c$. The readout that minimises the balanced risk
under this Gaussian model is uniform-prior LDA: $w_c = \Sigma^{-1}\mu_c$, $b_c = -\tfrac12\mu_c^\top\Sigma^{-1}\mu_c$.

**What the client's readout sees.** Within a round, class $c$'s evidence on client $k$ is a mixture of its local
real samples (mean $\hat\mu_{k,c} = \mu_c + \varepsilon_{k,c}$,
$\varepsilon_{k,c}\sim\mathcal N(0,\Sigma/n_{k,c})$) and synthetic samples, with real share
$$\alpha_{k,c} = \frac{T_r\,n_{k,c}}{T_r\,n_{k,c} + T_s\,n_k/C}.$$
The fitted class mean is $m_{k,c} = \alpha_{k,c}\hat\mu_{k,c} + (1-\alpha_{k,c})(\mu_c + \Delta_c)$, so its error is
$$\delta_{k,c} = m_{k,c} - \mu_c = \underbrace{(1-\alpha_{k,c})\,\Delta_c}_{\text{synthetic bias}} + \underbrace{\alpha_{k,c}\,\varepsilon_{k,c}}_{\text{local noise}} . \tag{1}$$

**Lemma R1 [proved].** Logit offsets (LA, BBC, any prior correction) map LDA$(m)$ to LDA$(m')$ only if
$\Sigma^{-1}(m_c - m'_c)$ is the same vector for every $c$.
*Proof.* LDA$(m)$'s logit difference between classes $c, c'$ at $e$ is
$(m_c - m_{c'})^\top\Sigma^{-1}e + \text{const}$. An offset changes only the constant, so the $e$-dependent part is
preserved iff $\Sigma^{-1}\big((m_c - m'_c) - (m_{c'} - m'_{c'})\big) = 0$ for all pairs. $\square$
Numerical check (C = 4, d = 6): a common shift of all means is fully undone by BBC's offsets (88.3 → 76.9 → 88.3);
a class-specific shift of the same size is only partly undone (88.3 → 70.9 → 75.6).

So the class-specific part of (1) survives every prior correction. It is zero when $\alpha_{k,c} = 1$ (plenty of
local data: IID) or $\Delta_c = 0$ (a faithful generator), which is why the readout gap is small in IID runs and on
MNIST and largest on CIFAR-10 LT (largest synthetic–real embedding mismatch, `gap_held` 15 %, and tail classes
learned almost only from synthetic features).

**Excess risk.** LDA is optimal at $\delta = 0$, so the balanced risk is stationary there and the excess is
second order: $R(m) - R(\mu) \approx \tfrac12\sum_{c,c'}\delta_c^\top H_{cc'}\delta_{c'}$. To first approximation
the per-class damage scales with $\|\delta_{k,c}\|^2_{\Sigma^{-1}}$, whose expectation is
$$\mathbb E\|\delta_{k,c}\|^2_{\Sigma^{-1}} = (1-\alpha_{k,c})^2\|\Delta_c\|^2_{\Sigma^{-1}} + \alpha_{k,c}^2\,\frac{d_g}{n_{k,c}} . \tag{2}$$

**Exact-statistics readout (ER).** Every client embeds its real features with *every* architecture's current body
and uploads, per architecture, the class sums $\sum e$, the counts, and $\sum e e^\top$; secure aggregation gives the
pooled $\mu_c$ and $\Sigma$, and the server sets the LDA readout (shrinkage $\gamma = 0.1$ fixed a priori). Then
$\delta = 0$ up to estimation error $\Sigma/n_c$ with the *federation's* $n_c$, not the client's. Special cases: with
one architecture and a frozen pretrained body this is Fed3R's setting (ridge instead of LDA) and CCVR's (Gaussian
virtual features); the cross-architecture statistics are what heterogeneous heads add.

**Cost.** One extra round. Each client: G forward passes of its data through small heads. Upload per architecture
$C d_g + d_g^2$ numbers ($d_g$ = 80-1024 here; under 1.1 M numbers for the largest head). Disclosure: federation-level
class means and second moments of each head's embedding, the same kind of release as MC-S and FSG.

**Where it should fail [conjecture].** (i) Embedding distributions far from a shared-covariance Gaussian (LDA
misspecified): ER below the CE-refit oracle. (ii) Representation-limited regimes (CIFAR-10 IID: readout gap 1.1,
representation gap 9.6): ER cannot help. (iii) Under DP, tail-class embedding means carry noise of order
$\sqrt{d_g}/(n_c\varepsilon)$, which grows with $d_g$.

## 5. Relationship to ESA

ESA's thesis is "anchor every stage to exact statistics instead of FedAvg-estimated parameters". Until now it was
applied to the **generator** (prototypes, moments, kernel embeddings) and to a **prior offset** (BBC, fitted on
synthetic features). The readout analysis shows the remaining error of the final method under the long tail sits in
the **classifier's last layer**, on classes a client does not hold, and that offsets cannot reach it (Lemma R1, BBC <
oracle in every run). ER applies the same thesis to that last stage: the readout of every architecture is set from
exact federated statistics of *real* embeddings. It is not another generator component; it removes a failure the
generator components can only shrink indirectly.

## 6. Related-work comparison

| Dimension | CCVR (NeurIPS 2021) | CReFF (IJCAI 2022) | Fed3R (ICML 2024) | ER (this report) |
|---|---|---|---|---|
| Setting | one global model, non-IID | one global model, long tail | frozen pretrained body, non-IID | heterogeneous heads (G architectures), generator-based FL, long tail |
| Statistic | per-class feature mean + covariance (global model) | gradients of real features (federated features learned by gradient matching) | ridge sufficient statistics $\sum ee^\top$, $\sum e y^\top$ | per-architecture class sums, counts, $\sum ee^\top$ — every client embeds with **every** architecture |
| Classifier | CE on Gaussian virtual features | CE on learned federated features | closed-form ridge | closed-form uniform-prior LDA |
| Main claim | the classifier is the most biased layer | the biased classifier is the main cause in FL-LT | closed form is immune to heterogeneity | same diagnosis holds in generator-based heterogeneous FL; offsets (LA/BBC) cannot fix it; the bias sits on unheld classes |
| Equivalence | ER = CCVR with the Gaussian solved in closed form, if G = 1 | different estimator | ER ≈ Fed3R with LDA instead of ridge, if G = 1 and the body frozen | — |

**Novelty assessment.** The *principle* (re-fit the classifier from federated feature statistics) is known. What is
new as far as this search found: (i) the cross-architecture statistics that heterogeneous heads require; (ii) the
measured decomposition of GeFL-F's residual into a K-invariant readout gap and a representation gap that vanishes as
clients per architecture grow; (iii) the finding that classifier recalibration on plain GeFL-F (the CCVR/CReFF upper
bound) nearly matches the full ESA method on FMNIST and SVHN — a baseline the paper currently omits. Claim (ii)-(iii)
as analysis; claim ER only as "the known classifier-recalibration principle, extended to heterogeneous heads".

## 7. Theoretical status

* **Proved.** Lemma R1 (offsets reach only common shifts; numerically checked). Eq. (1)-(2) are exact under the
  stated Gaussian/LDA model.
* **Falsified (pre-registered P3).** That the per-class readout gap is explained by the synthetic class-mean
  mismatch $\Delta_c$ in embedding space: $\rho(\|\Delta\|)$ = 0.10-0.33, and $B = (1-\alpha)\|\Delta\|$ never beats
  $(1-\alpha)$ alone.
* **Supported (post hoc, not pre-registered).** The gap concentrates on classes the client does not hold and grows
  with the client's lack of real evidence for the class, also within a class (ρ 0.37-0.79).
* **Partly supported (pre-registered T03b).** H-R′, the unheld-class readout error is a *scale* (weight-norm) error:
  the norm deficit tracks the per-class gap (ρ 0.44-0.69, Q1 pass), but τ-normalisation recovers only 16-34 % of
  the gap (Q2 fail, needed ≥ 50 %). Scale is about a quarter of the error; the rest is in the weight *direction*.
* **Open, leading hypothesis (H-R″, untested): source–label confounding.** On client $k$ an unheld class is seen
  *only* as synthetic features, held classes mostly as real ones. A discriminative readout can therefore use
  "looks synthetic" as evidence for the unheld class; at test time real samples of that class look real and are
  pulled toward held classes. This predicts the per-class gap from the *separability* of real vs synthetic class-$c$
  embeddings (a domain classifier), not from their mean distance — consistent with ρ(‖Δ‖) being weak while
  (1 − α) is strong. It also says ER works because the exact readout never sees synthetic data.

## 8. Algorithm (ER)

```
after training (or every R rounds):
  server broadcasts the bodies e_1 .. e_G of all architectures            # heads are small (5k-110k params)
  every client k, for every architecture g:
      E = e_g(h) for its real features h
      upload  S_gc = sum_{i: y_i = c} E_i,  n_c,  M_g = sum_i E_i E_i^T     # secure aggregation
  server, per g:  mu_c = S_gc / n_c ;  Sigma = (M_g - sum_c n_c mu_c mu_c^T) / (N - C)
                  Sigma <- (1 - 0.1) Sigma + 0.1 tr(Sigma)/d I
                  W_g = Sigma^-1 [mu_1 .. mu_C] ;  b_gc = -1/2 mu_c^T Sigma^-1 mu_c
  heads keep their bodies; readout replaced by (W_g, b_g); prediction unchanged in form
```
Cost: one round; G forward passes per client; upload $\sum_g (C d_g + d_g^2)$ numbers. In `core.py` as
`exact_readout()` (diagnostic `er_bal`, logged by every run from now on).

## 9. Experimental protocol and results (T03)

FMNIST / MNIST, IF = 100, Dirichlet(0.5), K = 10, seed 0, schedule × 0.4 (CPU), methods GeFL-F, + HWA + LA,
Ours-A (T_s = 1). Balanced accuracy:

| run | head | oracle (pooled-real CE refit) | **ER-LDA** (exact sums) | BBC | ρ(B) | ρ(1−α) | ρ(‖Δ‖) |
|---|---|---|---|---|---|---|---|
| FMNIST GeFL-F | 61.2 | 78.7 | **79.1** | 65.1 | 0.64 | 0.77 | 0.33 |
| FMNIST + HWA + LA | 73.0 | 79.4 | **79.9** | 73.6 | 0.43 | 0.57 | 0.16 |
| FMNIST Ours-A | 75.7 | 80.7 | **80.3** | 77.9 | 0.41 | 0.57 | 0.10 |
| MNIST GeFL-F | 72.4 | 91.0 | **90.8** | 75.5 | 0.82 | 0.90 | 0.32 |
| MNIST + HWA + LA | 82.1 | 91.8 | **91.3** | 82.0 | 0.65 | 0.74 | 0.23 |
| MNIST Ours-A | 91.1 | 93.6 | **92.8** | 92.2 | 0.29 | 0.33 | 0.14 |

Pre-registered criteria: **P1 pass** (ER ≥ head in all runs, within 0.8 of the oracle), **P2 pass** (+17.9, +18.4 on
GeFL-F), **P4 pass** (BBC < oracle in all LA runs), **P3 fail** (see §7).

Recall lost by the head relative to the oracle, by the client's count of the class (post hoc):

| run | unheld (n = 0) | held, 1-20 | held, > 20 |
|---|---|---|---|
| FMNIST GeFL-F / HWA+LA / Ours-A | 74.4 / 28.2 / 17.4 | 19.8 / 7.7 / 7.2 | −2.2 / −1.4 / −1.4 |
| MNIST GeFL-F / HWA+LA / Ours-A | 76.7 / 44.3 / 8.2 | 22.1 / 10.2 / 3.1 | −1.8 / 0.1 / 0.1 |

Caveats: one seed, reduced schedule (larger gaps than the full schedule; there the oracle gap of Ours-A is 0.1
MNIST, 1.9 FMNIST, 6.2 CIFAR-10 LT). The ER gains above are therefore upper-end; the full-schedule estimate is the
oracle gap in §2.

### T03b (pre-registered follow-up, FMNIST)

| run | head | oracle | ER-LDA | τ-norm | τ-norm share of gap | ρ(norm deficit) |
|---|---|---|---|---|---|---|
| GeFL-F | 61.2 | 78.7 | **79.2** | 65.7 | 26 % | 0.69 |
| + HWA + LA | 73.1 | 79.4 | **80.0** | 74.2 | 16 % | 0.45 |
| Ours-A | 75.7 | 80.7 | **80.3** | 77.4 | 34 % | 0.44 |

Q1 pass, Q2 fail, Q3 pass (the T03 numbers reproduce to within 0.2: same seed, same code).

## 10. Interpretation: what the evidence establishes

**Established (1 seed, reduced schedule, CPU; consistent with the 3-seed full-schedule oracle numbers in §2):**
1. The residual error of every GeFL-F variant under the long tail splits into a readout gap and a representation
   gap. The readout gap does not shrink with more clients; the representation gap does.
2. The readout gap is not prior bias: offsets (LA, BBC) leave it in place, as Lemma R1 predicts.
3. It sits on classes a client does not hold (and, less, on classes it holds with few samples).
4. A closed-form readout from exact federated statistics of real embeddings (ER) closes it: within 0.8 points of
   the pooled-real oracle in 9/9 runs, +18 on plain GeFL-F, +4.7 / +1.8 on Ours-A at this schedule.

**Not established:** the mechanism of the readout error. The synthetic mean-mismatch account (H-R) is falsified;
weight-norm mis-scaling (H-R′) explains about a quarter; source–label confounding (H-R″) is a hypothesis.

**The consequence that matters most for the paper.** Classifier recalibration from federated statistics is
published (CCVR, CReFF, Fed3R). Applied to plain GeFL-F it reaches 90.7 / 79.2 / 64.8 (MNIST / FMNIST / SVHN LT,
full-schedule oracle, 3 seeds), within +4.1 / +0.5 / +0.8 of the final method's head. With ER on both sides the
final method keeps an advantage of +4.2 / +2.4 / +3.6 (oracle vs oracle). The paper must report "GeFL-F +
recalibration" as a baseline and state the method's advantage against it, not only against GeFL-F.

## 11. Decision

* **ER: provisionally accept** as the classifier-side instance of exact-statistics anchoring, presented honestly
  as the known recalibration principle extended to heterogeneous heads (cross-architecture statistics). Logged
  automatically as `er_bal` in every run from now on (K14, K15 included) at no extra GPU cost.
* **Readout-bias mechanism: revise.** H-R rejected; H-R′ partial; H-R″ to be tested before any mechanistic claim.
* **Paper: add the "GeFL-F + recalibration" baseline** and restate the advantage of the generator-side
  contributions against it.

## 12. Best next experiment

**CIFAR-10 LT and SVHN LT, 3 seeds: GeFL-F and the final method (Ours-A; rows + MC + KH on SVHN), with `er_bal`
logged.** It is where the readout gap is largest (6.2 on CIFAR-10) and where the threat to the paper is greatest
(the recalibrated-GeFL-F bound is unknown on CIFAR-10 and within 0.8 on SVHN). About 6 GPU-hours (≈ 3 h on Kaggle
T4 × 2). It re-runs existing configurations only to log the new metric, so it is a deliberate exception to the
ledger's no-duplicates rule and needs the user's go-ahead.

**Cheapest mechanism test (CPU, ≈ 1 h):** H-R″ — add a per-class real-vs-synthetic domain classifier in each head's
embedding to the T03 harness; pre-register that its accuracy predicts the per-class gap better than (1 − α), and
that mixing synthetic samples of *held* classes into the real phase (breaking the confound) shrinks the unheld-class
gap.
