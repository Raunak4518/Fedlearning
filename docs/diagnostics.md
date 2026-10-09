# What each knob tells us: hyperparameter responses, their causes, and where to go next

For every knob we varied, this lists: how the results moved, the mechanism (with the math where it decides the
answer), and what follows. Numbers are balanced accuracy (long tail, IF = 100) or `best_mean_acc` (IID), %, mean
over seeds unless marked n=1. Runs: F01, F04, F09, F10, E02, E12, E15, E16, K01, K03 (`experiments/results`).

---

## 1. Generator weight decay and aggregation → the collapse (understood, fixed)

| FMNIST LT, quick (E02) | GeFL-F | +LCD | +NOWD | +HWA | +NOWD+HWA |
|---|---|---|---|---|---|
| tail fidelity | 20.9 | 24.6 | 32.7 | 41.1 | 42.3 |

**Response.** Removing the decay (NOWD) or the non-holders (HWA) each restores the tail rows. HWA does more,
because it also removes the *dilution* by $m_c/K$ that remains without decay.

**Mechanism.** Under flat FedAvg with Adam's coupled decay, a class row changes by at most
$(2m_c/K - 1)S\eta$ per round (§3.1), which is negative when $m_c < K/2$. The measured GeFL-F tail/head row ratio
is 0.06–0.33 (0.07 at K = 50, 0.06 at K = 100). HWA brings it to about 1.0 everywhere.

**Implication.** Closed. HWA is necessary under any label skew, and harmless in IID.

## 2. Number of clients K → which generator wins flips (understood; selection rule open)

| MNIST LT | K = 10 | K = 50 | K = 100 |
|---|---|---|---|
| GeFL-F | 74.2 | 78.1 | 76.0 |
| +HWA+LA (CVAE-F) | **87.8** | 86.5 | 83.1 |
| FSG+LA (Gaussian) | 87.0 | **87.7** | **86.1** |

Tail fidelity: CVAE-F + HWA gives 27 / 20 / 21; FSG gives 78 / 79 / 77.

**Response.** As K grows, the CVAE degrades while the Gaussian generator does not.

**Mechanism.**
* *FSG.* Its estimate is built from client sums, so the federated estimate *equals* the centralised one at any K
  (§6 of the method notes).
* *CVAE.* Each class row is fitted by holders that each see $n/K$ samples (about 60 images at K = 100) for a few
  local epochs. Its estimation error grows as per-client data shrinks, even when HWA removes the aggregation bias.
* *Where FSG fails.* SVHN (44 vs 62) is multi-modal within a class, and one Gaussian per class cannot fit it.
* *Bias–variance.* Neither generator wins everywhere. A low-capacity, exactly aggregated estimator wins when
  per-client data is small. A flexible one wins when within-class structure is rich.

**Implication.** We need a *label-free* way to tell, per class, which generator is more faithful. See §7, H-G.

## 3. Synthetic budget $T_s$ → interacts with label quality (predicted, confirmed; MNIST explained)

| | $T_s$=1 | 3 | 5 | 10 |
|---|---|---|---|---|
| FMNIST IID, GeFL-F (hard labels) | 82.96 | 83.20 | 83.18 | – |
| FMNIST IID, +CSL | 83.31 | 83.90 | 84.14 | **84.37** |
| FMNIST LT, Ours | 76.94 | – | 78.76 | **79.23** |
| MNIST IID, +CSL | **96.67** | 96.65 | 96.54 | 96.61 |

**Mechanism (Proposition 3).** The optimal synthetic share is $w^\star = \sigma^2/(2 n_r b^2)$.
* *FMNIST.* The generator is faithful (IID fidelity 86%), and CSL lowers $b$ further, so $w^\star$ is large and the
  budget pays.
* *MNIST.* The CVAE-F is a poor generator: fidelity 58% IID and 27% LT, so $b$ is large. Meanwhile real data already
  makes $\sigma^2/n_r$ tiny (head 96.7 against oracle 97.2), so $w^\star$ is small.
* *Hard labels* keep $b$ at the generator's label error, which is the paper's Fig. 10.

**Implication.** The budget should follow label quality, which is what Bayesian CSL (§7, H-B) estimates.

## 4. CSL weight β → interior optimum (as predicted); its fixed value is a weakness

| MNIST IID, seed 0 | GeFL-F (β = 0) | β = 0.5 | β = 1 |
|---|---|---|---|
| K = 10 | 95.97 | **96.54** | 96.10 |
| K = 50 | 95.16 | **95.41** | 95.13 |
| K = 100 | 94.61 | **94.72** | 94.49 |

**Mechanism.** The target $t = (1-\beta)e_y + \beta\bar p$ has two bias sources: the generator's label error
(weight $1-\beta$) and the ensemble's error (weight $\beta$). The optimum is interior, and it should depend on
their *ratio*, which differs by dataset and generator. A fixed β = 0.5 cannot be right everywhere. It is worst when
the generator is very poor (MNIST LT) or very good.

**Implication.** Replace the fixed mixture with the Bayes posterior (§7, H-B). E19 maps the β × $T_s$ surface to
test the prediction that the best β does not move with $T_s$.

## 5. Synthetic diversity per architecture → the K > 10 CSL result was an implementation artefact (fixed)

At K = 50 / 100, CSL went negative: −0.2 / −0.4 IID, and −1.0 / −2.0 LT on FMNIST. The cause is that every client of
an architecture trained on the *same* synthetic slice before averaging, which cut synthetic diversity by 5–10×. GeFL-F
draws fresh samples per client, so the comparison was confounded. Fixed: each client now gets a disjoint slice, and
K = 10 is untouched. Reruns: F04 locally; NB4b/NB5b on Kaggle.

## 6. Server-side distillation steps (SED) → hurts IID, helps a collapsed generator (understood)

| FMNIST, quick | without SED | +SED(20) |
|---|---|---|
| IID, CSL | 81.60 | 80.82 |
| IID, Ours | 81.69 | 80.96 |
| LT, GeFL-F | 59.05 | 62.45 |
| LT, Ours | 70.68 | 71.58 |

**Mechanism.** $R_{\text{real}} \le R_{\text{syn}} + \ell_{\max}\mathrm{TV}(p_{\text{syn}}, p_{\text{real}})$.
Distillation lowers $R_{\text{syn}}$ but pulls the heads toward the synthetic distribution. When the heads are
already good (IID), the TV term dominates. When they are poor (collapsed GeFL-F), the distillation term dominates.

**Implication.** No knowledge transfer can beat the generator's fidelity. That makes generator choice (§2) the
limiting factor in the hard regimes.

## 7. Next hypotheses, each derived from a failure above

**H-B: Bayesian consensus labels** (fixes §4; implemented, running as E18).
* The posterior of a generated feature's class, given the ensemble and the conditioning label under symmetric
  generator noise $\rho$, is $t_c \propto \bar p_c(\tilde h)\,[\rho 1\{c=y\} + (1-\rho)/C]$.
* $\rho$ is estimated label-free. For a calibrated ensemble of confidence $\kappa = \mathbb E\max_c \bar p_c$, the
  agreement is $a = \mathbb E\,\bar p_y = \rho\kappa + (1-\rho)/C$, since a uniformly drawn class has $\mathbb E\,\bar p_y = 1/C$. So $\hat\rho = (a - 1/C)/(\kappa - 1/C)$.
* There is no β. A perfect generator gives hard labels, and a useless one gives the ensemble.
* *Prediction.* BCSL ≥ CSL, and the budget pays where CSL's did not (MNIST LT).

**H-G: label-free per-class generator selection** (fixes §2; to implement after E18 and E19).
* The stage-(i) heads are trained on *real* local data only, before any synthetic data exists. Their ensemble is
  therefore an unbiased referee of a generator's class-conditional samples, available at the server.
* Compute the per-class agreement $a_c(G)$ for CVAE-F and FSG, and sample class $c$ from the generator with the
  higher $a_c$, or mix them in proportion to $a_c$.
* *Prediction.* It matches the better generator in every regime: CVAE on SVHN, FSG on MNIST at large K and on
  CIFAR-10 LT. It costs one forward pass, adds no new message, and needs no real data at the server.
* *Test first.* Check that $a_c$ ranks the generators as the held-out referee does (rank correlation over classes),
  then measure accuracy.

**H-PC: take class identity out of the parameters (PC-VAE; running as E20).**
* *Problem.* The CVAE-F's only class-specific parameter is one learned row per class, fitted from that class's
  few samples. It is the object that FedAvg dilutes and decays (§1), and its estimation error does not shrink with
  more clients (§2).
* *Model.* Replace it with the exact federated class mean $\mu_c$, from secure-aggregated sums (the same disclosure
  as FSG). The decoder gets a class-*shared* projection of $\mu_y$, and models only the residual:
  $\tilde h = \mathrm{ReLU}(\mu_y + \mathrm{dec}(z, \mu_y))$.
* *Why it should work.*
  * No class-specific parameter is left to dilute or collapse.
  * Tail classes borrow within-class variation from all classes (amortisation), instead of learning a lookup row
    from 24 samples.
  * A linear decoder recovers FSG, and a nonlinear one can model the non-Gaussian shape that FSG lacks.
* *Prediction.* Accuracy ≥ max(CVAE + HWA, FSG) in every regime, with no regime-dependent choice of generator.
* *Add-on ZP.* An ex-post latent prior fitted by exact sums of encoder means closes the VAE prior hole.

**H-MC: moment-calibrated sampling (running as E20).**
* *Problem.* The new diversity diagnostic shows that both VAE generators under-disperse. The ratio of generated to
  real within-class spread is 0.1–0.3: decoder means are blurry and too narrow, so heads see classes as 3–10× too
  tight.
* *Method.* The server knows each class's exact mean $\mu_c$ and spread $V_c = \mathbb E\|h-\mu_c\|^2$ from
  secure-aggregated sums ($\sum h$, $\sum\|h\|^2$, $n$ per class). Map each sample by
  $x' = \mathrm{ReLU}(\mu_c + s_c(x - m_c))$ with $s_c = \sqrt{V_c/\tilde V_c}$. Among affine corrections, this is
  the $W_2$-optimal one that gives the generator's class distribution the exact first moment and total spread.
* *Properties.* It works with any generator. It fixes only the first two moments, where the generator is
  measurably wrong, and keeps the learned shape.
* *Smoke test (3 % of rounds).* On CVAE + HWA, tail fidelity rose from 0.02 to 0.65, spread from 0.20 to 0.73, and
  balanced accuracy from 59.0 to 64.4.

**H-S: larger budgets with more clients** (from Proposition 3; running as E17). $w^\star \propto 1/n_r$, so the
optimal synthetic share is larger at K = 50 (about 120 real images per client) than at K = 10. $T_s = 10$ with CSL
should gain more there.

---

## 8. What is *not* a knob problem

* **FMNIST IID is close to its feature-extractor ceiling.** The oracle head is 83.6–84.3, and we are at 84.34.
  Beyond this, gains need a better shared FE, which the GeFL-F protocol fixes.
* **CIFAR-10 is limited by the FE in the same way** (oracle ≈ 50 under the long tail). The paper's 59–62 come from
  image-space GeFL, which has no shared FE.
