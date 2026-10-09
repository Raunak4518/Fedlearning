# Rare Classes Vanish from Federated Feature Generators: Holder-Weighted Aggregation, Prior-Adjusted Headers and Consensus Labels for GeFL-F

*Working draft. Every number below comes from `experiments/results/*/runs.jsonl` and is reproduced by
`experiments/make_results_page.py`. Entries marked **[PENDING: …]** are filled when the named run finishes.*

---

## Abstract

GeFL-F lets clients with different model architectures learn from each other. They share a feature extractor and a
conditional feature generator, and train only their own classifier heads on real and generated features. We show that
GeFL-F fails under long-tailed client data, for a reason that can be stated exactly. Under flat federated averaging
with Adam's coupled weight decay, the generator's per-class conditioning rows of rare classes shrink every round
whenever fewer than half the clients hold the class. The generator then emits class-agnostic features under rare
labels. We measured this collapse on MNIST, FashionMNIST and SVHN: the tail-to-head row-norm ratio falls to 0.03–0.72.

We propose three changes, each derived from a specific failure.
1. **Holder-weighted aggregation (HWA)** averages each conditioning row only over the clients that hold its class.
2. **Prior-adjusted headers (LA)** train each head with logit adjustment by its client's own label prior.
3. **Consensus soft labels (CSL)** label synthetic features with a mix of the one-hot label and the mean prediction of
   all architectures' heads.

None of the three shares raw data, real features, or per-client histograms in the clear. HWA and LA together add
under 2% compute. Under a 100:1 long tail with Dirichlet(0.5) clients, balanced accuracy rises by +13.3 points on MNIST
(p = 0.004) and +17.4 on FashionMNIST (p = 0.015) with HWA + LA. The full method adds CSL and reaches +15.6 and
+17.6 (F10). Tail recall rises by 30 or more points. On SVHN the gain is +12.0. In GeFL-F's own IID setting, CSL improves on every
seed: +0.69 on MNIST, +0.49 on FashionMNIST and +0.48 on SVHN. We validated the GeFL-F baseline against the
authors' released code and the published numbers; it is within 0.6 points of both.

---

## 1. Introduction

*Motivation.* Federated deployments rarely have one model architecture or balanced labels. Hospitals run different
networks on different hardware. Rare conditions appear at only a few sites. Generator-based model-heterogeneous FL
(GeFL, and its feature-level variant GeFL-F) handles the first problem. It trains a shared generative model whose
samples carry knowledge between architectures that cannot be averaged. The original evaluation uses IID clients only.

*Gap.* The generator is itself trained by federated averaging. When a class is held by few clients, everything the
generator knows about that class passes through that averaging. We show that GeFL-F's aggregation and optimiser
interact to erase rare classes from the generator. This is a structural failure, not a tuning issue: it worsens as the
number of clients grows and as the tail lengthens.

*Contributions.*
1. **Diagnosis (§3).** An exact recursion for a conditioning row under flat averaging with coupled weight decay. It
   predicts collapse whenever $m_c < K/2$, where $m_c$ is the number of clients holding class $c$. We confirm it with
   row norms and with generator fidelity measured by an independent referee classifier.
2. **Method (§4).** HWA, LA and CSL. Each fixes one measured failure. Each reduces to GeFL-F when its failure is absent.
   All three are compatible with secure aggregation.
3. **Validation of the baseline (§5.2).** We ran the authors' code next to ours, so the improvement is measured from a
   faithful baseline.
4. **Evidence (§5).** Paired-seed experiments in the long-tail and IID settings, more clients, SVHN and CIFAR-10,
   ablations, a DP-noised variant, and a record of the ideas that did not work.

---

## 2. Background: GeFL-F

There are $K$ clients. Client $k$ holds data $D_k$ and runs a head $h_{a(k)}$ of architecture $a(k)$ from a set of $G$
architectures. GeFL-F proceeds in three stages.

1. **Feature extractor (FE) warm-up.** A small shared FE $\phi$ (one conv block; output $3\times16\times16$) is trained
   with the heads by FedAvg, then frozen.
2. **Feature generator.** A conditional VAE $G_F(h\mid y)$ (CVAE-F) is trained on $\phi(x)$ by FedAvg. It uses Adam
   with learning rate $10^{-3}$ and *coupled* weight decay $\lambda = 10^{-3}$ (paper, Table XV). The label enters
   through a learned linear projection whose $c$-th column (the *conditioning row* $w_c$) is the only class-specific
   parameter.
3. **Heads.** Each round, each client trains its head for $T_s = 1$ epoch on generated features with uniform random
   labels. It then trains for $T_r = 5$ epochs on its real features. Heads of the same architecture are averaged.

The paper reports `best_mean_acc`: the best test accuracy of each architecture over rounds, averaged over the
architectures.

---

## 3. Why rare classes vanish

### 3.1 The conditioning-row recursion

Take a class $c$ held by $m_c$ of the $K$ clients. A client that does not hold $c$ gets no data gradient on $w_c$.
Its only gradient on $w_c$ is the decay term $\lambda w_c$. Adam normalises each coordinate's step by the root of its
second-moment estimate, so a pure-decay gradient yields a step of about $\eta\,\mathrm{sign}(w_c)$, the full learning
rate, whatever the value of $\lambda$. Over a round of $S$ local steps, a non-holder therefore returns
$w_c - S\eta\,\mathrm{sign}(w_c)$ coordinate-wise (until it reaches zero). A holder returns at most
$w_c + S\eta$ in the useful direction. Flat averaging gives, per coordinate, an increase in $|w_c|$ of at most

$$\Delta|w_c| \;\le\; \tfrac{m_c}{K} S\eta \;-\; \bigl(1-\tfrac{m_c}{K}\bigr) S\eta \;=\; \bigl(\tfrac{2m_c}{K}-1\bigr) S\eta ,$$

which is **negative for every class held by fewer than half the clients**. The row is driven toward zero round after
round. The decoder then receives almost no class signal for $c$, and samples "of class $c$" become class-agnostic
features.

Without weight decay (or with SGD), the decay term vanishes. The data step is still scaled by $m_c/K$ (Theorem 1 of
`docs/improvement_derivations.html`), so rare classes learn $m_c/K$ times slower. That is dilution, not collapse.

**Predictions.** (i) GeFL-F's tail-to-head row-norm ratio is $\ll 1$ under a long tail. (ii) It worsens with $K$ for
fixed data, because $m_c/K$ falls. (iii) Any change that removes non-holders from row $c$'s average restores it.

### 3.2 Measurement

The setting is IF = 100 with Dirichlet(0.5), $K = 10$, full schedule and 3 seeds (F01). *Fidelity* is the accuracy of
a referee classifier, trained on real held-out features, on generated features of each class.

| | MNIST row ratio | MNIST tail fidelity | FMNIST row ratio | FMNIST tail fidelity |
|---|---|---|---|---|
| GeFL-F | 0.33 (0.09, 0.17, 0.72) | 8.8 | 0.22 (0.55, 0.09, 0.03) | 19.6 |
| + HWA + LA | 0.80 (0.71, 0.80, 0.88) | 27.2 | 1.01 (1.15, 1.03, 0.85) | 66.2 |

On SVHN (Kaggle, 3 seeds) the GeFL-F ratio is 0.49, 0.32 and 0.18, against 0.99, 0.96 and 1.05 with HWA. A quick-pass
ablation (E02) shows that removing weight decay, lazy decay, and HWA each restore the ratio. Prediction (iii) holds.

---

## 4. Method

### 4.1 HWA: holder-weighted aggregation of conditioning rows

Let $n_{k,c}$ be client $k$'s count of class $c$, and $E(n) = (1-\beta^n)/(1-\beta)$ the effective number of samples
(Cui et al., 2019), with $\beta = 0.999$. The server aggregates

$$w_c \leftarrow \frac{\sum_k E(n_{k,c})\, w_c^{(k)}}{\sum_k E(n_{k,c})}, \qquad\text{all other parameters by FedAvg.}$$

* **Effect on §3.1.** Non-holders have $E(0) = 0$, so the decay-only copies are excluded. The recursion's negative
  term disappears for every class with at least one holder.
* **Why effective number.** A holder's row estimate has variance that falls with its sample count but saturates
  (correlated samples, a finite number of local steps). Inverse-variance weighting with a saturating variance model is
  $E(n)$-weighting. Raw counts $n$ and uniform-over-holders are the two limits, $\beta \to 1$ and $\beta \to 0$. Both
  are ablated in K02.
* **Do-no-harm.** When every client has the same class counts, HWA *equals* FedAvg. Random IID splits give nearly
  equal counts. On SVHN IID, HWA + LA scores 0.18 points below GeFL-F (75.64 vs 75.82), a small but consistent cost.
* **Privacy.** The server needs $\sum_k E(n_{k,c}) w_c^{(k)}$ and $\sum_k E(n_{k,c})$, which are sums. Under secure
  aggregation it learns neither any client's row nor any client's histogram. We also test Laplace-noised counts
  ($\varepsilon \in \{10, 1, 0.1\}$; K02).

### 4.2 LA: heads trained with the client's own prior

Each head trains on real features with logits $f(h) + \tau \log \pi_k$, where $\pi_k$ is client $k$'s smoothed label
distribution and $\tau = 1$. Testing uses plain $f(h)$. Cross-entropy on prior-adjusted logits is Fisher-consistent
for the balanced error (Menon et al., 2021). Using the *client's own* prior means each local update estimates the
balanced risk for that client's label shift, so averaged heads are balanced as well. The prior never leaves the client.
In IID data $\pi_k$ is uniform and LA is a no-op.

Why it is needed in addition to HWA: the real phase ($T_r = 5$ epochs) follows the synthetic phase ($T_s = 1$) every
round, and gradient descent contracts toward the real-risk optimum. The final head therefore reflects the client's
skew unless the real phase itself is debiased.

### 4.3 CSL: consensus soft labels for synthetic features

A generated feature $\tilde h \sim G_F(\cdot\mid y)$ is not always a clean example of $y$. Its true posterior
$p(\cdot\mid\tilde h)$ differs from $\mathrm{onehot}(y)$ by the generator's error. The minimiser of expected
cross-entropy is that posterior, so one-hot training on generated features injects label noise. The server already
holds every architecture's aggregated head. For a shared synthetic pool drawn from a public seed it computes

$$t(\tilde h) = (1-\beta)\,\mathrm{onehot}(y) + \beta\,\frac1G\sum_{g=1}^{G}\mathrm{softmax}\,h_g(\tilde h), \qquad \beta = 0.5,$$

and clients use $t$ as the synthetic-phase target. The $G$ heads have different architectures and are trained on
different clients' data. Averaging their errors lowers variance by up to $1/G$ for uncorrelated errors, and the one-hot
term bounds the bias that a shared ensemble error would introduce. This is the bias–variance reason why $\beta = 0.5$
beats $\beta = 1$ (§5.4). CSL uses no real data and no client message beyond GeFL-F's. The server broadcasts the pool
seed and the $P \times C$ soft labels (about 24 KB per round here).

### 4.4 The method in one line

GeFL-F with HWA in stage (ii), and LA and CSL in stage (iii). F10 (§5.5) confirmed that the three parts add. In IID
data the method reduces to GeFL-F + CSL.

---

## 5. Experiments

### 5.1 Protocol

We follow the paper's appendix exactly.
* **Models.** FE per Table XIV (CIFAR-10: 10 channels). Heads CNN-1…10 (Tables XXI–XXIII). CVAE-F per Table XX.
* **Optimisation.** SGD with learning rate 0.1 and momentum 0. Rounds $T_{FE}/T_{KA}/T_{TN}$ = 20/100/50 (MNIST,
  FMNIST); SVHN and CIFAR-10 per the paper.
* **Data.** Fraction 0.1 (0.5 for CIFAR-10). Client-to-architecture assignment is contiguous.
* **Long tail.** $n_c \propto \mathrm{IF}^{c/(C-1)}$ with the total data budget fixed, and Dirichlet(0.5) client
  shares. Test sets are balanced.
* **Metrics.** Balanced accuracy, tail recall (the three rarest classes), worst-class recall, and `best_mean_acc`.
* **Statistics.** 3 seeds; each seed fixes the split and the initialisation for every method. p-values come from
  two-sided paired t-tests over seeds.

### 5.2 Is our GeFL-F the paper's GeFL-F?

| Setting | Paper | Authors' code | Ours |
|---|---|---|---|
| MNIST IID, K=10 | 95.47 | 96.07 | 96.02 ± 0.04 |
| MNIST IID, K=50 | 95.04 | – | 95.16 |
| MNIST IID, K=100 | 94.63 | – | 94.61 |
| FMNIST IID, K=10 | 83.14 | – | 82.67 ± 0.31 |
| MNIST LT, seed 0 (final balanced) | – | 75.84 | 73.92 |

Under the long tail our baseline is 1.9 points *below* the authors' code, so the gains below are, if anything,
understated.

### 5.3 Long-tailed clients (IF = 100, Dir 0.5, K = 10; F01)

| Method | MNIST bal. | MNIST tail | FMNIST bal. | FMNIST tail |
|---|---|---|---|---|
| FedAvg (per architecture) | 62.9 ± 1.8 | 40.8 | 52.8 ± 3.0 | 33.7 |
| LG-FedAvg | 63.2 ± 2.2 | 42.0 | 52.5 ± 2.9 | 33.3 |
| LG-FedAvg + LA | 67.8 ± 1.2 | 50.6 | 57.6 ± 2.4 | 41.0 |
| GeFL-F | 74.6 ± 1.8 | 51.9 | 58.6 ± 3.4 | 35.4 |
| GeFL-F + LA | 78.9 ± 2.3 | 60.1 | 64.8 ± 3.3 | 45.5 |
| **GeFL-F + HWA + LA** | **87.9 ± 1.6** | **81.1** | **76.0 ± 0.7** | **68.8** |

Against GeFL-F, the gain is +13.3 (p = 0.004) on MNIST and +17.4 (p = 0.015) on FMNIST. Tail recall rises by +29 and
+33. On the paper's own metric (`best_mean_acc`) the gain is +13.5 and +17.3. LA alone accounts for +4.3 and +6.2;
HWA adds +9.0 and +11.2 on top of it.

**More clients** (F04, MNIST long tail, full schedule, 3 seeds, balanced accuracy):

| | K = 10 (F01/F10) | K = 50 | K = 100 |
|---|---|---|---|
| GeFL-F | 74.2 | 78.1 | 76.0 |
| + LA | 78.9 | 81.4 | 79.5 |
| + HWA + LA | 87.8 | 86.5 | 83.1 |
| + HWA + LA + CSL (ours) | **89.9** | 86.8 | 82.7 |
| FSG + LA | 87.0 | 87.7 | **86.1** |
| FSG + LA + CSL | – | 87.2 | 85.5 |
| MIX + HWA + LA + CSL | – | **87.9** | 85.2 |

* Every variant beats GeFL-F at every K. The full method gains +8.7 at K = 50 and +6.7 at K = 100.
* The best *generator* changes with K. The repaired CVAE-F wins at K = 10. The Gaussian sufficient-statistics
  generator (FSG) wins at K = 100: 86.1 against 82.7 for HWA + LA + CSL, p = 0.105 with 3 seeds. At K = 50 the
  variants are within noise of each other (ours vs FSG, p = 0.47).
* This is the expected structure. FSG's federated estimate equals the centralised one at any K, because it
  aggregates exact sums. The CVAE's rows, even under HWA, are fitted by clients that each hold fewer samples as K
  grows: about 60 images per client at K = 100. The server can see the per-client sample size from aggregated
  counts, so a principled rule (CVAE-F when clients are data-rich, FSG when they are data-poor) is available.
  Validating that rule on FMNIST (K03) and CIFAR-10 (K01) is pending.

**SVHN** (Kaggle, 3 seeds). Balanced accuracy is 62.45 ± 1.76 for ours, 50.49 ± 2.24 for GeFL-F and 56.44 for + LA,
a gain of +12.0. Tail recall is 47.8 against 24.8. On `best_mean_acc` the result is 66.86 vs 57.36 (p = 0.045). The
Gaussian generator falls to 44.2.
**CIFAR-10** (Kaggle, partial, long tail). GeFL-F collapses hardest here, with a row ratio of 0.06. HWA + LA scores
40.4 against 33.7 for GeFL-F and 37.3 for + LA. The Gaussian generator does better, at 43.4. Its exact class means
beat a CVAE trained on about 25 tail samples on CIFAR features, the opposite of SVHN. Choosing the generator per
dataset is left open. **[PENDING: CIFAR-10 IID and the full method.]**

### 5.4 GeFL-F's own IID setting (F09)

| | Paper GeFL-F | Our GeFL-F | + CSL | Gain per seed | p |
|---|---|---|---|---|---|
| MNIST, K=10 | 95.47 | 96.02 | **96.71** | +0.57, +0.69, +0.80 | 0.010 |
| FMNIST, K=10 | 83.14 | 82.67 | **83.16** | +0.60, +0.47, +0.40 | 0.016 |
| MNIST, K=50 (seed 0) | 95.04 | 95.16 | 95.41 | +0.26 | – |
| MNIST, K=100 (seed 0) | 94.63 | 94.61 | 94.72 | +0.11 | – |
| SVHN, K=10 | 76.26 | 75.82 | **76.30** | +0.78, +0.37, +0.29 | 0.085 |

$\beta = 1$ is worse than $\beta = 0.5$ at every K (96.10 / 95.13 / 94.49), which is the bias term of §4.3.
Interleaving synthetic features into real batches (CSLM) is also worse.

**On data augmentation.** The paper's augmentation results (Table IV: MixUp, CutMix, AugMix, AutoAugment) are for
image-space GeFL with DCGAN on CIFAR-10, never for GeFL-F. Augmentation is orthogonal to all three of our changes, so
we leave it out of every method.

### 5.5 One method for both regimes (F10)

Matched seeds and splits, 3 seeds.

| | MNIST LT bal. | FMNIST LT bal. | MNIST IID `best_mean_acc` | FMNIST IID `best_mean_acc` |
|---|---|---|---|---|
| GeFL-F | 74.24 | 58.74 | 96.01 | 82.80 |
| + CSL | 76.35 | 58.59 | **96.82** | **83.25** |
| + HWA + LA | 87.78 | 75.99 | 96.01 | 82.74 |
| **+ HWA + LA + CSL (ours)** | **89.87** | **76.37** | 96.77 | 83.21 |
| ours − GeFL-F (p) | +15.6 (0.008) | +17.6 (0.019) | +0.76 (0.004) | +0.41 (0.008) |

* **Long tail.** The full method beats HWA + LA on every seed: +3.29, +2.21 and +0.77 on MNIST, and +0.28, +0.65 and
  +0.20 on FMNIST. With 3 seeds this is not yet significant (p ≈ 0.1). Tail recall is 82.7 against 51.4 on MNIST and
  69.8 against 35.6 on FMNIST.
* **CSL alone under the long tail.** It gives +2.1 on MNIST (p = 0.003) and nothing on FMNIST. It works best once HWA
  has made the tail features faithful, as §4.3 predicts: relabelling class-agnostic features cannot help.
* **IID.** The full method is within 0.05 of + CSL. HWA + LA neither helps nor hurts there.

### 5.6 Regimes, privacy and ablations (K02)

**[PENDING:]**
* label skew without a tail (IF = 1, Dir 0.5) and a mild tail (IF = 10);
* Laplace-noised counts at $\varepsilon \in \{10, 1, 0.1\}$, and feature-space memorisation (MND);
* ablations: each component removed, HWA weighting ($E(n)$ vs $n$ vs uniform), $\tau \in \{1, 1.5, 2\}$, and
  CSL $\beta \in \{0.25, 0.5, 0.75\}$.

### 5.7 What did not work

These ideas were tested on a reduced schedule first and dropped on the evidence. We report them because they sharpen
what the working method actually fixes.

* **Prior-completing mixed batches.** These fill a client's missing classes with synthetic features, which works only
  when tail synthetic features are faithful. Before HWA they are not; FMNIST went from 65.2 to 59.6 with LA.
* **Lazy conditioning decay.** It removes the collapse from the client side, with no shared counts. With HWA present
  it adds nothing (78.6 vs 79.1). It remains the fallback when even aggregated counts are disallowed.
* **A federated Gaussian generator (exact aggregation of sufficient statistics).** It was competitive on MNIST
  (87.0 vs 87.9). It is worse in IID data and fails on SVHN (≈ 45 vs 62), where tail fidelity is 0.14: a single
  Gaussian per class cannot model real feature distributions.
* **A hybrid CVAE/Gaussian generator.** The quick pass favoured it; the full schedule reversed that (87.1 vs 87.9).
  This shows why a quick pass alone is not enough to decide.
* **Server-side classifier re-calibration, and gap-filling synthetic labels.** Both are weaker than LA.
* **Server-side ensemble distillation on generated features (SED).**
  * *Motivation.* The ensemble of the G heads beats the mean head by 3.2 points on FMNIST IID, 4.5 on MNIST LT and
    7.9 on SVHN IID. SED distils that ensemble into every head at the server on fresh generated features, using
    CSL's target, with no client cost.
  * *Result (FMNIST quick pass).* It *lowers* IID accuracy (81.69 → 80.96) and raises the long tail only a little
    (70.65 → 71.30).
  * *Why.* The bound $R_{\text{real}} \le R_{\text{syn}} + \ell_{\max}\,\mathrm{TV}(p_{\text{syn}}, p_{\text{real}})$
    explains it. In IID data the heads are already good, so the generator's infidelity dominates what distillation
    adds. The ensemble's knowledge can reach the heads only through features as faithful as the generator's.
  * *Implication.* The remaining IID gap to the paper's best FMNIST number (DDPM-F) is a *generator* gap. §5.9 tests
    exactly that.

### 5.9 Generator-agnostic: our method on DDPM-F, the paper's best feature generator

The best FMNIST result of any GeFL variant in the paper is 84.28, from GeFL-F with the feature diffusion model
DDPM-F. Our components do not depend on the generator:
* HWA averages the class columns of DDPM-F's two context-embedding layers over holders.
* LA and CSL act on the heads.

We ported the authors' DDPM-F unchanged: the ddpm16 ContextUnet, $n_T = 200$, a linear $eta$ schedule from
$10^{-4}$ to 0.02, context dropout 0.1, and Adam at $10^{-4}$ decayed linearly to 0 over $T_{KA}$ rounds, then frozen.
It runs GeFL-F, + CSL and Ours on it. Every round, all clients' synthetic phases draw from one shared pool of fresh
uniform-label samples. For each head this is exactly the reference's per-client draw in distribution, at 1/K of the
sampling cost, and it is the same for every method compared.

On seed 0 of FMNIST IID, GeFL-F on DDPM-F scores 83.16, not the published 84.28. + CSL scores 83.04 and Ours 83.10.
In our hands DDPM-F is only +0.28 over CVAE-F on the same seed, and CSL does not add to it: diffusion samples are
more diverse but less class-pure, with referee fidelity 0.55 against 0.86. **[PENDING: seeds 1–2, and K04 on
Kaggle.]**

### 5.9b Consensus labels make a larger synthetic budget useful (E16, FMNIST IID, 3 seeds)

**Prediction.** Train a head on real data plus a share $\lambda$ of synthetic data. Its error is roughly
$\sigma^2/(n_{\text{real}}+\lambda n_{\text{syn}}) + \lambda^2 b(t)^2$, where $b(t)$ is the bias of the synthetic label
against the true posterior of a generated feature. The optimal $\lambda$ grows as $b$ shrinks. Hard labels have a large
$b$, so the paper finds $T_s = 5$ no better than $T_s = 1$ (its Fig. 10). CSL lowers $b$, so with CSL more synthetic
epochs should help.

| `best_mean_acc` | $T_s=1$ | $T_s=3$ | $T_s=5$ | $T_s=10$ |
|---|---|---|---|---|
| GeFL-F (hard labels) | 82.96 | 83.20 | 83.18 | – |
| + CSL | 83.31 | 83.90 | 84.14 | **84.37** (84.34 / 84.16 / 84.62) |
| Ours (HWA + LA + CSL) | 83.21 | – | 84.12 | **84.34** (84.55 / 83.88 / 84.59; p = 0.002 vs GeFL-F) |

**The same budgets under the long tail** (FMNIST, IF = 100, Dir 0.5, 2 seeds, balanced accuracy):

| | GeFL-F | Ours, $T_s=1$ | Ours, $T_s=5$ | Ours, $T_s=10$ |
|---|---|---|---|---|
| Balanced accuracy | 61.33 | 76.98 | 78.76 | **79.33** |
| Tail recall | 38.86 | 70.17 | 72.37 | 72.44 |

Under the long tail the larger budget also helps: +2.35 over $T_s = 1$, and +18.0 over GeFL-F. There, HWA has made
the tail features faithful and CSL has lowered their label bias. This is the synergy of §4.3, seen through the
budget.

* The interaction appears exactly as predicted.
* At $T_s = 10$, CSL with the cheap CVAE-F generator reaches the paper's best FMNIST number of *any* variant, 84.28
  (GeFL-F with feature diffusion). It is +1.23 over the paper's GeFL-F with the same generator.
* MNIST is near its ceiling (oracle ≈ 97.2), and a larger budget does not help there.
* With one fixed setting ($T_s = 5$), the full method gives 84.12 on FMNIST and 96.6 on MNIST. With $T_s = 10$ the
  full method gives 84.34 on FMNIST, a tie with the paper's best (84.28).

### 5.10 Against every GeFL variant in the paper (IID, `best_mean_acc`)

The paper's Figure 4 has ten methods: FedProx, FedALA, and GeFL and GeFL-F each with DCGAN, CVAE, DDPM w = 0 and
DDPM w = 2. In each setting, the comparison is with the best of them.

| Setting | GeFL-F CVAE-F | Best in paper (which) | Ours | Ours − best |
|---|---|---|---|---|
| MNIST, K=10 | 95.47 | 96.44 (GeFL, image DDPM) | **96.77** | +0.33 |
| MNIST, K=50 | 95.04 | 95.04 (GeFL-F CVAE-F) | **95.41** (CSL, 1 seed) | +0.37 |
| MNIST, K=100 | 94.63 | 94.63 (GeFL-F CVAE-F) | 94.72 (CSL, 1 seed) | +0.09 |
| FMNIST, K=10 | 83.14 | 84.28 (GeFL-F DDPM-F) | **84.34** (Ours, $T_s=10$); 84.12 ($T_s=5$) | +0.06 / −0.16 (tie) |
| SVHN, K=10 | 76.26 | 76.26 (GeFL-F CVAE-F) | 76.30 (CSL) | +0.04 (tie) |
| CIFAR-10, K=10 | 55.86 | 59.36 (GeFL, image DDPM); 62.67 with MixUp | **[PENDING: K01, K04]** | – |

The two strongest CIFAR-10 numbers come from **image-space** GeFL, with an image diffusion model or with DCGAN +
MixUp. Neither has a shared feature extractor, and the paper notes that GeFL-F on CIFAR-10 is limited by its small FE.
Our method keeps GeFL-F's FE and privacy model. The fair claims are therefore:
1. over GeFL-F with the same FE and generator;
2. over the best of *all* variants wherever the FE is not the bottleneck: MNIST, SVHN, and FMNIST pending DDPM-F.

### 5.8 Cost

Summed over all stages, our method takes 373 s per run against GeFL-F's 366 s (+1.9%), measured on MNIST under the
long tail, averaged over 3 seeds. Communication adds $C$ numbers per client per round for HWA, and one broadcast of
soft labels for CSL.

---

## 6. Related work (to expand)

* **Model-heterogeneous FL.** GeFL / GeFL-F (Kang et al.); FedGen (Zhu et al., 2021), a data-free generator for
  knowledge distillation; FedDF (Lin et al., 2020), ensemble distillation; LG-FedAvg (Liang et al., 2020).
* **FL under class imbalance.** CReFF (Shang et al., 2022), which re-trains the classifier on federated features;
  FedLC (Zhang et al., 2022), logit calibration.
* **Long-tail learning.** Class-balanced loss with effective numbers (Cui et al., 2019); logit adjustment
  (Menon et al., 2021).
* **Secure aggregation** (Bonawitz et al., 2017).

What sets this work apart: none of these analyses how *generator* aggregation treats rare classes. The collapse in
§3 is specific to class-conditional generators trained by FedAvg with decoupled class rows, and HWA is the minimal
fix to it.

---

## 7. Limitations

* Results so far are on small images and on the paper's backbone sizes. Larger FEs could change the generator's
  failure profile. SVHN and CIFAR-10 are under way.
* HWA uses aggregated class counts. LCD is the fully local alternative but is weaker.
* CSL's gain in IID data is small (+0.5 to +0.7 points), although it is consistent across seeds.
* MND in feature space is a proxy for privacy, not a guarantee. A formal DP accounting of the generator itself is
  future work.
