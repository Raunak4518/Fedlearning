# Rare Classes Vanish from Federated Feature Generators — and How Exact Statistics Bring Them Back

*Working draft. Every number below comes from `experiments/results/*/runs.jsonl` and is reproduced by
`experiments/make_results_page.py`. Entries marked **[PENDING: …]** are filled when the named run finishes.*

---

## Abstract

GeFL-F lets clients with different model architectures learn from each other. They share a small feature extractor
and a conditional feature generator, and each trains its own classifier head on real and generated features. We
show that GeFL-F fails under long-tailed client data, for a reason that can be stated exactly. Under federated
averaging with Adam's coupled weight decay, the generator's class-conditioning rows shrink every round for every class
held by fewer than half the clients. Rare classes vanish from the generator, and every head is then taught from
class-agnostic features.

The diagnosis suggests a principle: **anchor every stage of generator-based heterogeneous FL to exactly aggregated
statistics** (sums that secure aggregation delivers) instead of to federated-averaged parameters. We instantiate it
at every point where the data decides the result:
* **class identity:** a generator conditioned on the exact federated class mean, with no class-specific parameters;
* **moments:** the minimum-displacement correction of generated samples to the exact class mean and spread;
* **distribution:** kernel herding of samples toward the exact class kernel mean embedding;
* **classifier bias:** server-side calibration on the anchored generator.

On the heads we use logit adjustment by each client's own prior and cross-architecture consensus soft labels. With
these labels, a larger synthetic budget pays off once the generator is faithful (Proposition 3, confirmed).

Under a 100:1 long tail with Dirichlet(0.5) clients:
* balanced accuracy rises from 75.3 to 95.2 on MNIST and from 58.5 to 80.6 on FashionMNIST (3 seeds,
  p ≤ 0.011);
* tail recall rises from 52 to 91 and from 35 to 74;
* on MNIST the method reaches the accuracy of its own heads re-fit on pooled real data;
* with 100 clients it does not degrade (94.3, against GeFL-F's 76.0), because its generator has no class-specific
  parameters to fit from shrinking per-client data;
* on CIFAR-10 it rises from 33.7 to 46.1, or about 49 with the calibration, ahead of every other method.

The anchor has a precise limit. A generator whose class identity comes from the class mean can separate two classes
only as far as their means are apart, up to its Lipschitz constant (Proposition 5). SVHN's class means are nearly
identical, and there the anchored generator loses to the paper's generator repaired with holder-weighted aggregation
(55.9 vs 62.5).

A minimal fix for the paper's own generators, holder-weighted aggregation of the class rows, already gives +12 to +20
points on MNIST, FashionMNIST, SVHN and CIFAR-10. In GeFL-F's own IID setting the final method reaches 97.62 on
MNIST and 60.93 on CIFAR-10: +1.2 and +1.6 over the best of all ten methods in the paper (96.44 and 59.36). On
CIFAR-10 our own GeFL-F run scores 59.10, 3.2 above the paper's, so the margin over GeFL-F there is +1.8. On FashionMNIST and
SVHN our CVAE-based variant matches the best of all ten (84.34 vs 84.28, and 76.30 vs 76.26). We validated the GeFL-F baseline against the authors' released code. Formal privacy is cheap:
ε = 1 on every client's class counts costs at most 1.3 points.

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
2. **A minimal fix (§4.1–4.3).** HWA, LA and CSL. Each fixes one measured failure and each reduces to GeFL-F when
   its failure is absent. All three are compatible with secure aggregation.
3. **A principle and its instantiation (§4.5).** Anchoring to exact statistics:
   * PC-VAE: the generator has no class-specific parameters;
   * MC: the minimum-displacement moment correction (Proposition 4);
   * KH: kernel herding on the MMD term of the head's risk bound;
   * BBC: server-side bias calibration, gated by exact global counts;
   * KME-Gen, the extreme case: a generator trained only from exact kernel embeddings, with no federated training.
4. **Validation of the baseline (§5.2).** We ran the authors' code next to ours, so the improvement is measured from a
   faithful baseline.
5. **Evidence (§5).** Paired-seed experiments in the long-tail and IID settings, more clients, SVHN and CIFAR-10,
   ablations, a DP-noised variant, and a record of the ideas that did not work and why (§5.7).

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
beats $\beta = 1$ (§5.4). The bias–variance view of distillation from teachers that
approximate the Bayes posterior is due to Menon et al. (ICML 2021); CSL applies it to generated features labelled by a
cross-architecture ensemble. CSL uses no real data and no client message beyond GeFL-F's. The server broadcasts the pool
seed and the $P \times C$ soft labels (about 24 KB per round here).

### 4.4 Why consensus labels make more synthetic data useful (Proposition 3)

**Stylised model.** A head estimates a parameter $\theta^\star$ from $n_r$ real samples and $n_s$ synthetic samples,
weighting the synthetic part by $\lambda \in [0, 1]$. Real samples are unbiased with per-sample variance $\sigma^2$.
Synthetic samples carry the same variance, but their labels are biased by $b(t)$, the distance between the label
target $t$ and the true posterior of the generated feature. The weighted estimator's mean-squared error is

$$\mathcal{E}(\lambda) \;=\; \frac{\sigma^2}{n_r + \lambda n_s} \;+\; \Big(\frac{\lambda n_s}{n_r+\lambda n_s}\Big)^{2} b^2 .$$

**Proposition 3.** For $b > 0$, $\mathcal{E}$ has a unique minimiser $\lambda^\star(b)$ in $[0,1]$ (or the boundary
$\lambda^\star = 1$). It is non-increasing in $b$, and $\lambda^\star \to 1$ as $b \to 0$.

*Proof.* Write $w = \lambda n_s/(n_r+\lambda n_s) \in [0, w_{\max}]$, with $w_{\max} = n_s/(n_r+n_s)$. This is an
increasing bijection of $\lambda$. Since $n_r + \lambda n_s = n_r/(1-w)$, the error becomes

$$\mathcal{E}(w) = \frac{\sigma^2 (1-w)}{n_r} + w^2 b^2 .$$

This is a convex parabola in $w$, minimised at $w^\star = \sigma^2/(2 n_r b^2)$ and clipped to $[0, w_{\max}]$.
$w^\star$ is decreasing in $b$, and $\lambda$ is increasing in $w$. So $\lambda^\star$ is non-increasing in $b$. As
$b \to 0$, $w^\star \to \infty$, so the clip gives $w^\star = w_{\max}$, that is $\lambda^\star = 1$. $\square$

**Reading.** With hard labels, $b$ is the generator's label error, which no amount of synthetic data averages away.
The optimal synthetic share is then small, and adding synthetic epochs ($T_s\!\uparrow$, which acts like raising
$\lambda n_s$) past it *hurts*, or at best does nothing. This is the paper's Fig. 10. CSL's target
$t = (1-\beta)e_y + \beta\,\bar p$ moves $t$ toward the posterior, which lowers $b$ and pushes $\lambda^\star$ up, so the
same extra epochs now help. The proposition predicts an *interaction*, not a main effect: the gain from more synthetic
epochs should be larger with CSL than without it. That is exactly what E16 measures (FMNIST IID: +0.22 from $T_s = 1$
to 5 without CSL, +0.83 with it). It also explains why the budget helps more under the long tail once HWA has made
tail features faithful, since a lower generator error is a lower $b$ as well.

### 4.5 Anchoring every stage to exact statistics

HWA repairs a FedAvg-trained generator. The diagnosis of §3, together with the measurements of §5, points to a
cleaner design. Wherever the data decides the result, use a statistic that secure aggregation computes **exactly** —
sums over clients of per-class quantities — instead of a parameter that FedAvg *estimates*. Every statistic below
costs one upload per client and reveals only federation-level sums.

**PC-VAE: class identity from the exact class mean.**
* *Prototypes.* The server computes $m_c = \sum_k \sum_{i \in k, y_i = c} h_i \,/\, \sum_k n_{kc}$.
* *Decoder.* It models only the within-class residual, $\tilde h = \mathrm{ReLU}(m_y + \mathrm{dec}(z, m_y))$, and
  receives $m_y$ through a class-*shared* projection.
* *Encoder.* It sees $m_y$ as an extra channel.
* *Consequence.* There is no class-specific parameter, so the collapse recursion of §3 has nothing to act on.
  Tail classes borrow within-class variation from every class through the shared decoder. A linear decoder recovers
  a Gaussian class-conditional model.

**What a mean-anchored generator can express.** The anchor removes the collapse, but it also limits what the
generator can say about a class. Write the class-$c$ sampler as $X_c = \mathrm{ReLU}(m_c + g(Z, A m_c))$, with
$Z \sim \mathcal N(0, I)$ the same for every class, $A$ the class-shared projection, and $g$ $L$-Lipschitz in its
second argument.

*Proposition 5.* For any two classes, $W_1(G_c, G_{c'}) \le (1 + L\|A\|_2)\,\|m_c - m_{c'}\|$.

*Proof.* Couple both classes through the same $Z$. Since ReLU is 1-Lipschitz,
$\|X_c - X_{c'}\| \le \|m_c - m_{c'}\| + \|g(Z, A m_c) - g(Z, A m_{c'})\| \le (1 + L\|A\|_2)\|m_c - m_{c'}\|$.
Taking expectations bounds $W_1$. $\square$

* *What it means for a head.* If a head's logit differences are $K$-Lipschitz, the synthetic data can separate $c$
  from $c'$ by an expected margin of at most $K(1 + L\|A\|_2)\|m_c - m_{c'}\|$, however different the real classes
  are. MC adds at most the difference of the exact spreads, and KH only re-weights candidates drawn from $G_c$.
* *The quantity that decides.* In units of the within-class spread the anchors are, on average, $\sqrt{2F}$ apart,
  where $F = \overline{\|m_c - \bar m\|^2}\,/\,\overline{V_c}$. The server computes $F$ from the sums it already
  receives. In pixel space $F$ is 0.48 (FashionMNIST), 0.135 (MNIST), 0.067 (CIFAR-10) and **0.0015 (SVHN)**.
* *Why training does not escape it.* On SVHN the anchors differ by about 5% of the within-class spread. Reproducing
  class differences of the order of the spread needs a decoder about $1/\sqrt{2F} \approx 18$ times more sensitive
  to the anchor than to its noise. The ELBO offers a cheaper route: the encoder sees $x$, so it can carry the label
  in $z$ for a KL cost of order $\log C$ nats per sample, small against the reconstruction term. Samples drawn from
  the prior are then class-agnostic. This is what we measure (§5.11): after KH, tail fidelity is 24% on SVHN, against
  36% for the CVAE + HWA and 0.9 on MNIST.
* *The fix.* Free class rows $r_y$ inside the decoder, $g(Z, A m_y + r_y)$, remove the bound: the separation
  $\|r_c - r_{c'}\|$ is then a learned parameter, and HWA keeps it from collapsing. This is the hybrid generator PCR.

**MC: the minimum-displacement moment correction.** The server also knows each class's exact spread
$V_c = \mathbb E\|h - m_c\|^2$, from the sums of $\|h\|^2$. Let $\tilde m_c$ and $\tilde V_c$ be the mean and spread
of the generator's class-$c$ samples. MC maps $x \mapsto \mathrm{ReLU}\!\big(m_c + \sqrt{V_c/\tilde V_c}\,(x - \tilde m_c)\big)$.

*Proposition 4.* Among **all** maps $T$ with $\mathbb E\,T(X) = m_c$ and $\mathbb E\|T(X) - m_c\|^2 = V_c$ (with
$X$ the generator's class-$c$ sample), the map $T^\star(x) = m_c + \sqrt{V_c/\tilde V_c}\,(x - \tilde m_c)$ minimises
the expected displacement $\mathbb E\|T(X) - X\|^2$.

*Proof.* Let $Y = T(X)$. Since $\mathbb E[Y - m_c] = \mathbb E[X - \tilde m_c] = 0$,
$\mathbb E\|Y - X\|^2 = V_c + \tilde V_c + \|m_c - \tilde m_c\|^2 - 2\,\mathbb E\langle Y - m_c,\, X - \tilde m_c\rangle$.
By Cauchy–Schwarz, the last expectation is at most $\sqrt{V_c\tilde V_c}$, with equality iff
$Y - m_c = \sqrt{V_c/\tilde V_c}\,(X - \tilde m_c)$. $\square$

MC therefore fixes exactly the two moments the generator is measurably wrong about, and changes the samples as little
as possible. The ReLU projects onto the non-negative orthant where FE features live.

**KH: kernel herding toward the exact class embedding.**
* *Embedding.* With bounded random Fourier features $\phi$, the server receives the exact class mean embeddings
  $\mu_c = \sum_k \sum_{i \in k, y_i = c}\phi(h_i) / n_c$ in one upload.
* *Selection.* For each class, from $4n$ calibrated candidates, KH greedily selects $n$ by Frank–Wolfe on
  $\|\frac1n\sum_t \phi(x_t) - \mu_c\|$ (kernel herding).
* *Why this target.* For a head whose loss lies in the RKHS with norm at most $B$,
  $|\mathbb E_{G_c}\ell - \mathbb E_{P_c}\ell| \le B\,\mathrm{MMD}(G_c, P_c)$. KH lowers exactly the term of the
  head's risk bound that limits all knowledge transfer (§5.7, SED).

**BBC: balanced bias calibration on the anchored generator.**
* *Fit.* After training, the server fits $C$ logit offsets $b_g$ per head $g$ so that the head predicts every class
  equally often on class-balanced synthetic data. This is the prior-correction fixed point
  $b \leftarrow b - \log(C\,\bar q(b))$ of Saerens et al. (2002) in logit form.
* *Gate.* It is switched on only when the exact global class counts are imbalanced (max/min > 2; any threshold
  between 1 and 10 gives the same decisions on our regimes). It uses no test or held-out data.
* *Role.* It corrects the residual bias that LA's prior correction leaves when tail classes are also intrinsically
  hard (FashionMNIST's shirt and coat), without tuning LA's temperature.

**KME-Gen: the extreme case.** Train the generator *at the server only*, by MMD against the exact class embeddings
$\mu_c$, with no federated generator training at all.
* *Partition invariance.* $\mu_c$ equals the centralised embedding for every partition, so collapse, dilution and
  client drift cannot occur.
* *Formal privacy.* Since $\|\phi\| \le \sqrt2$, Gaussian-mechanism DP on the single release covers the whole
  synthetic channel by post-processing.
* *Cost.* One upload of about 0.3 MB replaces 100 rounds of generator FedAvg.
* *Result.* It beats GeFL-F by 13–16 points, but a learned decoder anchored by PC + MC + KH is better (§5.11).

### 4.6 The method in one line

**Ours-A.**
* *Generator:* PC-VAE trained by FedAvg on the frozen FE's features.
* *Sampling:* MC, then KH.
* *Heads:* LA + CSL with a 10-epoch synthetic budget.
* *Calibration:* gated BBC.

**Low class separation.** Where the released sums show uninformative class means ($F$ small, as on SVHN), the
hybrid PCR adds HWA-aggregated class rows to the anchored decoder (under test, §5.11).

**The minimal variant** for an existing GeFL-F deployment is HWA + LA + CSL. It keeps the paper's generator
(CVAE-F or DDPM-F) unchanged apart from aggregation.

## 5. Experiments

### 5.1 Protocol

We follow the paper's appendix exactly.
* **Models.** FE per Table XIV (CIFAR-10: 10 channels). Heads CNN-1…10 (Tables XXI–XXIII). CVAE-F per Table XX.
* **Optimisation.** SGD with learning rate 0.1 and momentum 0. Rounds $T_{FE}/T_{KA}/T_{TN}$ = 20/100/50 (MNIST,
  FMNIST); SVHN and CIFAR-10 per the paper.
* **Data.** Fraction 0.1 (0.5 for CIFAR-10). Client-to-architecture assignment is contiguous.
* **Long tail.** $n_c \propto \mathrm{IF}^{c/(C-1)}$ with the total data budget fixed, and Dirichlet(0.5) client
  shares. Test sets are balanced.
* **Metrics.** Balanced accuracy, tail recall (the four rarest classes, 6–9), worst-class recall, and `best_mean_acc`.
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

### 5.6 Regimes, privacy and ablations (K02, Kaggle; minimal variant HWA + LA + CSL)

**The gain grows with imbalance** (balanced accuracy, K = 10, 3 seeds):

| | MNIST: GeFL-F → ours | FashionMNIST: GeFL-F → ours |
|---|---|---|
| label skew only (IF = 1, Dir 0.5) | 89.8 → 95.6 (+5.8) | 74.1 → 79.2 (+5.1) |
| mild tail (IF = 10) | 86.9 → 94.3 (+7.4) | 72.6 → 79.4 (+6.8) |
| heavy tail (IF = 100) | 75.3 → 90.2 (+14.8) | 58.9 → 76.5 (+17.6) |

Dirichlet(0.5) already leaves some classes on fewer than half the clients, so the collapse term of §3 is active even
without a global tail.

**Ablations** (IF = 100):
* *Necessary components.* Removing HWA costs 7–8 points, and removing LA 5–8.
* *HWA's weighting rule is irrelevant.* Effective number, raw counts and uniform weighting over holders are within
  0.4 points of each other. This is predicted: the damage comes from *non-holders'* decay.
* *CSL.* It matters where the generator's labels are poor (MNIST: −2.5 without it) and not where they are decent
  (FashionMNIST).
* *LA temperature.* τ = 2 helps FashionMNIST (+1.4) and slightly hurts MNIST. BBC captures this residual bias
  without tuning.

**Privacy.**
* *DP on class counts.* Laplace noise at ε = 1 on every client's class histogram costs 0.2 (FashionMNIST) and 1.3
  points (MNIST). At ε = 0.1 the method is still 11–14 points above GeFL-F.
* *Memorisation.* Feature-space MND equals GeFL-F's.

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
  * *Implication.* The remaining IID gap to the paper's best FMNIST number (DDPM-F) is a *generator* gap. §5.8 tests
    exactly that.

* **Bayesian consensus labels.**
  * *Idea.* Replace CSL's fixed mixture by the Bayes posterior under symmetric generator noise, with the generator's
    fidelity estimated without labels from ensemble agreement.
  * *Result.* Worse than CSL: 83.16 vs 84.22 on FashionMNIST IID ($p = 0.004$), and 87.3 vs 90.5 on MNIST LT.
  * *Why.* The estimate is circular. The ensemble was trained on the generator's own samples, so it agrees with
    their labels ($\hat\rho \to 0.999$), and the target collapses to the hard label.
  * *Lesson.* CSL works as *distillation* of the ensemble's class-similarity structure, not as label correction.
* **Held-class real–synthetic alignment.** Aligning each head's embedding of synthetic features to that of real
  features on the classes a client holds *increased* the synthetic–real gap on unheld classes. Dropped.

### 5.8 Generator-agnostic: our method on DDPM-F, the paper's best feature generator

The best FMNIST result of any GeFL variant in the paper is 84.28, from GeFL-F with the feature diffusion model
DDPM-F. Our components do not depend on the generator:
* HWA averages the class columns of DDPM-F's two context-embedding layers over holders.
* LA and CSL act on the heads.

We ported the authors' DDPM-F unchanged: the ddpm16 ContextUnet, $n_T = 200$, a linear $\beta$ schedule from
$10^{-4}$ to 0.02, context dropout 0.1, and Adam at $10^{-4}$ decayed linearly to 0 over $T_{KA}$ rounds, then frozen.
It runs GeFL-F, + CSL and Ours on it. Every round, all clients' synthetic phases draw from one shared pool of fresh
uniform-label samples. For each head this is exactly the reference's per-client draw in distribution, at 1/K of the
sampling cost, and it is the same for every method compared.

On seed 0 of FMNIST IID, GeFL-F on DDPM-F scores 83.16, not the published 84.28. + CSL scores 83.04 and Ours 83.10.
In our hands DDPM-F is only +0.28 over CVAE-F on the same seed, and CSL does not add to it: diffusion samples are
more diverse but less class-pure, with referee fidelity 0.55 against 0.86. **[PENDING: seeds 1–2, and K04 on
Kaggle.]**

### 5.9 Consensus labels make a larger synthetic budget useful (E16, FMNIST IID, 3 seeds)

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

### 5.11 The anchored method (E20, E24, E25; K = 10, 3 seeds unless marked)

| | MNIST LT | FMNIST LT | MNIST IID | FMNIST IID |
|---|---|---|---|---|
| GeFL-F | 75.25 | 58.47 | 96.01 | 82.76 |
| minimal variant, CVAE + HWA + LA + CSL | 90.40 | 76.60 | 96.77 | 83.19 |
| minimal variant, $T_s = 10$ | 90.15 | 79.23 | – | **84.34** |
| PC-VAE + MC | 92.31 | 77.56 | – | 82.92 |
| KME-Gen, server-only generator (1 seed) | 89.88 | 75.11 | – | 81.82 |
| Ours-A: PC + MC + KH + LA + CSL, $T_s = 1$ | 93.11 | 78.22 | 96.67 | 83.35 |
| **Ours-A, $T_s = 10$** | **94.82** | **79.67** | **97.62** | 83.96 |
| **Ours-A, $T_s = 10$, + gated BBC** | **95.17** | **80.55** | (BBC off) | (BBC off) |
| oracle: its heads, last layer re-fit on pooled real data | 94.93 | 81.55 | 97.78 | 84.09 |

Long-tail columns report final balanced accuracy and IID columns report `best_mean_acc` (the paper's metric).
The MNIST IID GeFL-F and minimal-variant numbers come from F10 (same seeds).

**Against GeFL-F and the minimal variant.**
* Over GeFL-F: +19.6 ($p = 0.002$) and +21.2 ($p = 0.011$) under the long tail. In IID: +1.6 on MNIST
  ($p = 0.001$; 97.62, +1.18 over the best of all ten methods in the paper) and +1.2 on FashionMNIST ($p = 0.021$).
* Over the minimal variant: +4.7 and +0.5 under the long tail (positive on every seed). In IID: +0.85 on MNIST
  ($p = 0.001$) and −0.4 on FashionMNIST ($p = 0.18$).
* *Why the IID gap.* In balanced data FedAvg does not collapse class rows, so the CVAE's class-specific parameters
  cost nothing and add detail.

**Mechanism, measured.**
* *Raw samples.* The anchored decoder's raw samples are under-dispersed (tail spread 0.39–0.49 of real). On
  FashionMNIST they are also *purer than real data*: referee fidelity 0.89–0.91, against the referee's 0.84 accuracy
  on real test data.
* *After MC and KH.* Spread is restored to 0.93–0.98, and fidelity moves to real-data difficulty (IID: 0.837 vs
  0.839).
* *KH.* It helps in all 9 paired comparisons: +0.80, +0.66 and +0.43, sign test $p = 0.004$.
* *Proposition 3.* With the anchored generator, $T_s = 10$ adds +1.70 on MNIST LT ($p = 0.04$), where it added −0.05
  to the CVAE (tail fidelity 0.28). The interaction is +1.75, positive on every seed. The same holds on MNIST IID:
  +0.95 for the anchored generator, against −0.18 for consensus labels on the CVAE.

**More clients (E22, MNIST long tail, K = 100, 3 seeds).**

| | GeFL-F | CVAE + HWA + LA | FSG + LA | Ours-A, $T_s = 1$ | **Ours-A, $T_s = 10$** |
|---|---|---|---|---|---|
| balanced accuracy | 75.98 | 83.07 | 86.05 | 90.41 | **94.30** |
| tail recall | 50.3 | 66.6 | 76.0 | 83.4 | **90.5** |

* *Robust to K.* Ours-A loses only 0.5 points from K = 10 to K = 100, while the CVAE + HWA loses 4.7.
* *Margins.* +18.3 over GeFL-F ($p = 0.0004$) and +8.3 over the Gaussian generator ($p = 0.001$).
* *Proposition 3 again.* The synthetic budget is worth +3.9 at K = 100 against +1.7 at K = 10: the optimal synthetic
  share $w^\star \propto 1/n_{\text{real}}$ grows as each client holds fewer real samples (about 60 here).

**CIFAR-10 and SVHN (Kaggle K01, K08; 3 seeds, same seeds and splits).**

| | CIFAR-10 LT | SVHN LT | CIFAR-10 IID | SVHN IID |
|---|---|---|---|---|
| GeFL-F | 33.74 | 50.49 | 59.10 | 75.82 |
| + HWA + LA | 40.73 | **62.45** | – | 75.64 |
| minimal variant (CVAE + HWA + LA + CSL) / + CSL | 42.10 | – | 60.02 | **76.30** |
| FSG + LA (Gaussian, exact statistics) | 44.17 | 44.23 | – | – |
| Ours-A, $T_s = 1$ | **46.06** | 55.89 | 60.39 | 73.86 |
| Ours-A, $T_s = 1$, + gated BBC | **≈ 49** | ≈ 57 | (off) | (off) |
| Ours-A, $T_s = 10$ | 44.38 | 54.41 | **60.93** | 74.38 |
| Ours-A, $T_s = 10$, + gated BBC | ≈ 47 | ≈ 55 | (off) | (off) |
| paper: GeFL-F / best of all ten methods | – | – | 55.86 / 59.36 | 76.26 / 76.26 |

Long-tail columns report final balanced accuracy and IID columns report `best_mean_acc`. The Kaggle summaries
printed BBC values to two significant digits. The tests below are unpaired (Welch, from means and standard
deviations), since the per-seed files have not been copied back yet.

* *CIFAR-10: the anchored method wins in both regimes.* Under the long tail it is +12.3 over GeFL-F ($p = 0.001$),
  +4.0 over the minimal variant ($p = 0.017$) and +1.9 over FSG ($p = 0.11$). With BBC it is +4.8 over FSG
  ($p = 0.003$). Its tail fidelity after KH is 41%, against 23% for the CVAE + HWA. In IID it reaches 60.93, +1.8 over
  our GeFL-F ($p \approx 0.1$ unpaired).
* *A registered prediction that failed.* Before these runs we predicted a tie or a small loss on CIFAR-10, because
  its class means are weak in pixel space (nearest-class-mean accuracy 27%). It won. What decides is the anchored
  generator's fidelity *relative to the CVAE's*, and on CIFAR-10 the CVAE's own fidelity is low.
* *SVHN: the anchored generator fails.* It is −6.6 against + HWA + LA under the long tail and −2.0 against GeFL-F
  in IID. This is Proposition 5: SVHN's anchors are $\sqrt{2F} \approx 0.05$ within-class spreads apart, and the
  synthetic classes are nearly indistinguishable (tail fidelity 24%).
* *The synthetic budget on hard data.* On CIFAR-10 and SVHN, $T_s = 10$ raises the best-round metric in all four
  settings but lowers final balanced accuracy (CIFAR-10 long tail: 44.4 against 46.1, $p = 0.03$). Proposition 3
  predicts this: $w^\star = \sigma^2/(2 n_r b^2)$ falls with the generator's error $b$, and there tail fidelity
  after KH (24–46%) is far below real-data accuracy. On MNIST and FashionMNIST, where fidelity matches real data,
  $T_s = 10$ is better on both metrics.
* *Not beaten.* Image-space GeFL + MixUp (62.67 on CIFAR-10 IID, Table IV) remains above every result here. It uses
  a different pipeline (no shared feature extractor) and data augmentation, which is orthogonal to our method.

**Pending.**
* The hybrid PCR on MNIST and FashionMNIST, including K = 100 (E29), and on SVHN and CIFAR-10 (Kaggle K10).
* FashionMNIST at K = 50 / 100 (Kaggle K06, K08); the IID setting at K = 50 / 100 (E27).
* Formal DP on the anchoring statistics (E26); the component ablation (K09).

### 5.12 Cost

Summed over all stages, our method takes 373 s per run against GeFL-F's 366 s (+1.9%), measured on MNIST under the
long tail, averaged over 3 seeds. Communication adds $C$ numbers per client per round for HWA, and one broadcast of
soft labels for CSL.

---

## 6. Related work

**Model-heterogeneous FL by distillation and partial sharing.** FedAvg [2] assumes that every client trains the
same architecture, so that parameters can be averaged. A first family of methods replaces parameter averaging with
an exchange of predictions. FedMD [3] and DS-FL [6] share client outputs on a public or unlabelled dataset and
distil the consensus back into each model. FedDF [4] fuses heterogeneous client models at the server by ensemble
distillation on unlabelled or generated inputs. FedGKT [5] moves a large model to the server and trains it by
distilling from small edge models. AvgKD [7] analyses these protocols in a kernel-regression model and shows that
distillation-based fusion degrades under data heterogeneity. A second family shares only part of the network.
LG-FedAvg [8] keeps local representations and averages a global head. FedRep [9] does the reverse: it learns a
shared representation and personal heads. FedProto [10] exchanges class prototypes instead of parameters. GeFL-F
[1] combines both ideas: it shares a frozen feature extractor and a federated feature generator, and it leaves
the heads architecture-specific. Our work keeps GeFL-F's communication pattern and architecture freedom
unchanged. It studies what that pattern does to rare classes, a question the distillation and partial-sharing
literature has not asked of a shared *generator*.

**Generator-based and data-free knowledge transfer.** FedGen [11] trains a lightweight generator in feature space at
the server. The generator is fitted so that the ensemble of client predictors classifies its samples as the
conditioning label, and clients then regularise local training with these samples under that hard label. Its
supplementary variant weights each client's predictor per label by that client's label count. FedFTG [12] uses a
server-side generator to fine-tune the global model by data-free distillation. It samples labels in proportion to
global class counts and weights each client's teacher loss per class by the client's share of that class. DENSE
[13] trains a generator against the client ensemble and distils the ensemble into a global model in one round.
FedCG [14] has clients share conditional GANs in place of their private extractors. FedCVAE [15] fits client
conditional VAEs for one-shot FL under extreme heterogeneity. Felo/Velo [16] trains a server-side conditional VAE
on clients' mid-level features and sends synthetic features and per-class averaged logits back to heterogeneous
clients. FedVTC [17] shares feature mean and covariance statistics, from which clients generate data to fine-tune
heterogeneous models. GeFL [1] trains the generator itself by FedAvg across clients, with CVAE, DCGAN or DDPM
backbones, in image space (GeFL) or feature space (GeFL-F). FedGen and FedFTG apply label-count weights to the
*teacher predictions or losses* that train a server-side generator. HWA instead weights the *parameters* of a
client-trained generator, namely its per-class conditioning rows, during aggregation. It is derived from a failure
that is specific to that setting: under FedAvg with Adam's coupled weight decay, rows of classes held by fewer
than half the clients shrink every round.

**FL under class imbalance and long tails.** Much of this work locates the bias in the classifier. CCVR [18]
re-calibrates the classifier after training on virtual features sampled from a Gaussian mixture built from
federated feature statistics. Our federated Gaussian sufficient-statistics generator is close to this idea. CReFF
[19] re-trains the classifier on "federated features" optimised so that their gradients match those of real data.
FEDIC [20] uses ensemble distillation with logit adjustment and a calibration gating network. A second group
changes the local loss. Wang et al. [22] infer each round's class composition with a monitor and counter it with
a Ratio Loss. FedLC [21]
calibrates logits by each client's label frequencies through pairwise margins. FedRS [27] restricts softmax updates
for classes a client does not hold, and MAP [28] develops this restricted-softmax view for incomplete classes.
BalanceFL [24] changes local training so that the uploaded model behaves as if trained on balanced data. CLIMB
[23] casts class imbalance as a constrained learning problem. Fed-GraB [25] re-weights gradients using an
estimate of the global long-tailed prior, and FedLoGe [26] combines a shared backbone with a static ETF
classifier and per-client classifiers. Class-aware aggregation also appears for classifier layers: cwFedAvg [29]
averages per class with weights given by client class distributions, and FedLA [30] weights client updates using
both data size and label distribution. Our LA component is a direct application of logit adjustment by the client's
own prior. It is closely related to FedLC and to the logit adjustment in FEDIC, and we do not claim it as new on its
own. What is new is the setting, a generator and heads trained in separate phases, and the finding that LA helps
only after the generator is repaired. None of these methods trains a class-conditional generator by FedAvg, so none
encounters the row collapse we analyse.

**Long-tailed recognition.** The class-balanced loss [31] re-weights classes by the effective number of samples
$E(n)=(1-\beta^n)/(1-\beta)$, a saturating model of how much new information extra samples add. LDAM [33] enforces
label-dependent margins and defers re-weighting. Decoupling [34] shows that instance-balanced representations with
a re-balanced classifier are hard to beat. Logit adjustment [32] adds $\tau\log\pi$ to the logits during training
(or subtracts it post hoc), which is Fisher-consistent for the balanced error. We borrow the effective number as an
inverse-variance weight for *holders* of a class, not as a loss weight. We use logit adjustment with a per-client
rather than a global prior, which needs no communication. Our diagnosis also depends on the difference between
coupled and decoupled weight decay in Adam [43].

**Synthetic data, label quality and soft labels.** Knowledge distillation [35] trains a student on a teacher's soft
predictions, and label smoothing [36] mixes the one-hot target with a uniform distribution, which can remove
inter-class information useful to a student [37]. Menon et al. [38] show that a teacher that approximates Bayes
class-probabilities lowers the variance of the student's objective, with a bias–variance trade-off when the
approximation is poor. Work on training classifiers with generated data finds that label fidelity of samples, not
only visual quality, limits accuracy. The Classification Accuracy Score [39] trains on generated samples and shows
large drops relative to real data, with class-specific failures. Later studies of diffusion-generated training data
report gains from synthetic data together with clear limits [40, 41, 42]. CSL applies this view to generated
*features* in heterogeneous FL. Its target mixes the one-hot label with the mean softmax of all architectures' heads,
so it is label smoothing toward a learned ensemble posterior rather than toward a uniform distribution. The
bias–variance argument of [38] explains why the mixture ($\beta=0.5$) beats the pure ensemble ($\beta=1$), and why
lower-bias labels make a larger synthetic budget useful. Unlike FedDF, DENSE or our negative server-side variant,
CSL does not distil at the server. It only relabels the clients' synthetic phase.

**Privacy.** Secure aggregation [44] lets a server learn only the sum of client vectors. HWA needs only the sums
$\sum_k E(n_{k,c})\,w_c^{(k)}$ and $\sum_k E(n_{k,c})$, so it runs inside such a protocol and reveals neither
individual rows nor individual histograms. Differential privacy [45] and its use in deep learning [46] and in
federated training at the user or client level [47, 48] give formal guarantees. Our DP experiment adds Laplace
noise to class counts only. A formal accounting for the generator itself is left to future work.

---

### 6.7 Positioning relative to the closest works

GeFL/GeFL-F [1] was evaluated only with IID clients and aggregates every generator parameter by flat FedAvg. We
show that, combined with Adam's coupled weight decay, this shrinks the conditioning rows of every class held by
fewer than half the clients, and HWA removes exactly this failure while reducing to FedAvg when class counts are
equal. FedGen [11] and FedFTG [12] also use label-count weights, but on the teacher predictions or losses that train
a *server-side* generator; we weight the *parameters* of a *client-trained* generator. Both label generated samples
with the hard conditioning label, whereas CSL relabels them with a one-hot/cross-architecture-consensus mixture.
CReFF [19] and CCVR [18] use synthetic or virtual features to re-train one homogeneous classifier after training,
while we keep per-architecture heads and change how the generator is aggregated and its samples labelled throughout
training. FedLC [21] already debiases local training with client label frequencies; our LA belongs to the same
family, and our claim for it is only empirical: it is needed, but it pays off fully only after HWA. FedDF-style
server distillation [4] of the ensemble on generated features lowers IID accuracy in our hands. CSL uses the same
ensemble only as a low-variance label inside the clients' synthetic phase, and this is what makes a larger
synthetic budget pay off. The closest labelling scheme is Felo/Velo [16]. Its server averages clients' logits
*per class* (one vector per class, computed on real data) and uses them as KL targets on clients' *real* samples; its
generated features serve only as MSE targets for the feature extractor. CSL instead labels *each generated feature*
with the consensus of all architectures' heads on that very feature, and trains the heads on it. The per-sample
consensus is what tracks the generator's per-sample label error, the bias $b$ of Proposition 3.

---

## 7. Limitations

* **Balanced data.** In exactly balanced (IID) data the anchored generator is marginally below the CVAE + HWA
  variant (−0.4, not significant). Under any label skew it is better.
* **Upper bound.** Every method is bounded by the frozen shared feature extractor, which GeFL-F's protocol fixes. On
  CIFAR-10 this bound (oracle ≈ 50 under the long tail) is far below the image-space GeFL numbers in the paper.
* **What the statistics reveal.** The exact statistics are federation-level sums, plus counts. Classes held by a
  single client are not hidden by secure aggregation; DP noise (§5.6) is the remedy, at a measured cost.

* **Class means that carry no class information.** On SVHN the anchored generator loses to the CVAE + HWA
  (Proposition 5). The server can detect the regime from the released sums, at no extra privacy cost, and the hybrid
  PCR is the candidate fix.
* Results are on small images and on the paper's backbone sizes. Larger FEs could change the generator's failure
  profile.
* HWA uses aggregated class counts. LCD is the fully local alternative but is weaker.
* CSL's gain in IID data is small (+0.5 to +0.7 points), although it is consistent across seeds.
* MND in feature space is a proxy for privacy, not a guarantee. A formal DP accounting of the generator itself is
  future work.

## References

[1] H. Kang, S. Cha, J. Kang. "GeFL: Model-Agnostic Federated Learning with Generative Models." IEEE Transactions on
Mobile Computing, 2025 (arXiv:2412.18460). https://arxiv.org/abs/2412.18460

[2] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas. "Communication-Efficient Learning of Deep
Networks from Decentralized Data." AISTATS, PMLR 54:1273–1282, 2017. https://proceedings.mlr.press/v54/mcmahan17a.html

[3] D. Li, J. Wang. "FedMD: Heterogenous Federated Learning via Model Distillation." NeurIPS Workshop on Federated
Learning for Data Privacy and Confidentiality, 2019 (arXiv:1910.03581). https://arxiv.org/abs/1910.03581

[4] T. Lin, L. Kong, S. U. Stich, M. Jaggi. "Ensemble Distillation for Robust Model Fusion in Federated Learning."
NeurIPS, 2020 (arXiv:2006.07242). https://proceedings.neurips.cc/paper_files/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf

[5] C. He, M. Annavaram, S. Avestimehr. "Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge."
NeurIPS, 2020 (arXiv:2007.14513). https://arxiv.org/abs/2007.14513

[6] S. Itahara, T. Nishio, Y. Koda, M. Morikura, K. Yamamoto. "Distillation-Based Semi-Supervised Federated Learning
for Communication-Efficient Collaborative Training with Non-IID Private Data." IEEE Transactions on Mobile
Computing (DOI 10.1109/TMC.2021.3070013) (arXiv:2008.06180). https://arxiv.org/abs/2008.06180

[7] A. Afonin, S. P. Karimireddy. "Towards Model Agnostic Federated Learning Using Knowledge Distillation." ICLR,
2022 (arXiv:2110.15210). https://iclr.cc/virtual/2022/poster/6644

[8] P. P. Liang, T. Liu, L. Ziyin, N. B. Allen, R. P. Auerbach, D. Brent, R. Salakhutdinov, L.-P. Morency. "Think
Locally, Act Globally: Federated Learning with Local and Global Representations." NeurIPS Workshop on Federated
Learning, 2019 (arXiv:2001.01523). https://arxiv.org/abs/2001.01523

[9] L. Collins, H. Hassani, A. Mokhtari, S. Shakkottai. "Exploiting Shared Representations for Personalized
Federated Learning." ICML, 2021 (arXiv:2102.07078). https://icml.cc/virtual/2021/poster/10309

[10] Y. Tan, G. Long, L. Liu, T. Zhou, Q. Lu, J. Jiang, C. Zhang. "FedProto: Federated Prototype Learning across
Heterogeneous Clients." AAAI, 36(8):8432–8440, 2022. https://ojs.aaai.org/index.php/AAAI/article/view/20819

[11] Z. Zhu, J. Hong, J. Zhou. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." ICML, PMLR
139:12878–12889, 2021. https://proceedings.mlr.press/v139/zhu21b.html

[12] L. Zhang, L. Shen, L. Ding, D. Tao, L.-Y. Duan. "Fine-tuning Global Model via Data-Free Knowledge Distillation
for Non-IID Federated Learning." CVPR, 2022 (arXiv:2203.09249). https://arxiv.org/abs/2203.09249

[13] J. Zhang, C. Chen, B. Li, L. Lyu, S. Wu, S. Ding, C. Shen, C. Wu. "DENSE: Data-Free One-Shot Federated
Learning." NeurIPS, 2022. https://proceedings.neurips.cc/paper_files/paper/2022/hash/868f2266086530b2c71006ea1908b14a-Abstract.html

[14] Y. Wu, Y. Kang, J. Luo, Y. He, L. Fan, R. Pan, Q. Yang. "FedCG: Leverage Conditional GAN for Protecting
Privacy and Maintaining Competitive Performance in Federated Learning." IJCAI, 2334–2340, 2022.
https://www.ijcai.org/proceedings/2022/324

[15] C. Heinbaugh, E. Luz-Ricca, H. Shao. "Data-Free One-Shot Federated Learning Under Very High Statistical
Heterogeneity." ICLR, 2023. https://iclr.cc/virtual/2023/poster/11962

[16] Y.-H. Chan, E. C.-H. Ngai. "Exploiting Features and Logits in Heterogeneous Federated Learning." Computer
Networks, 2025 (DOI 10.1016/j.comnet.2025.111271) (arXiv:2210.15527). https://arxiv.org/abs/2210.15527

[17] Z. Niu, H. Dong, A. K. Qin. "Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative
Models." ICLR, 2026 (arXiv:2508.01669). https://arxiv.org/abs/2508.01669

[18] M. Luo, F. Chen, D. Hu, Y. Zhang, J. Liang, J. Feng. "No Fear of Heterogeneity: Classifier Calibration for
Federated Learning with Non-IID Data." NeurIPS, 2021 (arXiv:2106.05001). https://neurips.cc/virtual/2021/poster/27280

[19] X. Shang, Y. Lu, G. Huang, H. Wang. "Federated Learning on Heterogeneous and Long-Tailed Data via Classifier
Re-Training with Federated Features." IJCAI, 2218–2224, 2022. https://www.ijcai.org/proceedings/2022/308

[20] X. Shang, Y. Lu, Y.-M. Cheung, H. Wang. "FEDIC: Federated Learning on Non-IID and Long-Tailed Data via
Calibrated Distillation." IEEE ICME, 2022 (arXiv:2205.00172). https://arxiv.org/abs/2205.00172

[21] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, C. Wu. "Federated Learning with Label Distribution Skew via
Logits Calibration." ICML, PMLR 162:26311–26329, 2022. https://proceedings.mlr.press/v162/zhang22p.html

[22] L. Wang, S. Xu, X. Wang, Q. Zhu. "Addressing Class Imbalance in Federated Learning." AAAI, 35(11), 2021
(arXiv:2008.06217). https://ojs.aaai.org/index.php/AAAI/article/view/17219

[23] Z. Shen, J. Cervino, H. Hassani, A. Ribeiro. "An Agnostic Approach to Federated Learning with Class Imbalance"
(CLIMB). ICLR, 2022. https://openreview.net/forum?id=Xo0lbDt975 (listing: https://iclr.cc/virtual/2022/poster/6104)

[24] X. Shuai, Y. Shen, S. Jiang, Z. Zhao, Z. Yan, G. Xing. "BalanceFL: Addressing Class Imbalance in Long-Tail
Federated Learning." ACM/IEEE IPSN, 2022.
https://conferences.computer.org/cpsiot/pdfs/IPSN2022-6R1M30NXCSXmbVKUqzz1Of/962400a259/962400a259.pdf

[25] Z. Xiao, Z. Chen, S. Liu, H. Wang, Y. Feng, J. Hao, J. T. Zhou, J. Wu, H. H. Yang, Z. Liu. "Fed-GraB:
Federated Long-tailed Learning with Self-Adjusting Gradient Balancer." NeurIPS, 2023 (arXiv:2310.07587).
https://proceedings.neurips.cc/paper_files/paper/2023/hash/f4b8ddb9b1aa3cb11462d64a70b84db2-Abstract-Conference.html

[26] Z. Xiao, Z. Chen, et al. (corresponding author Z. Liu). "FedLoGe: Joint Local and Generic Federated Learning
under Long-tailed Data." ICLR, 2024 (arXiv:2401.08977).
https://proceedings.iclr.cc/paper_files/paper/2024/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html

[27] X.-C. Li, D.-C. Zhan. "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data."
ACM SIGKDD, 995–1005, 2021. https://dl.acm.org/doi/10.1145/3447548.3467254 *(metadata consistent across citing
papers, e.g. [21]; the ACM page itself could not be loaded)*

[28] X.-C. Li, S. Song, Y. Li, B. Li, Y. Shao, Y. Yang, D.-C. Zhan. "MAP: Model Aggregation and Personalization in
Federated Learning with Incomplete Classes." IEEE Transactions on Knowledge and Data Engineering, 2024
(arXiv:2404.09232). https://arxiv.org/abs/2404.09232

[29] G. Lee, D. Choi. "Class-Wise Federated Averaging for Efficient Personalization." ICCV, 1773–1782, 2025
(arXiv:2406.07800).
https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Class-Wise_Federated_Averaging_for_Efficient_Personalization_ICCV_2025_paper.html

[30] A. Khalil, A. Wainakh, E. Zimmer, J. Parra-Arnau, A. Fernández Anta, T. Meuser, R. Steinmetz. "Label-Aware
Aggregation for Improved Federated Learning." IEEE FMEC, 2023. https://dspace.networks.imdea.org/handle/20.500.12761/1742

[31] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. "Class-Balanced Loss Based on Effective Number of Samples."
CVPR, 9268–9277, 2019.
https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html

[32] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, S. Kumar. "Long-tail learning via logit
adjustment." ICLR, 2021 (arXiv:2007.07314). https://iclr.cc/virtual/2021/poster/2675

[33] K. Cao, C. Wei, A. Gaidon, N. Arechiga, T. Ma. "Learning Imbalanced Datasets with Label-Distribution-Aware
Margin Loss." NeurIPS, 2019 (arXiv:1906.07413).
https://proceedings.neurips.cc/paper/2019/hash/621461af90cadfdaf0e8d4cc25129f91-Abstract.html

[34] B. Kang, S. Xie, M. Rohrbach, Z. Yan, A. Gordo, J. Feng, Y. Kalantidis. "Decoupling Representation and
Classifier for Long-Tailed Recognition." ICLR, 2020 (arXiv:1910.09217). https://arxiv.org/abs/1910.09217

[35] G. Hinton, O. Vinyals, J. Dean. "Distilling the Knowledge in a Neural Network." NIPS Deep Learning and
Representation Learning Workshop, 2015 (arXiv:1503.02531). https://arxiv.org/abs/1503.02531

[36] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, Z. Wojna. "Rethinking the Inception Architecture for Computer
Vision." CVPR, 2818–2826, 2016 (arXiv:1512.00567). https://arxiv.org/abs/1512.00567

[37] R. Müller, S. Kornblith, G. Hinton. "When Does Label Smoothing Help?" NeurIPS, 2019 (arXiv:1906.02629).
https://papers.nips.cc/paper/8717-when-does-label-smoothing-help

[38] A. K. Menon, A. S. Rawat, S. Reddi, S. Kim, S. Kumar. "A Statistical Perspective on Distillation." ICML, PMLR
139:7632–7642, 2021. https://proceedings.mlr.press/v139/menon21a.html

[39] S. Ravuri, O. Vinyals. "Classification Accuracy Score for Conditional Generative Models." NeurIPS, 2019
(arXiv:1905.10887). https://proceedings.neurips.cc/paper/2019/hash/fcf55a303b71b84d326fb1d06e332a26-Abstract.html

[40] R. He et al. "Is Synthetic Data from Generative Models Ready for Image Recognition?" ICLR, 2023
(arXiv:2210.07574). https://arxiv.org/abs/2210.07574

[41] S. Azizi, S. Kornblith, C. Saharia, M. Norouzi, D. J. Fleet. "Synthetic Data from Diffusion Models Improves
ImageNet Classification." Transactions on Machine Learning Research, 2023 (arXiv:2304.08466).
https://arxiv.org/abs/2304.08466

[42] M. B. Sarıyıldız, K. Alahari, D. Larlus, Y. Kalantidis. "Fake It Till You Make It: Learning Transferable
Representations from Synthetic ImageNet Clones." CVPR, 8011–8021, 2023.
https://openaccess.thecvf.com/content/CVPR2023/html/Sariyildiz_Fake_It_Till_You_Make_It_Learning_Transferable_Representations_From_CVPR_2023_paper.html

[43] I. Loshchilov, F. Hutter. "Decoupled Weight Decay Regularization." ICLR, 2019 (arXiv:1711.05101).
https://arxiv.org/abs/1711.05101

[44] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, K. Seth.
"Practical Secure Aggregation for Privacy-Preserving Machine Learning." ACM CCS, 2017. https://eprint.iacr.org/2017/281
(also https://research.google/pubs/pub47246/)

[45] C. Dwork, F. McSherry, K. Nissim, A. Smith. "Calibrating Noise to Sensitivity in Private Data Analysis." TCC,
LNCS 3876:265–284, 2006. https://www.microsoft.com/en-us/research/publication/calibrating-noise-to-sensitivity-in-private-data-analysis/

[46] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, L. Zhang. "Deep Learning with
Differential Privacy." ACM CCS, 308–318, 2016 (arXiv:1607.00133). https://arxiv.org/abs/1607.00133

[47] H. B. McMahan, D. Ramage, K. Talwar, L. Zhang. "Learning Differentially Private Recurrent Language Models."
ICLR, 2018 (arXiv:1710.06963). https://iclr.cc/virtual/2018/poster/187

[48] R. C. Geyer, T. Klein, M. Nabi. "Differentially Private Federated Learning: A Client Level Perspective." arXiv
preprint arXiv:1712.07557, 2017. https://arxiv.org/abs/1712.07557
