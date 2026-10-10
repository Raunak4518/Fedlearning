# Rare Classes Vanish from Federated Feature Generators — and Exact Statistics Bring Them Back

*Single-idea draft. The method is called ESA (exact-statistics anchoring); the name is a placeholder. Every number is
from 3 seeds unless marked. Cells marked PENDING name the run that fills them.*

## Abstract

Generator-based federated learning lets clients with different model architectures learn from one another. In
GeFL-F, the clients share a small frozen feature extractor and a conditional feature generator, and each client
trains its own classifier head on real and generated features. We show that GeFL-F fails when client data are
long-tailed, for a reason that can be stated exactly. Under federated averaging, the generator's class-specific
parameters shrink every round for every class held by fewer than half of the clients. Rare classes vanish from the
generator, and every head is taught from class-agnostic features.

Our single idea is that **class-level information should not be a parameter that federated averaging estimates. It
should be anchored to statistics that secure aggregation computes exactly.** Per-class counts, sums and second
moments are sums over clients. The server receives them exactly, they do not depend on how the data are split, and
they reveal nothing beyond federation-level totals. We use them at each point where class information enters the
pipeline:
* the generator's class identity comes from the exact class means;
* generated samples are moved to the exact class mean and covariance by the minimum-displacement (Gelbrich) map;
* generated samples are then herded to the exact class kernel embedding;
* where the exact means do not identify the classes, the same statistics detect it, and class identity comes from
  class parameters aggregated with exact class counts instead.

Each step has a guarantee: the collapse cannot occur, the corrections are displacement-optimal, and the herding lowers
the term of each head's risk bound that limits transfer.

Under a 100:1 long tail, ESA raises balanced accuracy over GeFL-F from 75.3 to 95.2 on MNIST, 58.5 to 80.6 on
FashionMNIST, 33.7 to about 47 on CIFAR-10 and 50.5 to about 67 on SVHN. With 100 clients it holds 94.3 on MNIST. In
GeFL-F's own balanced setting it exceeds the best of the ten methods in the original paper on MNIST, CIFAR-10 and SVHN,
and is level on FashionMNIST. Every released statistic can be made formally $(\varepsilon, \delta)$-differentially
private. At $\varepsilon = 2$ the method stays 7–10 points above GeFL-F, which offers no formal guarantee.

---

## 1. Introduction

Federated learning across clients that run *different* model architectures cannot average models. GeFL [1] instead
shares a generative model: clients train a conditional generator by federated averaging (FedAvg), and every client
then trains its own model on real and generated data. GeFL-F, its feature-level variant, shares a small frozen feature
extractor (FE) and a generator over FE features. The heads stay private and heterogeneous.

Real federations are long-tailed: some classes are common everywhere and others are held by a few clients in small
numbers. We find that GeFL-F breaks under exactly this condition. At a 100:1 imbalance, GeFL-F's balanced accuracy
falls 20 points below what its own heads can reach on pooled real data, and its tail classes are nearly never
predicted. The cause is not a detail of the architecture (§3). The generator's class identity lives in class-specific
parameters, and FedAvg drives the parameters of every class held by fewer than half of the clients toward zero, round
after round. Generated samples of a rare class become class-agnostic, and every head learns from them.

**The idea.** Secure aggregation [44] lets the server compute sums over clients exactly, without seeing any client's
term. Many quantities that decide how a generator represents a class are sums over clients: counts $n_c$, feature sums
$\sum h$, second moments $\sum h h^\top$, kernel features $\sum \phi(h)$. They are partition-invariant: the pooled class
mean computed from summed client sums equals the centralised class mean for every way of splitting the data. A
FedAvg-trained parameter has no such property. ESA therefore anchors every class-level quantity to these exact
statistics instead of to averaged parameters.

**Contributions.**
1. *Diagnosis.* We show why rare classes vanish from FedAvg-trained conditional generators: the conditioning-row
   recursion (Proposition 1). We measure it in GeFL-F on four datasets.
2. *One principle, instantiated where class information enters.*
   * class identity from the exact class mean, with no class-specific parameter left to collapse;
   * sample moments corrected to the exact class mean and covariance;
   * the sample distribution herded to the exact kernel embedding.

   The same statistics decide when the mean cannot identify a class, and then class parameters are aggregated with
   exact counts.
3. *Theory for each anchor.*
   * The mean-anchored generator's expressiveness limit (Proposition 2).
   * The minimum-displacement optimality of the moment corrections (Propositions 3 and 4).
   * A kernel risk bound for herding (Lemma 1).
   * When more synthetic training helps (Proposition 5).
4. *Experiments* in GeFL-F's exact protocol, validated against the authors' code.
   * Under the long tail: +15 to +22 points over GeFL-F on four datasets.
   * In GeFL-F's own balanced setting: above the original paper's best method on three of four datasets.
   * Robust to 100 clients.
   * Formal privacy at a measured cost.

---

## 2. Setting: GeFL-F

There are $K$ clients. Client $k$ holds data $D_k$ and a head $h_{a(k)}$ of one of $G$ architectures (CNN-1 … CNN-10
of [1]). GeFL-F runs three stages.
1. **FE warm-up.** A small FE $\phi$ (one or two convolution blocks; for example $3\times14\times14$ features on MNIST
   and $10\times16\times16$ on CIFAR-10) is trained with the heads by FedAvg, then frozen.
2. **Generator.** A conditional VAE $G(h \mid y)$ over FE features is trained by FedAvg with Adam (learning rate
   $10^{-3}$, coupled weight decay $10^{-3}$). The label enters through a learned projection whose $c$-th row $w_c$
   is the only class-specific parameter.
3. **Heads.** Each round, each client trains its head for $T_s$ epochs on generated features ($T_s = 1$ in [1]) and
   $T_r = 5$ epochs on its real features. Heads of the same architecture are averaged.

**Long tail.** Class sizes $n_c \propto \mathrm{IF}^{c/(C-1)}$ with imbalance factor IF = 100 (the rarest class has
24 MNIST training samples). Client shares follow Dirichlet(0.5), and test sets are balanced. The paper's own setting
(IID, balanced) is reported separately.

---

## 3. Why rare classes vanish

**Proposition 1 (conditioning-row recursion).** Let class $c$ be held by $m_c$ of $K$ clients. A non-holder receives
no data gradient on $w_c$, only the coupled decay $\lambda w_c$. Adam normalises each coordinate's step by its
second-moment estimate, so a pure-decay gradient moves the coordinate by about $\eta\,\mathrm{sign}(w_c)$ per step,
whatever $\lambda$ is. Over $S$ local steps a holder can grow $|w_c|$ by at most $S\eta$ per coordinate, and a
non-holder shrinks it by $S\eta$ until it reaches zero. After flat averaging,
$$\Delta |w_c| \;\le\; \Big(\frac{2 m_c}{K} - 1\Big) S\eta ,$$
which is negative for every class held by fewer than half of the clients. $\square$ (Appendix A.1.)

**Consequences.**
* *Collapse.* Rows of rare classes are driven toward zero. The decoder receives almost no class signal for them, and
  "class-$c$" samples become class-agnostic.
* *Dilution without weight decay.* Without the decay (or with SGD), the data step is still scaled by $m_c/K$, so rare
  classes learn $m_c/K$ times slower.

**Measured** (IF = 100, Dir(0.5), K = 10). The tail-to-head row-norm ratio of GeFL-F's generator is 0.33 on MNIST,
0.22 on FashionMNIST and 0.18–0.49 on SVHN. The *fidelity* of its tail classes — the accuracy, on generated features,
of a referee trained on real held-out features — is 8.8% on MNIST and 19.6% on FashionMNIST. Excluding non-holders
from the rows' average restores the ratio to 0.80–1.05 (Table 6).

---

## 4. Exact-statistics anchoring (ESA)

### 4.1 What the server can know exactly

Under secure aggregation each client uploads, once, per-class sums over its features:
$$n_c = \textstyle\sum_k n_{kc}, \quad S_c = \sum_k \sum_{i\in k,\,y_i=c} h_i, \quad Q_c = \sum_k \sum_{i\in k,\,y_i=c} \|h_i\|^2,$$
$$M = \textstyle\sum_k \sum_i h_i h_i^\top, \qquad Z_c = \sum_k \sum_{i\in k,\,y_i=c} z_i z_i^\top \ (z = B^\top h), \qquad \Phi_c = \sum_k \sum_{i\in k,\,y_i=c} \phi(h_i).$$

The server sees the totals and never a client's term. From them it gets:
* the class means $m_c = S_c/n_c$;
* the class spreads $V_c = Q_c/n_c - \|m_c\|^2$;
* the pooled covariance and its top-$k$ principal basis $B$ (from $M$);
* the class covariances in that basis (from $Z_c$; $k = 256$);
* the class kernel mean embeddings $\mu_c = \Phi_c/n_c$, with bounded random Fourier features $\|\phi\| \le \sqrt2$.

All of these are exact for any partition of the data.

### 4.2 Class identity from the exact mean

The generator models only the within-class residual around the exact mean,
$$\tilde h = \mathrm{ReLU}\big(m_y + \mathrm{dec}(z,\ A m_y)\big), \qquad z \sim \mathcal N(0, I),$$
with $A$ a class-shared projection; the encoder also sees $m_y$. There is no class-specific parameter, so
Proposition 1 has nothing to act on. Tail classes borrow within-class variation from every class through the shared
decoder.

**The anchor's limit.** A mean-anchored generator can only say as much about a class as the mean does.

**Proposition 2 (separation bound).** If $\mathrm{dec}$ is $L$-Lipschitz in its second argument, then for any two
classes $W_1(G_c, G_{c'}) \le (1 + L\|A\|_2)\,\|m_c - m_{c'}\|$. $\square$ (Appendix A.2.)

* *What it means.* The synthetic classes can be told apart only in proportion to how far apart their exact means are.
* *The decisive quantity.* In units of the within-class spread, the means are on average $\sqrt{2F}$ apart, where
  $F = \overline{\|m_c - \bar m\|^2}\,/\,\overline{V_c}$. The server computes $F$ from §4.1, with no extra disclosure.
* *The rule.* If $F < 0.05$, the mean cannot carry class identity. ESA then takes class identity from the original
  generator's class rows, aggregated with exact class counts so that Proposition 1 cannot act:
  $w_c \leftarrow \sum_k E(n_{kc})\,w_c^{(k)} / \sum_k E(n_{kc})$, where $E(n) = (1-\beta^n)/(1-\beta)$ and
  $\beta = 0.999$ [31]. Non-holders have $E(0) = 0$.
* *The threshold is not tuned.* $F$ is 0.01 on SVHN and 0.14–0.60 on the other datasets, so any threshold between
  0.02 and 0.1 makes the same choice everywhere.

### 4.3 Moments: the minimum-displacement correction

A trained generator is measurably wrong about class moments; a VAE is under-dispersed, for example. ESA moves each
generated class onto its exact moments with the map that changes the samples least.

**Proposition 3 (mean and spread).** Among all maps $T$ with $\mathbb E\,T(X) = m_c$ and
$\mathbb E\|T(X) - m_c\|^2 = V_c$, the map $T^\star(x) = m_c + \sqrt{V_c/\tilde V_c}\,(x - \tilde m_c)$ minimises
$\mathbb E\|T(X) - X\|^2$, where $\tilde m_c, \tilde V_c$ are the generator's class moments. $\square$ (Appendix A.3.)

**Proposition 4 (mean and covariance).** Among all maps $T$ with $\mathbb E\,T(X) = m_c$ and
$\mathrm{Cov}\,T(X) = \Sigma_c$, the map $T^\star(x) = m_c + A_c(x - \tilde m_c)$ minimises $\mathbb E\|T(X)-X\|^2$, with
$A_c = \tilde\Sigma_c^{-1/2}(\tilde\Sigma_c^{1/2}\Sigma_c\tilde\Sigma_c^{1/2})^{1/2}\tilde\Sigma_c^{-1/2}$. The minimum
is the squared Gelbrich distance [Gelbrich 1990; Olkin and Pukelsheim 1982]. $\square$ (Appendix A.4.)

* *How ESA applies them.* The covariance map acts in the principal subspace $B$, and Proposition 3 restores the exact
  spread outside it.
* *Tail classes.* Their $k\times k$ covariance is shrunk toward the pooled one with prior strength $k$, then rescaled
  to the class's exact variance. Only the *shape* is borrowed.
* *Which moments.* With informative means ($F \ge 0.05$) the mean-and-spread correction suffices. With uninformative
  means, class identity lives in the second moments: on SVHN pixels a quadratic classifier built from class
  covariances reaches 54%, against 21% for the shared-covariance (LDA) rule and 10% for nearest mean. There ESA
  corrects the full covariance.
* *Projection.* A final ReLU projects onto the non-negative orthant, where FE features live.

### 4.4 Distribution: herding to the exact kernel embedding

**Lemma 1.** For a head whose loss lies in the RKHS ball of radius $B$,
$|\mathbb E_{G_c}\ell - \mathbb E_{P_c}\ell| \le B\,\mathrm{MMD}(G_c, P_c)$.

For each class, ESA draws $4n$ corrected candidates and selects $n$ of them by Frank–Wolfe herding on
$\|\frac1n\sum_t \phi(x_t) - \mu_c\|$ (kernel herding [Chen et al. 2010]). This lowers exactly the term that bounds
how much a head can transfer from synthetic to real data.

### 4.5 Heads

These are standard components, used for every method we report wherever they apply:
* **Logit adjustment** with the client's own prior on real batches [32].
* **Consensus soft labels** on the synthetic pool, $t = \tfrac12 e_y + \tfrac12 \bar p$, where $\bar p$ is the mean
  prediction of the $G$ architectures' heads. This is the cross-architecture distillation of [3, 4] applied to
  generated features. It uses no client message beyond prediction sums.
* **Synthetic budget** $T_s = 10$. Proposition 5 (Appendix A.5) explains why: in a stylised estimator model the optimal
  synthetic share $w^\star = \sigma^2/(2 n_r b^2)$ falls with the generator's label bias $b$. A faithful, anchored
  generator therefore makes a larger budget pay off. It also predicts the converse, which we observe on CIFAR-10 and
  SVHN (§5.5).
* **Balanced bias calibration.** After training, each head's $C$ logit offsets are fitted so that it predicts every
  class equally often on balanced *anchored* synthetic features (the prior correction of Saerens et al. 2002). It is
  switched on only when the exact global counts are imbalanced (max/min > 2).

### 4.6 Algorithm

**ESA, server side, once after the FE warm-up.**
1. Receive the sums of §4.1 by secure aggregation.
2. Compute $F$ and choose the class-identity source (§4.2).

**Generator.** Train it by FedAvg: the mean-anchored generator if $F \ge 0.05$; otherwise class rows aggregated with
exact counts.

**Each head round.**
1. Draw a synthetic pool.
2. Correct its moments: mean and spread if $F \ge 0.05$, mean and covariance otherwise.
3. Herd it to the exact $\mu_c$.
4. Label it by consensus.
5. Clients train their heads with $T_s = 10$ synthetic and $T_r = 5$ real epochs, with logit adjustment.

**End of training.** Apply the gated calibration.

### 4.7 What is disclosed

ESA adds one upload per client of federation-level sums: about 0.3 MB, plus $C k^2$ numbers when the covariance is
used. GeFL-F's other messages are unchanged. Every release can be made $(\varepsilon, \delta)$-DP:
* features are clipped to a public norm bound;
* each release gets Gaussian noise, with the budget composed by zCDP [Bun and Steinke 2016];
* prototypes, corrections, herding targets and both gates are post-processing.

Where a class's noisy statistic would be too noisy to help, the correction and herding fall back to the plain
generator for that class. This is decided from the noisy counts, at no extra cost.

---

## 5. Experiments

### 5.1 Protocol

We follow [1] exactly:
* FE per Table XIV, heads per Tables XXI–XXIII, generator per Table XX.
* SGD with learning rate 0.1 for the heads; the paper's round schedules.
* Data fraction 0.1 (0.5 for CIFAR-10).
* Metrics: balanced accuracy and tail recall (the four rarest classes) under the long tail; the paper's
  `best_mean_acc` in its IID setting.
* Each seed fixes the split and the initialisation for every method; p-values are paired over seeds.

**Validation.** Our GeFL-F matches the paper and the authors' released code: MNIST IID 96.02 against 96.07 (code) and
95.47 (paper). Under the long tail it is 1.9 points *below* the authors' code, so gains are if anything understated.
On CIFAR-10 IID it scores 59.10 against the paper's 55.86; we compare against our own run there.

### 5.2 Main result: long-tailed clients

**Table 1.** Balanced accuracy, IF = 100, Dir(0.5), K = 10.

| | MNIST | FashionMNIST | CIFAR-10 | SVHN |
|---|---|---|---|---|
| GeFL-F [1] | 75.25 | 58.47 | 33.74 | 50.49 |
| GeFL-F + logit adjustment | 78.92 | 64.78 | 37.30 | 56.44 |
| GeFL-F + count-anchored rows + LA | 87.89 | 76.02 | 40.73 | 62.45 |
| Gaussian generator from exact statistics + LA | 87.04 | 73.60 | 44.17 | 44.23 |
| **ESA** | **95.17** | **80.55** | **≈ 47** | **≈ 67** |
| oracle: ESA's heads, last layer re-fit on pooled real data | 94.93 | 81.55 | 50.6 | – |

* *Ranges.* ESA is +19.9, +22.1, about +13 and about +16 over GeFL-F. Tail recall rises from 52 to 91 on MNIST and
  from 35 to 74 on FashionMNIST.
* *Near the ceiling.* On MNIST ESA reaches the accuracy of its own heads re-fit on pooled real data.
* *The rule at work.* SVHN is the dataset where $F < 0.05$: there ESA uses the count-anchored rows with the covariance
  correction.
* *Notes.* "≈" marks values that the Kaggle summaries printed with two significant digits; the per-seed files are
  PENDING. The SVHN value is at $T_s = 1$; $T_s = 10$ is PENDING (NB25).

### 5.3 GeFL-F's own balanced setting

**Table 2.** `best_mean_acc`, IID, K = 10, against all ten methods in [1] (GeFL and GeFL-F with DCGAN, CVAE and DDPM,
plus FedProx and FedALA).

| | MNIST | FashionMNIST | CIFAR-10 | SVHN |
|---|---|---|---|---|
| GeFL-F (paper) | 95.47 | 83.14 | 55.86 | 76.26 |
| best of the ten methods in [1] | 96.44 | 84.28 | 59.36 | 76.26 |
| GeFL-F (ours) | 96.01 | 82.76 | 59.10 | 75.82 |
| **ESA** | **97.62** | 83.96 | **60.93** | **77.07** |

* *Against the paper's best.* +1.18, −0.32 (level), +1.57 and +0.81.
* *Against our own GeFL-F.* +1.6 on MNIST ($p = 0.001$) and +1.2 on FashionMNIST ($p = 0.021$).
* *Gains are smaller than under the long tail,* as they should be: with balanced clients FedAvg has no rare class to
  lose.
* *K = 50 / 100.* The paper's other two columns are PENDING (local E27; Kaggle NB17, NB26–NB29).

Image-space GeFL with data augmentation (Table IV of [1]) reaches 62.67 (MixUp) and 61.66 (CutMix) on CIFAR-10. It
trains full models on raw images without a frozen FE, and augmentation is orthogonal to ESA (Appendix C).

### 5.4 More clients

**Table 3.** MNIST, IF = 100, K = 100.

| GeFL-F | count-anchored rows + LA | Gaussian from exact statistics + LA | **ESA** |
|---|---|---|---|
| 75.98 | 83.07 | 86.05 | **94.30** |

* *No loss with K.* ESA loses 0.5 points from K = 10 to K = 100, while the count-anchored CVAE loses 4.8. The
  mean-anchored generator has no class parameter to estimate from shrinking per-client data.
* *Margins.* +18.3 over GeFL-F ($p = 0.0004$) and +8.3 over the Gaussian generator ($p = 0.001$), on every seed.

### 5.5 What each anchor contributes

**Table 4.** Adding the anchors one at a time (balanced accuracy, IF = 100, K = 10, same seeds).

| | MNIST | FashionMNIST |
|---|---|---|
| GeFL-F + LA + consensus labels | 90.40 | 76.60 |
| class identity from the exact mean | 90.75 | 75.96 |
| + exact mean and spread | 92.31 | 77.56 |
| + herding to the exact embedding | 93.11 | 78.22 |
| + $T_s = 10$ | 94.82 | 79.67 |
| + gated calibration (= ESA) | 95.17 | 80.55 |

**Each anchor's effect.**
* *Herding* helped in all 9 paired comparisons (sign test $p = 0.004$).
* *The 10-epoch budget* adds +1.7 with the anchored generator and nothing with the original one, which is
  Proposition 5's interaction.
* *Covariance correction where means fail* (SVHN, 3 seeds, paired): +6.0 under the long tail (every seed,
  $p = 0.016$) and +2.7 in IID ($p = 0.026$) over the mean-and-spread correction.
* *Leave-one-out ablation* of the final method is PENDING (NB18).

**Where more synthetic training stops paying.** On CIFAR-10 and SVHN, $T_s = 10$ raises the paper's best-round metric
in all four settings but lowers the final balanced accuracy (CIFAR-10 long tail: 44.4 against 46.1). There tail
fidelity after herding (24–46%) is far below real-data accuracy, so Proposition 5 predicts a smaller optimal synthetic
share.

### 5.6 Why it works

**Table 5.** Mechanism measurements.

| | GeFL-F | ESA |
|---|---|---|
| tail fidelity, MNIST / FashionMNIST (same seeds) | 10.3 / 19.9 | 88.6 / 78.7 |
| tail spread relative to real (raw → after correction and herding) | – | 0.39–0.49 → 0.93–0.98 |
| IID fidelity vs the referee's real-data accuracy (FashionMNIST) | – | 0.837 vs 0.839 |

**Table 6.** Separation $F$ of the exact class means, which the rule of §4.2 uses (FE feature space).

| MNIST | FashionMNIST | CIFAR-10 | SVHN |
|---|---|---|---|
| 0.29 | 0.53–0.60 | ≈ 0.14 (reduced schedule; full PENDING) | 0.01 |

**Reading the measurements.**
* *Spread.* The anchored decoder's raw samples are under-dispersed. After the correction and herding, spread is
  realistic and fidelity equals the difficulty of real data: the samples are as hard as real ones, not purer.
* *SVHN.* Nearest-class-mean accuracy in feature space is 12% (chance 10%), so Proposition 2 applies, and the rule
  selects count-anchored rows.

### 5.7 Formal privacy

**Table 7.** All ESA releases made $(\varepsilon, 10^{-5})$-DP (long tail, K = 10).

| | MNIST | FashionMNIST |
|---|---|---|
| GeFL-F (no formal guarantee) | 75.25 | 58.47 |
| ESA, exact statistics | 94.82 | 79.67 |
| ESA, $\varepsilon = 8$ | 90.36 | 72.77 |
| ESA, $\varepsilon = 2$ | 85.06 | 65.59 |

* *Where the cost falls.* Mostly on the rarest classes, whose 24 samples cannot be both private and accurate. The
  correction is applied to 7 of 10 classes at $\varepsilon = 8$ and to 5 at $\varepsilon = 2$.
* *Memorisation.* Feature-space MND equals GeFL-F's.
* *Scope.* The generator's FedAvg training is not DP, in either method.

---

## 6. Related work

**Generator-based and data-free FL.** FedGen [11], FedFTG [12], DENSE [13], FedCG [14] and GeFL [1] share or distil
through generators. All of them learn class conditioning by FedAvg or by distillation. We show this conditioning
collapses under the long tail, and we replace it with exact statistics.

**Federated long-tail learning.**
* *Logit calibration:* FedLC [21] and FedRS [27].
* *Classifier re-training on federated features:* CReFF [19].
* *Calibrated distillation:* FEDIC [20].
* *Reweighting and gradient balancing:* CLIMB [23], BalanceFL [24], Fed-GraB [25] and FedLoGe [26].

All of these assume shared or aggregable classifiers. In model-heterogeneous FL the heads cannot be averaged, and the
generator is the only shared class-conditional object. We use logit adjustment [32] (the principle behind FedLC) as a
component, and test re-training on synthetic features (the principle behind CReFF) as an alternative (Appendix B).

**Prototypes and statistics in FL.** FedProto [10] shares class prototypes for regularisation. Prototypes have not
been used to *anchor a generator*, nor have second moments, kernel embeddings or the displacement-optimal correction.

**Moment matching and transport.** Proposition 4 is the Gelbrich bound [Gelbrich 1990; Olkin and Pukelsheim 1982].
Herding [Chen et al. 2010] and MMD [Gretton et al. 2012] are classical. Our contribution is using exact federated
statistics as their targets.

---

## 7. Limitations

* **The rule is set from four datasets.** $F$ separates them by more than a factor of ten. A dataset with intermediate
  separation would need both branches evaluated.
* **Balanced data.** Gains are +1 to +2 points. Image-space GeFL with MixUp or CutMix stays above ESA on CIFAR-10 IID.
* **The frozen FE bounds every method,** as GeFL-F's protocol fixes it.
* **Privacy.** Secure aggregation hides individual clients but not a class held by a single client. Formal DP removes
  this at the measured cost of Table 7. The generator's own FedAvg is not DP.
* **The synthetic budget.** $T_s = 10$ trades final-round accuracy for peak accuracy on data where the generator's
  fidelity is low.

---

## Appendix A. Proofs

**A.1 Proposition 1.**
* *Non-holder.* A non-holder's gradient on $w_c$ is $\lambda w_c$. Adam's update is $\eta\,\hat m/\sqrt{\hat v}$, and
  for a constant-sign gradient $\hat m/\sqrt{\hat v} \to \mathrm{sign}(w_c)$, so each step moves $|w_c|$ by $\eta$
  toward zero. Over $S$ steps that is $S\eta$, until zero.
* *Holder.* A holder's normalised step is at most $\eta$ in magnitude, so $|w_c|$ grows by at most $S\eta$.
* *Average.* Flat averaging weights holders by $m_c/K$ and non-holders by $1 - m_c/K$.

**A.2 Proposition 2.** Couple both classes through the same $z$. ReLU is 1-Lipschitz, so
$\|X_c - X_{c'}\| \le \|m_c - m_{c'}\| + L\|A(m_c - m_{c'})\|$. Taking expectations bounds $W_1$.

**A.3 Proposition 3.** With $Y = T(X)$,
$\mathbb E\|Y-X\|^2 = V_c + \tilde V_c + \|m_c - \tilde m_c\|^2 - 2\,\mathbb E\langle Y - m_c, X - \tilde m_c\rangle$.
By Cauchy–Schwarz the last expectation is at most $\sqrt{V_c\tilde V_c}$, with equality if and only if
$Y - m_c = \sqrt{V_c/\tilde V_c}\,(X - \tilde m_c)$.

**A.4 Proposition 4.** Let $K = \mathbb E[(Y - m_c)(X - \tilde m_c)^\top]$. Then
$\mathbb E\|Y - X\|^2 = \|m_c - \tilde m_c\|^2 + \mathrm{tr}\,\Sigma_c + \mathrm{tr}\,\tilde\Sigma_c - 2\,\mathrm{tr}\,K$.
* *Upper bound on $\mathrm{tr}\,K$.* The joint covariance is positive semi-definite, so
  $K = \Sigma_c^{1/2} R\,\tilde\Sigma_c^{1/2}$ with $\|R\|_2 \le 1$. Hence
  $\mathrm{tr}\,K \le \|\tilde\Sigma_c^{1/2}\Sigma_c^{1/2}\|_* = \mathrm{tr}(\tilde\Sigma_c^{1/2}\Sigma_c\tilde\Sigma_c^{1/2})^{1/2}$.
* *Attained by $T^\star$.* It gives $K = A_c\tilde\Sigma_c$, with exactly this trace, and
  $\mathrm{Cov}\,T^\star(X) = \Sigma_c$.
* *Subspace form.* The displacement splits orthogonally between $B$ and its complement, and the in-subspace map
  attains this bound while the complement's scalar map attains Proposition 3.

**A.5 Proposition 5.** Model a head estimating $\theta^\star$ from $n_r$ real samples and $n_s$ synthetic samples
weighted by $\lambda$. Real samples are unbiased with variance $\sigma^2$; synthetic ones have the same variance and a
label bias $b$. The mean-squared error is $\frac{\sigma^2}{n_r+\lambda n_s} + \big(\frac{\lambda n_s}{n_r+\lambda n_s}\big)^2 b^2$.
* *Change of variable.* With $w = \lambda n_s/(n_r+\lambda n_s)$ it becomes $\frac{\sigma^2(1-w)}{n_r} + w^2 b^2$.
* *Optimum.* This parabola is minimised at $w^\star = \sigma^2/(2 n_r b^2)$, clipped to
  $[0, \frac{n_s}{n_r+n_s}]$. So $\lambda^\star$ is non-increasing in $b$, and $\lambda^\star \to 1$ as $b \to 0$.

## Appendix B. What did not work

These were tested and dropped on the evidence:
* a Bayesian posterior label correction, whose fidelity estimate is circular;
* server-side ensemble distillation, bounded by the synthetic–real total-variation gap;
* a synthetic–real alignment loss, after which the gap grew;
* an ex-post latent prior;
* interleaving or prior-completing synthetic samples in real batches;
* local decay schedules instead of count weighting;
* balanced re-training of the classifier and gradient filtering;
* a hybrid of rows and anchor;
* a server-only generator trained by MMD to the exact embeddings (+15 over GeFL-F at 1/3000 of the communication,
  but below ESA);
* ESA's components on the paper's diffusion generator.

## Appendix C. Extension under test: vicinal consensus transfer

On CIFAR-10 the ten heads *together* reach 69.8% while the average head reaches 60%.

**The extension.** Mix each client's real features with partners from the anchored, consensus-labelled synthetic
pool, $\tilde x = \lambda x + (1-\lambda) x_s$, with target $\lambda e_y + (1-\lambda)\,\bar p(x_s)$.

**Proposition 6.** To second order in $(1-\lambda)\|x_s - x\|$, the MixUp objective depends on its partners only
through the class prior, class means and class covariances [Zhang et al. 2018; 2021].

**Consequence.** The anchored partners therefore give each client the MixUp regulariser of the whole federation's
data, plus ensemble distillation, with no data leaving any client. A numerical check agrees to within 0.001 of the
loss. Whether this beats image-space MixUp (62.67) is PENDING (NB30, NB31). It will enter the method only if it does.

---

## References

[1] H. Kang, S. Cha, J. Kang. "GeFL: Model-Agnostic Federated Learning with Generative Models." IEEE Transactions on
Mobile Computing, 2025 (arXiv:2412.18460).
[3] D. Li, J. Wang. "FedMD: Heterogenous Federated Learning via Model Distillation." NeurIPS FL Workshop, 2019.
[4] T. Lin, L. Kong, S. U. Stich, M. Jaggi. "Ensemble Distillation for Robust Model Fusion in Federated Learning."
NeurIPS, 2020.
[10] Y. Tan et al. "FedProto: Federated Prototype Learning across Heterogeneous Clients." AAAI, 2022.
[11] Z. Zhu, J. Hong, J. Zhou. "Data-Free Knowledge Distillation for Heterogeneous Federated Learning." ICML, 2021.
[12] L. Zhang et al. "Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning."
CVPR, 2022.
[13] J. Zhang et al. "DENSE: Data-Free One-Shot Federated Learning." NeurIPS, 2022.
[14] Y. Wu et al. "FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in
Federated Learning." IJCAI, 2022.
[19] X. Shang, Y. Lu, G. Huang, H. Wang. "Federated Learning on Heterogeneous and Long-Tailed Data via Classifier
Re-Training with Federated Features." IJCAI, 2022.
[20] X. Shang et al. "FEDIC: Federated Learning on Non-IID and Long-Tailed Data via Calibrated Distillation." ICME,
2022.
[21] J. Zhang et al. "Federated Learning with Label Distribution Skew via Logits Calibration." ICML, 2022.
[23] Z. Shen, J. Cervino, H. Hassani, A. Ribeiro. "An Agnostic Approach to Federated Learning with Class Imbalance."
ICLR, 2022.
[24] X. Shuai et al. "BalanceFL: Addressing Class Imbalance in Long-Tail Federated Learning." IPSN, 2022.
[25] Z. Xiao et al. "Fed-GraB: Federated Long-tailed Learning with Self-Adjusting Gradient Balancer." NeurIPS, 2023.
[26] Z. Xiao et al. "FedLoGe: Joint Local and Generic Federated Learning under Long-tailed Data." ICLR, 2024.
[27] X.-C. Li, D.-C. Zhan. "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data."
KDD, 2021.
[31] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, S. Belongie. "Class-Balanced Loss Based on Effective Number of Samples."
CVPR, 2019.
[32] A. K. Menon et al. "Long-tail learning via logit adjustment." ICLR, 2021.
[44] K. Bonawitz et al. "Practical Secure Aggregation for Privacy-Preserving Machine Learning." ACM CCS, 2017.
* M. Bun, T. Steinke. "Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds." TCC, 2016.
* Y. Chen, M. Welling, A. Smola. "Super-Samples from Kernel Herding." UAI, 2010.
* M. Gelbrich. "On a formula for the L2 Wasserstein metric between measures on Euclidean and Hilbert spaces."
  Mathematische Nachrichten, 147:185–203, 1990.
* A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola. "A Kernel Two-Sample Test." JMLR, 13:723–773,
  2012.
* I. Olkin, F. Pukelsheim. "The distance between two random vectors with given dispersion matrices." Linear Algebra
  and its Applications, 48:257–263, 1982.
* M. Saerens, P. Latinne, C. Decaestecker. "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A
  Simple Procedure." Neural Computation, 14(1):21–41, 2002.
* H. Zhang, M. Cisse, Y. N. Dauphin, D. Lopez-Paz. "mixup: Beyond Empirical Risk Minimization." ICLR, 2018.
* L. Zhang, Z. Deng, K. Kawaguchi, A. Ghorbani, J. Zou. "How Does Mixup Help With Robustness and Generalization?"
  ICLR, 2021.
