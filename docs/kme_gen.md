# KME-Gen: a federated generator that is never trained federatedly

*Status: new method, implemented in `experiments/core.py` (`stage_ii_kme`, `KMEGen`, `RFFMap`). Evidence is pending
(E24 locally, then Kaggle). This note gives the motivation, the method and its three guarantees.*

## Why

Every failure of generator-based heterogeneous FL that we measured comes from **training the generator by FedAvg**:

| Failure | Measured | Root cause |
|---|---|---|
| Rare-class conditioning rows collapse | tail/head row norm 0.06–0.33 | flat averaging + Adam's coupled decay on rows held by < K/2 clients |
| Dilution grows with K | CVAE+HWA 87.8 → 83.1 from K = 10 to 100 | each row is fitted from n/K samples per client |
| Location / spread errors of generated classes | MNIST tail fidelity 0.28; spread ratio 0.1–0.6 | an ELBO-trained decoder under federated data |
| Cost and privacy | 100 rounds of generator up- and download per client | no formal privacy for the generator |

Patching these one at a time (HWA, PC, MC) works, but it suggests a cleaner design: **do not optimise the generator
across clients at all.**

## Method

1. **One upload per client, through secure aggregation.** Random Fourier features
   $\phi(h) = \sqrt{2/D}\cos(Wh + b)$ approximate a mixture of Gaussian kernels; their bandwidths come from the exact
   within-class spread. Each client sends, once:
   * the class sums $\sum_{i:y_i=c}\phi(h_i)$, a $C \times D$ matrix;
   * the class sums of $h$, for the prototypes;
   * the class counts.
2. **The server forms the exact pooled class embeddings** $\mu_c = \sum_k S_{kc} / \sum_k n_{kc}$, together with the
   prototypes $m_c$.
3. **The server trains the generator.** It uses the prototype-anchored residual decoder of PC-VAE,
   $G(z,c) = \mathrm{ReLU}(m_c + \mathrm{dec}(z, m_c))$, trained by
   $$\min_G \sum_c \big\| \mathbb E_z\,\phi(G(z,c)) - \mu_c \big\|_2^2 = \sum_c \widehat{\mathrm{MMD}}_k(G_c, P_c)^2 .$$
4. **Heads are trained exactly as before** (LA, consensus labels), on samples from $G$.

## Guarantees

**Theorem 1 (partition invariance).** For any partition of the federation's data, any number of clients $K$ and any
degree of imbalance across clients, $\mu_c$ equals the centralised empirical kernel mean embedding of class $c$. So
the server's objective is the same function of $G$ as in centralised training, and the trained generator depends on
the data only through the pooled class distributions.

*Proof.* $\sum_k S_{kc} = \sum_{i \in D,\, y_i = c}\phi(h_i)$ and $\sum_k n_{kc} = n_c$ for every partition. $\square$

*Consequence.* Row collapse, $m_c/K$ dilution and client drift of the generator **cannot occur**: they are properties
of federated optimisation, which KME-Gen does not do. The remaining dependence on $K$ comes only from the shared
feature extractor's warm-up.

**Theorem 2 (the server minimises the head's risk gap).** Let $\mathcal H$ be the RKHS of the kernel $k$. For a head
whose loss $x \mapsto \ell(f(x), c)$ lies in $\mathcal H$ with norm at most $B$,
$$\big| \mathbb E_{G_c}\ell - \mathbb E_{P_c}\ell \big| \le B\,\mathrm{MMD}_k(G_c, P_c),$$
and therefore $R_{\text{real}} \le R_{\text{syn}} + B \max_c \mathrm{MMD}_k(G_c, P_c)$.

*Proof.* The reproducing property gives $\mathbb E_P \ell = \langle \ell, \mu_P\rangle_{\mathcal H}$; apply
Cauchy–Schwarz. $\square$

The term the server drives down is the same gap that our earlier analysis showed limits all knowledge transfer
(server-side distillation, the synthetic budget, held/unheld classes). A FedAvg-trained CVAE optimises an ELBO, which
controls this gap only indirectly. With $D$ random features, the embedding is uniformly accurate to
$O(1/\sqrt D)$ (Rahimi & Recht).

**Theorem 3 (formal privacy at no extra cost).**
* *Sensitivity.* $\|\phi(h)\|_2 \le \sqrt 2$, and one sample changes one row of the $C \times D$ release. The release
  therefore has $L_2$ sensitivity $\sqrt 2$.
* *Guarantee.* Adding $\mathcal N(0, \sigma^2 I)$ with $\sigma = \sqrt 2\,\sqrt{2\ln(1.25/\delta)}/\varepsilon$ makes it
  $(\varepsilon, \delta)$-DP, with Laplace noise on the counts and standard composition.
* *Everything downstream is post-processing.* Generator training, sampling and head training add no privacy cost, so
  the entire synthetic channel is DP. GeFL-F's generator, trained by FedAvg on raw gradients, has no formal guarantee.
* *Limitation, stated honestly.* The signal of class $c$ grows with $n_c$, but the noise does not shrink. A class with
  about 24 samples cannot be both private and useful at small $\varepsilon$, which holds for any mechanism.

**Communication and compute.** One upload of $C(D + d + 1)$ floats, about 0.28 MB for $D = 6144$ and $d = 768$,
replaces 100 rounds of generator up- and download. Clients run no generator training at all; the server does a
few thousand cheap steps.

## Relation to prior work

* **DP-MERF** (Harder et al., AISTATS 2021) trains generators from random-feature mean embeddings, but centralised.
* **FedHypeVAE, FedDPMS** and DP-CVAE data sharing train generators by federated optimisation.
* **FSG / CCVR / FedCOF** aggregate first and second moments only; KME-Gen matches the whole embedding.

No prior method we found trains a *server-side, class-conditional* generator from exactly aggregated kernel mean
embeddings for model-heterogeneous FL. Taken together, KME-Gen generalises our moment calibration (MC) from two
moments to all of them.

## What would falsify it

KME-Gen is promoted to the final method only if it **matches or beats the best federated-trained generator** (PC-VAE
+ MC, E20) under the long tail and does not lose in IID. The partition-invariance claim also predicts that at
K = 100 it does not degrade the way CVAE + HWA does (87.8 → 83.1).
