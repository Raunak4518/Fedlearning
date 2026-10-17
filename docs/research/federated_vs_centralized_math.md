# Federated vs centralised GeFL-F under long-tailed clients: where accuracy is lost, stage by stage

*Analysis report, 16 October 2026. Read-only with respect to the code and the results: nothing in `experiments/` or
the paper was changed. The numerical checks (§8) are scratch scripts outside the repository; their paths are given
there. Every system-specific statement cites `experiments/core.py` (written `core.py:L`) or a results folder under
`experiments/results/`.*

**Labels.** Every statement is tagged.
- **THEOREM / PROPOSITION**: exact, with a proof here, under the stated model.
- **LEMMA**: auxiliary exact statement.
- **APPROXIMATION**: stated with its error order or the regime where it holds.
- **HEURISTIC**: an argument that is not a proof.
- **CONJECTURE**: a claim we do not prove.
- **MEASURED**: a number from our runs or from a check in §8.

"Toy" means a synthetic numerical model (§8). Nothing below repeats a claim that the adversarial dossier lists as
false: the theorem-1 threshold is not called predictive, "a class held by every client cannot contract" is not used,
and the MC / MC-S / KH optimality statements are treated as statements about idealised maps.

---

## 0. Findings on one page

1. **The partition is pure label shift (Lemma 1.0).** Every client's class-$c$ samples are a uniformly random subset
   of the class-$c$ pool, so $P_k(x\mid y)=P(x\mid y)$ for all clients. Consequences:
   - For **class-conditional quantities** (generator rows, class means, readouts), the data term of every client
     points at the same target. Federation therefore causes weighting, variance and rate errors, never a target bias.
   - Bias comes only from:
     - the optimiser, e.g. coupled decay's shrinkage of rows toward 0;
     - parameters coupled to the **class prior**: logit biases, last layers, and shared trunks trained on
       prior-weighted losses.
2. **Stage I (frozen FE) is first-order consistent with pooled ERM.**
   - Uniform $1/K$ averaging combined with $\tau_k\propto n_k$ local steps gives FedNova surrogate weights within
     0.9–1.8% (total variation) of $n_k/N$ at $K=10$ (Cor. 1.2).
   - The regime is not perturbative: $\eta\tau_k = 1$–$17$.
   - With one head per client, the population objective is $H(Y\mid\phi(X),K)$ instead of $H(Y\mid\phi(X))$
     (Prop. 2.1).
   - **MEASURED (N7, seed 0):** replacing GeFL-F's federated FE by a centralised class-balanced FE changes the
     accuracy of a pooled-trained body with ER readout by **−0.0** on MNIST and **+1.1** on FashionMNIST. Against a
     centralised *ERM* FE the differences are −0.5 / +0.4, so most of FMNIST's 1.1 is the long-tailed weighting of
     the FE loss rather than federation.
     [[N7_C100_FE]]
   - Exact BatchNorm recalibration changes it by ≤ 0.3. The FE has 36 (MNIST/FMNIST), 90 (SVHN) or 390 (CIFAR)
     parameters.
3. **Generator rows.**
   - Under coupled Adam decay with flat averaging, a **count-only** step ratio $\kappa_c$ predicts collapse
     (Approximation 2.5).
   - **MEASURED** over 900 logged rows (36 partitions, 4 datasets, $K=10/50/100$): every row with
     $\kappa_c<1$ collapsed (≤ 0.15 of its initial norm, 352 rows, 0 exceptions), and
     Spearman(log $\kappa_c$, final norm) = 0.94. This is the predictive model the action plan asked for (E-4), at
     least for ranking.
   - Without decay, flat averaging has the right target but a $m_c/K$-diluted rate. In the saturated regime its fixed
     point is the *unweighted* mean of holders' local means, whose MSE is $V_c$ times the pooled MLE's
     ($V_c = 1.2$–$7.8$ for tail classes). HWA restores count weighting (Prop. 2.4, checked to 2 decimals).
4. **Per-client logit corrections are reparametrisations (Prop. 2.7, exact).**
   - LA, FedLC and balanced softmax are each equivalent to plain cross-entropy with a shifted initial bias. FedRS is
     plain cross-entropy with a rescaled learning rate on absent-class rows. None can change a single-client body
     except through transients.
   - **MEASURED (K17):** real-data-only heads with CE, LA, FedLC or FedRS have the same ER accuracy per seed within
     0.1 points (one outlier, 0.48).
   - LA's role in GeFL-F is twofold:
     - it is a readout correction;
     - it aligns the bias optima of the synthetic and real phases, so that the real phase does not erase what the
       synthetic phase taught about absent classes (toy: retention 0.97–0.99 with LA vs 0.49–0.85 without).
5. **Synthetic data whose label is independent of the feature act as an exact logit-variance penalty
   (Prop. 4.3).** The synthetic phase then *hurts* the representation.
   - **MEASURED (K17):** the synthetic-phase gain over real-only heads (same ER readout) changes sign with the
     referee-measured signal fraction $s$.
   - CIFAR-100: GeFL-F at $s=0.002$ gives **−2.2**; decay-fixed GeFL-F at $s=0.010$ gives −1.2; ours at
     $s=0.088$ gives +1.8.
   - CIFAR-10: $s=0.034/0.074/0.23$ give −0.8 / +0.1 / +1.5.
6. **The one-client body is the federated loss that matters, and our generator closes most of it.**
   - **MEASURED (N7, seed 0):** a body trained on its client's real data only loses **3.0** (MNIST) and
     **3.8** (FMNIST) ER points against a body trained on the pooled real data with the same FE.
   - With our generator's synthetic data the federated bodies reach the pooled-body level on MNIST (94.4 vs 94.4)
     and come within 0.8 on FMNIST (81.1 vs 81.9). [[N7_C100_BODY]]
7. **Consequence.**
   - On MNIST and FMNIST, our federated pipeline with ER is within 0–2 points of a *centralised* two-stage pipeline
     (centralised FE, body trained on pooled data, LDA readout) on the same long-tailed data.
   - The long-tail vs balanced gap (3.0 / 3.3 points) is therefore mostly **not** a federated loss: it is present in
     centralised training on the same long-tailed pool.
   - **MEASURED (N7, seed 0):** a pooled-trained body on the balanced (IID) data reaches ER 97.6 / 85.7, against
     94.4 / 83.0 on the long-tailed pool with the best FE. The centralised long-tail cost is 3.2 / 2.7, against our
     pipeline's 2.9 / 3.3 (§2.4).
   - SVHN and CIFAR-10 were not measured.
8. **Top changes that the analysis supports** (§6, §9):
   - **(i)** class-wise gating or relabelling of synthetic data by an exact-statistics estimate of its signal
     fraction, which removes the harm regime of item 5;
   - **(ii)** a class-balanced Stage-I loss with weights from the exact global counts, which is first-order
     equivalent to the centralised balanced FE that gained +1.1 on FMNIST;
   - **(iii)** on the hard datasets only, and only if a measurement shows a pooled-body gap: a cross-client body
     refresh, provably first-order equal to pooled ERM.
   - Interleaving synthetic data into real batches is *not* supported as a gain (toy: equal for faithful generators).

---

## 1. Setup and notation

### 1.1 Data and partition

| Symbol | Meaning |
|---|---|
| $K$, $k$ | clients; $a(k)$ = architecture of client $k$, contiguous blocks (`core.py:279–282`); at $K=10$, $a(k)=k$ |
| $C$, $c$ | classes; $G=10$ architectures |
| $n_{kc}$, $n_k$, $n_c$, $N$ | class-$c$ count on client $k$; client size; class size; total |
| $\pi_{kc}=n_{kc}/n_k$, $\pi_c=n_c/N$ | client prior; pooled prior |
| $\tilde\pi_{kc}=(n_{kc}+1)/(n_k+C)$ | smoothed client prior used by LA (`core.py:780–785`) |
| $H_c=\{k:n_{kc}>0\}$, $m_c=\lvert H_c\rvert$ | holders of class $c$ |
| $B$; $b(n,B)=\lfloor n/B\rfloor+\mathbf 1[n \bmod B\ge 2]$ | batch size (64; 128 on CIFAR); number of batches, since batches of size 1 are skipped (`core.py:946`, `:1181`, `:2039`) |
| $\eta$, $\lambda$, $(\beta_1,\beta_2,\epsilon)$ | learning rate, weight decay, Adam constants $(0.9, 0.999, 10^{-8})$ |
| $f_{kc}$ | probability that a local batch of client $k$ contains class $c$ (hypergeometric) |

Long tail: $n_c=\lfloor n_{\rm tot}\,\mu^c/\sum_j\mu^j\rfloor$ with $\mu=\mathrm{IF}^{1/(C-1)}$ (the code writes $\mathrm{IF}=0.01$, so class 0 is the largest), capped by availability (`core.py:244–251`). Client shares come from
Dirichlet(0.5) per class (`core.py:256–263`). Recorded partition sizes (`n_train`, `min_client`, `tail_holders` in
`runs.jsonl`) are reproduced exactly by our reconstruction for all 36 MNIST / FMNIST / CIFAR-10 / CIFAR-100 settings
checked (N6).

**LEMMA 1.0 (pure label shift).** For every client $k$ and class $c$ with $n_{kc}>0$, the multiset of client $k$'s
class-$c$ images is a uniformly random subset of the class-$c$ training pool. Hence the class-conditional distribution
of a client sample equals that of the pool, $P_k(x\mid y)=P(x\mid y)$. Because $\phi$ is shared, the same holds for
features $h=\phi(x)$.

*Proof.* Each class's indices are randomly permuted (`core.py:248`). The pool is a prefix of that permutation
(`:251`), and the Dirichlet step cuts it into contiguous chunks (`:259–263`). Given the counts, a chunk is a uniformly
random subset. The top-up step (`:268–272`) moves uniformly random samples between clients, which preserves
exchangeability within each class. $\square$

The statement is about exchangeability (sampling without replacement from a finite pool). It does not assume
independence across clients.

### 1.2 The three stages as optimisation problems

Let $\ell$ be the cross-entropy. The pooled empirical risk of a model $\theta$ is
$\mathcal L(\theta)=\sum_k \frac{n_k}{N}\mathcal L_k(\theta)$, with $\mathcal L_k(\theta) = \frac1{n_k}\sum_{i\in D_k}\ell(\theta;x_i,y_i)$.
The **balanced risk**, the population target of the reported metric, is
$\mathcal R_{\rm bal}(\theta)=\frac1C\sum_c\mathbb E_{x\sim P(\cdot\mid c)}\ell(\theta;x,c)$.
Its empirical version reweights each sample by $N/(C\,n_{y_i})$. The reported metric is balanced accuracy, the mean
of per-class recalls on balanced test sets (`core.py:834–848`, `:888–906`).

**Stage I (FE).** Parameters $\phi$ (FE: conv–BN–ReLU–maxpool, 36 / 90 / 390 parameters; `core.py:307–320`) and heads
$h_a$ ($5\times10^3$–$1.1\times10^5$ parameters).
- *As implemented.* Client $k$ starts from $(\phi, h_{a(k)})$ and runs $\tau_k^{\rm I}=5\,b(n_k,B)$ SGD steps
  (lr 0.1, no momentum) on $\mathcal L_k(\phi,h_{a(k)})$ (`core.py:937–954`).
- *Aggregation.* The server sets $\phi\leftarrow\frac1K\sum_k\phi_k$. The average is **uniform over clients**,
  including the BN buffers (`Averager.result`, `core.py:603`; `:955–961`). Heads are averaged within an
  architecture, which is the identity at $K=10$.
- *Rounds.* $T_{FE}=20$ (MNIST, FMNIST) or 50 rounds; then $\phi$ is frozen and features are computed in eval mode
  (`core.py:985–987`).
- *Centralised counterparts.*
  - **(C-ERM)**: $\min_{\phi,\{h_a\}}\sum_a\mathcal L(\phi,h_a)$.
  - **(C-bal)**: the same with class-balanced sampling (`stage_i_oracle`, `core.py:990–1016`).

**Stage II (generator).** CVAE-F, $\psi=(\psi_0, r_1,\dots,r_C)$, where $r_c\in\mathbb R^{512}$ is class $c$'s
conditioning row (`core.py:344–393`).
- *Decoder.* $\mathrm{ReLU}(\mathrm{dec}(\mathrm{ReLU}(A z+a_0+r_y)))$.
- *Encoder.* It receives the class only as a constant scalar channel equal to the class index (`core.py:384`). This is
  an ordinal encoding that cannot distinguish classes well, so class information tends to flow into $z$; that is
  relevant for §4.3.
- *Loss.* $\sum_i\|h_i-\hat h_i\|^2+\mathrm{KL}$ with sum reduction (`core.py:389`). This is the negative ELBO of a
  Gaussian decoder with variance $1/2$, up to constants.
- *Optimiser.* Adam(lr $10^{-3}$) with *coupled* decay $\lambda=10^{-3}$, meaning $g\leftarrow\nabla+\lambda\psi$
  (`core.py:1147–1157`). Each client keeps its own optimiser state across rounds while its parameters are reset to
  the global ones (`core.py:1141`, `:1422`).
- *Local work and aggregation.* $\tau^{\rm II}_k = 5\,b(n_k,64)$ local steps (`core.py:1175–1209`). Aggregation is
  flat, or HWA for the rows: $r_c\leftarrow\sum_kE(n_{kc})r_{k,c}/\sum_kE(n_{kc})$ with $E(n)=(1-0.999^n)/(1-0.999)$
  (`core.py:571–613`, `:1229–1240`, `:1427–1435`).
- *Centralised counterpart.* Minimise $\sum_i[\cdot]$ on the pooled features. `stage_ii_oracle` (`core.py:1444`)
  does it with balanced sampling.

**Stage III (heads).** A head is a body $u_g:h\mapsto e\in\mathbb R^{d_g}$ followed by a last layer
$(W_g,b_g)$ with a free bias (`core.py:323–341`). Per round, client $k$ runs:
- *Synthetic phase.* $S_s=T_s\lfloor n_k/B\rfloor$ SGD steps (lr 0.1) on batches of $B$ generated features with uniform
  labels and target $e_y$, or the CSL target $(1-\beta)e_y+\beta\bar p(\tilde h)$ (`core.py:1937–1981`, `:2012–2033`).
- *Real phase.* $S_r=T_r\,b(n_k,B)$ SGD steps with $T_r=5$, loss $\ell(f(h)+\log\tilde\pi_k,y)$ (LA;
  `core.py:2001–2009`, `:2035–2121`).
- *Aggregation.* Heads are averaged within an architecture.

After $T_{TN}$ rounds come the readouts: ER, the class-balanced LDA from secure-aggregatable sums of every client's
embeddings (`core.py:2347–2366`), plus BBC (`core.py:2221–2238`), CCVR, Fed3R and FedNCM.
- *Centralised counterpart.* A body trained on the pooled real features (ERM or balanced) and a balanced readout. N7
  (§8) trains exactly this.

### 1.3 What FedAvg computes (general facts, in this notation)

**LEMMA 1.1 (one-round expansion).** Let $\mathcal L_k$ be $L$-smooth and $G_k$ bound $\|\nabla\mathcal L_k\|$ on the
local path. Local GD from $x$ with $\tau_k$ steps of size $\eta$ returns
$\Delta_k=-\eta\tau_k\nabla\mathcal L_k(x)+e_k$ with $\|e_k\|\le\eta^2LG_k\tau_k(\tau_k-1)/2$. Hence the aggregate
$x^+=x+\sum_kp_k\Delta_k$ equals $x-\eta\bar\tau\,\nabla\tilde{\mathcal L}(x)+O(\eta^2LG\tau^2)$, with
$\tilde{\mathcal L}=\sum_k\frac{p_k\tau_k}{\bar\tau}\mathcal L_k$ and $\bar\tau=\sum_kp_k\tau_k$.

*Proof.* $y_t-x=-\eta\sum_{s<t}\nabla\mathcal L_k(y_s)$ and
$\|\nabla\mathcal L_k(y_s)-\nabla\mathcal L_k(x)\|\le L\|y_s-x\|\le L\eta sG_k$. Summing over $s<\tau_k$ gives the bound.
$\square$

This is the "objective inconsistency" of Wang et al. (NeurIPS 2020, FedNova), who prove that FedAvg-type methods with
heterogeneous local work converge to stationary points of $\tilde{\mathcal L}$ rather than of $\mathcal L$.

**COROLLARY 1.2 (Stage I is first-order consistent with pooled ERM).** In Stage I, $p_k=1/K$ and
$\tau_k=5\,b(n_k,B)\approx 5n_k/B$, so $p_k\tau_k/\bar\tau=n_k/N+\varepsilon_k$. **MEASURED (N6):** the total
variation $\tfrac12\sum|\varepsilon_k|$ is 0.009–0.018 at $K=10$, 0.057–0.083 at $K=50$ and 0.114–0.122 at $K=100$.
The error comes from rounding up and from the floor of one batch.
- To first order, the FE therefore minimises $\sum_k\frac{n_k}{N}\mathcal L_k(\phi,h_{a(k)})$: the pooled sample
  weighting, but **with one head per client**.
- With equal $\tau_k$ it would instead minimise the unweighted client average.
- The remainder is *not* small: $\eta\tau_k=1.0$–$17$ at $K=10$ (N6: $\tau_k\in[10,170]$ over the four datasets).

**PROPOSITION 1.3 (exact fixed point for quadratic clients).** Let $\mathcal L_k(x)=\frac12(x-x_k)^\top A_k(x-x_k)$,
$A_k\succ0$, $\eta<2/\lambda_{\max}(A_k)$, full-batch local GD, and aggregation weights $p_k$. Then FedAvg converges to
$$x_{\rm FA}=\Big(\sum_kp_kQ_k\Big)^{-1}\sum_kp_kQ_kx_k,\qquad Q_k=I-(I-\eta A_k)^{\tau_k}.$$
- As $\eta\tau_k\|A_k\|\to0$, $Q_k\to\eta\tau_kA_k$ and $x_{\rm FA}\to\arg\min\tilde{\mathcal L}$.
- As $\eta\tau_k\lambda_{\min}(A_k)\to\infty$, $Q_k\to I$ and $x_{\rm FA}\to\sum_kp_kx_k$, the $p$-weighted average of
  the local optima (with $p_k = 1/K$: client-uniform, whatever the $n_k$).

*Proof.* One round maps $x\mapsto\sum_kp_k[(I-Q_k)x+Q_kx_k]$. Its fixed point solves
$\sum_kp_kQ_k(x-x_k)=0$, and the map is a contraction because $\|I-Q_k\|<1$. $\square$

**MEASURED (N1).**
- Iteration matches the formula to $10^{-12}$.
- With $\tau_k\propto n_k$, $\|x_{\rm FA}-x^*\|$ is linear in $\eta$ (0.0023 at $\eta\bar\tau=0.007$; 0.19 at 0.66).
- At the Stage-I value $\eta\bar\tau=5.45$, $x_{\rm FA}$ lies closer to the client-uniform average of local optima
  (distance 0.25) than to the pooled optimum (0.59).
- **Charles & Konečný (AISTATS 2021)** analyse this accuracy–speed trade-off of local-update methods through such
  surrogate objectives, and **Pathak & Wainwright (NeurIPS 2020)** show that FedAvg-type fixed points are in general
  not stationary points of $\mathcal L$. We use only the formula proved above.

**LEMMA 1.4 (exact drift correction).** Take full participation, exact gradients, and control variates
$c_k=\nabla\mathcal L_k(x)$, $c=\sum_kp_kc_k$ fixed during the round, with local direction
$\nabla\mathcal L_k(y)-c_k+c$ (SCAFFOLD-type; Karimireddy et al., ICML 2020). Then:
(i) every stationary point of $\mathcal L=\sum_kp_k\mathcal L_k$ is a fixed point;
(ii) for quadratic clients with $\eta<2/\lambda_{\max}$, the fixed point is unique and equals $\arg\min\mathcal L$ for
every $\tau_k$.

*Proof.* (i) If $\nabla\mathcal L(x)=0$, the local direction at $y=x$ is $0$, so $y$ never moves. (ii) With
$e=y-x$: $e_{t+1}=(I-\eta A_k)e_t-\eta\nabla\mathcal L(x)$, so $e_{\tau}=-A_k^{-1}Q_k\nabla\mathcal L(x)$. The
aggregate update is $-(\sum_kp_kA_k^{-1}Q_k)\nabla\mathcal L(x)$, and the matrix is positive definite. $\square$

**MEASURED (N1):** $|x_{\rm cv}-x^*|<10^{-13}$ at $\eta=0.1$.

**Adam (Stage II).** Local Adam is preconditioned descent with a per-client diagonal preconditioner
$P_k=\mathrm{diag}(\sqrt{\hat v_k}+\epsilon)^{-1}$ that persists across rounds.
- **APPROXIMATION** (treat $\hat v_k$ as constant within a round; error $O(1-\beta_2^{\tau_k})$): the first-order
  aggregate direction is $\sum_kp_k\tau_kP_k\nabla\mathcal L_k$. This is not the gradient of a single objective unless
  the $P_k$ coincide.
- For rows, $P_k$ differs most, because a row's gradient is zero or pure decay on some clients (§2.2).

---

## 2. Centralised vs federated aggregate, stage by stage

### 2.1 The feature extractor (Stage I)

**PROPOSITION 2.1 (what the federated FE optimises).**
- **(a) THEOREM, population, unconstrained heads.** Let $Z=\phi(X)$, $P(K=k)=n_k/N$. Then
$$\min_{\{q_k\}}\sum_k\tfrac{n_k}{N}\,\mathbb E_k[-\log q_k(Y\mid Z)]=H(Y\mid Z,K)=H(Y\mid Z)-I(Y;K\mid Z),$$
  while one shared head achieves $H(Y\mid Z)$. So Stage I's first-order objective (Cor. 1.2, one head per client)
  rewards the FE for $I(Y;Z\mid K)$, the label information *within* clients, not $I(Y;Z)$. Label information that the
  client identity already provides is free.
- **(b) APPROXIMATION, low separation.** Assume Gaussian class-conditionals of $Z$ with a shared covariance $\Sigma$ and
  class means $\mu_c$ (pure label shift, Lemma 1.0). Let
  $D^2_{cc'}=(\mu_c-\mu_{c'})^\top\Sigma^{-1}(\mu_c-\mu_{c'})$. To second order in the separations (error
  $O(\max D^4)$),
  $$I_k(Y;Z)=\tfrac14\sum_{c,c'}\pi_{kc}\pi_{kc'}D^2_{cc'},$$
  so the three objectives differ only in the weight each class pair gets:
  - federated: $w^{\rm fed}_{cc'}=\sum_k\frac{n_k}{N}\pi_{kc}\pi_{kc'}$;
  - centralised ERM: $w^{\rm cen}_{cc'}=\pi_c\pi_{c'}$;
  - balanced: $1/C^2$.

*Proof of (a).* Gibbs' inequality: $\mathbb E[-\log q(Y\mid Z)]\ge H(Y\mid Z)$ with equality at the true posterior.
Apply it per client and use the chain rule.

*Proof of (b).* $I_k(Y;Z)=\sum_c\pi_{kc}\mathrm{KL}(P_c\|P_k)$ with $P_k=\sum_c\pi_{kc}P_c$. For Gaussians with a
common $\Sigma$ and small mean differences, $\mathrm{KL}(P_c\|P_k)=\frac12\|\mu_c-\bar\mu_k\|^2_{\Sigma^{-1}}+O(D^4)$.
Then $\sum_c\pi_{kc}\|\mu_c-\bar\mu_k\|^2=\frac12\sum_{c,c'}\pi_{kc}\pi_{kc'}\|\mu_c-\mu_{c'}\|^2$. $\square$

**MEASURED (N6b): the federated weighting is about as long-tailed as centralised ERM.**
- Over unordered pairs, the standard deviation of $\log_{10}w$ is 0.71–0.87 (federated) vs 0.77–0.85 (centralised).
- Their mean distance from the balanced weight is 0.82–1.00 vs 0.80–0.96 decades.
- The federated weighting puts 1.1–3.3× more weight on tail–tail pairs relative to head–head pairs than
  centralised ERM does. The ratio $\rho_{cc'}=w^{\rm fed}_{cc'}/w^{\rm cen}_{cc'}$ has mean ≈ 1 and minimum 0.02–0.6
  (N6).
- **PREDICTION:** to first order the federated FE is neither better nor worse than a centralised ERM FE for balanced
  accuracy. Both are worse than a centralised balanced FE by an amount that depends on how much the bottleneck forces
  trade-offs between pairs.

In the strongly-local regime of Prop. 1.3, FedAvg moves toward client-uniform weighting of local solutions.
**MEASURED:** the client-uniform prior $\frac1K\sum_k\pi_k$ is less imbalanced than the pooled prior on MNIST / FMNIST
(max/min 39–63 vs 100), about equal on CIFAR-10 (40–52 vs 48), and not on CIFAR-100 (106–110 vs 96).
- **HEURISTIC:** this is a mild implicit rebalancing of the FE on the 10-class datasets.

**MEASURED (toy N2: shared 4- or 10-dimensional bottleneck, personalised heads, actual MNIST seed-0 client
composition, 5 seeds).** Balanced LDA accuracy on the frozen bottleneck:

| FE trained by | $p=4$ | $p=10$ |
|---|---|---|
| federated, one head per client (GeFL-F Stage I) | 79.5 | 96.3 |
| federated + SCAFFOLD on the FE | 79.6 | 97.3 |
| federated + an auxiliary shared classifier | 73.6 | 97.6 |
| centralised, instance sampling (C-ERM) | 56.8 | 79.9 |
| centralised, class-balanced (C-bal) | 78.0 | 97.7 |

- The ratio of federated to centralised pair separation tracks $\rho_{cc'}$ (Spearman 0.54 and 0.51, $p<10^{-6}$).
- The large advantage of the federated FE over C-ERM is **not** explained by the first-order weights above; we
  attribute it, as a CONJECTURE, to the non-perturbative regime (local optimisation to near-optimal per-client heads).

**MEASURED (N7, real data, the repository's own Stage I, seed 0).**
- Setup: FE variants trained on the same split; every variant then scored with (a) a balanced LDA on the raw FE
  features and (b) each architecture's body trained on the pooled real features (instance sampling, same number of
  SGD steps as Stage III), read out by ER.
- Readout: balanced test accuracy, mean over the 10 architectures.

| FE | MNIST raw LDA | MNIST pooled body + ER | FMNIST raw LDA | FMNIST pooled body + ER |
|---|---|---|---|---|
| GeFL-F federated (`stage_i`) | 88.6 | **94.4** | 78.8 | **81.9** |
| federated, exact pooled BN statistics | 88.6 | 94.6 | 78.8 | 81.8 |
| centralised, class-balanced (`stage_i_oracle`) | 90.4 | 94.4 | 79.9 | **83.0** |
| centralised, instance sampling | 89.5 | 93.9 | 79.4 | 82.3 |

- Against a centralised *balanced* FE, the FE-attributable loss is ≈ 0 on MNIST and ≈ 1.1 on FMNIST.
- Against a centralised *ERM* FE it is −0.5 and +0.4, so most of FMNIST's 1.1 is the long-tailed weighting of the FE
  loss (as predicted by Prop. 2.1(b)), not federation.
- Both come from one seed; the seed-to-seed s.d. of comparable quantities is 0.3–1.
- [[N7_C100_FE_TABLE]]

**PROPOSITION 2.2 (BatchNorm buffers).**
- Each client's running mean and variance are EMAs (momentum 0.1) of batch moments computed under its **own drifted
  weights** $\phi_k^{(t)}$. The server averages them uniformly.
- Even without drift, with exact client moments $(\mu_k,\sigma_k^2)$ of a common $\phi$, the averaged buffers differ
  from the pooled ones:
  - $\bar\mu-\mu_{\rm pool}=\sum_k(\frac1K-\frac{n_k}N)\mu_k$;
  - $\sigma^2_{\rm pool}-\bar\sigma^2=\sum_k(\frac{n_k}N-\frac1K)\sigma_k^2+\sum_k\frac{n_k}N\|\mu_k-\mu_{\rm pool}\|^2$.
  The last term is the between-client variance, which is ≥ 0 (the identity is the law of total variance).
- With drift, the averaged buffers describe no single model.
- **MEASURED (N7):**
  - averaged-to-pooled variance ratio per channel: 0.99–1.06 (MNIST) and 0.98–1.38 (FMNIST);
  - the between-client term can only push the ratio below 1, so the ratio of 1.38 on FMNIST comes from drift or from
    the uniform client weighting;
  - replacing the buffers by the exact pooled statistics changes the downstream accuracies by ≤ 0.3
    (table above).
- **Conclusion:** a real but inconsequential discrepancy here. The heads are retrained on the eval-mode features, so a
  per-channel affine change matters only through the ReLU's active set.

**Drift.** $\eta\tau_k$ is large (1–17), so Lemma 1.1 is directional only. Lemma 1.4 removes the drift exactly, but
in the toy it changes the FE by ≤ 1 point (table above). We know of no measurement suggesting that FE drift costs
accuracy here.

**Summary for Stage I.**
- To first order, Stage I optimises pooled ERM with one head per client.
- Its population objective ignores label information carried by the client identity.
- It is about as long-tailed as centralised ERM.
- **MEASURED:** a centralised balanced FE helps by 0 (MNIST) and 1.1 (FMNIST) points.

### 2.2 The generator (Stage II)

The row $r_c$ enters only through samples labelled $c$ (`core.py:380`). Its data gradient is
$g^{\rm data}_{kc}=\sum_{i\in{\rm batch},y_i=c}J_i^\top\nabla_{\hat h}\ell_i$. It is zero on every step of a
non-holder and on every step of a holder whose batch lacks $c$.

**LEMMA 2.3 (non-holder updates under each optimiser setting; exact).** The table gives one local round of a
non-holder $k\notin H_c$ with persistent Adam state.

| setting | non-holder's gradient on $r_c$ | non-holder's update to $r_c$ per step |
|---|---|---|
| coupled Adam, $\lambda>0$ (GeFL-F) | $\lambda r_c$ on every step | $-\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ with $\hat m,\hat v$ built from $\lambda r_c$ only; for a constant-sign coordinate this tends to $-\eta\,\mathrm{sign}(r_{cj})\cdot\frac{\lambda\lvert r_{cj}\rvert}{\lambda\lvert r_{cj}\rvert+\epsilon}$ |
| no decay (`NOWD`), no decay on rows (`ROWNOWD`) | $0$ | exactly $0$: $\hat m=\hat v=0$ and the update is $0/(0+\epsilon)$ |
| AdamW ($\lambda$ decoupled) | $0$ | multiplication by $1-\eta\lambda$: $1-10^{-6}$ (`ADAMW3`, $\lambda=10^{-3}$) or $1-10^{-5}$ (`ADAMW`, $\lambda=10^{-2}$) |
| lazy decay (`LCD`) | $0$ | rows of classes absent from the batch are restored after the step (`core.py:1196–1207`): $0$ |

Under flat averaging, therefore,
$$r_c^+-r_c=\frac1K\sum_{k\in H_c}\Delta_{kc}+\frac1K\sum_{k\notin H_c}\Delta^{\rm decay}_{kc},$$
with $\Delta^{\rm decay}=0$ for no decay, row-only no decay and lazy decay. For AdamW it is
$-(1-(1-\eta\lambda)^{\tau_k})r_c$, at most $1.6\times10^{-4}r_c$ ($\lambda=10^{-3}$) or $1.6\times10^{-3}r_c$
($\lambda=10^{-2}$) per round. For coupled Adam it is
$\approx-\eta\tau_k\,\mathrm{sign}(r_c)$ per coordinate while $\lambda|r_{cj}|\gg\epsilon$. This is the
λ-invariance of the paper's Lemma 2. HWA deletes the second sum.

- **Centralised Adam with the same coupled decay does not collapse rare rows.** There, $\hat v$ for $r_c$ is
  dominated by data gradients from the batches that contain $c$, so the decay steps are $\lambda r/\sqrt{\hat v}\ll\eta$.
  The collapse needs three things together, all federated:
  - a client whose gradient on $r_c$ is pure decay;
  - Adam's normalisation of that pure-decay gradient;
  - averaging that client's row back in.
- Counting the decay as an L2 penalty in the pooled objective, its weight is $\approx\frac{N}{64}\frac\lambda2\|r_c\|^2$
  against $n_c$ data terms. That is negligible for every class.

**PROPOSITION 2.4 (row fixed points without decay; quadratic row model).**
- *Model (APPROXIMATION).* Linearise the decoder at the current trunk, so that the class-$c$ loss on client $k$ is
  $\frac12\sum_{i\in D_{kc}}\|r-t_i\|_A^2$ with a fixed $A\succ0$ and targets $t_i$. By Lemma 1.0, the targets are
  exchangeable across clients with mean $t_c$ and covariance $\Sigma_t$.
- A local round maps $r\mapsto M_{kc}r+(I-M_{kc})\bar t_{kc}$, where $M_{kc}=\prod_s(I-\eta b_sA)$ over the steps
  and $b_s$ is the number of class-$c$ samples in step $s$'s batch. Take $\bar t_{kc}$ = the local mean (exact for
  full-batch steps; in expectation otherwise).

Then:
1. **Flat FedAvg** converges to $r^*=(\sum_{H_c}(I-M_{kc}))^{-1}\sum_{H_c}(I-M_{kc})\bar t_{kc}$. Non-holders do not
   enter.
   - *Small-step regime:* $I-M_{kc}\approx 5\eta A\,n_{kc}$, so $r^*\to\sum_kn_{kc}\bar t_{kc}/n_c$, the pooled MLE.
   - *Saturated regime:* $M_{kc}\to0$, so $r^*\to\frac1{m_c}\sum_{H_c}\bar t_{kc}$, the unweighted mean of holders.
   - The rate per round is $\frac1K\sum_{H_c}(I-M_{kc})$, i.e. diluted by $m_c/K$ relative to the holder average.
2. In the saturated regime, the MSE of the flat fixed point relative to the pooled MLE is
   $$V_c=\frac{n_c}{m_c^2}\sum_{k\in H_c}\frac1{n_{kc}}\;\ge1,$$
   with equality iff all holders have equal counts (Cauchy–Schwarz).
3. **HWA** with weights $\omega_k=E(n_{kc})$ converges to $(\sum\omega_k(I-M_{kc}))^{-1}\sum\omega_k(I-M_{kc})\bar t_{kc}$.
   - *Saturated:* $E(n)$-weighting, with $V^{\rm HWA}_c\approx1$ because $E(n)\approx n$ for $n\ll10^3$.
   - *Small-step:* weights $\approx n_{kc}^2$, giving $V^{\rm HWA}_c=n_c\sum n_{kc}^3/(\sum n_{kc}^2)^2\ge1$.
   - So neither flat nor HWA is right in both regimes.
4. The **per-class normalised rule** $r^+=r+\sum_{H_c}\frac{n_{kc}}{n_c}(I-M_{kc})^{-1}(r_{k}^{+}-r)$ reaches
   $\sum_k\frac{n_{kc}}{n_c}\bar t_{kc}$ (the pooled MLE) after one round, in every regime.
   - In practice, $(I-M_{kc})^{-1}$ is approximated by dividing by the number of class-$c$ steps. This is FedNova's
     normalisation applied per class, accurate to first order in $\eta$.

*Proof.* Item 1 is Prop. 1.3 restricted to the row; non-holders contribute the identity map, so they drop out of the
fixed-point equation. For item 2, the local means are independent and unbiased with covariance $\Sigma_t/n_{kc}$
(Lemma 1.0), so $\mathrm{MSE}=\mathrm{tr}\Sigma_t\cdot m_c^{-2}\sum1/n_{kc}$, against $\mathrm{tr}\Sigma_t/n_c$.
Item 3 follows like item 1. Item 4 by substitution. $\square$

**MEASURED (N3b):** simulated MSE ratios on the actual MNIST seed-0 counts match $V_c$ to 2 decimals.
- MNIST seed 0: $V_c$ = 5.7, 5.4, 7.8, 9.2, 6.1 (head and medium classes) and 2.6, 2.0, 2.3, 2.8, 2.2 (tail).
- Across settings (N6): tail mean 1.2–7.8, head mean 3.9–26.
- Head classes are hit hardest because Dirichlet splits create holders with 1–2 samples of a head class.

**APPROXIMATION 2.5 (count-only collapse threshold under coupled decay and flat averaging).**
- *Model.* Per coordinate $j$, an Adam step has size $\eta\psi$ with $\psi=\hat m/\sqrt{\hat v}\in[-1,1]$.
  - Non-holders see a constant-sign decay gradient, so $\psi=\mathrm{sign}(r_{cj})$ on each of their $\tau_k$ steps.
  - Holders move only on their $\tau_kf_{kc}$ class-$c$ steps, toward the target, with $\psi=\psi_h(\delta)$, where
    $\delta=t_j-r_{cj}$ and $\psi_h$ is odd, increasing and $|\psi_h|\le1$.
- *Drift per round* (flat averaging):
  $$\Delta r_{cj}=\tfrac{\eta}{K}\Big[\psi_h(\delta)\sum_{k\in H_c}\tau_kf_{kc}-\mathrm{sign}(r_{cj})\sum_{k\notin H_c}\tau_k\Big].$$
  Define the step ratio
  $$\kappa_c=\frac{\sum_{k\in H_c}\tau_kf_{kc}}{\sum_{k\notin H_c}\tau_k}.$$
- *Consequences.*
  - (i) If $\kappa_c<1$, the drift points toward 0 for every coordinate and every target, so the row collapses
    (up to an oscillation of size $O(\eta\tau/K)$).
  - (ii) If $\kappa_c>1$, the row equilibrates where $\psi_h(\delta^*)=1/\kappa_c$. For the Gaussian row model,
    $\psi_h(\delta)=\delta/\sqrt{\delta^2+\sigma_j^2/\bar b}$ ($\bar b$ = class-$c$ samples per batch that contains
    $c$), which gives $|\delta^*|=\sigma_j/\sqrt{\bar b(\kappa_c^2-1)}$. The equilibrium shrink falls with $\kappa_c$
    and with the holders' signal-to-noise ratio.
- *Error terms.* $O(\eta\,(1-\beta_1)^{-1})$ momentum carry-over between class-$c$ steps, which raises the holder
  term by up to $1/\sqrt{f_{kc}}$ (sparse-gradient Adam). Also $O(1-\beta_2^{\tau})$ non-stationarity of $\hat v$.
  Note that Adam's per-step update can exceed $\eta$ by up to $(1-\beta_1)/\sqrt{1-\beta_2}\approx3.2$ in the sparse
  case (Kingma & Ba, 2015, §2.1).
- *Status.* Unlike Theorem 1's $\theta_c$, $\kappa_c$ uses only counts and step numbers, which are known before
  training.

**MEASURED.**
- T01 (MNIST seed 0, the real CVAE): $\kappa$ = 8.0, 7.3, 0.84, 0.31, 0.48 for classes 5–9. The final row norms are
  0.75, 0.65, 0.10, 0.04, 0.09 of the initial norm: collapse exactly where $\kappa<1$.
- Toy N3 on the same counts (rows-only Gaussian model, the exact local Adam loop): classes 7–9 end at 0.27 / 0.08 /
  0.24 of their target norm and classes 5–6 at 0.84 / 0.76. With AdamW or no decay, all classes reach 0.95–1.15.
  With HWA (decay kept), 0.98–1.04.
- Out of sample (N6c): over **every** logged GeFL-F run with coupled decay and flat averaging (MNIST and FMNIST at
  $K=10/50/100$, CIFAR-10, CIFAR-100; 900 class rows):
  - every row with $\kappa_c<1$ ended at ≤ 0.153 of its initial norm (352 rows, 0 exceptions);
  - Spearman(log $\kappa_c$, final norm) = 0.94;
  - median final/initial norm by $\kappa$ bin:

    | $\kappa_c$ | <0.5 | 0.5–1 | 1–2 | 2–4 | 4–8 | ≥8 | ∞ (no non-holder) |
    |---|---|---|---|---|---|---|---|
    | median final/initial norm | 0.04 | 0.08 | 0.12 | 0.22 | 0.34 | 0.81 | 2.25 |

  - So $\kappa_c\ge1$ is not sufficient to survive. That matches (ii) with a low holder SNR ($\psi_h(\delta^*)=1/\kappa_c$
    is reached at a large shrink).
- This answers action-plan item E-4 for ranking and as a sufficient condition. A calibrated equilibrium curve would
  need $\sigma_j$ and $\bar b$ per class.

**Rates without decay.** The flat per-round rate of row $c$ is $\frac1K\sum_k\tau_kf_{kc}$ (in units of $\eta$),
against $\sum_k\omega_k\tau_kf_{kc}/\sum\omega_k$ for HWA.
- **MEASURED (N6):** normalised to class 0 as in T02, the HWA/flat rate ratio of MNIST's rarest class (seed 0) is
  1.15 / 2.28 / 3.14 at $K=10/50/100$. T02's measured row movements give 1.05 / 2.24 / about 2–3×
  (`experiments/results/T02_dilution_rates/summary.md`). Over all K16-type partitions, the ratio for the four rarest
  classes is 0.7–1.8 at $K=10$ (MNIST / FMNIST), 0.8–2.4 at $K=50$ and 1.0–3.1 at $K=100$.
- This explains why HWA's advantage over the decay fixes appears only at $K=50/100$ (K15: +1.3 to +1.8 on MNIST).
  It also explains why tail fidelity under the decay fixes falls with $K$ (MNIST 0.25 → 0.16 → 0.15; K15).

**The shared trunk.**
- The trunk's per-class loss weight is $\approx n_c/N$ in both worlds: Lemma 1.1, up to Adam's client-wise
  preconditioning.
- The rarest class carries 0.4% of the loss (T02).
- The trunk's long-tail problem is therefore **not** federated. A class-balanced ELBO is the centralised fix and
  needs only the exact counts (§6).

**PC-VAE.**
- PC-VAE replaces the rows by the exact class means $m_c$, which are partition-invariant (Thm. 3.1). That removes
  every term of this subsection except the trunk's.
- Its limits are those of the anchor:
  - Prop. 5 of the paper (separation bounded by the mean distances);
  - the statistical floor of §3.3.

### 2.3 The heads (Stage III)

At $K=10$, head $g$ is client $g$'s model: averaging is the identity. What a federated body learns differs from a
centralised body in four ways:
- **(a) sample size:** $n_k\approx N/10$ real samples;
- **(b) coverage:** absent classes get no real gradient;
- **(c) prior:** the client's label prior;
- **(d) synthetic data:** generator bias and variance.

**PROPOSITION 2.7 (per-client logit corrections are reparametrisations; exact).** Let a head have a free last-layer
bias $b$, trained by SGD without weight decay on $b$ (true in Stage III: `core.py:2036`). Let $a_k\in\mathbb R^C$ be
any client constant: LA $a_k=\tau\log\tilde\pi_k$; FedLC $a_k=-\tau(n_{kc}+1)^{-1/4}$ (`core.py:763–777`); balanced
softmax.
1. Training with loss $\ell(f+a_k,y)$ from $(\theta_0,b_0)$ produces exactly the trajectory of training with
   $\ell(f,y)$ from $(\theta_0,b_0+a_k)$, with $b^{\rm LA}_t=b^{\rm CE}_t-a_k$ and every other parameter identical
   at every step.
2. FedRS (`core.py:776`: logits of absent classes multiplied by $\alpha$) is exactly plain CE in the variables
   $\tilde w_c=\alpha w_c,\ \tilde b_c=\alpha b_c$ for absent $c$, with the learning rate on those rows multiplied by
   $\alpha^2$. Every body parameter follows the same trajectory.

*Proof.*
1. $f+a_k=Wu(h)+(b+a_k)$. The map $b\mapsto b+a_k$ is a translation, so it has identity Jacobian. Gradients with
   respect to all parameters coincide at corresponding points; induct over steps.
2. $\alpha f_c=\tilde w_c^\top u+\tilde b_c$. The gradient with respect to $\tilde w_c$ is $1/\alpha$ times the gradient
   with respect to $w_c$, so an SGD step of size $\eta$ on $w_c$ is a step of $\eta\alpha^2$ on $\tilde w_c$. The body
   sees the same logits. $\square$

**COROLLARY.**
- At $K=10$, without a synthetic phase, LA / FedLC / FedRS / balanced softmax can change the body only through the
  different initial bias (or row learning rate). They are readout-level corrections.
- **MEASURED.** N4: the trajectory identity holds to $4\times10^{-16}$ (body) and $7\times10^{-15}$ (bias); the FedRS
  identity holds exactly.
- **MEASURED.** K17, five datasets, seeds 3–5: real-data-only heads under CE, LA, FedLC and FedRS have per-seed ER
  accuracies within 0.11 points of each other (one outlier, 0.48 on CIFAR-100). Their mean ER is 89.96–89.97 on MNIST,
  77.73–77.78 on FMNIST and 54.29–54.34 on SVHN.
- **Limits of the identity.**
  - With $K>10$, clients with different $a_k$ are averaged. Then LA becomes "translate to the balanced frame before
    averaging", which matters.
  - With a synthetic phase that uses unadjusted logits, the two phases see different biases (N4: body difference 0.16
    instead of $10^{-16}$).

**PROPOSITION 2.8 (alternation; exact for quadratic losses).**
- Let the synthetic and real losses be quadratic with Hessians $H_s,H_r\succeq0$ and minimisers $\theta_s,\theta_r$
  (any minimiser if singular). Let $P_s=(I-\eta H_s)^{S_s}$ and $P_r=(I-\eta H_r)^{S_r}$, with $\eta<1/\lambda_{\max}$,
  so that every $\rho\in[0,1]$.
- If $\|P_rP_s\|<1$, the round map $\theta\mapsto\theta_r+P_r(\theta_s+P_s(\theta-\theta_s)-\theta_r)$ converges to
  $$\theta^\star=\theta_r+\Omega(\theta_s-\theta_r),\qquad\Omega=(I-P_rP_s)^{-1}P_r(I-P_s).$$
- When $H_r$ and $H_s$ commute, the synthetic weight in the eigendirection $i$ is
  $$\omega_i=\frac{\rho_{r,i}(1-\rho_{s,i})}{1-\rho_{r,i}\rho_{s,i}}\le\rho_{r,i}=(1-\eta h_{r,i})^{S_r}.$$
  - Along directions the client's real loss does not constrain ($h_{r,i}=0$), $\omega_i=1$: the synthetic data decide.
  - Along well-constrained directions ($\eta h_{r,i}S_r\gg1$), $\omega_i\to0$: the real phase forgets.
  - More synthetic epochs (larger $S_s$, so $\rho_s\to0$) cannot raise $\omega_i$ above $\rho_{r,i}$.
- Fine interleaving ($S_s,S_r\to$ small with a fixed ratio) gives $\Omega\to(S_rH_r+S_sH_s)^{-1}S_sH_s$: the
  minimiser of the joint objective $S_r\mathcal L_r+S_s\mathcal L_s$, up to $O(\eta)$.

*Proof.* Solve the fixed-point equation as in Prop. 1.3. The scalar form follows in a common eigenbasis. For
interleaving, expand $P\approx I-\eta SH$. $\square$

**MEASURED (N1):** iteration equals the formula to $10^{-15}$; unconstrained directions get weight 1.000.

**Refinement (APPROXIMATION, second order around the post-synthetic point).**
- What decides forgetting is the real loss's curvature **along the synthetic displacement** $\delta$. The retained
  fraction is $\delta^\top P_r\delta/\|\delta\|^2$.
- For cross-entropy the Hessian is $\sum_iJ_i^\top(\mathrm{diag}(q_i)-q_iq_i^\top)J_i$, which is small whenever the
  client's real data are already fit with confidence, $q_{i,y_i}\to1$.
- Under LA, the absent-class directions have $q^{\rm LA}_{ic}\propto\tilde\pi_{kc}e^{f_c}=O(1/(n_k+C))$: almost no
  gradient and almost no curvature.
- Without LA, right after a synthetic phase that balanced the logits, $q_{ic}=O(1/C)$. The real phase then applies a
  non-quadratic drift $-\eta\sum_iq_{ic}u_i$ to $w_c$, which has no finite minimiser: the logit of a never-seen class
  decreases without bound, as for logistic regression on separable data (Soudry et al., JMLR 2018).
- **MEASURED (toy N5b, counterfactual retention of the synthetic phase's change through the real phase, $T_s=10$,
  3 seeds):**
  - absent-class rows: 0.97 / 0.97 / 0.99 with LA, 0.57 / 0.49 / 0.85 without;
  - held-class rows: 0.70–0.85 with LA.
  - So in the confidently-fit regime the real phase forgets little even where it has data, and the "cap" $\rho_r$ is
    close to 1 in most directions.
  - Consistent with this, alternation and a joint objective give the same ER in the toy for a faithful generator
    (97.3 vs 97.2; §4.4).

**APPROXIMATION 2.9 (sample-size gap in real-supported directions; HEURISTIC for networks).**
- For a regular M-estimator from $n$ samples, the excess risk is $\mathrm{tr}(H^{-1}\Sigma_g)/(2n)+o(1/n)$
  (van der Vaart, 1998, Ch. 5). For a well-specified log-loss this is $p/(2n)$.
- So a body fitted to $n_k=N/K$ real samples has about $K$ times the centralised excess risk in the directions both
  constrain.
- Over-parametrised networks are not regular M-estimators, so we use this only as a scale argument and measure the
  real gap directly. **MEASURED (N7, seed 0):**

| bodies (federated FE, ER readout) | MNIST | FMNIST |
|---|---|---|
| one client's real data only (K17 "Real only"; here from random init / from the Stage-I head) | 91.5 / 90.8 | 78.1 / 78.5 |
| our method, K16 seed 0 (same split, a different realisation of the FE) | 94.4 | 81.1 |
| pooled real data, instance sampling | **94.4** | **81.9** |
| pooled real data, class-balanced sampling | 94.2 | 81.5 |

- The one-client body loses 3.0 / 3.8 points against the pooled body. Our synthetic data recover all of it on MNIST
  and 80% on FMNIST (one seed each).
- On the other datasets K17 gives the synthetic gain over real-only bodies, ours vs real-only with the same ER:
  +4.7 (MNIST), +3.4 (FMNIST), +11.1 (SVHN), +1.6 (CIFAR-10), +1.8 (CIFAR-100). The pooled-body ceiling there is
  unmeasured except [[N7_C100_BODY_SHORT]].

**Per-class exposure.**
- Per round, class $c$ contributes about $T_rn_k\pi_{kc}$ real and $T_s\lfloor n_k/B\rfloor B/C$ synthetic samples
  to head $a(k)$, against $\propto\pi_c$ for a pooled body.
- The synthetic phase therefore sets a floor of $T_s/(CT_r)$ relative exposure per class. With $T_s=10$ and $C=10$,
  synthetic data dominate every class with $\pi_{kc}<0.2$.

**ERL (E33): CONJECTURE.**
- Training the body under a fixed, rescaled LDA readout lowered the pooled-refit accuracy of the bodies (MNIST 89.2 vs
  94.9) and raised ECE.
- Our conjecture: with a fixed readout of small norm, the local CE can only grow its margins by growing the embedding
  along the readout rows, which inflates within-class spread and confidence. The readout is also re-centred every
  round, so the body chases a moving target.
- A fixed readout is not harmful as such: FedBABU (Oh et al., ICLR 2022) trains bodies under a fixed head.
- Measurement M10 (§7) would test this.

### 2.4 Putting the stages together: a decomposition against centralised training (MNIST, FMNIST)

- *Centralised comparator.* GeFL-F's two-stage structure trained centrally: an FE (federated, or centralised and
  class-balanced), then each architecture's body trained on the **pooled** real features, then the same LDA readout.
  It is not end-to-end joint training; it is the comparator that matches GeFL-F's frozen-FE design.
- *Sources.* Rows marked K16 are seed 0 of the Kaggle runs; the others are N7, seed 0, CPU. Balanced accuracy, mean
  over the 10 architectures.

| | MNIST | FMNIST |
|---|---|---|
| Ours + ER, balanced (IID) data (K16) | 97.4 | 84.4 |
| Ours + ER, long tail (K16) | 94.4 | 81.1 |
| **drop of our pipeline** | **2.9** | **3.3** |
| centralised: pooled body, balanced (IID) data | 97.6 | 85.7 |
| centralised: pooled body, long tail, federated FE | 94.4 | 81.9 |
| centralised: pooled body, long tail, centralised class-balanced FE | 94.4 | 83.0 |
| **centralised long-tail cost** (IID minus best long-tail) | **3.2** | **2.7** |
| federated residual, long tail (best centralised − ours) | 0.0 | 1.9 (≈ 1.1 FE + 0.8 body) |
| of the FE part: federated FE vs centralised ERM FE / centralised ERM FE vs centralised balanced FE | −0.5 / +0.5 | +0.4 / +0.7 |
| federated residual, IID (centralised − ours) | 0.3 | 1.3 |

- **Reading.** The long-tail cost of our federated pipeline (2.9 / 3.3) is about the long-tail cost of centralised
  training (3.2 / 2.7).
- The federated residual is
  - ≈ 0 on MNIST;
  - ≈ 1.9 on FMNIST, of which ≈ 1.3 is present on balanced data too (one-client bodies) and ≈ 0.6 is specific to the
    long tail.
- One seed, so differences below ~1 point are within noise (seed-1 check: [[N7_S1]]).
- SVHN and CIFAR-10 are unmeasured. CIFAR-100: [[N7_C100_DECOMP]].

---

## 3. The maximum-likelihood / probabilistic view

**THEOREM 3.1 (partition invariance of sufficient statistics).**
- Let an estimator be a function $\Psi(S)$ of $S=\sum_k\sum_{i\in D_k}T(x_i,y_i)$, with $T$ fixed in advance. Its value
  is the same for every partition of the pooled data into clients, and equals the centralised estimator computed on
  the pool.
- In particular, every exponential-family maximum-likelihood estimate, which solves
  $\nabla A(\hat\theta)=\frac1N\sum_iT(x_i,y_i)$, is computed exactly by one round of secure aggregation.

*Proof.* Addition is associative and commutative, so $S$ does not depend on the partition. $\square$

**Which parts of the pipeline are partition-invariant** (the column "as implemented" says whether the federated value
equals the centralised one *for the same frozen networks*):

| component | statistic | partition-invariant? |
|---|---|---|
| ER readout (LDA, `core.py:2347`) | $n_c$, $\sum e$, $\sum ee^\top$ per architecture | **yes** (equals centralised LDA on the same bodies) |
| Fed3R, FedNCM, CCVR (all clients) | same sums | **yes**; CCVR's virtual-feature SGD adds Monte-Carlo noise |
| FSG generator | $n_c$, $\sum h$, $\sum hh^\top$ | **yes** (Gaussian MLE plus shrinkage) |
| PC-VAE anchor $m_c$; MC targets $(m_c,V_c)$; MC-S targets | first and second moments | **yes** for the targets; the maps are applied to a FedAvg-trained generator, so the output is **not** invariant |
| KH target $\mu_c$; KME-Gen objective | RFF sums | **yes** (the selection uses generator samples) |
| gate statistic $F$ (`core.py:2638`) | same sums | **yes** |
| exact BN recalibration (§6) | per-channel $\sum u,\sum u^2$ | **yes** |
| HWA rows | weights depend on $n_{kc}$ | no; ≈ pooled MLE in the saturated row model (Prop. 2.4) |
| FE, CVAE trunk, PC-VAE decoder, head bodies | FedAvg / local SGD | **no** |
| LA, FedLC, FedRS | client-local priors | by design client-specific (Prop. 2.7) |
| CSL soft labels, BBC offsets | heads + generator | no (they depend on trained models) |

**3.2 What the federated generator estimates.**
- The CVAE is fitted by maximising an ELBO, an approximate MLE. FedAvg–Adam's fixed point is not a function of sums, so
  Thm. 3.1 does not apply. §2.2 characterises the class-specific part:
  - **bias** 0 for the targets (Lemma 1.0);
  - **variance** inflated by $V_c$ under flat averaging in the saturated regime, and ≈ 1 under HWA;
  - **rate** diluted by $m_c/K$ (flat) or by holder weights (HWA);
  - **shrinkage** to zero when $\kappa_c<1$ under coupled decay.
- The shared trunk is fitted to the pooled, prior-weighted ELBO in both worlds (to first order, up to Adam's
  preconditioning). Its tail deficit is a long-tail effect, not a federated one.

**3.3 The statistical floor that exactness cannot remove.**
- The exact class mean is the centralised estimator, with error $\mathbb E\|\hat m_c-m_c\|^2=V_c/n_c$
  ($V_c$ = within-class spread).
- The typical squared distance between two class means is $2F\bar V$ (the paper's $F$, `core.py:2638`). The anchor
  noise relative to the class separation is therefore about $1/(2Fn_c)$ for the rarest class. **MEASURED** $F$ and
  $n_{\min}$:

| | MNIST | FMNIST | SVHN | CIFAR-10 | CIFAR-100 |
|---|---|---|---|---|---|
| $F$ (long tail) | 0.31 | 0.59 | 0.01 | 0.07 | 0.21 |
| rarest $n_c$ | 24 | 24 | ≈ 29 | 100 | 5 |
| $1/(2Fn_{\min})$ | 0.07 | 0.04 | ≈ 1.7 | 0.07 | 0.48 |

- On SVHN the exact mean of a tail class is statistically uninformative even centrally: the anchor noise exceeds the
  separation. This is a second, independent reason for the anchor's failure there, besides Prop. 5's bound.
- On CIFAR-100 the rarest classes are at half the separation.
- No aggregation rule can beat this floor; only more data or better features can.
- Class covariances are worse: with $n_c=5$ in $D=2560$ dimensions, $\hat\Sigma_c$ has rank ≤ 4. MC-S / MC-B
  therefore need shrinkage centrally too.

**3.4 ER vs the best readout.**
- ER equals centralised LDA on the same bodies (Thm. 3.1). LDA is Bayes-optimal for Gaussian class-conditionals with
  a shared covariance under a uniform prior; otherwise it is a plug-in approximation.
- **MEASURED (K16):** ER, CCVR (all clients) and the pooled-real CE refit agree within about 1–2 points on all five
  datasets. Ours + ER is 0.9 / 0.4 / 1.6 below the refit on MNIST / FMNIST / SVHN and 2.0 / 1.2 above it on CIFAR-10 /
  CIFAR-100. The refit is 300 SGD steps, so it is not converged (`core.py:2301–2322`) and is not an upper bound.
- The federated readout is therefore no longer a source of loss **relative to centralised training of the same
  bodies**. The residual is LDA's model error, about 1 point.

---

## 4. Data-space and representation geometry

### 4.1 Class manifolds in a frozen, tiny feature space

- By Lemma 1.0, the feature distribution of each class is the same on every client. Clients differ only in *which*
  classes they see and how often.
- The FE is a 36–390-parameter filter bank. It maps $32\times32\times\{1,3\}$ images to $D=768$ (MNIST / FMNIST /
  SVHN) or $2560$ (CIFAR) post-ReLU features.
- Class separation in that space is $F=0.01$ (SVHN) to 0.6 (FMNIST).
- In the balanced setting, the frozen-FE ceiling is low: CIFAR-100 ≈ 21% and CIFAR-10 ≈ 62% with ER (K16, ours), in
  both worlds.
- N7 shows that training the FE centrally (balanced) moves a pooled body's ER by 0.0 / +1.1 (MNIST / FMNIST). The
  FE's *capacity*, not its *federated training*, is what limits the hard datasets in absolute terms. On CIFAR-100,
  [[N7_C100_GEOM]].

### 4.2 Minority collapse and absent classes

- In the layer-peeled model of Fang, He, Long and Su (PNAS 2021), training with imbalanced classes beyond a threshold
  ratio makes the minority-class classifier vectors collapse toward one direction ("minority collapse").
- A head trained on one client's data is the extreme case: absent classes have imbalance ratio ∞. Under CE their
  logits are only pushed down, with no restoring force (§2.3). Their last-layer rows carry no class-specific geometry
  except what the synthetic phase put there.
- ER sidesteps this. It never uses the trained last layer, only class means and the within-class covariance of the
  embeddings over **all** clients. ER accuracy is therefore a direct measure of the body's class geometry (Fisher
  separability of the embeddings), which is the representation quantity §2.3 compares.

### 4.3 Synthetic features: transfer error, moments, and when they hurt

**APPROXIMATION 4.1 (transfer error, second order in the input).**
- For a head loss $\ell(f(\cdot),t)$ twice differentiable in its input, with input Hessian $\mathcal H$ at the class
  mean $\tilde m_c$ of generated samples:
  $$\mathbb E_{G_c}\ell-\mathbb E_{P_c}\ell=\nabla^\top(\tilde m_c-m_c)+\tfrac12\mathrm{tr}\,\mathcal H(\tilde\Sigma_c-\Sigma_c)+\tfrac12(\tilde m_c-m_c)^\top\mathcal H(\tilde m_c-m_c)+O(\mathbb E\|x-\tilde m_c\|^3+\|\tilde m_c-m_c\|^3).$$
- MC sets $\tilde m_c=m_c$ and $\mathrm{tr}\tilde\Sigma_c=\mathrm{tr}\Sigma_c$ (idealised map).
- The remaining second-order term is $\frac12\mathrm{tr}((\mathcal H-hI)(\tilde\Sigma_c-\Sigma_c))$ for any scalar $h$,
  since the trace difference is 0. It is bounded by $\frac12\|\mathcal H-\bar hI\|_{\rm op}\|\tilde\Sigma_c-\Sigma_c\|_*$
  with $\bar h$ the mean eigenvalue.
- So **MC removes the interaction with the isotropic part of the head's input curvature, and MC-S removes it inside
  its $k$-dimensional subspace.**
- As the dossier notes, the deployed MC is approximate (post-ReLU and plug-in moments): this is a statement about the
  idealised map.
- The first-order version, $|\mathbb E_{G_c}\ell-\mathbb E_{P_c}\ell|\le\mathrm{Lip}(\ell\circ f)\,W_1(G_c,P_c)$
  (Kantorovich–Rubinstein), needs a Lipschitz head, and its constant can be large.

**Off-manifold samples.**
- VAE samples drawn from the prior are under-dispersed. MEASURED spread ratios (generated / real within-class spread)
  are 0.25–0.98 before MC (K16 `spread_tail`). The final ReLU projects them onto the non-negative orthant.
- Samples off the real class manifold train the body on inputs that test data never visit.
- By Prop. 2.8, their effect persists only in directions the real data do not constrain. For an absent class, those
  are exactly the directions that matter at test time. An off-manifold sample of an absent class teaches the body a
  wrong region for that class, and nothing on the client corrects it.

**PROPOSITION 4.3 (label-independent synthetic data are a logit-variance penalty; exact).**
- If the synthetic label $y$ is uniform on $[C]$ and independent of $\tilde h$, then for every logit vector $f$:
  $$\mathbb E_y[\ell(f,y)]=\mathrm{LSE}(f)-\bar f=\log C+\mathrm{KL}(u\,\|\,\mathrm{softmax}f),\qquad\bar f=\tfrac1C\sum_cf_c,$$
  which equals $\log C+\frac1{2C}\sum_c(f_c-\bar f)^2+O(\|f-\bar f\|^3)$.
- Mixed case: suppose a generated sample carries its label's information with probability $s$ (the signal fraction)
  and is label-independent otherwise. Then the synthetic loss is
  $$s\,\mathbb E[\ell(f(\tilde h),y)\mid\text{faithful}]+(1-s)\,\mathbb E\,\mathrm{KL}(u\,\|\,p_f(\tilde h)).$$

*Proof.* $\frac1C\sum_c(\mathrm{LSE}(f)-f_c)=\mathrm{LSE}(f)-\bar f$, and
$\mathrm{KL}(u\|p)=-\log C-\frac1C\sum_c\log p_c=-\log C+\mathrm{LSE}(f)-\bar f$. The Taylor expansion of LSE at
constant logits gives the quadratic term. $\square$

**Consequences.**
- At $s\approx0$, the synthetic phase minimises the spread of the logits on the generator's support. That support
  overlaps the real data's, because the generator was fitted to real features. The phase is therefore an explicit
  *anti-discrimination* regulariser on real-like inputs. By Prop. 2.8, it survives exactly in the absent-class
  directions.
- **PREDICTION:** the synthetic phase hurts the body (relative to real-only) when $s$ is near 0, and helps once $s$
  is large enough for the faithful term to dominate.
- **MEASURED (K17, seeds 3–5).** The signal fraction is estimated from the held-out referee's per-class fidelity
  $a_c$ (`core.py:1585`) as $s=(a-1/C)/(1-1/C)$. "Synthetic gain" is ER(arm) − ER(real-only heads), same seeds:

| | GeFL-F | GeFL-F + AdamW + LA | Ours ($T_s=10$) |
|---|---|---|---|
| MNIST: $s$ / gain | 0.31 / +1.0 | 0.53 / +2.6 | 0.97 / +4.7 |
| FMNIST | 0.50 / +2.1 | 0.72 / +2.8 | 0.87 / +3.4 |
| SVHN | 0.36 / +9.2 | 0.46 / +9.9 | 0.46 / +11.1 |
| CIFAR-10 | 0.034 / **−0.8** | 0.074 / +0.1 | 0.23 / +1.5 |
| CIFAR-100 | 0.002 / **−2.2** | 0.010 / **−1.2** | 0.088 / +1.8 |

  - Within every dataset the gain increases with $s$.
  - The sign changes between $s\approx0.03$ and $0.07$ on the CIFARs, which is what Prop. 4.3 predicts for
    near-label-independent synthetic data.
  - The cross-dataset scale differs: SVHN gains most at moderate $s$ because its real-only bodies are weakest (54.3).
- **MEASURED (toy N5, 3 seeds; real-only ER 85.7):**
  - class-agnostic synthetic data lower ER to 77.9 at $T_s=1$ and 68.4 at $T_s=10$: more epochs, more harm, as the
    proposition predicts;
  - a faithful generator raises it to 96.0 / 97.3;
  - a generator with $s=0.5$ to 92.3 / 96.3.

**CSL as distillation.**
- With target $t=(1-\beta)e_y+\beta\bar p(\tilde h)$, the loss at $s=0$ is $(1-\beta)$ times the penalty plus
  $\beta\,\mathbb E\,\ell(f,\bar p(\tilde h))$: distillation of the cross-architecture ensemble on the generator's
  support, which is a feature-dependent and consistent target.
- **MEASURED (toy N5):** with a class-agnostic generator, CSL turns the harm into a gain (ER 68.4 → 90.2 at
  $T_s=10$ and 77.9 → 85.9 at $T_s=1$, against 85.7 real-only).
- This is consistent with CSL being the only component whose effect survived a readout refit in the dossier's
  ablation analysis (R8). In the bias–variance language of Menon et al. (ICML 2021), CSL lowers the target's bias
  $\|t-p^*(\cdot\mid\tilde h)\|$.

### 4.4 The synthetic-then-real alternation as a mixed objective

- Prop. 2.8 shows that alternation does not minimise $\mathcal L_r+\alpha\mathcal L_s$. It keeps synthetic information
  in proportion to $\rho_r=(1-\eta h_r)^{S_r}$ along each direction, so the real phase "forgets" a synthetic update
  exactly when the update increases the real loss at second order.
- With $S_r=T_r\,b(n_k,B)\approx$ 10–155 steps at $\eta=0.1$, directions with curvature $h_r\gtrsim0.06$–$1$ are
  forgotten.
- **For $T_s$:**
  - more synthetic epochs push $\rho_s\to0$ in more directions, so a faithful generator gains from $T_s\uparrow$ until
    $\omega_i\approx\rho_{r,i}$, and an unfaithful one loses (Prop. 4.3; toy: 77.9 → 68.4);
  - K17 ours, $T_s=1\to10$ with ER: +0.9 / +0.6 / +0.1 / −0.1 / +1.4 on MNIST / FMNIST / SVHN / CIFAR-10 / CIFAR-100,
    small and positive where fidelity is high.
- **Joint vs alternating (toy N5b, same synthetic budget, 3 seeds; ER):**
  - faithful: 97.3 vs 97.2;
  - $s=0.5$: 96.3 vs 96.1;
  - class-agnostic: 68.4 (alternating) vs 74.7 (joint).
  - Interleaving does not help a good generator (the confidently-fit regime makes the two schemes nearly equal), but
    it halves the harm of a bad one, because the real gradient counteracts the penalty at every step instead of only
    afterwards.
  - The paper's interleaved variant CSLM was only ever run with GeFL-F's collapsed generator (F09, IID, one seed);
    this toy agrees with that negative result for good generators.

---

## 5. How each of our changes moves the federated solution toward the centralised one

| change | stage | discrepancy term it removes or reduces | assumptions | removes it exactly? | evidence |
|---|---|---|---|---|---|
| AdamW / no decay on rows / lazy decay | II | non-holder drift $\frac1K\sum_{k\notin H_c}\Delta^{\rm decay}$ (Lemma 2.3) | — | **yes** for no-decay, row-only and lazy; to $1.6\times10^{-4}r_c$ per round for AdamW | rows 0.73–0.97, +7 to +11 points at $K=10$ (K15) |
| HWA | II | non-holder drift **and** $m_c/K$ dilution; count weighting ($V_c\to\approx1$, saturated regime) | quadratic row model for the weighting claim | drift yes; the weighting is exact only in the saturated regime (Prop. 2.4) | +1.3 to +1.8 over the decay fixes at $K=50/100$ (MNIST, K15); rarest-class rate ratio 1.15 / 2.28 / 3.14 at $K=10/50/100$ (N6, normalised as T02) |
| PC-VAE | II | every row term (no class rows); class identity from the exact $m_c$ (Thm. 3.1) | the decoder uses the anchor (fails when $F$ is small: SVHN) | rows: yes; trunk: untouched | tail fidelity 0.89–0.97 (MNIST / FMNIST, K17) |
| gate on $F$ | II | chooses which of the two error terms to accept | threshold set after SVHN | — | P3 of K17: CIFAR-10 branches within 0.13 |
| MC | III (sampler) | first-moment and isotropic second-moment transfer error (Approx. 4.1) | idealised map; deployed map approximate (post-ReLU) | no | K14: −1.1 when removed (MNIST, as trained) |
| KH | III (sampler) | MMD to the exact $\mu_c$ (heuristic) | kernel-mean motivation only | no | −0.7 when removed (MNIST), n.s. on FMNIST |
| CSL | III | label bias of synthetic targets; turns the $s\approx0$ penalty into distillation (Prop. 4.3) | ensemble better than chance | no | survives the readout refit (R8); toy 68 → 90 |
| LA | III | readout-level prior shift; aligns the synthetic- and real-phase bias optima, so absent-class synthetic information survives the real phase (Props. 2.7, 2.8) | — | readout: yes (Menon et al.); body: no effect without a synthetic phase | ER effect vanishes (K14, K17 real-only) |
| ER | post | the whole last-layer discrepancy: equals centralised LDA on the same bodies (Thm. 3.1) | LDA model error (~1 point vs a CE refit) | **yes**, relative to centralised LDA | +7 to +20 on GeFL-F (K16) |
| BBC | post | prior shift of the logits, estimated on synthetic data | label shift only; synthetic ≈ real per class | no | weaker than ER (K16) |

**Terms that no change of ours touches:**
1. **The FE (Stage I) is untouched.** It is first-order consistent with pooled ERM (Cor. 1.2) and as long-tailed as
   centralised ERM (Prop. 2.1(b)). MEASURED cost against a centralised *balanced* FE: 0.0 (MNIST), 1.1 (FMNIST),
   [[N7_C100_FE_SHORT]].
2. **Body sample size.** At $K=10$ every body sees $\approx N/10$ real samples. Only the synthetic phase transfers
   information from other clients' data, and it is bounded by the generator's per-class signal fraction and by
   Prop. 2.8. MEASURED residual against a pooled body: 0.0 (MNIST), 0.8 (FMNIST), [[N7_C100_BODY_SHORT2]]; unmeasured
   on SVHN and CIFAR-10.
3. **The generator trunk's long tail.** It is present centrally too (§2.2, §3.2).
4. **The statistical floor of rare classes.** No aggregation rule can remove it (§3.3).

**Why the remaining gain over strong baselines is small (K17: 0.5–2.0 points).**
- The decay fix removes the largest generator term.
- ER removes the whole readout term.
- What is left for a better generator is the body term (≤ 3.8 points on MNIST / FMNIST by N7), part of which a
  decay-fixed GeFL-F generator already recovers. K17: the decay-fixed GeFL-F's synthetic gain over real-only bodies is
  +2.6 / +2.8 on MNIST / FMNIST, ours +4.7 / +3.4.
- **CONJECTURE:** a body trained on the pooled real data is a ceiling for any body trained on this data. It may not
  be, because synthetic data can also act as augmentation. If it is, the margin over the decay fix is bounded by the
  pooled-body ER minus the decay-fixed GeFL-F's ER:
  - MNIST: 94.4 − 92.5 = 1.9 (N7 seed 0 / K17). The measured margin is +2.0, at the putative ceiling.
  - FMNIST: 81.9 − 80.5 = 1.4. Measured +0.5.
  - These compare one N7 seed with three K17 seeds, so they are indicative only.
- **Confirmed for MNIST / FMNIST:** the remaining loss is not in the federated aggregation of anything we can fix. It
  is in the long-tailed data itself (§2.4: the centralised long-tail cost is 3.2 on MNIST and 2.7 on FMNIST) and,
  on FMNIST, about 1 point in the FE.

---

## 6. Making the federated solution closer to the centralised one: concrete proposals

Each proposal states what it approximates, its error order, its cost, and a falsifiable prediction. CLAUDE.md
applies to any run: check the ledger first and reuse seeds 0–5. **P0 is a measurement that decides whether P3 is
worth running.**

**P0. Measure the pooled-body ceiling on SVHN / CIFAR-10 / CIFAR-100 (diagnostic, no new method).**
- *What.* For each saved run (A12: models are saved from K16 on), train each architecture's body on the pooled real
  features of all clients with the Stage-III schedule, then apply ER. This is the N7 protocol, at no extra cost in
  federated rounds.
- *Cost.* Head-stage compute only, *if* the K16 / K17 model archives (`<folder>__models.zip`, action plan D2) were
  kept. They are not in the repository. Without them P0 needs the FE and heads re-trained, which duplicates K16 / K17
  arms: run `python experiments/ledger.py --check` first and ask before re-running.
- *Prediction.* The pooled-body ER exceeds Ours + ER by ≤ 1 point on SVHN and by 1–3 points on CIFAR-10 / CIFAR-100,
  where $s$ is low (0.09–0.46). If the gap is ≤ 1 everywhere, no body-side federated fix can be worth more than 1
  point, and P3 should be dropped.

**P1. Class-wise gating or relabelling of synthetic data by an exact-statistics signal-fraction estimate.**
- *What.* Before Stage III, the server computes from existing sums the class-balanced LDA on frozen features
  (Thm. 3.1: exact). It scores $n$ generated samples per class and estimates $\hat s_c=(\hat a_c-1/C)/(1-1/C)$, where
  $\hat a_c$ is the LDA's agreement with the conditioning label. This is label-free on the server and needs no
  held-out data.
- *Rule.* Draw synthetic labels only from classes with $\hat s_c\ge s_0$, or set the CSL weight per class to
  $\beta_c=1-\hat s_c$ (pure ensemble labels where the generator carries no class information).
- *Approximation.* The LDA agreement is a biased proxy for the true $s_c$: LDA error inflates disagreement. The
  error is of the order of the LDA's own per-class error. Choose $s_0$ on the validation split.
- *Cost.* None in communication: the sums are already released.
- *Prediction (Prop. 4.3).* On CIFAR-100 and CIFAR-10, GeFL-F (+ decay fix) with gating moves from −2.2 / −0.8
  (synthetic gain vs real-only) to ≥ 0. For ours, the change is ≥ 0 on CIFAR-100 (where $s=0.088$ is near the
  harmful range) and 0 ± 0.3 on MNIST / FMNIST, where $\hat s_c$ is high for all classes and the gate does nothing.

**P2. Class-balanced Stage-I loss with weights from the exact global counts.**
- *What.* Each client multiplies its Stage-I loss on sample $i$ by $w_{y_i}=N/(Cn_{y_i})$ (one released count
  vector; secure aggregation).
- *Approximation.* By Cor. 1.2 (first order, error $O(\eta^2\tau^2)$ with the same TV error as above), the FedAvg
  surrogate becomes the empirical balanced risk with one head per client. This is the federated analogue of the
  centralised class-balanced FE (`stage_i_oracle`).
- *Caveat.* Loss reweighting and balanced sampling have the same expected objective but different variance. With
  rare classes at 24 samples, clip $w$ (e.g. ≤ 20) to bound the variance. This is a design choice, to be validated.
- *Cost.* One count release, which ER already implies.
- *Prediction (N7).* +1 ± 1 point of Ours + ER on FMNIST, 0 ± 0.5 on MNIST. On the CIFARs, the centralised-balanced
  FE measurement of P0 / N7 decides.

**P3. Cross-client body refresh (only if P0 shows a gap ≥ 1.5 points).**
- *What.* After Stage III, broadcast every architecture's body (ER already requires this). For $R$ rounds, every
  client runs $T_r$ local epochs of each body on its real features with LA. Aggregate each architecture by FedAvg with
  uniform weights (then $\tau_k\propto n_k$ gives pooled weighting by Cor. 1.2). Finish with ER.
- *Approximation.* First-order equal to pooled ERM of the body (Cor. 1.2, TV error ≤ 0.02 at $K=10$). Drift can be
  removed exactly by control variates (Lemma 1.4; 2× upload). LA is needed because averaging now mixes clients with
  different priors (Prop. 2.7, limits).
- *Cost.* Each client trains all $G$ bodies, so head compute is about 10× for $R$ rounds. Upload: $G$ bodies per round
  ($\le1.1\times10^5$ parameters each). **This departs from GeFL-F's premise that a client runs only its own
  architecture** and must be presented as such.
- *Prediction.* Ours + ER moves to within 0.5 of the pooled-body ER from P0 with $R=10$.
- *Cheaper variant P3b: per-class gradient matching.*
  - Each client returns, for each architecture $g$ and class $c$, the sum of its real per-sample body gradients
    $\sum_{i\in D_{kc}}\nabla_{\theta_g}\ell_i$. This is one forward–backward pass per body, no training, and the sum is
    secure-aggregatable.
  - The server then optimises the synthetic features of class $c$ so that their gradient on $\theta_g$ matches
    $\frac1{n_c}\sum_k\sum_{i\in D_{kc}}\nabla\ell_i$, as CReFF does for a classifier (Shang et al., IJCAI 2022).
  - *Approximation:* a synthetic step then equals a pooled-real step to first order in $\eta$ at the current
    $\theta_g$; it degrades as $\theta_g$ moves away, $O(\eta\,S_s\,L)$.
  - *Cost:* $C\times|\theta_g|$ numbers per architecture per refresh, up to $10^7$ for the largest heads at $C=100$.
    That is too large for CIFAR-100 unless restricted to the last body layers.

**P4. Per-class step-normalised row aggregation (rows branch only: SVHN / CVAE + HWA).**
- *What.* $r_c\leftarrow r_c+\sum_{k\in H_c}\frac{n_{kc}}{n_c}\frac{r_{k,c}-r_c}{\nu_{kc}}\bar\nu$, where
  $\nu_{kc}$ is the number of local steps whose batch contained $c$ and $\bar\nu$ a common step scale.
- *Approximation.* Prop. 2.4(4) with $(I-M_{kc})^{-1}\approx(\eta A\nu_{kc})^{-1}$. Exact in the quadratic row model
  to first order in $\eta$. Unlike HWA's $E(n)$ weights, it targets the pooled MLE in both regimes.
- *Cost.* Clients add $\nu_{kc}$ to the HWA sums (secure-aggregatable).
- *Prediction.* Rows closer to the pooled class targets, but a small accuracy effect: ≤ 0.5 on SVHN. Low priority,
  because the anchored branch has no rows.

**P5. Class-balanced ELBO weights for the generator trunk.**
- *What.* Weight each sample's ELBO term by $N/(Cn_{y_i})$ (clipped). This is the centralised fix for the trunk's
  long tail (§2.2). `stage_ii_oracle` is its centralised version and was never run (E09_oracles, NOT RUN in
  `experiments/LEDGER.md`).
- *Prediction.* Tail fidelity and the tail spread ratio up by ≥ 0.05 for the rows branch. Accuracy effect uncertain
  (≤ 1 point). Run only together with P1's diagnostic.

**P6. Do not prioritise.**
- *FE drift correction (SCAFFOLD on Stage I).* Toy: ≤ 1 point (Prop. 2.1 table).
- *Exact BN recalibration.* N7: ≤ 0.3.
- *Interleaving synthetic and real data.* Toy N5b: no gain for a faithful generator.
- *Shorter real phases ($T_r\downarrow$) to retain more synthetic information.* By Prop. 2.8 this raises
  $\rho_r=(1-\eta h_r)^{S_r}$. The measured retention is already ≥ 0.97 for absent classes under LA, and 0.7–0.85
  for held classes. **PREDICTION:** $T_r\in\{2,5\}$ changes ER by ≤ 0.5 when the generator is faithful.
- *ERL.* E33 was negative.
- *Fisher-weighted aggregation of rows.* The row Fisher is block-diagonal and $\propto n_{kc}$ in the linear model,
  so Fisher-weighted merging (Matena & Raffel, NeurIPS 2022) reduces to HWA with linear weights, already ablated
  (K02).

---

## 7. Measurements that would confirm or refute each claim

All are per-round quantities computable inside the existing loops of `core.py`. "Before / after aggregation" refers to
the server's `Averager`.

| # | claim tested | quantity (formula) | where in `core.py` |
|---|---|---|---|
| M1 | Cor. 1.2, drift | FE gradient heterogeneity at the global $\phi$: $\zeta^2=\sum_k\frac{n_k}N\|\nabla_\phi\mathcal L_k-\nabla_\phi\mathcal L\|^2$, and the pooled gradient norm $\|\sum_k\frac{n_k}N\nabla_\phi\mathcal L_k(\phi^{\rm end})\|$ relative to $\max_k\|\nabla_\phi\mathcal L_k\|$ | `stage_i` after each `fe_avg.result()` (`:961`); one extra full-batch pass per client |
| M2 | Prop. 2.2 | per-BN-channel ratio of averaged running variance to the exact pooled variance of the aggregated FE ($\sum u^2/n-(\sum u/n)^2$ over all clients) | after Stage I, before `compute_features` |
| M3 | Prop. 2.1(b) | count-only pair weights $w^{\rm fed}_{cc'}$, $\rho_{cc'}$, and the per-pair Mahalanobis separation $D^2_{cc'}$ of the frozen features from exact sums | after Stage I (statistics as in `anchor_stats`, `:1267`) |
| M4 | Approx. 2.5 | $\kappa_c$ from counts before training; per round and class: holder gain $\bar G_c$, non-holder loss $\bar D_c$, and the fraction of coordinates with $\mathrm{sign}(r_{k,cj}-r_{cj})=-\mathrm{sign}(r_{cj})$ on non-holders | `stage_ii` loop (`:1428–1436`); counts from `ctx.part["counts"]` |
| M5 | Prop. 2.4 | per class: $\|r_c-\hat t_c^{\rm pool}\|$ is not observable for the CVAE, so log instead the row change per class-$c$ step, $\|r_{k,c}^{+}-r_c\|/\nu_{kc}$, per holder (its dispersion across holders estimates the weighting error) | `gen_local_update` (`:1175`): count $\nu_{kc}$ |
| M6 | Prop. 2.7 | $\max\lvert b^{\rm LA}-b^{\rm CE}+a_k\rvert$ is identity-guaranteed; log instead the body distance between LA and CE arms with a synthetic phase (non-zero, measures phase conflict) | heads saved (A12) |
| M7 | Prop. 2.8 refinement | counterfactual retention per class group: run the real phase from the pre-synthetic head copy and compute $R=\langle w^{\rm after}-w^{\rm cf},\,w^{\rm syn}-w^{\rm before}\rangle/\|w^{\rm syn}-w^{\rm before}\|^2$ for absent and held rows; prediction ≥ 0.9 for absent rows with LA, lower without | `stage_iii` around `syn_phase()` / `real_phase()` (`:2127–2130`); one extra real phase on one client every 10 rounds |
| M8 | Prop. 4.3 | per class: $\hat s_c$ from the exact-sum LDA agreement on generated samples (P1), and the logit spread $\frac1C\sum_c(f_c-\bar f)^2$ on generated features before and after the synthetic phase (prediction: it falls when $\hat s_c\approx0$) | `stage_iii`, pool construction (`:1937–1981`) |
| M9 | §2.3, §5 | pooled-body ER ceiling (P0) and ER of one-client real-only bodies on the same saved FE | post hoc on saved models |
| M10 | ERL conjecture | embedding norm $\mathbb E\|u(h)\|$, within-class covariance condition number and ECE per round under ERL vs normal training | `stage_iii` with `erl` on |
| M11 | §3.3 | per class: $V_c/n_c$ divided by the nearest-other-class $\|m_c-m_{c'}\|^2$ (anchor noise-to-separation) | after Stage I, from `anchor_stats` |
| M12 | Lemma 2.3 | per round, the per-class row change on non-holders (must be exactly 0 for NOWD / ROWNOWD / LCD and $(1-\eta\lambda)^{\tau_k}$ for AdamW) | `stage_ii` loop |

---

## 8. Numerical checks run for this report

Scripts and outputs are in `/tmp/claude-0/-home-user-Fedlearning/76f672ce-438e-501b-a061-1360cd7e1c9d/scratchpad/math/`.
That folder is session scratch space, outside the repository and not committed. All checks ran on CPU; the scripts
import `experiments/core.py` read-only and read the cached tensors in `experiments/data/`.

| id | script | what it checks | result | agrees? |
|---|---|---|---|---|
| N1 | `n1_quadratic.py` | Prop. 1.3 formula; O(η) consistency with $\tau\propto n$; Lemma 1.4; Prop. 2.8 formula | formula vs iteration ≤ $10^{-12}$; error 0.0023 → 0.19 linear in $\eta$; control variates $10^{-14}$; alternation $10^{-15}$, unconstrained directions weight 1.000 | yes |
| N2 | `n2_fe_toy.py` | Stage I in miniature: federated vs centralised FE, SCAFFOLD, auxiliary shared classifier | table in §2.1; pair-ratio vs $\rho$ Spearman 0.51–0.54 | partly: the size of the federated advantage over C-ERM is not explained by first-order theory |
| N3 | `n3_rows_adam.py` | rows under FedAvg with coupled Adam / HWA / AdamW / no decay on the real MNIST-s0 counts | collapse exactly for $\kappa<1$ classes; decay fixes and HWA reach 0.95–1.15 of the target norm; HWA closest to the pooled mean | yes |
| N3b | `n3b_vinfl.py` | $V_c$ of Prop. 2.4 | simulated / predicted equal to 2 decimals; HWA 0.99–1.03 | yes |
| N4 | `n4_la_reparam.py` | Prop. 2.7 (LA, FedRS) | $4\times10^{-16}$; exact; breaks with a synthetic phase (0.16) | yes |
| N5 | `n5_stage3_toy.py` | Stage III alternation with generators of varying $s$, CSL | real-only ER 85.7; class-agnostic 77.9 / 68.4 ($T_s=1/10$); CSL 85.9 / 90.2; faithful 96.0 / 97.3 | yes |
| N5b | `n5b_joint_vs_alt.py` | joint vs alternating; counterfactual retention with / without LA | §4.4 | the "joint helps a good generator" prediction was **refuted**; retention as predicted |
| N6 | `n6_partition_stats.py` | exact partition reconstruction; FedNova TV; $\rho$; $V_c$; $\kappa$; rates | partitions match the logs exactly | — |
| N6b | `n6b_pair_weights.py` | first-order pair weights fed vs cen | §2.1 | — |
| N6c | `n6c_kappa_vs_logs.py` | $\kappa_c$ vs the logged final row norms of every GeFL-F run | 352 / 352 collapse for $\kappa<1$; Spearman 0.94 | yes |
| N7 | `n7_fe_body_decomp.py`, `n7b_fe_body_decomp_sched.py` | real data: FE variants × body variants × ER | §2.1, §2.3; [[N7_ALL]] | — |

Caveats:
- N7 uses one seed per dataset, on CPU, so it is not bit-identical to the Kaggle runs. Its one-client real-only bodies
  (91.5 / 78.1) reproduce K17's "Real only + ER" (90.0 / 77.8, seeds 3–5) and K16's GeFL-F + ER on seed 0 (90.8 /
  79.5) within about 1 point.
- The toys are models of mechanisms, not of the networks.

---

## 9. Summary

**Where the federated pipeline loses accuracy relative to centralised training on the same pooled long-tailed data,
with the same frozen-FE architecture.** Ranked by measured size for GeFL-F, followed by the size left after our
changes.

| rank | loss source | GeFL-F | after our changes | type |
|---|---|---|---|---|
| 1 | last layer trained on one client's skewed data | 7–20 points (ER lift, K16) | ≈ 0 relative to centralised LDA (ER, Thm. 3.1); LDA model error ~1 | federated, fixed exactly |
| 2 | generator rows: coupled-decay collapse ($\kappa_c<1$) | 7–11 points at $K=10$ (K15) | 0 (decay fix / HWA / PC-VAE) | federated, fixed exactly |
| 3 | body sample size and coverage (one client per architecture) | 3.0 / 3.8 (MNIST / FMNIST, real-only vs pooled body, N7); larger where real-only bodies are weak (SVHN: real-only 54.3 vs ours 65.5) | 0.0 / 0.8 (MNIST / FMNIST); unmeasured elsewhere ([[N7_C100_BODY_SHORT3]]) | federated, reduced by synthetic data; residual bounded by the signal fraction $s$ |
| 4 | synthetic data with $s\approx0$ (anti-discrimination penalty) | −0.8 / −2.2 vs real-only (CIFAR-10 / -100, K17) | removed (ours $s$ = 0.23 / 0.088: +1.5 / +1.8) | federated through the generator |
| 5 | FE trained federatedly and long-tailed | 0.0 / 1.1 (MNIST / FMNIST, vs centralised balanced FE, N7) | unchanged | on FMNIST ≈ 0.4 federated + 0.7 long-tail weighting (vs a centralised ERM FE); on MNIST the centralised ERM FE is 0.5 worse; not touched by any change |
| 6 | row dilution $m_c/K$ (no-decay flat averaging) | 1.3–1.8 at $K=50/100$ (MNIST) | 0 with HWA; absent in PC-VAE | federated |
| — | long-tailed data itself (present centrally) | the remainder of the LT vs balanced gap; centralised cost 3.2 (MNIST) / 2.7 (FMNIST), against our drops of 2.9 / 3.3 (N7, seed 0) | unchanged | not federated |

**The 2–3 most promising theory-backed changes:**
1. **P1, exact-statistics signal-fraction gating / CSL weighting of synthetic data.**
   - *Backed by:* Prop. 4.3 (exact), the K17 sign pattern and toy N5.
   - *Cost:* none.
   - *Effect:* removes the regime where synthetic data hurt, which matters exactly on the hard datasets (CIFAR-10 /
     CIFAR-100), where our margin is smallest and least certain.
2. **P2, class-balanced Stage-I loss from exact counts.**
   - *Backed by:* first-order equivalence to the centralised balanced FE (Cor. 1.2), and N7's +1.1 on FMNIST for
     that FE.
   - *Effect:* the only change that touches the stage none of our components touches.
3. **P0 → P3, measure the pooled-body ceiling on the hard datasets; then, only if it shows ≥ 1.5 points, a
   cross-client body refresh.**
   - *Backed by:* Cor. 1.2 and Lemma 1.4, which make it provably first-order equal to pooled ERM.
   - *Caveat:* it departs from GeFL-F's one-architecture-per-client premise, and P0 alone already tells the paper how
     much any body-side method could still gain.

**What the analysis rules out** (to save compute):
- FE drift correction and BN recalibration (≤ 0.3–1 point);
- interleaving synthetic data for a good generator;
- any further per-client loss correction for single-client bodies: LA, FedLC, FedRS and balanced softmax are
  reparametrisations (Prop. 2.7).

---

### References (only those cited above)

- Charles, Z. and Konečný, J. Convergence and accuracy trade-offs in federated learning and meta-learning. AISTATS 2021.
- Fang, C., He, H., Long, Q. and Su, W. J. Exploring deep neural networks via layer-peeled model: minority collapse in imbalanced training. PNAS 2021.
- Kang, M. et al. GeFL: model-agnostic federated learning with generative models. IEEE TMC 2025 (as cited in the paper draft).
- Karimireddy, S. P. et al. SCAFFOLD: stochastic controlled averaging for federated learning. ICML 2020.
- Kingma, D. P. and Ba, J. Adam: a method for stochastic optimization. ICLR 2015.
- Li, X.-C. and Zhan, D.-C. FedRS: federated learning with restricted softmax for label distribution non-IID data. KDD 2021.
- Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. ICLR 2019.
- Matena, M. and Raffel, C. Merging models with Fisher-weighted averaging. NeurIPS 2022.
- McMahan, B. et al. Communication-efficient learning of deep networks from decentralized data. AISTATS 2017.
- Menon, A. K. et al. Long-tail learning via logit adjustment. ICLR 2021.
- Menon, A. K. et al. A statistical perspective on distillation. ICML 2021.
- Oh, J., Kim, S. and Yun, S.-Y. FedBABU: toward enhanced representation for federated image classification. ICLR 2022.
- Pathak, R. and Wainwright, M. J. FedSplit: an algorithmic framework for fast federated optimization. NeurIPS 2020.
- Soudry, D. et al. The implicit bias of gradient descent on separable data. JMLR 2018.
- van der Vaart, A. W. Asymptotic Statistics. Cambridge University Press, 1998.
- Shang, X., Lu, Y., Huang, G. and Wang, H. Federated learning on heterogeneous and long-tailed data via classifier re-training with federated features (CReFF). IJCAI 2022.
- Wang, J., Liu, Q., Liang, H., Joshi, G. and Poor, H. V. Tackling the objective inconsistency problem in heterogeneous federated optimization. NeurIPS 2020.
- Zhang, J. et al. Federated learning with label distribution skew via logits calibration (FedLC). ICML 2022.
