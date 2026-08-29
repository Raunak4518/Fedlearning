# Paper skeleton — Class-Balanced GeFL

Working title: **Class-Balanced GeFL: Frequency-Aware Aggregation and
Fidelity-Gated Sampling for Model-Heterogeneous Federated Learning under Long-Tailed Data**

Target venue (in priority): IEEE TMC (same journal as GeFL) → NeurIPS/ICLR
FL Workshop → AAAI/CVPR (only if empirics reach top-conference bar).

---

## 0. Abstract (≤ 200 words, filled last)

- Problem: model-heterogeneous FL via a shared generator (GeFL, Kang 2025)
  is class-blind — flat aggregation of generator updates and uniform sampling
  of synthetic conditioning labels ignore local and global class imbalance
  entirely.
- Contribution: two coupled, architecture-agnostic mechanisms — Mechanism A
  (frequency-weighted conditioning aggregation) and Mechanism B (fidelity-
  gated adaptive sampling) — that operate only on the shared generator.
- Result headline: +__% tail accuracy over vanilla GeFL on CIFAR-10-LT
  (IF=0.01) without loss on overall or head accuracy, across MNIST-LT,
  FMNIST-LT, CIFAR-10-LT, and CIFAR-100-LT.
- Ancillary result: a naive class-support floor variant fails; we diagnose
  the quality-quantity tension driving that failure.

---

## 1. Introduction

### Motivation paragraph
- Realistic federated deployments simultaneously face architectural
  heterogeneity (hospitals with different imaging hardware and models) and
  class imbalance (rare diseases, industrial defects).
- Existing FL methods handle each in isolation.

### Gap paragraph
- GeFL (Kang et al. 2025, IEEE TMC) is the direct precedent for model-
  heterogeneous FL via a shared generator. It solves the architecture
  problem but is class-blind: `w_g ← (1/|C|) Σ w_k` and `y_i ~ Uniform(y)`.
- No existing method combines generator-mediated model-heterogeneous FL
  with class-frequency-aware aggregation and sampling.

### Contribution list
1. **Diagnosis:** demonstrate empirically that vanilla GeFL exhibits
   near-zero tail accuracy under CIFAR-10-LT IF=0.01 (baseline tail = X.X%).
2. **Mechanism A:** frequency-weighted per-class conditioning aggregation
   using effective-number-of-samples weights (adapting Cui et al. 2019).
3. **Mechanism B:** fidelity-gated adaptive sampling that interpolates from
   natural class frequency toward inverse-frequency rebalancing.
4. **Empirical validation** across 4 datasets × 3 imbalance factors × 3 seeds.
5. **Negative-result contribution:** class-support floor fails; we analyze
   why (§__).

---

## 2. Related Work

**Federated Learning under Class Imbalance.** CReFF (Shang, IJCAI 2022),
Ratio Loss (Chen 2022), BalanceFL (IPSN 2022), FedCBA, FedSat (2024).
All operate on a shared classifier, not a generator.

**Model-Heterogeneous FL.** FedDF (Lin, NeurIPS 2020) via ensemble
distillation; AvgKD (Afonin, ICLR 2022) via pairwise logits; LG-FedAvg
(Liang 2020) via shared feature extractor. GeFL and GeFL-F (Kang, TMC 2025)
via a shared generative model — the direct precedent.

**Federated generative models for knowledge distillation.** FedGen (Zhu,
ICML 2021) trains a generator on the server from client-uploaded label
predictions. FedFTG (Zhang, CVPR 2022) is data-free distillation via a
generator. DENSE (Zhang, NeurIPS 2022) extends this to heterogeneous
architectures. **None reason about class balance.**

**Class-balanced generative modeling.** GAMO (Mullick, ICCV 2019) uses a
classifier-guided GAN to synthesize minority-class samples. Centralized;
does not address federated aggregation. Our Mechanism B is arguably a
federated + non-circular-fidelity descendant of GAMO's core idea.

**Effective number of samples.** Cui et al. (CVPR 2019). We adapt their
weight from a loss-reweighting term to an aggregation-weighting term; the
direction question (weight-as-weight vs. weight-as-inverse) is discussed
in §__.

---

## 3. Preliminaries

Recap in one page:
- Notation: clients $k \in \mathcal{C}$, local data $\mathcal{D}_k$,
  target networks $T_{\theta_{k,m}}$ (architecture index $m \in
  \{1,\dots,M\}$), generator $G_{w_k}$.
- GeFL Algorithm 1 (Kang 2025): alternate FedAvg on $w$ and grouped FedAvg
  on $\theta_m$; augment target training with $B$ synthetic samples per
  batch drawn as $y_i \sim \text{Uniform}$, $x_i \sim G(z_i \mid y_i)$.
- The two class-blind lines:
  $$w_g \leftarrow \frac{1}{|\mathcal{C}|} \sum_k w_k
     \qquad\text{(class-blind aggregation)}$$
  $$y_i \sim \text{Uniform}(\{1,\dots,C\})
     \qquad\text{(class-blind sampling)}$$

---

## 4. Method

### 4.1 Mechanism A — Frequency-weighted conditioning aggregation

Each generator has a **conditioning pathway** — a small set of parameters
whose rows correspond to individual classes (label embedding, one-hot
projection, or classifier-free-guidance embedding). Denote its parameters
$w^{\text{cond}}$ and the trunk as $w^{\text{trunk}}$.

Trunk aggregation stays volume-weighted:
$$w^{\text{trunk}}_g \leftarrow \sum_k \frac{n_k}{\sum_j n_j} w^{\text{trunk}}_k$$
where $n_k = |\mathcal{D}_k|$.

Per-class conditioning row $r$ uses effective-number weighting:
$$E_n = \frac{1 - \beta^{n}}{1 - \beta}
       \quad\text{(Cui et al. 2019)}$$
$$\alpha^{(r)}_k = \frac{E_{n_{k,r}}}{\sum_j E_{n_{j,r}}}
                 \qquad n_{k,r} = |\{(x, y) \in \mathcal{D}_k : y = r\}|$$
$$w^{\text{cond}, r}_g \leftarrow \sum_k \alpha^{(r)}_k \cdot w^{\text{cond}, r}_k$$

**Direction justification.** Rare-class row $r$: only clients holding
class $r$ can inform its conditioning; effective-number weight scales
monotonically with count, giving these clients proportional influence.
The inverse direction (Cui's original *loss* weight) would upweight
clients *without* class $r$, which is nonsensical for per-class
conditioning parameters. Empirically confirmed in ablation (Table __).

### 4.2 Mechanism B — Fidelity-gated adaptive sampling

Maintain a per-class fidelity EMA:
$$f_c^{(t)} \leftarrow \lambda f_c^{(t-1)} + (1 - \lambda) \bar{p}_c^{(t)}$$
where $\bar{p}_c^{(t)}$ is the mean target-network softmax confidence on
this round's synthetic samples of class $c$. (No extra forward passes:
reuses the local training pass.)

Sampling distribution blends natural and inverse-frequency:
$$g = \frac{1}{C} \sum_c f_c
     \qquad\text{(mean fidelity gate, } g \in [0,1] \text{)}$$
$$p^{\text{inv}}_c = \frac{\left(\pi_c^{-1}\right)^\gamma \cdot f_c}{Z}$$
$$p(y = c) = (1 - g) \cdot \pi_c + g \cdot p^{\text{inv}}_c$$
where $\pi_c$ is the natural global class frequency, $\gamma$ softens the
inverse (default 1.0), and $Z$ normalizes.

Early training ($g$ low): $p \approx \pi$ (natural). Late ($g$ high):
$p \approx p^{\text{inv}}$ (rebalanced). Per-class $f_c$ inside
$p^{\text{inv}}$ prevents oversampling a rare-but-still-unreliable class.

### 4.3 Threat model / privacy note

Both mechanisms operate only on generator parameters and per-class counts.
No raw data leaves clients. MND privacy ratio (Kang 2025) is used as the
sanity check (§__).

---

## 5. Experiments

### 5.1 Setup

**Datasets:** MNIST-LT, FMNIST-LT, CIFAR-10-LT, CIFAR-100-LT with
imbalance factor $\rho \in \{0.01, 0.1, 1.0\}$ (CReFF convention).

**Client partition:** Dirichlet($\alpha = 0.3$) — non-IID default matching
existing FL long-tail literature.

**Federated setup:** 10 clients (MNIST/FMNIST/CIFAR-10), 20 clients
(CIFAR-100). 3 heterogeneous target networks (`cnn_small`, `cnn_deep`,
`mobilenet_lite`).

**Federated schedule** (paper Tables XIV/XV): $T_{KA}/2 = 100$
generator warm-up rounds, $T_{TN} = 100$ interleaved rounds, $T_s = 1$
synthetic-only local epoch, $T_r = 5$ real-only local epochs, $T_g = 5$
generator local epochs.

**Generator:** Conditional Convolutional VAE (paper CVAE), latent 32
(MNIST/FMNIST) / 50 (CIFAR-10/100), sum-reduction ELBO, Adam lr=1e-3.

**Baselines:** vanilla GeFL (flat aggregation, uniform sampling), FedAvg
(grouped per architecture), FedProx, CReFF-style class-balanced classifier
retraining where the target-net pool allows.

**Evaluation:** overall accuracy plus head/medium/tail-bucket accuracy
(paper convention: top 33% / middle 33% / bottom 33% by class frequency),
macro-F1, class-balanced accuracy, MND privacy ratio.

**Reporting:** mean ± std over 3 seeds. Bold = best; underline = second.

### 5.2 Main results

**Table 1 — Overall vs. tail accuracy across datasets (IF=0.01, α=0.3).**

| Method           | MNIST-LT | FMNIST-LT | CIFAR-10-LT | CIFAR-100-LT |
|------------------|----------|-----------|-------------|--------------|
|                  | ovr / tail | ovr / tail | ovr / tail | ovr / tail |
| FedAvg (grouped) | __ / __  | __ / __   | __ / __     | __ / __      |
| FedProx          | __ / __  | __ / __   | __ / __     | __ / __      |
| CReFF (adapted)  | __ / __  | __ / __   | __ / __     | __ / __      |
| GeFL             | __ / __  | __ / __   | __ / __     | __ / __      |
| + Mech A         | __ / __  | __ / __   | __ / __     | __ / __      |
| + Mech B         | __ / __  | __ / __   | __ / __     | __ / __      |
| **+ A + B (ours)** | __ / __ | __ / __  | __ / __     | __ / __      |

**Table 2 — Accuracy vs. imbalance severity (CIFAR-10-LT, α=0.3).**

| Method | IF=1.0 (bal) | IF=0.1 | IF=0.01 |
|---|---|---|---|
| GeFL | __ | __ | __ |
| **+ A + B** | __ | __ | __ |

**Figure 1 — Per-class accuracy heatmap.** Grid: methods × class index,
ordered by class frequency. Shows tail lift.

**Figure 2 — Training curves.** Overall and tail accuracy vs.
communication round for baseline / A / B / A+B on CIFAR-10-LT.

### 5.3 Ablations

**Table 3 — Mechanism ablation (CIFAR-10-LT IF=0.01).** A alone / B alone
/ A+B / A+B + support-floor / A+B with inverse Mech-A direction.

**Table 4 — Hyperparameter sensitivity.** Mech A: $\beta \in \{0.99, 0.999, 0.9999\}$.
Mech B: $\lambda \in \{0.4, 0.6, 0.8\}$, $\gamma \in \{0.5, 1.0, 2.0\}$.

### 5.4 Analysis

**5.4.1 Why the support floor hurts.** Support-floor variant falls back
to flat mean when total round support < 20. Empirically worse than no
floor (Table 3). Analysis: fallback erases the very signal Mech A is
designed to inject; adding a floor is equivalent to disabling Mech A
for the classes that need it most.

**5.4.2 Fidelity dynamics.** Figure __ shows per-class $f_c$ over
training. Head-class $f_c$ rises within ~20 rounds; tail-class $f_c$
lags until ~50 rounds — matching the intuition that Mechanism B's gate
only opens for classes the generator has actually learned.

**5.4.3 Privacy check.** MND ratio for baseline vs. proposed across
all datasets. Both < 1 (no memorization signal). Mechanism A's
frequency-weighted upweighting of rare-class clients could in principle
increase memorization risk; empirically no measurable effect at the
default $\beta$.

**5.4.4 Comparison to CReFF.** CReFF operates on a shared feature
classifier and cannot be applied to fully-heterogeneous target networks.
Where applicable (partial-heterogeneity setting), CReFF and our method
are complementary — combining them yields __.

---

## 6. Limitations

- Mechanism A requires an architectural conditioning pathway with per-class
  parameters; not every conditional generator has this natively (paper
  reference CCVAE uses one-hot channel concatenation, which we replace
  with an embedding — architectural change quantified in ablation).
- No formal convergence analysis of the frequency-weighted aggregation.
- MND is a proximity-based memorization metric; it does not rule out
  gradient-based membership inference.
- Real-world imbalanced-medical-imaging validation deferred to future work.

---

## 7. Conclusion

- Contribution restated in one sentence.
- One-sentence roadmap: quality-gated fidelity signals (contingency Path B),
  formal analysis, real-world medical validation.

---

## Appendix (in supplementary)

- **A.** Full hyperparameter tables.
- **B.** All ablation results with error bars.
- **C.** Per-class accuracy breakdowns for every (method, dataset, IF).
- **D.** Fidelity evolution plots for all classes.
- **E.** MND ratio breakdowns.
- **F.** Compute budget disclosure (Kaggle 2×T4, total GPU-hours per experiment table).
- **G.** Code availability (GitHub URL + commit hash for each reported number).

---

## Notes to self while filling in

- Numbers land from `docs/results_tables/*.csv` produced by
  `scripts/kaggle_full_sweep.ipynb`. Aggregator cell already emits
  mean ± std per (mech, imb, dir).
- Every result cell in Tables 1-4 must have a matching seed count
  (footer: "n=3 seeds, mean ± std").
- Figures 1-2 generated by [`utils/visualize.py`](../utils/visualize.py)
  functions already present in the repo — reuse, don't rewrite.
- Related-work paragraph on FedGen / FedFTG / DENSE is NON-NEGOTIABLE.
  Confirmed still missing from the current proposal draft.
- Support-floor negative result stays in the paper. Instructive.
