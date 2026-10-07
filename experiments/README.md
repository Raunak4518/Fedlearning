# GeFL-F experiments: one standalone file per experiment

Each `E*.py` file in this folder is **self-contained**: it needs only PyTorch,
torchvision and NumPy, so it can be copied alone to Kaggle, Colab or a lab
machine. It is built from two structured sources:

```
experiments/core.py            shared GeFL-F implementation (paper appendix exact)
experiments/specs/E*.py        what each experiment runs, and why
experiments/build.py           core.py + spec -> experiments/E*.py
```

Edit `core.py` or a spec, then run `python experiments/build.py`.

```bash
python experiments/E03_method_components.py --quick          # viability check, minutes
python experiments/E03_method_components.py                  # full run, 3 seeds
python experiments/E03_method_components.py --list           # planned runs and which are done
python experiments/E03_method_components.py --datasets mnist --only GeFL-F "Ours (HWA+LCD+PCM+LA)"
```

Results go to `results/<experiment>/`: `runs.jsonl` (one line per run,
including per-round history and per-class recall), `runs.csv` and
`summary.md` (mean ± std over seeds, paired t-test against the reference).
Runs already in `runs.jsonl` are skipped, so a killed job resumes.

| File | Question | Plan ref |
|---|---|---|
| `E00_paper_parity.py` | Does our GeFL-F match Figure 4 (IID, K=10)? | E0.2, E0.3 |
| `E01_imbalance_gap.py` | How much does imbalance cost GeFL-F? | E1.1 |
| `E02_conditioning_collapse.py` | Does weight decay collapse rare-class conditioning? | §2.1, E1.3, E4.9 |
| `E03_method_components.py` | Each component alone, combinations, baselines | E2.1–E2.3 |
| `E04_client_scaling.py` | Does the gain grow with K (10, 50, 100)? | E3.1 |
| `E05_datasets.py` | SVHN and CIFAR-10 under a long tail | E3.3 |
| `E06_iid_paper_setting.py` | Do-no-harm; can PCM beat GeFL-F in the paper's own setting? | E3.5 |
| `E07_ablations.py` | Justify every hyperparameter of the method | P4 |
| `E08_generators.py` | Is the fix generator-agnostic (DCGAN-F)? | E3.4 |
| `E09_oracles.py` | Which stage loses the tail? | E1.2 |
| `E10_cifar10_track2.py` | Beat GeFL-F on CIFAR-10 in its own setting | Track 2 |
| `E11_privacy_cost.py` | Memorisation and DP class histograms | P5 |

## Protocol (all experiments)

Paper appendix, Tables XIV–XXIII: FE `conv(3)-bn-relu-pool` (CIFAR-10: two
convs, 10 channels) → 3×16×16 features; the ten headers CNN-1…CNN-10;
CVAE-F of Table XX with Adam 1e-3 and weight decay 1e-3; DCGAN-F of Table
XVIII; SGD lr 0.1, momentum 0; T_w/T_g/T_s/T_r = 5/5/1/5; rounds
T_FE/T_KA/T_TN = 20/100/50 (MNIST, FMNIST), 50/100/100 (SVHN), 50/200/100
(CIFAR-10); data fraction 0.1 (0.5 for CIFAR-10), split over K clients;
client→architecture in contiguous blocks; each client keeps its own Adam
state for the generator. Reported:

* `best_mean_acc`: the paper's metric, mean over architectures of each one's best test accuracy over rounds;
* `final_bal`, `final_tail`, `final_worst`: final-round class-balanced, tail-bucket (classes 6–9) and worst-class recall, averaged over architectures. These are the long-tail metrics;
* `oracle_bal`: the same model with each header's last layer re-fit on pooled real features with balanced sampling. It is a diagnostic upper bound, reported for every run;
* `fidelity_tail`: a frozen referee classifier's accuracy on synthetic tail features (generator quality per class);
* `cond_norm_tail_over_head_end`: norm of the tail classes' conditioning rows over the head classes', after stage (ii).

Long-tailed data keeps the paper's total budget (plan §2.2):
`n_c ∝ IF^{c/(C−1)}`, rescaled so Σn_c = frac·N, capped at availability.
Non-IID clients draw class shares from Dirichlet(α=0.5). 200 images per
class are held out from every client, for the referee and the MND reference.

Known deviations, all minor: CIFAR-10 augmentation (crop + flip) is applied
in stage (i) only, because features are precomputed once the FE is frozen.
Balanced runs sample exactly frac·N/C images per class rather than a random
10%. DP noise (E11) is Laplace on each count, added once.

## The method, and why each part exists

The problem: in GeFL-F a class reaches a header only through a client's
real data, or through the shared feature generator G_F(h | y). Under
long-tailed, non-IID data both channels fail for rare classes, for four
separate reasons. Each component below fixes one of them, and each is
derived from that cause rather than tuned.

### 1. HWA: holder-weighted aggregation of conditioning rows (dilution)

The class-r conditioning row w_r gets a data gradient only on clients that
hold class r. Under flat averaging with m_r holders out of K clients, the
non-holders return w_r unchanged (Theorem 1, exact for SGD and for Adam
without decay):

    w_r ← w_r + (m_r / K) · Δ̄_r

so a rare class learns m_r/K times slower than a common one. HWA averages
row r over holders only, weighted by the effective number
E(n) = (1 − β^n)/(1 − β). That is inverse-variance weighting of the
holders' estimates (Proposition 2). It equals flat averaging when every
client holds every class equally (the paper's IID setting, so no harm is
done there). The cost is C integers per client, sent once; E11 measures an
ε-DP version.

### 2. LCD: lazy conditioning decay (collapse)

Table XV gives CVAE-F coupled L2 weight decay λ inside Adam. A client with
no class-r sample in a batch has gradient g_r = λ·w_r, which Adam
normalises to a step of about η·sign(w_r): the row shrinks at full
learning-rate speed whatever λ is. Over a round of S steps, and with flat
averaging, a coordinate of |w_r| can grow by at most

    (m_r/K)·Sη − (1 − m_r/K)·Sη = (2·m_r/K − 1)·Sη          (plan eq. 2.2)

This is negative whenever fewer than half the clients hold class r. The
row collapses toward 0, so the decoder receives no class information and
class-r samples come out class-agnostic. LCD applies the decay to a row
only in steps whose batch contains that class, and freezes absent rows,
in the spirit of LazyAdam. HWA removes the collapse caused by
non-holders; LCD also removes the part caused by holders' own batches that
miss the class. Prediction (E02): GeFL-F's tail rows fall far below its
head rows; HWA, LCD and no-weight-decay each prevent it; DCGAN-F (no weight
decay in the paper) does not collapse.

### 3. PCM: prior-completing mixed batches (new; replaces the sequential synthetic epoch)

GeFL-F trains each header for T_s synthetic epochs, then T_r = 5 real
epochs. Near a local optimum, gradient descent on the real risk contracts
the distance to the real-data optimum geometrically:
θ_t − θ*_real ≈ (I − ηH)^t (θ_syn − θ*_real). The synthetic phase
therefore survives only as an initialisation that is mostly forgotten after
T_r·n/B real steps, and the final header minimises the client's own skewed
risk.

PCM instead minimises one mixed objective. Take a real batch with class
counts b_c (Σ b_c = B). Add s_c synthetic features of class c drawn from
G_F (Σ s_c = S). With a perfect generator, the expected mixed-batch loss is

    E[L_mix] = Σ_c ((b_c + s_c)/(B + S)) · E_{h|c}[ℓ(h, c)]

This is exactly the class-balanced risk R_bal = (1/C) Σ_c E_{h|c}[ℓ] when
b_c + s_c = (B + S)/C for every c. With a limited budget S, the s ≥ 0 that
brings the mixture prior q = (b + s)/(B + S) closest to uniform (in L2)
is, by the KKT conditions, **water-filling**: s_c = [λ − b_c]_+ with λ set
by Σ s_c = S. The remaining imbalance q is removed by logit adjustment with
log q (Menon et al.: cross-entropy on f(h) + log q is Fisher-consistent for
the balanced error when the training prior is q). So **PCM + LA(q) is
consistent for balanced error at any budget S**. It also lowers the
variance of plain logit adjustment, because rare classes get more
effective samples. The default S = 0.2·B matches the paper's synthetic
budget (T_s : T_r = 1 : 5), so no extra compute is used. Everything is
computed from the client's own batch; nothing new is shared.

Where it can fail: with an imperfect generator, the bias is

    Σ_c (s_c/(B + S)) · (E_syn,c − E_c)[ℓ]  ≤  ℓ_max · Σ_c (s_c/(B + S)) · TV(G_F(·|c), p(·|c))

PCM puts the most synthetic mass (the largest s_c) on exactly the classes
where the generator is worst: the tail, whose conditioning was diluted
and collapsed. So PCM *needs* HWA + LCD, and the components are
complementary by construction rather than stacked. That is testable: E03
has +PCM alone, +PCM+LA, +HWA+LCD+PCM and Ours, and predicts the PCM gain
is larger with HWA + LCD than without them.

In the paper's own IID setting, b ≈ uniform, so the completion is uniform
and LA(q) ≈ 0. PCM then reduces to interleaving synthetic features into
every batch instead of a separate, forgotten epoch. E06 and E10 test
whether that alone beats the published GeFL-F numbers.

### 4. LA: logit adjustment by the training prior (residual calibration)

Train on f(h) + τ·log q_k(y), where q_k is the client's own training
prior: its smoothed label counts, or the mixture prior when PCM is on.
At test time the plain f(h) is used. Using the client's own prior rather
than a global one makes each client's update estimate the *balanced* risk
for its own label shift, so the averaged headers are balanced too. It
discloses nothing.

### 5. BCR: server-side generative classifier re-calibration (optional)

After each round, the server re-fits each header's last layer on
class-balanced synthetic features from G_F (a CReFF-style re-training
that needs no client data). Ablated in E03 and E07.

### 6. FSG: federated sufficient-statistics generator (new, E13)

The quick pass showed two things. First, every method sits about 13 pp
below the classifier oracle. At K = 10 each architecture lives on a single
client, so classifier knowledge about rare classes can reach it only
through the generator. Second, the CVAE-F generator models rare classes
poorly: its tail fidelity is 0.15–0.40, even after HWA. PCM failed for the
same reason, because it puts its synthetic mass on exactly those classes.

FSG replaces the learned generator with a class-conditional Gaussian in
the shared FE feature space, h | y ~ N(μ_y, Σ), with a pooled
within-class covariance. Samples are clipped at 0, because FE features
come after a ReLU. Client k uploads, once:

    n_k,c  (counts),   S_k,c = Σ_{h∈c} h  (class sums),   M_k = Σ_h h hᵀ  (second moment)

The server forms the exact pooled estimates

    μ_c = Σ_k S_k,c / Σ_k n_k,c,    W = Σ_k M_k − Σ_c n_c μ_c μ_cᵀ,
    Σ = (1 − γ)·W/(N − C) + γ·(tr/D)·I

with γ = 0.1 shrinkage, because rare classes have n_c ≪ D.

* **No aggregation loss.** Because these are sums, the federated FSG equals
  the centralised one. Theorem 1's dilution and eq. 2.2's collapse cannot
  occur.
* **The right capacity for rare classes.** With n_c ≈ 24, a mean has
  estimation error tr(Σ)/n_c. A 5.5M-parameter generator fitted to the
  same 24 samples through 100 rounds of averaging is a far noisier
  estimator. The pooled Σ borrows second-moment strength from every
  class, as LDA does.
* **Privacy.** The server needs only sums, so it works under secure
  aggregation, and the counts can take DP noise (E13 tests ε = 1). A
  Gaussian cannot reproduce an individual training sample.
* **Communication.** One round of D² + C·D + C numbers (about 600k for
  3×16×16 features), against 5.5M parameters × 100 rounds for CVAE-F:
  about 1000× less. Stage (ii) takes seconds instead of minutes.

HYB keeps CVAE-F for classes with at least 100 samples federation-wide and
uses FSG below that, in case the Gaussian is too crude for common classes.

### Privacy: what each component discloses

| Component | Runs where | New information leaving a client | Under secure aggregation |
|---|---|---|---|
| LA | client | none (uses the client's own label counts locally) | none |
| PCM | client | none (uses the shared generator, already in GeFL-F) | none |
| LCD | client | none (an optimiser change) | none |
| HWA | server | the class histogram E(n_k,·), C numbers | only federation-wide totals Σ_k E(n_k,c) |
| BCR | server | none (uses only the generator) | none |

Two points matter here.

1. **Original GeFL-F already leaks class presence.** A conditioning row of a
   class the client lacks gets no data gradient, so its update is just the
   weight-decay shrinkage, which a server can tell apart from a data
   update. HWA's counts add *how many* samples, on top of *whether*.
2. **HWA works with secure aggregation.** Each client uploads
   E(n_k,c)·w_k,c for the conditioning rows, and the vector E(n_k,c),
   inside the same masked sum as the rest of its update. The server
   recovers Σ_k E(n_k,c)·w_k,c / Σ_k E(n_k,c) without seeing any one
   client's histogram. E11 adds Laplace noise to the histograms (ε-DP) to
   price even the plain version.

**Ours-private = LCD + PCM + LA** sends nothing beyond what GeFL-F
already sends. It runs beside Ours in E03, E04, E05, E06 and E11, so the
privacy price of HWA is measured rather than asserted.

### Federated trade-offs the experiments cover

* **Client count and data scarcity** (E04): K = 10, 50, 100 with the
  paper's fixed total data, so clients get smaller and rarer classes have
  fewer holders.
* **Statistical heterogeneity** (E01): IID, Dirichlet label skew, and long
  tails of increasing severity.
* **The paper's own regime** (E06, E10): a method that only helps under
  imbalance but hurts IID clients would be rejected.
* **Communication**: HWA adds C numbers per client once. Nothing else adds
  traffic. PCM generates S = 0.2·B features per batch, the same synthetic
  budget as GeFL-F's T_s = 1 epoch, so client compute does not change.
* **Model heterogeneity**: all ten architectures are trained and reported.
  At K = 10 each is unique and never averaged; at K = 50 and 100, five or
  ten clients share each one.
* **Memorisation** (E11): feature-space MND, so our generator does not
  memorise more than the baseline's.

**Ours = HWA + LCD + PCM + LA.** The decisive comparison is Ours against
**+LA** alone, the cheap classifier-side fix (FedLC-style). If Ours wins,
the generator-side parts contribute something a classifier-side
correction cannot.
