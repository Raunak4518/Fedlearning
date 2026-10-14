# T03 pre-registration (written before any T03 run; do not edit after results exist)

Date: 2026-10-14. Code: experiments/T03_exact_readout.py (isolated; core.py and all reported runs unchanged).

## Hypothesis

H-R (readout bias). In GeFL-F and its variants, a head g = readout ∘ embedding is trained on a mixture of
its client's real features and synthetic features. For class c, the readout sees the class through the
mixture mean  m̂_gc = α_kc μ^R_kc + (1 − α_kc) μ^S_gc  (embedding space of head g), where
α_kc = T_r n_kc / (T_r n_kc + T_s n_k / C) is the share of real evidence for class c on client k.
The readout's class-c error relative to the pooled-real readout therefore has a synthetic-bias term
(1 − α_kc) Δ_gc, Δ_gc = μ^S_gc − μ_gc (synthetic minus pooled-real embedding mean). Prior corrections
(LA, BBC) shift logit offsets only and cannot remove a mean (direction) error. A readout fit to exact
federated statistics of REAL embeddings (class sums and pooled second moments, computed by every client
for every architecture) removes the term.

## Primary metrics and predictions

Runs: FashionMNIST, IF = 100, Dirichlet(0.5), K = 10, seed 0, reduced schedule (quick_scale 0.4),
methods GeFL-F, +HWA+LA (minimal), Ours-A (T_s = 1). Optional: the same on MNIST.

P1  ER-LDA (closed-form LDA readout from exact sums, uniform prior, covariance shrinkage γ = 0.1 fixed
    a priori) is ≥ head − 0.5 balanced-accuracy points in every run, and within 3 points of the oracle
    (class-balanced CE refit of the fc on pooled real embeddings).
P2  For GeFL-F, ER-LDA gains ≥ +10 points over the head (the oracle gap measured in F10/E20 is 16-20).
P3  Mechanism: over the 100 (head, class) pairs of each run with LA, Spearman(B_gc, ΔR_gc) ≥ 0.3, where
    B_gc = (1 − α_kc) · ||Δ_gc||_{Σ_g^{-1}} and ΔR_gc = recall_oracle − recall_head for head g, class c;
    and B beats both single factors (1 − α_kc) alone and ||Δ_gc|| alone.
P4  BBC (prior-only server correction) stays below the oracle in the LA runs.

## Falsification

* ER-LDA more than 3 points below the oracle in the LA runs: the closed-form federated route does not
  reach the readout bound (the gap is not a first/second-moment effect).
* Spearman(B, ΔR) < 0.1, or B no better than (1 − α) alone: the synthetic-mean mechanism does not explain
  the readout gap (it is prior bias or something else).

## Competing explanation

The readout gap is pure class-prior bias. Then BBC would reach the oracle, and (1 − α) alone (a prior
proxy) would predict ΔR as well as B.

No test-set tuning: γ = 0.1 and all thresholds above are fixed here.
