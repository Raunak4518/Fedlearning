# T03b pre-registration (revised hypothesis after T03; written before any T03b run)

## What T03 established
* ER-LDA closes the readout gap (P1, P2 pass); BBC stays below the oracle (P4 pass).
* P3 failed: the synthetic class-mean mismatch Δ does not explain the per-class readout gap; the real-evidence
  share (1 − α) does (ρ 0.33-0.90). Post hoc: the gap sits on classes the client does not hold
  (FMNIST Ours-A: −17 pts recall on unheld classes, −7 held-rare, ≈0 held > 20).

## Revised hypothesis H-R′ (readout scale, not location)
A head's readout weight for class c, w_gc, is trained on real data only where its client holds c; unheld classes
get only the synthetic phase. Their weight norms are therefore mis-scaled relative to held classes. A norm error
changes the logit by a factor that depends on the input (‖w_c‖‖e‖cos θ), so no logit offset (LA, BBC) removes it
(Lemma R1 generalised: offsets reach only common shifts of w·e + b in the bias term).

## Predictions (FMNIST LT, K = 10, seed 0, quick_scale 0.4; GeFL-F, +HWA+LA, Ours-A T_s = 1)
Q1  Per (head, class): Spearman(norm deficit, dR) ≥ 0.3, where norm deficit = 1 − ‖w_gc‖ / median_c ‖w_gc‖.
Q2  τ-normalisation (Kang et al. 2020: w_c ← w_c / ‖w_c‖, bias kept) recovers ≥ 50 % of the readout gap
    (oracle − head) in at least 2 of the 3 runs.
Q3  ER-LDA ≥ τ-norm in every run (exact statistics fix more than the scale).

## Falsification
Q1 ρ < 0.1 and Q2 < 25 % of the gap in all runs: the readout error is not a norm (scale) effect.

Fixed in advance: τ = 1 (full normalisation), no tuning on the test set.
