# K09 ablation (Kaggle, final): analysis

Source: `K09_kaggle_ablation_final__mnist/` and `__fmnist/` (`runs.jsonl`, 18 runs each:
6 arms x 3 seeds; long tail IF = 100, Dirichlet(0.5), K = 10, T_s = 10).
Differences are **paired by seed** (arm minus Ours-A, percentage points). The head-component
arms (-LA, -CSL, -MC, -KH) share Ours-A's FE and generator per seed, so only the head stage differs.

Note: each run's `runs.csv` lacks the three `-PC` rows (each worker overwrites `runs.csv`
with its own rows; `-PC` ran on the other worker). `runs.jsonl` and `summary.md` are complete.

## Balanced accuracy (final_bal), removal minus Ours-A

| removed | MNIST mean [min, max] | p (paired t) | FMNIST mean [min, max] | p (paired t) |
|---|---|---|---|---|
| PC (-> CVAE+HWA) | **-4.72** [-5.97, -3.96] | 0.018 | -0.67 [-0.98, -0.20] | 0.106 |
| LA  | **-2.98** [-3.55, -2.26] | 0.016 | **-6.18** [-7.30, -4.66] | 0.016 |
| CSL | -1.63 [-2.50, -0.57] | 0.102 | **-1.38** [-1.58, -1.14] | 0.008 |
| MC  | -0.96 [-1.28, -0.74] | 0.028 | -0.49 [-0.76, -0.18] | 0.100 |
| KH  | -0.71 [-0.95, -0.36] | 0.059 | -0.30 [-0.49, -0.17] | 0.097 |

Ours-A: MNIST 95.18 +- 0.08, FMNIST 79.79 +- 0.31.

## Findings

1. **Direction holds everywhere.** All 30 seed-level differences are negative (every arm,
   every seed, both datasets). Two-sided sign test on the 30: p = 2e-9. The stated hypothesis
   ("removing any one component lowers balanced accuracy") is supported as a whole.
2. **Per-component significance does not survive correction.** With n = 3 (df = 2), no single
   arm passes Holm over the 10 tests (smallest adjusted p = 0.085, -CSL on FMNIST). A per-component
   sign test is 6/6 negative for every component (p = 0.031 each), also not Holm-significant across 5.
   Five seeds per dataset (10/10 negative -> p = 0.002) would be enough for the small effects.
3. **Three tiers of effect size.**
   - Large: LA (3-6 pts; also the largest worst-class loss, -15.7 / -11.3), PC on MNIST (4.7).
   - Medium: CSL (1.4-1.6 on both).
   - Small: MC (0.5-1.0), KH (0.3-0.7). Report these as consistent but small.
4. **LA overlaps with BBC.** Without LA, post-hoc balanced bias calibration recovers nearly all of
   it: bbc_bal drops only 0.26 (MNIST) and 0.53 (FMNIST; 3/3 seeds negative, p = 0.002). If the
   reported method applies gated BBC (it is applied here: global imbalance 100 > 2), LA's
   marginal contribution on top of BBC is ~0.3-0.5 pts, not 3-6. Expect a reviewer to ask "why LA
   if BBC does the same?". Say which number the paper reports (final_bal or bbc_bal) and ablate on that one.
5. **PC is dataset-dependent.** Large on MNIST (-4.7 bal, -8.3 tail); small on FMNIST (-0.67 bal,
   -2.6 tail), where the CVAE+HWA arm even has a better worst class (+1.3) and the same
   ensemble accuracy (-0.05). Under BBC the FMNIST gap widens to -1.5.
6. **MC works through spread, not referee fidelity.** Removing MC drops tail sample spread
   0.93 -> 0.59 (MNIST) / 0.96 -> 0.63 (FMNIST), but the referee's tail fidelity is unchanged or
   higher (FMNIST 83.9 vs 79.2). This is consistent with MC restoring the exact class spread rather than
   making samples more classifiable.

## Suggested follow-ups

- Re-run -MC, -KH and -PC (FMNIST) with seeds 3-4 (the head arms reuse the cached generator, so
  this is cheap) to make the small effects significant per component.
- Decide final_bal vs bbc_bal as the headline metric; if bbc_bal, present LA as a training-time
  alternative to BBC rather than an additive gain.
- Fix the `runs.csv` overwrite in the experiment template (write the CSV from the merged `runs.jsonl`).
