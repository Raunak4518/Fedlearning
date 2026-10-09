# E16_csl_synthetic_budget: Consensus labels and the synthetic budget T_s

Mode: full

Hypothesis: CSL gains from T_s = 3 / 5; GeFL-F does not (paper Fig. 10).

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 2 | 77.55 ± 0.98 | 76.98 ± 0.75 | 81.92 ± 0.48 | 81.17 ± 0.73 | 0.076 |
| Ours (HWA+LA+CSL) Ts=5 | 2 | 79.13 ± 0.70 | 78.76 ± 0.89 | 82.66 ± 0.82 | 81.73 ± 1.02 | 0.0614 |
| Ours (HWA+LA+CSL) Ts=10 | 2 | 79.64 ± 0.59 | 79.33 ± 0.67 | 82.87 ± 0.65 | 81.85 ± 1.10 | 0.0569 |
| GeFL-F | 2 | 62.24 ± 1.61 | 61.33 ± 1.53 | 73.45 ± 0.60 | 79.63 ± 0.36 | - |

