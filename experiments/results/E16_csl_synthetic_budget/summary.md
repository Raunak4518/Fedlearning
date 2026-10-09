# E16_csl_synthetic_budget: Consensus labels and the synthetic budget T_s

Mode: full

Hypothesis: CSL gains from T_s = 3 / 5; GeFL-F does not (paper Fig. 10).

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 77.54 ± 0.69 | 76.94 ± 0.54 | 81.90 ± 0.34 | 81.09 ± 0.54 | 0.0191 |
| Ours (HWA+LA+CSL) Ts=5 | 3 | 79.05 ± 0.51 | 78.76 ± 0.63 | 82.52 ± 0.63 | 81.60 ± 0.76 | 0.0148 |
| Ours (HWA+LA+CSL) Ts=10 | 3 | 79.56 ± 0.44 | 79.23 ± 0.51 | 82.79 ± 0.48 | 81.75 ± 0.80 | 0.0139 |
| GeFL-F | 3 | 60.02 ± 4.02 | 59.13 ± 3.96 | 73.04 ± 0.84 | 79.42 ± 0.44 | - |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 90.57 ± 0.79 | 90.20 ± 0.98 | 94.39 ± 0.45 | 93.53 ± 0.11 | 0.00355 |
| Ours (HWA+LA+CSL) Ts=5 | 3 | 91.61 ± 0.82 | 90.78 ± 1.09 | 93.32 ± 0.46 | 93.38 ± 0.26 | 0.00195 |
| Ours (HWA+LA+CSL) Ts=10 | 3 | 91.52 ± 0.83 | 90.15 ± 1.35 | 92.82 ± 0.86 | 92.89 ± 0.19 | 0.00151 |
| GeFL-F | 3 | 75.69 ± 1.64 | 74.94 ± 2.05 | 85.26 ± 2.10 | 91.00 ± 0.34 | - |

