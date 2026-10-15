# E18_bayesian_csl: Bayesian consensus soft labels (product of label prior and ensemble)

Mode: full

Hypothesis: BCSL >= CSL at equal budget; BCSL makes T_s = 10 pay off where CSL did not (MNIST long tail).

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | p vs +CSL Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours-B (HWA+LA+BCSL) Ts=10 | 3 | 83.26 ± 0.48 | 82.89 ± 0.52 | 83.16 ± 0.95 | 85.06 ± 2.34 | 0.00582 |
| +CSL Ts=10 | 3 | 84.22 ± 0.39 | 84.17 ± 0.43 | 84.61 ± 0.77 | 85.61 ± 1.08 | - |
| +BCSL | 3 | 82.78 ± 0.29 | 82.51 ± 0.35 | 82.86 ± 0.64 | 85.61 ± 1.08 | 0.00225 |
| +BCSL Ts=10 | 3 | 83.16 ± 0.30 | 82.80 ± 0.45 | 83.29 ± 0.91 | 85.61 ± 1.08 | 0.00403 |

| label | seed | best_mean_acc | final_bal | final_tail | fidelity_tail |
|---|---|---|---|---|---|
| Ours-B (HWA+LA+BCSL) Ts=10 | 0 | 83.56 | 83.20 | 84.17 | 85.58 |
| Ours-B (HWA+LA+BCSL) Ts=10 | 1 | 82.71 | 82.30 | 82.28 | 87.08 |
| Ours-B (HWA+LA+BCSL) Ts=10 | 2 | 83.52 | 83.19 | 83.02 | 82.50 |
| +CSL Ts=10 | 0 | 84.53 | 84.51 | 85.45 | 85.67 |
| +CSL Ts=10 | 1 | 83.78 | 83.69 | 83.95 | 86.67 |
| +CSL Ts=10 | 2 | 84.35 | 84.30 | 84.44 | 84.50 |
| +BCSL | 0 | 82.96 | 82.67 | 83.53 | 85.67 |
| +BCSL | 1 | 82.44 | 82.11 | 82.26 | 86.67 |
| +BCSL | 2 | 82.93 | 82.76 | 82.78 | 84.50 |
| +BCSL Ts=10 | 0 | 83.34 | 83.16 | 84.22 | 85.67 |
| +BCSL Ts=10 | 1 | 82.82 | 82.30 | 82.41 | 86.67 |
| +BCSL Ts=10 | 2 | 83.33 | 82.96 | 83.23 | 84.50 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | p vs +CSL Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 90.18 ± 0.96 | 89.74 ± 1.04 | 82.16 ± 3.00 | 28.69 ± 4.47 | - |
| Ours (HWA+LA+CSL) Ts=10 | 3 | 91.70 ± 1.35 | 90.51 ± 1.61 | 84.37 ± 3.37 | 28.69 ± 4.47 | - |
| Ours-B (HWA+LA+BCSL) | 3 | 89.33 ± 1.44 | 88.68 ± 1.67 | 81.16 ± 4.20 | 28.69 ± 4.47 | - |
| Ours-B (HWA+LA+BCSL) Ts=10 | 3 | 89.92 ± 1.35 | 87.32 ± 1.80 | 80.12 ± 3.92 | 28.69 ± 4.47 | - |

| label | seed | best_mean_acc | final_bal | final_tail | fidelity_tail |
|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 0 | 89.08 | 88.58 | 79.01 | 26.58 |
| Ours (HWA+LA+CSL) | 1 | 90.85 | 90.58 | 84.98 | 25.67 |
| Ours (HWA+LA+CSL) | 2 | 90.61 | 90.06 | 82.50 | 33.83 |
| Ours (HWA+LA+CSL) Ts=10 | 0 | 90.37 | 88.92 | 80.58 | 26.58 |
| Ours (HWA+LA+CSL) Ts=10 | 1 | 91.65 | 90.45 | 85.50 | 25.67 |
| Ours (HWA+LA+CSL) Ts=10 | 2 | 93.07 | 92.15 | 87.03 | 33.83 |
| Ours-B (HWA+LA+BCSL) | 0 | 87.68 | 86.78 | 76.60 | 26.58 |
| Ours-B (HWA+LA+BCSL) | 1 | 90.35 | 89.93 | 84.85 | 25.67 |
| Ours-B (HWA+LA+BCSL) | 2 | 89.97 | 89.33 | 82.03 | 33.83 |
| Ours-B (HWA+LA+BCSL) Ts=10 | 0 | 88.43 | 85.69 | 75.80 | 26.58 |
| Ours-B (HWA+LA+BCSL) Ts=10 | 1 | 90.28 | 87.01 | 81.09 | 25.67 |
| Ours-B (HWA+LA+BCSL) Ts=10 | 2 | 91.05 | 89.25 | 83.46 | 33.83 |

