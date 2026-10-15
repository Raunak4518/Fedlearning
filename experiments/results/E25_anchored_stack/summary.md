# E25_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC) with LA + CSL

Mode: full

Hypothesis: beats every earlier variant under the long tail (3 seeds), no IID loss; T_s = 10 adds on top.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 78.22 ± 0.69 | 72.29 ± 0.26 | 78.60 ± 0.55 | 79.80 ± 0.70 | 76.33 ± 1.06 | 81.19 ± 0.45 | - |
| Ours-A Ts=10 | 3 | 79.67 ± 0.54 | 73.56 ± 1.13 | 80.01 ± 0.54 | 80.55 ± 0.65 | 75.11 ± 1.28 | 81.55 ± 0.76 | 0.00392 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal |
|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 77.49 | 72.34 | 77.97 | 79.08 | 75.70 | 80.73 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 78.87 | 72.53 | 79.00 | 80.49 | 77.55 | 81.63 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 78.30 | 72.02 | 78.81 | 79.84 | 75.74 | 81.20 |
| Ours-A Ts=10 | 0 | 79.12 | 72.51 | 79.40 | 79.84 | 74.84 | 80.67 |
| Ours-A Ts=10 | 1 | 80.21 | 74.77 | 80.43 | 81.11 | 76.51 | 82.04 |
| Ours-A Ts=10 | 2 | 79.69 | 73.41 | 80.19 | 80.71 | 73.98 | 81.95 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 83.24 ± 0.24 | 83.41 ± 0.51 | 83.35 ± 0.22 | 83.03 ± 0.14 | 82.92 ± 1.08 | 83.87 ± 0.29 | - |
| Ours-A Ts=10 | 3 | 83.81 ± 0.42 | 84.18 ± 0.58 | 83.96 ± 0.39 | 83.79 ± 0.35 | 83.52 ± 0.98 | 84.09 ± 0.42 | 0.0368 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal |
|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 83.20 | 83.97 | 83.38 | 82.98 | 84.14 | 83.79 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 83.03 | 83.27 | 83.11 | 82.92 | 82.10 | 83.63 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 83.50 | 82.99 | 83.55 | 83.18 | 82.51 | 84.19 |
| Ours-A Ts=10 | 0 | 83.62 | 84.85 | 83.80 | 83.61 | 84.62 | 83.98 |
| Ours-A Ts=10 | 1 | 83.51 | 83.77 | 83.67 | 83.56 | 82.75 | 83.74 |
| Ours-A Ts=10 | 2 | 84.28 | 83.92 | 84.41 | 84.19 | 83.21 | 84.55 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 93.11 ± 0.42 | 89.15 ± 0.58 | 93.24 ± 0.40 | 93.67 ± 0.36 | 90.53 ± 0.47 | 94.07 ± 0.30 | - |
| Ours-A Ts=10 | 3 | 94.82 ± 0.57 | 91.08 ± 1.39 | 95.09 ± 0.50 | 95.17 ± 0.55 | 92.06 ± 1.09 | 94.93 ± 0.70 | 0.042 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal |
|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 92.96 | 89.26 | 93.18 | 93.57 | 90.55 | 94.11 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 93.59 | 89.66 | 93.67 | 94.07 | 90.99 | 94.35 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 92.79 | 88.52 | 92.88 | 93.37 | 90.04 | 93.75 |
| Ours-A Ts=10 | 0 | 95.38 | 92.52 | 95.47 | 95.69 | 93.18 | 95.58 |
| Ours-A Ts=10 | 1 | 94.84 | 90.96 | 95.28 | 95.23 | 92.01 | 95.03 |
| Ours-A Ts=10 | 2 | 94.23 | 89.75 | 94.52 | 94.59 | 90.99 | 94.19 |

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 96.65 ± 0.01 | 95.46 ± 0.06 | 96.67 ± 0.01 | 96.67 ± 0.04 | 95.48 ± 0.17 | 97.02 ± 0.05 | - |
| Ours-A Ts=10 | 3 | 97.58 ± 0.09 | 96.66 ± 0.12 | 97.62 ± 0.06 | 97.61 ± 0.08 | 96.75 ± 0.12 | 97.78 ± 0.05 | 0.00387 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal |
|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 96.66 | 95.53 | 96.68 | 96.71 | 95.65 | 97.04 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 96.65 | 95.42 | 96.67 | 96.68 | 95.47 | 96.96 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 96.64 | 95.43 | 96.66 | 96.63 | 95.31 | 97.06 |
| Ours-A Ts=10 | 0 | 97.48 | 96.52 | 97.56 | 97.52 | 96.62 | 97.72 |
| Ours-A Ts=10 | 1 | 97.66 | 96.77 | 97.67 | 97.67 | 96.86 | 97.78 |
| Ours-A Ts=10 | 2 | 97.61 | 96.70 | 97.62 | 97.65 | 96.76 | 97.82 |

