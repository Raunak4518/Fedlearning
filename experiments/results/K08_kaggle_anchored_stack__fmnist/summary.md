# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## fmnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 76.20 ± 1.12 | 66.74 ± 1.97 | 76.57 ± 0.80 | 77.58 ± 1.05 | 71.03 ± 3.04 | 80.17 ± 0.70 | 76.89 ± 1.18 | - |
| Ours-A Ts=10 | 3 | 79.02 ± 0.98 | 70.59 ± 1.24 | 79.25 ± 0.92 | 79.35 ± 1.00 | 73.04 ± 1.50 | 81.03 ± 0.76 | 76.97 ± 0.84 | 0.00176 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 76.79 | 67.33 | 77.09 | 78.23 | 72.89 | 80.53 | 76.17 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 74.91 | 64.54 | 75.65 | 76.36 | 67.52 | 79.36 | 78.25 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 76.91 | 68.35 | 76.96 | 78.13 | 72.68 | 80.61 | 76.25 |
| Ours-A Ts=10 | 0 | 79.38 | 71.19 | 79.62 | 79.55 | 73.63 | 81.15 | 77.75 |
| Ours-A Ts=10 | 1 | 77.91 | 69.16 | 78.21 | 78.27 | 71.33 | 80.22 | 77.08 |
| Ours-A Ts=10 | 2 | 79.76 | 71.42 | 79.93 | 80.23 | 74.16 | 81.73 | 76.08 |

## fmnist  IF=0.01  alpha=0.5  K=50

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 77.24 ± 0.45 | 69.19 ± 0.87 | 77.68 ± 0.14 | 78.56 ± 0.19 | 73.54 ± 0.29 | 80.55 ± 0.07 | 77.47 ± 2.98 | - |
| Ours-A Ts=10 | 3 | 79.47 ± 0.38 | 71.69 ± 0.52 | 79.81 ± 0.24 | 79.92 ± 0.36 | 73.69 ± 0.21 | 81.25 ± 0.31 | 77.14 ± 3.74 | 0.0155 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 76.77 | 68.29 | 77.77 | 78.59 | 73.72 | 80.48 | 74.92 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 77.66 | 69.26 | 77.75 | 78.36 | 73.20 | 80.63 | 80.75 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 77.30 | 70.03 | 77.52 | 78.74 | 73.70 | 80.53 | 76.75 |
| Ours-A Ts=10 | 0 | 79.17 | 71.10 | 79.59 | 79.75 | 73.48 | 81.03 | 74.50 |
| Ours-A Ts=10 | 1 | 79.34 | 71.99 | 79.77 | 79.68 | 73.90 | 81.10 | 81.42 |
| Ours-A Ts=10 | 2 | 79.90 | 72.00 | 80.07 | 80.33 | 73.68 | 81.60 | 75.50 |

## fmnist  IF=1.0  alpha=None  K=100
Paper Figure 4 GeFL-F (CVAE-F): 81.65%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 81.18 ± 0.07 | 81.36 ± 0.74 | 81.21 ± 0.07 | 80.75 ± 0.00 | 80.83 ± 1.39 | 82.63 ± 0.18 | 82.42 ± 2.38 | - |
| Ours-A Ts=10 | 3 | 81.82 ± 0.56 | 82.14 ± 0.60 | 82.04 ± 0.52 | 81.54 ± 0.54 | 82.13 ± 0.89 | 83.09 ± 0.34 | 81.97 ± 2.14 | 0.156 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 81.13 | 82.20 | 81.18 | 80.76 | 82.33 | 82.80 | 80.17 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 81.14 | 81.04 | 81.16 | 80.75 | 80.60 | 82.45 | 84.92 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 81.26 | 80.83 | 81.28 | 80.75 | 79.57 | 82.64 | 82.17 |
| Ours-A Ts=10 | 0 | 81.32 | 82.74 | 81.62 | 81.05 | 83.06 | 82.73 | 80.17 |
| Ours-A Ts=10 | 1 | 81.69 | 82.15 | 81.88 | 81.45 | 82.06 | 83.14 | 84.33 |
| Ours-A Ts=10 | 2 | 82.43 | 81.54 | 82.63 | 82.12 | 81.28 | 83.40 | 81.42 |

