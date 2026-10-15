# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## svhn  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 55.89 ± 1.20 | 37.10 ± 3.60 | 61.62 ± 0.90 | 57.19 ± 1.56 | 41.23 ± 4.74 | 61.77 ± 1.19 | 24.28 ± 5.34 | - |
| Ours-A Ts=10 | 3 | 54.41 ± 1.13 | 35.51 ± 3.12 | 62.30 ± 0.39 | 54.86 ± 1.42 | 36.11 ± 2.55 | 59.13 ± 1.29 | 25.94 ± 5.53 | 0.102 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 55.06 | 35.89 | 61.04 | 56.33 | 39.39 | 60.61 | 23.17 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 55.34 | 34.25 | 61.16 | 56.25 | 37.69 | 61.72 | 19.58 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 57.26 | 41.15 | 62.65 | 58.98 | 46.61 | 62.98 | 30.08 |
| Ours-A Ts=10 | 0 | 54.59 | 35.59 | 62.33 | 54.69 | 35.84 | 58.36 | 23.67 |
| Ours-A Ts=10 | 1 | 53.21 | 32.35 | 61.90 | 53.53 | 33.71 | 58.41 | 21.92 |
| Ours-A Ts=10 | 2 | 55.45 | 38.59 | 62.68 | 56.36 | 38.78 | 60.62 | 32.25 |

## svhn  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 76.26%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 72.52 ± 0.96 | 71.05 ± 1.06 | 73.86 ± 0.60 | 72.45 ± 0.96 | 71.02 ± 1.43 | 73.81 ± 0.67 | 56.33 ± 3.34 | - |
| Ours-A Ts=10 | 3 | 71.82 ± 1.45 | 70.98 ± 1.72 | 74.38 ± 0.79 | 71.67 ± 1.58 | 70.89 ± 2.13 | 73.24 ± 0.98 | 56.72 ± 0.80 | 0.136 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 73.01 | 71.76 | 73.92 | 72.82 | 70.86 | 74.10 | 59.75 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 73.13 | 71.56 | 74.42 | 73.17 | 72.52 | 74.28 | 56.17 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 71.42 | 69.83 | 73.23 | 71.37 | 69.68 | 73.03 | 53.08 |
| Ours-A Ts=10 | 0 | 72.54 | 72.03 | 74.84 | 72.31 | 71.41 | 73.81 | 57.58 |
| Ours-A Ts=10 | 1 | 72.78 | 71.91 | 74.83 | 72.83 | 72.71 | 73.80 | 56.58 |
| Ours-A Ts=10 | 2 | 70.15 | 68.99 | 73.47 | 69.87 | 68.55 | 72.11 | 56.00 |

