# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 46.06 ± 1.29 | 37.73 ± 1.67 | 46.78 ± 1.20 | 48.71 ± 0.96 | 47.93 ± 2.05 | 52.25 ± 0.74 | 41.39 ± 4.52 | - |
| Ours-A Ts=10 | 3 | 44.38 ± 1.07 | 34.24 ± 2.34 | 48.07 ± 0.96 | 47.26 ± 1.23 | 46.06 ± 4.48 | 50.64 ± 0.85 | 40.44 ± 3.26 | 0.0277 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 45.97 | 37.56 | 46.31 | 48.57 | 48.25 | 52.36 | 36.17 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 44.81 | 36.15 | 45.88 | 47.83 | 45.74 | 51.46 | 44.00 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 47.39 | 39.49 | 48.14 | 49.72 | 49.79 | 52.93 | 44.00 |
| Ours-A Ts=10 | 0 | 44.77 | 34.94 | 48.20 | 48.13 | 47.55 | 51.11 | 36.83 |
| Ours-A Ts=10 | 1 | 43.17 | 31.64 | 47.05 | 45.85 | 41.02 | 49.66 | 41.33 |
| Ours-A Ts=10 | 2 | 45.21 | 36.16 | 48.96 | 47.81 | 49.61 | 51.15 | 43.17 |

