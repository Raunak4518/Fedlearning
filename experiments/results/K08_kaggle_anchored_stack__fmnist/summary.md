# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## fmnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 76.20 ± 1.12 | 66.74 ± 1.97 | 76.57 ± 0.80 | 0.78 ± 0.01 | 0.71 ± 0.03 | 80.17 ± 0.70 | 76.89 ± 1.18 | - |
| Ours-A Ts=10 | 3 | 79.02 ± 0.98 | 70.59 ± 1.24 | 79.25 ± 0.92 | 0.79 ± 0.01 | 0.73 ± 0.02 | 81.03 ± 0.76 | 76.97 ± 0.84 | 0.00176 |

## fmnist  IF=0.01  alpha=0.5  K=50

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 77.24 ± 0.45 | 69.19 ± 0.87 | 77.68 ± 0.14 | 0.79 ± 0.00 | 0.74 ± 0.00 | 80.55 ± 0.07 | 77.47 ± 2.98 | - |
| Ours-A Ts=10 | 3 | 79.47 ± 0.38 | 71.69 ± 0.52 | 79.81 ± 0.24 | 0.80 ± 0.00 | 0.74 ± 0.00 | 81.25 ± 0.31 | 77.14 ± 3.74 | 0.0155 |

## fmnist  IF=1.0  alpha=None  K=100
Paper Figure 4 GeFL-F (CVAE-F): 81.65%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 81.18 ± 0.07 | 81.36 ± 0.74 | 81.21 ± 0.07 | 0.81 ± 0.00 | 0.81 ± 0.01 | 82.63 ± 0.18 | 82.42 ± 2.38 | - |
| Ours-A Ts=10 | 3 | 81.82 ± 0.56 | 82.14 ± 0.60 | 82.04 ± 0.52 | 0.82 ± 0.01 | 0.82 ± 0.01 | 83.09 ± 0.34 | 81.97 ± 2.14 | 0.156 |

