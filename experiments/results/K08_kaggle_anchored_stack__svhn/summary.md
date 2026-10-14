# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## svhn  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 55.89 ± 1.20 | 37.10 ± 3.60 | 61.62 ± 0.90 | 0.57 ± 0.02 | 0.41 ± 0.05 | 61.77 ± 1.19 | 24.28 ± 5.34 | - |
| Ours-A Ts=10 | 3 | 54.41 ± 1.13 | 35.51 ± 3.12 | 62.30 ± 0.39 | 0.55 ± 0.01 | 0.36 ± 0.03 | 59.13 ± 1.29 | 25.94 ± 5.53 | 0.102 |

## svhn  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 76.26%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 72.52 ± 0.96 | 71.05 ± 1.06 | 73.86 ± 0.60 | 0.72 ± 0.01 | 0.71 ± 0.01 | 73.81 ± 0.67 | 56.33 ± 3.34 | - |
| Ours-A Ts=10 | 3 | 71.82 ± 1.45 | 70.98 ± 1.72 | 74.38 ± 0.79 | 0.72 ± 0.02 | 0.71 ± 0.02 | 73.24 ± 0.98 | 56.72 ± 0.80 | 0.136 |

