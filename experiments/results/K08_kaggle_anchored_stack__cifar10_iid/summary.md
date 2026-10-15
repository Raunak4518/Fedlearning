# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## cifar10  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 55.86%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 58.62 ± 1.43 | 67.30 ± 1.97 | 60.39 ± 0.84 | 58.47 ± 1.51 | 66.84 ± 2.44 | 59.12 ± 1.12 | 46.14 ± 6.53 | - |
| Ours-A Ts=10 | 3 | 56.82 ± 2.16 | 65.47 ± 2.77 | 60.93 ± 1.27 | 56.40 ± 2.12 | 65.08 ± 3.47 | 57.92 ± 1.74 | 44.89 ± 5.28 | 0.0514 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 58.71 | 67.92 | 60.06 | 58.70 | 68.17 | 59.10 | 53.25 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 57.15 | 65.10 | 59.76 | 56.85 | 64.02 | 58.02 | 40.42 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 59.99 | 68.89 | 61.34 | 59.85 | 68.33 | 60.25 | 44.75 |
| Ours-A Ts=10 | 0 | 56.98 | 66.49 | 60.80 | 56.85 | 67.27 | 58.04 | 50.58 |
| Ours-A Ts=10 | 1 | 54.58 | 62.34 | 59.74 | 54.10 | 61.08 | 56.13 | 40.17 |
| Ours-A Ts=10 | 2 | 58.89 | 67.59 | 62.26 | 58.26 | 66.89 | 59.60 | 43.92 |

