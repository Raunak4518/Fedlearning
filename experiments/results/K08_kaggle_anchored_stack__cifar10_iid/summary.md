# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## cifar10  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 55.86%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 58.62 ± 1.43 | 67.30 ± 1.97 | 60.39 ± 0.84 | 0.58 ± 0.02 | 0.67 ± 0.02 | 59.12 ± 1.12 | 46.14 ± 6.53 | - |
| Ours-A Ts=10 | 3 | 56.82 ± 2.16 | 65.47 ± 2.77 | 60.93 ± 1.27 | 0.56 ± 0.02 | 0.65 ± 0.03 | 57.92 ± 1.74 | 44.89 ± 5.28 | 0.0514 |

