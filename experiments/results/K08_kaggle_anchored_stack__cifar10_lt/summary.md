# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100

Mode: full

Hypothesis: Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 46.06 ± 1.29 | 37.73 ± 1.67 | 46.78 ± 1.20 | 0.49 ± 0.01 | 0.48 ± 0.02 | 52.25 ± 0.74 | 41.39 ± 4.52 | - |
| Ours-A Ts=10 | 3 | 44.38 ± 1.07 | 34.24 ± 2.34 | 48.07 ± 0.96 | 0.47 ± 0.01 | 0.46 ± 0.04 | 50.64 ± 0.85 | 40.44 ± 3.26 | 0.0277 |

