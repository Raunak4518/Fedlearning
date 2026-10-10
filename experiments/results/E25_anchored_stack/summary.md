# E25_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC) with LA + CSL

Mode: full

Hypothesis: beats every earlier variant under the long tail (3 seeds), no IID loss; T_s = 10 adds on top.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 78.22 ± 0.69 | 72.29 ± 0.26 | 78.60 ± 0.55 | 0.80 ± 0.01 | 0.76 ± 0.01 | 81.19 ± 0.45 | - |
| Ours-A Ts=10 | 3 | 79.67 ± 0.54 | 73.56 ± 1.13 | 80.01 ± 0.54 | 0.81 ± 0.01 | 0.75 ± 0.01 | 81.55 ± 0.76 | 0.00392 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 83.24 ± 0.24 | 83.41 ± 0.51 | 83.35 ± 0.22 | 0.83 ± 0.00 | 0.83 ± 0.01 | 83.87 ± 0.29 | - |
| Ours-A Ts=10 | 3 | 83.81 ± 0.42 | 84.18 ± 0.58 | 83.96 ± 0.39 | 0.84 ± 0.00 | 0.84 ± 0.01 | 84.09 ± 0.42 | 0.0368 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 93.11 ± 0.42 | 89.15 ± 0.58 | 93.24 ± 0.40 | 0.94 ± 0.00 | 0.91 ± 0.00 | 94.07 ± 0.30 | - |
| Ours-A Ts=10 | 3 | 94.82 ± 0.57 | 91.08 ± 1.39 | 95.09 ± 0.50 | 0.95 ± 0.01 | 0.92 ± 0.01 | 94.93 ± 0.70 | 0.042 |

