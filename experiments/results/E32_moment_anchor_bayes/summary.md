# E32_moment_anchor_bayes: MC-B: exact covariance where the class's data determine it, the generator's shape where not

Mode: full

Hypothesis: MC-B >= max(MC, MC-S) in every setting: MNIST LT >= 94.8, FMNIST LT >= 79.7, FMNIST IID >= 85.2.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh | p vs Ours-A-B Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 3 | 81.57 ± 0.31 | 74.54 ± 0.82 | 33.19 ± 4.02 | 81.75 ± 0.27 | 81.96 ± 0.30 | 80.08 ± 1.91 | 0.98 ± 0.00 | - |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 0 | 81.51 | 74.93 | 37.83 | 81.68 | 82.09 | 78.42 | 0.98 |
| Ours-A-B Ts=10 | 1 | 81.90 | 75.09 | 30.68 | 82.05 | 82.18 | 82.17 | 0.98 |
| Ours-A-B Ts=10 | 2 | 81.29 | 73.61 | 31.07 | 81.53 | 81.62 | 79.67 | 0.99 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh | p vs Ours-A-B Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 3 | 85.60 ± 0.49 | 85.79 ± 0.38 | 60.21 ± 0.76 | 85.75 ± 0.45 | 85.56 ± 0.54 | 83.53 ± 1.46 | 0.99 ± 0.01 | - |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 0 | 85.33 | 85.86 | 60.51 | 85.54 | 85.26 | 82.00 | 1.00 |
| Ours-A-B Ts=10 | 1 | 85.32 | 85.38 | 59.34 | 85.44 | 85.24 | 84.92 | 0.99 |
| Ours-A-B Ts=10 | 2 | 86.17 | 86.12 | 60.77 | 86.26 | 86.19 | 83.67 | 0.99 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh | p vs Ours-A-B Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 3 | 94.76 ± 1.95 | 90.66 ± 3.72 | 86.08 ± 6.74 | 94.90 ± 1.81 | 94.99 ± 1.84 | 89.67 ± 2.34 | 0.96 ± 0.00 | - |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | bbc_bal | fidelity_tail_kh | spread_tail_kh |
|---|---|---|---|---|---|---|---|---|
| Ours-A-B Ts=10 | 0 | 95.89 | 92.93 | 88.87 | 95.93 | 96.05 | 89.83 | 0.96 |
| Ours-A-B Ts=10 | 1 | 95.87 | 92.68 | 90.98 | 95.95 | 96.04 | 91.92 | 0.96 |
| Ours-A-B Ts=10 | 2 | 92.50 | 86.37 | 78.39 | 92.81 | 92.86 | 87.25 | 0.96 |

