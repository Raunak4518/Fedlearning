# K15_kaggle_collapse_fixes: Existing fixes for row collapse (AdamW, lazy decay) against HWA, K = 10 / 50 / 100

Mode: full

Hypothesis: ADAMW3, ROWNOWD and LCD restore the tail rows (ratio ~1) but trail +HWA+LA, by more as K grows; ADAMW (1e-2) = ADAMW3 at K = 10.

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ADAMW3+LA | 3 | 86.44 ± 2.00 | 77.52 ± 5.39 | 87.39 ± 1.64 | 0.88 ± 0.07 | 26.64 ± 2.42 | - |
| +ROWNOWD+LA | 3 | 86.81 ± 2.40 | 78.38 ± 6.50 | 87.57 ± 2.45 | 0.87 ± 0.06 | 25.19 ± 1.08 | 0.618 |
| +LCD+LA | 3 | 85.06 ± 1.47 | 74.20 ± 4.11 | 85.98 ± 1.52 | 0.73 ± 0.04 | 21.36 ± 1.21 | 0.0682 |
| +ADAMW+LA | 3 | 86.53 ± 2.46 | 77.98 ± 6.62 | 87.32 ± 2.53 | 0.87 ± 0.05 | 25.00 ± 2.50 | 0.918 |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ADAMW3+LA | 0 | 84.14 | 71.37 | 85.51 | 0.81 | 24.17 |
| +ADAMW3+LA | 1 | 87.70 | 81.45 | 88.52 | 0.88 | 26.75 |
| +ADAMW3+LA | 2 | 87.50 | 79.72 | 88.13 | 0.94 | 29.00 |
| +ROWNOWD+LA | 0 | 84.27 | 70.92 | 84.77 | 0.81 | 25.25 |
| +ROWNOWD+LA | 1 | 87.12 | 81.43 | 88.61 | 0.88 | 24.08 |
| +ROWNOWD+LA | 2 | 89.04 | 82.79 | 89.34 | 0.92 | 26.25 |
| +LCD+LA | 0 | 83.39 | 69.45 | 84.22 | 0.69 | 20.58 |
| +LCD+LA | 1 | 85.63 | 76.63 | 87.00 | 0.75 | 20.75 |
| +LCD+LA | 2 | 86.16 | 76.52 | 86.71 | 0.77 | 22.75 |
| +ADAMW+LA | 0 | 83.97 | 70.42 | 84.45 | 0.81 | 25.92 |
| +ADAMW+LA | 1 | 86.71 | 80.81 | 88.29 | 0.89 | 22.17 |
| +ADAMW+LA | 2 | 88.89 | 82.72 | 89.23 | 0.92 | 26.92 |

## mnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ADAMW3+LA | 3 | 81.60 ± 0.33 | 62.64 ± 0.13 | 81.98 ± 0.26 | 0.89 ± 0.01 | 14.53 ± 5.35 | - |
| +ROWNOWD+LA | 3 | 81.77 ± 0.44 | 63.16 ± 0.49 | 82.06 ± 0.37 | 0.89 ± 0.01 | 14.11 ± 4.38 | 0.137 |
| +LCD+LA | 3 | 81.65 ± 0.45 | 62.94 ± 0.37 | 81.93 ± 0.40 | 0.88 ± 0.01 | 13.53 ± 4.25 | 0.601 |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ADAMW3+LA | 0 | 81.43 | 62.49 | 82.02 | 0.87 | 20.08 |
| +ADAMW3+LA | 1 | 81.39 | 62.73 | 81.70 | 0.89 | 14.08 |
| +ADAMW3+LA | 2 | 81.98 | 62.71 | 82.21 | 0.89 | 9.42 |
| +ROWNOWD+LA | 0 | 81.58 | 62.74 | 81.95 | 0.87 | 18.42 |
| +ROWNOWD+LA | 1 | 81.45 | 63.02 | 81.75 | 0.89 | 14.25 |
| +ROWNOWD+LA | 2 | 82.27 | 63.70 | 82.47 | 0.89 | 9.67 |
| +LCD+LA | 0 | 81.52 | 62.67 | 81.86 | 0.86 | 17.75 |
| +LCD+LA | 1 | 81.28 | 62.79 | 81.58 | 0.89 | 13.58 |
| +LCD+LA | 2 | 82.16 | 63.36 | 82.36 | 0.88 | 9.25 |

## mnist  IF=0.01  alpha=0.5  K=50

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 3 | 84.44 ± 0.66 | 69.36 ± 1.34 | 84.95 ± 0.68 | 0.86 ± 0.01 | 15.89 ± 3.15 | 0.377 |
| +LCD+LA | 3 | 83.92 ± 0.49 | 68.08 ± 0.89 | 84.41 ± 0.55 | 0.83 ± 0.01 | 14.03 ± 3.19 | 0.0602 |
| +ADAMW3+LA | 3 | 84.71 ± 0.29 | 69.91 ± 0.51 | 85.19 ± 0.21 | 0.86 ± 0.01 | 16.19 ± 3.63 | - |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 0 | 84.10 | 68.82 | 84.67 | 0.85 | 19.25 |
| +ROWNOWD+LA | 1 | 84.01 | 68.38 | 84.44 | 0.87 | 15.42 |
| +ROWNOWD+LA | 2 | 85.20 | 70.89 | 85.72 | 0.86 | 13.00 |
| +LCD+LA | 0 | 83.86 | 68.18 | 84.40 | 0.82 | 17.42 |
| +LCD+LA | 1 | 83.46 | 67.14 | 83.87 | 0.84 | 13.58 |
| +LCD+LA | 2 | 84.43 | 68.91 | 84.97 | 0.82 | 11.08 |
| +ADAMW3+LA | 0 | 84.46 | 69.40 | 84.98 | 0.85 | 19.92 |
| +ADAMW3+LA | 1 | 84.67 | 70.42 | 85.19 | 0.87 | 16.00 |
| +ADAMW3+LA | 2 | 85.02 | 69.90 | 85.40 | 0.86 | 12.67 |

