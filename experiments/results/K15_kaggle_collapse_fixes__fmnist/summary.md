# K15_kaggle_collapse_fixes: Existing fixes for row collapse (AdamW, lazy decay) against HWA, K = 10 / 50 / 100

Mode: full

Hypothesis: ADAMW3, ROWNOWD and LCD restore the tail rows (ratio ~1) but trail +HWA+LA, by more as K grows; ADAMW (1e-2) = ADAMW3 at K = 10.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 3 | 75.89 ± 0.60 | 69.19 ± 1.70 | 76.76 ± 0.41 | 0.96 ± 0.05 | 61.83 ± 6.44 | 0.987 |
| +ADAMW3+LA | 3 | 75.90 ± 0.57 | 68.80 ± 0.61 | 76.61 ± 0.58 | 0.96 ± 0.06 | 61.47 ± 3.32 | - |
| +ADAMW+LA | 3 | 76.08 ± 0.32 | 69.34 ± 0.89 | 76.88 ± 0.23 | 0.97 ± 0.06 | 63.00 ± 7.85 | 0.714 |
| +LCD+LA | 3 | 75.79 ± 0.30 | 68.97 ± 0.23 | 76.45 ± 0.23 | 0.78 ± 0.01 | 51.86 ± 6.88 | 0.788 |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 0 | 75.21 | 67.25 | 76.30 | 1.02 | 65.08 |
| +ROWNOWD+LA | 1 | 76.31 | 69.88 | 77.11 | 0.96 | 66.00 |
| +ROWNOWD+LA | 2 | 76.15 | 70.43 | 76.87 | 0.91 | 54.42 |
| +ADAMW3+LA | 0 | 75.74 | 68.75 | 76.41 | 1.03 | 64.33 |
| +ADAMW3+LA | 1 | 75.42 | 68.22 | 76.16 | 0.95 | 62.25 |
| +ADAMW3+LA | 2 | 76.53 | 69.44 | 77.27 | 0.91 | 57.83 |
| +ADAMW+LA | 0 | 75.78 | 68.44 | 76.61 | 1.04 | 69.42 |
| +ADAMW+LA | 1 | 76.42 | 69.37 | 77.01 | 0.95 | 65.33 |
| +ADAMW+LA | 2 | 76.04 | 70.21 | 77.01 | 0.91 | 54.25 |
| +LCD+LA | 0 | 75.45 | 69.20 | 76.30 | 0.78 | 57.42 |
| +LCD+LA | 1 | 76.01 | 68.74 | 76.33 | 0.78 | 54.00 |
| +LCD+LA | 2 | 75.90 | 68.97 | 76.72 | 0.76 | 44.17 |

## fmnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 3 | 71.85 ± 1.56 | 57.79 ± 2.89 | 72.71 ± 1.06 | 0.89 ± 0.01 | 28.39 ± 2.38 | 0.811 |
| +ADAMW3+LA | 3 | 71.82 ± 1.40 | 57.65 ± 2.36 | 72.57 ± 1.20 | 0.89 ± 0.01 | 29.69 ± 3.48 | - |
| +LCD+LA | 3 | 71.64 ± 1.28 | 57.36 ± 2.57 | 72.45 ± 1.14 | 0.87 ± 0.01 | 26.11 ± 1.96 | 0.216 |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 0 | 73.62 | 61.13 | 73.80 | 0.88 | 30.67 |
| +ROWNOWD+LA | 1 | 71.25 | 56.18 | 72.63 | 0.89 | 25.92 |
| +ROWNOWD+LA | 2 | 70.68 | 56.06 | 71.69 | 0.89 | 28.58 |
| +ADAMW3+LA | 0 | 73.41 | 60.26 | 73.93 | 0.88 | 32.33 |
| +ADAMW3+LA | 1 | 71.29 | 55.69 | 72.11 | 0.90 | 25.75 |
| +ADAMW3+LA | 2 | 70.77 | 56.99 | 71.67 | 0.89 | 31.00 |
| +LCD+LA | 0 | 73.06 | 60.28 | 73.73 | 0.86 | 28.00 |
| +LCD+LA | 1 | 71.29 | 55.51 | 72.08 | 0.88 | 24.08 |
| +LCD+LA | 2 | 70.57 | 56.28 | 71.55 | 0.88 | 26.25 |

## fmnist  IF=0.01  alpha=0.5  K=50

| label | n | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail | p vs +ADAMW3+LA (final_bal) |
|---|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 3 | 74.52 ± 0.56 | 63.71 ± 0.90 | 75.16 ± 0.56 | 0.88 ± 0.02 | 36.72 ± 4.63 | 0.288 |
| +ADAMW3+LA | 3 | 74.03 ± 0.72 | 62.53 ± 1.80 | 74.59 ± 0.86 | 0.88 ± 0.02 | 35.86 ± 7.46 | - |
| +LCD+LA | 3 | 74.19 ± 0.65 | 63.11 ± 1.30 | 74.78 ± 0.84 | 0.83 ± 0.01 | 31.69 ± 4.72 | 0.607 |

| label | seed | final_bal | final_tail | best_mean_acc | cond_norm_tail_over_head_end | fidelity_tail |
|---|---|---|---|---|---|---|
| +ROWNOWD+LA | 0 | 74.95 | 64.61 | 75.70 | 0.89 | 38.83 |
| +ROWNOWD+LA | 1 | 74.73 | 63.71 | 75.20 | 0.90 | 39.92 |
| +ROWNOWD+LA | 2 | 73.88 | 62.81 | 74.59 | 0.87 | 31.42 |
| +ADAMW3+LA | 0 | 74.87 | 64.50 | 75.50 | 0.89 | 40.50 |
| +ADAMW3+LA | 1 | 73.56 | 60.97 | 74.48 | 0.90 | 39.83 |
| +ADAMW3+LA | 2 | 73.67 | 62.11 | 73.79 | 0.86 | 27.25 |
| +LCD+LA | 0 | 74.81 | 64.54 | 75.59 | 0.82 | 34.50 |
| +LCD+LA | 1 | 74.26 | 62.74 | 74.83 | 0.84 | 34.33 |
| +LCD+LA | 2 | 73.51 | 62.03 | 73.92 | 0.82 | 26.25 |

