# E29_hybrid_generator: Hybrid generator PCR on MNIST / FMNIST (where the mean anchor is informative)

Mode: full

Hypothesis: Ours-R (PCR + MC + KH + LA + CSL, T_s = 10) >= Ours-A under the long tail and >= the CVAE variant in IID.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end | p vs Ours-R Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 3 | 79.60 ± 0.39 | 73.21 ± 0.77 | 79.91 ± 0.33 | 80.46 ± 0.38 | 0.60 ± 0.14 | 68.07 ± 2.03 | 0.91 ± 0.05 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 0 | 79.23 | 73.16 | 79.59 | 80.03 | 0.71 | 69.96 | 0.97 |
| Ours-R Ts=10 | 1 | 80.01 | 74.00 | 80.25 | 80.77 | 0.66 | 65.92 | 0.90 |
| Ours-R Ts=10 | 2 | 79.57 | 72.47 | 79.90 | 80.58 | 0.44 | 68.34 | 0.87 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end | p vs Ours-R Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 3 | 84.00 ± 0.56 | 84.34 ± 0.37 | 84.11 ± 0.57 | 83.91 ± 0.51 | 0.53 ± 0.06 | 69.30 ± 1.99 | 1.03 ± 0.02 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 0 | 83.74 | 84.73 | 83.93 | 83.72 | 0.59 | 71.14 | 1.02 |
| Ours-R Ts=10 | 1 | 83.61 | 83.99 | 83.66 | 83.52 | 0.54 | 67.18 | 1.02 |
| Ours-R Ts=10 | 2 | 84.64 | 84.29 | 84.75 | 84.49 | 0.46 | 69.58 | 1.05 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end | p vs Ours-R Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 3 | 95.03 ± 0.32 | 91.46 ± 0.65 | 95.24 ± 0.30 | 95.34 ± 0.26 | 0.29 ± 0.03 | 81.98 ± 0.27 | 0.86 ± 0.01 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 0 | 95.30 | 92.07 | 95.47 | 95.55 | 0.30 | 82.28 | 0.84 |
| Ours-R Ts=10 | 1 | 95.12 | 91.53 | 95.35 | 95.43 | 0.32 | 81.76 | 0.86 |
| Ours-R Ts=10 | 2 | 94.67 | 90.77 | 94.90 | 95.04 | 0.25 | 81.89 | 0.87 |

## mnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end | p vs Ours-R Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 2 | 94.36 ± 0.15 | 90.53 ± 0.07 | 94.40 ± 0.17 | 94.49 ± 0.20 | 0.30 ± 0.01 | 81.31 ± 1.24 | 0.88 ± 0.02 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|---|
| Ours-R Ts=10 | 0 | 94.46 | 90.58 | 94.52 | 94.63 | 0.29 | 82.19 | 0.87 |
| Ours-R Ts=10 | 1 | 94.25 | 90.49 | 94.28 | 94.34 | 0.31 | 80.43 | 0.89 |

