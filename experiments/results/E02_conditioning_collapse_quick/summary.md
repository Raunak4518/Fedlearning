# E02_conditioning_collapse: Rare-class conditioning collapse under weight decay and flat averaging

Mode: QUICK (reduced rounds; viability only)

Hypothesis: GeFL-F tail rows collapse (norm ratio << 1, low tail fidelity); HWA / LCD / NOWD prevent it.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | cond_norm_tail_over_head_end | cond_norm_tail_rel_init | fidelity_tail | fidelity_head | final_bal | final_tail | p vs GeFL-F (cond_norm_tail_over_head_end) |
|---|---|---|---|---|---|---|---|---|
| +NOWD+HWA | 1 | 1.02 ± 0.00 | 1.59 ± 0.00 | 42.25 ± 0.00 | 77.11 ± 0.00 | 63.33 ± 0.00 | 51.92 ± 0.00 | n<2 |
| +HWA | 1 | 1.00 ± 0.00 | 1.55 ± 0.00 | 41.08 ± 0.00 | 75.44 ± 0.00 | 63.43 ± 0.00 | 52.20 ± 0.00 | n<2 |
| +HWA+LCD | 1 | 0.86 ± 0.00 | 1.34 ± 0.00 | 31.92 ± 0.00 | 77.33 ± 0.00 | 62.23 ± 0.00 | 49.37 ± 0.00 | n<2 |
| +NOWD | 1 | 0.90 ± 0.00 | 1.33 ± 0.00 | 32.67 ± 0.00 | 75.22 ± 0.00 | 62.55 ± 0.00 | 49.93 ± 0.00 | n<2 |
| GeFL-F | 1 | 0.56 ± 0.00 | 0.81 ± 0.00 | 20.92 ± 0.00 | 72.78 ± 0.00 | 57.91 ± 0.00 | 39.40 ± 0.00 | - |
| +LCD | 1 | 0.79 ± 0.00 | 1.14 ± 0.00 | 24.58 ± 0.00 | 74.56 ± 0.00 | 60.69 ± 0.00 | 45.33 ± 0.00 | n<2 |

| label | seed | cond_norm_tail_over_head_end | cond_norm_tail_rel_init | fidelity_tail | fidelity_head | final_bal | final_tail |
|---|---|---|---|---|---|---|---|
| +NOWD+HWA | 0 | 1.02 | 1.59 | 42.25 | 77.11 | 63.33 | 51.92 |
| +HWA | 0 | 1.00 | 1.55 | 41.08 | 75.44 | 63.43 | 52.20 |
| +HWA+LCD | 0 | 0.86 | 1.34 | 31.92 | 77.33 | 62.23 | 49.37 |
| +NOWD | 0 | 0.90 | 1.33 | 32.67 | 75.22 | 62.55 | 49.93 |
| GeFL-F | 0 | 0.56 | 0.81 | 20.92 | 72.78 | 57.91 | 39.40 |
| +LCD | 0 | 0.79 | 1.14 | 24.58 | 74.56 | 60.69 | 45.33 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | cond_norm_tail_over_head_end | cond_norm_tail_rel_init | fidelity_tail | fidelity_head | final_bal | final_tail | p vs GeFL-F (cond_norm_tail_over_head_end) |
|---|---|---|---|---|---|---|---|---|
| +NOWD+HWA | 1 | 0.73 ± 0.00 | 1.19 ± 0.00 | 22.17 ± 0.00 | 77.78 ± 0.00 | 70.75 ± 0.00 | 47.88 ± 0.00 | n<2 |
| +HWA | 1 | 0.71 ± 0.00 | 1.15 ± 0.00 | 21.00 ± 0.00 | 77.56 ± 0.00 | 70.70 ± 0.00 | 47.87 ± 0.00 | n<2 |
| +HWA+LCD | 1 | 0.69 ± 0.00 | 1.12 ± 0.00 | 21.25 ± 0.00 | 76.11 ± 0.00 | 70.28 ± 0.00 | 46.91 ± 0.00 | n<2 |
| +NOWD | 1 | 0.83 ± 0.00 | 1.12 ± 0.00 | 19.58 ± 0.00 | 72.33 ± 0.00 | 69.87 ± 0.00 | 46.73 ± 0.00 | n<2 |
| GeFL-F | 1 | 0.22 ± 0.00 | 0.30 ± 0.00 | 15.00 ± 0.00 | 70.89 ± 0.00 | 68.29 ± 0.00 | 44.39 ± 0.00 | - |
| +LCD | 1 | 0.79 ± 0.00 | 1.06 ± 0.00 | 17.67 ± 0.00 | 70.56 ± 0.00 | 69.66 ± 0.00 | 45.82 ± 0.00 | n<2 |

| label | seed | cond_norm_tail_over_head_end | cond_norm_tail_rel_init | fidelity_tail | fidelity_head | final_bal | final_tail |
|---|---|---|---|---|---|---|---|
| +NOWD+HWA | 0 | 0.73 | 1.19 | 22.17 | 77.78 | 70.75 | 47.88 |
| +HWA | 0 | 0.71 | 1.15 | 21.00 | 77.56 | 70.70 | 47.87 |
| +HWA+LCD | 0 | 0.69 | 1.12 | 21.25 | 76.11 | 70.28 | 46.91 |
| +NOWD | 0 | 0.83 | 1.12 | 19.58 | 72.33 | 69.87 | 46.73 |
| GeFL-F | 0 | 0.22 | 0.30 | 15.00 | 70.89 | 68.29 | 44.39 |
| +LCD | 0 | 0.79 | 1.06 | 17.67 | 70.56 | 69.66 | 45.82 |

