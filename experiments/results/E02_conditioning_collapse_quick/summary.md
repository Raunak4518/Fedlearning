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

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | cond_norm_tail_over_head_end | cond_norm_tail_rel_init | fidelity_tail | fidelity_head | final_bal | final_tail | p vs GeFL-F (cond_norm_tail_over_head_end) |
|---|---|---|---|---|---|---|---|---|
| +NOWD+HWA | 1 | 0.73 ± 0.00 | 1.19 ± 0.00 | 22.17 ± 0.00 | 77.78 ± 0.00 | 70.75 ± 0.00 | 47.88 ± 0.00 | n<2 |
| +HWA | 1 | 0.71 ± 0.00 | 1.15 ± 0.00 | 21.00 ± 0.00 | 77.56 ± 0.00 | 70.70 ± 0.00 | 47.87 ± 0.00 | n<2 |
| +HWA+LCD | 1 | 0.69 ± 0.00 | 1.12 ± 0.00 | 21.25 ± 0.00 | 76.11 ± 0.00 | 70.28 ± 0.00 | 46.91 ± 0.00 | n<2 |
| +NOWD | 1 | 0.83 ± 0.00 | 1.12 ± 0.00 | 19.58 ± 0.00 | 72.33 ± 0.00 | 69.87 ± 0.00 | 46.73 ± 0.00 | n<2 |
| GeFL-F | 1 | 0.22 ± 0.00 | 0.30 ± 0.00 | 15.00 ± 0.00 | 70.89 ± 0.00 | 68.29 ± 0.00 | 44.39 ± 0.00 | - |
| +LCD | 1 | 0.79 ± 0.00 | 1.06 ± 0.00 | 17.67 ± 0.00 | 70.56 ± 0.00 | 69.66 ± 0.00 | 45.82 ± 0.00 | n<2 |

