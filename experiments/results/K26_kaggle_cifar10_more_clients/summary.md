# K26_kaggle_cifar10_more_clients: CIFAR-10 with 50 clients (5 per architecture): decay fix vs HWA vs ours, seeds 3-5

Mode: full

Hypothesis: D1-D2 in the docstring.

## cifar10  IF=0.01  alpha=0.5  K=50

| label | n | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | cond_norm_tail_over_head_end | p vs GeFL-F + AdamW + LA (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| GeFL-F + HWA + LA | 3 | 35.90 ± 0.34 | 45.22 ± 0.51 | 44.15 ± 0.15 | 0.36 ± 0.01 | 0.45 ± 0.00 | 22.45 ± 0.82 | 1.09 ± 0.09 | 0.0231 |
| Ours (gated) | 3 | 40.37 ± 0.35 | 46.13 ± 0.22 | 45.10 ± 0.20 | 0.40 ± 0.01 | 0.46 ± 0.01 | 31.21 ± 1.47 | 0.00 ± 0.00 | 0.00331 |
| GeFL-F + AdamW + LA | 3 | 35.66 ± 0.36 | 45.12 ± 0.42 | 44.08 ± 0.00 | 0.36 ± 0.00 | 0.45 ± 0.00 | 21.43 ± 0.92 | 0.90 ± 0.03 | - |

| label | seed | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|---|
| GeFL-F + HWA + LA | 3 | 35.51 | 44.67 | 44.00 | 0.36 | 0.45 | 23.30 | 1.01 |
| GeFL-F + HWA + LA | 4 | 36.18 | 45.32 | 44.17 | 0.35 | 0.45 | 22.39 | 1.10 |
| GeFL-F + HWA + LA | 5 | 36.00 | 45.67 | 44.29 | 0.36 | 0.45 | 21.67 | 1.18 |
| Ours (gated) | 3 | 40.15 | 46.01 | 45.14 | 0.41 | 0.47 | 32.40 | 0.00 |
| Ours (gated) | 4 | 40.18 | 45.99 | 44.89 | 0.40 | 0.46 | 29.57 | 0.00 |
| Ours (gated) | 5 | 40.78 | 46.38 | 45.28 | 0.41 | 0.46 | 31.67 | 0.00 |
| GeFL-F + AdamW + LA | 3 | 35.29 | 44.65 | 44.09 | 0.36 | 0.45 | 22.32 | 0.87 |
| GeFL-F + AdamW + LA | 4 | 36.00 | 45.29 | 44.08 | 0.35 | 0.44 | 21.49 | 0.91 |
| GeFL-F + AdamW + LA | 5 | 35.69 | 45.43 | 44.08 | 0.36 | 0.45 | 20.48 | 0.93 |

