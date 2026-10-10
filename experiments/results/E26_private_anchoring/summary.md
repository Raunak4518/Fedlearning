# E26_private_anchoring: (eps, delta)-DP anchoring statistics for the final method

Mode: full

Hypothesis: at eps = 8 the method keeps most of its gain over GeFL-F; at eps = 2 it degrades gracefully, still well above GeFL-F.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | bbc_bal | feature_mnd | feature_mnd_final | mc_classes_used | kh_classes_used | p vs Ours-A Ts=10, DP eps=8 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10, DP eps=2 | 3 | 65.59 ± 0.58 | 49.20 ± 0.82 | 62.86 ± 0.11 | 1.12 ± 0.01 | 1.01 ± 0.01 | 5.00 ± 0.00 | 3.00 ± 0.00 | 0.0101 |
| Ours-A Ts=10, DP eps=8 | 3 | 72.77 ± 1.67 | 60.20 ± 2.60 | 70.98 ± 0.45 | 1.14 ± 0.00 | 1.07 ± 0.03 | 7.00 ± 0.00 | 5.00 ± 0.00 | - |

| label | seed | final_bal | final_tail | bbc_bal | feature_mnd | feature_mnd_final | mc_classes_used | kh_classes_used |
|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10, DP eps=2 | 0 | 65.74 | 49.24 | 62.85 | 1.13 | 1.02 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=2 | 1 | 64.95 | 48.36 | 62.76 | 1.11 | 1.02 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=2 | 2 | 66.07 | 50.01 | 62.98 | 1.11 | 1.00 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=8 | 0 | 72.01 | 58.61 | 71.19 | 1.14 | 1.10 | 7.00 | 5.00 |
| Ours-A Ts=10, DP eps=8 | 1 | 71.62 | 58.78 | 70.47 | 1.13 | 1.08 | 7.00 | 5.00 |
| Ours-A Ts=10, DP eps=8 | 2 | 74.69 | 63.20 | 71.29 | 1.13 | 1.03 | 7.00 | 5.00 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | bbc_bal | feature_mnd | feature_mnd_final | mc_classes_used | kh_classes_used | p vs Ours-A Ts=10, DP eps=8 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10, DP eps=2 | 3 | 85.06 ± 0.54 | 68.56 ± 1.74 | 84.87 ± 0.48 | 1.05 ± 0.00 | 0.99 ± 0.01 | 5.00 ± 0.00 | 3.00 ± 0.00 | 0.000768 |
| Ours-A Ts=10, DP eps=8 | 3 | 90.36 ± 0.30 | 79.60 ± 0.27 | 89.83 ± 0.14 | 1.04 ± 0.01 | 1.01 ± 0.00 | 7.00 ± 0.00 | 5.00 ± 0.00 | - |

| label | seed | final_bal | final_tail | bbc_bal | feature_mnd | feature_mnd_final | mc_classes_used | kh_classes_used |
|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10, DP eps=2 | 0 | 85.41 | 70.38 | 85.21 | 1.05 | 0.98 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=2 | 1 | 84.43 | 66.91 | 84.32 | 1.05 | 1.00 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=2 | 2 | 85.33 | 68.40 | 85.08 | 1.04 | 0.99 | 5.00 | 3.00 |
| Ours-A Ts=10, DP eps=8 | 0 | 90.60 | 79.88 | 89.91 | 1.04 | 1.01 | 7.00 | 5.00 |
| Ours-A Ts=10, DP eps=8 | 1 | 90.02 | 79.60 | 89.67 | 1.04 | 1.02 | 7.00 | 5.00 |
| Ours-A Ts=10, DP eps=8 | 2 | 90.46 | 79.33 | 89.91 | 1.03 | 1.01 | 7.00 | 5.00 |

