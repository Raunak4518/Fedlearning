# F10_combined_method: Combined method HWA + LA + CSL: long tail and IID

Mode: full

Hypothesis: +HWA+LA+CSL >= +HWA+LA under the long tail and >= +CSL in IID.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | final_acc | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 75.99 ± 0.47 | 68.64 ± 1.04 | 76.56 ± 0.54 | 75.99 ± 0.47 | 0.0197 |
| +HWA+LA+CSL | 3 | 76.37 ± 0.64 | 69.83 ± 0.30 | 76.85 ± 0.69 | 76.36 ± 0.64 | 0.0185 |
| GeFL-F | 3 | 58.74 ± 3.94 | 35.57 ± 7.05 | 59.41 ± 3.58 | 58.74 ± 3.94 | - |
| +CSL | 3 | 58.59 ± 3.67 | 36.10 ± 5.71 | 59.24 ± 3.27 | 58.59 ± 3.67 | 0.637 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | final_acc | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 82.54 ± 0.28 | 82.91 ± 0.68 | 82.74 ± 0.28 | 82.54 ± 0.28 | 0.109 |
| +HWA+LA+CSL | 3 | 83.17 ± 0.33 | 83.74 ± 0.54 | 83.21 ± 0.28 | 83.17 ± 0.33 | 0.00935 |
| GeFL-F | 3 | 82.63 ± 0.24 | 82.89 ± 0.46 | 82.80 ± 0.22 | 82.63 ± 0.24 | - |
| +CSL | 3 | 83.18 ± 0.37 | 83.75 ± 0.40 | 83.25 ± 0.31 | 83.18 ± 0.37 | 0.0257 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | final_acc | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 87.78 ± 2.07 | 80.80 ± 4.06 | 88.49 ± 2.06 | 87.91 ± 2.05 | 0.0132 |
| +HWA+LA+CSL | 3 | 89.87 ± 0.96 | 82.67 ± 2.31 | 90.43 ± 0.50 | 89.98 ± 0.97 | 0.0079 |
| GeFL-F | 3 | 74.24 ± 1.91 | 51.40 ± 5.49 | 74.87 ± 1.80 | 74.48 ± 1.84 | - |
| +CSL | 3 | 76.35 ± 1.96 | 51.38 ± 4.92 | 76.68 ± 1.82 | 76.60 ± 1.89 | 0.00312 |

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | final_bal | final_tail | best_mean_acc | final_acc | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 95.72 ± 0.06 | 94.47 ± 0.21 | 96.01 ± 0.05 | 95.76 ± 0.07 | 0.744 |
| +HWA+LA+CSL | 3 | 96.71 ± 0.07 | 95.60 ± 0.09 | 96.77 ± 0.06 | 96.72 ± 0.06 | 0.000903 |
| GeFL-F | 3 | 95.75 ± 0.06 | 94.40 ± 0.13 | 96.01 ± 0.14 | 95.79 ± 0.05 | - |
| +CSL | 3 | 96.75 ± 0.07 | 95.63 ± 0.08 | 96.82 ± 0.07 | 96.77 ± 0.07 | 0.00125 |

