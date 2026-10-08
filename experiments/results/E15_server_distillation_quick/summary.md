# E15_server_distillation: Server-side ensemble distillation on generated features

Mode: QUICK (reduced rounds; viability only)

Hypothesis: SED adds to CSL in IID (towards the 84.28 FMNIST best) and under the long tail.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_tail | final_ens_acc | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 1 | 70.68 ± 0.00 | 70.65 ± 0.00 | 65.64 ± 0.00 | 79.87 ± 0.00 | n<2 |
| Ours+SED(20) | 1 | 71.58 ± 0.00 | 71.30 ± 0.00 | 66.52 ± 0.00 | 78.00 ± 0.00 | n<2 |
| GeFL-F | 1 | 59.05 ± 0.00 | 58.57 ± 0.00 | 40.16 ± 0.00 | 70.53 ± 0.00 | - |
| +CSL | 1 | 58.43 ± 0.00 | 58.29 ± 0.00 | 39.28 ± 0.00 | 71.37 ± 0.00 | n<2 |
| +CSL+SED(20) | 1 | 61.51 ± 0.00 | 60.57 ± 0.00 | 40.96 ± 0.00 | 70.35 ± 0.00 | n<2 |
| +CSL+SED(60) | 1 | 61.74 ± 0.00 | 59.57 ± 0.00 | 40.62 ± 0.00 | 71.85 ± 0.00 | n<2 |
| +SED(20) | 1 | 62.45 ± 0.00 | 61.58 ± 0.00 | 42.67 ± 0.00 | 70.29 ± 0.00 | n<2 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_bal | final_tail | final_ens_acc | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 1 | 81.69 ± 0.00 | 81.62 ± 0.00 | 82.50 ± 0.00 | 85.30 ± 0.00 | n<2 |
| Ours+SED(20) | 1 | 80.96 ± 0.00 | 80.72 ± 0.00 | 81.48 ± 0.00 | 85.14 ± 0.00 | n<2 |
| GeFL-F | 1 | 81.57 ± 0.00 | 81.51 ± 0.00 | 82.56 ± 0.00 | 85.47 ± 0.00 | - |
| +CSL | 1 | 81.60 ± 0.00 | 81.56 ± 0.00 | 82.71 ± 0.00 | 85.32 ± 0.00 | n<2 |
| +CSL+SED(20) | 1 | 80.82 ± 0.00 | 80.59 ± 0.00 | 81.41 ± 0.00 | 85.12 ± 0.00 | n<2 |
| +CSL+SED(60) | 1 | 80.78 ± 0.00 | 80.48 ± 0.00 | 83.48 ± 0.00 | 84.83 ± 0.00 | n<2 |
| +SED(20) | 1 | 80.56 ± 0.00 | 80.17 ± 0.00 | 81.61 ± 0.00 | 84.98 ± 0.00 | n<2 |

