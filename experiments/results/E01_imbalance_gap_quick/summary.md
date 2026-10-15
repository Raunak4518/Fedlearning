# E01_imbalance_gap: GeFL-F accuracy gap by imbalance regime

Mode: QUICK (reduced rounds; viability only)

Hypothesis: final_bal and final_tail decrease monotonically from IID to IF=100; the GeFL-F minus LG-FedAvg gap on the tail shrinks as imbalance grows.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|---|
| GeFL-F | 1 | 58.34 ± 0.00 | 39.78 ± 0.00 | 19.43 ± 0.00 | 58.57 ± 0.00 | 78.27 ± 0.00 | - |
| LG-FedAvg | 1 | 52.87 ± 0.00 | 35.30 ± 0.00 | 19.44 ± 0.00 | 53.01 ± 0.00 | - | n<2 |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal |
|---|---|---|---|---|---|---|
| GeFL-F | 0 | 58.34 | 39.78 | 19.43 | 58.57 | 78.27 |
| LG-FedAvg | 0 | 52.87 | 35.30 | 19.44 | 53.01 | - |

## fmnist  IF=0.1  alpha=0.5  K=10

| label | n | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|---|
| GeFL-F | 1 | 66.68 ± 0.00 | 64.92 ± 0.00 | 38.18 ± 0.00 | 66.95 ± 0.00 | 80.52 ± 0.00 | - |
| LG-FedAvg | 1 | 59.29 ± 0.00 | 53.52 ± 0.00 | 32.68 ± 0.00 | 59.37 ± 0.00 | - | n<2 |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal |
|---|---|---|---|---|---|---|
| GeFL-F | 0 | 66.68 | 64.92 | 38.18 | 66.95 | 80.52 |
| LG-FedAvg | 0 | 59.29 | 53.52 | 32.68 | 59.37 | - |

## fmnist  IF=1.0  alpha=0.5  K=10

| label | n | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|---|
| GeFL-F | 1 | 66.51 ± 0.00 | 74.01 ± 0.00 | 39.68 ± 0.00 | 68.24 ± 0.00 | 80.29 ± 0.00 | - |
| LG-FedAvg | 1 | 59.11 ± 0.00 | 62.57 ± 0.00 | 34.46 ± 0.00 | 59.66 ± 0.00 | - | n<2 |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal |
|---|---|---|---|---|---|---|
| GeFL-F | 0 | 66.51 | 74.01 | 39.68 | 68.24 | 80.29 |
| LG-FedAvg | 0 | 59.11 | 62.57 | 34.46 | 59.66 | - |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal | p vs GeFL-F (final_bal) |
|---|---|---|---|---|---|---|---|
| GeFL-F | 1 | 81.64 ± 0.00 | 82.63 ± 0.00 | 54.58 ± 0.00 | 81.69 ± 0.00 | 83.26 ± 0.00 | - |
| LG-FedAvg | 1 | 79.27 ± 0.00 | 79.65 ± 0.00 | 47.57 ± 0.00 | 79.33 ± 0.00 | - | n<2 |

| label | seed | final_bal | final_tail | final_worst | best_mean_acc | oracle_bal |
|---|---|---|---|---|---|---|
| GeFL-F | 0 | 81.64 | 82.63 | 54.58 | 81.69 | 83.26 |
| LG-FedAvg | 0 | 79.27 | 79.65 | 47.57 | 79.33 | - |

