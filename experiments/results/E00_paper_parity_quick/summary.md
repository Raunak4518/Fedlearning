# E00_paper_parity: GeFL-F paper parity (IID, K=10)

Mode: QUICK (reduced rounds; viability only)

Hypothesis: best_mean_acc is within 1 pp of Figure 4 for each dataset.

## cifar10  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 55.86%

| label | n | best_mean_acc | final_acc | final_bal | final_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| FedAvg (grouped) | 1 | 38.58 ± 0.00 | 37.79 ± 0.00 | 37.79 ± 0.00 | 41.55 ± 0.00 | n<2 |
| GeFL-F | 1 | 37.21 ± 0.00 | 36.97 ± 0.00 | 36.97 ± 0.00 | 42.84 ± 0.00 | - |
| LG-FedAvg | 1 | 39.27 ± 0.00 | 38.49 ± 0.00 | 38.49 ± 0.00 | 44.17 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_bal | final_tail |
|---|---|---|---|---|---|
| FedAvg (grouped) | 0 | 38.58 | 37.79 | 37.79 | 41.55 |
| GeFL-F | 0 | 37.21 | 36.97 | 36.97 | 42.84 |
| LG-FedAvg | 0 | 39.27 | 38.49 | 38.49 | 44.17 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_acc | final_bal | final_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| FedAvg (grouped) | 1 | 80.25 ± 0.00 | 80.15 ± 0.00 | 80.15 ± 0.00 | 81.07 ± 0.00 | n<2 |
| GeFL-F | 1 | 81.74 ± 0.00 | 81.68 ± 0.00 | 81.68 ± 0.00 | 82.56 ± 0.00 | - |
| LG-FedAvg | 1 | 79.33 ± 0.00 | 79.13 ± 0.00 | 79.13 ± 0.00 | 79.56 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_bal | final_tail |
|---|---|---|---|---|---|
| FedAvg (grouped) | 0 | 80.25 | 80.15 | 80.15 | 81.07 |
| GeFL-F | 0 | 81.74 | 81.68 | 81.68 | 82.56 |
| LG-FedAvg | 0 | 79.33 | 79.13 | 79.13 | 79.56 |

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | best_mean_acc | final_acc | final_bal | final_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| FedAvg (grouped) | 1 | 93.16 ± 0.00 | 93.14 ± 0.00 | 93.08 ± 0.00 | 91.78 ± 0.00 | n<2 |
| GeFL-F | 1 | 94.70 ± 0.00 | 94.67 ± 0.00 | 94.63 ± 0.00 | 93.10 ± 0.00 | - |
| LG-FedAvg | 1 | 93.19 ± 0.00 | 93.15 ± 0.00 | 93.09 ± 0.00 | 91.63 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_bal | final_tail |
|---|---|---|---|---|---|
| FedAvg (grouped) | 0 | 93.16 | 93.14 | 93.08 | 91.78 |
| GeFL-F | 0 | 94.70 | 94.67 | 94.63 | 93.10 |
| LG-FedAvg | 0 | 93.19 | 93.15 | 93.09 | 91.63 |

## svhn  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 76.26%

| label | n | best_mean_acc | final_acc | final_bal | final_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| FedAvg (grouped) | 1 | 47.78 ± 0.00 | 47.70 ± 0.00 | 45.63 ± 0.00 | 42.64 ± 0.00 | - |

| label | seed | best_mean_acc | final_acc | final_bal | final_tail |
|---|---|---|---|---|---|
| FedAvg (grouped) | 0 | 47.78 | 47.70 | 45.63 | 42.64 |

