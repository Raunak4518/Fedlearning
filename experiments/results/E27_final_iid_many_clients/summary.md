# E27_final_iid_many_clients: Final method (Ours-A, T_s = 10) in the IID setting at K = 50 / 100

Mode: full

Hypothesis: Ours-A beats the paper's best at every K, as at K = 10.

## fmnist  IF=1.0  alpha=None  K=50
Paper Figure 4 GeFL-F (CVAE-F): 82.21%

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs Ours-A Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours-A Ts=10 | 3 | 82.96 ± 0.57 | 82.84 ± 0.59 | 85.56 ± 0.52 | 83.61 ± 0.52 | - |

| label | seed | best_mean_acc | final_bal | final_ens_acc | oracle_bal |
|---|---|---|---|---|---|
| Ours-A Ts=10 | 0 | 82.74 | 82.57 | 85.26 | 83.42 |
| Ours-A Ts=10 | 1 | 82.53 | 82.42 | 85.26 | 83.22 |
| Ours-A Ts=10 | 2 | 83.61 | 83.52 | 86.16 | 84.20 |

## mnist  IF=1.0  alpha=None  K=100
Paper Figure 4 GeFL-F (CVAE-F): 94.63%

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs Ours-A Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours-A Ts=10 | 3 | 96.77 ± 0.18 | 96.76 ± 0.18 | 97.29 ± 0.16 | 97.16 ± 0.13 | - |

| label | seed | best_mean_acc | final_bal | final_ens_acc | oracle_bal |
|---|---|---|---|---|---|
| Ours-A Ts=10 | 0 | 96.58 | 96.57 | 97.14 | 97.01 |
| Ours-A Ts=10 | 1 | 96.94 | 96.93 | 97.46 | 97.21 |
| Ours-A Ts=10 | 2 | 96.79 | 96.78 | 97.27 | 97.25 |

## mnist  IF=1.0  alpha=None  K=50
Paper Figure 4 GeFL-F (CVAE-F): 95.04%

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs Ours-A Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours-A Ts=10 | 3 | 96.93 ± 0.21 | 96.92 ± 0.21 | 97.55 ± 0.23 | 97.30 ± 0.19 | - |

| label | seed | best_mean_acc | final_bal | final_ens_acc | oracle_bal |
|---|---|---|---|---|---|
| Ours-A Ts=10 | 0 | 96.73 | 96.72 | 97.28 | 97.11 |
| Ours-A Ts=10 | 1 | 97.14 | 97.13 | 97.68 | 97.48 |
| Ours-A Ts=10 | 2 | 96.91 | 96.90 | 97.69 | 97.31 |

