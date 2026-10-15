# E22_pc_many_clients: Anchored stack at K = 100 clients (MNIST long tail)

Mode: full

Hypothesis: PC + MC does not degrade with K: >= FSG (86.1) at K = 100.

## mnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | fidelity_tail_mc | spread_tail_mc | p vs Ours-PC+MC (final_bal) |
|---|---|---|---|---|---|---|---|
| Ours-PC+MC | 3 | 89.20 ± 0.47 | 81.29 ± 0.46 | 89.55 ± 0.50 | 82.83 ± 0.88 | 0.82 ± 0.00 | - |
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 90.41 ± 0.52 | 83.41 ± 0.89 | 90.60 ± 0.57 | 83.64 ± 1.58 | 0.82 ± 0.01 | 0.00742 |
| Ours-A Ts=10 | 3 | 94.30 ± 0.31 | 90.54 ± 0.28 | 94.45 ± 0.18 | 83.69 ± 0.13 | 0.83 ± 0.02 | 0.000444 |

| label | seed | final_bal | final_tail | best_mean_acc | fidelity_tail_mc | spread_tail_mc |
|---|---|---|---|---|---|---|
| Ours-PC+MC | 0 | 89.47 | 81.72 | 89.80 | 82.17 | 0.82 |
| Ours-PC+MC | 1 | 88.65 | 81.36 | 88.98 | 83.83 | 0.83 |
| Ours-PC+MC | 2 | 89.48 | 80.80 | 89.88 | 82.50 | 0.82 |
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 90.87 | 84.36 | 91.03 | 81.83 | 0.82 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 89.85 | 82.61 | 89.95 | 84.75 | 0.83 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 90.52 | 83.27 | 90.82 | 84.33 | 0.83 |
| Ours-A Ts=10 | 0 | 94.55 | 90.85 | 94.60 | 83.58 | 0.85 |
| Ours-A Ts=10 | 1 | 93.95 | 90.33 | 94.24 | 83.67 | 0.83 |
| Ours-A Ts=10 | 2 | 94.41 | 90.42 | 94.51 | 83.83 | 0.81 |

