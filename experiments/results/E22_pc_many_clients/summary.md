# E22_pc_many_clients: Anchored stack at K = 100 clients (MNIST long tail)

Mode: full

Hypothesis: PC + MC does not degrade with K: >= FSG (86.1) at K = 100.

## mnist  IF=0.01  alpha=0.5  K=100

| label | n | final_bal | final_tail | best_mean_acc | fidelity_tail_mc | spread_tail_mc | p vs Ours-PC+MC (final_bal) |
|---|---|---|---|---|---|---|---|
| Ours-PC+MC | 3 | 89.20 ± 0.47 | 81.29 ± 0.46 | 89.55 ± 0.50 | 82.83 ± 0.88 | 0.82 ± 0.00 | - |
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 90.41 ± 0.52 | 83.41 ± 0.89 | 90.60 ± 0.57 | 83.64 ± 1.58 | 0.82 ± 0.01 | 0.00742 |
| Ours-A Ts=10 | 3 | 94.30 ± 0.31 | 90.54 ± 0.28 | 94.45 ± 0.18 | 83.69 ± 0.13 | 0.83 ± 0.02 | 0.000444 |

