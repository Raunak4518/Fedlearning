# E25_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC) with LA + CSL

Mode: full

Hypothesis: beats every earlier variant under the long tail (3 seeds), no IID loss; T_s = 10 adds on top.

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 96.65 ± 0.01 | 95.46 ± 0.06 | 96.67 ± 0.01 | 0.97 ± 0.00 | 0.95 ± 0.00 | 97.02 ± 0.05 | - |
| Ours-A Ts=10 | 3 | 97.58 ± 0.09 | 96.66 ± 0.12 | 97.62 ± 0.06 | 0.98 ± 0.00 | 0.97 ± 0.00 | 97.78 ± 0.05 | 0.00387 |

