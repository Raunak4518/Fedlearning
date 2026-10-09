# E18_bayesian_csl: Bayesian consensus soft labels (product of label prior and ensemble)

Mode: full

Hypothesis: BCSL >= CSL at equal budget; BCSL makes T_s = 10 pay off where CSL did not (MNIST long tail).

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | p vs +CSL Ts=10 (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 90.18 ± 0.96 | 89.74 ± 1.04 | 82.16 ± 3.00 | 28.69 ± 4.47 | - |
| Ours (HWA+LA+CSL) Ts=10 | 3 | 91.70 ± 1.35 | 90.51 ± 1.61 | 84.37 ± 3.37 | 28.69 ± 4.47 | - |
| Ours-B (HWA+LA+BCSL) | 3 | 89.33 ± 1.44 | 88.68 ± 1.67 | 81.16 ± 4.20 | 28.69 ± 4.47 | - |
| Ours-B (HWA+LA+BCSL) Ts=10 | 3 | 89.92 ± 1.35 | 87.32 ± 1.80 | 80.12 ± 3.92 | 28.69 ± 4.47 | - |

