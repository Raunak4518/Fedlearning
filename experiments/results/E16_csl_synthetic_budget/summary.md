# E16_csl_synthetic_budget: Consensus labels and the synthetic budget T_s

Mode: full

Hypothesis: CSL gains from T_s = 3 / 5; GeFL-F does not (paper Fig. 10).

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | best_mean_acc | final_bal | final_ens_acc | oracle_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) Ts=5 | 3 | 96.62 ± 0.12 | 96.26 ± 0.07 | 97.37 ± 0.21 | 96.89 ± 0.11 | 0.0202 |
| GeFL-F | 3 | 95.97 ± 0.08 | 95.65 ± 0.09 | 97.55 ± 0.20 | 96.57 ± 0.12 | - |
| GeFL-F Ts=3 | 3 | 95.81 ± 0.14 | 94.92 ± 0.01 | 97.07 ± 0.05 | 96.06 ± 0.10 | 0.054 |
| GeFL-F Ts=5 | 3 | 95.71 ± 0.05 | 94.66 ± 0.07 | 96.86 ± 0.19 | 95.87 ± 0.12 | 0.0231 |
| +CSL | 3 | 96.67 ± 0.13 | 96.61 ± 0.10 | 97.89 ± 0.17 | 97.15 ± 0.16 | 0.0247 |
| +CSL Ts=3 | 3 | 96.65 ± 0.11 | 96.41 ± 0.11 | 97.53 ± 0.19 | 97.01 ± 0.17 | 0.0198 |
| +CSL Ts=5 | 3 | 96.54 ± 0.14 | 96.15 ± 0.07 | 97.31 ± 0.19 | 96.82 ± 0.15 | 0.0388 |
| +CSL Ts=10 | 3 | 96.49 ± 0.10 | 95.97 ± 0.10 | 97.20 ± 0.16 | 96.62 ± 0.16 | 0.0328 |

