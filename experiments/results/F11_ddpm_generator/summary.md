# F11_ddpm_generator: Our method on DDPM-F, the paper's best feature generator

Mode: full

Hypothesis: IID FMNIST: Ours (DDPM-F) > 84.28 (paper's best). LT: Ours (DDPM-F) >> GeFL-F (DDPM-F).

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_bal | final_tail | final_ens_acc | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (DDPM-F) (best_mean_acc) |
|---|---|---|---|---|---|---|---|---|
| Ours (DDPM-F) | 2 | 82.82 ± 0.39 | 82.80 ± 0.38 | 83.53 ± 0.21 | 85.90 ± 0.10 | 58.38 ± 5.13 | 1.00 ± 0.01 | n<2 |
| GeFL-F (DDPM-F) | 1 | 83.16 ± 0.00 | 83.12 ± 0.00 | 83.97 ± 0.00 | 86.60 ± 0.00 | 55.25 ± 0.00 | 1.01 ± 0.00 | - |
| +CSL (DDPM-F) | 1 | 83.04 ± 0.00 | 83.02 ± 0.00 | 83.64 ± 0.00 | 86.21 ± 0.00 | 55.25 ± 0.00 | 1.01 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_bal | final_tail | final_ens_acc | fidelity_tail | cond_norm_tail_over_head_end |
|---|---|---|---|---|---|---|---|
| Ours (DDPM-F) | 0 | 83.10 | 83.07 | 83.68 | 85.97 | 54.75 | 1.01 |
| Ours (DDPM-F) | 1 | 82.55 | 82.54 | 83.38 | 85.83 | 62.00 | 1.00 |
| GeFL-F (DDPM-F) | 0 | 83.16 | 83.12 | 83.97 | 86.60 | 55.25 | 1.01 |
| +CSL (DDPM-F) | 0 | 83.04 | 83.02 | 83.64 | 86.21 | 55.25 | 1.01 |

