# K04_kaggle_ddpm_cifar10_svhn: Our method on DDPM-F: SVHN and CIFAR-10, paper's IID setting

Mode: full

Hypothesis: Ours (DDPM-F) > GeFL-F (DDPM-F) on both; SVHN > 76.26.

## svhn  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 76.26%

| label | n | best_mean_acc | final_bal | final_ens_acc | fidelity_tail | p vs GeFL-F (DDPM-F) (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (DDPM-F) | 3 | 70.37 ± 0.65 | 68.69 ± 0.58 | 80.08 ± 0.46 | 28.36 ± 5.94 | 0.0987 |
| GeFL-F (DDPM-F) | 3 | 69.77 ± 0.85 | 67.86 ± 1.08 | 79.52 ± 0.17 | 28.89 ± 5.01 | - |
| +CSL (DDPM-F) | 3 | 70.25 ± 0.68 | 68.57 ± 0.76 | 80.02 ± 0.16 | 28.89 ± 5.01 | 0.0454 |

| label | seed | best_mean_acc | final_bal | final_ens_acc | fidelity_tail |
|---|---|---|---|---|---|
| Ours (DDPM-F) | 0 | 69.63 | 68.10 | 79.96 | 27.08 |
| Ours (DDPM-F) | 1 | 70.82 | 69.27 | 79.69 | 23.17 |
| Ours (DDPM-F) | 2 | 70.66 | 68.70 | 80.59 | 34.83 |
| GeFL-F (DDPM-F) | 0 | 68.92 | 66.72 | 79.55 | 25.67 |
| GeFL-F (DDPM-F) | 1 | 70.62 | 68.87 | 79.68 | 26.33 |
| GeFL-F (DDPM-F) | 2 | 69.79 | 67.99 | 79.34 | 34.67 |
| +CSL (DDPM-F) | 0 | 69.54 | 67.79 | 80.20 | 25.67 |
| +CSL (DDPM-F) | 1 | 70.89 | 69.32 | 79.93 | 26.33 |
| +CSL (DDPM-F) | 2 | 70.32 | 68.59 | 79.91 | 34.67 |

