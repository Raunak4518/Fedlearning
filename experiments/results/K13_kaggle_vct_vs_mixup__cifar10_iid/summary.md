# K13_kaggle_vct_vs_mixup: VCT vs MixUp in the final method (CIFAR-10, IID, K = 10)

Mode: full

Hypothesis: Ours-A + VCT > Ours-A + MixUp, and above the paper's GeFL + MixUp (62.67).

## cifar10  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 55.86%

| label | n | best_mean_acc | final_bal | final_tail | bbc_bal | final_ens_acc | fidelity_tail_kh | class_sep_F | ncm_acc | p vs Ours-A Ts=10 + MixUp (best_mean_acc) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10 + MixUp | 3 | 59.98 ± 0.91 | 54.76 ± 1.81 | 63.33 ± 1.49 | 54.62 ± 3.43 | 68.00 ± 1.25 | 46.14 ± 7.00 | 0.06 ± 0.00 | 42.16 ± 0.69 | - |
| Ours-A Ts=10 + VCT | 3 | 60.29 ± 1.24 | 53.18 ± 2.88 | 61.54 ± 3.92 | 53.05 ± 4.06 | 66.84 ± 1.59 | 46.64 ± 7.10 | 0.06 ± 0.00 | 42.16 ± 0.69 | 0.257 |

| label | seed | best_mean_acc | final_bal | final_tail | bbc_bal | final_ens_acc | fidelity_tail_kh | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A Ts=10 + MixUp | 0 | 60.47 | 55.38 | 64.74 | 56.32 | 67.90 | 50.75 | 0.05 | 42.09 |
| Ours-A Ts=10 + MixUp | 1 | 60.54 | 56.17 | 63.47 | 56.87 | 69.30 | 49.58 | 0.06 | 41.51 |
| Ours-A Ts=10 + MixUp | 2 | 58.93 | 52.72 | 61.78 | 50.68 | 66.80 | 38.08 | 0.06 | 42.89 |
| Ours-A Ts=10 + VCT | 0 | 60.84 | 54.87 | 64.07 | 55.50 | 67.70 | 52.00 | 0.05 | 42.09 |
| Ours-A Ts=10 + VCT | 1 | 61.16 | 54.80 | 63.52 | 55.29 | 67.82 | 49.33 | 0.06 | 41.51 |
| Ours-A Ts=10 + VCT | 2 | 58.87 | 49.86 | 57.03 | 48.36 | 65.01 | 38.58 | 0.06 | 42.89 |

