# K13_kaggle_vct_vs_mixup: VCT vs MixUp in the final method (CIFAR-10, IID, K = 10)

Mode: full

Hypothesis: Ours-A + VCT > Ours-A + MixUp, and above the paper's GeFL + MixUp (62.67).

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_tail | bbc_bal | final_ens_acc | fidelity_tail_kh | class_sep_F | ncm_acc | p vs Ours-A Ts=10 + MixUp (best_mean_acc) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A + MixUp | 3 | 47.76 ± 1.07 | 46.20 ± 1.13 | 41.23 ± 1.81 | 48.50 ± 0.98 | 59.00 ± 2.22 | 36.86 ± 1.61 | 0.07 ± 0.01 | 40.94 ± 0.41 | - |
| Ours-A + VCT | 3 | 48.09 ± 0.54 | 45.65 ± 0.55 | 51.25 ± 0.52 | 46.89 ± 0.83 | 59.49 ± 1.17 | 36.17 ± 2.35 | 0.07 ± 0.01 | 40.94 ± 0.41 | - |

| label | seed | best_mean_acc | final_bal | final_tail | bbc_bal | final_ens_acc | fidelity_tail_kh | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A + MixUp | 0 | 47.95 | 45.69 | 42.04 | 48.39 | 59.63 | 35.58 | 0.07 | 40.84 |
| Ours-A + MixUp | 1 | 46.60 | 45.41 | 39.17 | 47.58 | 56.54 | 36.33 | 0.07 | 40.60 |
| Ours-A + MixUp | 2 | 48.72 | 47.50 | 42.50 | 49.53 | 60.84 | 38.67 | 0.08 | 41.39 |
| Ours-A + VCT | 0 | 48.51 | 46.12 | 50.67 | 47.69 | 60.26 | 35.58 | 0.07 | 40.84 |
| Ours-A + VCT | 1 | 47.48 | 45.04 | 51.67 | 46.03 | 58.14 | 34.17 | 0.07 | 40.60 |
| Ours-A + VCT | 2 | 48.29 | 45.79 | 51.41 | 46.95 | 60.07 | 38.75 | 0.08 | 41.39 |

