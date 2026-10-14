# K10_kaggle_hybrid_svhn_cifar10: Hybrid generator PCR vs CVAE+HWA (+MC+KH) on SVHN and CIFAR-10

Mode: full

Hypothesis: Ours-R >= max(CVAE + HWA, Ours-A) on SVHN and CIFAR-10 in both regimes.

## svhn  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | fidelity_tail | fidelity_tail_kh | p vs Ours+MC+KH (CVAE+HWA) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-R | 3 | 60.86 ± 0.84 | 42.79 ± 2.82 | 65.97 ± 1.00 | 0.62 ± 0.01 | 0.01 ± 0.00 | 0.12 ± 0.00 | 34.78 ± 4.23 | 32.39 ± 2.35 | 0.0242 |
| Ours+MC+KH (CVAE+HWA) | 3 | 65.60 ± 1.50 | 49.60 ± 2.27 | 70.03 ± 1.37 | 0.67 ± 0.02 | 0.01 ± 0.00 | 0.12 ± 0.01 | 36.25 ± 5.54 | 41.72 ± 3.08 | - |
| Ours-R Ts=10 | 3 | 60.51 ± 0.59 | 42.04 ± 2.53 | 66.58 ± 0.83 | 0.61 ± 0.01 | 0.01 ± 0.00 | 0.12 ± 0.00 | 34.78 ± 4.23 | 31.36 ± 4.13 | 0.0229 |

## svhn  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 76.26%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | class_sep_F | ncm_acc | fidelity_tail | fidelity_tail_kh | p vs Ours+MC+KH (CVAE+HWA) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours+MC+KH (CVAE+HWA) | 3 | 75.89 ± 0.44 | 74.14 ± 1.09 | 77.07 ± 0.19 | 0.76 ± 0.00 | 0.01 ± 0.00 | 0.18 ± 0.05 | 68.03 ± 1.28 | 67.28 ± 2.45 | - |
| Ours-R | 3 | 74.89 ± 0.21 | 73.18 ± 0.91 | 76.06 ± 0.22 | 0.75 ± 0.00 | 0.01 ± 0.00 | 0.18 ± 0.04 | 69.36 ± 4.63 | 66.36 ± 3.22 | 0.0802 |
| Ours-R Ts=10 | 3 | 74.95 ± 0.19 | 73.83 ± 0.31 | 76.50 ± 0.08 | 0.75 ± 0.00 | 0.01 ± 0.00 | 0.18 ± 0.04 | 69.36 ± 4.63 | 66.33 ± 3.84 | 0.0227 |

