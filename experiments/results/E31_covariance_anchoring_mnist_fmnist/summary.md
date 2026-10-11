# E31_covariance_anchoring_mnist_fmnist: MC-S in the final method on MNIST / FashionMNIST (T_s = 10)

Mode: full

Hypothesis: Ours-A-S Ts=10 >= Ours-A Ts=10 (E25) in every setting.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc | p vs Ours-A-S Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 3 | 79.65 ± 0.91 | 71.37 ± 1.69 | 79.83 ± 0.91 | 79.14 ± 1.20 | 79.89 ± 0.67 | 75.08 ± 1.26 | 0.74 ± 0.02 | 0.60 ± 0.12 | 68.10 ± 1.93 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 0 | 79.81 | 72.28 | 80.08 | 79.74 | 80.00 | 74.67 | 0.73 | 0.71 | 70.16 |
| Ours-A-S Ts=10 | 1 | 80.46 | 72.42 | 80.59 | 79.92 | 80.50 | 76.50 | 0.76 | 0.62 | 67.83 |
| Ours-A-S Ts=10 | 2 | 78.67 | 69.42 | 78.82 | 77.75 | 79.17 | 74.08 | 0.72 | 0.48 | 66.32 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc | p vs Ours-A-S Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 3 | 85.17 ± 0.70 | 85.56 ± 0.26 | 85.25 ± 0.65 | 85.08 ± 0.67 | 83.61 ± 1.43 | 82.00 ± 2.00 | 0.89 ± 0.01 | 0.54 ± 0.08 | 69.17 ± 2.49 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 0 | 84.95 | 85.85 | 85.06 | 84.87 | 83.83 | 79.75 | 0.90 | 0.59 | 70.91 |
| Ours-A-S Ts=10 | 1 | 84.61 | 85.44 | 84.72 | 84.55 | 84.92 | 83.58 | 0.88 | 0.57 | 66.32 |
| Ours-A-S Ts=10 | 2 | 85.96 | 85.38 | 85.98 | 85.84 | 82.08 | 82.67 | 0.89 | 0.45 | 70.28 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc | p vs Ours-A-S Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 3 | 93.25 ± 0.77 | 86.87 ± 1.63 | 94.01 ± 0.71 | 93.08 ± 1.32 | 80.06 ± 2.38 | 82.83 ± 0.93 | 0.87 ± 0.01 | 0.31 ± 0.03 | 82.12 ± 0.51 | - |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A-S Ts=10 | 0 | 94.05 | 88.54 | 94.78 | 94.33 | 81.08 | 83.67 | 0.86 | 0.30 | 82.32 |
| Ours-A-S Ts=10 | 1 | 92.53 | 85.29 | 93.38 | 91.69 | 77.33 | 81.83 | 0.88 | 0.35 | 81.54 |
| Ours-A-S Ts=10 | 2 | 93.19 | 86.77 | 93.86 | 93.20 | 81.75 | 83.00 | 0.86 | 0.29 | 82.49 |

