# K11_kaggle_covariance_anchoring: MC-S: exact class-covariance anchoring (SVHN)
(pasted by the user from Kaggle NB22; per-seed values included.)

## svhn  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 55.76 ± 1.68 | 36.82 ± 4.33 | 61.65 ± 1.28 | 57.27 ± 2.04 | 22.33 ± 4.26 | 24.53 ± 6.88 | 0.92 ± 0.01 | 0.01 ± 0.00 | 12.45 ± 2.40 | - |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 3 | 62.89 ± 1.07 | 42.88 ± 0.47 | 68.20 ± 1.06 | 64.57 ± 0.75 | 23.19 ± 3.29 | 29.22 ± 3.40 | 0.95 ± 0.02 | 0.01 ± 0.00 | 13.17 ± 0.76 | 0.0443 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 3 | 61.80 ± 1.28 | 42.07 ± 2.84 | 67.34 ± 1.14 | 63.30 ± 1.49 | 21.56 ± 3.30 | 28.47 ± 4.32 | 0.95 ± 0.02 | 0.01 ± 0.00 | 12.45 ± 2.40 | 0.0155 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 56.76 | 38.81 | 62.27 | 58.41 | 24.92 | 28.33 | 0.92 | 0.01 | 13.23 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 53.82 | 31.85 | 60.18 | 54.91 | 17.42 | 16.58 | 0.91 | 0.01 | 9.76 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 56.71 | 39.79 | 62.49 | 58.49 | 24.67 | 28.67 | 0.92 | 0.01 | 14.36 |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 0 | 61.87 | 42.33 | 67.21 | 63.78 | 22.00 | 32.17 | 0.96 | 0.01 | 12.59 |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 1 | 64.00 | 43.16 | 69.32 | 65.27 | 20.67 | 25.50 | 0.93 | 0.01 | 12.90 |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 2 | 62.78 | 43.15 | 68.07 | 64.65 | 26.92 | 30.00 | 0.97 | 0.01 | 14.03 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 0 | 61.29 | 42.13 | 66.77 | 62.99 | 23.58 | 31.75 | 0.97 | 0.01 | 13.23 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 1 | 60.85 | 39.20 | 66.60 | 62.00 | 17.75 | 23.58 | 0.96 | 0.01 | 9.76 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 2 | 63.25 | 44.88 | 68.65 | 64.92 | 23.33 | 30.08 | 0.93 | 0.01 | 14.36 |

## svhn  IF=1.0  alpha=None  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 3 | 76.35 ± 0.33 | 74.29 ± 0.33 | 77.39 ± 0.41 | 76.27 ± 0.34 | 50.64 ± 0.59 | 52.31 ± 2.97 | 0.96 ± 0.01 | 0.01 ± 0.00 | 17.93 ± 4.86 | 0.0263 |
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 73.04 ± 0.64 | 71.58 ± 0.31 | 74.38 ± 0.66 | 73.02 ± 0.58 | 54.11 ± 0.68 | 56.03 ± 0.89 | 0.92 ± 0.01 | 0.01 ± 0.00 | 18.03 ± 4.26 | - |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 3 | 75.87 ± 0.03 | 73.66 ± 0.30 | 77.05 ± 0.10 | 75.84 ± 0.09 | 45.94 ± 2.13 | 51.14 ± 0.83 | 0.94 ± 0.01 | 0.01 ± 0.00 | 18.03 ± 4.26 | 0.0154 |

| label | seed | final_bal | final_tail | best_mean_acc | bbc_bal | fidelity_tail_mc | fidelity_tail_kh | spread_tail_mc | class_sep_F | ncm_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 0 | 76.37 | 74.50 | 77.26 | 76.28 | 51.25 | 52.67 | 0.95 | 0.01 | 23.54 |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 1 | 76.01 | 73.92 | 77.06 | 75.92 | 50.08 | 49.17 | 0.97 | 0.01 | 15.45 |
| Ours-R-S (PCR+MCS+KH+LA+CSL) | 2 | 76.67 | 74.47 | 77.84 | 76.60 | 50.58 | 55.08 | 0.96 | 0.01 | 14.81 |
| Ours-A (PC+MC+KH+LA+CSL) | 0 | 73.31 | 71.79 | 74.65 | 73.09 | 54.42 | 56.58 | 0.91 | 0.01 | 22.95 |
| Ours-A (PC+MC+KH+LA+CSL) | 1 | 73.49 | 71.73 | 74.87 | 73.56 | 54.58 | 56.50 | 0.92 | 0.01 | 15.57 |
| Ours-A (PC+MC+KH+LA+CSL) | 2 | 72.31 | 71.22 | 73.63 | 72.41 | 53.33 | 55.00 | 0.94 | 0.01 | 15.58 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 0 | 75.90 | 74.01 | 76.98 | 75.93 | 45.25 | 50.83 | 0.94 | 0.01 | 22.95 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 1 | 75.88 | 73.46 | 77.00 | 75.85 | 44.25 | 50.50 | 0.95 | 0.01 | 15.57 |
| Ours-A-S (PC+MCS+KH+LA+CSL) | 2 | 75.85 | 73.51 | 77.16 | 75.75 | 48.33 | 52.08 | 0.93 | 0.01 | 15.58 |
