# K21_kaggle_signal_gate: Signal-fraction gate (P1) and class-balanced Stage-I loss (P2), seeds 3-5, paired with K17

Mode: full

Hypothesis: Q1-Q4 in the docstring.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | final_worst | sfrac_hat_mean | sgate_kept |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours Ts=10 + BALFE | 3 | 79.24 ± 1.37 | 81.20 ± 0.87 | 82.34 ± 0.69 | 0.80 ± 0.01 | 0.82 ± 0.01 | 73.56 ± 1.41 | 31.28 ± 3.87 | - | - |
| Ours Ts=10 + SCSL | 3 | 78.82 ± 1.25 | 80.95 ± 0.97 | 81.48 ± 0.68 | 0.80 ± 0.01 | 0.82 ± 0.01 | 72.41 ± 2.04 | 32.16 ± 2.74 | 0.99 ± 0.00 | - |
| GeFL-F + AdamW + LA + SGATE | 3 | 76.25 ± 0.61 | 80.40 ± 0.80 | 81.37 ± 0.36 | 0.77 ± 0.01 | 0.81 ± 0.01 | 68.14 ± 0.48 | 28.48 ± 5.01 | 0.89 ± 0.04 | 10.00 ± 0.00 |

| label | seed | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | final_worst | sfrac_hat_mean | sgate_kept |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours Ts=10 + BALFE | 3 | 77.73 | 80.73 | 81.57 | 0.78 | 0.82 | 73.99 | 29.09 | - | - |
| Ours Ts=10 + BALFE | 4 | 80.41 | 82.20 | 82.90 | 0.81 | 0.82 | 74.70 | 35.75 | - | - |
| Ours Ts=10 + BALFE | 5 | 79.58 | 80.67 | 82.55 | 0.80 | 0.81 | 71.98 | 29.01 | - | - |
| Ours Ts=10 + SCSL | 3 | 77.63 | 80.63 | 80.74 | 0.79 | 0.82 | 72.67 | 31.89 | 1.00 | - |
| Ours Ts=10 + SCSL | 4 | 80.13 | 82.04 | 82.05 | 0.81 | 0.82 | 74.29 | 35.03 | 1.00 | - |
| Ours Ts=10 + SCSL | 5 | 78.69 | 80.18 | 81.65 | 0.80 | 0.81 | 70.25 | 29.56 | 0.99 | - |
| GeFL-F + AdamW + LA + SGATE | 3 | 75.65 | 80.56 | 81.04 | 0.76 | 0.82 | 68.40 | 24.61 | 0.93 | 10.00 |
| GeFL-F + AdamW + LA + SGATE | 4 | 76.23 | 81.10 | 81.76 | 0.77 | 0.82 | 67.58 | 34.14 | 0.88 | 10.00 |
| GeFL-F + AdamW + LA + SGATE | 5 | 76.87 | 79.53 | 81.31 | 0.77 | 0.80 | 68.44 | 26.70 | 0.86 | 10.00 |

