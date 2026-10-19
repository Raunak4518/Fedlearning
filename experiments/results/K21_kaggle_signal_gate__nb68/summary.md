# K21_kaggle_signal_gate: Signal-fraction gate (P1) and class-balanced Stage-I loss (P2), seeds 3-5, paired with K17

Mode: full

Hypothesis: Q1-Q4 in the docstring.

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | final_worst | sfrac_hat_mean | sgate_kept |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours Ts=10 + BALFE | 3 | 95.57 ± 0.46 | 94.78 ± 0.78 | 95.78 ± 0.61 | 0.95 ± 0.01 | 0.94 ± 0.01 | 92.19 ± 1.03 | 88.86 ± 1.63 | - | - |
| Ours Ts=10 + SCSL | 3 | 94.25 ± 0.30 | 94.07 ± 0.35 | 94.91 ± 0.18 | 0.94 ± 0.00 | 0.94 ± 0.01 | 90.35 ± 0.72 | 85.90 ± 2.37 | 1.00 ± 0.00 | - |
| GeFL-F + AdamW + LA + SGATE | 3 | 82.05 ± 6.36 | 92.12 ± 0.06 | 93.33 ± 0.29 | 0.81 ± 0.06 | 0.91 ± 0.00 | 63.40 ± 15.62 | 37.39 ± 27.41 | 0.64 ± 0.10 | 8.67 ± 1.15 |

| label | seed | final_bal | er_bal | ccvr_bal | val_final_bal | val_er_bal | final_tail | final_worst | sfrac_hat_mean | sgate_kept |
|---|---|---|---|---|---|---|---|---|---|---|
| Ours Ts=10 + BALFE | 3 | 95.19 | 94.03 | 95.36 | 0.94 | 0.93 | 91.31 | 87.24 | - | - |
| Ours Ts=10 + BALFE | 4 | 95.42 | 94.72 | 95.50 | 0.95 | 0.94 | 91.93 | 88.84 | - | - |
| Ours Ts=10 + BALFE | 5 | 96.08 | 95.59 | 96.48 | 0.96 | 0.95 | 93.33 | 90.50 | - | - |
| Ours Ts=10 + SCSL | 3 | 94.46 | 93.77 | 94.72 | 0.94 | 0.93 | 89.90 | 85.99 | 1.00 | - |
| Ours Ts=10 + SCSL | 4 | 94.37 | 93.97 | 94.94 | 0.94 | 0.94 | 91.18 | 88.24 | 0.99 | - |
| Ours Ts=10 + SCSL | 5 | 93.91 | 94.45 | 95.07 | 0.94 | 0.94 | 89.96 | 83.49 | 1.00 | - |
| GeFL-F + AdamW + LA + SGATE | 3 | 89.36 | 92.17 | 93.58 | 0.88 | 0.91 | 81.43 | 69.04 | 0.75 | 10.00 |
| GeFL-F + AdamW + LA + SGATE | 4 | 78.90 | 92.13 | 93.01 | 0.78 | 0.91 | 54.97 | 21.65 | 0.59 | 8.00 |
| GeFL-F + AdamW + LA + SGATE | 5 | 77.88 | 92.06 | 93.40 | 0.77 | 0.92 | 53.81 | 21.50 | 0.57 | 8.00 |

