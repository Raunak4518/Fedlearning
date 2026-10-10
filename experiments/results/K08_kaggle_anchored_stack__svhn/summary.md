# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): SVHN
(pasted by the user from Kaggle NB16; runs.jsonl not yet copied back. bbc_* columns were printed as fractions, converted to % here.)

## svhn  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 55.89 ± 1.20 | 37.10 ± 3.60 | 61.62 ± 0.90 | 57.00 ± 2.00 | 41.00 ± 5.00 | 61.77 ± 1.19 | 24.28 ± 5.34 | - |
| Ours-A Ts=10 | 3 | 54.41 ± 1.13 | 35.51 ± 3.12 | 62.30 ± 0.39 | 55.00 ± 1.00 | 36.00 ± 3.00 | 59.13 ± 1.29 | 25.94 ± 5.53 | 0.102 |

## svhn  IF=1.0  alpha=None  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 72.52 ± 0.96 | 71.05 ± 1.06 | 73.86 ± 0.60 | 72.00 ± 1.00 | 71.00 ± 1.00 | 73.81 ± 0.67 | 56.33 ± 3.34 | - |
| Ours-A Ts=10 | 3 | 71.82 ± 1.45 | 70.98 ± 1.72 | 74.38 ± 0.79 | 72.00 ± 2.00 | 71.00 ± 2.00 | 73.24 ± 0.98 | 56.72 ± 0.80 | 0.136 |
