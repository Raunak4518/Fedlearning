# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10 long tail
(pasted by the user from Kaggle NB14; runs.jsonl not yet copied back. bbc_* columns were printed as fractions with two decimals, converted to % here.)

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 46.06 ± 1.29 | 37.73 ± 1.67 | 46.78 ± 1.20 | 49.00 ± 1.00 | 48.00 ± 2.00 | 52.25 ± 0.74 | 41.39 ± 4.52 | - |
| Ours-A Ts=10 | 3 | 44.38 ± 1.07 | 34.24 ± 2.34 | 48.07 ± 0.96 | 47.00 ± 1.00 | 46.00 ± 4.00 | 50.64 ± 0.85 | 40.44 ± 3.26 | 0.0277 |
