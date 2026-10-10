# K08_kaggle_anchored_stack: Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10 IID
(pasted by the user from Kaggle NB15; runs.jsonl not yet copied back. bbc_* columns were printed as fractions with two decimals, converted to % here; BBC is gated off in IID.)

## cifar10  IF=1.0  alpha=None  K=10

| label | n | final_bal | final_tail | best_mean_acc | bbc_bal | bbc_tail | oracle_bal | fidelity_tail_kh | p vs Ours-A (PC+MC+KH+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| Ours-A (PC+MC+KH+LA+CSL) | 3 | 58.62 ± 1.43 | 67.30 ± 1.97 | 60.39 ± 0.84 | 58.00 ± 2.00 | 67.00 ± 2.00 | 59.12 ± 1.12 | 46.14 ± 6.53 | - |
| Ours-A Ts=10 | 3 | 56.82 ± 2.16 | 65.47 ± 2.77 | 60.93 ± 1.27 | 56.00 ± 2.00 | 65.00 ± 3.00 | 57.92 ± 1.74 | 44.89 ± 5.28 | 0.0514 |
