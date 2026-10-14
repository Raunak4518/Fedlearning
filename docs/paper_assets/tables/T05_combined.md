### T05_combined: One method for both regimes (F10; LT: final balanced acc., IID: best_mean_acc)

*Paper section 5.5.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | MNIST LT | FMNIST LT | MNIST IID | FMNIST IID |
|---|---|---|---|---|
| GeFL-F | 74.24 ± 1.91 | 58.74 ± 3.94 | 96.01 ± 0.14 | 82.80 ± 0.22 |
| + CSL | 76.35 ± 1.96 (p=0.00312) | 58.59 ± 3.67 (p=0.637) | 96.82 ± 0.07 (p=0.00261) | 83.25 ± 0.31 (p=0.0235) |
| + HWA + LA | 87.78 ± 2.07 (p=0.0132) | 75.99 ± 0.47 (p=0.0197) | 96.01 ± 0.05 (p=1) | 82.74 ± 0.28 (p=0.233) |
| + HWA + LA + CSL (ours) | 89.87 ± 0.96 (p=0.0079) | 76.37 ± 0.64 (p=0.0185) | 96.77 ± 0.06 (p=0.00406) | 83.21 ± 0.28 (p=0.00775) |
