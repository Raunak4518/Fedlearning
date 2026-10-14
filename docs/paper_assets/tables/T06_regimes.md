### T06_regimes: The gain grows with imbalance (balanced accuracy, K = 10; K02, minimal variant)

*Paper section 5.6.* Mean ± sd over seeds.

| | MNIST GeFL-F | MNIST ours | FMNIST GeFL-F | FMNIST ours |
|---|---|---|---|---|
| label skew only (IF = 1, Dir 0.5) | 89.79 ± 1.63 | 95.59 ± 0.28 | 74.06 ± 1.96 | 79.19 ± 0.16 |
| mild tail (IF = 10) | 86.91 ± 1.85 | 94.34 ± 0.19 | 72.63 ± 2.30 | 79.41 ± 0.21 |
| heavy tail (IF = 100) | 75.31 ± 1.11 | 90.15 ± 0.94 | 58.90 ± 3.62 | 76.51 ± 0.49 |
