### T04_iid_csl: GeFL-F's own IID setting: consensus soft labels (best_mean_acc; F09, K01)

*Paper section 5.4.* Mean ± sd over seeds.

| | Paper GeFL-F | Our GeFL-F | + CSL |
|---|---|---|---|
| MNIST, K=10 | 95.47 | 96.02 ± 0.04 | 96.71 ± 0.16 |
| FMNIST, K=10 | 83.14 | 82.67 ± 0.31 | 83.16 ± 0.32 |
| MNIST, K=50 | 95.04 | 95.16 (1 seed) | 95.41 (1 seed) |
| MNIST, K=100 | 94.63 | 94.61 (1 seed) | 94.72 (1 seed) |
| SVHN, K=10 | 76.26 | 75.82 ± 0.55 | 76.30 ± 0.30 |
| CIFAR-10, K=10 | 55.86 | 59.10 ± 0.63 | 59.85 ± 0.50 |

p for + CSL vs our GeFL-F is computed per row in numbers.csv.
