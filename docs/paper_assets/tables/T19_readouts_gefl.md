### T19_readouts_gefl: Balanced accuracy under every readout, long tail (K16; same trained heads per row group)

*Paper section 5.11.* Mean ± sd over seeds.

| | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 |
|---|---|---|---|---|---|
| GeFL-F, heads as trained | 74.48 ± 1.67 | 59.16 ± 2.84 | 49.62 ± 3.04 | – | – |
| GeFL-F, BBC | 78.32 ± 1.30 | 63.24 ± 3.44 | 55.51 ± 1.19 | – | – |
| GeFL-F, ER (LDA, all clients) | 90.41 ± 0.52 | 79.28 ± 0.40 | 64.67 ± 0.81 | – | – |
| GeFL-F, Fed3R (all clients) | 75.09 ± 0.39 | 68.20 ± 1.49 | 44.82 ± 1.32 | – | – |
| GeFL-F, FedNCM (all clients) | 87.18 ± 0.79 | 74.11 ± 1.06 | 60.48 ± 1.01 | – | – |
| GeFL-F, CCVR (all clients) | 91.91 ± 0.45 | 80.49 ± 0.30 | 64.81 ± 1.32 | – | – |
| GeFL-F, CCVR (own clients) | 75.27 ± 1.09 | 61.43 ± 2.99 | 49.41 ± 2.89 | – | – |
| GeFL-F, oracle: pooled-real refit | 91.15 ± 0.55 | 79.26 ± 0.39 | 64.28 ± 0.74 | – | – |
| + LA, heads as trained | 79.18 ± 2.59 | 65.20 ± 3.18 | 55.55 ± 1.34 | – | – |
| + LA, BBC | 79.60 ± 2.00 | 64.67 ± 3.32 | 55.97 ± 0.89 | – | – |
| + LA, ER (LDA, all clients) | 90.37 ± 0.78 | 79.49 ± 0.54 | 64.53 ± 1.04 | – | – |
| + LA, Fed3R (all clients) | 75.36 ± 1.41 | 68.41 ± 1.60 | 44.80 ± 1.48 | – | – |
| + LA, FedNCM (all clients) | 87.42 ± 1.21 | 74.42 ± 1.07 | 60.74 ± 1.13 | – | – |
| + LA, CCVR (all clients) | 92.20 ± 0.64 | 80.61 ± 0.40 | 64.86 ± 1.40 | – | – |
| + LA, CCVR (own clients) | 73.04 ± 1.79 | 60.55 ± 3.28 | 47.05 ± 2.93 | – | – |
| + LA, oracle: pooled-real refit | 91.44 ± 0.90 | 79.43 ± 0.41 | 64.30 ± 0.89 | – | – |

CIFAR-10 / CIFAR-100 columns fill in when NB43 / NB44 results are copied back.

**Missing (no run found):** GeFL-F, heads as trained / CIFAR10; GeFL-F, heads as trained / CIFAR100; GeFL-F, BBC / CIFAR10; GeFL-F, BBC / CIFAR100; GeFL-F, ER (LDA, all clients) / CIFAR10; GeFL-F, ER (LDA, all clients) / CIFAR100; GeFL-F, Fed3R (all clients) / CIFAR10; GeFL-F, Fed3R (all clients) / CIFAR100; GeFL-F, FedNCM (all clients) / CIFAR10; GeFL-F, FedNCM (all clients) / CIFAR100; GeFL-F, CCVR (all clients) / CIFAR10; GeFL-F, CCVR (all clients) / CIFAR100; GeFL-F, CCVR (own clients) / CIFAR10; GeFL-F, CCVR (own clients) / CIFAR100; GeFL-F, oracle: pooled-real refit / CIFAR10; GeFL-F, oracle: pooled-real refit / CIFAR100; + LA, heads as trained / CIFAR10; + LA, heads as trained / CIFAR100; + LA, BBC / CIFAR10; + LA, BBC / CIFAR100; + LA, ER (LDA, all clients) / CIFAR10; + LA, ER (LDA, all clients) / CIFAR100; + LA, Fed3R (all clients) / CIFAR10; + LA, Fed3R (all clients) / CIFAR100; + LA, FedNCM (all clients) / CIFAR10; + LA, FedNCM (all clients) / CIFAR100; + LA, CCVR (all clients) / CIFAR10; + LA, CCVR (all clients) / CIFAR100; + LA, CCVR (own clients) / CIFAR10; + LA, CCVR (own clients) / CIFAR100; + LA, oracle: pooled-real refit / CIFAR10; + LA, oracle: pooled-real refit / CIFAR100
