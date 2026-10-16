### T21_readouts_iid: Balanced accuracy under every readout, balanced (IID) setting (K16)

*Paper section 5.4 / 5.11.* Mean ± sd over seeds.

| | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 |
|---|---|---|---|---|---|
| GeFL-F, heads as trained | 95.67 ± 0.03 | 83.04 ± 0.34 | – | 56.79 ± 0.63 | – |
| GeFL-F, BBC | 95.54 ± 0.11 | 82.74 ± 0.26 | – | 56.54 ± 0.68 | – |
| GeFL-F, ER (LDA, all clients) | 96.87 ± 0.11 | 84.46 ± 0.38 | – | 61.44 ± 0.67 | – |
| GeFL-F, Fed3R (all clients) | 96.76 ± 0.08 | 84.64 ± 0.37 | – | 60.86 ± 0.58 | – |
| GeFL-F, FedNCM (all clients) | 94.62 ± 0.05 | 80.33 ± 0.47 | – | 56.55 ± 0.82 | – |
| GeFL-F, CCVR (all clients) | 96.79 ± 0.10 | 84.53 ± 0.35 | – | 60.06 ± 0.63 | – |
| GeFL-F, CCVR (own clients) | 96.03 ± 0.13 | 83.01 ± 0.31 | – | 55.45 ± 0.57 | – |
| GeFL-F, oracle: pooled-real refit | 96.59 ± 0.08 | 83.83 ± 0.38 | – | 58.20 ± 0.58 | – |
| minimal, heads as trained | 96.68 ± 0.04 | 83.54 ± 0.29 | – | 58.58 ± 0.43 | – |
| minimal, BBC | 96.48 ± 0.13 | 82.95 ± 0.14 | – | 58.25 ± 0.38 | – |
| minimal, ER (LDA, all clients) | 97.12 ± 0.13 | 84.50 ± 0.42 | – | 62.12 ± 0.54 | – |
| minimal, Fed3R (all clients) | 97.04 ± 0.14 | 84.64 ± 0.33 | – | 61.57 ± 0.52 | – |
| minimal, FedNCM (all clients) | 94.99 ± 0.06 | 80.32 ± 0.42 | – | 56.89 ± 0.82 | – |
| minimal, CCVR (all clients) | 97.20 ± 0.06 | 84.64 ± 0.46 | – | 60.83 ± 0.42 | – |
| minimal, CCVR (own clients) | 96.71 ± 0.10 | 83.37 ± 0.24 | – | 56.81 ± 0.36 | – |
| minimal, oracle: pooled-real refit | 97.20 ± 0.08 | 84.23 ± 0.34 | – | 59.32 ± 0.58 | – |
| ours (gated), heads as trained | 97.36 ± 0.17 | 83.97 ± 1.04 | – | 57.82 ± 1.41 | – |
| ours (gated), BBC | 97.38 ± 0.14 | 83.87 ± 0.94 | – | 57.65 ± 1.49 | – |
| ours (gated), ER (LDA, all clients) | 97.43 ± 0.12 | 84.79 ± 0.77 | – | 61.56 ± 0.79 | – |
| ours (gated), Fed3R (all clients) | 97.31 ± 0.14 | 84.91 ± 0.75 | – | 61.00 ± 0.74 | – |
| ours (gated), FedNCM (all clients) | 95.31 ± 0.16 | 80.79 ± 0.80 | – | 56.34 ± 1.38 | – |
| ours (gated), CCVR (all clients) | 97.52 ± 0.10 | 84.91 ± 0.78 | – | 60.16 ± 1.04 | – |
| ours (gated), CCVR (own clients) | 97.29 ± 0.16 | 83.81 ± 0.95 | – | 56.26 ± 1.39 | – |
| ours (gated), oracle: pooled-real refit | 97.60 ± 0.07 | 84.33 ± 1.04 | – | 58.37 ± 1.02 | – |

SVHN / CIFAR-100 columns fill in when NB46 / NB48 results are copied back.

**Missing (no run found):** GeFL-F, heads as trained / SVHN; GeFL-F, heads as trained / CIFAR100; GeFL-F, BBC / SVHN; GeFL-F, BBC / CIFAR100; GeFL-F, ER (LDA, all clients) / SVHN; GeFL-F, ER (LDA, all clients) / CIFAR100; GeFL-F, Fed3R (all clients) / SVHN; GeFL-F, Fed3R (all clients) / CIFAR100; GeFL-F, FedNCM (all clients) / SVHN; GeFL-F, FedNCM (all clients) / CIFAR100; GeFL-F, CCVR (all clients) / SVHN; GeFL-F, CCVR (all clients) / CIFAR100; GeFL-F, CCVR (own clients) / SVHN; GeFL-F, CCVR (own clients) / CIFAR100; GeFL-F, oracle: pooled-real refit / SVHN; GeFL-F, oracle: pooled-real refit / CIFAR100; minimal, heads as trained / SVHN; minimal, heads as trained / CIFAR100; minimal, BBC / SVHN; minimal, BBC / CIFAR100; minimal, ER (LDA, all clients) / SVHN; minimal, ER (LDA, all clients) / CIFAR100; minimal, Fed3R (all clients) / SVHN; minimal, Fed3R (all clients) / CIFAR100; minimal, FedNCM (all clients) / SVHN; minimal, FedNCM (all clients) / CIFAR100; minimal, CCVR (all clients) / SVHN; minimal, CCVR (all clients) / CIFAR100; minimal, CCVR (own clients) / SVHN; minimal, CCVR (own clients) / CIFAR100; minimal, oracle: pooled-real refit / SVHN; minimal, oracle: pooled-real refit / CIFAR100; ours (gated), heads as trained / SVHN; ours (gated), heads as trained / CIFAR100; ours (gated), BBC / SVHN; ours (gated), BBC / CIFAR100; ours (gated), ER (LDA, all clients) / SVHN; ours (gated), ER (LDA, all clients) / CIFAR100; ours (gated), Fed3R (all clients) / SVHN; ours (gated), Fed3R (all clients) / CIFAR100; ours (gated), FedNCM (all clients) / SVHN; ours (gated), FedNCM (all clients) / CIFAR100; ours (gated), CCVR (all clients) / SVHN; ours (gated), CCVR (all clients) / CIFAR100; ours (gated), CCVR (own clients) / SVHN; ours (gated), CCVR (own clients) / CIFAR100; ours (gated), oracle: pooled-real refit / SVHN; ours (gated), oracle: pooled-real refit / CIFAR100
