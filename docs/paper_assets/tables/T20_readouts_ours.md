### T20_readouts_ours: Balanced accuracy under every readout, long tail (K16; same trained heads per row group)

*Paper section 5.11.* Mean ± sd over seeds.

| | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 |
|---|---|---|---|---|---|
| minimal, heads as trained | 90.07 ± 0.43 | 76.71 ± 0.25 | 62.44 ± 2.54 | – | – |
| minimal, BBC | 90.49 ± 0.50 | 78.37 ± 0.15 | 64.06 ± 2.34 | – | – |
| minimal, ER (LDA, all clients) | 92.41 ± 0.40 | 80.63 ± 0.04 | 66.36 ± 1.03 | – | – |
| minimal, Fed3R (all clients) | 77.22 ± 0.10 | 72.02 ± 0.38 | 45.37 ± 1.45 | – | – |
| minimal, FedNCM (all clients) | 90.11 ± 0.40 | 76.72 ± 0.48 | 62.62 ± 1.06 | – | – |
| minimal, CCVR (all clients) | 94.05 ± 0.16 | 81.75 ± 0.41 | 66.87 ± 1.82 | – | – |
| minimal, CCVR (own clients) | 81.06 ± 0.98 | 69.63 ± 1.61 | 48.64 ± 0.48 | – | – |
| minimal, oracle: pooled-real refit | 93.76 ± 0.24 | 81.12 ± 0.40 | 67.30 ± 1.62 | – | – |
| ours (gated), heads as trained | 95.10 ± 0.17 | 80.06 ± 0.14 | 65.42 ± 1.42 | – | – |
| ours (gated), BBC | 95.36 ± 0.15 | 80.83 ± 0.06 | 66.73 ± 1.46 | – | – |
| ours (gated), ER (LDA, all clients) | 94.35 ± 0.18 | 81.49 ± 0.46 | 66.65 ± 0.60 | – | – |
| ours (gated), Fed3R (all clients) | 81.06 ± 0.63 | 73.57 ± 0.56 | 45.88 ± 1.21 | – | – |
| ours (gated), FedNCM (all clients) | 93.31 ± 0.29 | 78.60 ± 0.18 | 63.79 ± 1.12 | – | – |
| ours (gated), CCVR (all clients) | 95.52 ± 0.12 | 82.56 ± 0.43 | 67.16 ± 1.33 | – | – |
| ours (gated), CCVR (own clients) | 91.26 ± 0.59 | 75.32 ± 0.53 | 51.61 ± 0.60 | – | – |
| ours (gated), oracle: pooled-real refit | 95.25 ± 0.03 | 81.85 ± 0.40 | 68.26 ± 1.29 | – | – |

CIFAR-10 / CIFAR-100 columns fill in when NB43 / NB44 results are copied back.

**Missing (no run found):** minimal, heads as trained / CIFAR10; minimal, heads as trained / CIFAR100; minimal, BBC / CIFAR10; minimal, BBC / CIFAR100; minimal, ER (LDA, all clients) / CIFAR10; minimal, ER (LDA, all clients) / CIFAR100; minimal, Fed3R (all clients) / CIFAR10; minimal, Fed3R (all clients) / CIFAR100; minimal, FedNCM (all clients) / CIFAR10; minimal, FedNCM (all clients) / CIFAR100; minimal, CCVR (all clients) / CIFAR10; minimal, CCVR (all clients) / CIFAR100; minimal, CCVR (own clients) / CIFAR10; minimal, CCVR (own clients) / CIFAR100; minimal, oracle: pooled-real refit / CIFAR10; minimal, oracle: pooled-real refit / CIFAR100; ours (gated), heads as trained / CIFAR10; ours (gated), heads as trained / CIFAR100; ours (gated), BBC / CIFAR10; ours (gated), BBC / CIFAR100; ours (gated), ER (LDA, all clients) / CIFAR10; ours (gated), ER (LDA, all clients) / CIFAR100; ours (gated), Fed3R (all clients) / CIFAR10; ours (gated), Fed3R (all clients) / CIFAR100; ours (gated), FedNCM (all clients) / CIFAR10; ours (gated), FedNCM (all clients) / CIFAR100; ours (gated), CCVR (all clients) / CIFAR10; ours (gated), CCVR (all clients) / CIFAR100; ours (gated), CCVR (own clients) / CIFAR10; ours (gated), CCVR (own clients) / CIFAR100; ours (gated), oracle: pooled-real refit / CIFAR10; ours (gated), oracle: pooled-real refit / CIFAR100
