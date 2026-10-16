### T21_readouts_iid: Balanced accuracy under every readout, balanced (IID) setting (K16)

*Paper section 5.4 / 5.11.* Mean ± sd over seeds.

| | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 |
|---|---|---|---|---|---|
| GeFL-F, heads as trained | 95.67 ± 0.03 | 83.04 ± 0.34 | 73.81 ± 0.52 | 56.79 ± 0.63 | 12.52 ± 0.21 |
| GeFL-F, BBC | 95.54 ± 0.11 | 82.74 ± 0.26 | 73.73 ± 0.57 | 56.54 ± 0.68 | 12.62 ± 0.20 |
| GeFL-F, ER (LDA, all clients) | 96.87 ± 0.11 | 84.46 ± 0.38 | 76.16 ± 0.50 | 61.44 ± 0.67 | 19.77 ± 0.39 |
| GeFL-F, Fed3R (all clients) | 96.76 ± 0.08 | 84.64 ± 0.37 | 74.51 ± 0.67 | 60.86 ± 0.58 | 18.07 ± 0.18 |
| GeFL-F, FedNCM (all clients) | 94.62 ± 0.05 | 80.33 ± 0.47 | 70.20 ± 0.27 | 56.55 ± 0.82 | 16.91 ± 0.15 |
| GeFL-F, CCVR (all clients) | 96.79 ± 0.10 | 84.53 ± 0.35 | 74.48 ± 0.65 | 60.06 ± 0.63 | 18.21 ± 0.44 |
| GeFL-F, CCVR (own clients) | 96.03 ± 0.13 | 83.01 ± 0.31 | 70.66 ± 0.84 | 55.45 ± 0.57 | 12.16 ± 0.18 |
| GeFL-F, oracle: pooled-real refit | 96.59 ± 0.08 | 83.83 ± 0.38 | 74.93 ± 0.32 | 58.20 ± 0.58 | 16.63 ± 0.34 |
| minimal, heads as trained | 96.68 ± 0.04 | 83.54 ± 0.29 | 74.65 ± 0.68 | 58.58 ± 0.43 | 14.46 ± 0.13 |
| minimal, BBC | 96.48 ± 0.13 | 82.95 ± 0.14 | 74.64 ± 0.68 | 58.25 ± 0.38 | 14.50 ± 0.12 |
| minimal, ER (LDA, all clients) | 97.12 ± 0.13 | 84.50 ± 0.42 | 76.39 ± 0.65 | 62.12 ± 0.54 | 20.09 ± 0.26 |
| minimal, Fed3R (all clients) | 97.04 ± 0.14 | 84.64 ± 0.33 | 74.72 ± 0.78 | 61.57 ± 0.52 | 18.41 ± 0.11 |
| minimal, FedNCM (all clients) | 94.99 ± 0.06 | 80.32 ± 0.42 | 70.38 ± 0.56 | 56.89 ± 0.82 | 16.79 ± 0.10 |
| minimal, CCVR (all clients) | 97.20 ± 0.06 | 84.64 ± 0.46 | 75.31 ± 0.67 | 60.83 ± 0.42 | 18.85 ± 0.19 |
| minimal, CCVR (own clients) | 96.71 ± 0.10 | 83.37 ± 0.24 | 72.08 ± 1.00 | 56.81 ± 0.36 | 13.40 ± 0.09 |
| minimal, oracle: pooled-real refit | 97.20 ± 0.08 | 84.23 ± 0.34 | 75.57 ± 0.57 | 59.32 ± 0.58 | 17.41 ± 0.09 |
| ours (gated), heads as trained | 97.36 ± 0.17 | 83.97 ± 1.04 | 76.00 ± 0.05 | 57.82 ± 1.41 | 18.00 ± 0.33 |
| ours (gated), BBC | 97.38 ± 0.14 | 83.87 ± 0.94 | 75.95 ± 0.14 | 57.65 ± 1.49 | 17.74 ± 0.44 |
| ours (gated), ER (LDA, all clients) | 97.43 ± 0.12 | 84.79 ± 0.77 | 77.30 ± 0.21 | 61.56 ± 0.79 | 21.41 ± 0.10 |
| ours (gated), Fed3R (all clients) | 97.31 ± 0.14 | 84.91 ± 0.75 | 75.59 ± 0.22 | 61.00 ± 0.74 | 19.47 ± 0.24 |
| ours (gated), FedNCM (all clients) | 95.31 ± 0.16 | 80.79 ± 0.80 | 71.58 ± 0.30 | 56.34 ± 1.38 | 17.62 ± 0.31 |
| ours (gated), CCVR (all clients) | 97.52 ± 0.10 | 84.91 ± 0.78 | 76.23 ± 0.25 | 60.16 ± 1.04 | 19.93 ± 0.04 |
| ours (gated), CCVR (own clients) | 97.29 ± 0.16 | 83.81 ± 0.95 | 73.42 ± 0.33 | 56.26 ± 1.39 | 15.87 ± 0.31 |
| ours (gated), oracle: pooled-real refit | 97.60 ± 0.07 | 84.33 ± 1.04 | 76.80 ± 0.14 | 58.37 ± 1.02 | 19.07 ± 0.24 |

SVHN / CIFAR-100 columns fill in when NB46 / NB48 results are copied back.
