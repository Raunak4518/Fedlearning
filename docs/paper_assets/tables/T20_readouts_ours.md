### T20_readouts_ours: Balanced accuracy under every readout, long tail (K16; same trained heads per row group)

*Paper section 5.11.* Mean ± sd over seeds.

| | MNIST | FMNIST | SVHN | CIFAR10 | CIFAR100 |
|---|---|---|---|---|---|
| minimal, heads as trained | 90.07 ± 0.43 | 76.71 ± 0.25 | 62.44 ± 2.54 | 42.50 ± 0.57 | 10.99 ± 0.80 |
| minimal, BBC | 90.49 ± 0.50 | 78.37 ± 0.15 | 64.06 ± 2.34 | 43.98 ± 0.45 | 11.25 ± 0.83 |
| minimal, ER (LDA, all clients) | 92.41 ± 0.40 | 80.63 ± 0.04 | 66.36 ± 1.03 | 53.42 ± 0.85 | 16.51 ± 1.12 |
| minimal, Fed3R (all clients) | 77.22 ± 0.10 | 72.02 ± 0.38 | 45.37 ± 1.45 | 34.51 ± 0.22 | 10.09 ± 0.63 |
| minimal, FedNCM (all clients) | 90.11 ± 0.40 | 76.72 ± 0.48 | 62.62 ± 1.06 | 48.90 ± 1.02 | 15.07 ± 0.86 |
| minimal, CCVR (all clients) | 94.05 ± 0.16 | 81.75 ± 0.41 | 66.87 ± 1.82 | 52.17 ± 0.56 | 15.96 ± 1.22 |
| minimal, CCVR (own clients) | 81.06 ± 0.98 | 69.63 ± 1.61 | 48.64 ± 0.48 | 36.09 ± 0.61 | 9.94 ± 0.71 |
| minimal, oracle: pooled-real refit | 93.76 ± 0.24 | 81.12 ± 0.40 | 67.30 ± 1.62 | 51.40 ± 0.72 | 15.16 ± 1.20 |
| ours (gated), heads as trained | 95.10 ± 0.17 | 80.06 ± 0.14 | 65.42 ± 1.42 | 45.73 ± 1.59 | 15.37 ± 0.98 |
| ours (gated), BBC | 95.36 ± 0.15 | 80.83 ± 0.06 | 66.73 ± 1.46 | 48.21 ± 1.45 | 15.97 ± 0.95 |
| ours (gated), ER (LDA, all clients) | 94.35 ± 0.18 | 81.49 ± 0.46 | 66.65 ± 0.60 | 53.89 ± 0.96 | 18.59 ± 0.89 |
| ours (gated), Fed3R (all clients) | 81.06 ± 0.63 | 73.57 ± 0.56 | 45.88 ± 1.21 | 34.63 ± 0.36 | 10.77 ± 0.64 |
| ours (gated), FedNCM (all clients) | 93.31 ± 0.29 | 78.60 ± 0.18 | 63.79 ± 1.12 | 49.84 ± 1.05 | 16.57 ± 0.69 |
| ours (gated), CCVR (all clients) | 95.52 ± 0.12 | 82.56 ± 0.43 | 67.16 ± 1.33 | 52.60 ± 0.70 | 17.77 ± 0.94 |
| ours (gated), CCVR (own clients) | 91.26 ± 0.59 | 75.32 ± 0.53 | 51.61 ± 0.60 | 37.42 ± 0.94 | 12.09 ± 0.82 |
| ours (gated), oracle: pooled-real refit | 95.25 ± 0.03 | 81.85 ± 0.40 | 68.26 ± 1.29 | 51.88 ± 0.90 | 17.42 ± 0.94 |

Long tail, IF = 100, Dirichlet 0.5, K = 10, 3 seeds.
