### T07_k02_ablations: Ablations and privacy of the minimal variant (IF = 100; K02; balanced accuracy)

*Paper section 5.6.* Mean ± sd over seeds; p = two-sided t-test against **Ours** (paired by seed when per-seed data exist, else Welch).

| | MNIST | FMNIST |
|---|---|---|
| GeFL-F | 75.31 ± 1.11 (p=0.00114) | 58.90 ± 3.62 (p=0.0176) |
| Ours | 90.15 ± 0.94 | 76.51 ± 0.49 |
| no HWA (flat averaging) | 83.08 ± 1.76 (n=2) (p=0.031) | 68.44 ± 1.08 (n=2) (p=0.0692) |
| no LA (plain CE) | 85.46 ± 2.14 (n=2) (p=0.0845) | 68.71 ± 2.34 (n=2) (p=0.145) |
| no CSL | 87.64 ± 1.89 (n=2) (p=0.115) | 76.52 ± 0.03 (n=2) (p=0.147) |
| HWA weights = counts n | 89.97 ± 1.82 (n=2) (p=0.926) | 76.45 ± 0.01 (n=2) (p=0.261) |
| HWA weights = uniform over holders | 89.02 ± 2.14 (n=2) (p=0.351) | 76.33 ± 0.28 (n=2) (p=0.559) |
| CSL beta=0.25 | 88.91 ± 1.82 (n=2) (p=0.213) | 76.05 ± 0.21 (n=2) (p=0.206) |
| CSL beta=0.75 | 90.70 ± 1.56 (n=2) (p=0.174) | 75.74 ± 0.04 (n=2) (p=0.0716) |
| LA tau=1.5 | 90.06 ± 1.30 (n=2) (p=0.156) | 77.76 ± 0.53 (n=2) (p=0.118) |
| LA tau=2.0 | 89.72 ± 1.21 (n=2) (p=0.105) | 77.94 ± 0.23 (n=2) (p=0.0301) |
| Ours, DP eps=10 | 90.10 ± 0.99 (p=0.802) | 76.40 ± 0.47 (p=0.217) |
| Ours, DP eps=1 | 88.87 ± 0.65 (p=0.11) | 76.33 ± 0.04 (p=0.554) |
| Ours, DP eps=0.1 | 86.73 ± 0.82 (p=0.00383) | 73.03 ± 0.80 (p=0.0381) |
