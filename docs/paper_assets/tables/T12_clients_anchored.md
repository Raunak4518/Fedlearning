### T12_clients_anchored: More clients under the long tail (balanced accuracy; E22 MNIST, K03/K08 FMNIST)

*Paper section 5.11.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | MNIST K=100 | FMNIST K=50 | FMNIST K=100 |
|---|---|---|---|
| GeFL-F | 75.98 ± 0.95 | 61.92 ± 1.31 | 61.95 ± 1.52 |
| CVAE + HWA + LA | 83.07 ± 0.56 (p=0.0095) | 75.45 ± 0.19 (p=0.00408) | 73.25 ± 0.74 (p=0.00263) |
| FSG + LA | 86.05 ± 0.84 (p=5.48e-05) | 75.06 ± 0.44 (p=0.00188) | 74.56 ± 0.84 (p=0.00196) |
| Ours-A, T_s = 1 | 90.41 ± 0.52 (p=0.000298) | 77.24 ± 0.45 (p=0.00284) | 76.20 ± 1.12 (p=0.00486) |
| Ours-A, T_s = 10 | 94.30 ± 0.31 (p=0.000409) | 79.47 ± 0.38 (p=0.00306) | 79.02 ± 0.98 (p=0.00362) |

MNIST GeFL-F/HWA/FSG at K=100 come from F04 (same splits as E22).
