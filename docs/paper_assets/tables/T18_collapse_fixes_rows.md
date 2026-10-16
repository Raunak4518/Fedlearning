### T18_collapse_fixes_rows: Fixes for row collapse, long tail (tail/head row norm at the end; K15 + F01/F04/K03 references)

*Paper section 3 / 4.1.* Mean ± sd over seeds; p = two-sided t-test against **+ HWA + LA** (paired by seed when per-seed data exist, else Welch).

| | MNIST K=10 | MNIST K=50 | MNIST K=100 | FMNIST K=10 | FMNIST K=50 | FMNIST K=100 |
|---|---|---|---|---|---|---|
| GeFL-F | 0.33 ± 0.34 (p=0.0947) | 0.05 ± 0.03 (p=0.000841) | 0.04 ± 0.01 (p=0.000475) | 0.22 ± 0.28 (p=0.0159) | 0.07 ± 0.02 (p=0.00268) | 0.06 ± 0.02 (p=0.000893) |
| + LA (coupled Adam decay) | 0.33 ± 0.34 (p=0.0947) | 0.05 ± 0.03 (p=0.000841) | 0.04 ± 0.01 (p=0.000475) | 0.22 ± 0.28 (p=0.0159) | 0.07 ± 0.02 (p=0.00268) | 0.06 ± 0.02 (p=0.000893) |
| + AdamW (1e-3, matched) + LA | 0.88 ± 0.07 (p=0.0158) | 0.86 ± 0.01 (p=0.374) | 0.89 ± 0.01 (p=0.276) | 0.96 ± 0.06 (p=0.464) | 0.88 ± 0.02 (p=0.431) | 0.89 ± 0.01 (p=0.16) |
| + AdamW (1e-2) + LA | 0.87 ± 0.05 (p=0.0597) | – | – | 0.97 ± 0.06 (p=0.482) | – | – |
| + no decay on class rows + LA | 0.87 ± 0.06 (p=0.0514) | 0.86 ± 0.01 (p=0.355) | 0.89 ± 0.01 (p=0.286) | 0.96 ± 0.05 (p=0.451) | 0.88 ± 0.02 (p=0.489) | 0.89 ± 0.01 (p=0.17) |
| + lazy decay + LA | 0.73 ± 0.04 (p=0.146) | 0.83 ± 0.01 (p=0.675) | 0.88 ± 0.01 (p=0.344) | 0.78 ± 0.01 (p=0.099) | 0.83 ± 0.01 (p=0.211) | 0.87 ± 0.01 (p=0.0928) |
| + HWA + LA | 0.80 ± 0.09 | 0.84 ± 0.04 | 0.86 ± 0.03 | 1.01 ± 0.15 | 0.92 ± 0.10 | 0.93 ± 0.03 |

Rows 3-6 fill in when K15 (NB38 / NB39) results are copied back.
