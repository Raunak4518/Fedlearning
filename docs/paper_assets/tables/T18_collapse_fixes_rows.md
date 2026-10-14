### T18_collapse_fixes_rows: Fixes for row collapse, long tail (tail/head row norm at the end; K15 + F01/F04/K03 references)

*Paper section 3 / 4.1.* Mean ± sd over seeds; p = two-sided t-test against **+ HWA + LA** (paired by seed when per-seed data exist, else Welch).

| | MNIST K=10 | MNIST K=50 | MNIST K=100 | FMNIST K=10 | FMNIST K=50 | FMNIST K=100 |
|---|---|---|---|---|---|---|
| GeFL-F | 0.33 ± 0.34 (p=0.0947) | 0.05 ± 0.03 (p=0.000841) | 0.04 ± 0.01 (p=0.000475) | 0.22 ± 0.28 (p=0.0159) | 0.07 ± 0.02 (p=0.00268) | 0.06 ± 0.02 (p=0.000893) |
| + LA (coupled Adam decay) | 0.33 ± 0.34 (p=0.0947) | 0.05 ± 0.03 (p=0.000841) | 0.04 ± 0.01 (p=0.000475) | 0.22 ± 0.28 (p=0.0159) | 0.07 ± 0.02 (p=0.00268) | 0.06 ± 0.02 (p=0.000893) |
| + AdamW + LA | – | – | – | – | – | – |
| + lazy decay + LA | – | – | – | – | – | – |
| + HWA + LA | 0.80 ± 0.09 | 0.84 ± 0.04 | 0.86 ± 0.03 | 1.01 ± 0.15 | 0.92 ± 0.10 | 0.93 ± 0.03 |

Rows 3-4 fill in when K15 (NB38 / NB39) results are copied back.

**Missing (no run found):** + AdamW + LA / MNIST K=10; + AdamW + LA / MNIST K=50; + AdamW + LA / MNIST K=100; + AdamW + LA / FMNIST K=10; + AdamW + LA / FMNIST K=50; + AdamW + LA / FMNIST K=100; + lazy decay + LA / MNIST K=10; + lazy decay + LA / MNIST K=50; + lazy decay + LA / MNIST K=100; + lazy decay + LA / FMNIST K=10; + lazy decay + LA / FMNIST K=50; + lazy decay + LA / FMNIST K=100
