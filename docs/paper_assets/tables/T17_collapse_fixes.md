### T17_collapse_fixes: Fixes for row collapse, long tail (balanced accuracy; K15 + F01/F04/K03 references)

*Paper section 3 / 4.1.* Mean ± sd over seeds; p = two-sided t-test against **+ HWA + LA** (paired by seed when per-seed data exist, else Welch).

| | MNIST K=10 | MNIST K=50 | MNIST K=100 | FMNIST K=10 | FMNIST K=50 | FMNIST K=100 |
|---|---|---|---|---|---|---|
| GeFL-F | 74.62 ± 1.76 (p=0.00368) | 78.10 ± 1.12 (p=0.008) | 75.98 ± 0.95 (p=0.0095) | 58.62 ± 3.41 (p=0.0152) | 61.92 ± 1.31 (p=0.00408) | 61.95 ± 1.52 (p=0.00263) |
| + LA (coupled Adam decay) | 78.92 ± 2.32 (p=0.013) | 81.35 ± 0.82 (p=0.00857) | 79.51 ± 0.76 (p=0.0271) | 64.78 ± 3.35 (p=0.0353) | 66.53 ± 1.08 (p=0.00675) | 66.56 ± 1.94 (p=0.0142) |
| + AdamW + LA | – | – | – | – | – | – |
| + lazy decay + LA | – | – | – | – | – | – |
| + HWA + LA | 87.89 ± 1.61 | 86.46 ± 0.84 | 83.07 ± 0.56 | 76.02 ± 0.67 | 75.45 ± 0.19 | 73.25 ± 0.74 |

Rows 3-4 fill in when K15 (NB38 / NB39) results are copied back.

**Missing (no run found):** + AdamW + LA / MNIST K=10; + AdamW + LA / MNIST K=50; + AdamW + LA / MNIST K=100; + AdamW + LA / FMNIST K=10; + AdamW + LA / FMNIST K=50; + AdamW + LA / FMNIST K=100; + lazy decay + LA / MNIST K=10; + lazy decay + LA / MNIST K=50; + lazy decay + LA / MNIST K=100; + lazy decay + LA / FMNIST K=10; + lazy decay + LA / FMNIST K=50; + lazy decay + LA / FMNIST K=100
