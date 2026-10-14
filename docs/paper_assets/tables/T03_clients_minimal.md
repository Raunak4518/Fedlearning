### T03_clients_minimal: More clients, MNIST long tail (balanced accuracy; F01/F10 at K = 10, F04 at K = 50/100)

*Paper section 5.3.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | K = 10 | K = 50 | K = 100 |
|---|---|---|---|
| GeFL-F | 74.24 ± 1.91 | 78.10 ± 1.12 | 75.98 ± 0.95 |
| + LA | 78.92 ± 2.32 (p=0.0218) | 81.35 ± 0.82 (p=0.00717) | 79.51 ± 0.76 (p=0.000919) |
| + HWA + LA | 87.78 ± 2.07 (p=0.0132) | 86.46 ± 0.84 (p=0.008) | 83.07 ± 0.56 (p=0.0095) |
| + HWA + LA + CSL | 89.87 ± 0.96 (p=0.0079) | – | 83.04 ± 0.06 (n=2) (p=0.0865) |
| FSG + LA | 87.04 ± 1.63 (p=0.00387) | 87.66 ± 0.96 (p=0.000388) | 86.05 ± 0.84 (p=5.48e-05) |
| FSG + LA + CSL | – | – | 86.34 (1 seed) |
| MIX + HWA + LA + CSL | – | – | 85.29 ± 1.78 (n=2) (p=0.0215) |

**Missing (no run found):** + HWA + LA + CSL / K = 50; FSG + LA + CSL / K = 50; MIX + HWA + LA + CSL / K = 50
