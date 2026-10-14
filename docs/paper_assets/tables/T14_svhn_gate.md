### T14_svhn_gate: SVHN: MC-S, the hybrid and the rows (K = 10; K01, K08, K10, K11)

*Paper section 5.11.* Mean ± sd over seeds; p = two-sided t-test against **Ours-A (PC + MC + KH)** (paired by seed when per-seed data exist, else Welch).

| | long tail | + gated BBC | IID best_mean_acc |
|---|---|---|---|
| GeFL-F | 50.49 ± 2.24 (p=0.117) | – | 75.82 ± 0.55 (p=0.153) |
| + HWA + LA | 62.45 ± 1.76 (p=0.0571) | – | 75.64 ± 0.51 (p=0.181) |
| Ours-A (PC + MC + KH) | 55.76 ± 1.68 | 57.27 ± 2.04 | 74.38 ± 0.66 |
| Ours-A-S (PC + MC-S + KH) | 61.80 ± 1.28 (p=0.0155) | 63.30 ± 1.49 (p=0.015) | 77.05 ± 0.10 (p=0.0259) |
| hybrid PCR + MC + KH | 60.86 ± 0.84 (p=0.0703) | 62.08 ± 1.21 (p=0.11) | 76.06 ± 0.22 (p=0.077) |
| hybrid PCR + MC-S + KH | 62.89 ± 1.07 (p=0.0443) | 64.57 ± 0.75 (p=0.0421) | 77.39 ± 0.41 (p=0.0396) |
| rows (CVAE + HWA) + MC + KH | 65.60 ± 1.50 (p=0.0189) | 66.96 ± 1.82 (p=0.0239) | 77.07 ± 0.19 (p=0.0131) |
| paper: GeFL-F / best of ten | – | – | 76.26 |
