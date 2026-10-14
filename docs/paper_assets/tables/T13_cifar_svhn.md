### T13_cifar_svhn: CIFAR-10 and SVHN (LT: final balanced acc., IID: best_mean_acc; K01, K08, K11)

*Paper section 5.11.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | CIFAR-10 LT | SVHN LT | CIFAR-10 IID | SVHN IID |
|---|---|---|---|---|
| GeFL-F | 33.74 ± 0.51 | 50.49 ± 2.24 | 59.10 ± 0.63 | 75.82 ± 0.55 |
| + HWA + LA | 40.73 ± 0.78 (p=0.000477) | 62.45 ± 1.76 (p=0.0279) | – | 75.64 ± 0.51 (p=0.0287) |
| minimal variant / + CSL | 42.10 ± 0.74 (p=0.00019) | – | 59.85 ± 0.50 (p=0.185) | 76.30 ± 0.30 (p=0.0851) |
| FSG + LA | 44.17 ± 0.87 (p=0.000242) | 44.23 ± 2.71 (p=0.0399) | – | – |
| Ours-A, T_s = 1 | 46.06 ± 1.29 (p=0.00123) | 55.89 ± 1.20 (p=0.112) | 60.39 ± 0.84 (p=0.105) | 73.86 ± 0.60 (p=0.0732) |
| Ours-A, T_s = 1, + gated BBC | 48.71 ± 0.96 (p=0.000143) | 57.19 ± 1.56 (p=0.0923) | – | – |
| Ours-A, T_s = 10 | 44.38 ± 1.07 (p=0.000761) | 54.41 ± 1.13 (p=0.17) | 60.93 ± 1.27 (p=0.112) | 74.38 ± 0.79 (p=0.187) |
| Ours-A, T_s = 10 + VCT | 45.65 ± 0.55 (p=1.12e-05) | – | 60.29 ± 1.24 (p=0.236) | – |
| Ours-A, T_s = 10 + MixUp | 46.20 ± 1.13 (p=0.000645) | – | 59.98 ± 0.91 (p=0.25) | – |
| paper: GeFL-F / best of ten | – | – | 55.86 | 76.26 |

K13 long-tail arms run at T_s = 1. paper best on CIFAR-10 IID is 59.36 (62.67 with MixUp).
