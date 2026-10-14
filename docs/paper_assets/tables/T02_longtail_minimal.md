### T02_longtail_minimal: Long-tailed clients (IF = 100, Dir 0.5, K = 10; F01)

*Paper section 5.3.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | MNIST bal. | MNIST tail | FMNIST bal. | FMNIST tail |
|---|---|---|---|---|
| FedAvg (per architecture) | 62.86 ± 1.84 (p=0.02) | 40.77 ± 2.27 (p=0.0719) | 52.79 ± 2.97 (p=0.0104) | 33.68 ± 3.25 (p=0.371) |
| LG-FedAvg | 63.22 ± 2.18 (p=0.0252) | 41.98 ± 2.33 (p=0.0803) | 52.45 ± 2.93 (p=0.0215) | 33.26 ± 3.92 (p=0.375) |
| LG-FedAvg + LA | 67.84 ± 1.16 (p=0.0521) | 50.62 ± 1.61 (p=0.682) | 57.57 ± 2.40 (p=0.213) | 40.98 ± 4.22 (p=0.0219) |
| GeFL-F | 74.62 ± 1.76 | 51.94 ± 6.05 | 58.62 ± 3.41 | 35.43 ± 5.67 |
| GeFL-F + LA | 78.92 ± 2.32 (p=0.00572) | 60.09 ± 6.62 (p=0.00772) | 64.78 ± 3.35 (p=0.00102) | 45.55 ± 5.44 (p=0.00047) |
| GeFL-F + HWA + LA | 87.89 ± 1.61 (p=0.00368) | 81.08 ± 3.04 (p=0.0086) | 76.02 ± 0.67 (p=0.0152) | 68.81 ± 1.79 (p=0.0134) |
