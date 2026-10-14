### T11_anchored: The anchored method (K = 10; LT: final balanced acc., IID: best_mean_acc)

*Paper section 5.11.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F** (paired by seed when per-seed data exist, else Welch).

| | MNIST LT | FMNIST LT | MNIST IID | FMNIST IID |
|---|---|---|---|---|
| GeFL-F | 75.25 ± 1.12 | 58.47 ± 3.59 | 96.01 ± 0.14 | 82.76 ± 0.33 |
| minimal variant (CVAE + HWA + LA + CSL) | 90.40 ± 1.42 (p=0.00148) | 76.60 ± 0.21 (p=0.0125) | 96.77 ± 0.06 (p=0.00406) | 83.19 ± 0.47 (p=0.0538) |
| PC-VAE + MC | 92.31 ± 0.61 (p=0.0013) | 77.56 ± 0.76 (p=0.0118) | – | 82.92 ± 0.66 (p=0.481) |
| KME-Gen (1 seed) | 89.88 (1 seed) | 75.11 (1 seed) | – | 81.82 (1 seed) |
| Ours-A, T_s = 1 | 93.11 ± 0.42 (p=0.0018) | 78.22 ± 0.69 (p=0.0135) | 96.67 ± 0.01 (p=0.0153) | 83.35 ± 0.22 (p=0.0216) |
| Ours-A, T_s = 10 | 94.82 ± 0.57 (p=0.00249) | 79.67 ± 0.54 (p=0.011) | 97.62 ± 0.06 (p=0.00138) | 83.96 ± 0.39 (p=0.0214) |
| Ours-A, T_s = 10, + gated BBC | 95.17 ± 0.55 (p=0.00234) | 80.55 ± 0.65 (p=0.011) | – | – |
| Ours-A-B (MC-B), T_s = 10 | 94.76 ± 1.95 (p=0.00786) | 81.57 ± 0.31 (p=0.0075) | – | 85.75 ± 0.45 (p=0.00482) |
| oracle: heads re-fit on pooled real data | 94.93 ± 0.70 (p=0.00285) | 81.55 ± 0.76 (p=0.011) | – | – |
