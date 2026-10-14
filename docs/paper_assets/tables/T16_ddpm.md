### T16_ddpm: Generator-agnostic: on the paper's DDPM-F (IID, best_mean_acc; F11, K04)

*Paper section 5.8.* Mean ± sd over seeds; p = two-sided t-test against **GeFL-F (DDPM-F)** (paired by seed when per-seed data exist, else Welch).

| | FMNIST | SVHN |
|---|---|---|
| GeFL-F (DDPM-F) | 83.16 (1 seed) | 69.77 ± 0.85 |
| + CSL (DDPM-F) | 83.04 (1 seed) | 70.25 ± 0.68 (p=0.0454) |
| Ours (DDPM-F) | 82.82 ± 0.39 (n=2) | 70.37 ± 0.65 (p=0.0987) |
| paper: GeFL-F DDPM-F | 84.28 | – |
