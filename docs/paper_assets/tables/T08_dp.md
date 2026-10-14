### T08_dp: Formal DP on every statistic the anchored method adds (E26; LT, K = 10)

*Paper section 5.6.* Mean ± sd over seeds; p = two-sided t-test against **final method, exact statistics** (paired by seed when per-seed data exist, else Welch).

| | MNIST | FMNIST |
|---|---|---|
| GeFL-F (no formal privacy) | 75.25 ± 1.12 (p=0.00249) | 58.47 ± 3.59 (p=0.011) |
| final method, exact statistics | 94.82 ± 0.57 | 79.67 ± 0.54 |
| (8, 1e-5)-DP statistics | 90.36 ± 0.30 (p=0.00581) | 72.77 ± 1.67 (p=0.0221) |
| (2, 1e-5)-DP statistics | 85.06 ± 0.54 (p=0.0021) | 65.59 ± 0.58 (p=0.00176) |
