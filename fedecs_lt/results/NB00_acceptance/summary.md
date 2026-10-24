# NB00_acceptance (Kaggle, T4 x2, 24 Oct 2026): ACCEPTANCE PASS

Values transcribed from `NB00_acceptance__report.pdf` (Tables III, IX, XVII, XXV); `runs.jsonl` not yet returned.
Seed 3; 2 runs per GPU, so times include GPU sharing. Code version: commit 9da4630.

| Check | Same splits | Bal. acc. new / old (tol.) | Exact readout new / old | F new / old |
|---|---|---|---|---|
| MNIST GeFL-F | yes | 74.2 / 75.0 (8.3) | 90.5 / 90.4 | - |
| MNIST FedECS-LT | yes | 94.3 / 94.5 (2.0) | 93.5 / 93.8 | 0.325 / 0.314 |
| CIFAR-10 GeFL-F | yes | 32.3 / 33.0 (2.4) | 49.3 / 50.3 | - |
| CIFAR-10 FedECS-LT | yes | 40.7 / 42.3 (3.9) | 51.0 / 52.3 | 0.082 / 0.076 |

| Run | Time (s) | Stage I (s) | Stage II (s) | Comm./client (MB) |
|---|---|---|---|---|
| CIFAR-10 FedECS-LT | 6,546 | 433 | 4,288 | 11,268 |
| CIFAR-10 GeFL-F | 6,713 | 433 | 6,161 | 8,659 |
| CIFAR-10 FedTGP | 7,565 | 433 (shared) | - | 18.8 |
| Track 2 CIFAR-10-LT FedECS-LT (K = 20, 40%) | 7,409 | 3,101 | 1,070 | 2,984 |
| Track 2 FedAvg (LongTailLib) | 12,018 | - | - | - |
| MNIST FedECS-LT | 2,066 | 99 | 1,161 | 4,739 |
| MNIST GeFL-F | 1,651 | 99 | 1,267 | 4,296 |

Track 2, seed 3 (one seed only): FedECS-LT bal. acc. 59.6, tail 48.0, worst class 33.7, ECE 18.5, exact readout
56.9; FedAvg (LongTailLib) 61.1, 44.2, 19.4, 28.6.

Problems found (fixed in commit 7e0f391): two CIFAR-10 runs failed once on a half-written dataset file (concurrent
downloads; both completed on retry); 'Peak GPU' measured only the server process and was never reset (values in
this report are not valid).
