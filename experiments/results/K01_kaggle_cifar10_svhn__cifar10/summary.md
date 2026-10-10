# K01_kaggle_cifar10_svhn: CIFAR-10 and SVHN: paper setting (IID) and long tail, full schedule
(pasted by the user from Kaggle NB1; runs.jsonl not yet copied back)

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | best_mean_acc | final_bal | final_tail | final_ens_acc | fidelity_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 41.81 ± 0.77 | 40.73 ± 0.78 | 28.78 ± 1.55 | 52.22 ± 0.84 | 22.64 ± 0.86 | 0.00169 |
| Ours (HWA+LA+CSL) | 3 | 42.92 ± 0.59 | 42.10 ± 0.74 | 28.33 ± 2.14 | 52.54 ± 1.22 | 22.64 ± 0.86 | 0.00368 |
| GeFL-F | 3 | 34.78 ± 0.62 | 33.74 ± 0.51 | 13.52 ± 0.83 | 43.46 ± 0.76 | 14.03 ± 3.37 | - |
| +LA | 3 | 39.34 ± 0.54 | 37.30 ± 0.59 | 20.73 ± 0.67 | 49.47 ± 1.08 | 14.03 ± 3.37 | 0.00144 |
| FSG+LA | 3 | 44.89 ± 0.80 | 44.17 ± 0.87 | 34.90 ± 1.28 | 54.30 ± 1.22 | 47.39 ± 2.47 | 0.00539 |
| FSG+LA+CSL | 3 | 43.66 ± 0.59 | 42.95 ± 0.56 | 31.52 ± 1.59 | 52.86 ± 1.25 | 47.39 ± 2.47 | 0.00489 |

## cifar10  IF=1.0  alpha=None  K=10

| label | n | best_mean_acc | final_bal | final_tail | final_ens_acc | fidelity_tail | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 60.02 ± 0.43 | 58.61 ± 0.55 | 67.08 ± 0.79 | 69.76 ± 0.14 | 33.44 ± 6.19 | 0.0145 |
| GeFL-F | 3 | 59.10 ± 0.63 | 56.68 ± 0.85 | 64.75 ± 1.41 | 69.09 ± 0.71 | 34.83 ± 5.72 | - |
| +CSL (beta=0.5) | 3 | 59.85 ± 0.50 | 58.51 ± 0.55 | 67.11 ± 0.93 | 69.76 ± 0.32 | 34.83 ± 5.72 | 0.0109 |
