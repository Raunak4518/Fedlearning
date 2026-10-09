# K03_kaggle_client_scaling: Client scaling K = 50, 100: paper setting (IID) and long tail
(pasted by the user from Kaggle NB5; runs.jsonl not yet copied back)

## fmnist  IF=0.01  alpha=0.5  K=50

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 75.96 ± 0.08 | 75.45 ± 0.19 | 66.07 ± 1.03 | 49.97 ± 6.80 | 0.92 ± 0.10 | 0.00655 |
| Ours (HWA+LA+CSL) | 3 | 75.68 ± 0.29 | 74.48 ± 0.40 | 64.14 ± 1.89 | 49.97 ± 6.80 | 0.92 ± 0.10 | 0.00528 |
| GeFL-F | 3 | 63.26 ± 1.86 | 61.92 ± 1.31 | 35.86 ± 3.65 | 6.33 ± 1.69 | 0.07 ± 0.02 | - |
| +LA | 3 | 67.91 ± 1.25 | 66.53 ± 1.08 | 45.67 ± 2.57 | 6.33 ± 1.69 | 0.07 ± 0.02 | 0.00568 |
| FSG+LA | 3 | 75.38 ± 0.61 | 75.06 ± 0.44 | 67.02 ± 0.70 | 79.42 ± 4.05 | 0.88 ± 0.03 | 0.00352 |
| FSG+LA+CSL | 3 | 75.08 ± 0.64 | 74.90 ± 0.70 | 66.58 ± 0.74 | 79.42 ± 4.05 | 0.88 ± 0.03 | 0.00418 |

## fmnist  IF=1.0  alpha=None  K=50

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 81.30 ± 0.21 | 81.17 ± 0.16 | 81.60 ± 0.52 | 83.42 ± 5.96 | 1.01 ± 0.00 | 0.0931 |
| GeFL-F | 3 | 81.50 ± 0.12 | 81.37 ± 0.16 | 81.57 ± 0.56 | 83.14 ± 5.85 | 1.01 ± 0.00 | - |
| +CSL (beta=0.5) | 3 | 81.29 ± 0.19 | 81.20 ± 0.19 | 81.62 ± 0.74 | 83.14 ± 5.85 | 1.01 ± 0.00 | 0.0688 |

## mnist  IF=1.0  alpha=None  K=50

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 95.53 ± 0.17 | 95.47 ± 0.18 | 93.64 ± 0.13 | 46.67 ± 2.48 | 0.99 ± 0.01 | 0.251 |
| GeFL-F | 3 | 95.43 ± 0.27 | 95.37 ± 0.26 | 93.65 ± 0.17 | 46.31 ± 2.94 | 0.98 ± 0.01 | - |
| +CSL (beta=0.5) | 3 | 95.53 ± 0.18 | 95.47 ± 0.17 | 93.70 ± 0.10 | 46.31 ± 2.94 | 0.98 ± 0.01 | 0.203 |
