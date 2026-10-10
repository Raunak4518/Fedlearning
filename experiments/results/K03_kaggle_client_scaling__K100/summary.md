# K03_kaggle_client_scaling: Client scaling K = 50, 100: paper setting (IID) and long tail
(pasted by the user from Kaggle NB4; runs.jsonl not yet copied back. CSL rows predate the per-client synthetic-slice fix.)

## fmnist  IF=0.01  alpha=0.5  K=100

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| +HWA+LA | 3 | 73.87 ± 0.85 | 73.25 ± 0.74 | 60.69 ± 1.61 | 44.25 ± 2.85 | 0.93 ± 0.03 | 0.00219 |
| Ours (HWA+LA+CSL) | 3 | 73.32 ± 1.13 | 71.24 ± 0.34 | 56.64 ± 2.69 | 44.25 ± 2.85 | 0.93 ± 0.03 | 0.00262 |
| GeFL-F | 3 | 62.61 ± 1.50 | 61.95 ± 1.52 | 35.40 ± 2.78 | 7.39 ± 1.51 | 0.06 ± 0.02 | - |
| +LA | 3 | 67.64 ± 1.65 | 66.56 ± 1.94 | 45.37 ± 3.57 | 7.39 ± 1.51 | 0.06 ± 0.02 | 0.00266 |
| FSG+LA | 3 | 74.70 ± 0.88 | 74.56 ± 0.84 | 65.33 ± 1.88 | 74.47 ± 7.79 | 0.87 ± 0.03 | 0.00253 |
| FSG+LA+CSL | 3 | 74.09 ± 0.70 | 73.77 ± 0.95 | 63.68 ± 2.42 | 74.47 ± 7.79 | 0.87 ± 0.03 | 0.00355 |

## fmnist  IF=1.0  alpha=None  K=100

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 80.76 ± 0.34 | 80.50 ± 0.47 | 81.31 ± 0.58 | 77.75 ± 11.77 | 1.01 ± 0.00 | 0.0648 |
| GeFL-F | 3 | 81.11 ± 0.20 | 80.97 ± 0.20 | 81.25 ± 0.83 | 77.22 ± 11.89 | 1.01 ± 0.01 | - |
| +CSL (beta=0.5) | 3 | 80.70 ± 0.31 | 80.54 ± 0.33 | 81.44 ± 0.62 | 77.22 ± 11.89 | 1.01 ± 0.01 | 0.0259 |

## mnist  IF=1.0  alpha=None  K=100

| label | n | best_mean_acc | final_bal | final_tail | fidelity_tail | cond_norm_tail_over_head_end | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|---|
| Ours (HWA+LA+CSL) | 3 | 94.87 ± 0.23 | 94.80 ± 0.23 | 93.17 ± 0.19 | 37.72 ± 4.73 | 0.98 ± 0.01 | 0.875 |
| GeFL-F | 3 | 94.88 ± 0.29 | 94.79 ± 0.28 | 92.94 ± 0.29 | 37.11 ± 5.38 | 0.98 ± 0.01 | - |
| +CSL (beta=0.5) | 3 | 94.89 ± 0.24 | 94.82 ± 0.24 | 93.23 ± 0.15 | 37.11 ± 5.38 | 0.98 ± 0.01 | 0.903 |
