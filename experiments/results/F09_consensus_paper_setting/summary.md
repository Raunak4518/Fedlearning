# F09_consensus_paper_setting: Paper setting (IID, Figure 4 protocol): consensus soft labels vs GeFL-F

Mode: full

Hypothesis: +CSL > GeFL-F on best_mean_acc (the paper's metric), beating the Figure 4 value.

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 3 | 96.02 ± 0.04 | 95.76 ± 0.09 | 97.69 ± 0.09 | 95.73 ± 0.08 | - |
| +CSL (beta=0.5) | 3 | 96.71 ± 0.16 | 96.67 ± 0.12 | 97.84 ± 0.06 | 96.65 ± 0.12 | 0.00957 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 3 | 82.67 ± 0.31 | 82.47 ± 0.32 | 86.13 ± 0.20 | 82.47 ± 0.32 | - |
| +CSL (beta=0.5) | 3 | 83.16 ± 0.32 | 83.13 ± 0.30 | 86.45 ± 0.33 | 83.13 ± 0.30 | 0.0155 |

