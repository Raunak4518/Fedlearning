# F09_consensus_paper_setting: Paper setting (IID, Figure 4 protocol): consensus soft labels vs GeFL-F

Mode: full

Hypothesis: +CSL > GeFL-F on best_mean_acc (the paper's metric), beating the Figure 4 value.

## mnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 95.47%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 3 | 96.02 ± 0.04 | 95.76 ± 0.09 | 97.69 ± 0.09 | 95.73 ± 0.08 | - |
| +CSL (beta=0.5) | 3 | 96.71 ± 0.16 | 96.67 ± 0.12 | 97.84 ± 0.06 | 96.65 ± 0.12 | 0.00957 |
| +CSL (beta=1) | 1 | 96.10 ± 0.00 | 96.10 ± 0.00 | 96.87 ± 0.00 | 96.08 ± 0.00 | n<2 |
| +CSLM (interleaved) | 1 | 96.16 ± 0.00 | 95.84 ± 0.00 | 97.84 ± 0.00 | 95.79 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_ens_acc | final_bal |
|---|---|---|---|---|---|
| GeFL-F | 0 | 95.97 | 95.70 | 97.59 | 95.67 |
| GeFL-F | 1 | 96.05 | 95.86 | 97.70 | 95.82 |
| GeFL-F | 2 | 96.05 | 95.72 | 97.77 | 95.69 |
| +CSL (beta=0.5) | 0 | 96.54 | 96.53 | 97.78 | 96.51 |
| +CSL (beta=0.5) | 1 | 96.74 | 96.72 | 97.84 | 96.70 |
| +CSL (beta=0.5) | 2 | 96.85 | 96.76 | 97.90 | 96.75 |
| +CSL (beta=1) | 0 | 96.10 | 96.10 | 96.87 | 96.08 |
| +CSLM (interleaved) | 0 | 96.16 | 95.84 | 97.84 | 95.79 |

## mnist  IF=1.0  alpha=None  K=100
Paper Figure 4 GeFL-F (CVAE-F): 94.63%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 1 | 94.61 ± 0.00 | 94.58 ± 0.00 | 96.43 ± 0.00 | 94.53 ± 0.00 | - |
| +CSL (beta=0.5) | 1 | 94.72 ± 0.00 | 94.71 ± 0.00 | 96.34 ± 0.00 | 94.66 ± 0.00 | n<2 |
| +CSL (beta=1) | 1 | 94.49 ± 0.00 | 94.49 ± 0.00 | 95.72 ± 0.00 | 94.44 ± 0.00 | n<2 |
| +CSLM (interleaved) | 1 | 94.36 ± 0.00 | 94.26 ± 0.00 | 96.11 ± 0.00 | 94.20 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_ens_acc | final_bal |
|---|---|---|---|---|---|
| GeFL-F | 0 | 94.61 | 94.58 | 96.43 | 94.53 |
| +CSL (beta=0.5) | 0 | 94.72 | 94.71 | 96.34 | 94.66 |
| +CSL (beta=1) | 0 | 94.49 | 94.49 | 95.72 | 94.44 |
| +CSLM (interleaved) | 0 | 94.36 | 94.26 | 96.11 | 94.20 |

## mnist  IF=1.0  alpha=None  K=50
Paper Figure 4 GeFL-F (CVAE-F): 95.04%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 1 | 95.16 ± 0.00 | 95.16 ± 0.00 | 97.27 ± 0.00 | 95.11 ± 0.00 | - |
| +CSL (beta=0.5) | 1 | 95.41 ± 0.00 | 95.41 ± 0.00 | 97.12 ± 0.00 | 95.37 ± 0.00 | n<2 |
| +CSL (beta=1) | 1 | 95.13 ± 0.00 | 95.12 ± 0.00 | 96.35 ± 0.00 | 95.08 ± 0.00 | n<2 |
| +CSLM (interleaved) | 1 | 95.30 ± 0.00 | 95.30 ± 0.00 | 96.94 ± 0.00 | 95.27 ± 0.00 | n<2 |

| label | seed | best_mean_acc | final_acc | final_ens_acc | final_bal |
|---|---|---|---|---|---|
| GeFL-F | 0 | 95.16 | 95.16 | 97.27 | 95.11 |
| +CSL (beta=0.5) | 0 | 95.41 | 95.41 | 97.12 | 95.37 |
| +CSL (beta=1) | 0 | 95.13 | 95.12 | 96.35 | 95.08 |
| +CSLM (interleaved) | 0 | 95.30 | 95.30 | 96.94 | 95.27 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | best_mean_acc | final_acc | final_ens_acc | final_bal | p vs GeFL-F (best_mean_acc) |
|---|---|---|---|---|---|---|
| GeFL-F | 3 | 82.67 ± 0.31 | 82.47 ± 0.32 | 86.13 ± 0.20 | 82.47 ± 0.32 | - |
| +CSL (beta=0.5) | 3 | 83.16 ± 0.32 | 83.13 ± 0.30 | 86.45 ± 0.33 | 83.13 ± 0.30 | 0.0155 |

| label | seed | best_mean_acc | final_acc | final_ens_acc | final_bal |
|---|---|---|---|---|---|
| GeFL-F | 0 | 82.75 | 82.61 | 86.16 | 82.61 |
| GeFL-F | 1 | 82.33 | 82.10 | 85.91 | 82.10 |
| GeFL-F | 2 | 82.94 | 82.69 | 86.31 | 82.69 |
| +CSL (beta=0.5) | 0 | 83.36 | 83.32 | 86.24 | 83.32 |
| +CSL (beta=0.5) | 1 | 82.79 | 82.78 | 86.27 | 82.78 |
| +CSL (beta=0.5) | 2 | 83.33 | 83.29 | 86.83 | 83.29 |

