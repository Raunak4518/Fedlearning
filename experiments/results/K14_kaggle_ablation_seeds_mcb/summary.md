# K14_kaggle_ablation_seeds_mcb: K09 ablation and MC-B at seeds 3-4 (MNIST / FMNIST long tail)

Mode: full

Hypothesis: every removal lowers balanced accuracy in 5/5 seeds; Ours-A-B >= Ours-A in 5/5 seeds (with K09 / E32 seeds 0-2).

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | bbc_bal | best_mean_acc | final_worst | fidelity_tail_kh | spread_tail_kh | p vs Ours-A Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| -PC (CVAE+HWA) | 2 | 79.96 ± 0.69 | 73.37 ± 0.37 | 80.61 ± 0.39 | 80.44 ± 0.81 | 33.36 ± 2.91 | 78.88 ± 2.30 | 0.97 ± 0.00 | - |

| label | seed | final_bal | final_tail | bbc_bal | best_mean_acc | final_worst | fidelity_tail_kh | spread_tail_kh |
|---|---|---|---|---|---|---|---|---|
| -PC (CVAE+HWA) | 3 | 79.48 | 73.63 | 80.33 | 79.86 | 31.30 | 80.50 | 0.96 |
| -PC (CVAE+HWA) | 4 | 80.44 | 73.11 | 80.89 | 81.01 | 35.42 | 77.25 | 0.97 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | bbc_bal | best_mean_acc | final_worst | fidelity_tail_kh | spread_tail_kh | p vs Ours-A Ts=10 (final_bal) |
|---|---|---|---|---|---|---|---|---|---|
| -PC (CVAE+HWA) | 2 | 92.50 ± 0.49 | 86.81 ± 1.57 | 91.93 ± 0.44 | 93.76 ± 0.46 | 81.14 ± 2.21 | 79.88 ± 0.53 | 0.89 ± 0.00 | - |

| label | seed | final_bal | final_tail | bbc_bal | best_mean_acc | final_worst | fidelity_tail_kh | spread_tail_kh |
|---|---|---|---|---|---|---|---|---|
| -PC (CVAE+HWA) | 3 | 92.85 | 87.91 | 92.24 | 94.08 | 82.70 | 79.50 | 0.90 |
| -PC (CVAE+HWA) | 4 | 92.16 | 85.70 | 91.62 | 93.43 | 79.57 | 80.25 | 0.89 |

