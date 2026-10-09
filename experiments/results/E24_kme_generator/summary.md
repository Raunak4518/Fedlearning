# E24_kme_generator: KME-Gen: server-trained generator from exact federated kernel mean embeddings

Mode: full

Hypothesis: KME-Gen >= PC+MC (E20) under the long tail, with no client-side generator training.

## fmnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | fidelity_tail | spread_tail | p vs Ours-KME (KME+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|
| Ours-KME (KME+LA+CSL) | 1 | 75.11 ± 0.00 | 71.16 ± 0.00 | 75.38 ± 0.00 | 72.75 ± 0.00 | 0.96 ± 0.00 | - |
| Ours-KME+MC | 1 | 75.05 ± 0.00 | 71.11 ± 0.00 | 75.33 ± 0.00 | 72.75 ± 0.00 | 0.96 ± 0.00 | n<2 |
| Ours-KME+KH | 1 | 75.31 ± 0.00 | 71.31 ± 0.00 | 75.71 ± 0.00 | 72.75 ± 0.00 | 0.96 ± 0.00 | n<2 |
| Ours-PC+MC+KH | 1 | 77.48 ± 0.00 | 72.45 ± 0.00 | 78.08 ± 0.00 | 83.58 ± 0.00 | 0.40 ± 0.00 | n<2 |

## fmnist  IF=1.0  alpha=None  K=10
Paper Figure 4 GeFL-F (CVAE-F): 83.14%

| label | n | final_bal | final_tail | best_mean_acc | fidelity_tail | spread_tail | p vs Ours-KME (KME+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|
| Ours-KME (KME+LA+CSL) | 1 | 81.76 ± 0.00 | 82.48 ± 0.00 | 81.82 ± 0.00 | 65.00 ± 0.00 | 0.99 ± 0.00 | - |
| Ours-KME+MC | 1 | 81.65 ± 0.00 | 82.35 ± 0.00 | 81.78 ± 0.00 | 65.00 ± 0.00 | 0.99 ± 0.00 | n<2 |
| Ours-KME+KH | 1 | 81.70 ± 0.00 | 82.41 ± 0.00 | 81.83 ± 0.00 | 65.00 ± 0.00 | 0.99 ± 0.00 | n<2 |
| Ours-PC+MC+KH | 1 | 83.46 ± 0.00 | 84.08 ± 0.00 | 83.50 ± 0.00 | 88.58 ± 0.00 | 0.49 ± 0.00 | n<2 |

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | final_tail | best_mean_acc | fidelity_tail | spread_tail | p vs Ours-KME (KME+LA+CSL) (final_bal) |
|---|---|---|---|---|---|---|---|
| Ours-KME (KME+LA+CSL) | 1 | 89.88 ± 0.00 | 84.43 ± 0.00 | 90.17 ± 0.00 | 84.17 ± 0.00 | 0.97 ± 0.00 | - |
| Ours-KME+MC | 1 | 90.08 ± 0.00 | 84.52 ± 0.00 | 90.33 ± 0.00 | 84.17 ± 0.00 | 0.97 ± 0.00 | n<2 |
| Ours-KME+KH | 1 | 89.55 ± 0.00 | 84.29 ± 0.00 | 89.77 ± 0.00 | 84.17 ± 0.00 | 0.97 ± 0.00 | n<2 |
| Ours-PC+MC+KH | 1 | 92.91 ± 0.00 | 89.59 ± 0.00 | 93.10 ± 0.00 | 92.25 ± 0.00 | 0.47 ± 0.00 | n<2 |

