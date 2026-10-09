# K04_kaggle_ddpm_cifar10_svhn: Our method on DDPM-F: SVHN and CIFAR-10, paper's IID setting
(pasted by the user from Kaggle NB6; runs.jsonl not yet copied back)

## svhn  IF=1.0  alpha=None  K=10

| label | n | best_mean_acc | final_bal | final_ens_acc | fidelity_tail | p vs GeFL-F (DDPM-F) (best_mean_acc) |
|---|---|---|---|---|---|---|
| Ours (DDPM-F) | 3 | 70.37 ± 0.65 | 68.69 ± 0.58 | 80.08 ± 0.46 | 28.36 ± 5.94 | 0.0987 |
| GeFL-F (DDPM-F) | 3 | 69.77 ± 0.85 | 67.86 ± 1.08 | 79.52 ± 0.17 | 28.89 ± 5.01 | - |
| +CSL (DDPM-F) | 3 | 70.25 ± 0.68 | 68.57 ± 0.76 | 80.02 ± 0.16 | 28.89 ± 5.01 | 0.0454 |
