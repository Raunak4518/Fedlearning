# Result slots: status

Written by `python docs/papers/fill.py` from 348 runs in fedecs_lt/results (NB00 excluded). Slots not listed under 'filled' still show a red placeholder.

## Group `main`: 78 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]
- CIFAR-100-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]

## Group `nat`: 208 runs, 6 settings

- DermaMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]
- OrganAMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]
- BloodMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]
- PathMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]
- EMNIST (ByClass) (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 5]  **(fewer than 3 seeds)**
- FER-2013 (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 13 methods, seeds [3, 4, 5]

## Group `t2`: 62 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 12 methods, seeds [3, 4, 5]
- CIFAR-100-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 12 methods, seeds [3, 4]  **(fewer than 3 seeds)**

## Slots in the paper

- Tables filled: 11 of 14; still empty: `abl_bal`, `abl_components`, `robust`
- Numbers filled: 38 of 49; still empty: `abl/cifar10/GeFL-F + LA/gain`, `abl/cifar10/rest`, `main/fmnist/margin`, `main/fmnist/strongest`, `main/mnist/margin`, `main/mnist/strongest`, `main/svhn/margin`, `main/svhn/strongest`, `repr/cifar10/FedECS-LT/knn`, `repr/cifar10/GeFL-F/knn`, `t2/cifar10-r100/fe200_diff`
- Figures filled: 0 of 9; still empty: `fig_forest`, `fig_hmt`, `fig_overview`, `fig_realgen`, `fig_recall_size`, `fig_rows`, `fig_shared_rounds`, `fig_spread`, `fig_tsne`

## Notebooks not returned yet (by folder)

- 1_main_comparison_mnist_fmnist_svhn_cifar10_cifar100: NB10_main_mnist, NB10_main_fmnist, NB10_main_svhn
- 2_one_shared_model_IF100_and_ablations: NB30_ablation_cifar10, NB31_components_cifar10, NB30_ablation_cifar100, NB31_components_cifar100, NB20_track2_cifar10_IF100_s5_part1of2, NB20_track2_cifar100_IF100_s5_part1of2, NB20_track2_cifar100_IF100_s5_part2of2
- 3_main_comparison_other_datasets: NB10_main_gtsrb_part2of3, NB10_main_gtsrb_part3of3
- 4_one_shared_model_IF050_IF010_and_long_runs: NB20_track2_cifar10_IF010_s3_part3of3, NB20_track2_cifar10_IF050_s3_part1of2, NB20_track2_cifar10_IF050_s3_part2of2, NB20_track2_cifar100_IF010_s3_part2of3, NB20_track2_cifar100_IF010_s3_part3of3, NB20_track2_cifar100_IF050_s3_part1of2, NB20_track2_cifar100_IF050_s3_part2of2, NB29_track2_long_runs_1
- 5_reruns_of_failed_runs: NB10_main_emnist_rerun_7bfd7d, NB20_track2_rerun_f2bb89, NB20_track2_rerun_2a90fa
- 6_robustness_sweeps: NB40_sweep_cifar10_a0.1, NB40_sweep_cifar10_a1.0, NB40_sweep_cifar10_K50, NB40_sweep_cifar10_K100, NB40_sweep_cifar10_IF050, NB40_sweep_cifar10_IF010
