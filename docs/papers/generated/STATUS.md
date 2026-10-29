# Result slots: status

Written by `python docs/papers/fill.py` from 350 runs in fedecs_lt/results (NB00 excluded). Slots not listed under 'filled' still show a red placeholder.

## Group `main`: 84 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- CIFAR-100-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]

## Group `nat`: 224 runs, 6 settings

- DermaMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- OrganAMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- BloodMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- PathMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- EMNIST (ByClass) (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 5]  **(fewer than 3 seeds)**
  - methods with fewer seeds: Balanced Softmax [3]; FedDF [3]; FedECS-LT [3]; FedECS-LT (heads only) [3]; FedGH [3]; FedGen [3]; FedLC [3]; FedMRL [5]; FedProto [3]; FedRS [3]; FedTGP [3]; GeFL-F [3]; LG-FedAvg [5]; Local [3]
- FER-2013 (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]

## Group `t2`: 67 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 13 methods, seeds [3, 4, 5]
  - methods with fewer seeds: CCVR [3, 4]; Fed-GraB [3, 4]; FedECS-LT [3, 4]; FedECS-LT (heads only) [3, 4]; FedLC [3, 4]; FedYoYo [3, 4]; RUCR [3, 4]
- CIFAR-100-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 13 methods, seeds [3, 4]  **(fewer than 3 seeds)**
  - methods with fewer seeds: FedECS-LT [4]; FedECS-LT (heads only) [4]; FedLoGe [4]

## Slots in the paper

- Tables filled: 14 of 17; still empty: `abl_bal`, `abl_components`, `robust`
- Numbers filled: 137 of 144; still empty: `abl/cifar10/GeFL-F + LA/gain`, `abl/cifar10/rest`, `main/cifar10/Centralised (pooled data)/bal`, `main/cifar10/Centralised (pooled data)/er_bal`, `main/cifar100/Centralised (pooled data)/bal`, `main/cifar100/Centralised (pooled data)/er_bal`, `t2/cifar10-r100/fe200_diff`
- Figures filled: 9 of 9; still empty: none

## Notebooks not returned yet (by folder)

- 1_main_comparison_mnist_fmnist_svhn_cifar10_cifar100: NB10_main_mnist, NB10_main_fmnist, NB10_main_svhn
- 2_one_shared_model_IF100_and_ablations: NB30_ablation_cifar10, NB31_components_cifar10, NB30_ablation_cifar100, NB31_components_cifar100, NB20_track2_cifar10_IF100_s5_part1of2, NB20_track2_cifar100_IF100_s5_part1of2, NB20_track2_cifar100_IF100_s5_part2of2
- 3_main_comparison_other_datasets: NB10_main_gtsrb_part2of3, NB10_main_gtsrb_part3of3
- 4_one_shared_model_IF050_IF010_and_long_runs: NB20_track2_cifar10_IF010_s3_part3of3, NB20_track2_cifar10_IF050_s3_part1of2, NB20_track2_cifar10_IF050_s3_part2of2, NB20_track2_cifar100_IF010_s3_part2of3, NB20_track2_cifar100_IF010_s3_part3of3, NB20_track2_cifar100_IF050_s3_part1of2, NB20_track2_cifar100_IF050_s3_part2of2
- 5_reruns_of_failed_runs: NB10_main_emnist_rerun_7bfd7d, NB20_track2_rerun_f2bb89, NB20_track2_rerun_2a90fa
- 6_robustness_sweeps: NB40_sweep_cifar10_a0.1, NB40_sweep_cifar10_a1.0, NB40_sweep_cifar10_K50, NB40_sweep_cifar10_K100, NB40_sweep_cifar10_IF050, NB40_sweep_cifar10_IF010
- 7_gefl_protocol_grouped_baselines_10_and_50_clients: cifar10_10clients_grouped_baselines, cifar10_50clients_all_methods_part1of3, cifar10_50clients_all_methods_part2of3, cifar10_50clients_all_methods_part3of3, cifar100_10clients_grouped_baselines, cifar100_50clients_all_methods_part1of2, cifar100_50clients_all_methods_part2of2, cifar10_cifar100_pooled_data_reference
