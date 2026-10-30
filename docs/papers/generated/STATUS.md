# Result slots: status

Written by `python docs/papers/fill.py` from 654 runs in fedecs_lt/results (NB00 excluded). Slots not listed under 'filled' still show a red placeholder.

## Group `main`: 234 runs, 5 settings

- MNIST-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- Fashion-MNIST-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- SVHN-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- CIFAR-10-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 21 methods, seeds [3, 4, 5]
- CIFAR-100-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 15 methods, seeds [3, 4, 5]

## Group `nat`: 224 runs, 6 settings

- DermaMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- OrganAMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- BloodMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- PathMNIST (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]
- EMNIST (ByClass) (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 5]  **(fewer than 3 seeds)**
  - methods with fewer seeds: Balanced Softmax [3]; FedDF [3]; FedECS-LT [3]; FedECS-LT (heads only) [3]; FedGH [3]; FedGen [3]; FedLC [3]; FedMRL [5]; FedProto [3]; FedRS [3]; FedTGP [3]; GeFL-F [3]; LG-FedAvg [5]; Local [3]
- FER-2013 (natural class imbalance), 10 clients, Dirichlet label split with alpha = 0.5: 14 methods, seeds [3, 4, 5]

## Group `t2`: 70 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 14 methods, seeds [3, 4, 5]
  - methods with fewer seeds: CCVR [3, 4]; Fed-GraB [3, 4]; FedECS-LT [3, 4]; FedECS-LT (200 extractor rounds) [3]; FedECS-LT (heads only) [3, 4]; FedLC [3, 4]; FedYoYo [3, 4]; RUCR [3, 4]
- CIFAR-100-LT (imbalance ratio 100), 20 clients, Dirichlet label split with alpha = 0.5, one ResNet-8 shared by all clients (standard CIFAR-LT class profile), 40% of the clients per round: 13 methods, seeds [3, 4]  **(fewer than 3 seeds)**
  - methods with fewer seeds: FedLoGe [4]

## Group `k50`: 74 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 50 clients, Dirichlet label split with alpha = 0.5: 15 methods, seeds [3, 4, 5]
- CIFAR-100-LT (imbalance ratio 100), 50 clients, Dirichlet label split with alpha = 0.5: 15 methods, seeds [3, 4, 5]
  - methods with fewer seeds: AvgKD [3]; Balanced Softmax [3]; FedALA [3, 4]; FedAvg [3]; FedLC [3]; FedMRL [3, 4]; FedProx [3]; FedRS [3]; GeFL-F [3]

  - not written to the paper yet: FedECS-LT has no run in every setting

## Group `fresh`: 30 runs, 2 settings

- CIFAR-10-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 5 methods, seeds [6, 7, 8]
- CIFAR-100-LT (imbalance ratio 100), 10 clients, Dirichlet label split with alpha = 0.5: 5 methods, seeds [6, 7, 8]

## Figures


## Slots in the paper

- Tables filled: 21 of 22; still empty: `k50_bal`
- Numbers filled: 222 of 229; still empty: `abl/cifar100/GeFL-F + AdamW/gain`, `k50/cifar10/er_margin`, `k50/cifar10/margin`, `k50/cifar10/strongest`, `k50/cifar100/er_margin`, `k50/cifar100/margin`, `k50/cifar100/strongest`
- Figures filled: 11 of 12; still empty: `fig_clients`

## Notebooks not returned yet (by folder)

- 2_one_shared_model_IF100: NB20_track2_cifar10_IF100_s5_part1of2 (6 of 6 runs missing), NB20_track2_cifar100_IF100_s3_part2of2 (1 of 6 runs missing), NB20_track2_cifar100_IF100_s5_part1of2 (6 of 6 runs missing), NB20_track2_cifar100_IF100_s5_part2of2 (6 of 6 runs missing)
- 3_main_comparison_other_datasets: NB10_main_gtsrb_part1of3 (21 of 21 runs missing), NB10_main_gtsrb_part2of3 (9 of 9 runs missing), NB10_main_gtsrb_part3of3 (9 of 9 runs missing), NB10_main_emnist (26 of 39 runs missing)
- 4_one_shared_model_IF050_IF010_and_long_runs: NB20_track2_cifar10_IF010_s3_part2of3 (1 of 4 runs missing), NB20_track2_cifar10_IF010_s3_part3of3 (3 of 3 runs missing), NB20_track2_cifar10_IF050_s3_part1of2 (6 of 6 runs missing), NB20_track2_cifar10_IF050_s3_part2of2 (6 of 6 runs missing), NB20_track2_cifar100_IF010_s3_part1of3 (1 of 4 runs missing), NB20_track2_cifar100_IF010_s3_part2of3 (4 of 4 runs missing), NB20_track2_cifar100_IF010_s3_part3of3 (3 of 3 runs missing), NB20_track2_cifar100_IF050_s3_part1of2 (6 of 6 runs missing), NB20_track2_cifar100_IF050_s3_part2of2 (6 of 6 runs missing)
- 5_reruns_of_failed_runs: NB10_main_emnist_rerun_7bfd7d (26 of 26 runs missing), NB20_track2_rerun_f2bb89 (3 of 3 runs missing)
- 6_robustness_sweeps: NB40_sweep_cifar10_a0.1 (12 of 12 runs missing), NB40_sweep_cifar10_a1.0 (12 of 12 runs missing), NB40_sweep_cifar10_K50 (12 of 12 runs missing), NB40_sweep_cifar10_K100 (12 of 12 runs missing), NB40_sweep_cifar10_IF050 (12 of 12 runs missing), NB40_sweep_cifar10_IF010 (12 of 12 runs missing)
- 7_gefl_protocol_grouped_baselines_10_and_50_clients: cifar100_10clients_grouped_baselines (18 of 18 runs missing), cifar100_50clients_all_methods_part1of3 (27 of 39 runs missing), cifar100_50clients_all_methods_part3of3 (1 of 9 runs missing), cifar100_50clients_all_methods_part1of3_rerun_9c26b1 (27 of 27 runs missing), cifar100_50clients_all_methods_part3of3_rerun_ae3320 (1 of 1 runs missing)
- 8_reviewer_requested_runs: cifar10_cifar100_components_of_fedecs_lt_part1of2 (5 of 18 runs missing), cifar10_cifar100_components_of_fedecs_lt_part2of2 (7 of 18 runs missing), cifar10_cifar100_gefl_f_variants (9 of 36 runs missing), cifar10_cifar100_components_of_fedecs_lt_part1of2_rerun_68f809 (5 of 5 runs missing), cifar10_cifar100_components_of_fedecs_lt_part2of2_rerun_a3aaa2 (7 of 7 runs missing), cifar10_cifar100_gefl_f_variants_rerun_f2ba51 (9 of 9 runs missing)
