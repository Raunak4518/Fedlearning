"""K07 - Kaggle run: the paper's federated baselines under imbalance on SVHN and CIFAR-10 (3 seeds).

The GeFL paper compares against FedAvg (within architecture groups) and
LG-FedAvg, in the IID setting only. Under the long tail they had never been run
on SVHN / CIFAR-10. This adds them (and LG-FedAvg + LA, the logit-calibrated
version) so the imbalanced comparison covers every baseline family: GeFL-F and
our methods come from K01 / K05 on the same seeds and splits. No generator is
trained, so the run is short. FedProx / FedALA coincide with grouped FedAvg at
K = 10 (one client per architecture).
"""
LABELS = [("FedAvg (grouped)", baseline_method("FedAvg")), ("LG-FedAvg", baseline_method("LG-FedAvg")),
          ("LG-FedAvg+LA", baseline_method("LG-FedAvg", la=True))]
EXPERIMENT = dict(
    name="K07_kaggle_baselines_cifar10_svhn",
    title="Federated baselines under imbalance (FedAvg, LG-FedAvg, LG-FedAvg+LA): SVHN and CIFAR-10",
    hypothesis="all baselines < GeFL-F < ours under the long tail.",
    reference="LG-FedAvg",
    metrics=["final_bal", "final_tail", "best_mean_acc"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=IF, alpha=a)
                      for ds in ["svhn", "cifar10"] for IF, a in [(0.01, 0.5), (1.0, None)]
                      for s in cfg["seeds"] for lab, m in LABELS],
)
