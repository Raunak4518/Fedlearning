"""E00 - Paper parity: does this GeFL-F reproduce Figure 4 of the paper?

Why: every later comparison is against GeFL-F, so ours must first match the
paper's own numbers in the paper's own setting (IID, equal clients, K=10,
Table XIV/XV hyperparameters). Targets (CVAE-F, Figure 4): MNIST 95.47,
FMNIST 83.14, SVHN 76.26, CIFAR-10 55.86. Metric = best_mean_acc (mean over
the 10 architectures of each one's best accuracy over rounds).
Also runs the two generator-free baselines the paper compares against
(FedAvg grouped by architecture = local training at K=10; LG-FedAvg).
Pass: within 1 pp of the paper.
"""
EXPERIMENT = dict(
    name="E00_paper_parity",
    title="GeFL-F paper parity (IID, K=10)",
    hypothesis="best_mean_acc is within 1 pp of Figure 4 for each dataset.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_acc", "final_bal", "final_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m)
                      for ds in ["mnist", "fmnist", "svhn", "cifar10"]
                      for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()),
                                     ("FedAvg (grouped)", baseline_method("FedAvg")),
                                     ("LG-FedAvg", baseline_method("LG-FedAvg"))]],
)
