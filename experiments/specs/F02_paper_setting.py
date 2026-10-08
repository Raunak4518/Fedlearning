"""F02 - The paper's own setting (full rounds, 3 seeds): parity and do-no-harm.

IID equal clients, K=10, Table XIV/XV. GeFL-F must reproduce Figure 4
(CVAE-F: MNIST 95.47, FMNIST 83.14, SVHN 76.26) and beat the paper's
generator-free baselines; Ours must not lose to it here (its rarity rule
assigns FSG to no class in IID data, and LA's prior is uniform).
Metric of the paper: best_mean_acc.
"""
EXPERIMENT = dict(
    name="F02_paper_setting",
    title="Paper setting (IID, K=10): parity with Figure 4 and do-no-harm",
    hypothesis="GeFL-F within 1 pp of Figure 4; Ours within noise of GeFL-F.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_acc", "final_bal", "final_tail", "n_gauss_classes"],
    plan=lambda cfg: [run_spec(lab, ds, s, m)
                      for ds in ["mnist", "fmnist", "svhn"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("Ours-hybrid (RHYB+HWA+LA)", compose("HWA+LA", gen=dict(type="hybrid", n_min_rel=1.0))), ("MIX+HWA+LA", compose("MIX+HWA+LA")),
                                     ("FedAvg (grouped)", baseline_method("FedAvg")),
                                     ("LG-FedAvg", baseline_method("LG-FedAvg"))]],
)
