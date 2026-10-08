"""F10 - One method for both settings: HWA + LA + consensus soft labels (full rounds, 3 seeds).

F01 showed HWA + LA wins under the long tail (+13 / +17 pp over GeFL-F).
F09 showed consensus soft labels (CSL, beta = 0.5) win in the paper's own
IID setting (+0.69 MNIST, +0.49 FMNIST over GeFL-F, every seed). The two act
on different parts of the pipeline - HWA on the generator's aggregation,
LA on the real-data classifier objective, CSL on the synthetic phase's
labels - so they should add. This tests the combination in both regimes,
against each part alone and GeFL-F, on identical splits and random streams.
"""
LABELS = [("GeFL-F", ""), ("+HWA+LA", "HWA+LA"), ("+CSL", "CSL"), ("+HWA+LA+CSL", "HWA+LA+CSL")]
EXPERIMENT = dict(
    name="F10_combined_method",
    title="Combined method HWA + LA + CSL: long tail and IID",
    hypothesis="+HWA+LA+CSL >= +HWA+LA under the long tail and >= +CSL in IID.",
    reference="GeFL-F",
    metrics=["final_bal", "final_tail", "best_mean_acc", "final_acc"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=IF, alpha=a)
                      for ds in ["mnist", "fmnist"] for IF, a in [(0.01, 0.5), (1.0, None)]
                      for s in cfg["seeds"] for lab, c in LABELS],
)
