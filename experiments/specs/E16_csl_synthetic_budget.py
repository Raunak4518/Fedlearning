"""E16 - Does consensus labelling unlock a larger synthetic budget? (full rounds, 3 seeds, FMNIST + MNIST IID)

The paper (Fig. 10) finds T_s = 5 synthetic epochs per round no better than
T_s = 1. With hard labels this is expected: a generated feature's label is
noisy, so more synthetic data adds more label noise along with more signal.
CSL replaces the hard label by (1 - beta) onehot + beta * consensus, whose
expected error is lower; if label noise was the limit, the benefit of more
synthetic epochs should now grow with T_s for CSL but not for GeFL-F.
Interaction hypothesis: (CSL, T_s=3) - (CSL, T_s=1) > (GeFL-F, T_s=3) - (GeFL-F, T_s=1).
"""
LABELS = [("GeFL-F", "", 1), ("GeFL-F Ts=3", "", 3), ("GeFL-F Ts=5", "", 5),
          ("+CSL", "CSL", 1), ("+CSL Ts=3", "CSL", 3), ("+CSL Ts=5", "CSL", 5), ("+CSL Ts=10", "CSL", 10),
          ("Ours (HWA+LA+CSL) Ts=5", "HWA+LA+CSL", 5), ("Ours (HWA+LA+CSL) Ts=10", "HWA+LA+CSL", 10)]
# Long tail: does the larger budget also hold (or hurt) where HWA + LA matter?
LABELS_LT = [("GeFL-F", "", 1), ("Ours (HWA+LA+CSL)", "HWA+LA+CSL", 1),
             ("Ours (HWA+LA+CSL) Ts=5", "HWA+LA+CSL", 5), ("Ours (HWA+LA+CSL) Ts=10", "HWA+LA+CSL", 10)]
EXPERIMENT = dict(
    name="E16_csl_synthetic_budget",
    title="Consensus labels and the synthetic budget T_s",
    hypothesis="CSL gains from T_s = 3 / 5; GeFL-F does not (paper Fig. 10).",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_ens_acc", "oracle_bal"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)))
                      for ds in ["fmnist", "mnist"] for s in cfg["seeds"] for lab, c, ts in LABELS]
                     + [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5)
                        for ds in ["fmnist", "mnist"] for s in cfg["seeds"] for lab, c, ts in LABELS_LT],
)
