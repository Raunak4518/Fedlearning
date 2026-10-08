"""F07 - Privacy cost (full rounds, 3 seeds).

Ours needs, at the server, only aggregates that secure aggregation can
deliver: FSG's per-class sums and counts, and HWA's holder-weighted row
sums. Here the per-client counts are additionally Laplace-noised
(epsilon-DP per histogram, epsilon in {10, 1, 0.1}) to price their
disclosure, and memorisation is measured by feature-space MND (> 1 =
memorisation) for GeFL-F and Ours.
"""
EXPERIMENT = dict(
    name="F07_privacy",
    title="DP class histograms and memorisation",
    hypothesis="Ours keeps most of its gain at eps=1; feature MND stays < 1.",
    reference="GeFL-F",
    mnd=True,
    metrics=["final_bal", "final_tail", "feature_mnd", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), (FINAL_LABEL, final_method()),
                                     (FINAL_LABEL + ", DP eps=10", final_method(gen=dict(dp_eps=10.0))),
                                     (FINAL_LABEL + ", DP eps=1", final_method(gen=dict(dp_eps=1.0))),
                                     (FINAL_LABEL + ", DP eps=0.1", final_method(gen=dict(dp_eps=0.1)))]],
)
