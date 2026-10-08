"""F03 - Heterogeneity regimes (full rounds, 3 seeds).

Label skew without a global tail (IF=1, Dir 0.5) and a milder tail
(IF=10, Dir 0.5) complete the picture between F02 (IID) and F01 (IF=100).
Prediction: the gain over GeFL-F grows with imbalance; without a global
tail HWA and FSG have little to fix, so the gain there comes from LA.
"""
EXPERIMENT = dict(
    name="F03_regimes",
    title="Heterogeneity regimes: Dir(0.5) balanced and IF=10",
    hypothesis="Ours >= +LA > GeFL-F in both regimes; gain over GeFL-F larger at IF=10 than at IF=1.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=IF, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for IF in [1.0, 0.1] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")),
                                     (FINAL_LABEL, final_method())]],
)
