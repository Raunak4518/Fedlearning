"""F01 - MAIN RESULT (full rounds, 3 seeds): the method vs GeFL-F and baselines under a long tail.

Setting: IF=100 long tail + Dirichlet(0.5) label skew, K=10, the paper's
full Table XIV/XV schedule (T_FE/T_KA/T_TN = 20/100/50) and data budget.
Compared: GeFL-F (paper); +LA (classifier-side fix only, FedLC-style);
+HWA+LA (best CVAE-F method); FSG+LA (Gaussian generator only); Ours; and
the generator-free baselines the paper uses (FedAvg grouped = local at
K=10, LG-FedAvg) plus LG-FedAvg+LA.
Chosen after the quick pass (E12-E14, 3 seeds, paired tests).
"""
EXPERIMENT = dict(
    name="F01_main_longtail",
    title="Main result: long tail IF=100, non-IID, K=10 (full schedule)",
    hypothesis="Ours > +HWA+LA > +LA > GeFL-F on final_bal and final_tail, p<0.05, on both datasets.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")),
                                     ("+HWA+LA", compose("HWA+LA")), ("FSG+LA", compose("GAUSS+LA")),
                                     (FINAL_LABEL, final_method()),
                                     ("FedAvg (grouped)", baseline_method("FedAvg")),
                                     ("LG-FedAvg", baseline_method("LG-FedAvg")),
                                     ("LG-FedAvg+LA", baseline_method("LG-FedAvg", la=True))]],
)
