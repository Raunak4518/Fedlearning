"""E03 - The method: every component alone, the key combinations, baselines.

Setting: long tail IF=100 + Dirichlet(0.5), K=10, paper budget.
Components (derivations in experiments/README.md):
  HWA  holder-weighted conditioning aggregation   (fixes dilution, Thm 1)
  LCD  lazy conditioning decay                    (fixes collapse, eq. 2.2)
  GF   gap-filling synthetic labels               (sequential water-filling)
  PCM  prior-completing mixed batches             (new: balanced-risk estimator)
  LA   logit adjustment by the training prior     (Bayes-consistent residual fix)
  BCR  server-side generative fc re-calibration
  LAFE LA during FE warm-up
Ours = HWA+LCD+PCM+LA. The decisive comparison is Ours vs +LA alone
(classifier-side fix): if Ours wins, the generator-side parts add value.
Baselines: LG-FedAvg (+LA = FedLC-style local calibration), FedAvg grouped.
"""
LABELS = [("GeFL-F", ""), ("+LA", "LA"), ("+GF", "GF"), ("+PCM", "PCM"), ("+BCR", "BCR"),
          ("+HWA", "HWA"), ("+LCD", "LCD"), ("+HWA+LCD", "HWA+LCD"), ("+HWA+LCD+GF", "HWA+LCD+GF"),
          ("+HWA+LCD+LA", "HWA+LCD+LA"), ("+HWA+LA", "HWA+LA"), ("+LCD+LA", "LCD+LA"), ("+PCM+LA", "PCM+LA"), ("+HWA+LCD+PCM", "HWA+LCD+PCM"),
          ("Ours (HWA+LCD+PCM+LA)", OURS), ("Ours-private (LCD+PCM+LA)", OURS_PRIVATE), ("Ours+BCR", OURS + "+BCR"), ("Ours+LAFE", OURS + "+LAFE")]
BASELINES = [("LG-FedAvg", "LG-FedAvg", False), ("LG-FedAvg+LA", "LG-FedAvg", True),
             ("FedAvg (grouped)", "FedAvg", False)]
EXPERIMENT = dict(
    name="E03_method_components",
    title="Method components, combinations and baselines under a long tail",
    hypothesis="Ours > +LA > GeFL-F on final_bal and final_tail (p<0.05); each generator-side "
               "component helps more when combined with LA.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, c in LABELS]
                     + [run_spec(lab, ds, s, baseline_method(b, la=la), IF=0.01, alpha=0.5)
                        for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, b, la in BASELINES],
)
