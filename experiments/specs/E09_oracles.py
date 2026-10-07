"""E09 - Where is the tail lost?  Component oracles.  (plan E1.2)

Replace one stage with a centralised, class-balanced oracle and see how
much tail accuracy each recovers (the CReFF diagnostic, adapted to GeFL-F's
three stages):
  FE oracle          stage (i) trained centrally on pooled data, balanced sampling
  generator oracle   stage (ii) trained centrally on pooled features, balanced
  classifier oracle  reported for EVERY run as oracle_bal / oracle_tail:
                     each header's fc re-fit on pooled real features, balanced.
Proposition 3 predicts the classifier oracle recovers the most.
"""
EXPERIMENT = dict(
    name="E09_oracles",
    title="Component oracles: FE, generator, classifier",
    hypothesis="classifier oracle gain > generator oracle gain > FE oracle gain.",
    reference="GeFL-F",
    metrics=["final_bal", "final_tail", "oracle_bal", "oracle_tail", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("FE oracle", compose(fe=dict(oracle=True))),
                                     ("Generator oracle", compose(gen=dict(oracle=True))),
                                     ("Ours (HWA+LCD+PCM+LA)", compose(OURS))]],
)
