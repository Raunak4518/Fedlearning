"""E08 - Is the fix generator-agnostic?  (plan E3.4)

DCGAN-F (Table XVIII) has no weight decay in the paper, so LCD does not
apply and the collapse of eq. 2.2 should NOT occur; only dilution (Thm 1)
remains. Prediction: GeFL-F with DCGAN-F loses less tail accuracy than with
CVAE-F (compare with E03), HWA still helps, and Ours (HWA+PCM+LA) helps.
"""
EXPERIMENT = dict(
    name="E08_generators",
    title="DCGAN-F feature generator under a long tail",
    hypothesis="DCGAN-F: Ours > GeFL-F; tail-row norms do not collapse under GeFL-F.",
    reference="GeFL-F (DCGAN-F)",
    metrics=["final_bal", "final_tail", "final_worst", "fidelity_tail", "cond_norm_tail_over_head_end"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, c in [("GeFL-F (DCGAN-F)", "DCGAN"), ("+HWA (DCGAN-F)", "DCGAN+HWA"),
                                     ("+LA (DCGAN-F)", "DCGAN+LA"),
                                     ("Ours (DCGAN-F: HWA+PCM+LA)", "DCGAN+HWA+PCM+LA")]],
)
