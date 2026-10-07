"""E11 - What does the gain cost in privacy and communication?  (plan P5)

(a) Memorisation: feature-space MND (paper Eq. 1 on frozen-FE features;
    > 1 = memorisation) for GeFL-F vs Ours.
(b) HWA uploads each client's class histogram once. Here the histogram is
    Laplace-noised for epsilon-DP (epsilon in {10, 1, 0.1}) before it is used
    for the weights, to measure the accuracy cost of protecting it.
(c) Communication: HWA adds C integers per client, once; PCM, LA and LCD
    add nothing (all local).
"""
EXPERIMENT = dict(
    name="E11_privacy_cost",
    title="Memorisation (feature MND) and DP class histograms",
    hypothesis="Ours' feature MND stays < 1 and close to GeFL-F; eps >= 1 keeps most of the gain.",
    reference="GeFL-F",
    mnd=True,
    metrics=["final_bal", "final_tail", "feature_mnd", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("Ours (HWA+LCD+PCM+LA)", compose(OURS)),
                                     ("Ours-private (LCD+PCM+LA)", compose(OURS_PRIVATE)),
                                     ("Ours, DP eps=10", compose(OURS, gen=dict(dp_eps=10.0))),
                                     ("Ours, DP eps=1", compose(OURS, gen=dict(dp_eps=1.0))),
                                     ("Ours, DP eps=0.1", compose(OURS, gen=dict(dp_eps=0.1)))]],
)
