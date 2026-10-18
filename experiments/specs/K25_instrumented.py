"""K25 - instrumented runs (action plan X6): what happens at each aggregation, measured on the real pipeline.
MNIST (CPU-feasible) and CIFAR-10 (Kaggle), seed 3 (K17's splits), long tail, K = 10.

Logged per run (core.instrument_*; math report §7):
  M1 inst_fe_grad       every 5 Stage-I rounds: client-gradient heterogeneity zeta^2, pooled gradient norm, largest
                        client gradient norm, at the global FE (Corollary 2)
  M2 inst_bn_var_ratio  averaged BN running variance / exact pooled variance, per channel (after Stage I)
  M4 inst_rows          every Stage-II round: per class, mean row-norm gain on holders and loss on non-holders
                        (Theorem 3 / Approximation 5; only generators with class rows)
  M7, M8 inst_phases    every 10 Stage-III rounds on one client: logit spread on generated features before and after
                        the synthetic phase (Proposition 9) and the counterfactual retention of the synthetic phase's
                        change through the real phase, absent vs held classes and the body (Proposition 8)
The counterfactual real phase changes the random stream, so these runs are diagnostics, not accuracy results.

PRE-REGISTERED (18 Oct 2026): (a) GeFL-F's non-holder loss exceeds the holder gain for the classes with kappa < 1 in
every round after the first few; the decay fix has non-holder loss ~ 0. (b) Retention of absent-class rows >= 0.9
with LA. (c) On CIFAR-10, GeFL-F's synthetic phase lowers the logit spread on generated features more than ours does.
"""
ARMS = {"mnist": [("GeFL-F [instrumented]", "", 1), ("GeFL-F + AdamW + LA [instrumented]", "ADAMW3+LA", 1),
                  ("Ours Ts=10 [instrumented]", "GATE+MC+KH+LA+CSL", 10)],
        "cifar10": [("GeFL-F [instrumented]", "", 1), ("GeFL-F + AdamW + LA [instrumented]", "ADAMW3+LA", 1),
                    ("Ours Ts=1 [instrumented]", "GATE+MC+KH+LA+CSL", 1)]}
EXPERIMENT = dict(
    name="K25_instrumented",
    title="Instrumented runs: per-round diagnostics of the three stages (MNIST, CIFAR-10; seed 3)",
    hypothesis="(a)-(c) in the docstring.",
    reference="GeFL-F [instrumented]",
    seeds=[3],
    instrument=True,
    readouts=["er"],
    metrics=["final_bal", "er_bal", "final_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5, K=10)
                      for ds in ["mnist", "cifar10"] for s in cfg["seeds"] for lab, c, ts in ARMS[ds]],
)
