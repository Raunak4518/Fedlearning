"""E26 - Formal privacy for the anchoring statistics of the final method (3 seeds, K = 10, long tail).

Ours-A releases, beyond GeFL-F's protocol, four federation-level sums per class:
counts, sums of h, sums of ||h||^2 and sums of the bounded random features phi(h).
Here every one of them is made (eps, delta)-DP at the sample level: features are
clipped to the 90th-percentile norm of the held-out pool (public data), and each
release gets Gaussian noise; budgets compose by zCDP (rho split equally over the
four releases), delta = 1e-5. Everything the server computes from them (PC
prototypes, MC moments, KH targets, the BBC gate) is post-processing.
The rarest classes carry the hard trade-off (24 samples behind a 768-dim mean):
in DP mode MC and KH are applied to a class only where the release's expected
noise norm is at most half the signal scale (decided from the noisy counts and
the public noise level - no extra privacy cost); other classes fall back to the
plain generator. Memorisation (feature-space MND) of the raw generator and of the
final sampler is recorded too (paper's own privacy metric; > 1 = memorisation).
Reference: E25 (Ours-A Ts=10, no DP) on the same seeds and splits.
"""
EPS = [8.0, 2.0]
EXPERIMENT = dict(
    name="E26_private_anchoring",
    title="(eps, delta)-DP anchoring statistics for the final method",
    hypothesis="at eps = 8 the method keeps most of its gain over GeFL-F; at eps = 2 it degrades gracefully, still well above GeFL-F.",
    reference="Ours-A Ts=10, DP eps=8",
    mnd=True,
    metrics=["final_bal", "final_tail", "bbc_bal", "feature_mnd", "feature_mnd_final", "mc_classes_used", "kh_classes_used"],
    plan=lambda cfg: [run_spec(f"Ours-A Ts=10, DP eps={e:g}", ds, s,
                               compose("PC+LA+CSL+MC+KH", gen=dict(stat_dp_eps=e), head=dict(ts=10)), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for e in EPS],
)
