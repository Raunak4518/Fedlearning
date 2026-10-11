"""E32 - MC-B: one moment-anchoring operator that uses each class's exact covariance as far as its sample size determines it.

E31 (MC-S, tail covariances shrunk toward the POOLED covariance) against MC (exact mean + spread):
FashionMNIST IID +1.29 (every seed, p = 0.013), but MNIST long tail -1.56 - all of it on the rare
classes (tail recall -4.2, common classes +0.2; FashionMNIST LT: common +1.4, tail -2.2). With 24
samples in a 256-dim subspace the pooled prior overwrites the class-specific shape the anchored
generator had learned. MC-B shrinks toward the GENERATOR's own class covariance instead (empirical
Bayes, weights n_c and k): a rare class keeps the generator's shape and the Gelbrich map reduces to
MC's scalar map; a common class gets its exact covariance. Reference: E25 (MC) and E31 (MC-S),
same seeds and splits.
"""
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E32_moment_anchor_bayes",
    title="MC-B: exact covariance where the class's data determine it, the generator's shape where not",
    hypothesis="MC-B >= max(MC, MC-S) in every setting: MNIST LT >= 94.8, FMNIST LT >= 79.7, FMNIST IID >= 85.2.",
    reference="Ours-A-B Ts=10",
    metrics=["final_bal", "final_tail", "final_worst", "best_mean_acc", "bbc_bal", "fidelity_tail_kh", "spread_tail_kh"],
    plan=lambda cfg: [run_spec("Ours-A-B Ts=10", ds, s, compose("PC+LA+CSL+MCB+KH", head=dict(ts=10)), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"]],
)
