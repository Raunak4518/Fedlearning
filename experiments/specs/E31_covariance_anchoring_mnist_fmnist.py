"""E31 - MC-S where the anchored method already wins (MNIST / FashionMNIST, long tail + FMNIST IID; 3 seeds).

MC-S (exact class mean + class covariance, Gelbrich map) replaces MC in the final method.
It must not hurt where Ours-A already wins; the proxy says it can only add class information
(MNIST, class-agnostic samples: MC 63 -> MC-S 78 under the long tail). Reference: E25
(Ours-A Ts=10, MC) on the same seeds and splits.
"""
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E31_covariance_anchoring_mnist_fmnist",
    title="MC-S in the final method on MNIST / FashionMNIST (T_s = 10)",
    hypothesis="Ours-A-S Ts=10 >= Ours-A Ts=10 (E25) in every setting.",
    reference="Ours-A-S Ts=10",
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "fidelity_tail_mc", "fidelity_tail_kh", "spread_tail_mc",
             "class_sep_F", "ncm_acc"],
    plan=lambda cfg: [run_spec("Ours-A-S Ts=10", ds, s, compose("PC+LA+CSL+MCS+KH", head=dict(ts=10)), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"]],
)
