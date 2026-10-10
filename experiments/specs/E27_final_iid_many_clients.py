"""E27 - The final method in the paper's IID setting with more clients (Figure 4 axis), 3 seeds.

Paper Fig. 4 (best_mean_acc): MNIST K = 50 / 100: GeFL-F 95.04 / 94.63 (also the best
of all ten methods); FMNIST K = 50: GeFL-F 82.21, best 82.96 (DDPM-F).
Ours-A (T_s = 10) has been measured at K = 10 only (MNIST 97.62, FMNIST 83.96).
NB17 (Kaggle) covers FMNIST at K = 100 IID; this covers the rest.
"""
SETTINGS = [("mnist", 50), ("mnist", 100), ("fmnist", 50)]
EXPERIMENT = dict(
    name="E27_final_iid_many_clients",
    title="Final method (Ours-A, T_s = 10) in the IID setting at K = 50 / 100",
    hypothesis="Ours-A beats the paper's best at every K, as at K = 10.",
    reference="Ours-A Ts=10",
    metrics=["best_mean_acc", "final_bal", "final_ens_acc", "oracle_bal"],
    plan=lambda cfg: [run_spec("Ours-A Ts=10", ds, s, compose("PC+LA+CSL+MC+KH", head=dict(ts=10)), K=K)
                      for ds, K in SETTINGS for s in cfg["seeds"]],
)
