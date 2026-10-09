"""E17 - The synthetic budget with more clients (FMNIST, K = 50, IID and long tail, 3 seeds).

Proposition 3: the optimal synthetic share is w* = sigma^2 / (2 n_r b^2),
which grows as the real sample size n_r per head shrinks. At K = 50 each
client holds ~120 real images (vs ~600 at K = 10), so a larger synthetic
budget with consensus labels should help MORE than at K = 10, and T_s = 1
(the paper's default) should be far from optimal. K03 on Kaggle measured
T_s = 1 at K = 50: CSL gave no gain in IID (81.29 vs 81.50).
Prediction: (CSL or Ours, T_s = 10) > GeFL-F (T_s = 1 and T_s = 10) at K = 50.
"""
IID = [("GeFL-F", "", 1), ("GeFL-F Ts=10", "", 10), ("+CSL Ts=10", "CSL", 10), ("Ours (HWA+LA+CSL) Ts=10", "HWA+LA+CSL", 10)]
LT = [("GeFL-F", "", 1), ("+HWA+LA", "HWA+LA", 1), ("Ours (HWA+LA+CSL) Ts=10", "HWA+LA+CSL", 10)]
EXPERIMENT = dict(
    name="E17_budget_more_clients",
    title="Synthetic budget at K = 50 clients (FMNIST)",
    hypothesis="At K = 50, consensus-labelled T_s = 10 beats GeFL-F; the budget gain exceeds the K = 10 gain.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_tail", "oracle_bal"],
    plan=lambda cfg: [run_spec(lab, "fmnist", s, compose(c, head=dict(ts=ts)), K=50) for s in cfg["seeds"] for lab, c, ts in IID]
                     + [run_spec(lab, "fmnist", s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5, K=50)
                        for s in cfg["seeds"] for lab, c, ts in LT],
)
