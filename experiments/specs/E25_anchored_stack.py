"""E25 - The exact-statistics-anchored stack, 3 seeds (K = 10).

Principle: anchor every stage of generator-based heterogeneous FL to exactly
aggregated statistics (secure-aggregated sums), not to federated-averaged
parameters.
  class identity  - PC-VAE: conditioning on the exact federated class mean
  moments         - MC: W2 projection onto the exact class mean and spread
  distribution    - KH: kernel herding toward the exact class kernel embedding
  classifier bias - BBC: server-side balanced bias calibration (recorded as bbc_*)
Heads: LA + consensus soft labels (CSL); T_s = 1 and 10 (Proposition 3: the better
generator lowers the label bias, so the larger budget should now pay on MNIST).
E24 seed 0: MNIST LT 92.9 (BBC 93.7), FMNIST LT 77.5 (79.3), FMNIST IID 83.5.
"""
LABELS = [("Ours-A (PC+MC+KH+LA+CSL)", "PC+LA+CSL+MC+KH", 1), ("Ours-A Ts=10", "PC+LA+CSL+MC+KH", 10)]
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E25_anchored_stack",
    title="Exact-statistics-anchored stack (PC + MC + KH + BBC) with LA + CSL",
    hypothesis="beats every earlier variant under the long tail (3 seeds), no IID loss; T_s = 10 adds on top.",
    reference="Ours-A (PC+MC+KH+LA+CSL)",
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "bbc_tail", "oracle_bal"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"] for lab, c, ts in LABELS],
)
