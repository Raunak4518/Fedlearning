"""E22 - PC-VAE + MC with many clients (MNIST long tail, K = 100, full rounds, 3 seeds).

F04: at K = 100 the CVAE + HWA falls to 83.1 while the exact-statistics Gaussian
generator (FSG) holds 86.1 - the CVAE's class rows are fitted from ~60 images
per client. PC-VAE has no class rows (class identity = exact federated mean),
so H-PC predicts it does not degrade with K: PC + MC >= FSG at K = 100.
References: F04 (GeFL-F, +HWA+LA, FSG+LA; same seeds and splits).
"""
LABELS = [("Ours-PC+MC", "PC+LA+CSL+MC"), ("Ours-PC+ZP+MC", "PC+ZP+LA+CSL+MC"), ("FSG+LA", "GAUSS+LA")]
EXPERIMENT = dict(
    name="E22_pc_many_clients",
    title="PC-VAE + MC at K = 100 clients (MNIST long tail)",
    hypothesis="PC + MC does not degrade with K: >= FSG (86.1) at K = 100.",
    reference="FSG+LA",
    metrics=["final_bal", "final_tail", "best_mean_acc", "fidelity_tail_mc", "spread_tail_mc"],
    plan=lambda cfg: [run_spec(lab, "mnist", s, compose(c), IF=0.01, alpha=0.5, K=100)
                      for s in cfg["seeds"] for lab, c in LABELS],
)
