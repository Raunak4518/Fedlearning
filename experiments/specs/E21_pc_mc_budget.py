"""E21 - The new generator stack with the larger synthetic budget (full rounds, 3 seeds, K = 10).

E20 (seed 0): PC-VAE + MC / + ZP lifted MNIST long-tail tail fidelity from 0.28
to 0.85 and balanced accuracy from 88.8 to 91.7-92.0. Proposition 3: a lower
label bias b raises the optimal synthetic share, so T_s = 10 - which did NOT
help the CVAE on MNIST (E16: 90.2 vs 90.2) - should now help there too.
Arms: PC+ZP+MC (both fixes), and PC+MC / PC+ZP+MC at T_s = 10; references come
from E16 (CVAE, T_s 1/10) and E20 (CVAE, PC, PC+MC, PC+ZP at T_s = 1), same seeds.
"""
LABELS = [("Ours-PC+ZP+MC", "PC+ZP+LA+CSL+MC", 1), ("Ours-PC+MC Ts=10", "PC+LA+CSL+MC", 10),
          ("Ours-PC+ZP+MC Ts=10", "PC+ZP+LA+CSL+MC", 10)]
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E21_pc_mc_budget",
    title="PC-VAE + MC (+ ZP) with the larger synthetic budget",
    hypothesis="with the better generator, T_s = 10 helps on MNIST long tail too; PC+ZP+MC >= PC+MC.",
    reference="Ours-PC+ZP+MC",
    metrics=["final_bal", "final_tail", "best_mean_acc", "fidelity_tail_mc", "spread_tail_mc"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"] for lab, c, ts in LABELS],
)
