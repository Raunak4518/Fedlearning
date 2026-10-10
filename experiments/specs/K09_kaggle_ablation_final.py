"""K09 - Kaggle run: ablation of the final method, one component removed at a time (MNIST, FMNIST long tail, 3 seeds).

Final method Ours-A (T_s = 10): PC-VAE generator + MC + KH sampling + LA + CSL heads,
gated BBC (recorded for every arm as bbc_bal). Each arm removes or swaps one part:
  - LA   : heads trained with plain cross-entropy
  - CSL  : hard labels on synthetic features
  - MC   : no moment calibration (KH herds uncalibrated candidates)
  - KH   : no kernel herding
  - PC   : the paper's CVAE-F generator with HWA instead of PC-VAE (MC + KH kept)
All PC arms share one generator per seed (two GPUs split the generators).
Long tail IF = 100, Dirichlet(0.5), K = 10 - the same splits as every other run.

Typical use (rough time on T4 x2: ~3 h):
    python K09.py --out_dir /kaggle/working/results
"""
ARMS = [("Ours-A Ts=10", "PC+LA+CSL+MC+KH"), ("-LA", "PC+CSL+MC+KH"), ("-CSL", "PC+LA+MC+KH"),
        ("-MC", "PC+LA+CSL+KH"), ("-KH", "PC+LA+CSL+MC"), ("-PC (CVAE+HWA)", "HWA+LA+CSL+MC+KH")]
EXPERIMENT = dict(
    name="K09_kaggle_ablation_final",
    title="Ablation of the final method (each component removed; MNIST / FMNIST long tail)",
    hypothesis="every component is needed: removing any one lowers balanced accuracy.",
    reference="Ours-A Ts=10",
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "bbc_bal", "best_mean_acc", "fidelity_tail_kh", "spread_tail_kh"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=10)), IF=0.01, alpha=0.5)
                      for ds in ["mnist", "fmnist"] for s in cfg["seeds"] for lab, c in ARMS],
)
