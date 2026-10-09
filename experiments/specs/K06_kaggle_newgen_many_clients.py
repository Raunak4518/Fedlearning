"""K06 - Kaggle run: the new generator stack with many clients (FashionMNIST, K = 50 and 100, 3 seeds).

Why: with many clients each class row of the CVAE-F is fitted from ~60-120 images
per client, and the exact-statistics Gaussian generator (FSG) overtook it
(F04 MNIST K = 100: 86.1 vs 83.1; K03 FMNIST K = 100: 74.6 vs 73.3). PC-VAE has
no class rows - class identity is the exact federated class mean - so it
should not degrade with K; MC adds the exact class spread. This run also
carries the fixed CSL (each client its own synthetic slice; the first K03 run
shared one slice among an architecture's clients).
Arms - ONLY new ones: Ours (HWA+LA+CSL) with the fixed CSL (shares its generator with
Ours+MC, so it costs ~2 min; it replaces the NB4b/NB5b reruns); Ours+MC; Ours-PC+MC;
Ours-PC+ZP+MC. GeFL-F, +LA, +HWA+LA and FSG+LA at these K / seeds come from K03 (NB4, NB5):
splits depend only on the seed, so the runs pair and nothing is repeated.
Settings: FMNIST long tail (IF = 100, Dir 0.5) at K = 50 and 100; FMNIST IID at K = 100
(paper Fig. 4: GeFL-F 81.65).

Typical use (rough time on T4 x2: ~5 h):
    python K06.py --out_dir /kaggle/working/results
"""
LABELS = [("Ours (HWA+LA+CSL)", "HWA+LA+CSL"), ("Ours+MC", "HWA+LA+CSL+MC"), ("Ours-PC+MC", "PC+LA+CSL+MC"),
          ("Ours-PC+ZP+MC", "PC+ZP+LA+CSL+MC")]
EXPERIMENT = dict(
    name="K06_kaggle_newgen_many_clients",
    title="New generator stack with many clients (FashionMNIST, K = 50 / 100)",
    hypothesis="Ours-PC+MC >= FSG+LA >= Ours at K = 50/100 under the long tail; no loss in IID.",
    reference="Ours+MC",
    metrics=["final_bal", "final_tail", "best_mean_acc", "fidelity_tail", "fidelity_tail_mc", "spread_tail_mc"],
    plan=lambda cfg: [run_spec(lab, "fmnist", s, compose(c), IF=IF, alpha=a, K=K)
                      for IF, a, K in [(0.01, 0.5, 50), (0.01, 0.5, 100), (1.0, None, 100)]
                      for s in cfg["seeds"] for lab, c in LABELS],
)
