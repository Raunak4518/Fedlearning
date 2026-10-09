"""K05 - Kaggle run: the new generator stack on SVHN and CIFAR-10 (long tail and the paper's IID setting, 3 seeds).

New since K01 (docs/diagnostics.md, E20):
  MC  - moment-calibrated sampling: x' = ReLU(mu_c + s_c (x - m_c)), the W2-optimal
        affine correction to the exact federated class mean and spread (sums).
  PC  - prototype-conditioned residual CVAE: no class rows; class identity from the
        exact federated class mean; decoder models only within-class variation.
  ZP  - ex-post latent prior fitted by exact sums of encoder means.
E20 seed 0, MNIST long tail: CVAE + HWA 88.8 -> PC + MC 91.7, PC + ZP 92.0.
Arms (same seeds and splits, generators split across the two GPUs):
  GeFL-F; Ours (HWA+LA+CSL); Ours+MC; Ours-PC+MC; Ours-PC+ZP+MC; Ours-PC+ZP+MC Ts=10.
Paper targets (IID, best_mean_acc): SVHN 76.26, CIFAR-10 55.86 (GeFL-F CVAE-F),
best of any variant 76.26 / 59.36.

Typical use (rough times on T4 x2):
    python K05.py --datasets svhn --out_dir /kaggle/working/results              # ~3 h
    python K05.py --datasets cifar10 --IF 0.01 --out_dir /kaggle/working/results # ~6 h
    python K05.py --datasets cifar10 --IF 1.0 --out_dir /kaggle/working/results  # ~6 h
"""
LABELS = [("GeFL-F", "", 1), ("Ours (HWA+LA+CSL)", "HWA+LA+CSL", 1), ("Ours+MC", "HWA+LA+CSL+MC", 1),
          ("Ours-PC+MC", "PC+LA+CSL+MC", 1), ("Ours-PC+ZP+MC", "PC+ZP+LA+CSL+MC", 1),
          ("Ours-PC+ZP+MC Ts=10", "PC+ZP+LA+CSL+MC", 10)]
EXPERIMENT = dict(
    name="K05_kaggle_newgen_cifar10_svhn",
    title="New generator stack (PC-VAE, MC, ZP) on SVHN and CIFAR-10",
    hypothesis="Ours-PC+(ZP)+MC >= Ours >= GeFL-F in both regimes; IID beats the published GeFL-F.",
    reference="GeFL-F",
    split_by_gen=True,
    metrics=["best_mean_acc", "final_bal", "final_tail", "fidelity_tail", "fidelity_tail_mc", "spread_tail_mc"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
                      for ds in ["svhn", "cifar10"] for IF, a in [(0.01, 0.5), (1.0, None)]
                      for s in cfg["seeds"] for lab, c, ts in LABELS],
)
