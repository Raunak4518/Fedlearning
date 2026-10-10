"""K10 - Kaggle run: why the anchored stack fails on SVHN, and the hybrid fix (SVHN, CIFAR-10; long tail + IID; 3 seeds).

Finding (K08 / NB16): on SVHN the anchored stack Ours-A loses to CVAE + HWA + LA
(55.9 vs 62.5 balanced accuracy under the long tail; 73.9 vs 75.8 GeFL-F in IID).
Cause: PC takes class identity from the exact class mean, and SVHN's class means are
not class-informative (pixel-space nearest-class-mean accuracy 13 %, chance 10 %;
MNIST 80 %, FMNIST 69 %, CIFAR-10 27 %). Every run here records the feature-space
separation F and nearest-class-mean accuracy (class_sep_F, ncm_acc).
Arms (two generators per seed, split across the GPUs):
  Ours+MC+KH (CVAE+HWA) - the paper's CVAE-F with HWA, plus MC and KH sampling
  Ours-R / Ours-R Ts=10 - the hybrid PCR generator: exact-mean anchor + HWA class rows
Baselines on the same seeds and splits come from K01 (NB1 / NB2), K05 (NB9-NB11), K08 (NB14-NB16).

Typical use (rough times on T4 x2):
    python K10.py --datasets svhn --out_dir /kaggle/working/results                # ~2 h
    python K10.py --datasets cifar10 --IF 0.01 --out_dir /kaggle/working/results   # ~3.5 h
    python K10.py --datasets cifar10 --IF 1.0 --out_dir /kaggle/working/results    # ~3.5 h
"""
ARMS = [("Ours+MC+KH (CVAE+HWA)", "HWA+LA+CSL+MC+KH", 1), ("Ours-R", "PCR+LA+CSL+MC+KH", 1), ("Ours-R Ts=10", "PCR+LA+CSL+MC+KH", 10)]
EXPERIMENT = dict(
    name="K10_kaggle_hybrid_svhn_cifar10",
    title="Hybrid generator PCR vs CVAE+HWA (+MC+KH) on SVHN and CIFAR-10",
    hypothesis="Ours-R >= max(CVAE + HWA, Ours-A) on SVHN and CIFAR-10 in both regimes.",
    reference="Ours+MC+KH (CVAE+HWA)",
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "class_sep_F", "ncm_acc", "fidelity_tail", "fidelity_tail_kh"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
                      for ds in ["svhn", "cifar10"] for IF, a in [(0.01, 0.5), (1.0, None)]
                      for s in cfg["seeds"] for lab, c, ts in ARMS],
)
