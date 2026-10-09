"""K08 - Kaggle run: the exact-statistics-anchored stack on CIFAR-10, SVHN and many-client FashionMNIST (3 seeds).

The method ("Ours-A"): anchor every stage of generator-based heterogeneous FL to
exactly aggregated statistics (secure-aggregated sums) instead of federated-averaged
parameters:
  PC  - generator conditioned on the exact federated class mean (no class rows)
  MC  - samples moved onto the exact class mean and spread (W2 projection)
  KH  - kernel herding of samples toward the exact class kernel mean embedding
  BBC - server-side balanced bias calibration of each head (reported as bbc_bal)
  + LA (client-prior logit adjustment) and CSL (consensus soft labels) on the heads.
Local evidence (MNIST / FMNIST long tail, K = 10, seed 0): 92.9 / 77.5, with BBC
93.7 / 79.3 (oracle 94.1 / 80.5), vs GeFL-F 74.2 / 62.0.
Only the two new arms run here (they share one generator per seed); every baseline
comes from NB1-NB5 and NB9-NB13 on the same seeds and splits.

Typical use (rough times on T4 x2):
    python K08.py --datasets cifar10 --IF 0.01 --out_dir /kaggle/working/results   # ~3.5 h
    python K08.py --datasets cifar10 --IF 1.0 --out_dir /kaggle/working/results    # ~3.5 h
    python K08.py --datasets svhn --out_dir /kaggle/working/results                # ~2 h
    python K08.py --datasets fmnist --out_dir /kaggle/working/results              # ~4 h (K = 50 / 100)
"""
LABELS = [("Ours-A (PC+MC+KH+LA+CSL)", "PC+LA+CSL+MC+KH", 1), ("Ours-A Ts=10", "PC+LA+CSL+MC+KH", 10)]


def _plan(cfg):
    runs = [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
            for ds in ["cifar10", "svhn"] for IF, a in [(0.01, 0.5), (1.0, None)]
            for s in cfg["seeds"] for lab, c, ts in LABELS]
    runs += [run_spec(lab, "fmnist", s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a, K=K)
             for IF, a, K in [(0.01, 0.5, 50), (0.01, 0.5, 100), (1.0, None, 100)]
             for s in cfg["seeds"] for lab, c, ts in LABELS]
    return runs


EXPERIMENT = dict(
    name="K08_kaggle_anchored_stack",
    title="Exact-statistics-anchored stack (PC + MC + KH + BBC, with LA + CSL): CIFAR-10, SVHN, FMNIST K = 50/100",
    hypothesis="Ours-A beats every earlier method in every setting (paired with NB1-NB13); BBC adds under the long tail.",
    reference="Ours-A (PC+MC+KH+LA+CSL)",
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "bbc_tail", "oracle_bal", "fidelity_tail_kh"],
    plan=_plan,
)
