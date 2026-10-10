"""K12 - Kaggle run: the final method's low-separation branch on SVHN, and the paper's 50 / 100-client axis on SVHN and CIFAR-10.

The final method chooses where class identity comes from with one statistic the server
already receives: F = between-class variance of the exact class means / mean within-class
spread (feature space; MNIST 0.29, FashionMNIST 0.55, CIFAR-10 ~0.14, SVHN 0.01).
Proposition 5: a mean-anchored generator separates classes only in proportion to sqrt(2F).
  F >= 0.05 : exact-mean anchor (PC-VAE) + MC + KH        (Ours-A: wins on MNIST, FMNIST, CIFAR-10)
  F <  0.05 : HWA-protected class rows (CVAE-F + HWA) + KH (SVHN: 65.6 under the long tail, best)
Open question for the low-separation branch: MC (exact mean + spread) or MC-S (exact mean +
class covariance)? MC-S gave +6.0 / +2.7 to the anchored generator on SVHN (NB22).
Parts (choose with --datasets / --K):
  svhn K=10        : rows + MC + KH (rerun, for pairing), rows + MC-S + KH, rows + MC-S + KH Ts=10; long tail + IID
  svhn K=50 / 100  : IID (paper Fig. 4: 73.64 / 76.00): GeFL-F, rows + MC + KH, rows + MC-S + KH
  cifar10 K=50/100 : IID (paper Fig. 4: GeFL-F 53.19 / 51.46, best 56.31 / 53.95): GeFL-F, Ours-A Ts=10

Typical use (rough times on T4 x2):
    python K12.py --datasets svhn --K 10 --out_dir /kaggle/working/results     # ~2 h
    python K12.py --datasets svhn --K 50 --out_dir /kaggle/working/results     # ~3 h
    python K12.py --datasets svhn --K 100 --out_dir /kaggle/working/results    # ~4 h
    python K12.py --datasets cifar10 --K 50 --seeds 0 1 --out_dir /kaggle/working/results    # ~6 h
    python K12.py --datasets cifar10 --K 100 --seeds 0 1 --out_dir /kaggle/working/results   # ~8 h
"""
ROWS = [("Ours+MC+KH (CVAE+HWA)", "HWA+LA+CSL+MC+KH", 1), ("Ours+MCS+KH (CVAE+HWA)", "HWA+LA+CSL+MCS+KH", 1)]


def _plan(cfg):
    runs = [run_spec(lab, "svhn", s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
            for IF, a in [(0.01, 0.5), (1.0, None)] for s in cfg["seeds"]
            for lab, c, ts in ROWS + [("Ours+MCS+KH (CVAE+HWA) Ts=10", "HWA+LA+CSL+MCS+KH", 10)]]
    runs += [run_spec(lab, "svhn", s, compose(c, head=dict(ts=ts)), K=K)
             for K in [50, 100] for s in cfg["seeds"] for lab, c, ts in [("GeFL-F", "", 1)] + ROWS]
    runs += [run_spec(lab, "cifar10", s, compose(c, head=dict(ts=ts)), K=K)
             for K in [50, 100] for s in cfg["seeds"]
             for lab, c, ts in [("GeFL-F", "", 1), ("Ours-A Ts=10", "PC+LA+CSL+MC+KH", 10)]]
    return runs


EXPERIMENT = dict(
    name="K12_kaggle_final_gated",
    title="Final method: low-separation branch on SVHN; SVHN and CIFAR-10 at K = 50 / 100 (IID)",
    hypothesis="rows + MC-S + KH >= rows + MC + KH on SVHN; the final method beats the paper's best at K = 50 / 100.",
    reference="Ours+MC+KH (CVAE+HWA)",
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "fidelity_tail", "fidelity_tail_mc", "fidelity_tail_kh",
             "class_sep_F", "ncm_acc"],
    plan=_plan,
)
