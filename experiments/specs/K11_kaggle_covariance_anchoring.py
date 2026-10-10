"""K11 - Kaggle run: MC-S, anchoring the generator to the exact class COVARIANCE (SVHN, CIFAR-10; long tail + IID; 3 seeds).

Why. The anchored method Ours-A wins on MNIST, FashionMNIST and CIFAR-10 but loses on SVHN
(55.9 vs 62.5 for CVAE + HWA under the long tail). Its class identity comes from the
class MEAN, and SVHN's class means carry none: nearest-class-mean 10 %, LDA 21 % (pixel
space), 11.5 % in the shared FE's feature space. The class COVARIANCES do: QDA 54 %.
MC-S replaces MC's mean + spread correction with the exact class mean AND class
covariance (top-256 principal subspace of the pooled within-class covariance; exact
spread outside it; tail classes shrink their shape toward the pooled covariance and
keep their exact spread). The map is the Gelbrich (W2-optimal) linear map, the
minimum-displacement map that achieves both moments.
Proxy (class-agnostic samples, a head trained only on corrected synthetic data, tested
on real data), MC -> MC-S: SVHN 12 -> 29 (long tail), 11 -> 44 (balanced);
CIFAR-10 23 -> 31, 23 -> 35; MNIST 63 -> 78, 65 -> 90.
Arms (same seeds and splits as NB1 / NB2 / NB14-NB16 / NB19-NB21):
  SVHN     : Ours-A (rerun, for exact pairing), Ours-A-S, Ours-R-S (hybrid PCR generator + MC-S)
  CIFAR-10 : Ours-A (rerun), Ours-A-S, Ours-A-S Ts=10
Uses: class counts, class sums of h and ||h||^2 (as MC), the pooled second moment
sum h h^T (as FSG), class sums of z z^T in a broadcast 256-dim basis - all secure-
aggregated sums, one upload each.

Typical use (rough times on T4 x2):
    python K11.py --datasets svhn --out_dir /kaggle/working/results                 # ~2.5 h
    python K11.py --datasets cifar10 --IF 0.01 --out_dir /kaggle/working/results    # ~3 h
    python K11.py --datasets cifar10 --IF 1.0 --out_dir /kaggle/working/results     # ~3 h
"""
SVHN_ARMS = [("Ours-A (PC+MC+KH+LA+CSL)", "PC+LA+CSL+MC+KH", 1), ("Ours-A-S (PC+MCS+KH+LA+CSL)", "PC+LA+CSL+MCS+KH", 1),
             ("Ours-R-S (PCR+MCS+KH+LA+CSL)", "PCR+LA+CSL+MCS+KH", 1)]
CIFAR_ARMS = [("Ours-A (PC+MC+KH+LA+CSL)", "PC+LA+CSL+MC+KH", 1), ("Ours-A-S (PC+MCS+KH+LA+CSL)", "PC+LA+CSL+MCS+KH", 1),
              ("Ours-A-S Ts=10", "PC+LA+CSL+MCS+KH", 10)]
EXPERIMENT = dict(
    name="K11_kaggle_covariance_anchoring",
    title="MC-S: exact class-covariance anchoring (SVHN, CIFAR-10; long tail + IID)",
    hypothesis="Ours-A-S > Ours-A everywhere; on SVHN it closes the gap to CVAE + HWA (62.5 LT, 76.3 IID).",
    reference="Ours-A (PC+MC+KH+LA+CSL)",
    split_by_gen=True,
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "fidelity_tail_mc", "fidelity_tail_kh", "spread_tail_mc",
             "class_sep_F", "ncm_acc"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=IF, alpha=a)
                      for ds, arms in [("svhn", SVHN_ARMS), ("cifar10", CIFAR_ARMS)]
                      for IF, a in [(0.01, 0.5), (1.0, None)] for s in cfg["seeds"] for lab, c, ts in arms],
)
