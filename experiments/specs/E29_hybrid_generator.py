"""E29 - Hybrid generator PCR (exact-mean anchor + HWA-aggregated class rows) where the anchored stack already wins.

Ours-A (PC: class identity from the exact class mean only) wins on MNIST / FMNIST,
whose class means identify the class (nearest-class-mean accuracy 80 / 69 % in pixel
space), and loses on SVHN, whose class means do not (13 %, chance 10 %). PCR adds the
CVAE-F's learned class rows, aggregated with HWA so they cannot collapse, to the
exact-mean anchor. This checks that PCR keeps Ours-A's gains here (K10 tests SVHN /
CIFAR-10). References (same seeds, splits): E25 (Ours-A), E16 (CVAE + HWA + LA + CSL).
"""
SETTINGS = [("mnist", 0.01, 0.5), ("fmnist", 0.01, 0.5), ("fmnist", 1.0, None)]
EXPERIMENT = dict(
    name="E29_hybrid_generator",
    title="Hybrid generator PCR on MNIST / FMNIST (where the mean anchor is informative)",
    hypothesis="Ours-R (PCR + MC + KH + LA + CSL, T_s = 10) >= Ours-A under the long tail and >= the CVAE variant in IID.",
    reference="Ours-R Ts=10",
    metrics=["final_bal", "final_tail", "best_mean_acc", "bbc_bal", "class_sep_F", "ncm_acc", "cond_norm_tail_over_head_end"],
    plan=lambda cfg: [run_spec("Ours-R Ts=10", ds, s, compose("PCR+LA+CSL+MC+KH", head=dict(ts=10)), IF=IF, alpha=a)
                      for ds, IF, a in SETTINGS for s in cfg["seeds"]],
)
