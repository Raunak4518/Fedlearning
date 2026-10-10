"""K13 - Kaggle run: VCT (vicinal consensus transfer) against MixUp, CIFAR-10, paper's IID setting, K = 10, 3 seeds.

Target: the paper's best CIFAR-10 number with augmentation, image-space GeFL + MixUp, 62.67
(Table IV; GeFL + CutMix 61.66). Ours-A at T_s = 10 gives 60.93 without augmentation (K08).
Headroom: the ten heterogeneous heads TOGETHER reach 69.8 % on CIFAR-10 while the average head
reaches 60 %; consensus labels on synthetic features transfer little of it (+0.75).
VCT mixes every real feature of a client with a partner from the round's consensus-labelled,
exact-moment-corrected synthetic pool: x = l x_real + (1 - l) x_syn, l = max(u, 1 - u),
u ~ Beta(1, 1); target l onehot(y) + (1 - l) consensus(x_syn). To second order MixUp depends
on its partners only through class prior, class means and class covariances (Proposition 7),
so these partners give the global-data MixUp regulariser without any data leaving a client;
the consensus labels add ensemble distillation along the path. Control arm: standard MixUp
on the client's own real features, same alpha, same pipeline (paired).

Long-tail part (IF = 100, Dir 0.5): the same two arms at T_s = 1 (the better final-round budget
there). This is where VCT's partners matter most: a client missing classes cannot MixUp toward
them locally, while the synthetic partners cover every class with exact moments.

Typical use (rough times on T4 x2):
    python K13.py --IF 1.0 --out_dir /kaggle/working/results     # IID, ~5 h
    python K13.py --IF 0.01 --out_dir /kaggle/working/results    # long tail, ~2 h
"""
ARMS = [("Ours-A Ts=10 + MixUp", "PC+LA+CSL+MC+KH+MIXUP"), ("Ours-A Ts=10 + VCT", "PC+LA+CSL+MC+KH+VCT")]
EXPERIMENT = dict(
    name="K13_kaggle_vct_vs_mixup",
    title="VCT vs MixUp in the final method (CIFAR-10, IID, K = 10)",
    hypothesis="Ours-A + VCT > Ours-A + MixUp, and above the paper's GeFL + MixUp (62.67).",
    reference="Ours-A Ts=10 + MixUp",
    metrics=["best_mean_acc", "final_bal", "final_tail", "bbc_bal", "final_ens_acc", "fidelity_tail_kh", "class_sep_F", "ncm_acc"],
    plan=lambda cfg: [run_spec(lab, "cifar10", s, compose(c, head=dict(ts=10)))
                      for s in cfg["seeds"] for lab, c in ARMS]
                     + [run_spec(lab.replace(" Ts=10", ""), "cifar10", s, compose(c), IF=0.01, alpha=0.5)
                        for s in cfg["seeds"] for lab, c in ARMS],
)
