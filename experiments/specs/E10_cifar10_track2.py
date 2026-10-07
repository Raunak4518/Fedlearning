"""E10 - Track 2: beat GeFL-F on CIFAR-10 in the paper's own setting.

The paper admits GeFL-F is weak on CIFAR-10 (55.86 vs FedProx 55.10 at
K=10) and blames the feature extractor. IID, K=10, paper protocol. Tests:
longer FE warm-up (T_FE 100, 150); updating G_F during stage (iii) (paper
Table VII shows this helps GeFL, untested for GeFL-F; T_KA split 100 + 100);
PCM (synthetic knowledge not overwritten). Target: > 55.86.
"""
EXPERIMENT = dict(
    name="E10_cifar10_track2",
    title="CIFAR-10, IID paper setting: FE warm-up, generator updates, PCM",
    hypothesis="at least one variant exceeds GeFL-F and the paper's 55.86 best_mean_acc.",
    reference="GeFL-F",
    seeds=[0, 1],
    metrics=["best_mean_acc", "final_acc", "final_bal"],
    plan=lambda cfg: [run_spec(lab, "cifar10", s, m) for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("T_FE=100", compose(fe=dict(T_FE=100))),
                                     ("T_FE=150", compose(fe=dict(T_FE=150))),
                                     ("+GENUPD", compose("GENUPD", gen=dict(T_KA=100))),
                                     ("+PCM", compose("PCM")),
                                     ("+PCM+GENUPD", compose("PCM+GENUPD", gen=dict(T_KA=100)))]],
)
