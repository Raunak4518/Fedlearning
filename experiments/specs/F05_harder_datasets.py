"""F05 - Harder datasets (full rounds): SVHN and CIFAR-10 under the long tail, CIFAR-10 IID.

Completes the paper's GeFL-F dataset set. CIFAR-10 uses the paper's
10-channel FE, batch 128 and T = 50/200/100; its long tail is capped at the
5,000 images per class available (plan section 2.2). CIFAR-10 IID checks
the paper's weakest cell (Figure 4d: 55.86).
"""
EXPERIMENT = dict(
    name="F05_harder_datasets",
    title="SVHN and CIFAR-10: long tail, and CIFAR-10 IID",
    hypothesis="Ours > +LA > GeFL-F under the long tail on both; no loss on CIFAR-10 IID.",
    reference="GeFL-F",
    metrics=["final_bal", "final_tail", "final_worst", "best_mean_acc", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["svhn", "cifar10"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")),
                                     (FINAL_LABEL, final_method())]]
                     + [run_spec(lab, "cifar10", s, m) for s in cfg["seeds"]
                        for lab, m in [("GeFL-F", compose()), (FINAL_LABEL, final_method())]],
)
