"""E05 - Does it hold on harder datasets?  (plan E3.3)

SVHN and CIFAR-10 complete the paper's GeFL-F dataset set (MNIST and
FMNIST are in E03). Long tail IF=100 + Dir(0.5), K=10. CIFAR-10's tail
cannot keep the paper's 25,000-image budget (5,000 per class available),
so it uses the capped profile (plan section 2.2).
"""
EXPERIMENT = dict(
    name="E05_datasets",
    title="SVHN and CIFAR-10 under a long tail",
    hypothesis="Ours > +LA > GeFL-F on final_bal on both datasets.",
    reference="GeFL-F",
    plan=lambda cfg: [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                      for ds in ["svhn", "cifar10"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+LA", compose("LA")),
                                     ("+HWA+LCD+LA", compose("HWA+LCD+LA")),
                                     ("Ours (HWA+LCD+PCM+LA)", compose(OURS)),
                                     ("LG-FedAvg+LA", baseline_method("LG-FedAvg", la=True))]],
)
