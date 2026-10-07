"""E06 - Do-no-harm, and can we beat GeFL-F in the paper's own setting?  (plan E3.5)

IID balanced clients, K=10, exactly the paper's protocol. Proposition 2
predicts HWA reduces to flat averaging here, and LA's prior is uniform, so
both are near no-ops. PCM is NOT a no-op: it interleaves synthetic
features into every real batch instead of a separate synthetic epoch that
the following 5 real epochs overwrite. Hypothesis: PCM >= GeFL-F in IID
(synthetic knowledge is no longer forgotten), i.e. an improvement over the
published numbers in the published setting.
"""
EXPERIMENT = dict(
    name="E06_iid_paper_setting",
    title="IID paper setting: do-no-harm and PCM",
    hypothesis="Ours within +-0.5 pp or better than GeFL-F; +PCM >= GeFL-F.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_acc", "final_bal", "final_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m)
                      for ds in ["mnist", "fmnist", "svhn"] for s in cfg["seeds"]
                      for lab, m in [("GeFL-F", compose()), ("+PCM", compose("PCM")),
                                     ("+PCM (r=0.5)", compose("PCM", head=dict(mix_ratio=0.5))),
                                     ("Ours (HWA+LCD+PCM+LA)", compose(OURS))]],
)
