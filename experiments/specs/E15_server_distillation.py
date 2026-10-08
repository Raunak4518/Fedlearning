"""E15 - Server-side ensemble distillation (SED) on generated features (quick pass, then full).

The ensemble of the G heterogeneous headers is far stronger than the average
header (SVHN IID: 82.5 vs 74.5; CIFAR-10 LT: 75.0 vs 62.5), yet the paper's
metric is the mean single-header accuracy. CSL passes the ensemble's
knowledge to clients only through one synthetic epoch that five real epochs
then partly overwrite. SED does it at the server, after aggregation, on fresh
generated features, with the same target as CSL - so no new hyperparameter
except the number of steps, and no new client cost or message.
"""
LABELS = [("GeFL-F", ""), ("+CSL", "CSL"), ("+CSL+SED(20)", "CSL+SED"),
          ("+CSL+SED(60)", ("CSL+SED", dict(head=dict(sed=60)))),
          ("+SED(20)", "SED"), ("Ours (HWA+LA+CSL)", "HWA+LA+CSL"), ("Ours+SED(20)", "HWA+LA+CSL+SED")]


def _m(c):
    return compose(c[0], **c[1]) if isinstance(c, tuple) else compose(c)


EXPERIMENT = dict(
    name="E15_server_distillation",
    title="Server-side ensemble distillation on generated features",
    hypothesis="SED adds to CSL in IID (towards the 84.28 FMNIST best) and under the long tail.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_tail", "final_ens_acc"],
    plan=lambda cfg: [run_spec(lab, ds, s, _m(c), IF=IF, alpha=a)
                      for ds in ["fmnist", "mnist"] for IF, a in [(1.0, None), (0.01, 0.5)]
                      for s in cfg["seeds"] for lab, c in LABELS],
)
