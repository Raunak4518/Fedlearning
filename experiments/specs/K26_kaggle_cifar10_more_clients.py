"""K26 - Kaggle run: several clients per architecture on CIFAR-10 (action plan X13 = D-1), K = 50, long tail
(IF = 100, Dirichlet 0.5), new seeds 3-5. SVHN is covered by K18 (K = 50 / 100), MNIST / FMNIST by K15.

Arms: the decay fix (GeFL-F + AdamW + LA), HWA (GeFL-F + HWA + LA, coupled decay kept) and ours (gated); every readout.

PRE-REGISTERED (18 Oct 2026, before any run):
  D1  HWA - decay fix >= 0 (final balanced accuracy and ER), as on MNIST at K = 50 (K15: +1.8).
  D2  ours + ER >= both baselines + ER (validation-selected readouts).
"""
OURS = "GATE+MC+KH+LA+CSL"
ARMS = [("GeFL-F + AdamW + LA", "ADAMW3+LA"), ("GeFL-F + HWA + LA", "HWA+LA"), ("Ours (gated)", OURS)]
EXPERIMENT = dict(
    name="K26_kaggle_cifar10_more_clients",
    title="CIFAR-10 with 50 clients (5 per architecture): decay fix vs HWA vs ours, seeds 3-5",
    hypothesis="D1-D2 in the docstring.",
    reference="GeFL-F + AdamW + LA",
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "er_own", "fed3r", "ncm", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "val_final_bal", "val_er_bal", "final_tail",
             "cond_norm_tail_over_head_end"],
    plan=lambda cfg: [run_spec(lab, "cifar10", s, compose(c), IF=0.01, alpha=0.5, K=50)
                      for s in cfg["seeds"] for lab, c in ARMS],
)
