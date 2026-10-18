"""K23 - Kaggle run: partial participation (action plan X11 = D-2), K = 100 clients, 10% and 20% of them per round,
MNIST and FashionMNIST, long tail (IF = 100, Dirichlet 0.5), new seeds 3-5.

Reviewers' objection: full participation is unrealistic. With partial participation a class's holders are often
absent from a round, so the flat-averaging dilution of §3.6 gets worse and the non-holder decay of §3 acts on more
of the sampled clients. The same seeded subset of clients is drawn for every method (Ctx.active).
Arms: GeFL-F, the decay fix (GeFL-F + AdamW + LA), HWA (GeFL-F + HWA + LA, coupled decay kept) and ours (gated).

PRE-REGISTERED (18 Oct 2026, before any run):
  R1  HWA's margin over the decay fix at 10% participation is >= its margin at full participation (K15, K = 100:
      +1.3 on MNIST, +1.4 on FMNIST), on both datasets.
  R2  ours + ER >= the best of the three baselines + ER (each with its validation-selected readout) in all 4 cells.
"""
OURS = "GATE+MC+KH+LA+CSL"
ARMS = [("GeFL-F", ""), ("GeFL-F + AdamW + LA", "ADAMW3+LA"), ("GeFL-F + HWA + LA", "HWA+LA"), ("Ours (gated)", OURS)]
EXPERIMENT = dict(
    name="K23_kaggle_partial_participation",
    title="Partial participation: K = 100, 10% / 20% of clients per round (MNIST, FMNIST; seeds 3-5)",
    hypothesis="R1-R2 in the docstring.",
    reference="GeFL-F",
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "ccvr", "fed3r", "ncm"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "val_final_bal", "val_er_bal", "final_tail",
             "cond_norm_tail_over_head_end"],
    plan=lambda cfg: [run_spec(f"{lab} [p={pf}]", ds, s, compose(c), IF=0.01, alpha=0.5, K=100, pfrac=pf)
                      for ds in ["mnist", "fmnist"] for pf in [0.1, 0.2] for s in cfg["seeds"] for lab, c in ARMS],
)
