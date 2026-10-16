"""E33 - ERL pilot: the exact balanced readout recomputed during training (new idea, 16 Oct 2026).

Finding behind it (K16): most of the long-tail loss sits in each head's last layer, which the clients learn by SGD on
their own skewed labels (at K = 10, one client per architecture). ER fixes it only after training; meanwhile the body
was trained under a biased last layer. ERL: every round (ERL) or every 5 rounds (ERL5) the server sets each
architecture's last layer to the class-balanced LDA readout from exact sums of every client's embeddings
(counts, sums of e and e e^T; secure-aggregatable; needs each body broadcast), and clients train only the body.

PRE-REGISTERED (written before any run; reduced-schedule CPU pilot, seed 0, MNIST + FMNIST long tail):
  P1  heads as trained under ERL are within 1 point of ER on the same run (no post-hoc step needed);
  P2  Ours + ERL beats Ours + ER (the K16 recipe) by >= 1 point on at least one dataset, and on none loses > 1;
  P3  tail recall of Ours + ERL >= Ours + ER.
  Kill rule: if P2 fails on both datasets, ERL is dropped (no Kaggle run).
References in this job (same seed and split): GeFL-F, Ours (gated); their er_* fields give "+ ER".
"""
TS = {"mnist": 10, "fmnist": 10, "svhn": 1, "cifar10": 1, "cifar100": 1}
ARMS = [("GeFL-F", "", False), ("GeFL-F + ERL", "ERL", False), ("Ours (gated)", "GATE+MC+KH+LA+CSL", True),
        ("Ours + ERL", "GATE+MC+KH+LA+CSL+ERL", True), ("Ours + ERL5", "GATE+MC+KH+LA+CSL+ERL5", True)]
EXPERIMENT = dict(
    name="E33_exact_readout_loop",
    title="ERL: exact balanced readout recomputed during training (pilot)",
    hypothesis="P1-P3 in the docstring: heads under ERL match ER without post-hoc step; Ours + ERL > Ours + ER by >= 1.",
    reference="Ours (gated)",
    save_gen=False,
    metrics=["final_bal", "er_bal", "final_tail", "er_tail", "final_worst", "oracle_bal", "final_ece"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=TS[ds] if rule else 1)), IF=0.01, alpha=0.5, K=10)
                      for ds in ["mnist", "fmnist", "cifar10", "svhn", "cifar100"] for s in cfg["seeds"]
                      for lab, c, rule in ARMS],
)
