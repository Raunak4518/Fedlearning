"""K22 - Kaggle run: regime breadth (action plan X10 = D-3), new seeds 3-5, FashionMNIST and SVHN, K = 10.

Reviewers' objection: one imbalance setting (IF = 100, Dirichlet 0.5) and the rare classes are always 6-9.
Settings (each against K17's IF = 100 / Dirichlet 0.5 cells, same seeds):
  IF = 10, Dirichlet 0.5                       milder long tail
  IF = 100, Dirichlet 0.1                      stronger label skew (many absent classes per client)
  IF = 100, Dirichlet 1.0                      weaker label skew
  IF = 100, Dirichlet 0.5, random rare classes the classes are relabelled by a seeded permutation before the split
                                               (core.permute_labels), so which classes are rare changes per seed
Arms: the strongest baseline (GeFL-F + AdamW + LA), real-data heads + LA (FedLC / FedRS / balanced softmax are the
same body, Proposition 7), and ours with K17's validation-selected T_s = 10. Every readout is logged.

PRE-REGISTERED (18 Oct 2026, before any run), with the selection rule of K17 (readout chosen on validation):
  B1  ours >= the strongest of the two baselines in every one of the 8 dataset-setting cells (mean over seeds).
  B2  the margin over GeFL-F + AdamW + LA is larger at Dirichlet 0.1 than at Dirichlet 1.0 on both datasets
      (more absent classes per client leave more room for synthetic data, Proposition 8).
  B3  with random rare classes, the margin over GeFL-F + AdamW + LA is within 1 point of K17's on both datasets.
"""
OURS = "GATE+MC+KH+LA+CSL"
ARMS = [("GeFL-F + AdamW + LA", "ADAMW3+LA", 1), ("Real only + LA", "REAL+LA", 0), ("Ours Ts=10", OURS, 10)]
SETTINGS = [(0.1, 0.5, False), (0.01, 0.1, False), (0.01, 1.0, False), (0.01, 0.5, True)]


def _plan(cfg):
    runs = []
    for ds in ["fmnist", "svhn"]:
        for IF, alpha, tp in SETTINGS:
            for s in cfg["seeds"]:
                for lab, c, ts in ARMS:
                    runs.append(run_spec(lab + (" [random tail]" if tp else ""), ds, s, compose(c, head=dict(ts=ts)),
                                         IF=IF, alpha=alpha, K=10, tail_perm=tp))
    return runs


EXPERIMENT = dict(
    name="K22_kaggle_regime_breadth",
    title="Regime breadth: imbalance 10, Dirichlet 0.1 / 1.0, random rare classes (FMNIST, SVHN; seeds 3-5)",
    hypothesis="B1-B3 in the docstring.",
    reference="GeFL-F + AdamW + LA",
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "val_final_bal", "val_er_bal", "final_tail", "final_worst"],
    plan=_plan,
)
