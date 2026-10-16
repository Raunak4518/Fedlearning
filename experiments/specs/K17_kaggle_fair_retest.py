"""K17 - Kaggle run: the fair re-test (action plan C4 + C6 + C5), new seeds 3-5, long tail, five datasets.

What it fixes from the reviews: (1) every setting so far was chosen on the test set; (2) no published long-tailed FL
method was compared; (3) the CIFAR-10 class-row branch was never run. Nothing here re-uses old seeds.

Arms (K = 10, IF = 100, Dirichlet 0.5; every run is also scored under every readout, as in K16):
  GeFL-F
  GeFL-F + AdamW + LA  the simple fix for the row collapse (K15: decoupled decay restores the rows) + logit
                       adjustment; added 16 Oct after K15, before any K17 run: the strongest simple baseline
  Real only            heads trained on real features only (no synthetic phase): a plain FL head baseline
  Real only + LA       = balanced softmax with the client's counts (FedRoD's generic head; FedLC's family)
  Real only + FedLC    FedLC logit calibration (Zhang et al., ICML 2022), tau = 1
  Real only + FedRS    FedRS restricted softmax (Li & Zhan, KDD 2021), alpha = 0.5
  Ours Ts=1, Ours Ts=10    our generator (GATE + MC + KH) + LA + CSL, both synthetic budgets
  Ours rows Ts=1       (CIFAR-10 only) the class-row branch forced, to test the gate near its threshold (C5)
Readouts per run: heads, BBC, ER, Fed3R, FedNCM, CCVR (all clients / own clients) and the oracle, each on the test
set AND on the validation set (val_*: the 2,000 held-out training images no client holds).

PRE-REGISTERED (16 Oct 2026, before any run):
  Selection, on validation only:  for each dataset, Ours' T_s is the one with the higher mean val_er_bal over the
    3 seeds; for every method, its readout is the federated readout (heads, BBC, ER, Fed3R, FedNCM, CCVR,
    all-client or own-client) with the highest mean val_*_bal. The oracle is never selectable.
  Primary test:  Ours (selected) vs each baseline (selected), paired over seeds 3-5, final balanced accuracy on the
    TEST set; Holm across the 6 baselines within each dataset.
  Predictions:  P1 Ours beats every baseline by >= 2 points on >= 4 of 5 datasets (Holm-adjusted p < 0.05 not
    required with n = 3; 95% CI of the paired difference excluding 0 is);  P2 the validation-selected T_s equals
    the per-dataset rule used so far (10 on MNIST / FMNIST, 1 elsewhere) on >= 4 of 5 datasets;  P3 on CIFAR-10 the
    gate's anchor branch is within 1 point of, or better than, the forced row branch;  P4 (idea 3, label-free gap)
    across all runs, ro_disagree (head vs ER disagreement on unlabelled held-out features) has Spearman >= 0.7 with
    the true readout gap er_bal - final_bal.
Reference arm for the summary's p-values: GeFL-F (same seeds, same job).
"""
TS = {"mnist": 10, "fmnist": 10, "svhn": 1, "cifar10": 1, "cifar100": 1}
OURS = "GATE+MC+KH+LA+CSL"
ARMS = [("GeFL-F", "", 1), ("GeFL-F + AdamW + LA", "ADAMW3+LA", 1), ("Real only", "REAL", 0), ("Real only + LA", "REAL+LA", 0),
        ("Real only + FedLC", "REAL+FEDLC", 0), ("Real only + FedRS", "REAL+FEDRS", 0),
        ("Ours Ts=1", OURS, 1), ("Ours Ts=10", OURS, 10)]
ROWS = ("Ours rows Ts=1", "HWA+MC+KH+LA+CSL", 1)
EXPERIMENT = dict(
    name="K17_kaggle_fair_retest",
    title="Fair re-test: validation-selected method vs published long-tail FL losses, new seeds 3-5",
    hypothesis="P1-P4 in the docstring.",
    reference="GeFL-F",
    seeds=[3, 4, 5],
    split_by_gen=True,
    save_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "bbc_bal", "val_final_bal", "val_er_bal", "final_tail",
             "final_worst", "final_f1", "ro_disagree"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=ts)), IF=0.01, alpha=0.5, K=10)
                      for ds in ["mnist", "fmnist", "svhn", "cifar10", "cifar100"] for s in cfg["seeds"]
                      for lab, c, ts in ARMS + ([ROWS] if ds == "cifar10" else [])],
)
