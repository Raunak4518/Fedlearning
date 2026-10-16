"""K19 - Kaggle run: the remaining published baselines of action plan C6, paired with K17 (same splits, seeds 3-5).

K17 compared against GeFL-F, the decay fix and the published long-tail FL losses (balanced softmax / FedRoD's
generic head, FedLC, FedRS). This adds:
  FSG + LA              the Gaussian generator built from exact federated statistics (CCVR-like virtual features
                        used during training), with logit adjustment
  FedDF                 server-side ensemble distillation (Lin et al., NeurIPS 2020) of every architecture's head into
                        each head, on generated features (FedDF allows generator-produced inputs), pure ensemble target
  FedDF + LA            the same, with logit adjustment in the clients' training
FedProto is not included: its prototype regulariser needs one embedding dimension shared by all architectures,
which GeFL-F's heterogeneous heads do not have; its nearest-prototype inference is the FedNCM readout, which every
K16 / K17 run already reports. FedMD needs a labelled public dataset; FedDF is its closest unlabelled analogue.
Every run is scored under every readout (test and validation), so k17_analysis.py treats these as further
baselines (validation-selected readout, paired by seed with K17's "Ours").
"""
ARMS = [("FSG + LA", "GAUSS+LA"), ("FedDF", "FEDDF"), ("FedDF + LA", "FEDDF+LA")]
EXPERIMENT = dict(
    name="K19_kaggle_more_baselines",
    title="Further published baselines (FSG + LA, FedDF, FedDF + LA), paired with K17 on seeds 3-5",
    hypothesis="K17's validation-selected Ours beats each of them by >= 2 points on >= 4 of 5 datasets.",
    reference="FedDF",
    seeds=[3, 4, 5],
    split_by_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "ccvr_bal", "bbc_bal", "val_final_bal", "val_er_bal", "final_tail", "final_worst"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c), IF=0.01, alpha=0.5, K=10)
                      for ds in ["mnist", "fmnist", "svhn", "cifar10", "cifar100"] for s in cfg["seeds"] for lab, c in ARMS],
)
