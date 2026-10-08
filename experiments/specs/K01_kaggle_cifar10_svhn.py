"""K01 - Kaggle run: CIFAR-10 and SVHN, paper setting and long tail (full schedule, 3 seeds).

Made to run on Kaggle (GPU T4 x2, Internet on): it uses both GPUs, downloads
CIFAR-10 (torchvision) and SVHN (Hugging Face mirror) itself.

Part A - the paper's own setting (IID, K=10, Table XIV/XV, Figure 4):
  targets: GeFL-F CVAE-F  CIFAR-10 55.86 (Fig. 4d), SVHN 76.26 (Fig. 4c)
           best CIFAR-10 number of any GeFL variant: 62.67 (GeFL+MixUp, Table IV)
  compared: GeFL-F; +HWA+LA (our final method, expected = GeFL-F in IID);
            +CSL / +CSLM (consensus soft labels for synthetic features,
            sequential / interleaved into every real batch).
  metric of the paper: best_mean_acc.
Part B - long tail IF=100 + Dirichlet(0.5), K=10:
  GeFL-F; +LA; +HWA+LA (ours); FSG+LA (Gaussian generator); +HWA+LA+CSLM.

Typical use (one Kaggle session can do one dataset; finished runs are
skipped when runs.jsonl is present, so a cut-off session can resume):
    python K01.py --datasets cifar10 --out_dir /kaggle/working/results
    python K01.py --datasets svhn    --out_dir /kaggle/working/results
"""
IID = [("GeFL-F", compose()), ("+HWA+LA (ours)", compose("HWA+LA")),
       ("+CSL (beta=0.5)", compose("CSL")), ("+CSLM (interleaved)", compose("CSLM"))]
LT = [("GeFL-F", compose()), ("+LA", compose("LA")), ("+HWA+LA (ours)", compose("HWA+LA")),
      ("FSG+LA", compose("GAUSS+LA")), ("+HWA+LA+CSLM", compose("HWA+LA+CSLM"))]

EXPERIMENT = dict(
    name="K01_kaggle_cifar10_svhn",
    title="CIFAR-10 and SVHN: paper setting (IID) and long tail, full schedule",
    hypothesis="IID: a CSL variant beats GeFL-F's best_mean_acc (55.86 / 76.26). Long tail: ours > +LA > GeFL-F.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_tail", "final_ens_acc", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m) for ds in ["cifar10", "svhn"] for s in cfg["seeds"] for lab, m in IID]
                     + [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                        for ds in ["cifar10", "svhn"] for s in cfg["seeds"] for lab, m in LT],
)
