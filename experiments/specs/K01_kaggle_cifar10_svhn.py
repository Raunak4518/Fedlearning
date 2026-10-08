"""K01 - Kaggle run: CIFAR-10 and SVHN, paper setting and long tail (full schedule, 3 seeds).

Made to run on Kaggle (GPU T4 x2, Internet on): it uses both GPUs, downloads
CIFAR-10 and SVHN itself (Hugging Face mirrors, verified identical).

Part A - the paper's own setting (IID, K=10, Table XIV/XV, Figure 4):
  targets: GeFL-F CVAE-F  CIFAR-10 55.86 (Fig. 4d), SVHN 76.26 (Fig. 4c)
           best CIFAR-10 number of any GeFL variant: 62.67 (GeFL+MixUp, Table IV)
  compared: GeFL-F; +CSL (consensus soft labels, beta=0.5); Ours = HWA+LA+CSL.
  metric of the paper: best_mean_acc.
Part B - long tail IF=100 + Dirichlet(0.5), K=10:
  GeFL-F; +LA; +HWA+LA; Ours = HWA+LA+CSL; FSG+LA and FSG+LA+CSL (the Gaussian
  generator lost on SVHN but led on CIFAR-10 in the first run; it costs ~2 min a run).
(CSLM was dropped after the first run: it lost to GeFL-F.)

Typical use (one Kaggle session can do one dataset; finished runs are
skipped when runs.jsonl is present, so a cut-off session can resume):
    python K01.py --datasets cifar10 --out_dir /kaggle/working/results   # ~6 h on T4 x2
    python K01.py --datasets svhn    --out_dir /kaggle/working/results   # ~2.5 h on T4 x2
"""
OURS_K = ("Ours (HWA+LA+CSL)", final_method())
IID = [("GeFL-F", compose()), ("+CSL (beta=0.5)", compose("CSL")), OURS_K]
LT = [("GeFL-F", compose()), ("+LA", compose("LA")), ("+HWA+LA", compose("HWA+LA")), OURS_K,
      ("FSG+LA", compose("GAUSS+LA")), ("FSG+LA+CSL", compose("GAUSS+LA+CSL"))]

EXPERIMENT = dict(
    name="K01_kaggle_cifar10_svhn",
    title="CIFAR-10 and SVHN: paper setting (IID) and long tail, full schedule",
    hypothesis="IID: Ours and +CSL beat GeFL-F's best_mean_acc (55.86 / 76.26). Long tail: Ours >= +HWA+LA > +LA > GeFL-F.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_tail", "final_ens_acc", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, m) for ds in ["cifar10", "svhn"] for s in cfg["seeds"] for lab, m in IID]
                     + [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5)
                        for ds in ["cifar10", "svhn"] for s in cfg["seeds"] for lab, m in LT],
)
