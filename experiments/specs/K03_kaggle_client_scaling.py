"""K03 - Kaggle run: more clients, K = 50 and K = 100 (MNIST, FMNIST; full schedule, 3 seeds).

Made for Kaggle (GPU T4 x2, Internet on); both GPUs are used.
Total data is fixed (paper footnote 5), so each client holds less as K grows,
and each rare class has a smaller fraction of holders. Flat averaging then
dilutes and decays its conditioning rows further (Theorem 1, eq. 2.2).

Part A - the paper's own setting (IID), Figure 4 targets (best_mean_acc):
  MNIST  K=50 95.04, K=100 94.63    FMNIST K=50 82.21, K=100 81.65   (GeFL-F CVAE-F)
  best variant of any kind in the paper: MNIST 95.04 / 94.63, FMNIST 82.96 / 81.65
  compared: GeFL-F; +CSL; Ours = HWA+LA+CSL.
Part B - long tail IF=100 + Dirichlet(0.5):
  GeFL-F; +LA; +HWA+LA; Ours; FSG+LA and FSG+LA+CSL (the Gaussian
  sufficient-statistics generator, whose aggregation is exact at any K).
  MNIST K=100 under the long tail is run locally (F04), so it is left out here.

Typical use (finished runs are skipped when runs.jsonl is present):
    python K03.py --out_dir /kaggle/working/results            # everything, ~6 h on T4 x2
    python K03.py --K 50 --out_dir /kaggle/working/results     # half, to split across accounts
    python K03.py --K 100 --out_dir /kaggle/working/results
"""
IID = [("GeFL-F", compose()), ("+CSL (beta=0.5)", compose("CSL")), ("Ours (HWA+LA+CSL)", final_method())]
LT = [("GeFL-F", compose()), ("+LA", compose("LA")), ("+HWA+LA", compose("HWA+LA")),
      ("Ours (HWA+LA+CSL)", final_method()), ("FSG+LA", compose("GAUSS+LA")), ("FSG+LA+CSL", compose("GAUSS+LA+CSL"))]


def _plan(cfg):
    runs = [run_spec(lab, ds, s, m, K=K) for K in [50, 100] for ds in ["mnist", "fmnist"]
            for s in cfg["seeds"] for lab, m in IID]
    runs += [run_spec(lab, ds, s, m, IF=0.01, alpha=0.5, K=K)
             for K, ds in [(50, "fmnist"), (100, "fmnist")] for s in cfg["seeds"] for lab, m in LT]
    return runs


EXPERIMENT = dict(
    name="K03_kaggle_client_scaling",
    title="Client scaling K = 50, 100: paper setting (IID) and long tail",
    hypothesis="IID: Ours and +CSL beat GeFL-F at K=50/100. Long tail: Ours > +LA > GeFL-F, gap growing with K.",
    reference="GeFL-F",
    metrics=["best_mean_acc", "final_bal", "final_tail", "fidelity_tail", "cond_norm_tail_over_head_end"],
    plan=_plan,
)
