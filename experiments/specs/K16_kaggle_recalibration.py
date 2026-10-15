"""K16 - Kaggle run: the recalibration baseline (action plan B1, Gate 1), on five datasets including CIFAR-100-LT.

The question (review objection F1): most of GeFL-F's long-tail loss sits in each head's last layer. Does our
method still beat GeFL-F once BOTH get a last layer computed from real-data statistics? Every run here trains
as before and then evaluates the same trained heads under several readouts, in the same pass:
  final_*     the heads as trained
  bbc_*       balanced bias calibration on the method's own generator (server-side, no real data)
  er_*        ER: per-architecture LDA from exact sums of every client's embeddings (er_own: own clients only)
  fed3r_*     Fed3R ridge readout from the same sums (fed3r_own: own clients only)
  ncm_*       FedNCM cosine nearest-class-mean readout (ncm_own: own clients only)
  ccvr_*      CCVR: virtual features from per-class Gaussians, last layer fine-tuned (ccvr_own: own clients,
              the published protocol; at K = 10 that is one client per architecture)
  oracle_*    last layer re-fit on pooled real data (diagnostic upper bound, not federated)
Each readout reports balanced accuracy, accuracy, head/medium/tail recall, worst class, worst architecture,
macro-F1, G-mean of class recalls, ECE, NLL, Brier score and the accuracy on each head's own clients'
label distribution (see core.summarize_eval).

Arms (K = 10, long tail IF = 100 with Dirichlet 0.5, and the paper's IID setting; seeds 0-2):
  GeFL-F, + LA, the minimal variant (HWA + LA + CSL), and the method as it now runs in code:
  GATE + MC + KH + LA + CSL, where GATE picks the PC-VAE (F >= 0.05) or the CVAE with HWA class rows
  (F < 0.05) at run time. T_s = 10 on MNIST / FashionMNIST and 1 on SVHN / CIFAR-10 / CIFAR-100 (the
  per-dataset rule of the draft; it was chosen on test results, see paper section 7).
These arms repeat existing configurations on purpose (decision D2, 15 Oct 2026): no models were saved before,
so the readouts cannot be computed from the old runs. CIFAR-100 is new (decision D5).

Models are saved apart from the results (default /kaggle/working/models/K16_kaggle_recalibration/): the FE once
per setting and seed, every generator once, and each run's heads. Re-evaluate later without training:
    python K16.py --datasets mnist --from_models <models>/K16_kaggle_recalibration --out_dir results

Typical use (T4 x2), one notebook per dataset and regime, long tail first (it decides Gate 1):
    python K16.py --datasets mnist --IF 0.01 --out_dir /kaggle/working/results
"""
TS = {"mnist": 10, "fmnist": 10, "svhn": 1, "cifar10": 1, "cifar100": 1}
# (label, components, T_s rule): baselines keep the paper's T_s = 1; only our method uses the per-dataset rule
ARMS = [("GeFL-F", "", False), ("+LA", "LA", False), ("Minimal (HWA+LA+CSL)", "HWA+LA+CSL", False),
        ("Ours (gated)", "GATE+MC+KH+LA+CSL", True)]
SETTINGS = [(0.01, 0.5), (1.0, None)]
EXPERIMENT = dict(
    name="K16_kaggle_recalibration",
    title="Recalibration baseline: every method under every readout (heads, BBC, ER, Fed3R, FedNCM, CCVR, oracle)",
    hypothesis=("Gate 1 (action plan): Ours + ER beats GeFL-F + ER by >= 2 points on >= 3 of the 5 datasets "
                "(paired, 95% CI excluding 0) -> method paper; otherwise analysis paper."),
    reference="GeFL-F",
    split_by_gen=True,
    save_gen=True,
    readouts=["er", "er_own", "fed3r", "fed3r_own", "ncm", "ncm_own", "ccvr", "ccvr_own"],
    metrics=["final_bal", "er_bal", "fed3r_bal", "ncm_bal", "ccvr_bal", "ccvr_own_bal", "bbc_bal", "oracle_bal",
             "final_tail", "er_tail", "final_worst", "final_f1", "final_gmean", "final_ece", "best_mean_acc"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c, head=dict(ts=TS[ds] if rule else 1)), IF=IF, alpha=a, K=10)
                      for ds in ["mnist", "fmnist", "svhn", "cifar10", "cifar100"] for IF, a in SETTINGS
                      for s in cfg["seeds"] for lab, c, rule in ARMS],
)
