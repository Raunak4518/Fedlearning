"""K04 - Kaggle run: our method on the paper's diffusion generator DDPM-F, SVHN and CIFAR-10 (paper's IID setting).

Made for Kaggle (GPU T4 x2, Internet on); both GPUs are used, and one seed's
generators are split across them (split_by_gen), because a DDPM-F run is
hours long on CIFAR-10 (T_KA = 200 rounds, 200-step sampling).

Targets (Fig. 4, best_mean_acc, K = 10):
  SVHN      GeFL-F DDPM-F 73.38   best of all variants 76.26 (GeFL-F CVAE-F)
  CIFAR-10  GeFL-F DDPM-F 56.61   best of all variants 59.36 (image-space GeFL + DDPM)
Compared, all on the authors' DDPM-F: GeFL-F, +CSL, Ours (HWA+LA+CSL).

Typical use (rough times on T4 x2; finished runs are skipped on rerun):
    python K04.py --datasets svhn    --out_dir /kaggle/working/results            # 3 seeds, ~7 h
    python K04.py --datasets cifar10 --seeds 0 --out_dir /kaggle/working/results  # 1 seed,  ~6 h
    python K04.py --datasets cifar10 --seeds 1 --out_dir /kaggle/working/results  # (other account)
"""
DD = "DDPM"
LABELS = [("GeFL-F (DDPM-F)", DD), ("+CSL (DDPM-F)", DD + "+CSL"), ("Ours (DDPM-F)", DD + "+HWA+LA+CSL")]
EXPERIMENT = dict(
    name="K04_kaggle_ddpm_cifar10_svhn",
    title="Our method on DDPM-F: SVHN and CIFAR-10, paper's IID setting",
    hypothesis="Ours (DDPM-F) > GeFL-F (DDPM-F) on both; SVHN > 76.26.",
    reference="GeFL-F (DDPM-F)",
    split_by_gen=True,
    metrics=["best_mean_acc", "final_bal", "final_ens_acc", "fidelity_tail"],
    plan=lambda cfg: [run_spec(lab, ds, s, compose(c)) for ds in ["svhn", "cifar10"] for s in cfg["seeds"] for lab, c in LABELS],
)
