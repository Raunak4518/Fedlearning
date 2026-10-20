"""Build one ready-to-run Kaggle notebook per job from the standalone K0*.py scripts.

    python make_kaggle_notebooks.py [out_dir]

Each notebook: instructions -> environment check (2 GPUs, internet) -> writes
the script -> runs it on both GPUs -> zips the results under the right name.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "kaggle_notebooks")
# Usage: python make_kaggle_notebooks.py NB38 NB39   (writes only the named notebooks; no names = list jobs)

JOBS = [  # (notebook name, script, arguments, result folder name, rough hours, what it does)
    ("NB1_cifar10_main", "K01_kaggle_cifar10_svhn", "--datasets cifar10", "K01_kaggle_cifar10_svhn__cifar10", 6,
     "CIFAR-10: paper's IID setting and the 100:1 long tail, our method vs GeFL-F."),
    ("NB2_svhn_main", "K01_kaggle_cifar10_svhn", "--datasets svhn", "K01_kaggle_cifar10_svhn__svhn", 2.5,
     "SVHN: paper's IID setting and the 100:1 long tail, our method vs GeFL-F."),
    ("NB3_regimes_privacy_ablations", "K02_kaggle_regimes_privacy_ablations", "", "K02_kaggle_regimes_privacy_ablations", 3.5,
     "MNIST/FMNIST: other data regimes, privacy (noised counts, memorisation), ablations."),
    ("NB4_clients_K100", "K03_kaggle_client_scaling", "--K 100", "K03_kaggle_client_scaling__K100", 5,
     "MNIST/FMNIST with 100 clients (paper Figure 4)."),
    ("NB5_clients_K50", "K03_kaggle_client_scaling", "--K 50", "K03_kaggle_client_scaling__K50", 2,
     "MNIST/FMNIST with 50 clients (paper Figure 4)."),
    ("NB9_newgen_svhn", "K05_kaggle_newgen_cifar10_svhn", "--datasets svhn", "K05_kaggle_newgen_cifar10_svhn__svhn", 2.5,
     "SVHN, long tail + IID: only the NEW generator arms (PC-VAE, moment calibration); baselines come from NB2."),
    ("NB10_newgen_cifar10_longtail", "K05_kaggle_newgen_cifar10_svhn", "--datasets cifar10 --IF 0.01",
     "K05_kaggle_newgen_cifar10_svhn__cifar10_lt", 4.5,
     "CIFAR-10 long tail: only the NEW generator arms; baselines come from NB1."),
    ("NB11_newgen_cifar10_iid", "K05_kaggle_newgen_cifar10_svhn", "--datasets cifar10 --IF 1.0",
     "K05_kaggle_newgen_cifar10_svhn__cifar10_iid", 4.5,
     "CIFAR-10 in the paper's IID setting (paper GeFL-F 55.86): only the NEW generator arms; baselines come from NB1."),
    ("NB12_newgen_fmnist_many_clients", "K06_kaggle_newgen_many_clients", "", "K06_kaggle_newgen_many_clients", 5,
     "FashionMNIST, 50 and 100 clients: only the NEW arms (plus Ours with the fixed CSL); baselines come from NB4/NB5."),
    ("NB13_baselines_cifar10_svhn", "K07_kaggle_baselines_cifar10_svhn", "", "K07_kaggle_baselines_cifar10_svhn", 2.5,
     "SVHN + CIFAR-10, long tail + IID: the paper's federated baselines (FedAvg, LG-FedAvg, LG-FedAvg+LA) - never run under imbalance before."),
    ("NB14_anchored_cifar10_longtail", "K08_kaggle_anchored_stack", "--datasets cifar10 --IF 0.01",
     "K08_kaggle_anchored_stack__cifar10_lt", 3.5,
     "CIFAR-10 long tail: the final anchored method (PC + MC + KH + BBC with LA + CSL), T_s = 1 and 10."),
    ("NB15_anchored_cifar10_iid", "K08_kaggle_anchored_stack", "--datasets cifar10 --IF 1.0",
     "K08_kaggle_anchored_stack__cifar10_iid", 3.5,
     "CIFAR-10 in the paper's IID setting (paper GeFL-F 55.86): the final anchored method, T_s = 1 and 10."),
    ("NB16_anchored_svhn", "K08_kaggle_anchored_stack", "--datasets svhn", "K08_kaggle_anchored_stack__svhn", 2,
     "SVHN long tail + IID: the final anchored method, T_s = 1 and 10."),
    ("NB17_anchored_fmnist_many_clients", "K08_kaggle_anchored_stack", "--datasets fmnist",
     "K08_kaggle_anchored_stack__fmnist", 4,
     "FashionMNIST with 50 and 100 clients: the final anchored method, T_s = 1 and 10."),
    ("NB18_ablation_final_method", "K09_kaggle_ablation_final", "", "K09_kaggle_ablation_final", 3,
     "MNIST + FashionMNIST long tail: the final method with each component removed in turn (LA, CSL, MC, KH, PC)."),
    ("NB19_hybrid_svhn", "K10_kaggle_hybrid_svhn_cifar10", "--datasets svhn", "K10_kaggle_hybrid_svhn_cifar10__svhn", 2,
     "SVHN long tail + IID: the hybrid generator (exact-mean anchor + HWA class rows) vs CVAE + HWA with MC + KH."),
    ("NB20_hybrid_cifar10_longtail", "K10_kaggle_hybrid_svhn_cifar10", "--datasets cifar10 --IF 0.01",
     "K10_kaggle_hybrid_svhn_cifar10__cifar10_lt", 3.5,
     "CIFAR-10 long tail: the hybrid generator vs CVAE + HWA with MC + KH."),
    ("NB21_hybrid_cifar10_iid", "K10_kaggle_hybrid_svhn_cifar10", "--datasets cifar10 --IF 1.0",
     "K10_kaggle_hybrid_svhn_cifar10__cifar10_iid", 3.5,
     "CIFAR-10 in the paper's IID setting (paper GeFL-F 55.86): the hybrid generator vs CVAE + HWA with MC + KH."),
    ("NB22_covariance_svhn", "K11_kaggle_covariance_anchoring", "--datasets svhn", "K11_kaggle_covariance_anchoring__svhn", 2.5,
     "SVHN long tail + IID: MC-S (exact class mean + class covariance) against Ours-A, plus the hybrid generator with MC-S."),
    ("NB23_covariance_cifar10_longtail", "K11_kaggle_covariance_anchoring", "--datasets cifar10 --IF 0.01",
     "K11_kaggle_covariance_anchoring__cifar10_lt", 3,
     "CIFAR-10 long tail: MC-S against Ours-A, at 1 and 10 synthetic epochs."),
    ("NB24_covariance_cifar10_iid", "K11_kaggle_covariance_anchoring", "--datasets cifar10 --IF 1.0",
     "K11_kaggle_covariance_anchoring__cifar10_iid", 3,
     "CIFAR-10 in the paper's IID setting: MC-S against Ours-A, at 1 and 10 synthetic epochs."),
    ("NB25_final_svhn", "K12_kaggle_final_gated", "--datasets svhn --K 10", "K12_kaggle_final_gated__svhn_K10", 2,
     "SVHN long tail + IID: the final method's low-separation branch (HWA class rows) with MC vs MC-S."),
    ("NB26_final_svhn_K50", "K12_kaggle_final_gated", "--datasets svhn --K 50", "K12_kaggle_final_gated__svhn_K50", 3,
     "SVHN, 50 clients, paper's IID setting (paper: 73.64): GeFL-F and the final method."),
    ("NB27_final_svhn_K100", "K12_kaggle_final_gated", "--datasets svhn --K 100", "K12_kaggle_final_gated__svhn_K100", 4,
     "SVHN, 100 clients, paper's IID setting (paper: 76.00): GeFL-F and the final method."),
    ("NB28_final_cifar10_K50", "K12_kaggle_final_gated", "--datasets cifar10 --K 50 --seeds 0 1", "K12_kaggle_final_gated__cifar10_K50", 6,
     "CIFAR-10, 50 clients, paper's IID setting (paper: GeFL-F 53.19, best 56.31): GeFL-F and the final method, 2 seeds."),
    ("NB29_final_cifar10_K100", "K12_kaggle_final_gated", "--datasets cifar10 --K 100 --seeds 0 1", "K12_kaggle_final_gated__cifar10_K100", 8,
     "CIFAR-10, 100 clients, paper's IID setting (paper: GeFL-F 51.46, best 53.95): GeFL-F and the final method, 2 seeds."),
    ("NB30_beat_mixup_cifar10_iid", "K13_kaggle_vct_vs_mixup", "--IF 1.0", "K13_kaggle_vct_vs_mixup__cifar10_iid", 5,
     "CIFAR-10, paper's IID setting (the paper's GeFL + MixUp: 62.67): our new VCT against MixUp, inside the final method."),
    ("NB31_vct_cifar10_longtail", "K13_kaggle_vct_vs_mixup", "--IF 0.01", "K13_kaggle_vct_vs_mixup__cifar10_lt", 2,
     "CIFAR-10 long tail: VCT against MixUp, inside the final method."),
    ("NB36_ablation_seeds_mcb_mnist", "K14_kaggle_ablation_seeds_mcb", "--datasets mnist",
     "K14_kaggle_ablation_seeds_mcb__mnist", 2,
     "MNIST long tail, seeds 3-4 only: the K09 ablation arms and MC-B (seeds 0-2 exist in K09 / E32)."),
    ("NB37_ablation_seeds_mcb_fmnist", "K14_kaggle_ablation_seeds_mcb", "--datasets fmnist",
     "K14_kaggle_ablation_seeds_mcb__fmnist", 2,
     "FashionMNIST long tail, seeds 3-4 only: the K09 ablation arms and MC-B (seeds 0-2 exist in K09 / E32)."),
    ("NB38_collapse_fixes_mnist", "K15_kaggle_collapse_fixes", "--datasets mnist",
     "K15_kaggle_collapse_fixes__mnist", 3.5,
     "MNIST long tail, K = 10/50/100: AdamW (matched 1e-3; 1e-2 at K=10), no decay on class rows, lazy decay, against HWA (references reused)."),
    ("NB39_collapse_fixes_fmnist", "K15_kaggle_collapse_fixes", "--datasets fmnist",
     "K15_kaggle_collapse_fixes__fmnist", 3.5,
     "FashionMNIST long tail, K = 10/50/100: AdamW (matched 1e-3; 1e-2 at K=10), no decay on class rows, lazy decay, against HWA (references reused)."),
    # K16 (action plan B1): one notebook per dataset and regime; the long-tail ones decide Gate 1, run them first
    ("NB40_recalib_mnist_lt", "K16_kaggle_recalibration", "--datasets mnist --IF 0.01", "K16_kaggle_recalibration__mnist_lt", 1.5,
     "MNIST long tail: GeFL-F, +LA, minimal variant, our gated method; every trained head evaluated under 8 readouts (ER, Fed3R, FedNCM, CCVR, ...)."),
    ("NB41_recalib_fmnist_lt", "K16_kaggle_recalibration", "--datasets fmnist --IF 0.01", "K16_kaggle_recalibration__fmnist_lt", 1.5,
     "FashionMNIST long tail: the same four methods under every readout."),
    ("NB42_recalib_svhn_lt", "K16_kaggle_recalibration", "--datasets svhn --IF 0.01", "K16_kaggle_recalibration__svhn_lt", 2,
     "SVHN long tail: the same four methods under every readout (the gate picks the class-row generator here)."),
    ("NB43_recalib_cifar10_lt", "K16_kaggle_recalibration", "--datasets cifar10 --IF 0.01", "K16_kaggle_recalibration__cifar10_lt", 4.5,
     "CIFAR-10 long tail: the same four methods under every readout."),
    ("NB44_recalib_cifar100_lt", "K16_kaggle_recalibration", "--datasets cifar100 --IF 0.01", "K16_kaggle_recalibration__cifar100_lt", 3.5,
     "CIFAR-100 long tail (new dataset, 100 classes): the same four methods under every readout."),
    ("NB45_recalib_mnist_fmnist_iid", "K16_kaggle_recalibration", "--datasets mnist fmnist --IF 1.0", "K16_kaggle_recalibration__mnist_fmnist_iid", 3,
     "MNIST + FashionMNIST, the paper's balanced setting: the same four methods under every readout."),
    ("NB46_recalib_svhn_iid", "K16_kaggle_recalibration", "--datasets svhn --IF 1.0", "K16_kaggle_recalibration__svhn_iid", 2,
     "SVHN, the paper's balanced setting: the same four methods under every readout."),
    ("NB47_recalib_cifar10_iid", "K16_kaggle_recalibration", "--datasets cifar10 --IF 1.0", "K16_kaggle_recalibration__cifar10_iid", 5.5,
     "CIFAR-10, the paper's balanced setting: the same four methods under every readout."),
    ("NB48_recalib_cifar100_iid", "K16_kaggle_recalibration", "--datasets cifar100 --IF 1.0", "K16_kaggle_recalibration__cifar100_iid", 4,
     "CIFAR-100, balanced: the same four methods under every readout."),
    # K17 (action plan C4 + C5 + C6): fair re-test on new seeds 3-5, long tail; validation-selected settings
    ("NB50_fair_mnist", "K17_kaggle_fair_retest", "--datasets mnist", "K17_kaggle_fair_retest__mnist", 2,
     "MNIST long tail, new seeds: our method (both synthetic budgets) vs GeFL-F and the published FL losses (FedLC, FedRS, balanced softmax)."),
    ("NB51_fair_fmnist", "K17_kaggle_fair_retest", "--datasets fmnist", "K17_kaggle_fair_retest__fmnist", 2,
     "FashionMNIST long tail, new seeds: the same comparison."),
    ("NB52_fair_svhn", "K17_kaggle_fair_retest", "--datasets svhn", "K17_kaggle_fair_retest__svhn", 2.5,
     "SVHN long tail, new seeds: the same comparison."),
    ("NB53_fair_cifar10", "K17_kaggle_fair_retest", '--datasets cifar10 --exclude "Ours rows Ts=1"', "K17_kaggle_fair_retest__cifar10", 7.5,
     "CIFAR-10 long tail, new seeds: the same comparison."),
    ("NB54_fair_cifar100", "K17_kaggle_fair_retest", "--datasets cifar100", "K17_kaggle_fair_retest__cifar100", 5.5,
     "CIFAR-100 long tail, new seeds: the same comparison."),
    ("NB55_fair_cifar10_rows", "K17_kaggle_fair_retest", '--datasets cifar10 --only "Ours rows Ts=1"', "K17_kaggle_fair_retest__cifar10_rows", 3,
     "CIFAR-10 long tail, new seeds: the class-row generator branch forced, to test the automatic switch near its threshold."),
    ("NB6_svhn_diffusion", "K04_kaggle_ddpm_cifar10_svhn", "--datasets svhn", "K04_kaggle_ddpm_cifar10_svhn__svhn", 7,
     "SVHN with the paper's diffusion feature generator (DDPM-F)."),
    ("NB7_cifar10_diffusion_seed0", "K04_kaggle_ddpm_cifar10_svhn", "--datasets cifar10 --seeds 0", "K04_kaggle_ddpm_cifar10_svhn__cifar10_s0", 6,
     "CIFAR-10 with the diffusion feature generator, seed 0."),
    ("NB8_cifar10_diffusion_seed1", "K04_kaggle_ddpm_cifar10_svhn", "--datasets cifar10 --seeds 1", "K04_kaggle_ddpm_cifar10_svhn__cifar10_s1", 6,
     "CIFAR-10 with the diffusion feature generator, seed 1."),
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(name, script, args, folder, hours, what):
    import build
    src = open(build.build_one(script), encoding="utf-8").read()  # always from current core.py + spec
    exp = script
    cells = [
        md(f"""
# {name}

**What this runs:** {what}
**Time:** about {hours} h. It runs by itself; you do not need to watch it.

## Before you start (once)
1. Right panel → **Settings** → **Accelerator** → choose **GPU T4 x2**. *Not* P100, not "None".
2. Right panel → **Settings** → **Internet** → switch **On**. (Kaggle may ask you to verify your phone number first.)

## How to run
1. Top right → **Save Version** → choose **Save & Run All (Commit)** → **Save**.
2. Close the tab if you like. It keeps running on Kaggle for up to 12 hours.
3. When it is done (the version shows a green tick), open that version → **Output** tab → download two files:
   - **`{folder}.zip`**: the results (small). Send this one back.
   - **`{folder}__models.zip`**: the trained models, kept separate. Keep it (for example as a private Kaggle
     dataset); it is only needed if the models are re-used later, so they do not have to be trained again.

Do **not** edit any cell. If cell 2 prints a red **STOP** message, fix the setting it names and run again.
"""),
        code("""
# Cell 2 - environment check: stops with a clear message if a setting is wrong
import subprocess, sys, urllib.request
import torch
n = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(n)]
print("GPUs found:", n, names)
if n < 2 or not all("T4" in x for x in names):
    raise RuntimeError("STOP: set Settings -> Accelerator -> 'GPU T4 x2', then run again.")
try:
    urllib.request.urlopen("https://huggingface.co", timeout=15)
    print("Internet: on")
except Exception as e:
    raise RuntimeError("STOP: set Settings -> Internet -> On, then run again. (" + str(e) + ")")
print("OK - environment is correct.")
"""),
        md("Cell 3 writes the experiment script to `run.py`. It is long; you do not need to read it."),
        {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}}, "execution_count": None, "outputs": [],
         "source": ("%%writefile run.py\n" + src).splitlines(keepends=True)},
        md("Cell 4 runs the experiment on both GPUs. Progress lines look like `[gpu0] 08:51:13 [2/13] ...`."),
        code(f"""
!python run.py {args} --out_dir /kaggle/working/results
"""),
        md("Cell 5 packs the results into one file for download."),
        code(f"""
import os, shutil
src = "/kaggle/working/results/{exp}"
dst = "/kaggle/working/{folder}"
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
shutil.make_archive(dst, "zip", "/kaggle/working", "{folder}")
print("Results (send this back):", dst + ".zip")
msrc = "/kaggle/working/models/{exp}"
if os.path.isdir(msrc):
    mdst = dst + "__models"
    if os.path.isdir(mdst):
        shutil.rmtree(mdst)
    shutil.copytree(msrc, mdst)
    shutil.make_archive(mdst, "zip", "/kaggle/working", "{folder}__models")
    shutil.rmtree(mdst)
    print("Models (keep; separate from the results):", mdst + ".zip")
print()
print(open(os.path.join(dst, "summary.md")).read() if os.path.exists(os.path.join(dst, "summary.md")) else "no summary.md yet")
"""),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                    "isGpuEnabled": True, "language": "python"}}}


# Combined notebooks: several jobs in one Kaggle session (12 h limit). A deadline (GEFL_DEADLINE) stops new runs after
# `budget_h` hours, so the session always ends with its zips written; skipped runs can be run in a later session.
COMBOS = [  # (name, [(script, args), ...], results folder, budget_h, rough hours, what)
    ("NB63_baselines_and_controls", [("K19_kaggle_more_baselines", ""), ("K20_kaggle_mechanism_controls", ""),
                                     ("K14_kaggle_ablation_seeds_mcb",  # the 4 runs NB60 lost to its time limit
                                      '--datasets mnist --seeds 4 --only "-CSL" "-MC" "-KH" "Ours-A-B Ts=10"')],
     "baselines_and_controls", 0, 9.3,
     "Further published baselines (FSG + LA, FedDF, FedDF + LA) on all five datasets, paired with the fair re-test, then the kernel-herding and moment-correction mechanism controls on FashionMNIST and CIFAR-10, then the 4 ablation runs (MNIST, seed 4) that NB60 lost to its time limit."),
    # X3-X15 of the action plan (§10), packed into three ~11 h sessions; most important step first in each
    ("NB68_gate_and_balanced_fe", [("K21_kaggle_signal_gate", "--datasets cifar10"),
                                   ("K21_kaggle_signal_gate", "--datasets cifar100"),
                                   ("K21_kaggle_signal_gate", "--datasets mnist")],
     "gate_and_balanced_fe", 0, 11.0,
     "X5, X7, X8: the new method suggested by the theory (signal-fraction gate on synthetic data, SGATE / SCSL) and the class-balanced Stage-I loss (BALFE) on CIFAR-10, CIFAR-100 and MNIST, seeds 3-5, paired with the fair re-test (K17)."),
    ("NB69_comparator_and_breadth", [("@docs/research/math_checks/n7b_fe_body_decomp_sched.py",
                                      [f"{ds} {sd} lt,iid" for ds in ["svhn", "cifar10", "cifar100"] for sd in [0, 1, 2]],
                                      {"NO_INIT": "1", "TAG": "x3", "NT": "2"}),
                                     ("K25_instrumented", ""),
                                     ("K21_kaggle_signal_gate", "--datasets fmnist"),
                                     ("K22_kaggle_regime_breadth", "--datasets fmnist"),
                                     ("K23_kaggle_partial_participation", "")],
     "comparator_and_breadth", 0, 11.0,
     "X3: the centralised comparison (federated vs centralised extractor; one-client vs pooled bodies; exact readout) on SVHN, CIFAR-10 and CIFAR-100, seeds 0-2. X6: instrumented runs (what happens at each aggregation), MNIST and CIFAR-10. X7/X8 on FashionMNIST. X10: other imbalance settings and random rare classes on FashionMNIST. X11: partial participation (K = 100, 10% / 20% of clients per round)."),
    ("NB70_more_clients_and_generators", [("K26_kaggle_cifar10_more_clients", ""),
                                          ("K22_kaggle_regime_breadth", "--datasets svhn"),
                                          ("K24_kaggle_dcgan_longtail", "")],
     "more_clients_and_generators", 0, 10.8,
     "X13: CIFAR-10 with 50 clients (decay fix vs HWA vs ours). X10 on SVHN: other imbalance settings and random rare classes. X12: DCGAN-F under the long tail (does collapse need weight decay?)."),
    ("NB71_comparator_rerun", [("K23_kaggle_partial_participation",
                                '--datasets mnist --only "GeFL-F + AdamW + LA [p=0.1]" "GeFL-F + AdamW + LA [p=0.2]" '
                                '"Ours (gated) [p=0.1]" "Ours (gated) [p=0.2]"'),
                               ("K23_kaggle_partial_participation",
                                '--datasets fmnist --seeds 5 --only "GeFL-F + AdamW + LA [p=0.1]" "GeFL-F + AdamW + LA [p=0.2]" '
                                '"Ours (gated) [p=0.1]" "Ours (gated) [p=0.2]"'),
                               ("@docs/research/math_checks/n7b_fe_body_decomp_sched.py",
                                [f"{ds} {sd} lt,iid" for ds in ["svhn", "cifar10", "cifar100"] for sd in [0, 1, 2]],
                                {"NO_INIT": "1", "TAG": "x3", "NT": "2"})],
     "comparator_rerun", 0, 5.0,
     "The 16 partial-participation runs NB69 did not reach (time limit), then the centralised comparison (X3) on SVHN, CIFAR-10 and CIFAR-100, seeds 0-2, rerun after the device fix (in NB69 every X3 run stopped at its first evaluation)."),
    ("NB72_embedding_collapse_lm", [("@experiments/fedembed.py",
                                     [f"--task lm --K {K} --seeds {sd} --rounds 20 "
                                      f"--out /kaggle/working/results/E41_fedembed_lm/s{sd}_K{K} --arms adam_l2 adamw adam "
                                      f"adam_l2_hwa sgd_l2 adam_l2_reset" + (" central_l2" if K == 10 else "")
                                      for sd in [0, 1, 2] for K in [10, 50, 100]],
                                     {"OUTDIR": "E41_fedembed_lm", "NT": "2"})],
     "embedding_collapse_lm", 0, 4.0,
     "Does the rare-row collapse happen in federated embedding models in general? Federated next-word prediction on WikiText-2 (clients = groups of Wikipedia articles), local Adam with coupled L2 vs AdamW / no decay / holder-weighted averaging / SGD / reset optimiser state / centralised, K = 10, 50, 100, seeds 0-2. Pre-registered predictions E1-E4 in experiments/fedembed.py."),
    ("NB62_more_clients_svhn", [("K18_kaggle_more_clients_svhn", "")], "more_clients_svhn", 0, 10.5,
     "SVHN long tail with 50 and 100 clients (3 seeds): GeFL-F, the weight-decay fix, holder-weighted averaging and our method, each under every last-layer readout."),
    ("NB60_combined_A", [("K17_kaggle_fair_retest", '--datasets cifar10 fmnist --exclude "Ours rows Ts=1"'),
                         ("K17_kaggle_fair_retest", '--datasets mnist --only "GeFL-F + AdamW + LA"'),  # rest of MNIST: NB50
                         ("K14_kaggle_ablation_seeds_mcb", "")],
     "combined_A", 10.0, 10.5,
     "Fair re-test (new seeds 3-5) on CIFAR-10, MNIST and FashionMNIST, then the ablation with 2 more seeds (MNIST, FashionMNIST)."),
    ("NB61_combined_B", [("K17_kaggle_fair_retest", "--datasets cifar100 svhn"),
                         ("K17_kaggle_fair_retest", '--datasets cifar10 --only "Ours rows Ts=1"'),
                         ("K16_kaggle_recalibration", "--datasets svhn cifar100 --IF 1.0")],
     "combined_B", 10.0, 11,
     "Fair re-test on CIFAR-100 and SVHN, the CIFAR-10 class-row branch, then the balanced-data readout check on SVHN and CIFAR-100."),
]


def combo_notebook(name, steps, folder, budget_h, hours, what):
    import build
    cells = [md(f"""
# {name}

**What this runs:** {what}
**Time:** about {hours} h (Kaggle's limit is 12 h per session).

## Before you start (once)
1. Right panel → **Settings** → **Accelerator** → **GPU T4 x2**. 2. **Settings** → **Internet** → **On**.

## How to run
**Save Version** → **Save & Run All (Commit)** → **Save**. Close the tab if you like.
When the version shows a green tick: **Output** tab → download
- **`{folder}.zip`** (results, small): send this one back;
- **`{folder}__models.zip`** (trained models): keep it.
"""),
             code(f"""
# Cell 2 - environment check
import os, time, torch, urllib.request
n = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(n)]
print("GPUs found:", n, names)
if n < 2 or not all("T4" in x for x in names):
    raise RuntimeError("STOP: set Settings -> Accelerator -> 'GPU T4 x2', then run again.")
try:
    urllib.request.urlopen("https://huggingface.co", timeout=15); print("Internet: on")
except Exception as e:
    raise RuntimeError("STOP: set Settings -> Internet -> On, then run again. (" + str(e) + ")")
print("OK - environment is correct.")
"""),
             code(f"""
# Cell 3 - pack(): zip results and models now (called after every step, so a timeout never loses finished work)
import os, shutil
def pack():
    for src, dst in [("/kaggle/working/results", "/kaggle/working/{folder}"), ("/kaggle/working/models", "/kaggle/working/{folder}__models")]:
        if os.path.isdir(src):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            shutil.make_archive(dst, "zip", "/kaggle/working", os.path.basename(dst))
            shutil.rmtree(dst)
            print("wrote", dst + ".zip")
""")]
    written = set()
    for step in steps:
        script, args = step[0], step[1]
        if script.startswith("@"):
            # an analysis script from the repository (e.g. docs/research/math_checks/*.py) that imports core.py:
            # write core.py and the script next to each other; outputs go to the results folder via GEFL_OUT
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", script[1:])
            for dep in ["experiments/core.py"] + list(step[3] if len(step) > 3 else []) + [script[1:]]:
                base = os.path.basename(dep)
                if base not in written:
                    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", dep), encoding="utf-8").read()
                    cells.append({"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}},
                                  "execution_count": None, "outputs": [],
                                  "source": (f"%%writefile {base}\n" + src).splitlines(keepends=True)})
                    written.add(base)
            env = dict(step[2] if len(step) > 2 else {})
            outdir = env.pop("OUTDIR", "X3_centralised_comparator")
            arglist = args if isinstance(args, list) else [args]
            # the argument sets run as a queue on the two GPUs (one process per GPU at a time)
            cells.append(code(f"""
import os, subprocess, threading, queue
out = "/kaggle/working/results/{outdir}"
os.makedirs(out, exist_ok=True)
jobs = queue.Queue()
for a in {arglist!r}:
    jobs.put(a)
def worker(gpu):
    while True:
        try:
            a = jobs.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GEFL_DATA_ROOT="/kaggle/working/data", GEFL_OUT=out, **{env!r})
        with open(os.path.join(out, "log_" + a.replace(" ", "_").replace(",", "-").replace("-", "")[:80] + ".txt"), "w") as f:
            rc = subprocess.run("python {os.path.basename(path)} " + a, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        print("GPU", gpu, a, "exit", rc, flush=True)
# data download once, before the two workers start (core datasets for the comparator script)
import sys; sys.path.insert(0, "/kaggle/working"); import core
for d in {sorted({a.split()[0] for a in arglist if a.split()[0] in ("mnist", "fmnist", "svhn", "cifar10", "cifar100")})!r}:
    core.load_dataset(d, "/kaggle/working/data")
ts = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
[t.start() for t in ts]; [t.join() for t in ts]
"""))
            cells.append(code("pack()"))
            continue
        if script not in written:
            src = open(build.build_one(script), encoding="utf-8").read()
            cells.append({"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}}, "execution_count": None,
                          "outputs": [], "source": (f"%%writefile {script}.py\n" + src).splitlines(keepends=True)})
            written.add(script)
        cells.append(code(f"!python {script}.py {args} --out_dir /kaggle/working/results --models_dir /kaggle/working/models"))
        cells.append(code("pack()"))
    cells.append(code(f"""
# Last cell - pack results and models into two separate zips
import os, shutil, glob, subprocess
for src, dst in [("/kaggle/working/results", "/kaggle/working/{folder}"), ("/kaggle/working/models", "/kaggle/working/{folder}__models")]:
    if os.path.isdir(src):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        shutil.make_archive(dst, "zip", "/kaggle/working", os.path.basename(dst))
        shutil.rmtree(dst)
        print("wrote", dst + ".zip")
"""))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                    "isGpuEnabled": True, "language": "python"}}}


def main():
    wanted = sys.argv[1:]
    for combo in [c for c in COMBOS if any(c[0] == w or c[0].startswith(w + "_") for w in wanted)]:
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, combo[0] + ".ipynb")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(combo_notebook(*combo), f, indent=1)
        print("wrote", os.path.normpath(path))
    if wanted and all(any(c[0] == w or c[0].startswith(w + "_") for c in COMBOS) for w in wanted):
        return
    if not wanted:
        for job in JOBS:
            print(f"{job[0]:40s} {job[1]} {job[2]}")
        print("\nName the notebooks to write, e.g.: python make_kaggle_notebooks.py NB38 NB39")
        return
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for job in [j for j in JOBS if any(j[0] == w or j[0].startswith(w + "_") for w in wanted)]:
        nb = notebook(*job)
        path = os.path.join(OUT, job[0] + ".ipynb")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1)
        rows.append(f"| {job[0]}.ipynb | {job[5]} | ~{job[4]} h | `{job[3]}.zip` |")
        print("wrote", os.path.normpath(path))
    with open(os.path.join(OUT, "START_HERE.md"), "w", encoding="utf-8") as f:
        f.write("""# Kaggle notebooks: one per person

Give each person **one** notebook. Each notebook explains itself in its first cell.

## For each person (5 minutes)
1. Kaggle → **Create** → **New Notebook** → **File** → **Import Notebook** → upload your `.ipynb`.
2. Right panel → **Settings**: **Accelerator = GPU T4 x2**, **Internet = On**.
3. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**. Then you can close the tab.
4. When the version shows a green tick: open it → **Output** → download the `.zip` → send it back.

If the run stops with a red **STOP** message, it says which setting to change. Change it and do step 3 again.

| Notebook | What it runs | Time | Send back |
|---|---|---|---|
""" + "\n".join(rows) + """

Most important first: NB1 (CIFAR-10), then NB2, NB3, NB4. NB6–NB8 are long and can run later.
""")
    print("wrote START_HERE.md")


if __name__ == "__main__":
    main()
