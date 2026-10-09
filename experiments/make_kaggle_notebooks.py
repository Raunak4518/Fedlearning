"""Build one ready-to-run Kaggle notebook per job from the standalone K0*.py scripts.

    python make_kaggle_notebooks.py [out_dir]

Each notebook: instructions -> environment check (2 GPUs, internet) -> writes
the script -> runs it on both GPUs -> zips the results under the right name.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kaggle_notebooks")

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
    src = open(os.path.join(HERE, script + ".py"), encoding="utf-8").read()
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
3. When it is done (the version shows a green tick), open that version → **Output** tab →
   download **`{folder}.zip`** and send it back. That one file is all that is needed.

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
print("Download this file from the Output tab:", dst + ".zip")
print()
print(open(os.path.join(dst, "summary.md")).read() if os.path.exists(os.path.join(dst, "summary.md")) else "no summary.md yet")
"""),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                    "isGpuEnabled": True, "language": "python"}}}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for job in JOBS:
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
