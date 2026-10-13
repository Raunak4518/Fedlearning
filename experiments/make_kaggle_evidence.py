"""Build the four Kaggle notebooks for the component evidence (run all four in parallel).

    python make_kaggle_evidence.py [out_dir]

NB32  K09 ablation, MNIST long tail        (each component removed in turn, 3 seeds)
NB33  K09 ablation, FashionMNIST long tail
NB34  D01 mechanism evidence, MNIST (GPU 0) + FashionMNIST (GPU 1), 3 seeds
NB35  D01 class separability, SVHN (GPU 0) + CIFAR-10 (GPU 1), 3 seeds
"""
import json
import os
import sys

from make_kaggle_notebooks import HERE, md, code, notebook

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "kaggle_notebooks")

HEADER = """
# {name}

**What this runs:** {what}
**Time:** about {hours} h. It runs by itself; you do not need to watch it.

## Before you start (once)
1. Right panel → **Settings** → **Accelerator** → choose **GPU T4 x2**. *Not* P100, not "None".
2. Right panel → **Settings** → **Internet** → switch **On**.

## How to run
1. Top right → **Save Version** → choose **Save & Run All (Commit)** → **Save**.
2. Close the tab if you like. It keeps running on Kaggle for up to 12 hours.
3. When it is done (green tick), open that version → **Output** tab → download **`{folder}.zip`** and send it back.

Do **not** edit any cell. If cell 2 prints a red **STOP** message, fix the setting it names and run again.
"""

ENV_CHECK = """
# Cell 2 - environment check: stops with a clear message if a setting is wrong
import urllib.request
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
"""


def hidden_file(name, src):
    return {"cell_type": "code", "metadata": {"jupyter": {"source_hidden": True}}, "execution_count": None,
            "outputs": [], "source": (f"%%writefile {name}\n" + src).splitlines(keepends=True)}


def d01_notebook(name, datasets, extra, folder, hours, what):
    core = open(os.path.join(HERE, "core.py"), encoding="utf-8").read()
    d01 = open(os.path.join(HERE, "D01_component_evidence.py"), encoding="utf-8").read()
    run = f"""
# Cell 5 - one dataset per GPU, in parallel; progress is printed every 2 minutes
import os, subprocess, sys, time
import core
for ds in {datasets!r}:
    core.load_dataset(ds, "./data")          # download once, before the two workers start
out = "/kaggle/working/results/D01_component_evidence"
os.makedirs(out, exist_ok=True)
procs = []
for i, ds in enumerate({datasets!r}):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i), PYTHONUNBUFFERED="1")
    log = open(f"/kaggle/working/d01_{{ds}}.log", "w")
    procs.append((ds, subprocess.Popen([sys.executable, "D01_component_evidence.py", "--datasets", ds,
                                        "--data_root", "./data", "--out_dir", out] + {extra!r},
                                       env=env, stdout=log, stderr=subprocess.STDOUT)))
while any(p.poll() is None for _, p in procs):
    time.sleep(120)
    for ds, _ in procs:
        lines = open(f"/kaggle/working/d01_{{ds}}.log", encoding="utf-8", errors="replace").read().splitlines()
        print(f"[{{ds}}]", lines[-1] if lines else "(starting)", flush=True)
for ds, p in procs:
    print(ds, "exit code", p.returncode)
    if p.returncode != 0:
        print(open(f"/kaggle/working/d01_{{ds}}.log", encoding="utf-8", errors="replace").read()[-3000:])
"""
    pack = f"""
import os, shutil, glob
src = "/kaggle/working/results/D01_component_evidence"
for f in glob.glob("/kaggle/working/d01_*.log"):
    shutil.copy(f, src)
dst = "/kaggle/working/{folder}"
if os.path.isdir(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
shutil.make_archive(dst, "zip", "/kaggle/working", "{folder}")
print("Download this file from the Output tab:", dst + ".zip")
print(sorted(os.listdir(dst)))
"""
    cells = [md(HEADER.format(name=name, what=what, hours=hours, folder=folder)), code(ENV_CHECK),
             md("Cells 3 and 4 write the code files. They are long; you do not need to read them."),
             hidden_file("core.py", core), hidden_file("D01_component_evidence.py", d01),
             md("Cell 5 runs the experiment, one dataset on each GPU."), code(run),
             md("Cell 6 packs the results into one file for download."), code(pack)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                    "isGpuEnabled": True, "language": "python"}}}


def main():
    os.makedirs(OUT, exist_ok=True)
    nbs = [
        ("NB32_ablation_mnist", notebook("NB32_ablation_mnist", "K09_kaggle_ablation_final", "--datasets mnist",
                                         "K09_kaggle_ablation_final__mnist", 2,
                                         "MNIST long tail: Ours-A with each component removed in turn (LA, CSL, MC, KH, PC), 3 seeds."),
         "K09_kaggle_ablation_final__mnist.zip", 2),
        ("NB33_ablation_fmnist", notebook("NB33_ablation_fmnist", "K09_kaggle_ablation_final", "--datasets fmnist",
                                          "K09_kaggle_ablation_final__fmnist", 2,
                                          "FashionMNIST long tail: Ours-A with each component removed in turn, 3 seeds."),
         "K09_kaggle_ablation_final__fmnist.zip", 2),
        ("NB34_evidence_mnist_fmnist", d01_notebook("NB34_evidence_mnist_fmnist", ["mnist", "fmnist"], [],
                                                    "D01_evidence_mnist_fmnist", 2.5,
                                                    "What each component does, measured on real features: collapse of class rows, "
                                                    "fidelity, spread, mean error, MMD, train-on-fake/test-on-real, LA and CSL checks. "
                                                    "MNIST and FashionMNIST long tail, 3 seeds."),
         "D01_evidence_mnist_fmnist.zip", 2.5),
        ("NB35_separability_svhn_cifar10", d01_notebook("NB35_separability_svhn_cifar10", ["svhn", "cifar10"], ["--sep_only"],
                                                        "D01_separability_svhn_cifar10", 1,
                                                        "How well the class MEAN identifies the class on SVHN and CIFAR-10 "
                                                        "(Fisher ratio, nearest-mean accuracy, 2-D view), 3 seeds."),
         "D01_separability_svhn_cifar10.zip", 1),
    ]
    rows = []
    for name, nb, zipname, hours in nbs:
        with open(os.path.join(OUT, name + ".ipynb"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1)
        rows.append(f"| {name}.ipynb | ~{hours} h | `{zipname}` |")
        print("wrote", name)
    with open(os.path.join(OUT, "START_HERE_evidence.md"), "w", encoding="utf-8") as f:
        f.write("# Component-evidence notebooks (run all four at the same time)\n\n"
                "Each: Kaggle → Create → New Notebook → File → Import Notebook → upload. Settings: GPU T4 x2, Internet On. "
                "Save Version → Save & Run All (Commit). When done, download the zip from Output.\n\n"
                "| Notebook | Time | Send back |\n|---|---|---|\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
