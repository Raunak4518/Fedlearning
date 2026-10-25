"""Builds the standalone Kaggle notebooks of the FedECS-LT experiment suite.

Every notebook contains every helper script (fedecs_lt/src/*.py, written to disk by hidden %%writefile cells), its own
run list, and the cells that run, check and pack it. Nothing is imported from GitHub; only pip packages (Flower, Ray,
reportlab) and the datasets are downloaded at run time. Each notebook returns four files:
  NAME__results.zip   every figure (PDF, PNG, CSV), metric file and log
  NAME__report.pdf    the readable report with every table and figure
  NAME__models.zip    trained weights, for re-plotting without re-running
  NAME__summary.zip   small: every metric and log as text (what is sent back when the others are too big)

    python fedecs_lt/build_notebooks.py            # all notebooks
    python fedecs_lt/build_notebooks.py NB00       # those whose name starts with NB00
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
OUT = os.path.join(HERE, "notebooks")
SEEDS = [3, 4, 5]
FLWR = "flwr[simulation]==1.42.0"
REPORTLAB = "reportlab==5.0.1"

# ---------------------------------------------------------------- notebook registry
# name -> dict(title, what (one line for START_HERE), hours (estimate before NB00 measures), runs (list of run dicts),
#              acceptance (bool))


def run(label, ds, seed, IF=0.01, alpha=0.5, K=10, **extra):
    return dict(label=label, ds=ds, seed=seed, IF=IF, alpha=alpha, K=K, **extra)


TRACK1_MAIN = ["FedECS-LT", "GeFL-F", "GeFL-F + AdamW + LA", "Real only", "Real only + LA (balanced softmax)", "FedLC",
               "FedRS", "Gaussian generator + LA", "FedDF", "FedDF + LA", "Local only", "FedProto", "FedTGP", "FedGen",
               "FedGH", "LG-FedAvg", "FedMRL"]

NOTEBOOKS = {
    "NB00_acceptance": dict(
        title="Acceptance of the Flower implementation and timing (MNIST-LT and CIFAR-10-LT)",
        what="Checks the new Flower code against the earlier results (identical client splits, accuracy within seed "
             "noise) and measures run times on Kaggle's GPUs. Must say ACCEPTANCE: PASS before any other notebook "
             "is run.",
        hours=6.0, acceptance=True,
        runs=[run(m, ds, 3) for ds in ("mnist", "cifar10") for m in ("GeFL-F", "FedECS-LT")]
        # timing only (not part of the acceptance check): one run of each other kind, to size the other notebooks
        + [run("FedTGP", "cifar10", 3),
           run("FedECS-LT", "cifar10_t2", 3, K=20, pfrac=0.4), run("FedAvg (LongTailLib)", "cifar10_t2", 3, K=20,
                                                                   pfrac=0.4)]),
}


def main_setting_notebook(ds, part=None, methods=None, seeds=SEEDS):
    name = f"NB10_main_{ds}" + (f"_part{part}" if part else "")
    from_settings = {}
    sys.path.insert(0, SRC)
    import settings
    spec = settings.DATASETS[ds]
    IF = 0.01 if spec["imbalance"] == "IF" else None
    title = (f"Main comparison on {spec['name']}" + ("-LT (imbalance ratio 100)" if IF else " (natural imbalance)")
             + ", 10 clients, Dirichlet 0.5")
    runs = [run(m, ds, s, IF=IF) for s in seeds for m in (methods or TRACK1_MAIN)]
    return name, dict(title=title, what=f"{title}: FedECS-LT against every Track-1 method, seeds 3-5.",
                      hours=None, acceptance=False, runs=runs, **from_settings)


# ---------------------------------------------------------------- cells
def md(text):
    return dict(cell_type="markdown", metadata={}, source=text.splitlines(True))


def code(text, hidden=False):
    meta = {"jupyter": {"source_hidden": True}} if hidden else {}
    return dict(cell_type="code", metadata=meta, execution_count=None, outputs=[], source=text.splitlines(True))


def code_version():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        h.update(open(f, "rb").read())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=HERE, capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return f"{commit} / sources sha256 {h.hexdigest()[:16]}"


def notebook(name, nb):
    work = "/tmp/fedecs/work"
    runs = []
    for i, r in enumerate(nb["runs"]):
        r = dict(r, data_root="/tmp/fedecs/data", out_dir=f"{work}/results", cache_dir="/tmp/fedecs/cache",
                 models_dir=f"{work}/models", tensorboard=True)
        runs.append(r)
    # runs ordered by (dataset, seed); the 4 workers (2 per GPU) take them from this queue in order; runs that share a
    # cached stage wait for whichever worker trains it first (lock files), so no stage is trained twice
    runs.sort(key=lambda r: (r["ds"], r["seed"]))
    n_runs = len(runs)
    cells = [md(
        f"# {nb['title']}\n\n"
        f"**What this notebook does:** {nb['what']}\n\n"
        f"**Runs:** {n_runs}. **Expected time:** {'about %.1f h' % nb['hours'] if nb.get('hours') else 'see START_HERE'} "
        f"(Kaggle's limit is 12 h; no new run starts after 11 h, and finished runs are always saved).\n\n"
        "## Before you start (once)\n"
        "1. Right panel → **Settings** → **Accelerator** → **GPU T4 x2**.\n"
        "2. **Settings** → **Internet** → **On**.\n\n"
        "## How to run\n"
        "**Save Version** → **Save & Run All (Commit)** → **Save**. You can close the tab.\n\n"
        "## What to send back\n"
        "When the version shows a green tick: open it → **Output** tab → download the files:\n"
        f"- `{name}__summary.zip` (small, a few MB: every metric and log; **send this one first**)\n"
        f"- `{name}__report.pdf` (everything in one readable document)\n"
        f"- `{name}__results.zip` (every image, metric, log and prediction; keep it)\n"
        f"- `{name}__models.zip` (trained weights; keep it)\n\n"
        "If the run stops with a red **STOP** message, it tells you which setting to change; change it and run again.")]
    cells.append(code(
        "# Environment check\n"
        "import os, sys, shutil, urllib.request\n"
        "import torch\n"
        "if sys.version_info < (3, 11):\n"
        "    raise RuntimeError(f'STOP: Kaggle Python {sys.version} is older than 3.11, which Flower 1.42 needs. '\n"
        "                       'Settings -> Environment -> Always use latest environment, then run again.')\n"
        "n = torch.cuda.device_count(); names = [torch.cuda.get_device_name(i) for i in range(n)]\n"
        "print('GPUs:', n, names)\n"
        "if n < 2 or not all('T4' in x for x in names):\n"
        "    raise RuntimeError(\"STOP: set Settings -> Accelerator -> 'GPU T4 x2', then run again.\")\n"
        "try:\n"
        "    urllib.request.urlopen('https://huggingface.co', timeout=20); print('Internet: on')\n"
        "except Exception as e:\n"
        "    raise RuntimeError('STOP: set Settings -> Internet -> On, then run again. (' + str(e) + ')')\n"
        "free = shutil.disk_usage('/tmp').free / 2**30\n"
        "print(f'Free disk in /tmp: {free:.0f} GB')\n"
        "print('OK - environment is correct.')"))
    extra = ""
    if any(r["label"].endswith("(LongTailLib)") for r in runs):   # what the authors' LongTailLib code imports
        extra = " h5py tqdm calmsize ftfy regex \"git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6\""
    cells.append(code(f"# Packages (fixed versions)\n!pip install -q \"{FLWR}\" \"{REPORTLAB}\"{extra} 2>&1 | tail -2\n"
                      "import flwr, ray; print('Flower', flwr.__version__, 'Ray', ray.__version__)"))
    cells.append(md("## Helper scripts\nThe cells below write the suite's code to disk (hidden; nothing to change)."))
    cells.append(code("import os\nos.makedirs('/kaggle/working/fedecs_lt', exist_ok=True)"))
    for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        src = open(f).read()
        cells.append(code(f"%%writefile /kaggle/working/fedecs_lt/{os.path.basename(f)}\n{src}", hidden=True))
    cells.append(code(
        "# Run list and code version\n"
        "import json, os, time\n"
        "os.makedirs('/kaggle/working/fedecs_lt', exist_ok=True)\n"
        f"RUNS = json.loads(r\"\"\"{json.dumps(runs, indent=1)}\"\"\")\n"
        "for d in ('/tmp/fedecs/data', '/tmp/fedecs/cache', '/tmp/fedecs/work/results', '/tmp/fedecs/work/models'):\n"
        "    os.makedirs(d, exist_ok=True)\n"
        "json.dump(RUNS, open('/tmp/fedecs/work/runs.json', 'w'))\n"
        f"open('/tmp/fedecs/work/results/code_version.txt', 'w').write({code_version()!r})\n"
        "print(len(RUNS), 'runs')"))
    cells.append(code(
        f"# Run everything: 2 GPUs x {nb.get('per_gpu', 2)} run(s) per GPU; resumable; no new run after 11 h\n"
        "import subprocess, os, sys, time\n"
        "env = dict(os.environ, FEDECS_DEADLINE=str(time.time() + 11 * 3600), FEDECS_PACK_OUT='/kaggle/working',\n"
        f"           FEDECS_NAME={name!r}, FEDECS_CODE_VERSION=open('/tmp/fedecs/work/results/code_version.txt').read(),\n"
        "           PYTHONPATH='/kaggle/working/fedecs_lt')\n"
        "p = subprocess.Popen([sys.executable, '/kaggle/working/fedecs_lt/runner.py', '--runs', '/tmp/fedecs/work/runs.json',\n"
        f"                      '--gpus', '2', '--per_gpu', '{nb.get('per_gpu', 2)}', '--log_dir', '/tmp/fedecs/work/results/logs'], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
        "                     text=True, cwd='/kaggle/working/fedecs_lt')\n"
        "for line in p.stdout:\n"
        "    print(line, end='')\n"
        "p.wait()\n"
        "print('finished with code', p.returncode)"))
    if nb.get("acceptance"):
        cells.append(code(
            "# Acceptance verdict\n"
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '/kaggle/working/fedecs_lt/acceptance.py', '/tmp/fedecs/work/results'],\n"
            "               cwd='/kaggle/working/fedecs_lt')"))
    cells.append(code(
        "# Report and the three output files\n"
        "import subprocess, sys, os\n"
        f"r = subprocess.run([sys.executable, '/kaggle/working/fedecs_lt/pack.py', '/tmp/fedecs/work', {name!r},\n"
        f"                    {nb['title']!r}, '/kaggle/working'], cwd='/kaggle/working/fedecs_lt',\n"
        "                   capture_output=True, text=True)\n"
        "print(r.stdout[-3000:], r.stderr[-3000:])\n"
        "for f in sorted(os.listdir('/kaggle/working')):\n"
        "    if f.startswith('" + name + "'):\n"
        "        print(f, round(os.path.getsize('/kaggle/working/' + f) / 2**20, 1), 'MB')"))
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}}
    return dict(nbformat=4, nbformat_minor=5, metadata=meta, cells=cells)


# CLIP2FL dropped (user, 24 Oct): it uses a pretrained CLIP model (outside knowledge) and is the slowest method; the CLIP
# package is still installed because LongTailLib's main.py imports it for every method
TRACK2_LT = ["FedAvg", "FedProx", "MOON", "FedLC", "CCVR", "CReFF", "FedETF", "Fed-GraB", "FedLoGe", "FedYoYo", "RUCR"]
TRACK2_SEEDS = {0.01: SEEDS, 0.02: [3], 0.1: [3]}   # imbalance ratio 100: seeds 3-5; ratios 50 and 10: seed 3


def track2_notebook(ds_key, IF, seed):
    """Track 2 (FedLTLib protocol): FedECS-LT in Flower and the 11 long-tail baselines in the authors' LongTailLib
    code, on the same split; one seed per notebook."""
    sys.path.insert(0, SRC)
    import settings
    spec = settings.DATASETS[ds_key]
    ratio = int(round(1 / IF))
    name = f"NB20_track2_{spec['raw']}_IF{ratio}_s{seed}"
    title = (f"One shared model (FedLTLib protocol): {spec['name']}-LT, imbalance ratio {ratio}, 20 clients, "
             f"40% per round, seed {seed}")
    runs = [run("FedECS-LT", ds_key, seed, IF=IF, K=20, pfrac=0.4)]
    if ds_key == "cifar10_t2" and IF == 0.01 and seed == 3:   # diagnosis: extractor trained for all 200 rounds
        runs.append(run("FedECS-LT (extractor 200 rounds)", ds_key, seed, IF=IF, K=20, pfrac=0.4,
                        rounds=dict(T_FE=200)))
    runs += [run(f"{m} (LongTailLib)", ds_key, seed, IF=IF, K=20, pfrac=0.4) for m in TRACK2_LT]
    return name, dict(title=title, what=f"{title}: FedECS-LT against {len(TRACK2_LT)} long-tail FL methods run with "
                                        "their authors' code (LongTailLib).", hours=None, acceptance=False, runs=runs)


TIER1 = ("mnist", "fmnist", "svhn", "cifar10", "cifar100")
TIER2 = ("dermamnist", "organamnist", "bloodmnist", "pathmnist", "gtsrb", "emnist", "fer2013")


def _parts(name, nb, parts):
    """One notebook per part (plan_time.py: each part's estimate fits one Kaggle session); a notebook that fits
    keeps its name. Parts of one experiment share `group`, so that their results can be merged."""
    import plan_time
    if len(parts) == 1:
        return {name: dict(nb, hours=plan_time.hours(parts[0]), group=name)}
    out = {}
    for i, runs in enumerate(parts, 1):
        ms = list(dict.fromkeys(r["label"] for r in runs))
        ss = sorted({r["seed"] for r in runs})
        what = (", ".join(ms) if len(ms) <= 6 else f"{len(ms)} methods") + (
            "; seed" + ("s " if len(ss) > 1 else " ") + ", ".join(map(str, ss)))
        out[f"{name}_part{i}of{len(parts)}"] = dict(
            nb, title=f"{nb['title']} (part {i} of {len(parts)})", what=f"{nb['what']} Part {i} of {len(parts)}: {what}.",
            runs=runs, hours=plan_time.hours(runs), group=name)
    return out


def all_notebooks():
    import plan_time
    nbs = {k: dict(v, group=k, hours=plan_time.hours(v["runs"])) for k, v in NOTEBOOKS.items()}
    pipe = [m for m in TRACK1_MAIN if m not in plan_time.HTFL_METHODS]
    htfl = [m for m in TRACK1_MAIN if m in plan_time.HTFL_METHODS]
    for ds in TIER1 + TIER2:
        name, nb = main_setting_notebook(ds)
        nbs.update(_parts(name, nb, plan_time.split_track1(nb["runs"], pipe, htfl, SEEDS)))
    long_runs = []
    for ds_key in ("cifar10_t2", "cifar100_t2"):
        for IF in (0.1, 0.02, 0.01):
            for seed in TRACK2_SEEDS[IF]:
                name, nb = track2_notebook(ds_key, IF, seed)
                parts, long_ = plan_time.split_track2(nb["runs"])
                nbs.update(_parts(name, nb, parts))
                long_runs += long_
    # runs too long to share a GPU within one session: two per notebook, each alone on its GPU
    for i in range(0, len(long_runs), 2):
        runs = long_runs[i:i + 2]
        name = f"NB29_track2_long_runs_{i // 2 + 1}"
        what = "; ".join(f"{r['label']} on {r['ds'].split('_')[0].upper().replace('CIFAR', 'CIFAR-')}-LT, imbalance "
                         f"ratio {round(1 / r['IF'])}, seed {r['seed']}" for r in runs)
        nbs[name] = dict(title="One shared model (FedLTLib protocol): runs too long to share a GPU",
                         what=f"Runs that need a GPU each to finish within one session: {what}.", runs=runs,
                         acceptance=False, per_gpu=1, group=name, hours=plan_time.hours(runs, per_gpu=1))
    return nbs


def priority(name):
    """Run order: 0 NB00 (done), 1 Track 1 on the tier-1 datasets, 2 Track 2 at imbalance ratio 100, 3 Track 1 on the
    tier-2 datasets, 4 Track 2 at ratios 50 and 10 and the long runs."""
    if name.startswith("NB00"):
        return 0
    if name.startswith("NB10"):
        return 1 if any(f"NB10_main_{d}" in name and not name.startswith(f"NB10_main_{d}mnist")
                        for d in TIER1) else 3
    if name.startswith("NB20") and "_IF100_" in name:
        return 2
    return 4


def main():
    os.makedirs(OUT, exist_ok=True)
    nbs = all_notebooks()
    if not sys.argv[1:]:   # a full build: remove notebooks that are no longer in the plan (e.g. now split in parts)
        for f in glob.glob(os.path.join(OUT, "*.ipynb")):
            if os.path.basename(f)[:-6] not in nbs:
                os.remove(f)
    sel = [n for n in nbs if not sys.argv[1:] or any(n.startswith(a) for a in sys.argv[1:])]
    for n in sel:
        path = os.path.join(OUT, n + ".ipynb")
        json.dump(notebook(n, nbs[n]), open(path, "w"), indent=1)
        print("wrote", path, len(nbs[n]["runs"]), "runs")
    rows = ["# Kaggle notebooks of the FedECS-LT suite: one per person and session", "",
            "**NB00_acceptance passed on 24 Oct 2026** (results in `fedecs_lt/results/NB00_acceptance/`); every other "
            "notebook can be run now.", "",
            "## For each person (5 minutes)",
            "1. Kaggle → **Create** → **New Notebook** → **File** → **Import Notebook** → upload your `.ipynb`.",
            "2. Right panel → **Settings**: **Accelerator = GPU T4 x2**, **Internet = On**.",
            "3. Top right: **Save Version** → **Save & Run All (Commit)** → **Save**. You can close the tab.",
            "4. When the version shows a green tick: open it → **Output** → download the four files. Send back "
            "`NAME__summary.zip` and `NAME__report.pdf` (small); keep `NAME__results.zip` and `NAME__models.zip` "
            "(large: predictions and weights) on Google Drive.",
            "", "Expected hours are estimates from NB00's measured run times (fedecs_lt/plan_time.py); a notebook "
            "starts no new run after 11 h and always saves what has finished.", "",
            f"**{len(nbs) - 1} notebooks to run, about "
            f"{sum(nb['hours'] for n, nb in nbs.items() if not n.startswith('NB00')):.0f} session-hours in total.**", "",
            "## Run order",
            "Priority 1 first (the paper's main tables), then 2, 3, 4. One notebook per account at a time; any free "
            "account takes the next notebook of the lowest priority number still open.", "",
            "| Priority | Notebook | What it runs | Runs | Expected hours | Send back |", "|---|---|---|---|---|---|"]
    for n, nb in sorted(nbs.items(), key=lambda kv: (priority(kv[0]), kv[0])):
        rows.append(f"| {priority(n)} | {n}.ipynb | {nb['what']} | {len(nb['runs'])} | {nb['hours']:.1f} | "
                    f"`{n}__summary.zip`, `{n}__report.pdf` |")
    open(os.path.join(OUT, "START_HERE.md"), "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
