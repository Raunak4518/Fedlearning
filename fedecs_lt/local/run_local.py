"""Runs the FedECS-LT notebooks' run lists on a local NVIDIA GPU (Windows, or Linux), one run at a time, with the
same code (fedecs_lt/src), the same client splits and seeds and the same four output files as the Kaggle notebooks.

    python fedecs_lt/local/run_local.py check            # once per machine (30-90 min): what fits on this GPU
    python fedecs_lt/local/run_local.py list             # every job: filed already?  fits here?  done locally?
    python fedecs_lt/local/run_local.py JOB [--hours H]  # runs one job; run the same command again to resume

On Windows the .bat files that setup_windows.bat writes into the repository folder call these commands (README.md in
this folder); on Linux, setup.sh. Jobs are the notebooks of fedecs_lt/notebooks/ (same names, local/jobs.json, written
by build_notebooks.py). Everything is kept under ~/fedecs_local (Windows: %USERPROFILE%/fedecs_local; or
$FEDECS_LOCAL_DIR): data/ (downloaded once), jobs/<JOB>/ (work, resumable), out/<JOB>/ (the four files to send back),
check/ (the memory check).

Small GPUs (4-8 GB): the runs keep no CUDA graphs (FEDECS_CUDA_GRAPHS=0; same computation, less memory held) and run
one at a time. `check` runs one short run (two rounds per stage, all the data) of every kind of run in the jobs and
records whether it fits; `list` and the job command use that record, so a job that cannot fit is not started.
On Windows the one-shared-model jobs (the authors' LongTailLib code, never run on Windows) are left to Kaggle.
"""
import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
FL = os.path.dirname(HERE)
SRC = os.path.join(FL, "src")
BASE = os.path.abspath(os.path.expanduser(os.environ.get("FEDECS_LOCAL_DIR", "~/fedecs_local")))
CHECK_FILE = os.path.join(BASE, "gpu_check.json")
TEST = os.environ.get("FEDECS_LOCAL_TEST") == "1"   # repository test on a CPU machine: 2 rounds, 20% of the data
WINDOWS = os.name == "nt" or os.environ.get("FEDECS_LOCAL_AS_WINDOWS") == "1"   # (the second: unit tests on Linux)
ACTIVATE = r"run the .bat files in the repository folder (setup_windows.bat writes them)" if WINDOWS else \
    "source ~/fedecs_env/bin/activate"
sys.path.insert(0, SRC)

# one short run per kind of run (two rounds per stage, all the data); the hardest dataset of each kind
CHECK = {
    "gen": [("FedECS-LT", "cifar100", 10), ("FedDF", "cifar100", 10)],
    "ce": [("Balanced Softmax", "cifar100", 10)],
    "htfl": [("FedTGP", "cifar100", 10), ("FedGen", "cifar100", 10)],
    "grouped": [("FedALA", "cifar100", 10), ("CCVR", "cifar100", 10)],
    "grouped_K50": [("FedAvg", "cifar10", 50)],
    "gen_K50": [("FedECS-LT", "cifar10", 50)],
    "t2_baseline": [("FedLoGe (LongTailLib)", "cifar100_t2", 20)],
    "t2_ours": [("FedECS-LT", "cifar10_t2", 20)],
    "gen_K100": [("FedECS-LT", "cifar10", 100)],
}
KIND_TEXT = {"gen": "generator methods (FedECS-LT, GeFL-F, FedDF, ablations), 10 clients",
             "ce": "Balanced Softmax, FedLC, FedRS, Local, pooled reference",
             "htfl": "HtFLlib methods (FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL)",
             "grouped": "grouped baselines (FedAvg, FedProx, AvgKD, FedALA, CCVR, FedETF), 10 clients",
             "grouped_K50": "grouped baselines, 50 clients",
             "gen_K50": "generator methods, 50 clients",
             "t2_baseline": "one shared model: LongTailLib methods",
             "t2_ours": "one shared model: FedECS-LT",
             "gen_K100": "generator methods, 100 clients"}


def stop(msg):
    sys.exit(f"\nSTOP: {msg}")


def jobs():
    return json.load(open(os.path.join(HERE, "jobs.json")))


def kind(r):
    from runner import method_spec
    from settings import DATASETS
    spec = method_spec(r["label"])
    p, K = spec.get("pipeline"), r.get("K", 10)
    if p == "longtaillib":
        return "t2_baseline"
    if DATASETS[r["ds"]].get("track") == 2:
        return "t2_ours"
    if p == "htfl":
        return "htfl"
    if p == "grouped":
        return "grouped" if K <= 10 else "grouped_K50"
    if spec.get("gen") is None:
        return "ce"
    return "gen" if K <= 10 else ("gen_K50" if K <= 50 else "gen_K100")


def gpu():
    if TEST:
        return "CPU (test)", 4096
    import torch
    if not torch.cuda.is_available():
        stop(f"PyTorch sees no GPU: update the NVIDIA driver (560 or newer), then {ACTIVATE}.")
    p = torch.cuda.get_device_properties(0)
    return p.name, round(p.total_memory / 2 ** 20)


def ram_gb():
    if os.name == "nt":
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.ullTotalPhys / 2 ** 30
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 0.0


def environment(need_github=False):
    """Stops with the fix when the machine cannot run the suite."""
    if platform.system() not in ("Linux", "Windows"):
        stop(f"{platform.system()} is not supported: Windows or Linux with an NVIDIA GPU.")
    if WINDOWS and sys.version_info[:2] != (3, 12):
        stop(f"Python {sys.version.split()[0]}: on Windows Flower's simulation needs Python 3.12 ({ACTIVATE}).")
    if sys.version_info[:2] < (3, 11):
        stop(f"Python {sys.version.split()[0]} is too old: {ACTIVATE}.")
    try:
        import flwr
        assert flwr.__version__ == "1.42.0", flwr.__version__
        import ray  # noqa: F401
    except Exception as e:
        stop(f"Flower 1.42.0 / Ray not installed in this Python ({e}): run the setup again, then {ACTIVATE}.")
    name, mb = gpu()
    os.makedirs(BASE, exist_ok=True)
    free = shutil.disk_usage(BASE).free / 2 ** 30
    print(f"GPU {name} ({mb / 1024:.1f} GB) | RAM {ram_gb():.0f} GB | free disk at {BASE}: {free:.0f} GB")
    if free < 20 and not TEST:
        stop(f"only {free:.0f} GB free at {BASE}; 20 GB are needed (datasets, caches, outputs).")
    for url in ["https://huggingface.co"] + (["https://github.com"] if need_github else []):
        try:
            urllib.request.urlopen(url, timeout=20)
        except urllib.error.HTTPError:
            pass   # the server answered: the connection works
        except Exception as e:
            stop(f"no connection to {url} ({e}); the datasets{' and LongTailLib' if 'github' in url else ''} are "
                 "downloaded from there.")
    return name, mb


def run_env(name_job, out_dir, deadline=None):
    _, mb = gpu()
    env = dict(os.environ, PYTHONPATH=SRC, FEDECS_PACK_OUT=out_dir, FEDECS_NAME=name_job,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               PYTHONUTF8="1")   # Windows: every text file in UTF-8 (captions, reports), as on Linux
    if os.name == "nt":   # Ray's session folders under a short path (Windows' 260-character path limit)
        env["RAY_TMPDIR"] = os.environ.get("FEDECS_RAY_TMPDIR", "C:\\fedecs_ray")
        try:
            os.makedirs(env["RAY_TMPDIR"], exist_ok=True)
        except OSError:   # C:\ not writable for this user
            env["RAY_TMPDIR"] = os.path.join(os.environ.get("LOCALAPPDATA", BASE), "fr")
            os.makedirs(env["RAY_TMPDIR"], exist_ok=True)
    if mb < 12 * 1024:
        env["FEDECS_CUDA_GRAPHS"] = "0"
    if deadline:
        env["FEDECS_DEADLINE"] = str(deadline)
    env["FEDECS_CODE_VERSION"] = code_version()
    return env


def code_version():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(SRC, "*.py"))):
        h.update(open(f, "rb").read())
    try:
        commit = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=FL, capture_output=True,
                                text=True).stdout.strip()
    except OSError:   # no git (a downloaded ZIP of the repository): the commit the job list was built from
        commit = ""
    if not commit and os.path.exists(os.path.join(HERE, "VERSION")):
        commit = open(os.path.join(HERE, "VERSION")).read().strip() + " (ZIP)"
    commit = commit or "unknown"
    name, mb = gpu()
    return f"{commit} / sources sha256 {h.hexdigest()[:16]} / local: {name} {mb / 1024:.0f} GB" + (
        ", CUDA graphs off" if mb < 12 * 1024 else "")


def place(runs, work):
    """The notebook's runs with this machine's paths (the datasets are shared by every job)."""
    out = []
    for r in runs:
        r = dict(r, data_root=os.path.join(BASE, "data"), out_dir=os.path.join(work, "results"),
                 cache_dir=os.path.join(work, "cache"), models_dir=os.path.join(work, "models"))
        if TEST:
            r.update(rounds=dict(T_FE=2, T_KA=2, T_TN=2, T=2), frac_scale=0.2)
        out.append(r)
    for d in (r["data_root"], r["out_dir"], r["cache_dir"], r["models_dir"]):
        os.makedirs(d, exist_ok=True)
    return out


def runner(runs, work, name_job, out_dir, deadline=None):
    rp = os.path.join(work, "runs.json")
    json.dump(runs, open(rp, "w"), indent=1)
    open(os.path.join(work, "results", "code_version.txt"), "w").write(code_version())
    env = run_env(name_job, out_dir, deadline)
    cmd = [sys.executable, os.path.join(SRC, "runner.py"), "--runs", rp, "--gpus", "0" if TEST else "1", "--per_gpu", "1",
           "--log_dir", os.path.join(work, "results", "logs")]
    return subprocess.run(cmd, env=env, cwd=SRC).returncode


def load_check():
    return json.load(open(CHECK_FILE)) if os.path.exists(CHECK_FILE) else None


def fits(job_runs, chk):
    """(True / False / None, the kinds that do not fit or were not checked)."""
    kinds = sorted({kind(r) for r in job_runs})
    if WINDOWS and any(k.startswith("t2") for k in kinds):
        return False, ["one shared model: Kaggle only on Windows"]
    if chk is None:
        return None, kinds
    bad = [k for k in kinds if chk["kinds"].get(k, {}).get("ok") is False]
    unknown = [k for k in kinds if k not in chk["kinds"]]
    if bad:
        return False, bad
    return (None, unknown) if unknown else (True, [])


# ---------------------------------------------------------------- commands
def cmd_check(args):
    t2 = [k for k in CHECK if k.startswith("t2") and (not args.kinds or k in args.kinds.split(","))]
    name, mb = environment(need_github=bool(t2) and not WINDOWS)
    chk = load_check() or {}
    if chk.get("gpu") != name:
        chk = dict(gpu=name, vram_mb=mb, kinds={})
    every = [r for j in jobs().values() for r in j["runs"]]
    for k, entries in CHECK.items():
        if WINDOWS and k.startswith("t2"):
            continue   # one shared model: Kaggle only on Windows
        if args.kinds and k not in args.kinds.split(","):
            continue
        if k in chk["kinds"] and not args.again:
            print(f"{k}: already checked ({'fits' if chk['kinds'][k]['ok'] else 'does not fit'})")
            continue
        print(f"\n== {k}: {KIND_TEXT[k]}", flush=True)
        work = os.path.join(BASE, "check", k)
        shutil.rmtree(work, ignore_errors=True)
        runs = []
        for label, ds, K in entries:   # the run as a job has it (every option), shortened to two rounds per stage
            r = next(dict(x) for x in every if x["label"] == label and x["ds"] == ds and x.get("K", 10) == K)
            r["rounds"] = dict(T_FE=2, T_KA=2, T_TN=2, T=2)
            runs.append(r)
        runs = place(runs, work)
        t0 = time.time()
        runner(runs, work, f"check_{k}", os.path.join(work, "out"))
        done = [json.loads(x) for x in open(os.path.join(work, "results", "runs.jsonl"))] \
            if os.path.exists(os.path.join(work, "results", "runs.jsonl")) else []
        fail = [json.loads(x) for x in open(os.path.join(work, "results", "failures.jsonl"))] \
            if os.path.exists(os.path.join(work, "results", "failures.jsonl")) else []
        oom = any("out of memory" in f["error"].lower() or "outofmemory" in f["error"].lower() for f in fail)
        ok = len(done) == len(runs)
        peak = max([r.get("gpu_peak_mb") or 0 for r in done] or [0])
        chk["kinds"][k] = dict(ok=ok, peak_mb=peak, minutes=round((time.time() - t0) / 60, 1),
                               error=None if ok else ("out of GPU memory" if oom else
                                                      (fail[-1]["error"][:300] if fail else "no result")))
        json.dump(chk, open(CHECK_FILE, "w"), indent=1)
        print(f"{k}: {'FITS' if ok else 'DOES NOT FIT / FAILED: ' + chk['kinds'][k]['error']}"
              f" (peak {peak / 1024:.1f} GB of {mb / 1024:.1f} GB; {chk['kinds'][k]['minutes']} min)", flush=True)
        if not ok and not oom:
            print("   not a memory problem: send ~/fedecs_local/check/" + k + "/results/ (failures.jsonl, logs/)")
    print(f"\nwritten: {CHECK_FILE}\nnext: python fedecs_lt/local/run_local.py list")


def status(n):
    filed = os.path.join(FL, "results", n, "runs.jsonl")
    if os.path.exists(filed):
        return "filed"
    p = os.path.join(BASE, "jobs", n, "results", "runs.jsonl")
    return f"local {sum(1 for _ in open(p))}" if os.path.exists(p) else ""


def cmd_list(args):
    chk = load_check()
    J = jobs()
    if chk is None:
        print("(no memory check yet on this machine: run `python fedecs_lt/local/run_local.py check` first)\n")
    else:
        print(f"GPU {chk['gpu']} ({chk['vram_mb'] / 1024:.1f} GB): " + ", ".join(
            f"{k} {'ok' if v['ok'] else 'NO'}" for k, v in chk["kinds"].items()) + "\n")
    print(f"{'job':58s} {'runs':>4s} {'Kaggle h':>8s}  {'status':10s} fits here")
    for n, j in J.items():
        st = status(n)
        if st == "filed" and not args.all:
            continue
        f, which = fits(j["runs"], chk)
        ftxt = "yes" if f else ("NO: " + ", ".join(which) if f is False else
                                "not checked" + (": " + ", ".join(which) if which and chk else ""))
        print(f"{n:58s} {len(j['runs']):4d} {j['kaggle_hours']:8.1f}  {st:10s} {ftxt}")
    print("\n'filed': results already in fedecs_lt/results (hidden unless --all). Kaggle h: hours on Kaggle's two T4 GPUs "
          "with four runs at a time; one local GPU running one run at a time needs about 2-4 times as long.\n"
          "Before you start a job, tell the others which one you took (and never start a notebook that is running "
          "on Kaggle).")


def cmd_job(args):
    J = jobs()
    n = args.cmd
    if n not in J:
        stop(f"no job {n!r}; `python fedecs_lt/local/run_local.py list` shows them.")
    j = J[n]
    if status(n) == "filed" and not args.force:
        stop(f"{n} has already been run and filed (fedecs_lt/results/{n}/); --force runs it again (a repeat).")
    environment(need_github=any(kind(r).startswith("t2") for r in j["runs"]))
    f, which = fits(j["runs"], load_check())
    if f is False and not args.force:
        stop(f"{n} contains runs that do not fit on this GPU ({', '.join(which)}; `check`). Give it to a machine with "
             "a larger GPU or to Kaggle.")
    if f is None:
        print(f"WARNING: not checked on this GPU: {', '.join(which) or 'everything'} "
              "(`python fedecs_lt/local/run_local.py check`). Starting anyway.")
    if any(r.get("K", 10) >= 50 for r in j["runs"]) and ram_gb() < 24:
        print(f"WARNING: {ram_gb():.0f} GB RAM; runs with 50 or more clients used about 15 GB on the CPU test. "
              "Close other programs; on WSL2 give it more memory (README.md, step 1).")
    work = os.path.join(BASE, "jobs", n)
    out = os.path.join(BASE, "out", n)
    os.makedirs(out, exist_ok=True)
    runs = place(j["runs"], work)
    deadline = time.time() + args.hours * 3600 if args.hours else None
    print(f"\n{n}: {len(runs)} runs, one at a time; Ctrl-C stops it (the run in progress is lost, finished ones are "
          "kept); the same command resumes.\n", flush=True)
    runner(runs, work, n, out, deadline)
    r = subprocess.run([sys.executable, os.path.join(SRC, "pack.py"), work, n, j["title"], out], cwd=SRC,
                       env=run_env(n, out), capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    done = os.path.join(work, "results", "runs.jsonl")
    nd = sum(1 for _ in open(done)) if os.path.exists(done) else 0
    print(f"\n{n}: {nd} of {len(runs)} runs finished. Send back {out}/{n}__summary.zip and {n}__report.pdf"
          + ("" if nd == len(runs) else "  (incomplete: run the same command again to continue)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", help="check | list | JOB")
    ap.add_argument("--hours", type=float, default=None, help="start no new run after this many hours")
    ap.add_argument("--force", action="store_true", help="run a filed job again, or one that failed the check")
    ap.add_argument("--again", action="store_true", help="check: measure every kind again")
    ap.add_argument("--kinds", default=None, help="check: only these kinds (comma-separated, e.g. gen,htfl)")
    ap.add_argument("--all", action="store_true", help="list: also the jobs already filed")
    args = ap.parse_args()
    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "list":
        cmd_list(args)
    else:
        cmd_job(args)


if __name__ == "__main__":
    main()
