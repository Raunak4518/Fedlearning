"""Runs experiments: one Flower simulation per run, two runs per GPU in parallel worker processes, resumable
(runs already in runs.jsonl are skipped), stops starting new runs at the deadline.

    python runner.py --exp NAME --runs runs.json --out /kaggle/working/results --gpus 2 --per_gpu 2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# Methods of the GeFL-F pipeline family (server.run_pipeline). Baselines with their own protocol are in baselines/.
PIPELINE = {
    "FedECS-LT": dict(pipeline="gefl", stats=True, secure=True, readout=True,
                      gen=dict(type="gate", opt="adam", wd=1e-3, beta=0.999),
                      head=dict(ts=10, tr=5, loss="la", tau=1.0, mc=True, csl=0.5)),
    "GeFL-F": dict(pipeline="gefl", gen=dict(type="cvae", opt="adam", wd=1e-3, agg="flat"),
                   head=dict(ts=1, tr=5, loss="ce")),
    "GeFL-F + LA": dict(pipeline="gefl", gen=dict(type="cvae", opt="adamw", wd=1e-3, agg="flat"),
                                head=dict(ts=1, tr=5, loss="la", tau=1.0)),
    "GeFL-F w/o generator": dict(pipeline="gefl", gen=None, head=dict(ts=0, tr=5, loss="ce")),
    "Balanced Softmax": dict(pipeline="gefl", gen=None, head=dict(ts=0, tr=5, loss="la", tau=1.0)),
    "FedLC": dict(pipeline="gefl", gen=None, head=dict(ts=0, tr=5, loss="lc", tau=1.0)),
    "FedRS": dict(pipeline="gefl", gen=None, head=dict(ts=0, tr=5, loss="rs")),
    "GeFL-F w/ Gaussian generator + LA": dict(pipeline="gefl", stats=False, gen=dict(type="gauss"),
                                    head=dict(ts=1, tr=5, loss="la", tau=1.0)),
    "FedDF": dict(pipeline="gefl", gen=dict(type="cvae", opt="adam", wd=1e-3, agg="flat"),
                  head=dict(ts=1, tr=5, loss="ce", sed=20, sed_beta=1.0, sed_lr=0.05, sed_bs=128)),
    "FedDF + LA": dict(pipeline="gefl", gen=dict(type="cvae", opt="adam", wd=1e-3, agg="flat"),
                       head=dict(ts=1, tr=5, loss="la", tau=1.0, sed=20, sed_beta=1.0, sed_lr=0.05, sed_bs=128)),
    "Local": dict(pipeline="local", head=dict(ts=0, tr=5, loss="ce")),
}
# Track 2 diagnosis (user, 24 Oct): FedECS-LT with its extractor trained for all 200 rounds of the baselines (the run
# sets rounds T_FE = 200; generator and heads as FedECS-LT, so 300 rounds in total, reported as such)
PIPELINE["FedECS-LT (extractor 200 rounds)"] = dict(PIPELINE["FedECS-LT"])
# Component ablation (paper, Table "Components of FedECS-LT"; user, 27 Oct 2026): one component changed at a time,
# everything else as FedECS-LT. The two generator variants replace the gate: class rows averaged by CCWA, or class
# rows averaged flat (GeFL-F's generator), both with the exact statistics, moment correction and the same heads.
_HEAD = PIPELINE["FedECS-LT"]["head"]
COMPONENTS = {
    "FedECS-LT w/ class rows (CCWA)": dict(gen=dict(type="cvae", opt="adam", wd=1e-3, agg="ccwa", beta=0.999)),
    "FedECS-LT w/ class rows (flat averaging)": dict(gen=dict(type="cvae", opt="adam", wd=1e-3, agg="flat")),
    "FedECS-LT w/o moment correction": dict(head=dict(_HEAD, mc=False)),
    "FedECS-LT w/o consensus soft labels": dict(head=dict(_HEAD, csl=0.0)),
    "FedECS-LT w/o logit adjustment": dict(head=dict(_HEAD, loss="ce")),
    "FedECS-LT w/ one synthetic epoch": dict(head=dict(_HEAD, ts=1)),
}
for _name, _change in COMPONENTS.items():
    PIPELINE[_name] = dict(PIPELINE["FedECS-LT"], **_change)


def method_spec(label):
    if label in PIPELINE:
        return PIPELINE[label]
    from baselines import BASELINES
    return BASELINES[label]


def run_id(r):
    extra = ("|tp" if r.get("tail_perm") else "") + (f"|p{r['pfrac']}" if r.get("pfrac", 1.0) < 1.0 else "")
    return f"{r['label']}|{r['ds']}|IF{r['IF']}|a{r['alpha']}|K{r['K']}|s{r['seed']}" + extra


def run_one(R):
    """One run = one Flower simulation (Ray backend, one persistent client process on this process's GPU); Track-2
    baselines run the authors' LongTailLib code instead (track2.py)."""
    if R["method"].get("pipeline") == "longtaillib":
        sys.path.insert(0, HERE)
        import track2
        return track2.run_for_suite(R)
    import torch
    from flwr.simulation import run_simulation
    sys.path.insert(0, HERE)
    os.environ["PYTHONPATH"] = HERE + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    import client
    import server
    server.RUN.clear()
    server.RUN.update(R)
    gpu = 1.0 if torch.cuda.is_available() else 0.0
    backend = {"client_resources": {"num_cpus": 1, "num_gpus": gpu},
               "init_args": {"num_cpus": 1, "num_gpus": int(gpu), "include_dashboard": False,
                             "object_store_memory": int(R.get("ray_object_store_gb", 2) * 2 ** 30),
                             "logging_level": "ERROR", "log_to_driver": True}}
    run_simulation(server_app=server.app, client_app=client.app, num_supernodes=R["K"], backend_config=backend)
    if "result" not in server.RUN:
        raise RuntimeError(f"run {R['run_id']} produced no result")
    return server.RUN["result"]


def done_ids(out_dir):
    p = os.path.join(out_dir, "runs.jsonl")
    if not os.path.exists(p):
        return set()
    out = set()
    for line in open(p):
        try:
            out.add(json.loads(line)["run_id"])
        except Exception:
            pass
    return out


def worker_main(args):
    """One worker: takes the next run of the list that no other worker has claimed (a shared queue, so long runs do
    not leave other workers idle) and runs it; runs are ordered by dataset and seed, so runs sharing cached stages
    are close together."""
    runs = json.load(open(args.runs))
    claims = os.path.join(os.path.dirname(os.path.abspath(args.runs)), "claims")
    os.makedirs(claims, exist_ok=True)
    deadline = float(os.environ.get("FEDECS_DEADLINE", "inf"))
    # pass 0: the queue; pass 1: every run that failed (out of memory while two heavy runs overlapped, a download
    # error) is tried once more after the queue is empty
    for attempt in (0, 1):
        for r in runs:
            if time.time() > deadline:
                print("deadline reached: no new run started", flush=True)
                return
            R = dict(r)
            R.setdefault("method", method_spec(R["label"]))
            R["run_id"] = run_id(R)
            if R["run_id"] in done_ids(R["out_dir"]):
                if attempt == 0:
                    print("already done:", R["run_id"], flush=True)
                continue
            if attempt == 1 and R["run_id"] not in failed_ids(R["out_dir"]):
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in R["run_id"])
            try:   # atomic claim: exactly one worker gets each run (and each retry)
                os.close(os.open(os.path.join(claims, ("retry_" if attempt else "") + safe),
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            execute(R, attempt)


def failed_ids(out_dir):
    path = os.path.join(out_dir, "failures.jsonl")
    if not os.path.exists(path):
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["run_id"])
        except Exception:
            pass
    return out


def needs_whole_gpu(R):
    """FedECS-LT with one shared model keeps 32,768-dimensional features of every image on the GPU: about 13 GB of a
    T4's 14.6 GB (Kaggle, 27 Oct 2026), so it ran out of memory next to any other run."""
    from settings import DATASETS
    return R["label"].startswith("FedECS-LT") and DATASETS.get(R["ds"], {}).get("track") == 2


class GpuShare:
    """The GPU of this worker (CUDA_VISIBLE_DEVICES): runs that need all of it take it exclusively, the others share
    it. A run waiting for exclusive use keeps new shared runs from starting there, so it is not starved. Only the
    scheduling changes, never a run's computation."""

    def __init__(self, work_dir, alone):
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
        self.path = os.path.join(work_dir, f"gpu_{dev}.lock")
        self.want = f"{self.path}.want.{os.getpid()}"   # one per waiting run
        self.alone = alone

    def __enter__(self):
        import fcntl
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.f = open(self.path, "a")
        if self.alone:
            open(self.want, "a").close()
            print("waiting for the whole GPU", flush=True)
            fcntl.flock(self.f, fcntl.LOCK_EX)
        else:
            import glob
            while glob.glob(self.path + ".want.*"):
                time.sleep(5)
            fcntl.flock(self.f, fcntl.LOCK_SH)
        return self

    def __exit__(self, *exc):
        import fcntl
        if self.alone:
            try:
                os.remove(self.want)
            except FileNotFoundError:
                pass
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def execute(R, attempt=0):
    t0 = time.time()
    print(("retry " if attempt else "") + "start", R["run_id"], flush=True)
    try:
        with GpuShare(os.path.dirname(os.path.abspath(R["out_dir"])), needs_whole_gpu(R)):
            res = run_one(R)
        print(f"done {R['run_id']}  bal={res.get('final_bal', float('nan')):.4f}  "
              f"er_bal={(res.get('er_bal') or float('nan')):.4f}  {time.time() - t0:.0f}s", flush=True)
    except Exception as e:  # record the failure, release this process's stage locks, continue with the next run
        import glob
        import traceback
        traceback.print_exc()
        os.makedirs(R["out_dir"], exist_ok=True)
        with open(os.path.join(R["out_dir"], "failures.jsonl"), "a") as f:
            f.write(json.dumps(dict(run_id=R["run_id"], attempt=attempt + 1, error=repr(e)[:4000],
                                    time_s=time.time() - t0)) + "\n")
        for lk in glob.glob(os.path.join(R["cache_dir"], "*", "*.lock")):
            try:
                if open(lk).read().strip() == str(os.getpid()):
                    os.remove(lk)
            except OSError:
                pass
    if os.environ.get("FEDECS_PACK_OUT"):   # protect finished work against a timeout
        try:
            import pack
            pack.quick(os.path.dirname(R["out_dir"]), os.environ["FEDECS_NAME"], os.environ["FEDECS_PACK_OUT"])
        except Exception as e:
            print("pack failed:", repr(e), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="JSON list of runs")
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--per_gpu", type=int, default=2)
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--log_dir", default=None)
    args = ap.parse_args()
    if args.worker is not None:
        return worker_main(args)
    n = max(1, args.gpus * args.per_gpu) if args.gpus else max(1, args.per_gpu)
    # a new start of the queue: claims of an earlier, interrupted start are void (finished runs are skipped anyway)
    shutil.rmtree(os.path.join(os.path.dirname(os.path.abspath(args.runs)), "claims"), ignore_errors=True)
    procs, threads = [], []
    for w in range(n):
        env = dict(os.environ)
        if args.gpus:
            env["CUDA_VISIBLE_DEVICES"] = str(w % args.gpus)
        # less fragmentation of GPU memory when two runs share a GPU (allocation only; the computation is unchanged)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = [sys.executable, os.path.abspath(__file__), "--runs", args.runs, "--worker", str(w), "--n_workers", str(n)]
        log_dir = args.log_dir or os.path.dirname(args.runs)
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, f"worker{w}.log"), "a")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)

        def stream(p=p, w=w, log=log):
            for line in p.stdout:
                log.write(line)
                log.flush()
                if line.startswith(("start", "retry", "done", "already", "deadline", "Traceback", "RuntimeError")):
                    print(f"[w{w}] {line}", end="", flush=True)
        t = threading.Thread(target=stream, daemon=True)
        t.start()
        procs.append(p)
        threads.append(t)
    for p in procs:
        p.wait()
    for t in threads:
        t.join(timeout=5)


if __name__ == "__main__":
    main()
