"""
experiments/run_queue.py

Runs the experiment files unattended, in priority order, in N parallel
lanes on one GPU (these models are small and launch-bound, so two
processes share a GPU well). One failure never stops the queue; each job
logs to results/logs/<job>.log, and results/queue_status.md is rewritten
after every job.

    python experiments/run_queue.py --phase quick      # viability pass, all 12
    python experiments/run_queue.py --phase full       # full runs, priority order
    python experiments/run_queue.py --phase all        # quick, then full
"""
import argparse
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ALL = ["E00_paper_parity", "E01_imbalance_gap", "E02_conditioning_collapse", "E03_method_components",
       "E04_client_scaling", "E05_datasets", "E06_iid_paper_setting", "E07_ablations",
       "E08_generators", "E09_oracles", "E10_cifar10_track2", "E11_privacy_cost"]

# Full phase: most decision-relevant first (plan section 9). Each lane is
# a list; lanes run in parallel.
FULL_LANES = [
    [("E03_method_components", ["--datasets", "mnist"]),
     ("E03_method_components", ["--datasets", "fmnist"]),
     ("E04_client_scaling", []),
     ("E09_oracles", []),
     ("E07_ablations", []),
     ("E11_privacy_cost", []),
     ("E10_cifar10_track2", [])],
    [("E00_paper_parity", ["--datasets", "mnist", "fmnist"]),
     ("E02_conditioning_collapse", []),
     ("E06_iid_paper_setting", []),
     ("E01_imbalance_gap", []),
     ("E08_generators", []),
     ("E05_datasets", []),
     ("E00_paper_parity", ["--datasets", "svhn", "cifar10"])],
]

status = {}
lock = threading.Lock()


def write_status(out_dir):
    with lock:
        lines = ["# Experiment queue status", "", f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                 "| job | state | minutes |", "|---|---|---|"]
        for job, (state, mins) in status.items():
            lines.append(f"| {job} | {state} | {mins} |")
        with open(os.path.join(out_dir, "queue_status.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_lane(jobs, extra, out_dir):
    for name, args in jobs:
        job = name + ("" if not args else " " + " ".join(args)) + (" [quick]" if "--quick" in extra else "")
        tag = job.replace(" ", "_").replace("[", "").replace("]", "")
        with lock:
            status[job] = ("running", "")
        write_status(out_dir)
        t0 = time.time()
        log_path = os.path.join(out_dir, "logs", tag + ".log")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        with open(log_path, "a", encoding="utf-8") as lf:
            rc = subprocess.call([sys.executable, os.path.join(HERE, name + ".py")] + args + extra,
                                 stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        with lock:
            status[job] = ("done" if rc == 0 else f"FAILED rc={rc}", f"{(time.time() - t0) / 60:.1f}")
        write_status(out_dir)


def run_phase(lanes, extra, out_dir):
    threads = [threading.Thread(target=run_lane, args=(lane, extra, out_dir)) for lane in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["quick", "full", "all"], default="all")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--lanes", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out_dir, "logs"), exist_ok=True)
    common = ["--data_root", a.data_root, "--out_dir", a.out_dir]
    if a.phase in ("quick", "all"):
        quick = [[(n, []) for n in ALL[i::a.lanes]] for i in range(a.lanes)]
        run_phase(quick, common + ["--quick"], a.out_dir)
    if a.phase in ("full", "all"):
        run_phase(FULL_LANES, common, a.out_dir)


if __name__ == "__main__":
    main()
