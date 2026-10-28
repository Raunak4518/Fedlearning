"""GeFL's measure for every returned run of the setting with different client models: each model's best test
accuracy over the evaluated rounds, then the mean over the models (GeFL's code, utils/util.py evaluate_models:
best_perf[i] = max over rounds; reported sum(best_perf) / len(best_perf)), at GeFL's evaluation rounds of the last
stage (round 1, every 10th round and the last; GeFL-F's args.sample_test = 10). Our main tables report the model after
the last round instead (no selection on the test set); the paper gives this measure in an appendix table so that
results can be compared with GeFL's.

Reads the per-round logs in results/_downloads/<notebook>__summary.zip (fetch_kaggle.py keeps them there) and writes
results/<notebook>/gefl_measure.jsonl: {run_id, gefl_best_acc, gefl_last_acc, rounds, models, source}.
- Per model: each architecture's aggregated head (`post_arch`, logged since 29 Oct 2026), else each client's model
  (`pre`): with one client per architecture (10 clients) the aggregated head of an architecture is that client's
  head, and the personal-model methods (HtFLlib, FedALA) are scored per client anyway.
- Local (no rounds): its final accuracy.

    python fedecs_lt/gefl_measure.py            # every results folder with a summary zip
"""
import glob
import json
import os
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def gefl_rounds(rounds):
    last = max(rounds)
    return [r for r in rounds if r == 1 or r % 10 == 0 or r == last]


def measure(events, K):
    """(best, last, rounds used, number of models) from one run's log events, or None."""
    s3 = [e for e in events if e.get("event") == "stage3_round"]
    by_round = {}
    for e in s3:
        if "post_arch" in e:
            by_round[e["round"]] = {g: v.get("acc") for g, v in e["post_arch"].items()}
        elif "pre" in e and (K <= 10 or "post_clients" in e):
            by_round[e["round"]] = {k: v.get("acc") for k, v in e["pre"].items()}
    if not by_round:
        return None
    rs = gefl_rounds(sorted(by_round))
    models = sorted({m for r in rs for m in by_round[r]})
    best = [max(by_round[r][m] for r in rs if by_round[r].get(m) is not None) for m in models]
    last = [by_round[rs[-1]][m] for m in models if by_round[rs[-1]].get(m) is not None]
    return float(np.mean(best)), float(np.mean(last)), rs, len(models)


def folder(name):
    z = os.path.join(RES, "_downloads", f"{name}__summary.zip")
    rj = os.path.join(RES, name, "runs.jsonl")
    if not (os.path.exists(z) and os.path.exists(rj)):
        return 0
    runs = {}
    for line in open(rj):
        if line.strip():
            r = json.loads(line)
            runs[r["run_id"]] = r
    zf = zipfile.ZipFile(z)
    logs = {}
    for n in zf.namelist():
        if n.startswith("results/logs/") and n.endswith(".jsonl") and "/tb/" not in n:
            ev = [json.loads(x) for x in zf.read(n).decode().splitlines() if x.strip()]
            start = next((e for e in ev if e.get("event") == "run_start"), None)
            if start and start.get("run_id") in runs:
                logs[start["run_id"]] = ev
    out = []
    for rid, r in runs.items():
        if str(r.get("dataset", "")).endswith("_t2"):
            continue   # one shared model: not GeFL's setting
        if r["label"] == "Local":
            out.append(dict(run_id=rid, gefl_best_acc=r.get("final_acc"), gefl_last_acc=r.get("final_acc"),
                            rounds=[], models=r["K"], source="final (no rounds)"))
            continue
        m = measure(logs.get(rid, []), r["K"])
        if m:
            out.append(dict(run_id=rid, gefl_best_acc=round(m[0], 4), gefl_last_acc=round(m[1], 4), rounds=m[2],
                            models=m[3], source="per-round logs"))
    if out:
        with open(os.path.join(RES, name, "gefl_measure.jsonl"), "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
    return len(out)


def main():
    for d in sorted(glob.glob(os.path.join(RES, "*", "runs.jsonl"))):
        name = os.path.basename(os.path.dirname(d))
        if name.startswith(("NB00", "_")):
            continue
        n = folder(name)
        if n:
            print(f"{name}: {n} runs")


if __name__ == "__main__":
    main()
