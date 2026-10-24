"""Logging audit of one experiment's results (run before the report; its findings are printed on the report's cover
and listed in an appendix, and written to results/audit.json).

For every run: every per-class metric (metrics.PER_CLASS) for every class on the test set, on the validation set
(methods that evaluate it) and with the exact readout (the GeFL-F pipeline family); the class order and names; the
run time. For every per-round evaluation in the run's log: every per-class metric for every class, for the clients'
models and for the aggregated (or global) model. For every figure: its PDF, PNG and CSV.

    python audit.py RESULTS_DIR
"""
import glob
import json
import os
import sys

from metrics import PER_CLASS

ROUND_EVENTS = ("stage1_round", "stage3_round", "round_eval")


def _vec_ok(v, C):
    return isinstance(v, list) and len(v) == C and any(x is not None for x in v)


def _logs(res_dir):
    out = {}
    for f in glob.glob(os.path.join(res_dir, "logs", "*.jsonl")):
        recs = []
        for line in open(f):
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        if recs and recs[0].get("event") == "run_start":
            out[recs[0]["run_id"]] = recs
    return out


def audit(res_dir, figures=True):
    problems = []
    path = os.path.join(res_dir, "runs.jsonl")
    runs = [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []
    logs = _logs(res_dir)
    checked = dict(runs=len(runs), round_evaluations=0, figures=0)
    for r in runs:
        rid, pipe = r["run_id"], (r.get("method") or {}).get("pipeline")
        C = len(r.get("final_recall_c") or r.get("class_order") or [])
        if not C:
            problems.append(f"{rid}: no per-class results at all")
            continue
        need = [("final", PER_CLASS)]
        if pipe != "longtaillib":
            need.append(("val_final", PER_CLASS))
        if pipe == "gefl" and (r.get("method") or {}).get("readout", True):
            need.append(("er", PER_CLASS))
        for prefix, keys in need:
            miss = [k for k in keys if not _vec_ok(r.get(f"{prefix}_{k}"), C)]
            if miss:
                problems.append(f"{rid}: {prefix} per-class metrics missing: {', '.join(miss)}")
        if len(r.get("class_order") or []) != C:
            problems.append(f"{rid}: class order not recorded")
        if r.get("class_names") is not None and len(r["class_names"]) != C:
            problems.append(f"{rid}: class names do not match the {C} classes")
        if not r.get("time_s"):
            problems.append(f"{rid}: run time not recorded")
        if pipe == "local":
            continue            # no rounds: each client trains alone, evaluated once at the end
        ev = [e for e in logs.get(rid, []) if e.get("event") in ROUND_EVENTS and (e.get("post") or e.get("pre"))]
        if not ev:
            problems.append(f"{rid}: no per-round evaluations in the log")
            continue
        for e in ev:
            checked["round_evaluations"] += 1
            post = e.get("post") or {}
            miss = [k for k in PER_CLASS if not _vec_ok(post.get(k), C)]
            if miss:
                problems.append(f"{rid}: {e['event']} {e.get('round')}: aggregated/global model lacks {', '.join(miss)}")
            for cid, pre in (e.get("pre") or {}).items():
                miss = [k for k in PER_CLASS if not _vec_ok(pre.get(k), C)]
                if miss:
                    problems.append(f"{rid}: {e['event']} {e.get('round')}: client {cid} lacks {', '.join(miss)}")
                    break
    for pdf in (glob.glob(os.path.join(res_dir, "figures", "**", "*.pdf"), recursive=True) if figures else []):
        checked["figures"] += 1
        for ext in (".png", ".csv"):
            if not os.path.exists(pdf[:-4] + ext):
                problems.append(f"figure {os.path.relpath(pdf, res_dir)}: no {ext} file")
    # one line per kind of problem and run is enough to act on; keep at most 200
    out = dict(checked=checked, problems=problems[:200], n_problems=len(problems))
    with open(os.path.join(res_dir, "audit.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    a = audit(sys.argv[1])
    print(json.dumps(a["checked"]), f"{a['n_problems']} problems")
    for p in a["problems"]:
        print(" -", p)
