"""Packs one experiment's outputs into the three files a helper sends back:
  <name>__results.zip  every figure (PDF, PNG, CSV), every metric (runs.jsonl, runs.csv), every log, failures
  <name>__report.pdf   the readable report (report.py)
  <name>__models.zip   trained weights (feature extractor, heads, generator, referee, readout) for later re-plotting
Called after every run, so a timeout never loses finished work."""
import csv
import json
import os
import shutil
import sys
import zipfile


def runs_csv(res_dir):
    p = os.path.join(res_dir, "runs.jsonl")
    if not os.path.exists(p):
        return
    rows = [json.loads(line) for line in open(p) if line.strip()]
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)})
    with open(os.path.join(res_dir, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])


def zip_dir(src, dst, skip=()):
    tmp = dst + ".part"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _, files in os.walk(src):
            if any(s in root for s in skip):
                continue
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, os.path.dirname(src)))
    os.replace(tmp, dst)


def pack(work, name, title, report=True, out_dir=None, models=True):
    """work holds results/ and models/; the three files go to out_dir (default: work)."""
    out_dir = out_dir or work
    os.makedirs(out_dir, exist_ok=True)
    res = os.path.join(work, "results")
    mdir = os.path.join(work, "models")
    os.makedirs(res, exist_ok=True)
    runs_csv(res)
    info = None
    if report:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import report as rep
        import audit
        figs = os.path.join(res, "figures")
        shutil.rmtree(figs, ignore_errors=True)
        audit.audit(res, figures=False)          # runs and logs; the report shows its findings
        info = rep.build(res, title, os.path.join(out_dir, f"{name}__report.pdf"), figs)
        a = audit.audit(res)                     # again with the figures the report has drawn
        info = dict(info or {}, audit_problems=a["n_problems"])
    zip_dir(res, os.path.join(out_dir, f"{name}__results.zip"))
    if models and os.path.isdir(mdir):
        zip_dir(mdir, os.path.join(out_dir, f"{name}__models.zip"))
    return info


def quick(work, name, out_dir):
    """After every run: results.zip only (fast), under a lock so parallel workers do not collide."""
    import fcntl
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(work, ".pack.lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        runs_csv(os.path.join(work, "results"))
        zip_dir(os.path.join(work, "results"), os.path.join(out_dir, f"{name}__results.zip"))


if __name__ == "__main__":
    # python pack.py WORK NAME "TITLE" OUT_DIR
    print(pack(sys.argv[1], sys.argv[2], sys.argv[3], out_dir=sys.argv[4] if len(sys.argv) > 4 else None))
