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


def pack(work, name, title, report=True):
    res = os.path.join(work, "results")
    models = os.path.join(work, "models")
    os.makedirs(res, exist_ok=True)
    runs_csv(res)
    info = None
    if report:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import report as rep
        figs = os.path.join(res, "figures")
        shutil.rmtree(figs, ignore_errors=True)
        info = rep.build(res, title, os.path.join(work, f"{name}__report.pdf"), figs)
    zip_dir(res, os.path.join(work, f"{name}__results.zip"))
    if os.path.isdir(models):
        zip_dir(models, os.path.join(work, f"{name}__models.zip"))
    return info


if __name__ == "__main__":
    print(pack(sys.argv[1], sys.argv[2], sys.argv[3]))
