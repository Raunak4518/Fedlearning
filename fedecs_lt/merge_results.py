"""Joins the results of several notebooks into one experiment and builds one report from them (repository tool; it
runs on any computer with the packages of the notebooks, no GPU needed).

Use it when one table of the paper needs runs from more than one notebook: the parts of a split notebook
(NB10_main_cifar10_part1of3 ... part3of3), the seeds of a Track-2 setting (NB20_..._IF100_s3 / s4 / s5), or an
ablation notebook together with the main notebook whose FedECS-LT and GeFL-F runs it is paired with.

    python fedecs_lt/merge_results.py OUT_DIR "Report title" A__results.zip B__results.zip ...
    python fedecs_lt/merge_results.py --no-report OUT_DIR "Title" A__summary.zip B__summary.zip ...   # tables only

Inputs are `NAME__results.zip` files (or folders unpacked from them, or `NAME__summary.zip` files: then the
figures that need the saved predictions, e.g. confusion matrices and t-SNE, are listed as not applicable).
A run that appears in more than one input (same run id) is taken from the input listed last. OUT_DIR receives
`results/` (all runs, logs and predictions, with `runs.csv` and the figures), `merged__report.pdf` and `tables/`
(the paper's numbers as LaTeX and CSV, paper_tables.py). With --no-report only `results/` and `tables/` are written
(seconds instead of up to an hour).
"""
import glob
import json
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))


def results_dir(path, tmp):
    """The folder holding runs.jsonl inside an input (a zip is unpacked into tmp first)."""
    if zipfile.is_zipfile(path):
        dst = tempfile.mkdtemp(dir=tmp)
        with zipfile.ZipFile(path) as z:
            z.extractall(dst)
        path = dst
    hits = sorted(glob.glob(os.path.join(path, "**", "runs.jsonl"), recursive=True), key=len)
    if not hits:
        raise SystemExit(f"no runs.jsonl in {path}")
    return os.path.dirname(hits[0])


def merge(inputs, out_res):
    os.makedirs(os.path.join(out_res, "logs"), exist_ok=True)
    os.makedirs(os.path.join(out_res, "predictions"), exist_ok=True)
    runs, failures, versions = {}, {}, []
    with tempfile.TemporaryDirectory() as tmp:
        for inp in inputs:
            d = results_dir(inp, tmp)
            for line in open(os.path.join(d, "runs.jsonl")):
                if line.strip():
                    r = json.loads(line)
                    if r["run_id"] in runs:
                        print(f"  {r['run_id']}: taken from {os.path.basename(inp)} (also in an earlier input)")
                    runs[r["run_id"]] = r
            fp = os.path.join(d, "failures.jsonl")
            if os.path.exists(fp):
                for line in open(fp):
                    if line.strip():
                        f = json.loads(line)
                        failures[f["run_id"]] = line if line.endswith("\n") else line + "\n"
            cv = os.path.join(d, "code_version.txt")
            if os.path.exists(cv):
                v = open(cv).read().strip()
                if v and v not in versions:
                    versions.append(v)
            for sub in ("logs", "predictions"):
                for f in glob.glob(os.path.join(d, sub, "*")):
                    if os.path.isfile(f):
                        shutil.copy2(f, os.path.join(out_res, sub, os.path.basename(f)))
            print(f"read {os.path.basename(inp)}")
    with open(os.path.join(out_res, "runs.jsonl"), "w") as f:
        for r in runs.values():
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out_res, "failures.jsonl"), "w") as f:   # only runs that never finished
        f.writelines(v for k, v in failures.items() if k not in runs)
    if versions:
        with open(os.path.join(out_res, "code_version.txt"), "w") as f:
            f.write("\n".join(versions) + "\n")
    return len(runs)


def main():
    args = [a for a in sys.argv[1:] if a != "--no-report"]
    if len(args) < 3:
        raise SystemExit(__doc__)
    out, title, inputs = args[0], args[1], args[2:]
    res = os.path.join(out, "results")
    if os.path.exists(res):
        raise SystemExit(f"{res} exists: choose a new OUT_DIR")
    n = merge(inputs, res)
    print(f"{n} runs from {len(inputs)} inputs")
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "paper_tables.py"), res, os.path.join(out, "tables")],
                   check=True)
    if "--no-report" in sys.argv:
        return
    os.environ["FEDECS_MERGED"] = "1"     # the report's software table then says where it was built
    import audit
    import pack
    import report
    pack.runs_csv(res)
    audit.audit(res, figures=False)
    info = report.build(res, title, os.path.join(out, "merged__report.pdf"), os.path.join(res, "figures"))
    info = dict(info, audit_problems=audit.audit(res)["n_problems"])
    print(json.dumps(info))
    print("report:", os.path.join(out, "merged__report.pdf"))


if __name__ == "__main__":
    main()
