"""merge_results.py joins notebooks' results without losing or doubling runs; paper_tables.py writes the report's
numbers (mean, SD, paired difference with Holm) into LaTeX / CSV."""
import csv
import json
import os
import subprocess
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
import merge_results  # noqa: E402


def run(label, seed, bal, ds="mnist"):
    return dict(run_id=f"{label}_{ds}_IF0.01_a0.5_K10_s{seed}", label=label, dataset=ds, IF=0.01, alpha=0.5,
                K=10, seed=seed, final_bal=bal, final_f1=bal - 0.05)


def results_zip(path, runs):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("results/runs.jsonl", "".join(json.dumps(r) + "\n" for r in runs))
        for r in runs:
            z.writestr(f"results/logs/{r['run_id']}.jsonl", json.dumps(dict(event="run_start", run_id=r["run_id"])))


def test_merge_and_tables(tmp_path):
    ours = [run("FedECS-LT", s, b) for s, b in ((3, 0.90), (4, 0.92), (5, 0.91))]
    ref = [run("GeFL-F", s, b) for s, b in ((3, 0.80), (4, 0.83), (5, 0.81))]
    results_zip(tmp_path / "A.zip", ours + ref[:1])
    stale = dict(ref[0], final_bal=0.10)                       # the same run, older result: the later input wins
    results_zip(tmp_path / "B.zip", [stale])
    other = [run(m, s, b, "fmnist") for m, b in (("FedECS-LT", 0.7), ("GeFL-F", 0.6)) for s in (3, 4, 5)]
    results_zip(tmp_path / "C.zip", ref + other)
    res = tmp_path / "merged" / "results"
    n = merge_results.merge([str(tmp_path / x) for x in ("A.zip", "B.zip", "C.zip")], str(res))
    assert n == 12
    kept = {json.loads(line)["run_id"]: json.loads(line) for line in open(res / "runs.jsonl")}
    assert kept[ref[0]["run_id"]]["final_bal"] == 0.80
    assert len(os.listdir(res / "logs")) == 12

    out = tmp_path / "tables"
    subprocess.run([sys.executable, os.path.join(ROOT, "paper_tables.py"), str(res), str(out)], check=True)
    main = list(csv.reader(open(out / "mnist_IF100_a0.5_K10_main.csv")))
    assert main[1][:2] == ["FedECS-LT", "91.0 ± 1.0"]
    paired = list(csv.reader(open(out / "mnist_IF100_a0.5_K10_paired.csv")))
    d = 100 * (np.array([0.90, 0.92, 0.91]) - np.array([0.80, 0.83, 0.81]))
    assert paired[1][0] == "GeFL-F" and paired[1][1] == "3" and paired[1][2] == f"{d.mean():+.1f}"
    assert r"\toprule" in open(out / "mnist_IF100_a0.5_K10_main.tex").read()
    across = list(csv.reader(open(out / "across_track1_final_bal.csv")))
    assert across[0] == ["Method", "MNIST", "Fashion-MNIST"] and across[1] == ["FedECS-LT", "91.0 ± 1.0", "70.0 ± 0.0"]
    assert list(csv.reader(open(out / "across_track1_paired.csv")))[1][0] == "GeFL-F"
