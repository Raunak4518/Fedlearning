"""Runs a generated Kaggle notebook end to end on this machine's CPU, from the notebook file itself: the helper
scripts are written from its %%writefile cells, its run list is shrunk (2 rounds per stage, 20% of the data), paths
under /kaggle/working and /tmp/fedecs are moved to a test folder, the GPU / Internet / pip cells are skipped.
Checks that the run cell, the acceptance cell and the packing cell complete and that the three output files exist.

    python fedecs_lt/tests/run_notebook_on_cpu.py fedecs_lt/notebooks/0_acceptance_done/NB00_acceptance.ipynb TEST_DIR [SEEDS] [LABELS]

SEEDS (e.g. "3") and LABELS ("FedECS-LT;FedProto") keep only those runs, so that a notebook with 51 runs can be tested
in about an hour.
"""
import json
import os
import re
import subprocess
import sys

nb_path, test_dir = sys.argv[1], os.path.abspath(sys.argv[2])
keep_seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else None
keep_labels = sys.argv[4].split(";") if len(sys.argv) > 4 and sys.argv[4] else None
nb = json.load(open(nb_path))
name = os.path.basename(nb_path)[:-6]
W, T = os.path.join(test_dir, "working"), os.path.join(test_dir, "tmp")
os.makedirs(W, exist_ok=True)


def sub(text):
    return text.replace("/kaggle/working", W).replace("/tmp/fedecs", T)


script = ["import os, sys, json"]
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.startswith("# Environment check") or src.startswith("# Packages"):
        continue
    if src.startswith("%%writefile "):
        first, body = src.split("\n", 1)
        path = sub(first.split(" ", 1)[1].strip())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(body)
        continue
    src = sub(src)
    if src.startswith("# Run list"):
        src += (f"\nRUNS = [r for r in RUNS if ({keep_seeds!r} is None or r['seed'] in {keep_seeds!r}) and "
                f"({keep_labels!r} is None or r['label'] in {keep_labels!r})]\n"
                "print('CPU test:', len(RUNS), 'runs kept')\n"
                "for r in RUNS:\n    r.update(rounds=dict(T_FE=2, T_KA=2, T_TN=2, T=2), frac_scale=0.2)\n"
                f"json.dump(RUNS, open('{T}/work/runs.json', 'w'))\n")
    src = re.sub(r"'--gpus', '2', '--per_gpu', '\d'", "'--gpus', '0', '--per_gpu', '2'", src)
    script.append(src)
open(os.path.join(test_dir, "nb_as_script.py"), "w").write("\n\n".join(script))
# LongTailLib runs: one local epoch instead of ten, so that the CPU test finishes (Kaggle runs the protocol's ten)
p = subprocess.run([sys.executable, os.path.join(test_dir, "nb_as_script.py")], cwd=test_dir,
                   env=dict(os.environ, FEDECS_LT_TEST_ARGS="-ls 1"))
outs = sorted(f for f in os.listdir(W) if f.startswith(name))
print("outputs:", outs)
assert p.returncode == 0, "notebook script failed"
for suffix in ("__results.zip", "__report.pdf", "__models.zip", "__summary.zip"):
    assert name + suffix in outs, f"missing {name + suffix}"
import zipfile  # noqa: E402
with zipfile.ZipFile(os.path.join(W, name + "__results.zip")) as z:
    info = json.loads(z.read(next(n for n in z.namelist() if n.endswith("report_info.json"))))
print("report:", info)
assert "report_error" not in info, info.get("report_error")
for key in ("failures", "audit_problems", "figure_errors", "table_errors", "layout_problems", "caption_problems"):
    assert info.get(key, 0) == 0, f"{key}: {info.get(key)}"
print("NOTEBOOK CPU TEST PASSED")
