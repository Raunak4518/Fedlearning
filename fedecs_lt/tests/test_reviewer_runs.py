"""The runs requested by the reviews of the paper (build_notebooks.reviewer_notebooks): each new arm differs from its
reference in exactly the stated setting, the notebooks hold the planned runs, no planned run has come back already,
and seeds 6-8 give new client splits."""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import build_notebooks as bn  # noqa: E402
from runner import PIPELINE, run_id  # noqa: E402


def diff(a, b, path=""):
    """Paths at which two method specs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in diff(a.get(k), b.get(k), f"{path}/{k}")]
    return [] if a == b else [path]


def test_new_arms_change_one_setting():
    assert diff(PIPELINE["GeFL-F + AdamW"], PIPELINE["GeFL-F"]) == ["/gen/opt"]
    assert PIPELINE["GeFL-F + AdamW"]["gen"]["opt"] == "adamw" and PIPELINE["GeFL-F + AdamW"]["head"]["loss"] == "ce"
    assert diff(PIPELINE["GeFL-F (T_s=10)"], PIPELINE["GeFL-F"]) == ["/head/ts"]
    assert diff(PIPELINE["FedDF (T_s=10)"], PIPELINE["FedDF"]) == ["/head/ts"]
    assert PIPELINE["GeFL-F (T_s=10)"]["head"]["ts"] == PIPELINE["FedECS-LT"]["head"]["ts"] == 10
    assert diff(PIPELINE["FedECS-LT w/ Gaussian sampler"], PIPELINE["FedECS-LT"]) == ["/gen/beta", "/gen/opt",
                                                                                       "/gen/type", "/gen/wd"]
    assert PIPELINE["FedECS-LT w/ Gaussian sampler"]["gen"] == dict(type="gauss")
    # the references are unchanged by the new entries
    assert PIPELINE["GeFL-F"]["gen"]["opt"] == "adam" and PIPELINE["GeFL-F"]["head"]["ts"] == 1


def test_reviewer_notebooks_hold_the_planned_runs():
    nbs = bn.all_notebooks()
    mine = {n: nb for n, nb in nbs.items() if n.startswith(bn.REVIEWER_NOTEBOOKS)}
    assert all(bn.folder(n) == "8_reviewer_requested_runs" for n in mine)
    runs = {}
    for n, nb in mine.items():
        for r in nb["runs"]:
            runs.setdefault(next(x for x in bn.REVIEWER_NOTEBOOKS if n.startswith(x)), []).append(r)
    labels = {k: {r["label"] for r in v} for k, v in runs.items()}
    assert labels["cifar10_cifar100_components_of_fedecs_lt"] == set(__import__("runner").COMPONENTS)
    assert labels["cifar10_cifar100_gefl_f_variants"] == set(bn.GEFL_VARIANTS)
    assert labels["cifar10_cifar100_matched_synthetic_epochs"] == set(bn.MATCHED_TS)
    assert labels["cifar10_cifar100_fresh_seeds_6_to_8"] == set(bn.FRESH_METHODS)
    for k, v in runs.items():
        seeds = bn.FRESH_SEEDS if "fresh" in k else bn.SEEDS
        assert len(v) == len(labels[k]) * 2 * 3 == len({run_id(r) for r in v})       # 2 datasets x 3 seeds, once
        assert {r["seed"] for r in v} == set(seeds) and {r["ds"] for r in v} == {"cifar10", "cifar100"}
        assert all(r["IF"] == 0.01 and r["alpha"] == 0.5 and r["K"] == 10 for r in v)
    assert not any(n.startswith(("NB30", "NB31")) for n in nbs)                        # replaced
    # nothing planned here has come back already (never run twice)
    done = set()
    for f in glob.glob(os.path.join(HERE, "..", "results", "*", "runs.jsonl")):
        done |= {json.loads(x)["run_id"] for x in open(f) if x.strip()}
    assert not done & {run_id(r) for v in runs.values() for r in v}


def test_fresh_seeds_give_new_splits():
    import torch
    from data import make_partition
    y = torch.from_numpy(np.repeat(np.arange(10), 400))
    parts = {s: make_partition(y, 10, 10, 1.0, 0.01, 0.5, s) for s in (3, 4, 5, 6, 7, 8)}
    again = make_partition(y, 10, 10, 1.0, 0.01, 0.5, 6)

    def flat(p):
        return [sorted(c.tolist()) for c in p["parts"]]
    assert flat(again) == flat(parts[6])                                               # deterministic
    for a in (6, 7, 8):
        for b in (3, 4, 5):
            assert flat(parts[a]) != flat(parts[b])
