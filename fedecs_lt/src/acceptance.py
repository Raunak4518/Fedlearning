"""Acceptance check of the Flower implementation against the earlier experiments (first notebook, NB00).

The new code cannot reproduce the old runs bit for bit: the old pipeline ran all clients in one process with one random
stream, the Flower simulation seeds every client action separately. It must reproduce what does not depend on that
stream exactly, and everything else within seed-to-seed variation:
  1. client splits: every client's class counts equal those logged by the old runs (exact);
  2. balanced accuracy as trained and with the exact readout: within max(2 points, 3 x the old seed SD) of the old
     run with the same configuration and seed 3;
  3. class separation F of the frozen features: within 25% of the old value (it depends on the trained extractor);
  4. the gate chooses the same generator branch as before (MNIST and CIFAR-10: the mean anchor).
Reference values: experiments/results (K17 GeFL-F and FedECS-LT; K14 for the final configuration without kernel
herding; K21 for the client counts). The CIFAR-10 FedECS-LT reference still had kernel herding (removing it costs
0.13-0.37 points with the readout), so its tolerance is 3 points.

    python acceptance.py RESULTS_DIR   ->  prints the verdict, writes RESULTS_DIR/acceptance.json
"""
import json
import os
import sys

import numpy as np

REF = {
    "mnist|GeFL-F": dict(final_bal={3: 0.7495, 4: 0.6941, 5: 0.7185}, er_bal={3: 0.9036, 4: 0.9137, 5: 0.9137}, F={3: 0.3255, 4: 0.3142, 5: 0.2911}, time=dict(fe=27, gen=483, head=33)),
    "mnist|-KH": dict(final_bal={3: 0.9451, 4: 0.9453}, er_bal={3: 0.9382, 4: 0.9427}, F={3: 0.3135, 4: 0.3108}, time=dict(fe=25, gen=487, head=100)),
    "mnist|Ours Ts=10": dict(final_bal={3: 0.9512, 4: 0.9488, 5: 0.9546}, er_bal={3: 0.9423, 4: 0.9452, 5: 0.9503}, F={3: 0.3165, 4: 0.3198, 5: 0.29}, time=dict(fe=25, gen=480, head=529)),
    "cifar10|GeFL-F": dict(final_bal={3: 0.3296, 4: 0.3452, 5: 0.3343}, er_bal={3: 0.5027, 4: 0.517, 5: 0.5018}, F={3: 0.0769, 4: 0.0596, 5: 0.0703}, time=dict(fe=128, gen=2970, head=173)),
    "cifar10|Ours Ts=10": dict(final_bal={3: 0.4234, 4: 0.4414, 5: 0.4486}, er_bal={3: 0.5235, 4: 0.5327, 5: 0.5349}, F={3: 0.0764, 4: 0.0584, 5: 0.0714}, time=dict(fe=132, gen=2974, head=2360)),
}
COUNTS = {"mnist": [[134, 604, 6, 0, 30, 101, 5, 13, 2, 0], [349, 5, 433, 15, 9, 7, 3, 0, 9, 2], [1257, 4, 34, 0, 109, 36, 18, 10, 16, 0], [114, 5, 6, 3, 1, 0, 9, 8, 0, 1], [143, 33, 42, 17, 80, 8, 58, 21, 1, 0], [85, 85, 191, 54, 6, 7, 1, 1, 0, 3], [141, 365, 129, 93, 10, 17, 1, 3, 7, 2], [170, 65, 7, 1, 14, 0, 5, 3, 3, 2], [22, 228, 3, 11, 0, 5, 6, 2, 0, 2], [2, 55, 17, 326, 53, 6, 6, 6, 2, 12]], "cifar10": [[18, 398, 7, 91, 75, 137, 70, 72, 41, 4], [148, 26, 1077, 1405, 1, 47, 0, 5, 7, 0], [385, 23, 81, 42, 250, 72, 17, 0, 1, 1], [67, 1, 0, 199, 379, 197, 21, 14, 3, 6], [1151, 147, 20, 21, 155, 3, 121, 17, 2, 0], [123, 236, 323, 30, 45, 126, 11, 13, 91, 2], [172, 2803, 1107, 71, 256, 5, 1, 57, 3, 63], [1740, 54, 106, 42, 136, 5, 22, 76, 1, 9], [765, 23, 567, 78, 0, 22, 165, 8, 0, 14], [231, 1089, 332, 191, 4, 165, 39, 18, 19, 1]]}
CHECKS = [  # (dataset, new label, reference key, minimum tolerance)
    ("mnist", "GeFL-F", "mnist|GeFL-F", 0.02),
    ("mnist", "FedECS-LT", "mnist|-KH", 0.02),
    ("cifar10", "GeFL-F", "cifar10|GeFL-F", 0.02),
    ("cifar10", "FedECS-LT", "cifar10|Ours Ts=10", 0.03),
]


def check(res_dir):
    runs = [json.loads(line) for line in open(os.path.join(res_dir, "runs.jsonl")) if line.strip()]
    rows, ok_all = [], True
    for ds, lab, ref_key, tol_min in CHECKS:
        r = next((x for x in runs if x["dataset"] == ds and x["label"] == lab and x["seed"] == 3 and x["K"] == 10
                  and x["IF"] == 0.01 and x["alpha"] == 0.5), None)
        ref = REF[ref_key]
        if r is None:
            rows.append(dict(check=f"{ds} {lab}", verdict="MISSING"))
            ok_all = False
            continue
        out = dict(check=f"{ds} {lab}")
        out["counts_identical"] = bool(np.array_equal(np.array(r["client_counts"]), np.array(COUNTS[ds])))
        for key in ("final_bal", "er_bal"):
            olds = np.array(list(ref[key].values()), dtype=float)
            tol = max(tol_min, 3 * olds.std(ddof=1)) if len(olds) > 1 else tol_min
            new = r.get(key)
            out[key] = dict(new=new, old_seed3=ref[key][3], tolerance=round(float(tol), 4),
                            ok=new is not None and abs(new - ref[key][3]) <= tol)
        if r.get("class_sep_F") is not None:
            out["F"] = dict(new=r["class_sep_F"], old_seed3=ref["F"][3],
                            ok=abs(r["class_sep_F"] - ref["F"][3]) <= 0.25 * ref["F"][3])
            out["gate_branch"] = dict(new=r.get("gate_branch"), expected="anchor", ok=r.get("gate_branch") == "anchor")
        out["time_s"] = dict(new=dict(fe=r.get("fe_time_s"), gen=r.get("gen_time_s"), total=r.get("time_s")),
                             old=ref["time"])
        ok = out["counts_identical"] and all(v["ok"] for k, v in out.items() if isinstance(v, dict) and "ok" in v)
        out["verdict"] = "PASS" if ok else "FAIL"
        ok_all &= ok
        rows.append(out)
    verdict = dict(overall="PASS" if ok_all else "FAIL", checks=rows)
    json.dump(verdict, open(os.path.join(res_dir, "acceptance.json"), "w"), indent=1)
    return verdict


if __name__ == "__main__":
    v = check(sys.argv[1])
    for row in v["checks"]:
        print(row["verdict"], row["check"], {k: row[k] for k in row if k not in ("check", "verdict", "time_s")})
    print()
    print("ACCEPTANCE:", v["overall"])
