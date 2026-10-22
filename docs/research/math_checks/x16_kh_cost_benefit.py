"""X16: is kernel herding (KH) worth its cost? Paired effect of removing KH (or replacing it by a random subset of the
same 4n candidates) on balanced accuracy, and the head-stage time ratio, from the existing ablation runs:
K09 + K14 (MNIST / FashionMNIST, seeds 0-4; ER logged for seeds 3-4 only) and K20 (FashionMNIST / CIFAR-10, seeds 3-4).
Output: x16_kh_cost_benefit.json and a printed table. No new runs."""
import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments", "results"))
KEYS = ["final_bal", "er_bal", "final_tail"]


def load(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, pattern, "runs.jsonl"))):
        out += [json.loads(line) for line in open(f)]
    return out


def compare(R, ds, ref, alt):
    A = {r["seed"]: r for r in R if r["dataset"] == ds and r["label"] == ref}
    B = {r["seed"]: r for r in R if r["dataset"] == ds and r["label"] == alt}
    seeds = sorted(set(A) & set(B))
    row = dict(dataset=ds, reference=ref, alternative=alt, seeds=seeds)
    for k in KEYS:
        d = [100 * (A[s][k] - B[s][k]) for s in seeds if A[s].get(k) is not None and B[s].get(k) is not None]
        row[k] = dict(mean=float(np.mean(d)), per_seed=[round(x, 2) for x in d],
                      p=float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 else None) if d else None
    row["head_time_ratio"] = float(np.mean([A[s]["head_time_s"] / B[s]["head_time_s"] for s in seeds]))
    return row


def main():
    rows = []
    R = load("K09_kaggle_ablation_final__*") + load("K14_kaggle_ablation_seeds_mcb")
    rows += [compare(R, ds, "Ours-A Ts=10", "-KH") for ds in ("mnist", "fmnist")]
    R = load("K20_kaggle_mechanism_controls")
    for ds in ("fmnist", "cifar10"):
        rows += [compare(R, ds, "Ours (reference)", "no KH"), compare(R, ds, "Ours (reference)", "KH -> random subset")]
    er = [x for r in rows if r["alternative"] != "KH -> random subset" for x in r["er_bal"]["per_seed"]]
    summary = dict(er_removal_per_seed=er, er_removal_positive=f"{sum(x > 0 for x in er)} of {len(er)}",
                   sign_test_p=float(stats.binomtest(sum(x > 0 for x in er), len(er)).pvalue))
    json.dump(dict(rows=rows, summary=summary), open(os.path.join(HERE, "x16_kh_cost_benefit.json"), "w"), indent=1)
    for r in rows:
        print(f"{r['dataset']:8s} KH vs {r['alternative']:20s} seeds {r['seeds']}: as trained {r['final_bal']['mean']:+.2f}"
              f" | ER {r['er_bal']['mean']:+.2f} | tail {r['final_tail']['mean']:+.2f} | head stage x{r['head_time_ratio']:.1f}")
    print(summary)


if __name__ == "__main__":
    main()
