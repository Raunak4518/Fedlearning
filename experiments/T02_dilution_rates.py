"""T02 - Generators without weight decay (the paper's DCGAN-F, DDPM-F): no collapse, but dilution. Zero GPU.

With lambda = 0 a non-holder's gradient on class c's conditioning parameters is exactly zero, so Adam
leaves them unchanged and flat FedAvg gives  r_c^+ - r_c = (m_c / K) * mean holder update  (no
shrink term). Under Adam a holder moves each coordinate by ~lr on the steps whose batch contains c,
so the expected per-round movement of row c is proportional to
    flat FedAvg : v_c = (1/K) sum_k S_k f_kc ,   HWA : v_c = sum_k w_kc S_k f_kc ,
with S_k the client's local steps, f_kc = 1 - (1 - pi_kc)^B the chance a batch contains c, and
w_kc = n_kc / n_c. This script computes v_c on the real splits (MNIST / FMNIST long tail,
K = 10 / 50 / 100) and checks the prediction against the T01 run without weight decay.

    python T02_dilution_rates.py   ->  results/T02_dilution_rates/{rates.json, summary.md}
"""
import json
import os
import sys

import numpy as np
from scipy import stats

sys.argv = sys.argv[:1]
import K09_kaggle_ablation_final as E  # noqa: E402

B, EP = 64, 5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "T02_dilution_rates")


def rates(counts):
    cnt = np.asarray(counts, float)
    n_k = cnt.sum(1)
    f = 1 - (1 - cnt / n_k[:, None]) ** B
    steps = EP * np.ceil(n_k / B)
    a = steps[:, None] * f
    flat = a.sum(0) / len(n_k)
    hwa = (cnt / cnt.sum(0, keepdims=True) * a).sum(0)
    return dict(holders=((cnt > 0).sum(0) / len(n_k)).tolist(), flat=(flat / flat[0]).tolist(),
                hwa=(hwa / hwa[0]).tolist(), hwa_over_flat=((hwa / hwa[0]) / (flat / flat[0])).tolist(),
                loss_share=(cnt.sum(0) / cnt.sum()).tolist())


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for ds in ["mnist", "fmnist"]:
        data = E.load_dataset(ds, "./data")
        for K in [10, 50, 100]:
            res[f"{ds}_K{K}"] = rates(E.make_partition(data["ytr"], K, 0.1, 0.01, 0.5, 0)["counts"])
    # validation on T01 (lambda = 0, flat FedAvg, MNIST LT, K = 10, seed 0)
    z = np.load(os.path.join(os.path.dirname(OUT), "T01_collapse_threshold", "mnist_s0_GeFL-F_wd0.npz"))
    g, loc, cnt = z["glob"], z["local"], z["counts"].astype(float)
    pred = np.array(rates(cnt)["flat"])
    H = cnt > 0
    val = {}
    for lo, hi in [(0, 10), (0, 40)]:
        meas = (g[hi] - g[lo]) / (hi - lo)
        r = stats.spearmanr(pred, meas)
        val[f"rounds_{lo}_{hi}"] = dict(spearman=float(r.correlation), p=float(r.pvalue),
                                        measured_rel=(meas / meas[0]).tolist())
    n_k = cnt.sum(1)
    f = 1 - (1 - cnt / n_k[:, None]) ** B
    steps = EP * np.ceil(n_k / B)
    gk, fk, nh = [], [], []
    for t in range(loc.shape[0]):
        for c in range(cnt.shape[1]):
            for k in range(cnt.shape[0]):
                (gk if H[k, c] else nh).append(loc[t, k, c] - g[t, c])
                if H[k, c]:
                    fk.append(steps[k] * f[k, c])
    r = stats.spearmanr(fk, gk)
    val["holder_gain_vs_class_steps"] = dict(spearman=float(r.correlation), p=float(r.pvalue), n=len(gk))
    val["max_nonholder_change"] = float(np.max(np.abs(nh)))
    res["validation_T01_wd0"] = val
    json.dump(res, open(os.path.join(OUT, "rates.json"), "w"), indent=1)

    L = ["# T02: row learning rates without weight decay (DCGAN-F / DDPM-F regime)", "",
         "Per-round movement of each class's conditioning row, relative to class 0 (Adam, no decay; "
         "seed-0 splits, IF = 100, Dir 0.5). `HWA/flat` > 1 means HWA speeds that class up.", ""]
    for key, v in res.items():
        if key.startswith("validation"):
            continue
        L += [f"## {key}", "", "| | " + " | ".join(str(c) for c in range(10)) + " |", "|---" * 11 + "|"]
        for row, lab in [("holders", "holder share m_c/K"), ("flat", "flat FedAvg rate"), ("hwa", "HWA rate"),
                         ("hwa_over_flat", "HWA / flat"), ("loss_share", "share of shared-parameter loss")]:
            L.append(f"| {lab} | " + " | ".join(f"{x:.2f}" if row != "loss_share" else f"{100 * x:.1f}%"
                                                 for x in v[row]) + " |")
        L.append("")
    L += ["## Check against T01 (lambda = 0, flat FedAvg, MNIST LT, K = 10)", "",
          f"* largest change of a row by a non-holder in 40 rounds: {val['max_nonholder_change']:.1e} (exactly zero)",
          f"* predicted vs measured row growth per class, rounds 0-10: Spearman {val['rounds_0_10']['spearman']:.2f} "
          f"(p = {val['rounds_0_10']['p']:.3f}); rounds 0-40: {val['rounds_0_40']['spearman']:.2f} "
          f"(p = {val['rounds_0_40']['p']:.3f})",
          f"* holder gain vs expected steps containing the class: Spearman "
          f"{val['holder_gain_vs_class_steps']['spearman']:.2f} over {val['holder_gain_vs_class_steps']['n']} holder-rounds"]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
