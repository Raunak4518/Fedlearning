"""Analyse T01 runs: checks the contraction theorem round by round on the real CVAE-F.

For every (arm, round t, class c) with flat FedAvg:
  bound_t,c = mean_k ||r_kc||  (= ||r_c|| + (1/K)(sum_H G_k - sum_N D_k))
  check 1   ||r_c^+|| <= bound_t,c            (the theorem; must hold always)
  theta_t,c = Dbar / (Gbar + Dbar)            (contraction certified iff m_c/K < theta)
  compare   certified contraction vs the old rule m_c < K/2 vs what actually happened.
Also: Gbar/Dbar against the holders' class frequency, and D per local step across lambda.

    python T01_analyze.py results/T01_collapse_threshold [> summary.md]
"""
import glob
import json
import math
import os
import sys

import numpy as np

B, EPOCHS, HEAD, TAIL = 64, 5, slice(0, 3), slice(6, 10)


def load(d):
    runs = []
    for js in sorted(glob.glob(os.path.join(d, "*.json"))):
        meta = json.load(open(js))
        z = np.load(js[:-5] + ".npz")
        runs.append((meta, z["glob"], z["local"], z["counts"], z["sizes"]))
    return runs


def steps(sizes):
    return np.array([EPOCHS * math.ceil(n / B) for n in sizes])


def analyse_flat(meta, g, loc, counts, sizes):
    R, K, C = loc.shape
    S = steps(sizes)
    H = counts > 0                                   # K x C holders
    m = H.sum(0)
    rows = []
    for t in range(R):
        for c in range(C):
            n0, n1, nk = g[t, c], g[t + 1, c], loc[t, :, c]
            Gk, Dk = nk[H[:, c]] - n0, n0 - nk[~H[:, c]]
            Gbar, Dbar = (Gk.mean() if len(Gk) else 0.0), (Dk.mean() if len(Dk) else 0.0)
            theta = Dbar / (Gbar + Dbar) if Gbar + Dbar > 0 else float("nan")
            rows.append(dict(t=t, c=c, m=int(m[c]), n0=n0, n1=n1, bound=nk.mean(), Gbar=Gbar, Dbar=Dbar,
                             theta=theta, Dstep=float(np.mean(Dk / S[~H[:, c]])) if len(Dk) else float("nan")))
    return rows


def main(d):
    runs = load(d)
    out = ["# T01: contraction theorem checked on the real CVAE-F", ""]
    for meta, g, loc, counts, sizes in runs:
        R, K, C = loc.shape
        H = counts > 0
        m = H.sum(0)
        frac = counts / counts.sum(1, keepdims=True)                  # pi_kc
        fbatch = 1 - (1 - frac) ** B                                  # P(class c in a batch of client k)
        out.append(f"## {meta['label']}  ({meta['dataset']}, seed {meta['seed']}, K={K}, {R} rounds, "
                   f"IF={meta['IF']}, alpha={meta['alpha']})")
        out.append("")
        out.append("holders m_c per class: " + " ".join(f"{c}:{int(m[c])}" for c in range(C)))
        out.append("total samples n_c:     " + " ".join(f"{c}:{int(counts[:, c].sum())}" for c in range(C)))
        rel = g[-1] / g[0]
        out.append(f"row norm end / init: head {rel[HEAD].mean():.3f}, tail {rel[TAIL].mean():.3f}; "
                   f"tail/head at end {g[-1, TAIL].mean() / g[-1, HEAD].mean():.3f}")
        out.append("per class end / init:  " + " ".join(f"{c}:{rel[c]:.2f}" for c in range(C)))
        out.append("")
        if meta["agg"] != "flat":
            continue
        rows = analyse_flat(meta, g, loc, counts, sizes)
        viol = [r for r in rows if r["n1"] > r["bound"] * (1 + 1e-5) + 1e-7]
        cert = [r for r in rows if r["bound"] < r["n0"]]
        shrink = [r for r in rows if r["n1"] < r["n0"]]
        old_no = [r for r in rows if 2 * r["m"] >= K]                 # old rule: no contraction predicted
        old_no_shrunk = [r for r in old_no if r["n1"] < r["n0"]]
        old_no_cert = [r for r in old_no if r["bound"] < r["n0"]]
        slack = np.array([r["n1"] / r["bound"] for r in rows])
        out.append(f"* theorem violations (||r+|| > bound): **{len(viol)} / {len(rows)}** row-rounds")
        out.append(f"* certified contractions (bound < ||r||): {len(cert)}; actual contractions: {len(shrink)}; "
                   f"certified but did not shrink: {sum(r['n1'] >= r['n0'] for r in cert)}")
        out.append(f"* alignment slack ||r+|| / bound: mean {slack.mean():.4f}, min {slack.min():.4f}")
        out.append(f"* row-rounds with m_c >= K/2 (old theorem: no contraction): {len(old_no)}; "
                   f"of these shrank: {len(old_no_shrunk)}; certified by the new theorem: {len(old_no_cert)}")
        out.append("")
        out.append("| class | m_c | m_c/K | mean holder class-in-batch freq | Gbar/Dbar (median) | theta (median) "
                   "| rounds certified to contract | rounds shrank |")
        out.append("|---|---|---|---|---|---|---|---|")
        for c in range(C):
            rc = [r for r in rows if r["c"] == c]
            ratio = [r["Gbar"] / r["Dbar"] for r in rc if r["Dbar"] > 0]
            th = [r["theta"] for r in rc if r["theta"] == r["theta"]]
            fb = fbatch[H[:, c], c].mean()
            out.append(f"| {c} | {int(m[c])} | {m[c] / K:.1f} | {fb:.3f} | "
                       f"{np.median(ratio) if ratio else float('nan'):.3f} | "
                       f"{np.median(th) if th else float('nan'):.3f} | "
                       f"{sum(r['bound'] < r['n0'] for r in rc)}/{len(rc)} | {sum(r['n1'] < r['n0'] for r in rc)}/{len(rc)} |")
        # holder gain vs class frequency, pooled over rounds (per holder)
        S = steps(sizes)
        gk, fk = [], []
        for t in range(R):
            for c in range(C):
                n0 = g[t, c]
                Dbar = np.mean(n0 - loc[t, ~H[:, c], c]) if (~H[:, c]).any() else np.nan
                if not Dbar > 0:
                    continue
                for k in np.where(H[:, c])[0]:
                    gk.append((loc[t, k, c] - n0) / Dbar)
                    fk.append(fbatch[k, c])
        gk, fk = np.array(gk), np.array(fk)
        if len(gk) > 2:
            rho = np.corrcoef(np.argsort(np.argsort(fk)), np.argsort(np.argsort(gk)))[0, 1]
            out.append("")
            out.append(f"* holder gain G_k / Dbar vs P(class in batch): Spearman rho = {rho:.3f} over {len(gk)} "
                       f"holder-rounds")
            for lo, hi in [(0, .1), (.1, .5), (.5, .9), (.9, 1.01)]:
                sel = (fk >= lo) & (fk < hi)
                if sel.any():
                    out.append(f"  * P in [{lo}, {min(hi, 1)}): median G_k/Dbar = {np.median(gk[sel]):.3f} (n={sel.sum()})")
        dstep = [r["Dstep"] for r in rows if r["t"] < 5 and r["Dstep"] == r["Dstep"]]
        out.append(f"* non-holder norm loss per local step, rounds 0-4: mean {np.mean(dstep):.2e}")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
