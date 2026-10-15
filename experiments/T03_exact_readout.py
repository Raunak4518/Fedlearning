"""T03 - Exact-statistics readout (ER) and the readout-bias mechanism. Isolated; see PREREGISTRATION.md.

Runs GeFL-F / +HWA+LA / Ours-A through the unchanged pipeline (K09 module) and, at the point where
the pipeline evaluates its pooled-real oracle, additionally computes:
  * ER-LDA: for every architecture g, every client embeds its real features with head g's body and
    contributes class sums  S_gc = sum e,  counts n_c  and the second moment  M_g = sum e e^T  (all sums,
    i.e. secure-aggregatable). The server fits the uniform-prior LDA readout
        w_c = Sigma^-1 mu_c ,  b_c = -1/2 mu_c^T Sigma^-1 mu_c ,
    Sigma = pooled within-class covariance, shrunk: (1 - g) Sigma + g tr(Sigma)/d I, g = 0.1.
  * the mechanism quantities B_gc = (1 - alpha_kc) ||Delta_gc||_{Sigma^-1} and dR_gc (oracle - head recall).

    python T03_exact_readout.py --datasets fmnist [mnist] --out_dir results/T03_exact_readout
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats

sys.argv, _argv = sys.argv[:1], sys.argv
import core as E  # noqa: E402  (shared pipeline, experiments/core.py)
sys.argv = _argv

GAMMA = 0.1
_STASH = {}


@torch.no_grad()
def _embed_all(h, X, bs=4096):
    return torch.cat([h.embed(X[i:i + bs]) for i in range(0, len(X), bs)]).double()


@torch.no_grad()
def er_lda(ctx, headers, feats, ys_list, feats_te):
    """Exact-statistics LDA readout per architecture; returns (eval dict, per-head stats for the mechanism)."""
    C = E.NUM_CLASSES
    yte = ctx.data["yte"]
    out, aux = {}, {}
    for g in ctx.groups:
        h = headers[g]
        h.eval()
        S, n, M = None, torch.zeros(C, dtype=torch.float64, device=E.DEV), None
        for X, Y in zip(feats, ys_list):              # every client, every architecture: sums only
            e = _embed_all(h, X)
            if S is None:
                S = torch.zeros(C, e.shape[1], dtype=torch.float64, device=E.DEV)
                M = torch.zeros(e.shape[1], e.shape[1], dtype=torch.float64, device=E.DEV)
            S.index_add_(0, Y, e)
            n += torch.bincount(Y, minlength=C).double()
            M += e.T @ e
        mu = S / n[:, None].clamp(min=1)
        N, d = float(n.sum()), mu.shape[1]
        Sw = (M - (mu.T * n) @ mu) / max(N - C, 1.0)
        Sw = (Sw + Sw.T) / 2
        Ss = (1 - GAMMA) * Sw + GAMMA * torch.trace(Sw) / d * torch.eye(d, dtype=torch.float64, device=E.DEV)
        P = torch.linalg.inv(Ss)
        W = mu @ P                                      # C x d
        b = -0.5 * (W * mu).sum(1)
        logits = _embed_all(h, feats_te) @ W.T + b
        correct = logits.argmax(1) == yte
        rec = torch.stack([correct[yte == c].float().mean() for c in range(C)]).cpu().numpy()
        out[g] = dict(acc=float(correct.float().mean()), recall=rec)
        aux[g] = dict(mu=mu, P=P)
        h.train()
    return out, aux


@torch.no_grad()
def mechanism(ctx, headers, aux, G, head_ev, orc_ev, T_s):
    C = E.NUM_CLASSES
    counts = np.asarray(ctx.part["counts"], float)
    rows = []
    G.eval()
    syn = {c: G.sample(torch.full((1000,), c, device=E.DEV, dtype=torch.long)) for c in range(C)}
    for k in range(ctx.K):
        g = ctx.arch[k]
        h = headers[g]
        h.eval()
        n_k = counts[k].sum()
        wn = h.fc.weight.norm(dim=1)
        wmed = float(wn.median())
        for c in range(C):
            alpha = 5 * counts[k, c] / (5 * counts[k, c] + T_s * n_k / C)
            delta = _embed_all(h, syn[c]).mean(0) - aux[g]["mu"][c]
            dist = float(torch.sqrt(delta @ aux[g]["P"] @ delta))
            rows.append(dict(client=k, g=int(g), c=c, n_kc=int(counts[k, c]), alpha=float(alpha), delta=dist,
                             B=float((1 - alpha) * dist), wdef=float(1 - float(wn[c]) / wmed),
                             dR=float(orc_ev[g]["recall"][c] - head_ev[g]["recall"][c])))
        if ctx.K > 10 and k >= 9:
            break
    return rows


def _hook_oracle(orig):
    def f(ctx, headers, feats, ys_list, feats_te, steps=300):
        _STASH.update(ctx=ctx, headers=headers, feats=feats, ys=ys_list, feats_te=feats_te)
        return orig(ctx, headers, feats, ys_list, feats_te, steps)
    return f


def _hook_bbc(orig):
    def f(ctx, headers, G, feats_te, n=500, iters=50):
        _STASH["G"] = G
        return orig(ctx, headers, G, feats_te, n, iters)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["fmnist"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick_scale", type=float, default=0.4)
    ap.add_argument("--out_dir", default="results/T03_exact_readout")
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_name", default="runs.jsonl")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    E.classifier_oracle = _hook_oracle(E.classifier_oracle)
    E.balanced_bias_calibration = _hook_bbc(E.balanced_bias_calibration)
    cfg = dict(quick=True, quick_scale=a.quick_scale, quick_frac={}, seeds=[a.seed], data_root=a.data_root,
               referee=True, mnd=False)
    arms = [("GeFL-F", E.compose("")), ("+HWA+LA", E.compose("HWA+LA")),
            ("Ours-A Ts=1", E.compose("PC+LA+CSL+MC+KH"))]
    cache, data_cache = {}, {}
    for ds in a.datasets:
        for lab, m in arms:
            run = E.run_spec(lab, ds, a.seed, m, IF=0.01, alpha=0.5)
            t0 = time.time()
            res = E.run_one(cfg, run, cache, data_cache)
            st = _STASH
            head_ev = E.eval_headers(st["ctx"], st["headers"], st["feats_te"], st["ctx"].data["yte"])
            orc_ev = None
            # recompute oracle per-head recalls for the mechanism (same procedure as the pipeline's oracle)
            import copy
            X, Y = torch.cat(st["feats"]), torch.cat(st["ys"])
            w = 1.0 / torch.bincount(Y, minlength=E.NUM_CLASSES).float().clamp(min=1)
            hs = {}
            torch.manual_seed(a.seed + 99)
            for g in st["ctx"].groups:
                hh = copy.deepcopy(st["headers"][g])
                hh.eval()
                with torch.no_grad():
                    emb = hh.embed(X)
                opt = torch.optim.SGD(hh.fc.parameters(), lr=0.05)
                for _ in range(300):
                    bidx = torch.multinomial(w[Y], 128, replacement=True)
                    loss = torch.nn.functional.cross_entropy(hh.fc(emb[bidx]), Y[bidx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                hs[g] = hh
            orc_ev = E.eval_headers(st["ctx"], hs, st["feats_te"], st["ctx"].data["yte"])
            er_ev, aux = er_lda(st["ctx"], st["headers"], st["feats"], st["ys"], st["feats_te"])
            tn = {}
            for g in st["ctx"].groups:                 # tau-normalisation (Kang et al. 2020), tau = 1
                hh = copy.deepcopy(st["headers"][g])
                with torch.no_grad():
                    hh.fc.weight.div_(hh.fc.weight.norm(dim=1, keepdim=True).clamp(min=1e-12))
                tn[g] = hh
            tn_ev = E.eval_headers(st["ctx"], tn, st["feats_te"], st["ctx"].data["yte"])
            T_s = m["head"]["ts"]
            rows = mechanism(st["ctx"], st["headers"], aux, st["G"], head_ev, orc_ev, T_s) if st.get("G") else []
            summ = {k: E.summarize_eval(v) for k, v in [("head", head_ev), ("oracle", orc_ev), ("er_lda", er_ev), ("tnorm", tn_ev)]}
            rec = dict(label=lab, dataset=ds, seed=a.seed, quick_scale=a.quick_scale,
                       head_bal=summ["head"]["bal"], oracle_bal=summ["oracle"]["bal"], er_lda_bal=summ["er_lda"]["bal"],
                       tnorm_bal=summ["tnorm"]["bal"], head_tail=summ["head"]["tail"], oracle_tail=summ["oracle"]["tail"], er_lda_tail=summ["er_lda"]["tail"],
                       ens_bal=summ["head"].get("ens_bal"), bbc_bal=res.get("bbc_bal"),
                       pipeline_final_bal=res.get("final_bal"), pipeline_oracle_bal=res.get("oracle_bal"),
                       time_s=time.time() - t0, mechanism=rows)
            if rows:
                B = np.array([r["B"] for r in rows]); dR = np.array([r["dR"] for r in rows])
                one_a = np.array([1 - r["alpha"] for r in rows]); dl = np.array([r["delta"] for r in rows])
                wd = np.array([r["wdef"] for r in rows])
                rec.update(spearman_wdef=float(stats.spearmanr(wd, dR).correlation),
                           spearman_B=float(stats.spearmanr(B, dR).correlation),
                           spearman_1ma=float(stats.spearmanr(one_a, dR).correlation),
                           spearman_delta=float(stats.spearmanr(dl, dR).correlation))
            with open(os.path.join(a.out_dir, a.out_name), "a") as f:
                f.write(json.dumps(rec) + "\n")
            E.log(f"[T03] {ds} {lab}: head {rec['head_bal']*100:.2f} | oracle {rec['oracle_bal']*100:.2f} | "
                  f"ER-LDA {rec['er_lda_bal']*100:.2f} | tau-norm {rec['tnorm_bal']*100:.2f} | rho(wdef)={rec.get('spearman_wdef', float('nan')):.2f} | bbc {100*(rec['bbc_bal'] or float('nan')):.2f} | "
                  f"rho(B)={rec.get('spearman_B', float('nan')):.2f} rho(1-a)={rec.get('spearman_1ma', float('nan')):.2f} "
                  f"rho(D)={rec.get('spearman_delta', float('nan')):.2f}")
            _STASH.clear()


if __name__ == "__main__":
    main()
