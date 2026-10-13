#!/usr/bin/env python3
"""D01 - What each Ours-A component does, measured on real features (long tail, K = 10).

Same partitions, seeds and stage order as run_one in core.py, so the generators are
the ones the E25 / E33 runs train. For each (dataset, seed):

  generators : CVAE-F (GeFL-F), CVAE-F + HWA, PC-VAE; samplers PC-VAE + MC, + KH
  per class  : conditioning-row norm (CVAE-F, HWA), referee fidelity (a head trained
               on held-out REAL features labels the fake samples), spread ratio,
               mean error / class std, MMD to the class's real TEST features
               (an RFF map independent of the one KH uses)
  seed 0     : a 2-D LDA view (fit on pooled real features, diagnostic only) of a
               rare class, and two Stage III runs of Ours-A (T_s = 10): with LA and
               with plain CE, recording per-class recall, the share of test
               predictions per class, and how close the CSL targets are to the
               real-data referee (total variation) against the one-hot labels.

python D01_component_evidence.py --data_root <dir> [--datasets mnist fmnist] [--seeds 0 1 2]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C  # noqa: E402

DEV = C.DEV
NC = C.NUM_CLASSES


def per_class(fn):
    return np.array([fn(c) for c in range(NC)])


@torch.no_grad()
def gen_metrics(G, s1, phi_eval, mu_test, m, V, n=1000):
    G.eval()
    out = dict(fid=[], spread=[], mean_err=[], mmd=[])
    means, within = [], []
    for c in range(NC):
        y = torch.full((n,), c, device=DEV, dtype=torch.long)
        x = G.sample(y)
        out["fid"].append(float((s1["ref"](x).argmax(1) == c).float().mean()))
        xf = x.flatten(1)
        mf = xf.mean(0)
        out["spread"].append(float((xf - mf).pow(2).sum(1).mean() / V[c]))
        out["mean_err"].append(float((mf - m[c]).norm() / V[c].sqrt()))
        out["mmd"].append(float((phi_eval(xf).mean(0) - mu_test[c]).norm()))
        means.append(mf)
        within.append((xf - mf).pow(2).sum(1).mean())
    res = {k: np.array(v) for k, v in out.items()}
    M = torch.stack(means)
    res["fisher_F"] = float((M - M.mean(0)).pow(2).sum(1).mean() / torch.stack(within).mean())
    Xt, yt = s1["feats_te"].flatten(1), s1["yte"]
    d = Xt.pow(2).sum(1, keepdim=True) - 2 * Xt @ M.T + M.pow(2).sum(1)[None]
    res["ncm_fake_means_on_real"] = float((d.argmin(1) == yt).float().mean())
    return res


def tstr(ctx, G, s1, n=1000, epochs=30):
    """Train a head ONLY on fake features (balanced, n per class), test on REAL test features."""
    with torch.no_grad():
        y = torch.arange(NC, device=DEV).repeat_interleave(n)
        x = torch.cat([G.sample(y[i:i + 4000]) for i in range(0, len(y), 4000)])
    h = C.PaperHeader(C.HEADER_WIDTHS[4], x.shape[1], True).to(DEV)
    opt = torch.optim.Adam(h.parameters(), lr=1e-3)
    for _ in range(epochs):
        for b in C.batches(len(y), 64):
            loss = torch.nn.functional.cross_entropy(h(x[b]), y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    h.eval()
    with torch.no_grad():
        ok = h(s1["feats_te"]).argmax(1) == s1["yte"]
        return [float(ok[s1["yte"] == c].float().mean()) for c in range(NC)]


def real_separability(X, Y, Xte, yte):
    """F = between-class spread of class means / within-class spread (what the
    server can compute from the sums), and nearest-class-mean accuracy on test."""
    M = torch.stack([X[Y == c].mean(0) for c in range(NC)])
    Vw = torch.stack([(X[Y == c] - M[c]).pow(2).sum(1).mean() for c in range(NC)])
    F_ = float((M - M.mean(0)).pow(2).sum(1).mean() / Vw.mean())
    d = Xte.pow(2).sum(1, keepdim=True) - 2 * Xte @ M.T + M.pow(2).sum(1)[None]
    return F_, float((d.argmin(1) == yte).float().mean())


def lda_2d(X, Y):
    X, Y = X.cpu().numpy().astype(np.float64), Y.cpu().numpy()
    mu = X.mean(0)
    Sw = np.zeros((X.shape[1],) * 2)
    Sb = np.zeros_like(Sw)
    for c in range(NC):
        Xc = X[Y == c]
        d = Xc - Xc.mean(0)
        Sw += d.T @ d
        e = (Xc.mean(0) - mu)[:, None]
        Sb += len(Xc) * e @ e.T
    Sw += 1e-3 * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    w, v = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-w.real)[:2]
    return v[:, order].real, mu


def run(ds, seed, root, out_dir, stage3, smoke=False, sep_only=False):
    t0 = time.time()
    cfg = dict(quick=smoke, quick_scale=0.03, quick_frac={}, seeds=[seed], data_root=root, referee=True, mnd=False)
    spec = C.DATASETS[ds]
    data = C.load_dataset(ds, root)
    part = C.make_partition(data["ytr"], 10, spec["frac"], 0.01, 0.5, seed)
    data, part = C.localize(data, part)
    ctx = C.Ctx(cfg, data, ds, part, 10, seed)
    base = C.compose("PC+LA+CSL+MC+KH", head=dict(ts=10))
    T = C.rounds_for(cfg, ds, base)

    # ---- stage (i), exactly as run_one
    C.set_seed(seed)
    fe_sd, hd_sd, fe = C.stage_i(ctx, T["T_FE"], "all", la=False, tracker=C.Tracker(), eval_every=10)
    fe.load_state_dict(fe_sd)
    feats = [C.compute_features(fe, data["xtr"][p]) for p in part["parts"]]
    ys = [data["ytr"][p] for p in part["parts"]]
    feats_te = C.compute_features(fe, data["xte"])
    ref = C.train_referee(ctx, fe)
    s1 = dict(fe=fe, hd_sd=hd_sd, feats=feats, ys=ys, feats_te=feats_te, ref=ref, yte=data["yte"])
    counts = part["counts"].sum(0)
    n, S1, S2 = C.anchor_stats(feats, ys)
    m = S1 / n[:, None]
    V = (S2 / n - m.pow(2).sum(1)).clamp(min=1e-8)
    # independent fingerprint for evaluation: other RFF seed, real TEST features
    phi_eval = C.RFFMap(feats[0][0].numel(), 3072, float(torch.sqrt(2 * V.mean())), seed=777 + seed).to(DEV)
    yte = data["yte"]
    Xte = feats_te.flatten(1)
    with torch.no_grad():
        mu_test = torch.stack([phi_eval(Xte[yte == c]).mean(0) for c in range(NC)])
        ref_acc = float((ref(feats_te).argmax(1) == yte).float().mean())
    res = dict(dataset=ds, seed=seed, class_counts=counts.tolist(), referee_test_acc=ref_acc,
               holders=(part["counts"] > 0).sum(0).tolist())
    Xtr = torch.cat(feats).flatten(1)
    Ytr = torch.cat(ys)
    res["real_F"], res["real_ncm_acc"] = real_separability(Xtr, Ytr, Xte, yte)
    if seed == 0:
        Wl, mul = lda_2d(Xtr, Ytr)
        proj = lambda x: ((x.flatten(1).cpu().numpy().astype(np.float64) - mul) @ Wl).astype(np.float32)
        rng = np.random.default_rng(0)
        bg = rng.choice(len(Xte), 3000, replace=False)
        sc = dict(bg=proj(Xte[bg]), bg_y=yte[bg].cpu().numpy(),
                  means=proj(torch.stack([Xtr[Ytr == c].mean(0) for c in range(NC)])))
    C.log(f"{ds} s{seed}: real F={res['real_F']:.3f} NCM acc={res['real_ncm_acc']:.3f}")
    if sep_only:
        if seed == 0:
            np.savez_compressed(os.path.join(out_dir, f"{ds}_s{seed}_lda.npz"), **sc)
        res["time_s"] = time.time() - t0
        with open(os.path.join(out_dir, f"{ds}_s{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f)
        return
    C.log(f"{ds} s{seed}: stage I done ({time.time() - t0:.0f}s), referee test acc {ref_acc:.3f}")

    gens = {}
    for name, parts in (("CVAE-F", ""), ("CVAE-F+HWA", "HWA"), ("PC-VAE", "PC")):
        gspec = C.compose(parts).get("gen")
        C.set_seed(seed + 1000)
        g_sd, norms, G, clients, W = C.stage_ii(ctx, gspec, feats, ys, T["T_KA"])
        del clients
        G.load_state_dict(g_sd)
        G.eval()
        gens[name] = G
        if norms.size and np.isfinite(norms).all() and norms.shape[-1] == NC and name != "PC-VAE":
            res[f"norms_{name}"] = norms.round(5).tolist()
        torch.cuda.empty_cache()
        C.log(f"{ds} s{seed}: {name} trained ({time.time() - t0:.0f}s)")
    samplers = dict(gens)
    samplers["PC-VAE+MC"] = C.MomentCalibratedGen(gens["PC-VAE"], feats, ys)
    phi, mu, use = C.federated_kme(ctx, feats, ys)
    samplers["PC-VAE+MC+KH"] = C.HerdedGen(samplers["PC-VAE+MC"], phi, mu, use=use)
    # real reference: the same metrics for real held-out features (what "perfect" looks like)
    with torch.no_grad():
        real = dict(fid=[], spread=[], mean_err=[], mmd=[])
        for c in range(NC):
            xc = Xte[yte == c]
            real["fid"].append(float((ref(feats_te[yte == c]).argmax(1) == c).float().mean()))
            real["spread"].append(float((xc - xc.mean(0)).pow(2).sum(1).mean() / V[c]))
            real["mean_err"].append(float((xc.mean(0) - m[c]).norm() / V[c].sqrt()))
            # MMD between two halves of the real test class: the sampling-noise floor
            h = len(xc) // 2
            real["mmd"].append(float((phi_eval(xc[:h]).mean(0) - phi_eval(xc[h:]).mean(0)).norm()))
    res["metrics"] = {"real (test)": {k: np.round(v, 5).tolist() for k, v in real.items()}}
    for name, G in samplers.items():
        C.set_seed(seed + 3000)
        mt = gen_metrics(G, s1, phi_eval, mu_test, m, V)
        C.set_seed(seed + 3500)
        mt["tstr_recall"] = np.array(tstr(ctx, G, s1))
        res["metrics"][name] = {k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) else v) for k, v in mt.items()}
        C.log(f"{ds} s{seed}: {name:14s} fid_tail={mt['fid'][6:].mean():.3f} spread_tail={mt['spread'][6:].mean():.3f} "
              f"mmd_tail={mt['mmd'][6:].mean():.4f} F={mt['fisher_F']:.3f} TSTR bal={mt['tstr_recall'].mean():.3f} tail={mt['tstr_recall'][6:].mean():.3f}")

    if seed == 0:
        # 2-D view of the rarest class
        for c in (9, 8):
            sc[f"real_{c}"] = proj(Xte[yte == c][:400])
            for name in ("CVAE-F", "PC-VAE", "PC-VAE+MC", "PC-VAE+MC+KH"):
                C.set_seed(seed + 4000)
                with torch.no_grad():
                    sc[f"{name}_{c}"] = proj(samplers[name].sample(torch.full((400,), c, device=DEV, dtype=torch.long)))
        np.savez_compressed(os.path.join(out_dir, f"{ds}_s{seed}_lda.npz"), **sc)

    if seed == 0 and stage3:
        G = samplers["PC-VAE+MC+KH"]
        for lab, loss in (("Ours-A", "la"), ("Ours-A with CE (no LA)", "ce")):
            hspec = C.compose("PC+LA+CSL+MC+KH", head=dict(ts=10, loss=loss))["head"]
            C.set_seed(seed + 2000)
            tr = C.Tracker()
            headers = C.stage_iii(ctx, hspec, fe, hd_sd, G, feats, ys, feats_te, T["T_TN"], tr, 10)
            last = tr.hist[-1]
            with torch.no_grad():
                share, per_client = [], []
                for g in ctx.groups:
                    headers[g].eval()
                    pred = headers[g](feats_te).argmax(1)
                    sh = torch.bincount(pred, minlength=NC).float().cpu().numpy() / len(pred)
                    share.append(sh)
                    for k in range(ctx.K):
                        if ctx.arch[k] == g:
                            pk = np.asarray(part["counts"][k], dtype=np.float64)
                            per_client.append(dict(client=k, prior=(pk / pk.sum()).round(5).tolist(), pred_share=sh.round(5).tolist()))
                share = np.mean(share, 0)
                # CSL targets on a fresh pool from the final heads vs the real-data referee
                C.set_seed(seed + 5000)
                py = torch.randint(0, NC, (5000,), device=DEV)
                px = G.sample(py)
                cons = sum(headers[g](px).softmax(1) for g in ctx.groups) / len(ctx.groups)
                onehot = torch.nn.functional.one_hot(py, NC).float()
                tgt = 0.5 * onehot + 0.5 * cons
                pref = ref(px).softmax(1)
                tv_hard = 0.5 * (onehot - pref).abs().sum(1)
                tv_csl = 0.5 * (tgt - pref).abs().sum(1)
                csl = dict(tv_onehot=per_class(lambda c: float(tv_hard[py == c].mean())).round(5).tolist(),
                           tv_csl=per_class(lambda c: float(tv_csl[py == c].mean())).round(5).tolist(),
                           label_agrees_referee=float((pref.argmax(1) == py).float().mean()))
            res[f"stage3::{lab}"] = dict(final_bal=last["bal"], final_tail=last["tail"], recall=last["recall"],
                                         pred_share=share.round(5).tolist(), csl=csl,
                                         per_client=per_client)
            C.log(f"{ds} s{seed}: {lab}: bal={last['bal'] * 100:.2f} tail={last['tail'] * 100:.2f} "
                  f"TV one-hot {np.mean(csl['tv_onehot']):.3f} -> CSL {np.mean(csl['tv_csl']):.3f}")
            torch.cuda.empty_cache()
    res["time_s"] = time.time() - t0
    with open(os.path.join(out_dir, f"{ds}_s{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    C.log(f"{ds} s{seed}: done in {res['time_s']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--out_dir", default="./results/D01_component_evidence")
    ap.add_argument("--datasets", nargs="+", default=["mnist", "fmnist"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--no_stage3", action="store_true")
    ap.add_argument("--sep_only", action="store_true", help="stage I + class separability only (e.g. svhn, cifar10)")
    ap.add_argument("--smoke", action="store_true", help="2-3 rounds per stage: code check only")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for s in a.seeds:
        for ds in a.datasets:
            if os.path.exists(os.path.join(a.out_dir, f"{ds}_s{s}.json")):
                continue
            run(ds, s, a.data_root, a.out_dir, not a.no_stage3, a.smoke, a.sep_only)
