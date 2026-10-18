"""N7: where is the representation gap? FE (stage I) vs head bodies (stage III), on CPU, real data (MNIST / FMNIST).

Uses the repository's own code (experiments/core.py, read-only) for the partition, the FE, the heads, stage I,
the stage-I oracle and the ER readout. For one (dataset, seed), long tail IF = 100 (core convention IF=0.01),
Dirichlet 0.5, K = 10:

FEs   A  'fed'      core.stage_i, 20 rounds (GeFL-F's federated FE)
      A' 'fed_bn'   A with BatchNorm running stats replaced by the exact pooled (all-client) batch statistics
      B  'cen_bal'  core.stage_i_oracle (pooled, class-balanced sampling, 30 epochs)
      C  'cen_erm'  same, but instance (uniform) sampling: plain centralised ERM on the long-tailed pool
      D  'fed_iid'  core.stage_i on the balanced (IF = 1, IID) partition with the same total (reference world)
Per FE (on its own data):
  lda_raw        balanced LDA on the raw FE features from all clients (partition-invariant; exact sums)
  one_client     each architecture's head trained on ITS client's real features only (= Stage III real phase,
                 50 rounds x 5 epochs, SGD 0.1, B = 64; K17's "Real only"), random init (and stage-I init for A)
  pooled         each architecture's head trained on the pooled real features (instance sampling), same number
                 of SGD steps as one_client (25 epochs of the pool)
  pooled_bal     same with class-balanced sampling
  each scored as trained ("heads") and with ER (core.readout_heads lda / all clients)
Outputs JSON lines to n7_<ds>_s<seed>.jsonl
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import sys, json, time, copy, math, os
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, _os.path.join(_REPO, "experiments"))
import core

ROOT = _os.path.join(_REPO, "experiments", "data")
OUT = _HERE


def bal_eval(ctx, hs, feats_te, yte):
    s = core.summarize_eval(core.eval_headers(ctx, hs, feats_te, yte))
    return dict(bal=round(100 * s["bal"], 2), tail=round(100 * s["tail"], 2), worst=round(100 * s["worst"], 2))


@torch.no_grad()
def lda_raw(feats, ys, Xte, yte, C, gamma=0.1):
    X = torch.cat(feats).flatten(1).double()
    Y = torch.cat(ys)
    n = torch.bincount(Y, minlength=C).double()
    mu = torch.zeros(C, X.shape[1], dtype=torch.float64).index_add_(0, Y, X) / n[:, None]
    Sw = (X.T @ X - (mu.T * n) @ mu) / (n.sum() - C)
    d = X.shape[1]
    Ss = (1 - gamma) * Sw + gamma * torch.trace(Sw) / d * torch.eye(d, dtype=torch.float64)
    W = mu @ torch.linalg.inv(Ss)
    b = -0.5 * (W * mu).sum(1)
    pred = (Xte.flatten(1).double() @ W.T + b).argmax(1)
    rec = torch.stack([(pred[yte == c] == c).double().mean() for c in range(C)])
    return round(100 * float(rec.mean()), 2)


def train_head(h, X, Y, steps_epochs, B=64, lr=0.1, balanced=False, C=None):
    C = C or core.NUM_CLASSES
    h.train()
    opt = torch.optim.SGD(h.parameters(), lr=lr, momentum=0.0)
    n = len(Y)
    w = 1.0 / torch.bincount(Y, minlength=C).float().clamp(min=1)
    for _ in range(steps_epochs):
        if balanced:
            order = torch.multinomial(w[Y], n, replacement=True)
        else:
            order = torch.randperm(n)
        for s in range(0, n, B):
            b = order[s:s + B]
            if b.numel() < 2:
                continue
            loss = F.cross_entropy(h(X[b]), Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return h


def bodies(ctx, feats, ys, feats_te, yte, tag, init_sd=None, seed=0):
    res = {}
    in_ch = feats[0].shape[1]
    X, Y = torch.cat(feats), torch.cat(ys)
    # one-client real-only heads: 50 rounds x 5 epochs on own data (K=10: one client per architecture)
    torch.manual_seed(seed + 11)
    hs = core.build_headers(ctx, in_ch)
    for k in range(ctx.K):
        g = ctx.arch[k]
        if g not in ctx.groups:
            continue
        if init_sd is not None:
            hs[g].load_state_dict(init_sd[g])
        train_head(hs[g], feats[k], ys[k], 5 * ctx.spec["T_TN"], B=ctx.B)
    res["one_client_heads"] = bal_eval(ctx, hs, feats_te, yte)
    res["one_client_ER"] = bal_eval(ctx, core.readout_heads(ctx, hs, feats, ys, "lda", "all"), feats_te, yte)
    for bal in (False, True):
        torch.manual_seed(seed + 12)
        hp = core.build_headers(ctx, in_ch)
        for g in ctx.groups:
            train_head(hp[g], X, Y, 5 * ctx.spec["T_TN"] // ctx.K, B=ctx.B, balanced=bal)
        nm = "pooled_bal" if bal else "pooled"
        res[nm + "_heads"] = bal_eval(ctx, hp, feats_te, yte)
        res[nm + "_ER"] = bal_eval(ctx, core.readout_heads(ctx, hp, feats, ys, "lda", "all"), feats_te, yte)
    return res


def stage_i_oracle_erm(ctx, epochs=30):
    """core.stage_i_oracle with uniform (instance) sampling instead of class-balanced sampling."""
    d, spec = ctx.data, ctx.spec
    fe = core.PaperFE(spec["in_ch"], spec["fe"]).to(core.DEV)
    headers = core.build_headers(ctx, fe.out_ch)
    allidx = torch.cat(ctx.part["parts"])
    y = d["ytr"][allidx]
    params = list(fe.parameters()) + [p for g in ctx.groups for p in headers[g].parameters()]
    opt = torch.optim.SGD(params, lr=0.1, momentum=0.0)
    n = len(allidx)
    for _ in range(epochs):
        draw = torch.randperm(n)
        for s in range(0, n, ctx.B):
            b = draw[s:s + ctx.B]
            if b.numel() < 2:
                continue
            h = fe(d["xtr"][allidx[b]])
            loss = sum(F.cross_entropy(headers[g](h), y[b]) for g in ctx.groups)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return core.sd_clone(fe), [core.sd_clone(h) for h in headers], fe


@torch.no_grad()
def exact_bn(fe, xs_list):
    """Replace the FE's BN running stats by the exact pooled batch statistics of its pre-BN activations
    (per-channel sums of u and u^2 over every client: secure-aggregatable)."""
    fe = copy.deepcopy(fe)
    fe.eval()
    mods = list(fe.net)
    out = []
    for i, m in enumerate(mods):
        if isinstance(m, torch.nn.BatchNorm2d):
            # pooled pre-BN activations: run the prefix in eval mode (earlier BNs already replaced)
            s1, s2, cnt = 0, 0, 0
            for x in xs_list:
                u = torch.nn.Sequential(*mods[:i])(x)
                s1 = s1 + u.sum((0, 2, 3)).double()
                s2 = s2 + u.pow(2).sum((0, 2, 3)).double()
                cnt += u.shape[0] * u.shape[2] * u.shape[3]
            mean = s1 / cnt
            var = s2 / cnt - mean ** 2
            out.append(dict(layer=i, avg_running_mean=m.running_mean.tolist(), avg_running_var=m.running_var.tolist(),
                            pooled_mean=mean.tolist(), pooled_var=var.tolist()))
            m.running_mean.copy_(mean.float())
            m.running_var.copy_((var * cnt / (cnt - 1)).float())
    return fe, out


def per_client_bn(fe, xs_list):
    """Per-client train-mode batch statistics of the first BN's input (heterogeneity diagnostic)."""
    mods = list(fe.net)
    with torch.no_grad():
        i = [j for j, m in enumerate(mods) if isinstance(m, torch.nn.BatchNorm2d)][0]
        stats = []
        for x in xs_list:
            u = torch.nn.Sequential(*mods[:i])(x)
            stats.append((u.mean((0, 2, 3)).tolist(), u.var((0, 2, 3)).tolist()))
    return stats


def run(ds, seed, which):
    torch.set_num_threads(int(os.environ.get("NT", "2")))
    core.set_num_classes(ds)
    data0 = core.load_dataset(ds, ROOT)
    cfg = core.default_cfg()
    fout = open(f"{OUT}/n7c_{ds}_s{seed}.jsonl", "a")

    def emit(d):
        d.update(ds=ds, seed=seed)
        print(json.dumps(d), flush=True)
        fout.write(json.dumps(d) + "\n")
        fout.flush()

    for world in which:
        IF, alpha = (0.01, 0.5) if world == "lt" else (1.0, None)
        part = core.make_partition(data0["ytr"], 10, core.DATASETS[ds]["frac"], IF, alpha, seed)
        data, part = core.localize(data0, part)
        ctx = core.Ctx(cfg, data, ds, part, 10, seed)
        xs = [data["xtr"][p] for p in part["parts"]]
        ys = [data["ytr"][p] for p in part["parts"]]
        yte = data["yte"]
        fes = {}
        t = time.time()
        core.set_seed(seed)
        fe_sd, hd_sd, fe = core.stage_i(ctx, core.DATASETS[ds]["T_FE"], "all")
        fe.load_state_dict(fe_sd)
        fes["fed" if world == "lt" else "fed_iid"] = (fe, hd_sd)
        emit(dict(world=world, event="stage_i_done", t=round(time.time() - t), bn_clients=per_client_bn(fe, xs)))
        if world == "lt":
            fe_bn, bnstats = exact_bn(fe, [x[:4000] for x in xs])
            fes["fed_bn"] = (fe_bn, None)
            emit(dict(world=world, event="bn", stats=bnstats))
            core.set_seed(seed)
            t = time.time()
            sd, hsd, feo = core.stage_i_oracle(ctx, epochs=30)
            feo.load_state_dict(sd)
            fes["cen_bal"] = (feo, None)
            core.set_seed(seed)
            sd, hsd, fee = stage_i_oracle_erm(ctx, epochs=30) if "cen_erm" in os.environ.get("FES", "cen_erm") else (sd, hsd, feo)
            fee.load_state_dict(sd)
            fes["cen_erm"] = (fee, None)
            emit(dict(world=world, event="oracle_fes_done", t=round(time.time() - t)))
        ctx.groups = [int(x) for x in os.environ.get("ARCHS", "0,4,8").split(",")]  # subset AFTER stage I
        keep = os.environ.get("FES")
        if keep:
            fes = {k: v for k, v in fes.items() if k in keep.split(",") or k == "fed_iid"}
        for name, (fe_, init) in fes.items():
            t = time.time()
            feats = [core.compute_features(fe_, x) for x in xs]
            fte = core.compute_features(fe_, data["xte"])
            r = dict(world=world, fe=name, lda_raw=lda_raw(feats, ys, fte, yte, core.NUM_CLASSES))
            r.update(bodies(ctx, feats, ys, fte, yte, name, seed=seed))
            if init is not None:
                torch.manual_seed(seed + 13)
                hs = core.build_headers(ctx, feats[0].shape[1])
                for k in [k for k in range(ctx.K) if ctx.arch[k] in ctx.groups]:
                    hs[ctx.arch[k]].load_state_dict(init[ctx.arch[k]])
                    train_head(hs[ctx.arch[k]], feats[k], ys[k], 5 * ctx.spec["T_TN"], B=ctx.B)
                r["one_client_stageIinit_heads"] = bal_eval(ctx, hs, fte, yte)
                r["one_client_stageIinit_ER"] = bal_eval(ctx, core.readout_heads(ctx, hs, feats, ys, "lda", "all"), fte, yte)
            r["t"] = round(time.time() - t)
            emit(r)


if __name__ == "__main__":
    ds, seed = sys.argv[1], int(sys.argv[2])
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["lt", "iid"]
    run(ds, seed, which)
