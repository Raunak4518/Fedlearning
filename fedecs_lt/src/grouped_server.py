"""Server side of the baselines GeFL compares in its grouped setting (grouped_client.py): every round the server
averages the uploaded heads within each architecture, unweighted (GeFL utils/avg.py model_wise_FedAvg; an architecture
without participants keeps its previous head). CCVR then calibrates each architecture's last layer on virtual
features drawn from the class Gaussians of its clients' embeddings (LongTailLib serverccvr.py: global class mean and
covariance from the clients' statistics, covariance + 1e-6 I, 2,000 virtual features per class, the last layer
re-trained for 10 epochs by SGD with lr 0.001, momentum 0.9, weight decay 1e-5, batch 32)."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import active_clients
from fl_common import arrays_of, cfg_of, pack, seed_all
from flops import forward_flops
from grouped_client import new_head
from metrics import summarize
from monitor import Monitor
from server import DEV, EVAL_EVERY, Fed, flat, gpu_peaks, save_run, stage_one, train_referee

CCVR_HP = dict(num_vr=2000, retrain_epoch=10, retrain_lr=0.001, momentum=0.9, weight_decay=1e-5, batch_size=32)
MU = 1e-2   # FedProx (GeFL args.mu)


def average(sds):
    """Unweighted mean of state dicts (integer buffers such as num_batches_tracked: the last client's)."""
    return {n: (sum(sd[n].float() for sd in sds) / len(sds)).to(sds[0][n].dtype) if sds[0][n].is_floating_point()
            else sds[-1][n] for n in sds[0]}


def ccvr_calibrate(S, glob, in_ch, seed):
    """CCVR for every architecture: class statistics of its clients' embeddings (chunks of classes per message), class
    Gaussians, virtual features, re-trained last layer."""
    C, hp = S.C, CCVR_HP
    srv_all = {f"{g}|{n}": v for g, sd in glob.items() for n, v in sd.items()}
    out = {}
    for g in S.groups:
        ks = [k for k in range(S.K) if S.arch[k] == g]
        h = new_head("CCVR", g, in_ch, S.spec, C, seed).to(DEV)
        h.load_state_dict(glob[g])
        d = h.emb_dim
        step = max(1, int(64 * 2 ** 20 // (4 * d * d)))   # classes per message: at most 64 MB of outer products
        # accumulated in host memory (CIFAR-100 with a 1,024-dimensional embedding: 0.8 GB of outer products)
        n = torch.zeros(C, dtype=torch.float64)
        S1 = torch.zeros(C, d, dtype=torch.float64)
        S2 = torch.zeros(C, d, d, dtype=torch.float64)
        own = {n_: v for n_, v in srv_all.items() if n_.startswith(f"{g}|")}
        for c0 in range(0, C, step):
            c1 = min(C, c0 + step)
            rep = S.ask("query.ccvr_stats", [(S.nid[k], pack(S.base(method="CCVR", c0=c0, c1=c1, fe_key=S.feat_key),
                                                               srv=own)) for k in ks])
            for m in rep:
                a = arrays_of(m.content, "s", "cpu")
                n[c0:c1] += a["n"]
                S1[c0:c1] += a["S1"]
                S2[c0:c1] += a["S2"].double()
        seed_all(seed, "ccvr", g)
        xs, ys = [], []
        for c in range(C):
            if n[c] < 1:
                continue
            mu = S1[c] / n[c]
            cov = (S2[c] - n[c] * torch.outer(mu, mu)) / max(float(n[c]) - 1.0, 1.0)
            cov = (cov + cov.T) / 2 + 1e-6 * torch.eye(d, dtype=torch.float64)
            lam, V = torch.linalg.eigh(cov)   # multivariate normal through the eigendecomposition
            L = (V * lam.clamp(min=0).sqrt()).to(DEV)
            z = torch.randn(hp["num_vr"], d, dtype=torch.float64, device=DEV)
            xs.append((mu.to(DEV) + z @ L.T).float())
            ys.append(torch.full((hp["num_vr"],), c, dtype=torch.long, device=DEV))
        X, Y = torch.cat(xs), torch.cat(ys)
        fc = nn.Linear(d, C).to(DEV)
        fc.load_state_dict(h.fc.state_dict())
        opt = torch.optim.SGD(fc.parameters(), lr=hp["retrain_lr"], momentum=hp["momentum"],
                              weight_decay=hp["weight_decay"])
        for _ in range(hp["retrain_epoch"]):
            perm = torch.randperm(len(Y), device=DEV)
            for s in range(0, len(Y), hp["batch_size"]):
                b = perm[s:s + hp["batch_size"]]
                loss = F.cross_entropy(fc(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        sd = dict(glob[g])
        sd["fc.weight"], sd["fc.bias"] = fc.weight.detach().clone(), fc.bias.detach().clone()
        out[g] = sd
        del X, Y
    return out


def grouped_flops(S, meth, in_ch, seed):
    f_head = {}
    for g in S.groups:
        h = new_head(meth, g, in_ch, S.spec, S.C, seed).to(DEV)
        f_head[str(g)] = forward_flops(h, torch.zeros(2, in_ch, 16, 16, device=DEV))
    per_round = sum(3 * f_head[str(S.arch[k])] * 5 * int(S.fed.sizes[k]) for k in range(S.K))
    return dict(flops_head=f_head, head_stage_gflops_per_round=S.R.get("pfrac", 1.0) * per_round / 1e9)


def run_grouped(grid, context, R):
    mon = Monitor(os.path.join(R["out_dir"], "logs"), R["run_id"], tensorboard=R.get("tensorboard", True))
    t0 = time.time()
    S = Fed(grid, context, R, mon)
    meth = R["method"]["name"]
    spec = S.spec
    T = dict(T_FE=spec["T_FE"], T_KA=spec["T_KA"], T_TN=spec["T_TN"])
    T.update(R.get("rounds") or {})
    hp = dict(epochs=5, mu=MU) if meth == "FedProx" else dict(epochs=5)
    res = dict(class_order=S.fed.class_order, class_names=S.fed.class_names,
               test_class_counts=np.bincount(S.fed.y_te.cpu().numpy(), minlength=S.C).tolist(),
               run_id=R["run_id"], label=R["label"], dataset=R["ds"], IF=R["IF"], alpha=R["alpha"], K=R["K"],
               seed=R["seed"], pfrac=R.get("pfrac", 1.0), tail_perm=R.get("tail_perm", False), method=R["method"],
               n_train=S.N, min_client=int(min(S.fed.sizes)), client_counts=S.fed.counts.astype(int).tolist(),
               rounds=T, framework="flower", hyperparameters=hp, setting="grouped by architecture (GeFL)")
    mon.log("run_start", **{k: v for k, v in res.items() if k != "client_counts"})
    feat_key, s1 = stage_one(S, T["T_FE"])
    S.feat_key = feat_key
    res["fe_time_s"] = s1["time_s"]
    S.ask("query.features", [(S.nid[k], pack(S.base(fe_key=feat_key), fe=s1["fe_sd"])) for k in range(S.K)])
    ref, Xval = train_referee(S, s1["fe_sd"])
    in_ch = Xval.shape[1]
    base_cfg = S.base(fe_key=feat_key, method=meth, **hp)
    glob = {}   # architecture -> averaged head
    history = []
    mon.start("stage3")
    for r in range(T["T_TN"]):
        act = active_clients(S.K, R.get("pfrac", 1.0), R["seed"], "iii", r)
        ev = r == 0 or (r + 1) % EVAL_EVERY == 0 or r == T["T_TN"] - 1
        items = []
        for k in act:
            g = S.arch[k]
            if meth == "AvgKD":   # every architecture's head (the pseudo-labels use the other architectures')
                srv = {f"{gg}|{n}": v for gg, sd in glob.items() for n, v in sd.items()}
            else:
                srv = {f"{g}|{n}": v for n, v in glob.get(g, {}).items()}
            named = dict(srv=srv) if srv else {}
            if r == 0:
                named["init"] = s1["hd_sd"][g]
            items.append((S.nid[k], pack(dict(base_cfg, round=r, eval=ev), **named)))
        rep = S.ask("train.grouped", items)
        ups, infos = {}, []
        for m in rep:
            k = S.pid[m.metadata.src_node_id]
            ups.setdefault(S.arch[k], []).append(arrays_of(m.content, "up", DEV))
            infos.append((k, cfg_of(m.content)))
        for g, sds in ups.items():   # model_wise_FedAvg
            glob[g] = average(sds)
        rec = dict(round=r + 1, clients=act, loss={str(k): i["loss"] for k, i in infos})
        if ev:
            rec["pre"] = {str(k): i["pre"] for k, i in infos if "pre" in i}
            rec["post_clients"] = {str(k): i["post"] for k, i in infos if "post" in i}
        mon.log("stage3_round", **rec)
        history.append(rec)
    mon.stop("stage3")
    if meth == "CCVR":
        tc = time.time()
        glob = ccvr_calibrate(S, glob, in_ch, R["seed"])
        res["ccvr_time_s"] = time.time() - tc
    srv_all = {f"{g}|{n}": v for g, sd in glob.items() for n, v in sd.items()}
    rep = S.ask("evaluate.grouped", [(S.nid[k], pack(dict(base_cfg, probs=True, embed=(k == 0)),
                                                     srv={n: v for n, v in srv_all.items()
                                                          if n.startswith(f"{S.arch[k]}|")}))
                                     for k in range(S.K)], account=False)
    per, probs, emb, personal = {}, {}, None, {}
    for m in rep:
        k = S.pid[m.metadata.src_node_id]
        o = cfg_of(m.content)
        per[k] = o
        personal[k] = arrays_of(m.content, "model", "cpu")
        pr = arrays_of(m.content, "probs", "cpu")
        if pr is not None:
            probs[str(k)] = pr["p"].numpy()
        if "embed" in o:
            emb = (o["embed"], o["embed_y"])
    for split, prefix in (("test", "final"), ("val", "val_final")):
        res.update(flat(prefix, summarize({k: v[split] for k, v in per.items()})))
    res["final_per_client"] = {str(k): {kk: v["test"][kk] for kk in ("acc", "bal", "f1", "tail", "worst", "local")
                                        if kk in v["test"]}
                               for k, v in per.items()}
    res["time_s"] = time.time() - t0
    res["stage_time_s"] = dict(mon.stage_t)
    res["bytes"] = {k: dict(down=v[0], up=v[1], messages=v[2]) for k, v in mon.bytes.items()}
    res["comm_mb_per_client"] = sum(v[0] + v[1] for v in mon.bytes.values()) / 2 ** 20 / S.K
    res.update(gpu_peaks(S))
    res.update(grouped_flops(S, meth, in_ch, R["seed"]))
    fin = dict(test=dict(probs=probs, embed={"0": emb[0]} if emb else {}, embed_y=emb[1] if emb else []))
    extra = dict(y_full=S.fed.y_te.cpu().numpy())
    w0 = personal.get(0, {})
    if "model.fc.weight" in w0:   # client 0's classifier (its embeddings are the saved ones)
        extra["fcw_0"] = np.asarray(w0["model.fc.weight"], dtype=np.float32)
        extra["fcb_0"] = np.asarray(w0["model.fc.bias"], dtype=np.float32)
    save_run(R, res, s1, personal, None, ref, fin, None, None, None, extra)
    mon.log("run_end", time_s=res["time_s"])
    mon.close()
    return res
