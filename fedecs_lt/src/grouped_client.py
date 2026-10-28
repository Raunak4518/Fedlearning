"""Client side of the baselines GeFL compares in its grouped setting (Kang et al., IEEE TMC 2025, Sec. V-A: "FedAvg is
applied in grouped settings: clients are partitioned by model architecture"): FedAvg, FedProx, FedALA, AvgKD, and the
long-tail methods CCVR and FedETF applied the same way. Every client trains its own CNN-g head (GeFL-F's target
network) on the shared frozen Stage-I features; the server averages the heads within each architecture.

Sources of the local updates:
- FedAvg, FedProx, AvgKD: GeFL's code (github.com/honggkang/hetero-model-fl-gen, commit 2e92735, MIT licence),
  utils/localUpdateTarget.py: LocalUpdate, LocalUpdate_fedprox (proximal term mu / 2 * sum of per-tensor L2 norms of
  w - w_global, mu = 0.01), LocalUpdate_avgKD (from the second round, soft targets softmax(one-hot(y) + sum over the
  other architectures' models of softmax(logits)) on the client's own data); SGD lr 0.1, no momentum or decay.
- FedALA (Zhang et al., AAAI 2023): the ALA module of the authors' code (github.com/TsingZ0/FedALA, Apache-2.0),
  system/utils/ALA.py, with its defaults eta 1.0, rand_percent 80, layer_idx 2, threshold 0.1, num_pre_loss 10.
- FedETF (Li et al., ICCV 2023): LongTailLib's clientfedetf.py (balanced-softmax loss with the client's class
  counts, SGD lr 0.04, momentum 0.9, weight decay 5e-4) on models.ETFHead.
- CCVR (Luo et al., NeurIPS 2021): FedAvg training as above, then the per-class feature statistics for the server's
  calibration (LongTailLib serverccvr.py / clientccvr.py).
Local epochs: 5, GeFL's default (args.local_ep) and that of our head stage; batches of the dataset's batch size.
"""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from client import DEV, STATE, app, fed_of, feats_of, pid_of, spec_of
from data import batches
from fl_common import arrays_of, cfg_of, reply, seed_all
from metrics import class_metrics, log_view
from models import ETFHead, GroupedHead
from settings import HEADER_WIDTHS

GROUPED = ("FedAvg", "FedProx", "FedALA", "AvgKD", "CCVR", "FedETF")
ALA_HP = dict(eta=1.0, rand_percent=80, layer_idx=2, threshold=0.1, num_pre_loss=10, max_epochs=1000)


def new_head(meth, g, in_ch, spec, C, seed):
    if meth == "FedETF":
        return ETFHead(HEADER_WIDTHS[g], in_ch, spec["header_bn"], C, seed=seed)
    return GroupedHead(HEADER_WIDTHS[g], in_ch, spec["header_bn"], C)


def group_sd(srv, g):
    """Architecture g's averaged head from the server's message (keys 'g|name')."""
    p = f"{g}|"
    return {n[len(p):]: v for n, v in srv.items() if n.startswith(p)}


def state_of(cfg, k, fed, in_ch, init=None):
    """The client's persistent head (FedALA keeps its personal model and ALA weights between rounds)."""
    key = ("grouped", cfg["run_id"], k)
    if key not in STATE:
        g = fed.arch[k]
        seed_all(cfg["seed"], "grouped_init", k)
        m = new_head(cfg["method"], g, in_ch, spec_of(cfg), fed.C, cfg["seed"]).to(DEV)
        if init is not None:   # the architecture's Stage-I head (FedETF: its body only)
            sd = {n: v for n, v in init.items() if cfg["method"] != "FedETF" or n.startswith("features.")}
            m.load_state_dict(sd, strict=cfg["method"] != "FedETF")
        STATE[key] = dict(model=m, ala_weights=None, ala_start=True)
    return STATE[key]


def adaptive_local_aggregation(st, global_sd, X, Y, B, hp):
    """FedALA's ALA.adaptive_local_aggregation (authors' code, unchanged apart from tensors instead of a DataLoader
    and a cap on the first-round weight learning, which in the original loops until the losses settle)."""
    local_model = st["model"]
    global_model = copy.deepcopy(local_model)
    global_model.load_state_dict(global_sd)
    n = len(Y)
    rand_num = int(hp["rand_percent"] / 100 * n)
    rand_idx = random.randint(0, n - rand_num)
    Xr, Yr = X[rand_idx:rand_idx + rand_num], Y[rand_idx:rand_idx + rand_num]
    params_g, params = list(global_model.parameters()), list(local_model.parameters())
    if torch.sum(params_g[0] - params[0]) == 0:   # deactivate ALA at the first communication round
        return
    L = hp["layer_idx"]
    for param, param_g in zip(params[:-L], params_g[:-L]):   # lower layers: the global update
        param.data = param_g.data.clone()
    model_t = copy.deepcopy(local_model)
    params_t = list(model_t.parameters())
    params_p, params_gp, params_tp = params[-L:], params_g[-L:], params_t[-L:]
    for param in params_t[:-L]:
        param.requires_grad = False
    optimizer = torch.optim.SGD(params_tp, lr=0)
    if st["ala_weights"] is None:
        st["ala_weights"] = [torch.ones_like(p.data) for p in params_p]
    weights = st["ala_weights"]
    for param_t, param, param_g, w in zip(params_tp, params_p, params_gp, weights):
        param_t.data = param + (param_g - param) * w
    losses, cnt = [], 0
    while True:
        for s in range(0, len(Yr), B):
            x, y = Xr[s:s + B], Yr[s:s + B]
            optimizer.zero_grad()
            loss_value = F.cross_entropy(model_t(x), y)
            loss_value.backward()
            for param_t, param, param_g, w in zip(params_tp, params_p, params_gp, weights):
                w.data = torch.clamp(w - hp["eta"] * (param_t.grad * (param_g - param)), 0, 1)
            for param_t, param, param_g, w in zip(params_tp, params_p, params_gp, weights):
                param_t.data = param + (param_g - param) * w
        losses.append(loss_value.item())
        cnt += 1
        if not st["ala_start"]:
            break
        if len(losses) > hp["num_pre_loss"] and np.std(losses[-hp["num_pre_loss"]:]) < hp["threshold"]:
            break
        if cnt >= hp["max_epochs"]:
            break
    st["ala_start"] = False
    for param, param_t in zip(params_p, params_tp):
        param.data = param_t.data.clone()


@torch.no_grad()
def avgkd_targets(cfg, fed, k, X, Y, srv, in_ch):
    """GeFL LocalUpdate_avgKD: softmax(one-hot(y) + sum over the other architectures' models of their softmax)."""
    s = F.one_hot(Y, fed.C).float()
    own = fed.arch[k]
    spec = spec_of(cfg)
    for g in sorted({int(n.split("|", 1)[0]) for n in srv}):
        if g == own:
            continue
        h = new_head("AvgKD", g, in_ch, spec, fed.C, cfg["seed"]).to(DEV)
        h.load_state_dict(group_sd(srv, g))
        h.eval()
        s += torch.cat([F.softmax(h(X[i:i + 2048]), 1) for i in range(0, len(Y), 2048)])
    return F.softmax(s, 1)


def evaluate_model(model, X, y, C):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(X[s:s + 2048]) for s in range(0, len(X), 2048)])
    model.train()
    return class_metrics(logits, y, C)


@app.train("grouped")
def grouped_train(msg, context):
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed, spec = fed_of(cfg), spec_of(cfg)
    F_ = feats_of(cfg, fed)
    X, Y = F_["x"][k], fed.y[k]
    st = state_of(cfg, k, fed, X.shape[1], arrays_of(msg.content, "init", DEV))
    model, meth, C, B = st["model"], cfg["method"], fed.C, spec["B"]
    srv = arrays_of(msg.content, "srv", DEV) or {}
    own = group_sd(srv, fed.arch[k])
    seed_all(cfg["seed"], "grouped", cfg["round"], k)
    if own:
        if meth == "FedALA":
            adaptive_local_aggregation(st, own, X, Y, B, ALA_HP)
        else:
            model.load_state_dict(own)
    out = dict(n=len(Y))
    if cfg.get("eval"):   # model as received (the architecture's average; FedALA: after ALA)
        out["post"] = log_view(evaluate_model(model, F_["te"], fed.y_te, C))
    targets = avgkd_targets(cfg, fed, k, X, Y, srv, X.shape[1]) if meth == "AvgKD" and cfg["round"] >= 1 else None
    if meth == "FedETF":
        opt = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=5e-4)
        log_spc = torch.bincount(Y, minlength=C).float().clamp(min=1e-8).log()
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.0, weight_decay=0.0)
    w0 = [p.detach().clone() for p in model.parameters()] if meth == "FedProx" else None
    model.train()
    losses = []
    for _ in range(cfg["epochs"]):
        for b in batches(len(Y), B, DEV):
            if b.numel() < 2:
                continue
            logits = model(X[b])
            if targets is not None:
                loss = F.cross_entropy(logits, targets[b])
            elif meth == "FedETF":
                loss = F.cross_entropy(logits + log_spc, Y[b])
            else:
                loss = F.cross_entropy(logits, Y[b])
            if w0 is not None:
                loss = loss + cfg["mu"] / 2 * sum((p - p0).norm(2) for p, p0 in zip(model.parameters(), w0))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    out["loss"] = float(np.mean(losses)) if losses else None
    if cfg.get("eval"):
        out["pre"] = log_view(evaluate_model(model, F_["te"], fed.y_te, C))
    return reply(msg, out, up={n: v.detach() for n, v in model.state_dict().items()})


@app.query("eval_groups")
def eval_groups(msg, context):
    """Every architecture's averaged head on the test features (the aggregated models of a round), one client."""
    cfg = cfg_of(msg.content)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    srv = arrays_of(msg.content, "srv", DEV) or {}
    out = {}
    for g in sorted({int(n.split("|", 1)[0]) for n in srv}):
        h = new_head(cfg["method"], g, F_["te"].shape[1], spec_of(cfg), fed.C, cfg["seed"]).to(DEV)
        h.load_state_dict(group_sd(srv, g))
        out[str(g)] = log_view(evaluate_model(h, F_["te"], fed.y_te, fed.C))
    return reply(msg, dict(groups=out))


@app.query("ccvr_stats")
def ccvr_stats(msg, context):
    """CCVR: per-class count, sum and sum of outer products of the client's embeddings under its architecture's
    averaged head, for classes [c0, c1) (sent in class chunks so that no message is too large)."""
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    X, Y = F_["x"][k], fed.y[k]
    st = state_of(cfg, k, fed, X.shape[1])
    model = st["model"]
    model.load_state_dict(group_sd(arrays_of(msg.content, "srv", DEV), fed.arch[k]))
    model.eval()
    with torch.no_grad():
        E = torch.cat([model.embed(X[s:s + 2048]) for s in range(0, len(Y), 2048)]).double()
    model.train()
    c0, c1 = cfg["c0"], cfg["c1"]
    d = E.shape[1]
    n = torch.zeros(c1 - c0, dtype=torch.float64, device=DEV)
    S1 = torch.zeros(c1 - c0, d, dtype=torch.float64, device=DEV)
    S2 = torch.zeros(c1 - c0, d, d, dtype=torch.float64, device=DEV)
    for c in range(c0, c1):
        Ec = E[Y == c]
        n[c - c0], S1[c - c0], S2[c - c0] = len(Ec), Ec.sum(0), Ec.T @ Ec
    return reply(msg, dict(ok=True), s=dict(n=n, S1=S1, S2=S2.float()))


@app.evaluate("grouped")
def grouped_eval(msg, context):
    """Final evaluation: the architecture's averaged head (FedALA: the client's personal head; CCVR: the calibrated
    head the server sends), all metrics, probabilities and embeddings, as htfl_client.htfl_eval."""
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    st = state_of(cfg, k, fed, F_["te"].shape[1])
    model = st["model"]
    srv = arrays_of(msg.content, "srv", DEV) or {}
    if cfg["method"] != "FedALA":
        model.load_state_dict(group_sd(srv, fed.arch[k]))
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(F_["te"][s:s + 2048]) for s in range(0, len(fed.y_te), 2048)])
        lv = torch.cat([model(F_["val"][s:s + 2048]) for s in range(0, len(fed.y_val), 2048)])
    prior = fed.counts[k] / max(fed.counts[k].sum(), 1)
    m = class_metrics(logits, fed.y_te, fed.C, prior=prior, keep_probs=cfg.get("probs", False))
    pr = m.pop("probs", None)
    out = dict(test={kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in m.items()})
    mv = class_metrics(lv, fed.y_val, fed.C, prior=prior)
    out["val"] = {kk: v for kk, v in mv.items() if kk != "probs"}
    if cfg.get("embed"):
        sub = np.random.RandomState(0).choice(len(fed.y_te), min(2000, len(fed.y_te)), replace=False)
        with torch.no_grad():
            out["embed"] = model.embed(F_["te"][torch.from_numpy(sub).to(DEV)]).cpu().numpy().tolist()
        out["embed_y"] = fed.y_te.cpu().numpy()[sub].tolist()
    model.train()
    weights = {f"model.{n}": v for n, v in model.state_dict().items()}
    if pr is not None:
        return reply(msg, out, model=weights, probs=dict(p=torch.from_numpy(pr).half()))
    return reply(msg, out, model=weights)
