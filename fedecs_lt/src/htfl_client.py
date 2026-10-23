"""Client side of the model-heterogeneous FL baselines (Track 1): FedProto, FedTGP, FedGen, FedGH, LG-FedAvg, FedMRL.

Ported from HtFLlib (Zhang et al., KDD 2025; https://github.com/TsingZ0/HtFLlib, commit e959261, Apache License 2.0):
system/flcore/clients/client{proto,tgp,gen,gh,lg,mrl}.py. The local updates below follow those files line by line
(losses, what is uploaded); only data loading is replaced by this suite's frozen-feature clients. As in HtFLlib, every
client model is split into base (here: the client's head body from Stage I, followed by HtFLlib's parameter-free
AdaptiveAvgPool1d to the common feature dimension, 512) and head (Linear(512, C)); models are personal and persist on
the client across rounds.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from client import DEV, STATE, app, fed_of, feats_of, pid_of, spec_of
from fl_common import arrays_of, cfg_of, reply, seed_all
from metrics import class_metrics
from models import PaperHeader
from settings import HEADER_WIDTHS


class HtModel(nn.Module):
    """HtFLlib BaseHeadSplit for a head body: base = body + AdaptiveAvgPool1d(feature_dim), head = Linear."""

    def __init__(self, widths, in_ch, bn, C, feature_dim=512):
        super().__init__()
        body = PaperHeader(widths, in_ch, bn, C)
        self.body = body.features
        self.pool = nn.AdaptiveAvgPool1d(feature_dim)
        self.head = nn.Linear(feature_dim, C)

    def base(self, x):
        return self.pool(self.body(x).flatten(1))

    def forward(self, x):
        return self.head(self.base(x))


class Generator(nn.Module):
    """FedGen's feature generator as in HtFLlib servergen.py (based on the official FedGen code)."""

    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device):
        super().__init__()
        self.noise_dim, self.num_classes, self.device = noise_dim, num_classes, device
        self.fc1 = nn.Sequential(nn.Linear(noise_dim + num_classes, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        eps = torch.rand((labels.shape[0], self.noise_dim), device=self.device)
        z = torch.cat((eps, F.one_hot(labels, self.num_classes)), dim=1)
        return self.fc(self.fc1(z))


def model_of(cfg, k, fed, in_ch, init=None):
    """The client's persistent personal model (and FedMRL's projection), created from the Stage-I head body."""
    key = ("htfl", cfg["run_id"], k)
    if key not in STATE:
        spec = spec_of(cfg)
        g = fed.arch[k]
        seed_all(cfg["seed"], "htfl_init", k)
        m = HtModel(HEADER_WIDTHS[g], in_ch, spec["header_bn"], fed.C, cfg["feature_dim"]).to(DEV)
        if init is not None:  # body weights of the architecture's Stage-I head
            m.body.load_state_dict({kk[len("features."):]: v for kk, v in init.items() if kk.startswith("features.")})
        extra = {}
        if cfg["method"] == "FedMRL":
            extra["proj"] = nn.Linear(cfg["feature_dim"] + cfg["sub_feature_dim"], cfg["feature_dim"]).to(DEV)
        STATE[key] = dict(model=m, **extra)
    return STATE[key]


def agg_protos(reps, ys, C):
    """HtFLlib agg_func: mean representation per class present on the client."""
    protos, mask = torch.zeros(C, reps.shape[1], device=reps.device), torch.zeros(C, device=reps.device)
    for c in torch.unique(ys):
        protos[c] = reps[ys == c].mean(0)
        mask[c] = 1
    return protos, mask


@app.train("htfl")
def htfl_train(msg, context):
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    X, Y = F_["x"][k], fed.y[k]
    st = model_of(cfg, k, fed, X.shape[1], arrays_of(msg.content, "init", DEV))
    model = st["model"]
    meth, C = cfg["method"], fed.C
    srv = arrays_of(msg.content, "srv", DEV) or {}
    out = dict(n=len(Y), counts=np.bincount(Y.cpu().numpy(), minlength=C).tolist())
    # set_parameters: methods that share a head replace the local head with the server's (LG-FedAvg, FedGH, FedGen)
    if meth in ("LG-FedAvg", "FedGH", "FedGen") and "head.weight" in srv:
        model.head.weight.data.copy_(srv["head.weight"])
        model.head.bias.data.copy_(srv["head.bias"])
    if cfg.get("eval"):   # model as received (after the server's aggregation step), before local training
        out["post"] = evaluate_client(cfg, st, F_, fed, srv)
    seed_all(cfg["seed"], "htfl", cfg["round"], k)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"])
    global_protos = srv.get("protos") if meth in ("FedProto", "FedTGP") else None
    gmask = srv.get("protos_mask")
    gen = None
    if meth == "FedGen" and any(n.startswith("gen.") for n in srv):
        gen = Generator(cfg["noise_dim"], C, cfg["hidden_dim"], cfg["feature_dim"], DEV).to(DEV)
        gen.load_state_dict({n[len("gen."):]: v for n, v in srv.items() if n.startswith("gen.")})
        gen.eval()
        qualified = np.array(cfg["qualified_labels"])
    gmodel, proj = None, None
    if meth == "FedMRL":
        gmodel = HtModel(HEADER_WIDTHS[0], X.shape[1], spec_of(cfg)["header_bn"], C, cfg["sub_feature_dim"]).to(DEV)
        gmodel.load_state_dict({n[len("g."):]: v for n, v in srv.items() if n.startswith("g.")})
        gmodel.train()
        proj = st["proj"]
        opt_p = torch.optim.SGD(proj.parameters(), lr=cfg["lr"])
        opt_g = torch.optim.SGD(gmodel.parameters(), lr=cfg["lr"])
    loss_ce, loss_mse = nn.CrossEntropyLoss(), nn.MSELoss()
    proto_reps = []
    losses = []
    for _ in range(cfg["epochs"]):
        # HtFLlib clientbase.load_train_data: DataLoader(batch_size, drop_last=True, shuffle=False) over the client's
        # (already shuffled) data
        for s0 in range(0, len(Y) - cfg["B"] + 1, cfg["B"]):
            x, y = X[s0:s0 + cfg["B"]], Y[s0:s0 + cfg["B"]]
            if meth == "FedMRL":
                rep_g, rep = gmodel.base(x), model.base(x)
                rep_new = proj(torch.cat((rep_g, rep), dim=1))
                loss = loss_ce(model.head(rep_new), y) + loss_ce(gmodel.head(rep_new[:, :cfg["sub_feature_dim"]]), y)
                opt.zero_grad(); opt_p.zero_grad(); opt_g.zero_grad()  # noqa: E702
                loss.backward()
                opt.step(); opt_p.step(); opt_g.step()  # noqa: E702
                losses.append(float(loss.detach()))
                continue
            rep = model.base(x)
            loss = loss_ce(model.head(rep), y)
            if global_protos is not None:     # FedProto / FedTGP: MSE to the global prototype of each sample's class
                proto_new = rep.detach().clone()
                has = gmask[y] > 0
                proto_new[has] = global_protos[y[has]]
                loss = loss + loss_mse(proto_new, rep) * cfg["lamda"]
            if gen is not None:               # FedGen: CE of the head on generated features of qualified labels
                labels = torch.from_numpy(np.random.choice(qualified, cfg["B"])).long().to(DEV)
                z = gen(labels)
                loss = loss + loss_ce(model.head(z), labels)
            if meth == "FedProto":
                proto_reps.append((rep.detach(), y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    out["loss"] = float(np.mean(losses)) if losses else None
    up = {}
    if meth == "FedProto":   # FedProto: prototypes from the training-time representations
        reps = torch.cat([r for r, _ in proto_reps])
        ys = torch.cat([y for _, y in proto_reps])
        p, mk = agg_protos(reps, ys, C)
        up.update(protos=p, mask=mk)
    elif meth in ("FedTGP", "FedGH"):   # collect_protos: representations of the trained model, eval mode
        model.eval()
        with torch.no_grad():
            reps = torch.cat([model.base(X[s:s + 4096]) for s in range(0, len(Y), 4096)])
        p, mk = agg_protos(reps, Y, C)
        up.update(protos=p, mask=mk)
        model.train()
    if meth in ("LG-FedAvg", "FedGen"):
        up.update({"head.weight": model.head.weight.detach(), "head.bias": model.head.bias.detach()})
    if meth == "FedMRL":
        up.update({f"g.{n}": v.detach() for n, v in gmodel.state_dict().items()})
    if cfg.get("eval"):
        out["pre"] = evaluate_client(cfg, st, F_, fed, srv, gmodel=gmodel)
    return reply(msg, out, up=up)


@torch.no_grad()
def predict(cfg, st, X, srv, gmodel=None):
    """Logits of the client's model. FedProto / FedTGP classify by the nearest global prototype (HtFLlib
    test_metrics: -MSE distance as the score); FedMRL uses the projected joint representation."""
    model = st["model"]
    model.eval()
    meth = cfg["method"]
    if meth in ("FedProto", "FedTGP") and srv.get("protos") is not None:
        P, mk = srv["protos"], srv["protos_mask"]
        out = []
        for s in range(0, len(X), 2048):
            rep = model.base(X[s:s + 2048])
            d = ((rep[:, None, :] - P[None, :, :]) ** 2).mean(-1)
            d = torch.where(mk[None, :] > 0, d, torch.full_like(d, float("inf")))
            out.append(-d)
        model.train()
        return torch.cat(out)
    if meth == "FedMRL" and gmodel is None and any(n.startswith("g.") for n in srv):
        gmodel = HtModel(HEADER_WIDTHS[0], X.shape[1], spec_of(cfg)["header_bn"], model.head.out_features,
                         cfg["sub_feature_dim"]).to(DEV)
        gmodel.load_state_dict({n[len("g."):]: v for n, v in srv.items() if n.startswith("g.")})
    if meth == "FedMRL" and gmodel is not None:
        gmodel.eval()
        out = torch.cat([model.head(st["proj"](torch.cat((gmodel.base(X[s:s + 2048]), model.base(X[s:s + 2048])), 1)))
                         for s in range(0, len(X), 2048)])
        model.train()
        return out
    out = torch.cat([model(X[s:s + 2048]) for s in range(0, len(X), 2048)])
    model.train()
    return out


def evaluate_client(cfg, st, F_, fed, srv, gmodel=None):
    logits = predict(cfg, st, F_["te"], srv, gmodel)
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9))
    m = class_metrics(logits, fed.y_te, fed.C)
    return {kk: m[kk] for kk in ("acc", "bal", "f1", "tail", "worst", "recall_c")}


@app.evaluate("htfl")
def htfl_eval(msg, context):
    """Final evaluation of the client's personal model on the test set (all metrics, probabilities, embeddings)."""
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    st = STATE[("htfl", cfg["run_id"], k)]
    srv = arrays_of(msg.content, "srv", DEV) or {}
    logits = predict(cfg, st, F_["te"], srv)
    logits = torch.where(torch.isfinite(logits), logits, torch.full_like(logits, -1e9))
    out = {}
    m = class_metrics(logits, fed.y_te, fed.C, keep_probs=cfg.get("probs", False))
    out["test"] = {kk: (v.tolist() if hasattr(v, "tolist") else v) for kk, v in m.items()}
    lv = predict(cfg, st, F_["val"], srv)
    lv = torch.where(torch.isfinite(lv), lv, torch.full_like(lv, -1e9))
    mv = class_metrics(lv, fed.y_val, fed.C)
    out["val"] = {kk: v for kk, v in mv.items() if kk != "probs"}
    if cfg.get("embed"):
        sub = np.random.RandomState(0).choice(len(fed.y_te), min(2000, len(fed.y_te)), replace=False)
        with torch.no_grad():
            st["model"].eval()
            out["embed"] = st["model"].base(F_["te"][torch.from_numpy(sub).to(DEV)]).cpu().numpy().tolist()
            st["model"].train()
        out["embed_y"] = fed.y_te.cpu().numpy()[sub].tolist()
    weights = {f"model.{n}": v for n, v in st["model"].state_dict().items()}
    if "proj" in st:
        weights.update({f"proj.{n}": v for n, v in st["proj"].state_dict().items()})
    return reply(msg, out, model=weights)
