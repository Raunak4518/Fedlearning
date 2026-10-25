"""Server side of the model-heterogeneous FL baselines (Track 1), ported from HtFLlib (Apache License 2.0):
system/flcore/servers/server{proto,tgp,gen,gh,lg,mrl}.py. Every method starts from the same Stage-I extractor and head
bodies as FedECS-LT (cached), trains for the same number of rounds and local epochs as our head stage, and uses the
hyperparameters of the authors' run scripts (HtFLlib README / total.sh): SGD lr 0.01, batch 10, feature dimension 512;
FedProto and FedTGP lambda 10; FedTGP server epochs 100, margin threshold 100; FedGen noise 32, hidden 512, generator
lr 0.1, server epochs 100; FedGH server lr 0.01, server epochs 1; FedMRL sub-feature dimension 128."""
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import active_clients
from fl_common import arrays_of, cfg_of, pack, seed_all
from metrics import summarize
from monitor import Monitor
from server import DEV, EVAL_EVERY, Fed, flat, gpu_peaks, save_run, stage_one, train_referee
from settings import HEADER_WIDTHS

HP = dict(lr=0.01, B=10, feature_dim=512)
METHOD_HP = {
    "FedProto": dict(lamda=10.0),
    "FedTGP": dict(lamda=10.0, server_epochs=100, margin_threthold=100.0),
    "FedGen": dict(noise_dim=32, hidden_dim=512, generator_learning_rate=0.1, server_epochs=100),
    "FedGH": dict(server_learning_rate=0.01, server_epochs=1),
    "LG-FedAvg": dict(),
    "FedMRL": dict(sub_feature_dim=128),
}


class TrainableGlobalPrototypes(nn.Module):
    """FedTGP's Trainable_Global_Prototypes (HtFLlib servertgp.py)."""

    def __init__(self, num_classes, server_hidden_dim, feature_dim):
        super().__init__()
        self.embedings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU()))
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        return self.fc(self.middle(self.embedings(class_id)))


def proto_cluster(protos, masks):
    """Unweighted mean over clients of each class's prototype (HtFLlib proto_aggregation / proto_cluster)."""
    P = torch.stack(protos)
    M = torch.stack(masks)
    cnt = M.sum(0)
    avg = (P * M[:, :, None]).sum(0) / cnt.clamp(min=1)[:, None]
    return avg, (cnt > 0).float()


def run_htfl(grid, context, R):
    from htfl_client import Generator, HtModel
    mon = Monitor(os.path.join(R["out_dir"], "logs"), R["run_id"], tensorboard=R.get("tensorboard", True))
    t0 = time.time()
    S = Fed(grid, context, R, mon)
    meth = R["method"]["name"]
    spec = S.spec
    T = dict(T_FE=spec["T_FE"], T_KA=spec["T_KA"], T_TN=spec["T_TN"])
    T.update(R.get("rounds") or {})
    hp = dict(HP, **METHOD_HP[meth], epochs=5)
    res = dict(class_order=S.fed.class_order, class_names=S.fed.class_names,
               test_class_counts=np.bincount(S.fed.y_te.cpu().numpy(), minlength=S.C).tolist(),
               run_id=R["run_id"], label=R["label"], dataset=R["ds"], IF=R["IF"], alpha=R["alpha"], K=R["K"],
               seed=R["seed"], pfrac=R.get("pfrac", 1.0), tail_perm=R.get("tail_perm", False), method=R["method"],
               n_train=S.N, min_client=int(min(S.fed.sizes)), client_counts=S.fed.counts.astype(int).tolist(),
               rounds=T, framework="flower", hyperparameters=hp)
    mon.log("run_start", **{k: v for k, v in res.items() if k != "client_counts"})
    feat_key, s1 = stage_one(S, T["T_FE"])
    res["fe_time_s"] = s1["time_s"]
    S.ask("query.features", [(S.nid[k], pack(S.base(fe_key=feat_key), fe=s1["fe_sd"])) for k in range(S.K)])
    ref, Xval = train_referee(S, s1["fe_sd"])
    in_ch = Xval.shape[1]
    C = S.C
    seed_all(R["seed"], "htfl_server")
    srv = {}
    TGP = gen = gmodel = None
    qualified = None
    if meth == "FedTGP":
        TGP = TrainableGlobalPrototypes(C, hp["feature_dim"], hp["feature_dim"]).to(DEV)
    if meth == "FedGen":
        gen = Generator(hp["noise_dim"], C, hp["hidden_dim"], hp["feature_dim"], DEV).to(DEV)
        # qualified labels: every client's samples per class (HtFLlib FedGen.__init__ reads them from the clients)
        counts = S.fed.counts.sum(0)
        qualified = [int(c) for c in range(C) for _ in range(int(counts[c]))]
        srv.update({f"gen.{n}": v for n, v in gen.state_dict().items()})
    if meth in ("LG-FedAvg", "FedGH", "FedGen"):
        seed_all(R["seed"], "htfl_init", 0)   # HtFLlib: the server's head starts as client 0's initial head
        h0 = HtModel(HEADER_WIDTHS[S.arch[0]], in_ch, spec["header_bn"], C, hp["feature_dim"]).to(DEV)
        srv.update({"head.weight": h0.head.weight.detach(), "head.bias": h0.head.bias.detach()})
    if meth == "FedMRL":
        gmodel = HtModel(HEADER_WIDTHS[0], in_ch, spec["header_bn"], C, hp["sub_feature_dim"]).to(DEV)
        srv.update({f"g.{n}": v for n, v in gmodel.state_dict().items()})
    base_cfg = S.base(fe_key=feat_key, method=meth, **hp)
    if qualified is not None:
        base_cfg["qualified_labels"] = qualified
    history = []
    mon.start("stage3")
    for r in range(T["T_TN"]):
        act = active_clients(S.K, R.get("pfrac", 1.0), R["seed"], "iii", r)
        ev = r == 0 or (r + 1) % EVAL_EVERY == 0 or r == T["T_TN"] - 1
        items = []
        for k in act:
            named = dict(srv=srv) if srv else {}
            if r == 0:
                named["init"] = s1["hd_sd"][S.arch[k]]
            items.append((S.nid[k], pack(dict(base_cfg, round=r, eval=ev), **named)))
        rep = S.ask("train.htfl", items)
        ups, infos = [], []
        for m in rep:
            k = S.pid[m.metadata.src_node_id]
            ups.append((k, arrays_of(m.content, "up", DEV)))
            infos.append((k, cfg_of(m.content)))
        w = np.array([i["n"] for _, i in infos], dtype=float)
        w = w / w.sum()
        rec = dict(round=r + 1, clients=act, loss={str(k): i["loss"] for k, i in infos})
        if meth in ("FedProto", "FedTGP", "FedGH"):
            protos = [u["protos"] for _, u in ups]
            masks = [u["mask"] for _, u in ups]
        if meth == "FedProto":
            P, M = proto_cluster(protos, masks)
            srv = dict(protos=P, protos_mask=M)
        elif meth == "FedTGP":
            avg, Mk = proto_cluster(protos, masks)
            # class-wise minimum distance between averaged prototypes (receive_protos)
            gap = torch.full((C,), 1e9, device=DEV)
            ks = [c for c in range(C) if Mk[c] > 0]
            for a in ks:
                for b in ks:
                    if a > b:
                        dis = torch.norm(avg[a] - avg[b], p=2)
                        gap[a] = torch.min(gap[a], dis)
                        gap[b] = torch.min(gap[b], dis)
            min_gap = torch.min(gap)
            gap = torch.where(gap > 1e8, min_gap, gap)
            max_gap = torch.max(gap)
            data = [(p[c], c) for p, mk in zip(protos, masks) for c in range(C) if mk[c] > 0]
            opt = torch.optim.SGD(TGP.parameters(), lr=hp["lr"])
            TGP.train()
            for _ in range(hp["server_epochs"]):
                for proto, y in DataLoader(data, hp["B"], drop_last=False, shuffle=True):
                    y = torch.as_tensor(y, dtype=torch.int64, device=DEV)
                    proto_gen = TGP(torch.arange(C, device=DEV))
                    dist = (proto.pow(2).sum(1, keepdim=True) - 2 * proto @ proto_gen.T
                            + proto_gen.pow(2).sum(1, keepdim=True).T).sqrt()
                    margin = min(max_gap.item(), hp["margin_threthold"])
                    dist = dist + F.one_hot(y, C) * margin
                    loss = F.cross_entropy(-dist, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            TGP.eval()
            with torch.no_grad():
                srv = dict(protos=TGP(torch.arange(C, device=DEV)).detach(), protos_mask=torch.ones(C, device=DEV))
            rec["server_loss"] = float(loss)
        elif meth == "FedGH":
            data = [(p[c], c) for p, mk in zip(protos, masks) for c in range(C) if mk[c] > 0]
            head = nn.Linear(hp["feature_dim"], C).to(DEV)
            head.weight.data.copy_(srv["head.weight"])
            head.bias.data.copy_(srv["head.bias"])
            opt = torch.optim.SGD(head.parameters(), lr=hp["server_learning_rate"])
            for _ in range(hp["server_epochs"]):
                for p, y in DataLoader(data, hp["B"], drop_last=False, shuffle=True):
                    loss = F.cross_entropy(head(p), torch.as_tensor(y, dtype=torch.int64, device=DEV))
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            srv = {"head.weight": head.weight.detach(), "head.bias": head.bias.detach()}
        elif meth in ("LG-FedAvg", "FedGen"):
            heads = [{n: u[n] for n in ("head.weight", "head.bias")} for _, u in ups]
            if meth == "FedGen":   # train_generator on the uploaded heads (weights by training samples), then average
                gen.train()
                gopt = torch.optim.Adam(gen.parameters(), lr=hp["generator_learning_rate"], betas=(0.9, 0.999),
                                        eps=1e-08, weight_decay=0, amsgrad=False)
                lin = []
                for h in heads:
                    lm = nn.Linear(hp["feature_dim"], C).to(DEV)
                    lm.weight.data.copy_(h["head.weight"])
                    lm.bias.data.copy_(h["head.bias"])
                    lm.eval()
                    lin.append(lm)
                for _ in range(hp["server_epochs"]):
                    labels = torch.from_numpy(np.random.choice(qualified, hp["B"])).long().to(DEV)
                    z = gen(labels)
                    logits = sum(lm(z) * wi for lm, wi in zip(lin, w))
                    loss = F.cross_entropy(logits, labels)
                    gopt.zero_grad()
                    loss.backward()
                    gopt.step()
                rec["server_loss"] = float(loss)
            agg = {n: sum(h[n] * wi for h, wi in zip(heads, w)) for n in ("head.weight", "head.bias")}
            srv = dict(agg)
            if meth == "FedGen":
                srv.update({f"gen.{n}": v for n, v in gen.state_dict().items()})
        elif meth == "FedMRL":
            gs = [{n: u[n] for n in u if n.startswith("g.")} for _, u in ups]
            srv = {n: sum(g[n].float() for g in gs) / len(gs) if gs[0][n].is_floating_point() else gs[-1][n]
                   for n in gs[0]}
        if ev:
            rec["pre"] = {str(k): i["pre"] for k, i in infos if "pre" in i}
            rec["post_clients"] = {str(k): i["post"] for k, i in infos if "post" in i}
            pre = [i["pre"] for _, i in infos if "pre" in i]
            if pre:   # mean over the clients' personal models of every scalar and every per-class metric
                rec["post"] = {}
                for k in pre[0]:
                    vals = [p[k] for p in pre if p.get(k) is not None]
                    if not vals:
                        continue
                    if isinstance(vals[0], list):
                        rec["post"][k] = [round(float(x), 4) for x in np.nanmean(np.array(vals, dtype=float), 0)]
                    else:
                        rec["post"][k] = round(float(np.mean(vals)), 4)
        mon.log("stage3_round", **rec)
        history.append(rec)
    mon.stop("stage3")
    # final evaluation: each client's personal model on the test and validation sets
    rep = S.ask("evaluate.htfl", [(S.nid[k], pack(dict(base_cfg, probs=True, embed=(k == 0)), srv=srv))
                                  for k in range(S.K)], account=False)
    per = {}
    probs, emb = {}, None
    personal = {}
    for m in rep:
        k = S.pid[m.metadata.src_node_id]
        o = cfg_of(m.content)
        per[k] = o
        personal[k] = arrays_of(m.content, "model", "cpu")
        if "probs" in o["test"]:
            probs[str(k)] = o["test"].pop("probs")
        if "embed" in o:
            emb = (o["embed"], o["embed_y"])
    for split, prefix in (("test", "final"), ("val", "val_final")):
        summ = summarize({k: v[split] for k, v in per.items()})
        res.update(flat(prefix, summ))
    res["final_per_client"] = {str(k): {kk: v["test"][kk] for kk in ("acc", "bal", "f1", "tail", "worst")}
                               for k, v in per.items()}
    res["time_s"] = time.time() - t0
    res["stage_time_s"] = dict(mon.stage_t)
    res["bytes"] = {k: dict(down=v[0], up=v[1], messages=v[2]) for k, v in mon.bytes.items()}
    res["comm_mb_per_client"] = sum(v[0] + v[1] for v in mon.bytes.values()) / 2 ** 20 / S.K
    res.update(gpu_peaks(S))
    fin = dict(test=dict(probs={k: v for k, v in probs.items()}, embed={"0": emb[0]} if emb else {},
                         embed_y=emb[1] if emb else []))
    extra = dict(y_full=S.fed.y_te.cpu().numpy())
    save_run(R, res, s1, personal, None, ref, fin, None, None, None, extra)
    mon.log("run_end", time_s=res["time_s"])
    mon.close()
    return res
