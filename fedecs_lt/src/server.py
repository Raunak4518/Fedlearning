"""Flower ServerApp: orchestrates one run of the GeFL-F pipeline family (GeFL-F, its variants, the controls and
FedECS-LT) and the baselines that share it. The run is described by the module-level dict RUN (set by runner.py).

Stage I    the feature extractor (FE) is averaged over all clients, each head within its architecture; then frozen.
Release    (FedECS-LT) per-class count, sum h, sum ||h||^2 by secure aggregation -> exact means, spreads, gate F.
Stage II   the feature generator: averaged (FedAvg); with the class-row branch, rows by class-count weighted
           averaging (CCWA, secure sums); with the anchor branch, conditioned on the exact class means.
Stage III  heads trained on generated features (moment-corrected, consensus soft labels for FedECS-LT), then real
           ones (logit-adjusted for FedECS-LT); averaged within each architecture.
Readout    exact class-balanced readout from secure sums of every client's embeddings (reported separately).
Clients evaluate every model the server sends (before and after aggregation); the server collects.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from flwr.app import Context, Message
from flwr.serverapp import Grid, ServerApp

import fedecs
import local_train as lt
from data import Federation, active_clients
from fl_common import Averager, arrays_of, cfg_of, key_of, nbytes, pack, seed_all
from flops import forward_flops
from metrics import PER_CLASS, log_view, summarize
from models import (GaussFeatureGen, PaperHeader, make_fe, make_generator, make_head, make_headers, n_params,
                    sd_clone)
from monitor import Monitor
from secure_sum import secure_sum
from settings import DATASETS, EVAL_EVERY, FEDECS, HEADER_WIDTHS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUN = {}
app = ServerApp()


class Fed:
    """Server-side view of the federation and the messaging helpers (with byte accounting)."""

    def __init__(self, grid, context, R, mon):
        self.grid, self.context, self.R, self.mon = grid, context, R, mon
        self.spec = DATASETS[R["ds"]]
        t0 = time.time()
        while len(list(grid.get_node_ids())) < R["K"]:
            if time.time() - t0 > 600:
                raise RuntimeError("clients did not register")
            time.sleep(0.2)
        nodes = sorted(grid.get_node_ids())
        if DEV == "cuda":   # a new run: the server's GPU memory peak is measured from here
            torch.cuda.reset_peak_memory_stats()
        rep = self.ask("query.hello", [(n, pack(self.base())) for n in nodes], count=False)
        self.nid = {cfg_of(m.content)["pid"]: m.metadata.src_node_id for m in rep}
        self.pid = {v: k for k, v in self.nid.items()}
        self.fed = Federation(R["ds"], R["data_root"], R["IF"], R["alpha"], R["K"], R["seed"], DEV,
                              tail_perm=R.get("tail_perm", False), frac_scale=R.get("frac_scale", 1.0), clients=False)
        self.C, self.K, self.arch, self.groups = self.fed.C, R["K"], self.fed.arch, self.fed.groups
        self.N = int(sum(self.fed.sizes))

    def base(self, **kw):
        R = self.R
        d = dict(ds=R["ds"], IF=R["IF"], alpha=R["alpha"], K=R["K"], seed=R["seed"], tail_perm=R.get("tail_perm", False),
                 frac_scale=R.get("frac_scale", 1.0), data_root=R["data_root"], run_id=R["run_id"])
        d.update(kw)
        return d

    def ask(self, mtype, items, count=True, account=True):
        """Send (node, content) pairs, return replies sorted by client id; raise on any client error."""
        msgs = [Message(content=c, dst_node_id=n, message_type=mtype) for n, c in items]
        down = sum(nbytes(c) for _, c in items)
        rep = list(self.grid.send_and_receive(msgs))
        for m in rep:
            if m.has_error():
                raise RuntimeError(f"{mtype}: client error {m.error.reason[:2000]}")
        if count and account:
            self.mon.count(mtype, down, sum(nbytes(m.content) for m in rep), len(items))
        if hasattr(self, "pid"):
            rep.sort(key=lambda m: self.pid[m.metadata.src_node_id])
        return rep

    def to(self, ks):
        return [self.nid[k] for k in ks]

    def ssum(self, cfg, bounds, clients, round_no=1, secure=True):
        """Sum over clients of their contributions: SecAgg+ when secure (FedECS-LT releases), else plain."""
        n_el = None
        if secure and len(clients) >= 3:
            out = secure_sum(self.grid, self.context, self.to(clients), dict(self.base(), **cfg), bounds, round_no)
            n_el = sum(a.size for a in out)
            self.mon.count("secsum." + cfg["what"], 0, 8 * n_el * len(clients), len(clients))
            return [torch.from_numpy(a).to(DEV) for a in out]
        rep = self.ask("query.sums", [(self.nid[k], pack(dict(self.base(), **cfg))) for k in clients],
                       account=False)
        arrs = None
        for m in rep:
            a = arrays_of(m.content, "s", DEV)
            a = [a[str(i)].double() for i in range(len(a))]
            arrs = a if arrs is None else [x + y for x, y in zip(arrs, a)]
        self.mon.count("sums." + cfg["what"], 0, 8 * sum(int(x.numel()) for x in arrs) * len(clients), len(clients))
        return arrs


# ---------------------------------------------------------------- helpers
def heads_rec(hd_sd, groups):
    return {f"{g}|{k}": v for g in groups for k, v in hd_sd[g].items()}


def eval_heads(S, hd_sd, feat_key, splits=("test",), probs=False, embed=False, lastlayer=None, tag=""):
    """Clients evaluate, the server collects: the first client evaluates every architecture's head."""
    named = dict(heads=heads_rec(hd_sd, S.groups))
    if lastlayer:
        named["lastlayer"] = lastlayer
    rec = pack(S.base(fe_key=feat_key, splits=list(splits), probs=probs, embed=embed), **named)
    rep = S.ask("evaluate.heads", [(S.nid[0], rec)], account=False)
    return cfg_of(rep[0].content)


def flat(prefix, summ):
    keep = ("acc", "bal", "precision", "f1", "mcc", "kappa", "gmean", "worst", "worst_arch", "head", "medium", "tail",
            "rec_std", "auc_roc", "auc_pr", "top5", "ece", "cw_ece", "nll", "brier", "local", "ens_acc", "ens_bal")
    out = {f"{prefix}_{k}": summ[k] for k in keep if k in summ}
    for k in PER_CLASS:
        if k in summ:
            out[f"{prefix}_{k}"] = [round(float(v), 4) for v in summ[k]]
    if "confusion" in summ:
        out[f"{prefix}_confusion"] = summ["confusion"]
    return out


def cache_path(R, kind, key):
    d = os.path.join(R["cache_dir"], kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".pt")


def claim(path, wait_s=8 * 3600):
    """Parallel workers sharing a stage: the first claims it (lock file holding its process id) and trains it; the
    others wait for its result instead of training the same stage twice. A lock whose process has died is removed.
    Returns True if this process must train the stage."""
    lock = path + ".lock"
    t0 = time.time()
    while True:
        if os.path.exists(path):
            return False
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(lock).read() or 0)
                os.kill(pid, 0)
                alive = True
            except (ValueError, ProcessLookupError, FileNotFoundError):
                alive = False
            except PermissionError:
                alive = True
            if not alive or time.time() - t0 > wait_s:
                try:
                    os.remove(lock)
                except FileNotFoundError:
                    pass
                continue
            time.sleep(5)


def release(path):
    try:
        os.remove(path + ".lock")
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------- stage I
def _bytes_delta(before, after):
    return {k: [a - b for a, b in zip(v, before.get(k, [0, 0, 0]))] for k, v in after.items()
            if v != before.get(k)}


def _add_bytes(mon, delta):
    """A stage loaded from cache still counts its communication (it was paid once per run)."""
    for k, (d, u, n) in (delta or {}).items():
        mon.count(k, d, u, n)


def stage_one(S, T_FE, la=False):
    R, spec, mon = S.R, S.spec, S.mon
    key = key_of("stage1", R["ds"], R["IF"], R["alpha"], R["K"], R["seed"], R.get("tail_perm", False),
                 R.get("frac_scale", 1.0), R.get("pfrac", 1.0), T_FE, la)
    path = cache_path(R, "stage1", key)
    if not claim(path):
        c = torch.load(path, weights_only=False)
        _add_bytes(mon, c.get("bytes"))
        mon.log("stage1_cached", key=key, time_s=c["time_s"])
        for rec in c["history"]:
            mon.log("stage1_round", **rec)
        return key, c
    mon.start("stage1")
    b0 = {k: list(v) for k, v in mon.bytes.items()}
    t0 = time.time()
    torch.manual_seed(R["seed"])   # same initial weights as core.py (set_seed(seed) before PaperFE, build_headers)
    np.random.seed(R["seed"])
    fe = make_fe(spec)
    headers = make_headers(spec, fe.out_ch, S.C, "cpu")
    fe_sd = sd_clone(fe)
    hd_sd = [sd_clone(h) for h in headers]
    history = []
    for r in range(T_FE):
        act = active_clients(S.K, R.get("pfrac", 1.0), R["seed"], "i", r)
        ev = r == 0 or (r + 1) % EVAL_EVERY == 0 or r == T_FE - 1
        items = [(S.nid[k], pack(S.base(round=r, la=la, eval=ev), fe=fe_sd, head=hd_sd[S.arch[k]])) for k in act]
        rep = S.ask("train.stage1", items)
        fe_avg, hd_avg = Averager(), {g: Averager() for g in S.groups}
        locs, infos = [], []
        for m in rep:
            k = S.pid[m.metadata.src_node_id]
            f_sd, h_sd = arrays_of(m.content, "fe"), arrays_of(m.content, "head")
            fe_avg.add(f_sd)
            hd_avg[S.arch[k]].add(h_sd)
            locs.append(f_sd)
            infos.append((k, cfg_of(m.content)))
        new_fe = fe_avg.result()
        # drift: distance of each client's FE from the new average (client drift), update norm of the average
        drift = [float(sum((l_[n] - new_fe[n]).float().pow(2).sum() for n in new_fe if new_fe[n].is_floating_point()) ** 0.5)
                 for l_ in locs]
        upd = float(sum((new_fe[n] - fe_sd[n]).float().pow(2).sum() for n in new_fe if new_fe[n].is_floating_point()) ** 0.5)
        bn_div = [float(torch.stack([l_[n].float() for l_ in locs]).var(0).mean()) for n in new_fe if "running_var" in n]
        fe_sd = new_fe
        for g in S.groups:
            if hd_avg[g].n:
                hd_sd[g] = hd_avg[g].result()
        rec = dict(round=r + 1, clients=act, loss={str(k): i["loss"] for k, i in infos}, drift=drift, update_norm=upd,
                   bn_divergence=bn_div, time={str(k): i["time"] for k, i in infos})
        if ev:
            rec["pre"] = {str(k): i["pre"] for k, i in infos if "pre" in i}
            post = S.ask("evaluate.stage1", [(S.nid[0], pack(S.base(), fe=fe_sd, heads=heads_rec(hd_sd, S.groups)))],
                         account=False)
            rec["post"] = log_view(cfg_of(post[0].content)["summary"])
        mon.log("stage1_round", **rec)
        history.append(rec)
    c = dict(fe_sd=fe_sd, hd_sd=hd_sd, history=history, time_s=time.time() - t0, bytes=_bytes_delta(b0, mon.bytes))
    torch.save(c, path + ".part")
    os.replace(path + ".part", path)
    release(path)
    mon.stop("stage1")
    return key, c


def train_referee(S, fe_sd):
    """Server-side referee on the validation set (public server data), as core.train_referee: CNN-5 with BN,
    Adam 1e-3, 40 epochs, batch 64. Used only to measure the generator's per-class fidelity."""
    spec = S.spec
    fe = make_fe(spec).to(DEV)
    fe.load_state_dict(fe_sd)
    fe.eval()
    with torch.no_grad():
        X = torch.cat([fe(S.fed.x_val[s:s + 4000]) for s in range(0, len(S.fed.y_val), 4000)])
    Y = S.fed.y_val
    seed_all(S.R["seed"], "referee")
    ref = PaperHeader(HEADER_WIDTHS[4], X.shape[1], True, S.C).to(DEV)
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
    for _ in range(40):
        perm = torch.randperm(len(Y), device=DEV)
        for s in range(0, len(Y), 64):
            b = perm[s:s + 64]
            loss = F.cross_entropy(ref(X[b]), Y[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    ref.eval()
    return ref, X


# ---------------------------------------------------------------- statistics release
def release_stats(S, feat_key, Xval):
    """Secure sum of (n_c, sum h, sum ||h||^2); public bounds from the federation size and the validation features."""
    N = S.N
    fmax = float(Xval.abs().max()) * 4 + 1e-6
    smax = float(Xval.flatten(1).pow(2).sum(1).max()) * 4 + 1e-6
    bounds = [N / 4.0, N * fmax / 4.0, N * smax / 4.0]
    n, S1, S2 = S.ssum(dict(what="stats", fe_key=feat_key), bounds, list(range(S.K)), secure=True)
    n = n.round()
    return n, S1, S2


# ---------------------------------------------------------------- stage II
def stage_two(S, gspec, feat_key, shape, T_KA, ref, proto=None, stats=None):
    R, mon = S.R, S.mon
    key = key_of("stage2", feat_key, gspec, T_KA, R.get("pfrac", 1.0),
                 None if proto is None else float(proto.double().sum()))
    path = cache_path(R, "stage2", key)
    if gspec["type"] == "gauss":
        n, Ssum, M = S.ssum(dict(what="gram", fe_key=feat_key),
                            [S.N / 4.0, S.N * 1e3, S.N * 1e6], list(range(S.K)), secure=False)
        mu, chol = fedecs.gaussian_from_sums(n, Ssum, M, shape)
        G = GaussFeatureGen(mu, chol, shape).to(DEV)
        return key, dict(g_sd=dict(mu=mu, chol=chol), G=G, history=[], time_s=0.0)
    seed_all(R["seed"], "gen_init")
    torch.manual_seed(R["seed"] + 1000)   # core.py: set_seed(seed + 1000) before make_gen
    G = make_generator(gspec["type"], shape[0], S.spec["latent"], S.C, proto)
    if not claim(path):
        c = torch.load(path, weights_only=False)
        G.load_state_dict(c["g_sd"])
        G = G.to(DEV)
        _add_bytes(mon, c.get("bytes"))
        mon.log("stage2_cached", key=key, time_s=c["time_s"])
        for rec in c["history"]:
            mon.log("stage2_round", **rec)
        c["G"] = G
        return key, c
    mon.start("stage2")
    b0 = {k: list(v) for k, v in mon.bytes.items()}
    t0 = time.time()
    g_sd = sd_clone(G)
    G = G.to(DEV)
    ccwa = gspec.get("agg") == "ccwa"
    history = []
    ref_rec = {k: v for k, v in ref.state_dict().items()}
    rnorm = lambda sd: sd["label_proj"].float().norm(dim=1).tolist() if "label_proj" in sd else None  # noqa: E731
    for r in range(T_KA):
        act = active_clients(S.K, R.get("pfrac", 1.0), R["seed"], "ii", r)
        ev = r == 0 or (r + 1) % EVAL_EVERY == 0 or r == T_KA - 1
        cfg = S.base(round=r, fe_key=feat_key, gen=gspec, gen_key=key, send_rows=not ccwa)
        extra = dict(ref=ref_rec) if ev else {}
        items = [(S.nid[k], pack(dict(cfg, eval=ev), g=g_sd, **extra)) for k in act]
        rep = S.ask("train.gen", items)
        avg, infos = Averager(), []
        for m in rep:
            k = S.pid[m.metadata.src_node_id]
            avg.add(arrays_of(m.content, "g"))
            infos.append((k, cfg_of(m.content)))
        new = avg.result()
        if ccwa:
            bound_rows = float(g_sd["label_proj"].abs().max()) * 20 + 1.0
            rs, ws = S.ssum(dict(what="rows", gen_key=key, beta=gspec.get("beta", FEDECS["ccwa_beta"])),
                            [1000.0 * bound_rows / 4.0, 1000.0 / 4.0], act, round_no=r + 1, secure=True)
            new["label_proj"] = fedecs.ccwa_rows(rs.float(), ws.float(), g_sd["label_proj"].to(DEV)).cpu()
        g_sd = {k: v.cpu() for k, v in new.items()}
        rec = dict(round=r + 1, clients=act, loss={str(k): i["loss"] for k, i in infos},
                   row_norm_global=rnorm(g_sd), time={str(k): i["time"] for k, i in infos})
        if any("row_norm" in i for _, i in infos):
            rec["row_norm_local"] = {str(k): i["row_norm"] for k, i in infos if "row_norm" in i}
            rec["row_v"] = {str(k): i["row_v"] for k, i in infos if "row_v" in i}
        if ev and any("fidelity_pre" in i for _, i in infos):
            rec["fidelity_pre"] = {str(k): i["fidelity_pre"] for k, i in infos if "fidelity_pre" in i}
        if ev:
            copies = {"post": g_sd}
            ep = pack(S.base(round=r, fe_key=feat_key, gen=gspec, copies=list(copies)),
                      ref=ref_rec, **{"g_" + k: v for k, v in copies.items()})
            rec["fidelity_post"] = cfg_of(S.ask("evaluate.gen", [(S.nid[0], ep)], account=False)[0].content)["post"]
        mon.log("stage2_round", **rec)
        history.append(rec)
    G.load_state_dict(g_sd)
    c = dict(g_sd=g_sd, history=history, time_s=time.time() - t0, bytes=_bytes_delta(b0, mon.bytes))
    torch.save(c, path + ".part")
    os.replace(path + ".part", path)
    release(path)
    c["G"] = G
    mon.stop("stage2")
    return key, c


@torch.no_grad()
def fidelity(ref, G, C, n=300, seed=0):
    seed_all(seed, "fidelity")
    G.eval()
    out = []
    for c in range(C):
        y = torch.full((n,), c, device=DEV, dtype=torch.long)
        out.append(float((ref(G.sample(y)).argmax(1) == c).float().mean()))
    return out


@torch.no_grad()
def spread_ratio(G, mu, V, C, n=300, seed=0):
    """Generated within-class spread over the exact real spread (1 = realistic)."""
    seed_all(seed, "spread")
    G.eval()
    out = []
    for c in range(C):
        g = G.sample(torch.full((n,), c, device=DEV, dtype=torch.long)).flatten(1)
        out.append(float((g - g.mean(0)).pow(2).sum(1).mean() / V[c].clamp(min=1e-12)))
    return out


# ---------------------------------------------------------------- stage III
def stage_three(S, hs, feat_key, hd_sd, T_TN, sampler, sampler_key, gen_cfg, secure, res):
    R, mon, C = S.R, S.mon, S.C
    mon.start("stage3")
    hd_sd = [dict(s) for s in hd_sd]
    csl = hs.get("csl", 0.0) > 0 and sampler is not None and hs["ts"] > 0
    P, offs = None, None
    if csl:
        P, offs, _ = fedecs.csl_pool_size(S.fed.sizes, hs["ts"], S.spec["B"], S.K, len(S.groups))
    history = []
    G = sampler
    for r in range(T_TN):
        act = active_clients(S.K, R.get("pfrac", 1.0), R["seed"], "iii", r)
        ev = r == 0 or (r + 1) % EVAL_EVERY == 0 or r == T_TN - 1
        cfg = S.base(round=r, fe_key=feat_key, head=hs, sampler_key=sampler_key if sampler is not None else None,
                     eval=ev, C=C, csl_beta=hs.get("csl", 0.0), round_offset=getattr(S, "round_offset", 0))
        cons = None
        rec = dict(round=r + 1, clients=act)
        if csl:
            pool_seed = int(key_of(R["seed"], "pool", r), 16) % (2 ** 31 - 1)
            cfg.update(pool_seed=pool_seed, P=P)
            hk = key_of(R["run_id"], "heads", r)
            S.ask("query.heads", [(S.nid[k], pack(dict(heads_key=hk), heads=heads_rec(hd_sd, S.groups))) for k in act])
            g_act = sorted({S.arch[k] for k in act})
            n_arch = {g: sum(1 for k in act if S.arch[k] == g) for g in g_act}
            w = {str(k): 1.0 / (len(g_act) * n_arch[S.arch[k]]) for k in act}
            out = S.ssum(dict(cfg, what="consensus", heads_key=hk, w=w), [1.0 / 4.0], act, round_no=r + 1,
                         secure=secure)
            cons = out[0].float().view(P, C)
            ent = float(-(cons.clamp(min=1e-12).log() * cons).sum(1).mean())
            rec.update(consensus_entropy=ent, consensus_confidence=float(cons.max(1).values.mean()))
        items = []
        for k in act:
            c = dict(cfg)
            named = dict(head=hd_sd[S.arch[k]])
            if csl:
                nsyn = hs["ts"] * ((S.fed.sizes[k] // S.spec["B"]) if S.fed.sizes[k] >= S.spec["B"] else 1) * \
                    (S.spec["B"] if S.fed.sizes[k] >= S.spec["B"] else S.fed.sizes[k])
                c["pool_off"] = offs[k]
                named["cons"] = dict(t=cons[offs[k]:offs[k] + nsyn].float())
            items.append((S.nid[k], pack(c, **named)))
        rep = S.ask("train.head", items)
        hd_avg, infos = {g: Averager() for g in S.groups}, []
        for m in rep:
            k = S.pid[m.metadata.src_node_id]
            hd_avg[S.arch[k]].add(arrays_of(m.content, "head"))
            infos.append((k, cfg_of(m.content)))
        for g in S.groups:
            if hd_avg[g].n:
                hd_sd[g] = hd_avg[g].result()
        agree = [i["consensus_agreement"] for _, i in infos if i.get("consensus_agreement") is not None]
        if agree:
            rec["consensus_agreement"] = float(np.mean(agree))
        if hs.get("sed") and G is not None:
            hd_sd = feddf_distill(S, hs, G, hd_sd, r)
        rec.update(syn_loss={str(k): i["syn_loss"] for k, i in infos}, real_loss={str(k): i["real_loss"] for k, i in infos},
                   time={str(k): i["time"] for k, i in infos},
                   probe={str(k): i["probe"] for k, i in infos if i.get("probe")},
                   fc_norm={str(g): hd_sd[g]["fc.weight"].float().norm(dim=1).tolist() for g in S.groups})
        if ev:
            rec["pre"] = {str(k): i["pre"] for k, i in infos if "pre" in i}
            post = eval_heads(S, hd_sd, feat_key)["test"]
            rec["post"] = log_view(post["summary"])
            rec["post_pred_counts"] = np.array(post["summary"]["confusion"]).sum(0).tolist()
        mon.log("stage3_round", **rec)
        history.append(rec)
    mon.stop("stage3")
    return hd_sd, history


def feddf_distill(S, hs, G, hd_sd, r):
    """FedDF-style server ensemble distillation on generated features (as core.stage_iii 'sed')."""
    C = S.C
    seed_all(S.R["seed"], "feddf", r)
    in_ch = hd_sd[S.groups[0]]["features.0.weight"].shape[1]
    headers = make_headers(S.spec, in_ch, C, DEV)
    n_sed = hs["sed"] * hs.get("sed_bs", 128)
    sy = torch.randint(0, C, (n_sed,), device=DEV)
    with torch.no_grad():
        sx = torch.cat([G.sample(sy[s:s + 4096]) for s in range(0, n_sed, 4096)])
        cons = torch.zeros(n_sed, C, device=DEV)
        for g in S.groups:
            headers[g].load_state_dict(hd_sd[g])
            headers[g].eval()
            cons += torch.cat([headers[g](sx[s:s + 4096]).softmax(1) for s in range(0, n_sed, 4096)])
        cons /= len(S.groups)
        tgt = (1 - hs.get("sed_beta", 1.0)) * F.one_hot(sy, C).float() + hs.get("sed_beta", 1.0) * cons
    out = list(hd_sd)
    for g in S.groups:
        h = headers[g]
        h.train()
        opt = torch.optim.SGD(h.parameters(), lr=hs.get("sed_lr", 0.05), momentum=0.0)
        ok = True
        bs = hs.get("sed_bs", 128)
        for s in range(0, n_sed, bs):
            loss = -(tgt[s:s + bs] * h(sx[s:s + bs]).log_softmax(1)).sum(1).mean()
            if not torch.isfinite(loss):
                ok = False
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ok:
            new_sd = {k: v.detach().cpu().clone() for k, v in h.state_dict().items()}
            for k, v in hd_sd[g].items():  # BN running statistics stay those of real features
                if "running_" in k or "num_batches" in k:
                    new_sd[k] = v
            out[g] = new_sd
    return out


# ---------------------------------------------------------------- exact readout
def exact_readout(S, hd_sd, feat_key, Xval, secure, gamma=0.1):
    """Per architecture: secure sums of every client's (n_c, sum e, sum e e^T) through that head's body ->
    class-balanced LDA last layer. Bounds from the validation embeddings (public server data)."""
    hk = key_of(S.R["run_id"], "heads", "readout")
    S.ask("query.heads", [(S.nid[k], pack(dict(heads_key=hk), heads=heads_rec(hd_sd, S.groups))) for k in range(S.K)])
    bounds = []
    in_ch = Xval.shape[1]
    for g in S.groups:
        h = make_head(S.spec, g, in_ch, S.C).to(DEV)
        h.load_state_dict(hd_sd[g])
        h.eval()
        with torch.no_grad():
            E = h.embed(Xval)
        emax = float(E.abs().max()) * 4 + 1e-6
        bounds += [S.N / 4.0, S.N * emax / 4.0, S.N * emax * emax / 4.0]
    out = S.ssum(dict(what="readout", heads_key=hk, groups=S.groups, fe_key=feat_key), bounds, list(range(S.K)),
                 secure=secure)
    last = {}
    for i, g in enumerate(S.groups):
        n, Ssum, M = out[3 * i], out[3 * i + 1], out[3 * i + 2]
        W, b = fedecs.lda_from_sums(n.round(), Ssum, M, gamma)
        last[f"{g}|W"], last[f"{g}|b"] = W.cpu(), b.cpu()
    return last


# ---------------------------------------------------------------- representation and cost measurements
def active_units(G, X, y, thr=0.01):
    """Active latent units of a VAE generator (Burda et al., ICLR 2016): latent dimensions whose posterior mean
    varies over the validation features by more than thr. None for generators without an encoder."""
    if G is None or not hasattr(G, "encode"):
        return {}
    modes = [(m, m.training) for m in G.modules()]
    G.eval()
    with torch.no_grad():
        mus = torch.cat([G.encode(X[s:s + 2048].float(), y[s:s + 2048])[0] for s in range(0, len(y), 2048)])
    for m, t in modes:
        m.training = t
    var = mus.float().var(0)
    return dict(active_units=int((var > thr).sum()), latent_dim=int(var.numel()), latent_var=var.tolist())


class _Sampler(torch.nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, y):
        return self.G.sample(y)


def model_flops(S, shape, G, hs, pfrac=1.0):
    """Forward FLOPs per image of the extractor, of every head architecture and of one generated feature, and the
    expected training FLOPs of one head-stage round over all clients (3 x forward: forward and backward pass;
    real epochs plus the synthetic pass; pfrac of the clients per round)."""
    fe = make_fe(S.spec).to(DEV)
    f_fe = forward_flops(fe, torch.zeros(2, S.spec["in_ch"], 32, 32, device=DEV))
    headers = make_headers(S.spec, shape[0], S.C, DEV)
    f_head = {str(g): forward_flops(headers[g], torch.zeros(2, *shape, device=DEV)) for g in S.groups}
    f_gen = forward_flops(_Sampler(G), torch.zeros(2, dtype=torch.long, device=DEV)) if G is not None else 0.0
    per_round = 0.0
    for k in range(S.K):
        n = int(S.fed.sizes[k])
        nsyn = lt.n_syn(n, S.spec["B"], hs["ts"]) if hs.get("ts", 0) > 0 and G is not None else 0
        per_round += 3 * f_head[str(S.arch[k])] * (hs["tr"] * n + nsyn)
    return dict(flops_fe=f_fe, flops_head=f_head, flops_gen=f_gen,
                head_stage_gflops_per_round=pfrac * per_round / 1e9)


# ---------------------------------------------------------------- one run
def run_pipeline(grid, context, R):
    mon = Monitor(os.path.join(R["out_dir"], "logs"), R["run_id"], tensorboard=R.get("tensorboard", True))
    t0 = time.time()
    S = Fed(grid, context, R, mon)
    m = R["method"]
    spec = S.spec
    T = dict(T_FE=spec["T_FE"], T_KA=spec["T_KA"], T_TN=spec["T_TN"])
    T.update(R.get("rounds") or {})
    res = dict(class_order=S.fed.class_order, class_names=S.fed.class_names,
               test_class_counts=np.bincount(S.fed.y_te.cpu().numpy(), minlength=S.C).tolist(),
               run_id=R["run_id"], label=R["label"], dataset=R["ds"], IF=R["IF"], alpha=R["alpha"], K=R["K"],
               seed=R["seed"], pfrac=R.get("pfrac", 1.0), tail_perm=R.get("tail_perm", False), method=m,
               n_train=S.N, min_client=int(min(S.fed.sizes)), client_counts=S.fed.counts.astype(int).tolist(),
               rounds=T, framework="flower")
    mon.log("run_start", **{k: v for k, v in res.items() if k != "client_counts"})
    if m["pipeline"] == "local":
        return run_local(S, R, T, res, mon, t0)
    feat_key, s1 = stage_one(S, T["T_FE"], la=m.get("fe_la", False))
    res["fe_time_s"] = s1["time_s"]
    res["stage1_final"] = s1["history"][-1].get("post")
    S.ask("query.features", [(S.nid[k], pack(S.base(fe_key=feat_key), fe=s1["fe_sd"])) for k in range(S.K)])
    ref, Xval = train_referee(S, s1["fe_sd"])
    shape = tuple(Xval.shape[1:])
    secure = m.get("secure", False)
    stats = None
    if m.get("stats"):
        n, S1, S2 = release_stats(S, feat_key, Xval)
        stats = (n, S1, S2)
        res["class_sep_F"] = fedecs.separation_F(n, S1, S2)
        _, mu, V = fedecs.moments_from_sums(n, S1, S2)
        mon.log("stats_release", F=res["class_sep_F"], n=n.tolist())
    G, gen_cfg, sampler_key = None, None, None
    gspec = m.get("gen")
    if gspec is not None:
        gspec = dict(gspec)
        if gspec["type"] == "gate":
            rows = res["class_sep_F"] < FEDECS["gate_F"]
            if R.get("force_branch") in ("rows", "anchor"):   # only for tests of the branch not chosen by the gate
                rows = R["force_branch"] == "rows"
            gspec.update(dict(type="cvae", agg="ccwa") if rows else dict(type="pcvae"))
            gspec.pop("gate", None)
            res["gate_branch"] = "rows" if rows else "anchor"
        proto = None
        if gspec["type"] == "pcvae":
            proto = mu.float().view(S.C, *shape)
        key2, s2 = stage_two(S, gspec, feat_key, shape, T["T_KA"], ref, proto=proto, stats=stats)
        res["gen_time_s"] = s2["time_s"]
        G = s2["G"]
        res["generator_params"] = n_params(G) if gspec["type"] != "gauss" else 0
        if "label_proj" in s2["g_sd"]:
            rn = s2["g_sd"]["label_proj"].float().norm(dim=1)
            res["cond_norms_end"] = rn.tolist()
        res["fidelity"] = fidelity(ref, G, S.C, seed=R["seed"])
        res.update(active_units(G, Xval, S.fed.y_val))
        sampler = G
        mc_arr = None
        if m["head"].get("mc"):
            sampler = fedecs.MomentCalibratedGen(G, mu.to(DEV), V.to(DEV), n=FEDECS["mc_samples"],
                                                 clip=FEDECS["mc_clip"], seed=R["seed"])
            mc_arr = dict(mu=sampler.mu, m=sampler.m, scale=sampler.scale)
            res["mc_scale"] = sampler.scale.tolist()
            res["fidelity_mc"] = fidelity(ref, sampler, S.C, seed=R["seed"])
        if stats is not None:
            res["spread_ratio"] = spread_ratio(sampler, mu.to(DEV), V.to(DEV), S.C, seed=R["seed"])
        sampler_key = key_of(key2, "mc" if mc_arr else "plain")
        gen_cfg = gspec
        lg = dict(sampler_key=sampler_key, gen=gspec)
        S.ask("query.load_gen", [(S.nid[k], pack(S.base(fe_key=feat_key, **lg), g=s2["g_sd"],
                                                 **({"mc": mc_arr} if mc_arr else {}))) for k in range(S.K)])
        G = sampler
    hs = dict(m["head"])
    if G is None:
        hs["ts"] = 0
    if spec.get("local_epochs"):   # Track 2 protocol: local epochs of the benchmark
        hs["tr"] = spec["local_epochs"]
    S.round_offset = T["T_FE"]
    hd_sd, h_hist = stage_three(S, hs, feat_key, s1["hd_sd"], T["T_TN"], G, sampler_key, gen_cfg, secure, res)
    res["head_time_s"] = sum(h.get("time", {}).get(str(0), 0) for h in h_hist)
    # final evaluation: test (with probabilities and embeddings for the plots) and validation
    fin = eval_heads(S, hd_sd, feat_key, splits=("test", "val"), probs=True, embed=True)
    res.update(flat("final", fin["test"]["summary"]))
    res.update(flat("val_final", fin["val"]["summary"]))
    res["final_per_arch"] = {g: {k: v for k, v in mm.items() if k in ("acc", "bal", "f1", "tail", "worst", "local")}
                             for g, mm in fin["test"]["per_arch"].items()}
    # exact readout (reported with and without)
    if m.get("readout", True):
        last = exact_readout(S, hd_sd, feat_key, Xval, secure)
        er = eval_heads(S, hd_sd, feat_key, splits=("test", "val"), lastlayer=last, probs=True, embed=False)
        res.update(flat("er", er["test"]["summary"]))
        res.update(flat("val_er", er["val"]["summary"]))
        er_probs = er["test"].get("probs")
    else:
        er_probs, last = None, None
    res.update(model_flops(S, shape, s2["G"] if gspec is not None and gspec["type"] != "gauss" else None, hs,
                           R.get("pfrac", 1.0)))
    res["time_s"] = time.time() - t0
    res["stage_time_s"] = dict(mon.stage_t)
    res["bytes"] = {k: dict(down=v[0], up=v[1], messages=v[2]) for k, v in mon.bytes.items()}
    res["comm_mb_per_client"] = sum(v[0] + v[1] for v in mon.bytes.values()) / 2 ** 20 / S.K
    res.update(gpu_peaks(S))
    extra = dict(y_full=S.fed.y_te.cpu().numpy())
    # last layers (classifier vectors) of the heads as trained and of the exact readout: neural-collapse NC2 / NC3
    # and the minority-collapse angle are computed from them with the test embeddings
    for g in S.groups:
        extra[f"fcw_{g}"] = hd_sd[g]["fc.weight"].float().cpu().numpy()
        extra[f"fcb_{g}"] = hd_sd[g]["fc.bias"].float().cpu().numpy()
        if last is not None and f"{g}|W" in last:
            extra[f"erw_{g}"] = last[f"{g}|W"].float().cpu().numpy()
    nv = min(2000, len(Xval))
    sub = np.random.RandomState(0).choice(len(Xval), nv, replace=False)
    extra.update(val_x=Xval[sub].flatten(1).cpu().numpy().astype(np.float16), val_y=S.fed.y_val[sub].cpu().numpy())
    if G is not None:
        seed_all(R["seed"], "gen_samples")
        gy = torch.arange(S.C, device=DEV).repeat_interleave(max(10, 1000 // S.C))
        with torch.no_grad():
            gx = torch.cat([G.sample(gy[i:i + 4096]) for i in range(0, len(gy), 4096)])
        extra.update(gen_x=gx.flatten(1).cpu().numpy().astype(np.float16), gen_y=gy.cpu().numpy())
    save_run(R, res, s1, hd_sd, G, ref, fin, er_probs, last, s2 if gspec is not None else None, extra)
    mon.log("run_end", time_s=res["time_s"])
    mon.close()
    return res


def run_local(S, R, T, res, mon, t0):
    """Local (no federation): every client trains its own extractor and head for the federated stages' epochs combined."""
    spec = S.spec
    torch.manual_seed(R["seed"])
    fe = make_fe(spec)
    headers = make_headers(spec, fe.out_ch, S.C, "cpu")
    epochs = 5 * T["T_FE"] + 5 * T["T_TN"]
    rep = S.ask("train.local", [(S.nid[k], pack(S.base(epochs=epochs), fe=sd_clone(fe),
                                                head=sd_clone(headers[S.arch[k]]))) for k in range(S.K)])
    per_client = {str(S.pid[m.metadata.src_node_id]): cfg_of(m.content)["metrics"] for m in rep}
    per_client_val = {str(S.pid[m.metadata.src_node_id]): cfg_of(m.content)["val_metrics"] for m in rep}
    # every scalar and every per-class metric, averaged over the clients (each client's own model)
    res.update(flat("final", summarize(per_client)))
    res.update(flat("val_final", summarize(per_client_val)))
    res["local_per_client"] = {k: log_view(v) for k, v in per_client.items()}
    res["time_s"] = time.time() - t0
    save_run(R, res, None, None, None, None, None, None, None, None)
    mon.close()
    return res


def gpu_peaks(S):
    """Peak GPU memory of the run (MB): the server process and the client process(es) share the worker's GPU, so
    their sum is an upper bound of what the run needed (CUDA's own context, ~0.3-0.5 GB per process, comes on top)."""
    rep = S.ask("query.gpu", [(S.nid[k], pack(S.base())) for k in range(S.K)], account=False)   # (also on CPU)
    procs = {}
    for m in rep:
        o = cfg_of(m.content)
        procs[o["pidproc"]] = max(procs.get(o["pidproc"], 0), o["peak_mb"])
    if DEV != "cuda":
        return {}
    server = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    client = sum(procs.values())
    return dict(gpu_peak_mb=server + client, gpu_peak_server_mb=server, gpu_peak_clients_mb=client)


def save_run(R, res, s1, hd_sd, G, ref, fin, er_probs, last, s2, extra=None):
    out = os.path.join(R["out_dir"], "runs.jsonl")
    os.makedirs(R["out_dir"], exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(res) + "\n")
    if fin is not None:
        pdir = os.path.join(R["out_dir"], "predictions")
        os.makedirs(pdir, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in R["run_id"])
        np.savez_compressed(os.path.join(pdir, safe + ".npz"),
                            y=np.array(fin["test"]["embed_y"]),
                            **{f"probs_{g}": np.array(p, dtype=np.float16) for g, p in fin["test"]["probs"].items()},
                            **{f"embed_{g}": np.array(e, dtype=np.float16) for g, e in fin["test"]["embed"].items()},
                            **({f"er_probs_{g}": np.array(p, dtype=np.float16) for g, p in er_probs.items()}
                               if er_probs else {}), **(extra or {}))
    if R.get("models_dir") and (s1 is not None or hd_sd is not None):
        mdir = os.path.join(R["models_dir"])
        os.makedirs(mdir, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in R["run_id"])
        torch.save(dict(fe=s1["fe_sd"] if s1 is not None else None, heads=hd_sd, gen=(s2["g_sd"] if s2 is not None else None),
                        referee=ref.state_dict() if ref is not None else None, readout=last, method=R["method"],
                        run_id=R["run_id"]), os.path.join(mdir, safe + ".pt"))


@app.main()
def main(grid: Grid, context: Context):
    R = RUN
    if R["method"].get("pipeline") in ("gefl", "local"):
        RUN["result"] = run_pipeline(grid, context, R)
    else:
        from baselines import run_baseline
        RUN["result"] = run_baseline(grid, context, R)
