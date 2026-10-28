"""Flower ClientApp: what each client does with its own data. One message type per action:

  query.hello        partition id (the server maps nodes to clients)
  train.stage1       Stage I: local training of the feature extractor and the client's head
  train.gen          Stage II: local generator training (persistent optimiser state per client)
  train.head         Stage III: local head training on generated, then real features
  train.local        Local-only baseline: the whole model trained on the client's data only
  train              secure-sum contributions (SecAgg+ mod): class statistics, CCWA rows, consensus, readout sums
  query.sums         the same contributions in plain (protocols of baselines that send statistics unprotected)
  evaluate.*         evaluation of models the server sends, on the shared test copy and on the validation set

The simulation engine runs one persistent client process per simulation; client data, frozen features, the
generator and per-client optimiser states are kept in that process (dictionary STATE), keyed by run and client.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from flwr.clientapp import ClientApp
from flwr.client.mod import secaggplus_mod

import fedecs
import local_train as lt
from data import Federation, batches, gpu_augment
from fl_common import arrays_of, cfg_of, reply, seed_all
from metrics import arch_priors, class_metrics, evaluate_heads, log_view
from models import GraphCache, PaperHeader, make_fe, make_generator, make_head, make_headers
from secure_sum import client_secsum
from settings import DATASETS, HEADER_WIDTHS

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
STATE = {}  # process-level store: federations, features, generators, optimiser states
app = ClientApp()


# ---------------------------------------------------------------- process store
def fed_of(cfg):
    k = ("fed", cfg["ds"], cfg["IF"], cfg["alpha"], cfg["K"], cfg["seed"], cfg.get("tail_perm", False),
         cfg.get("frac_scale", 1.0), cfg.get("pooled", False))
    if k not in STATE:
        for old in [x for x in STATE if x[0] in ("fed", "feat")]:  # one dataset split per process at a time
            del STATE[old]
        torch.cuda.empty_cache() if DEV == "cuda" else None
        STATE[k] = Federation(cfg["ds"], cfg["data_root"], cfg["IF"], cfg["alpha"], cfg["K"], cfg["seed"], DEV,
                              tail_perm=cfg.get("tail_perm", False), frac_scale=cfg.get("frac_scale", 1.0),
                              pooled=cfg.get("pooled", False))
    return STATE[k]


def pid_of(context):
    return int(context.node_config["partition-id"])


def spec_of(cfg):
    return DATASETS[cfg["ds"]]


def feats_of(cfg, fed):
    """Frozen features of every client, the test set and the validation set, for the FE the server broadcast."""
    k = ("feat", cfg["fe_key"])
    if k not in STATE:
        raise KeyError("features not computed: server must send query.features first")
    return STATE[k]


@torch.no_grad()
def compute_features(fe, x):
    fe.eval()
    return torch.cat([fe(x[s:s + 4000]) for s in range(0, len(x), 4000)])


def graphs_for(name):
    k = ("graphs", name)
    if k not in STATE:
        STATE[k] = GraphCache(DEV == "cuda")
    return STATE[k]


# ---------------------------------------------------------------- hello
@app.query("hello")
def hello(msg, context):
    if DEV == "cuda":   # a new run: this process's GPU memory peak is measured from here
        torch.cuda.reset_peak_memory_stats()
    return reply(msg, dict(pid=pid_of(context), device=DEV, pidproc=os.getpid()))


@app.query("gpu")
def gpu(msg, context):
    """Peak GPU memory of this client process since the run's hello (MB)."""
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20) if DEV == "cuda" else 0
    return reply(msg, dict(peak_mb=peak, pidproc=os.getpid()))


# ---------------------------------------------------------------- stage I
@app.train("stage1")
def stage1(msg, context):
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed, spec = fed_of(cfg), spec_of(cfg)
    seed_all(cfg["seed"], "i", cfg["round"], k)
    t0 = time.time()
    g = fed.arch[k]
    fe = make_fe(spec).to(DEV)
    head = make_head(spec, g, fe.out_ch, fed.C).to(DEV)
    fe.load_state_dict(arrays_of(msg.content, "fe", DEV))
    head.load_state_dict(arrays_of(msg.content, "head", DEV))
    prior = fedecs.local_log_prior(fed.counts[k], DEV) if cfg.get("la") else None
    x_k, y_k = fed.x[k], fed.y[k]
    lr, mom = lt.protocol_lr(spec, cfg["round"])
    losses = lt.local_stage1(fe, head, x_k, y_k, spec, DEV, prior=prior, epochs=spec.get("local_epochs", 5),
                             lr=lr, momentum=mom)
    out = dict(n=len(y_k), loss=float(np.mean(losses)) if losses else None, time=time.time() - t0)
    if cfg.get("eval"):  # pre-aggregation evaluation of the local model on the test set
        with torch.no_grad():
            fte = compute_features(fe, fed.x_te)
            head.eval()
            logits = torch.cat([head(fte[s:s + 2000]) for s in range(0, len(fed.y_te), 2000)])
        m = class_metrics(logits, fed.y_te, fed.C)
        out["pre"] = log_view(m)
    return reply(msg, out, fe=fe.state_dict(), head=head.state_dict())


@app.query("features")
def features(msg, context):
    """Compute (once per process) the frozen features of every client, the test and validation sets."""
    cfg = cfg_of(msg.content)
    fed, spec = fed_of(cfg), spec_of(cfg)
    k = ("feat", cfg["fe_key"])
    if k not in STATE:
        fe = make_fe(spec).to(DEV)
        fe.load_state_dict(arrays_of(msg.content, "fe", DEV))
        memo = {}   # clients holding the same tensor (pooled reference) share one copy of its features
        for x in fed.x:
            if id(x) not in memo:
                memo[id(x)] = compute_features(fe, x)
        xs = [memo[id(x)] for x in fed.x]
        STATE[k] = dict(x=xs, te=compute_features(fe, fed.x_te), val=compute_features(fe, fed.x_val))
    return reply(msg, dict(ok=True, shape=list(STATE[k]["te"].shape[1:])))


# ---------------------------------------------------------------- stage II
def gen_store(cfg, fed, shape):
    """The generator module and every client's persistent optimiser, bound to its parameters (as core.GenClient)."""
    k = ("gen", cfg["gen_key"])
    if k not in STATE:
        for old in [x for x in STATE if x[0] == "gen"]:
            del STATE[old]
        spec = spec_of(cfg)
        proto = None
        if cfg["gen"]["type"] == "pcvae":
            proto = torch.zeros(fed.C, *shape, device=DEV)  # the class means arrive in the weights (buffer)
        G = make_generator(cfg["gen"]["type"], shape[0], spec["latent"], fed.C, proto).to(DEV)
        G.graphs = GraphCache(DEV == "cuda")
        STATE[k] = dict(G=G, opts={}, local={})
    return STATE[k]


def gen_opt(store, k, gspec):
    if k not in store["opts"]:
        G = store["G"]
        cls = torch.optim.AdamW if gspec.get("opt", "adam") == "adamw" else torch.optim.Adam
        store["opts"][k] = cls([{"params": list(G.parameters()), "weight_decay": gspec.get("wd", 1e-3)}],
                               lr=1e-3, fused=DEV == "cuda")
    return store["opts"][k]


@app.train("gen")
def gen(msg, context):
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    X, Y = F_["x"][k], fed.y[k]
    store = gen_store(cfg, fed, tuple(X.shape[1:]))
    G = store["G"]
    G.load_state_dict(arrays_of(msg.content, "g", DEV))
    opt = gen_opt(store, k, cfg["gen"])
    seed_all(cfg["seed"], "ii", cfg["round"], k)
    t0 = time.time()
    losses = lt.local_gen(G, opt, X, Y, DEV)
    sd = {kk: v.detach().clone() for kk, v in G.state_dict().items()}
    store["local"][k] = sd
    out = dict(n=len(Y), loss=float(np.mean(losses)) if losses else None, time=time.time() - t0)
    if cfg.get("eval") and "ref" in msg.content.array_records:
        # pre-aggregation diagnostic: per-class fidelity of this client's local generator copy (section 17b)
        ref = PaperHeader(HEADER_WIDTHS[4], X.shape[1], True, fed.C).to(DEV)
        ref.load_state_dict(arrays_of(msg.content, "ref", DEV))
        ref.eval()
        G.eval()
        with torch.no_grad():
            out["fidelity_pre"] = [float((ref(G.sample(torch.full((200,), c, device=DEV, dtype=torch.long))).argmax(1)
                                          == c).float().mean()) for c in range(fed.C)]
        G.train()
    if G.cond:  # diagnostics: class-row norms of the local copy and Adam's second moment of the rows
        W = G.label_proj.detach()
        out["row_norm"] = W.norm(dim=1).tolist()
        st = opt.state.get(G.label_proj, {})
        if "exp_avg_sq" in st:
            out["row_v"] = st["exp_avg_sq"].mean(1).tolist()
    if cfg.get("send_rows", True):
        return reply(msg, out, g=sd)
    return reply(msg, out, g={kk: v for kk, v in sd.items() if kk not in G.cond})


@app.query("load_gen")
def load_gen(msg, context):
    """Stage III: the final generator (and moment-correction parameters) once per process."""
    cfg = cfg_of(msg.content)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    k = ("sampler", cfg["sampler_key"])
    if k not in STATE:
        # Stage II is over: its every-client Adam states and local generator copies are never used again; they held
        # about 5.5 GB of a T4 through Stage III (CIFAR-100, one shared model, 20 clients)
        for old in [x for x in STATE if x[0] in ("sampler", "gen")]:
            del STATE[old]
        if DEV == "cuda":
            torch.cuda.empty_cache()
        shape = tuple(F_["te"].shape[1:])
        spec = spec_of(cfg)
        if cfg["gen"]["type"] == "gauss":
            from models import GaussFeatureGen
            a = arrays_of(msg.content, "g", DEV)
            G = GaussFeatureGen(a["mu"], a["chol"], shape).to(DEV)
        else:
            proto = None
            if cfg["gen"]["type"] == "pcvae":
                proto = torch.zeros(fed.C, *shape, device=DEV)  # the class means arrive in the weights (buffer)
            G = make_generator(cfg["gen"]["type"], shape[0], spec["latent"], fed.C, proto).to(DEV)
            G.load_state_dict(arrays_of(msg.content, "g", DEV))
        G.eval()
        mc = arrays_of(msg.content, "mc", DEV)
        STATE[k] = dict(G=G, mc=fedecs.MCParams(mc["mu"], mc["m"], mc["scale"]) if mc else None, pool={},
                        shape=shape)
    return reply(msg, dict(ok=True))


def sampler_of(cfg):
    return STATE[("sampler", cfg["sampler_key"])]


@torch.no_grad()
@torch.no_grad()
def draw(sm, y):
    """Generated features for labels y (moment-corrected when the server sent MC parameters), written batch by batch
    into one output tensor (the correction is per sample, so this equals correcting the whole draw at once). Batches
    of 4096 as core.py; 1024 for very large features (Track 2's 32,768 dimensions), to bound the generator's memory."""
    d = int(np.prod(sm["shape"])) if "shape" in sm else 0
    bs = 4096 if d <= 8192 else 1024
    out = None
    for s in range(0, len(y), bs):
        yb = y[s:s + bs]
        xb = sm["G"].sample(yb)
        if sm["mc"] is not None:
            xb = sm["mc"].apply(xb, yb)
        if out is None:
            out = torch.empty((len(y),) + tuple(xb.shape[1:]), dtype=xb.dtype, device=xb.device)
        out[s:s + len(yb)] = xb
        del xb
    return out


POOL_CHUNK_BYTES = 2 ** 30   # the pool is generated in chunks of at most 1 GiB of features


def pool_chunk_size(cfg, sm):
    d = int(np.prod(sm["shape"]))
    return max(1, POOL_CHUNK_BYTES // (4 * d))


@torch.no_grad()
def pool_chunk(cfg, sm, j):
    """Chunk j of this round's shared synthetic pool, regenerated identically by every client from the server's seed
    (chunk j uses seed pool_seed + j). A pool that fits in one chunk (every Track-1 setting) is the same draw as one
    generated at once; with one shared architecture the pool covers all clients' slices and is far larger, so no
    client ever holds all of it."""
    L = pool_chunk_size(cfg, sm)
    n = min(L, cfg["P"] - j * L)
    gstate = torch.random.get_rng_state()
    cstate = torch.cuda.get_rng_state_all() if DEV == "cuda" else None
    torch.manual_seed(cfg["pool_seed"] + j)
    if DEV == "cuda":
        torch.cuda.manual_seed_all(cfg["pool_seed"] + j)
    allowed = torch.arange(cfg["C"], device=DEV)
    py = allowed[torch.randint(0, len(allowed), (n,), device=DEV)]
    px = draw(sm, py)
    torch.random.set_rng_state(gstate)
    if cstate is not None:
        torch.cuda.set_rng_state_all(cstate)
    return px, py


def pool_slice(cfg, sm, o, n):
    """Pool samples [o, o + n) (a client's slice): only the chunks that cover it are generated; the last slice is
    cached, so a client's head step reuses what its consensus step needed."""
    key = (cfg["round"], cfg["pool_seed"], o, n)
    if sm["pool"].get("key") != key:
        sm["pool"].clear()   # free the previous slice first: two slices at once ran a T4 out of memory (CIFAR-100,
        if DEV == "cuda":    # one shared model, ratio 10; 27 Oct 2026). The values drawn are unchanged.
            torch.cuda.empty_cache()
        L = pool_chunk_size(cfg, sm)
        xs = ys = None   # the slice is written chunk by chunk into one tensor, so at most one chunk is alive at a time
        for j in range(o // L, (o + n - 1) // L + 1):
            px, py = pool_chunk(cfg, sm, j)
            a, b = max(o, j * L) - j * L, min(o + n, (j + 1) * L) - j * L
            if xs is None:
                xs = torch.empty((n,) + tuple(px.shape[1:]), dtype=px.dtype, device=px.device)
                ys = torch.empty((n,), dtype=py.dtype, device=py.device)
            s = j * L + a - o
            xs[s:s + b - a] = px[a:b]
            ys[s:s + b - a] = py[a:b]
            del px, py
        sm["pool"].update(key=key, xy=(xs, ys))
    return sm["pool"]["xy"]


@torch.no_grad()
def pool_probs(cfg, sm, h, g):
    """softmax(h_g(pool)) over the whole pool, chunk by chunk (features are dropped after use); cached per round,
    pool and head, since every client of architecture g computes the same values."""
    key = ("pool_probs", cfg["round"], cfg["pool_seed"], cfg["heads_key"], g)
    if key not in STATE:
        for old in [x for x in STATE if x[0] == "pool_probs"]:
            del STATE[old]
        L = pool_chunk_size(cfg, sm)
        out = []
        for j in range((cfg["P"] + L - 1) // L):
            px, _ = pool_chunk(cfg, sm, j)
            out.append(torch.cat([h(px[s:s + 2048]).softmax(1) for s in range(0, len(px), 2048)]))
            del px
        STATE[key] = torch.cat(out)
    return STATE[key]


# ---------------------------------------------------------------- stage III
def head_of(cfg, fed, g, arrays, in_ch):
    """A fresh head module of architecture g with the given weights (inference only)."""
    spec = spec_of(cfg)
    h = make_head(spec, g, in_ch, fed.C).to(DEV)
    h.load_state_dict(arrays)
    return h


def train_head_module(cfg, fed, g, arrays, in_ch):
    """The persistent head module of architecture g for this run (weights loaded in place), so that CUDA graphs
    recorded on its parameters stay valid across rounds and clients (as core.py's headers list)."""
    k = ("headmod", cfg["run_id"], g)
    if k not in STATE:
        for old in [x for x in STATE if x[0] == "headmod" and x[1] != cfg["run_id"]]:
            del STATE[old]
        STATE[k] = head_of(cfg, fed, g, arrays, in_ch)
    h = STATE[k]
    h.load_state_dict(arrays)
    return h


@app.train("head")
def head(msg, context):
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed, spec = fed_of(cfg), spec_of(cfg)
    F_ = feats_of(cfg, fed)
    hs = cfg["head"]
    g = fed.arch[k]
    X, Y = F_["x"][k], fed.y[k]
    h = train_head_module(cfg, fed, g, arrays_of(msg.content, "head", DEV), X.shape[1])
    h.train()
    seed_all(cfg["seed"], "iii", cfg["round"], k)
    t0 = time.time()
    B, C, n = spec["B"], fed.C, len(Y)
    nb, bs = (n // B, B) if n >= B else (1, n)
    graphs = graphs_for(("head", cfg["run_id"], g))
    full = {B}

    def fwd(x):
        return graphs((len(x),), h, h, x) if len(x) in full else h(x)
    prior = fedecs.local_log_prior(fed.counts[k], DEV) if hs["loss"] == "la" else None
    adj = fedecs.client_loss_adj(hs["loss"], fed.counts[k], hs.get("tau", 1.0), DEV)
    out = dict(n=n)
    syn = None
    if hs["ts"] > 0 and cfg.get("sampler_key"):
        sm = sampler_of(cfg)
        nsyn = lt.n_syn(n, B, hs["ts"])
        if "cons" in msg.content.array_records:   # consensus soft labels (CSL) on the shared pool
            px, py = pool_slice(cfg, sm, int(cfg["pool_off"]), nsyn)
            cons = arrays_of(msg.content, "cons", DEV)["t"].float()
            syn = ("soft", px, fedecs.csl_targets(py, cons, cfg["csl_beta"], C))
            # agreement: share of the client's pool features whose consensus class is the generated label
            out["consensus_agreement"] = float((cons.argmax(1) == py).float().mean())
        else:                                     # uniform synthetic labels, hard targets (GeFL-F)
            ylab = lt.syn_hard_labels(C, nsyn).to(DEV)
            syn = ("hard", draw(sm, ylab), ylab)
    lr, mom = lt.protocol_lr(spec, cfg.get("round_offset", 0) + cfg["round"])
    syn_losses, real_losses, probes = lt.local_head(h, fwd, X, Y, B, hs, DEV, prior=prior, adj=adj, syn=syn, lr=lr,
                                                    momentum=mom, probe_C=C)
    out.update(probe=probes, syn_loss=float(np.mean(syn_losses)) if syn_losses else None,
               real_loss=float(np.mean(real_losses)) if real_losses else None, time=time.time() - t0)
    if cfg.get("eval"):
        h.eval()
        with torch.no_grad():
            logits = torch.cat([h(F_["te"][s:s + 2000]) for s in range(0, len(fed.y_te), 2000)])
        m = class_metrics(logits, fed.y_te, C)
        out["pre"] = log_view(m)
    return reply(msg, out, head=h.state_dict())


# ---------------------------------------------------------------- local-only baseline
@app.train("local")
def local(msg, context):
    """No federation: the client trains the extractor and its head on its own data for the same number of epochs
    as the federated stages; returns metrics of its own model on the test set (per-class benefit of collaboration)."""
    cfg, k = cfg_of(msg.content), pid_of(context)
    fed, spec = fed_of(cfg), spec_of(cfg)
    seed_all(cfg["seed"], "local", 0, k)
    g = fed.arch[k]
    fe = make_fe(spec).to(DEV)
    h = make_head(spec, g, fe.out_ch, fed.C).to(DEV)
    fe.load_state_dict(arrays_of(msg.content, "fe", DEV))
    h.load_state_dict(arrays_of(msg.content, "head", DEV))
    opt = torch.optim.SGD(list(fe.parameters()) + list(h.parameters()), lr=0.1, momentum=0.0)
    x_k, y_k = fed.x[k], fed.y[k]
    for _ in range(cfg["epochs"]):
        fe.train(), h.train()
        for b in batches(len(y_k), spec["B"], DEV):
            if b.numel() < 2:
                continue
            x = gpu_augment(x_k[b]) if spec["augment"] else x_k[b]
            loss = F.cross_entropy(h(fe(x)), y_k[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    out = {}
    for split, x, y in (("metrics", fed.x_te, fed.y_te), ("val_metrics", fed.x_val, fed.y_val)):
        with torch.no_grad():
            f_ = compute_features(fe, x)
            h.eval()
            logits = torch.cat([h(f_[s:s + 2000]) for s in range(0, len(y), 2000)])
        m = class_metrics(logits, y, fed.C, prior=fed.counts[k] / max(fed.counts[k].sum(), 1))   # own mix
        out[split] = {kk: v for kk, v in m.items() if kk != "probs"}
    return reply(msg, dict(n=len(y_k), **out))


# ---------------------------------------------------------------- statistics for sums (secure or plain)
def contributions(cfg, k):
    """What client k adds to a sum. Every item is a per-class sum of its own data (or of predictions on public
    generated features), so the server needs only the total."""
    fed = fed_of(cfg)
    what = cfg["what"]
    C = fed.C
    if what == "stats":                     # class statistics release: n, sum h, sum ||h||^2
        F_ = feats_of(cfg, fed)
        n, S1, S2 = fedecs.class_stats(F_["x"][k], fed.y[k], C)
        return [n.cpu().numpy(), S1.cpu().numpy(), S2.cpu().numpy()]
    if what == "gram":                      # Gaussian generator: n, class sums, sum h h^T
        F_ = feats_of(cfg, fed)
        Xf = F_["x"][k].flatten(1).double()
        n = torch.bincount(fed.y[k], minlength=C).double()
        S = torch.zeros(C, Xf.shape[1], dtype=torch.float64, device=DEV).index_add_(0, fed.y[k], Xf)
        return [n.cpu().numpy(), S.cpu().numpy(), (Xf.T @ Xf).cpu().numpy()]
    if what == "rows":                      # CCWA: E(n_kc) row_kc and E(n_kc)
        store = STATE[("gen", cfg["gen_key"])]
        rows = store["local"][k]["label_proj"].double()
        w = torch.tensor(fedecs.ccwa_weights(fed.counts[k], cfg["beta"]), dtype=torch.float64, device=DEV)
        return [(w[:, None] * rows).cpu().numpy(), w.cpu().numpy()]
    if what == "consensus":                 # w_k softmax(h_g(pool)), w_k = 1 / (G n_arch)
        F_ = feats_of(cfg, fed)
        sm = sampler_of(cfg)
        g = fed.arch[k]
        h = head_of(cfg, fed, g, {kk: torch.tensor(np.array(v)) for kk, v in STATE[("heads", cfg["heads_key"])][g].items()},
                    F_["te"].shape[1])
        h.eval()
        p = pool_probs(cfg, sm, h, g)
        return [(cfg["w"][str(k)] * p.double()).cpu().numpy()]
    if what == "readout":                   # per architecture: n, sum e, sum e e^T of the client's real features
        F_ = feats_of(cfg, fed)
        out = []
        for g in cfg["groups"]:
            h = head_of(cfg, fed, g, {kk: torch.tensor(np.array(v)) for kk, v in STATE[("heads", cfg["heads_key"])][g].items()},
                        F_["te"].shape[1])
            h.eval()
            with torch.no_grad():
                E = torch.cat([h.embed(F_["x"][k][s:s + 4096]) for s in range(0, len(fed.y[k]), 4096)])
            n, S, M = fedecs.embed_sums(E, fed.y[k], C)
            out += [n.cpu().numpy(), S.cpu().numpy(), M.cpu().numpy()]
        return out
    raise ValueError(what)


@app.train(mods=[secaggplus_mod])
def secsum(msg, context):
    k = pid_of(context)
    return client_secsum(msg, lambda cfg: contributions(cfg, k))


@app.query("sums")
def plain_sums(msg, context):
    cfg = cfg_of(msg.content)
    arrs = contributions(cfg, pid_of(context))
    return reply(msg, dict(n=len(arrs)), s={str(i): torch.from_numpy(np.ascontiguousarray(a)) for i, a in enumerate(arrs)})


@app.query("heads")
def set_heads(msg, context):
    """Cache the current global heads (needed by consensus and readout contributions)."""
    cfg = cfg_of(msg.content)
    w = arrays_of(msg.content, "heads", "cpu")
    heads = {}
    for name, v in w.items():
        g, kk = name.split("|", 1)
        heads.setdefault(int(g), {})[kk] = v.numpy()
    for old in [x for x in STATE if x[0] == "heads"]:
        del STATE[old]
    STATE[("heads", cfg["heads_key"])] = heads
    return reply(msg, dict(ok=True))


# ---------------------------------------------------------------- evaluation (clients evaluate, server collects)
@app.evaluate("heads")
def eval_heads(msg, context):
    cfg = cfg_of(msg.content)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    w = arrays_of(msg.content, "heads", DEV)
    in_ch = F_["te"].shape[1]
    heads = make_headers(spec_of(cfg), in_ch, fed.C, DEV)
    per_g = {}
    for name, v in w.items():
        g, kk = name.split("|", 1)
        per_g.setdefault(int(g), {})[kk] = v
    groups = sorted(per_g)
    for g in groups:
        heads[g].load_state_dict(per_g[g])
    lw = arrays_of(msg.content, "lastlayer", DEV)  # optional replacement last layers (readouts)
    if lw:
        for g in groups:
            heads[g].fc.weight.data.copy_(lw[f"{g}|W"])
            heads[g].fc.bias.data.copy_(lw[f"{g}|b"])
    out, probs = {}, {}
    for split in cfg.get("splits", ["test"]):
        X, y = (F_["te"], fed.y_te) if split == "test" else (F_["val"], fed.y_val)
        priors = arch_priors(fed.counts, fed.arch, groups)
        per, summ, emb = evaluate_heads(heads, groups, X, y, fed.C, priors=priors,
                                        keep_probs=cfg.get("probs", False) and split == "test",
                                        embed=cfg.get("embed", False) and split == "test")
        res = dict(summary=summ, per_arch={str(g): {kk: v for kk, v in m.items() if kk != "probs"} for g, m in per.items()})
        if cfg.get("probs") and split == "test":
            # float16 arrays (the precision save_run stores) instead of JSON lists of floats: EMNIST's 116,323 x 62
            # x 10 values took several GB of host memory per evaluation (Ray killed the process, Kaggle, 26 Oct 2026)
            probs = {str(g): torch.from_numpy(m["probs"]).half() for g, m in per.items()}
        if emb:
            sub = np.random.RandomState(0).choice(len(y), min(2000, len(y)), replace=False)
            res["embed"] = {str(g): e[sub].tolist() for g, e in emb.items()}
            res["embed_y"] = fed.y_te.cpu().numpy()[sub].tolist()
        out[split] = res
    return reply(msg, out, probs=probs) if probs else reply(msg, out)


@app.evaluate("stage1")
def eval_stage1(msg, context):
    cfg = cfg_of(msg.content)
    fed, spec = fed_of(cfg), spec_of(cfg)
    fe = make_fe(spec).to(DEV)
    fe.load_state_dict(arrays_of(msg.content, "fe", DEV))
    fte = compute_features(fe, fed.x_te)
    w = arrays_of(msg.content, "heads", DEV)
    heads = make_headers(spec, fe.out_ch, fed.C, DEV)
    per_g = {}
    for name, v in w.items():
        g, kk = name.split("|", 1)
        per_g.setdefault(int(g), {})[kk] = v
    groups = sorted(per_g)
    for g in groups:
        heads[g].load_state_dict(per_g[g])
    per, summ, _ = evaluate_heads(heads, groups, fte, fed.y_te, fed.C, priors=arch_priors(fed.counts, fed.arch, groups))
    return reply(msg, dict(summary=summ))


@app.evaluate("gen")
def eval_gen(msg, context):
    """Generator diagnostics on the client: per-class fidelity (share of generated class-c features the referee
    classifies as c, 200 per class) of the generator weights sent; pre- or post-aggregation."""
    cfg = cfg_of(msg.content)
    fed = fed_of(cfg)
    F_ = feats_of(cfg, fed)
    spec = spec_of(cfg)
    shape = tuple(F_["te"].shape[1:])
    proto = None
    if cfg["gen"]["type"] == "pcvae":
        proto = torch.zeros(fed.C, *shape, device=DEV)  # the class means arrive in the weights (buffer)
    G = make_generator(cfg["gen"]["type"], shape[0], spec["latent"], fed.C, proto).to(DEV)
    ref = PaperHeader(HEADER_WIDTHS[4], shape[0], True, fed.C).to(DEV)
    ref.load_state_dict(arrays_of(msg.content, "ref", DEV))
    ref.eval()
    out = {}
    for name in cfg["copies"]:
        G.load_state_dict(arrays_of(msg.content, "g_" + name, DEV))
        G.eval()
        seed_all(cfg["seed"], "evalgen", cfg["round"], name)
        fid = []
        with torch.no_grad():
            for c in range(fed.C):
                y = torch.full((200,), c, device=DEV, dtype=torch.long)
                fid.append(float((ref(G.sample(y)).argmax(1) == c).float().mean()))
        out[name] = fid
    return reply(msg, out)


import htfl_client  # noqa: E402,F401  (registers the Track-1 baseline handlers on this ClientApp)
import grouped_client  # noqa: E402,F401  (registers the handlers of GeFL's grouped baselines)
