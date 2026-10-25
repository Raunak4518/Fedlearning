"""One client's local training steps, exactly as experiments/core.py's inner loops (tests/test_local_steps.py checks
weight-for-weight equality with core.stage_i, core.stage_ii and core.stage_iii for one client and one round)."""
import numpy as np
import torch
import torch.nn.functional as F

import fedecs
from data import batches, gpu_augment


def local_stage1(fe, head, x_k, y_k, spec, device, prior=None, epochs=5, lr=0.1, momentum=0.0):
    """Stage I: SGD(0.1) on the extractor and the client's head jointly, 5 local epochs (T_w). Track 2 passes the
    FedLTLib protocol instead (10 epochs, momentum 0.9, decayed learning rate)."""
    fe.train(), head.train()
    opt = torch.optim.SGD(list(fe.parameters()) + list(head.parameters()), lr=lr, momentum=momentum)
    losses = []
    for _ in range(epochs):
        for b in batches(len(y_k), spec["B"], device):
            if b.numel() < 2:
                continue
            x, y = x_k[b], y_k[b]
            if spec["augment"]:
                x = gpu_augment(x)
            logits = head(fe(x))
            loss = F.cross_entropy(logits + prior if prior is not None else logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return losses


def local_gen(G, opt, X, Y, device, B=64, epochs=5):
    """Stage II: 5 local epochs of the generator loss with the client's persistent optimiser."""
    G.train()
    losses = []
    for _ in range(epochs):
        for b in batches(len(Y), B, device):
            if b.numel() < 2:
                continue
            x, y = X[b], Y[b]
            graphs = getattr(G, "graphs", None)
            loss = graphs(("loss", len(y)), G, G.loss, x, y) if (graphs is not None and len(y) == B) else G.loss(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()) / len(y))
    return losses


def syn_hard_labels(C, n):
    """GeFL-F: uniform synthetic labels drawn with NumPy's global generator (as core.stage_iii)."""
    return torch.from_numpy(np.random.choice(C, size=n, p=np.full(C, 1.0 / C))).long()


def probe_head(h, X, Y, C):
    """The head on the client's own real features, without changing it (eval mode, no gradients, so BatchNorm
    statistics and the random streams are untouched): logit spread (mean over images of the standard deviation of
    the logits over classes) and balanced accuracy over the classes the client holds."""
    was = h.training
    h.eval()
    with torch.no_grad():
        lg = torch.cat([h(X[s:s + 2000]) for s in range(0, len(X), 2000)]).float()
    h.train(was)
    pred = lg.argmax(1)
    rec = [float((pred[Y == c] == c).float().mean()) for c in range(C) if (Y == c).any()]
    return dict(logit_spread=float(lg.std(1).mean()), bal=float(np.mean(rec)) if rec else None)


def local_head(h, fwd, X, Y, B, hs, device, prior=None, adj=None, syn=None, lr=0.1, momentum=0.0, probe_C=None):
    """Stage III: synthetic phase (one pass, batch bs) then real phase (tr epochs, batch B), fresh SGD(0.1) each.
    syn: None, ("hard", xs, ylab) or ("soft", xs, targets). probe_C: number of classes; when given, the head is
    probed on the client's real features at the start, after the synthetic phase and at the end (probe_head);
    returned as a third value."""
    n = len(Y)
    nb, bs = (n // B, B) if n >= B else (1, n)
    syn_losses, real_losses = [], []
    probes = {}
    if probe_C:
        probes["start"] = probe_head(h, X, Y, probe_C)
    if syn is not None:
        kind, xs, t = syn
        opt = torch.optim.SGD(h.parameters(), lr=lr, momentum=momentum)
        for s in range(0, len(xs), bs):
            if kind == "soft":
                loss = -(t[s:s + bs] * fwd(xs[s:s + bs]).log_softmax(1)).sum(1).mean()
            else:
                loss = F.cross_entropy(fwd(xs[s:s + bs]), t[s:s + bs])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            syn_losses.append(float(loss.detach()))
        if probe_C:
            probes["after_syn"] = probe_head(h, X, Y, probe_C)
    opt = torch.optim.SGD(h.parameters(), lr=lr, momentum=momentum)
    for _ in range(hs["tr"]):
        for b in batches(n, B, device):
            if b.numel() < 2:
                continue
            loss = fedecs.head_loss(fwd(X[b]), Y[b], hs["loss"], prior, hs.get("tau", 1.0), adj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            real_losses.append(float(loss.detach()))
    if probe_C:
        probes["end"] = probe_head(h, X, Y, probe_C)
        return syn_losses, real_losses, probes
    return syn_losses, real_losses


def protocol_lr(spec, global_round):
    """(learning rate, momentum) of a local update: GeFL-F's SGD 0.1 without momentum (Track 1), or the FedLTLib
    protocol (Track 2: 0.1 x 0.99^round, momentum 0.9)."""
    o = spec.get("opt")
    if not o:
        return 0.1, 0.0
    return o["lr"] * o.get("decay", 1.0) ** global_round, o.get("momentum", 0.0)


def n_syn(n, B, ts):
    nb, bs = (n // B, B) if n >= B else (1, n)
    return ts * nb * bs
