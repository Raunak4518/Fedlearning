"""Each client's local training step equals experiments/core.py's, weight for weight, for one client and one round
(same initial weights, same random seed): Stage I, Stage II (generator), Stage III (hard and consensus labels,
plain and logit-adjusted losses)."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))

import core  # noqa: E402
import fedecs  # noqa: E402
import local_train as lt  # noqa: E402
import models  # noqa: E402
from settings import DATASETS  # noqa: E402

core.DEV = "cpu"
core.USE_GRAPHS = False
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def one_client_ctx(ds, n=300, seed=0):
    torch.manual_seed(123)
    spec = DATASETS[ds]
    C = spec["C"]
    x = torch.rand(n, spec["in_ch"], 32, 32)
    y = torch.arange(n) % C
    part = dict(parts=[torch.arange(n)], counts=np.bincount(y.numpy(), minlength=C)[None, :],
                heldout=torch.arange(0), class_counts=np.bincount(y.numpy(), minlength=C))
    core.set_num_classes(ds)
    data = dict(xtr=x, ytr=y, xte=x[:50], yte=y[:50])
    ctx = core.Ctx(dict(), data, ds, part, 1, seed)
    return ctx, x, y, spec, C


def sd_equal(a, b, tol=0.0):
    return all(torch.allclose(a[k].float(), b[k].float(), atol=tol, rtol=0) for k in a)


def test_stage1_step():
    for ds in ("mnist", "cifar10"):
        ctx, x, y, spec, C = one_client_ctx(ds)
        core.set_seed(5)
        fe_sd, hd_sd, _ = core.stage_i(ctx, 1, "all")
        core.set_seed(5)
        fe = models.PaperFE(spec["in_ch"], spec["fe"])
        heads = models.build_headers(fe.out_ch, spec["header_bn"], C, "cpu")
        lt.local_stage1(fe, heads[0], x, y, spec, "cpu")
        assert sd_equal(fe_sd, fe.state_dict()) and sd_equal(hd_sd[0], heads[0].state_dict())


def test_stage1_step_la():
    ctx, x, y, spec, C = one_client_ctx("fmnist")
    core.set_seed(6)
    fe_sd, hd_sd, _ = core.stage_i(ctx, 1, "all", la=True)
    core.set_seed(6)
    fe = models.PaperFE(spec["in_ch"], spec["fe"])
    heads = models.build_headers(fe.out_ch, spec["header_bn"], C, "cpu")
    lt.local_stage1(fe, heads[0], x, y, spec, "cpu", prior=fedecs.local_log_prior(ctx.part["counts"][0], "cpu"))
    assert sd_equal(fe_sd, fe.state_dict())


def test_generator_step():
    ctx, _, y, spec, C = one_client_ctx("mnist", n=200)
    feats = [torch.rand(200, 3, 16, 16)]
    gspec = dict(core.GEN_DEFAULTS)
    core.set_seed(7)
    g_sd, _, _, _, _ = core.stage_ii(ctx, gspec, feats, [y], 1)
    core.set_seed(7)
    G = models.PaperCVAE(3, spec["latent"], C)
    opt = torch.optim.Adam([{"params": list(G.parameters()), "weight_decay": 1e-3}], lr=1e-3)
    lt.local_gen(G, opt, feats[0], y, "cpu")
    assert sd_equal(g_sd, G.state_dict(), tol=1e-6)
    # decoupled decay (AdamW) as the decay-fix baseline
    core.set_seed(8)
    g2, _, _, _, _ = core.stage_ii(ctx, dict(gspec, opt="adamw"), feats, [y], 1)
    core.set_seed(8)
    G = models.PaperCVAE(3, spec["latent"], C)
    opt = torch.optim.AdamW([{"params": list(G.parameters()), "weight_decay": 1e-3}], lr=1e-3)
    lt.local_gen(G, opt, feats[0], y, "cpu")
    assert sd_equal(g2, G.state_dict(), tol=1e-6)


def _stage3(loss, csl):
    ctx, _, y, spec, C = one_client_ctx("mnist", n=200)
    feats = [torch.rand(200, 3, 16, 16).abs()]
    torch.manual_seed(1)
    G = core.PaperCVAE(3, 16, C=C)
    G.graphs = core.GraphCache()
    torch.manual_seed(2)
    hd0 = [models.sd_clone(h) for h in models.build_headers(3, spec["header_bn"], C, "cpu")]
    hspec = dict(core.HEAD_DEFAULTS, loss=loss, csl=csl)
    core.set_seed(9)
    tracker = core.Tracker()
    heads_core = core.stage_iii(ctx, hspec, None, hd0, G, feats, [y], feats[0][:50], 1, tracker, 10)
    # ours (core.stage_iii builds all ten heads first, consuming the random stream: do the same)
    core.set_seed(9)
    h = models.build_headers(3, spec["header_bn"], C, "cpu")[0]
    h.load_state_dict(hd0[0])
    h.train()
    prior = fedecs.local_log_prior(ctx.part["counts"][0], "cpu") if loss == "la" else None
    nsyn = lt.n_syn(200, spec["B"], 1)
    if csl:
        py = torch.arange(C)[torch.randint(0, C, (nsyn,))]
        px = G.sample(py)
        h.eval()
        with torch.no_grad():
            cons = h(px).softmax(1)
        h.train()
        syn = ("soft", px, fedecs.csl_targets(py, cons, csl, C))
    else:
        ylab = lt.syn_hard_labels(C, nsyn)
        syn = ("hard", G.sample(ylab), ylab)
    lt.local_head(h, h, feats[0], y, spec["B"], hspec, "cpu", prior=prior, syn=syn)
    return sd_equal(heads_core[0].state_dict(), h.state_dict(), tol=1e-6)


def test_head_step_hard_ce():
    assert _stage3("ce", 0.0)


def test_head_step_hard_la():
    assert _stage3("la", 0.0)


def test_head_step_consensus_la():
    assert _stage3("la", 0.5)
