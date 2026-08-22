"""
utils/localUpdateGen.py

One client's local generator training step, for one communication round.
Three implementations -- one per registered `--gen_model` -- selected by
`get_local_gen_update(args.gen_model)`, matching the original repo's
`LocalUpdate_CVAE` / equivalent-per-generator-type naming convention.

Every implementation has the same signature so the federated loop in
GeFL_*.py never branches on gen_model itself:

    new_state_dict, avg_loss, new_opt_state = update(
        net, dataloader, args, opt_state
    )

`opt_state` is threaded through across rounds (the same client keeps its
own optimizer momentum between rounds, as in the original repo), rather
than re-initialized from scratch every round.
"""
from typing import Callable, Dict

import torch
import torch.nn.functional as F

from registry import Registry

LOCAL_GEN_UPDATE_REGISTRY = Registry("local_gen_update")


def _make_optimizer(net, lr, opt_state=None, betas=(0.9, 0.999)):
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    return opt


@LOCAL_GEN_UPDATE_REGISTRY.register("vae")
def local_update_vae(net, dataloader, args, opt_state=None):
    net.train()
    opt = _make_optimizer(net, args.gen_lr, opt_state)
    total_loss, n_batches = 0.0, 0
    for _ in range(args.gen_local_ep):
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            recon, mu, logvar = net(x, y)
            loss = net.loss_function(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    return net.state_dict(), avg_loss, opt.state_dict()


@LOCAL_GEN_UPDATE_REGISTRY.register("gan")
def local_update_gan(net, dataloader, args, opt_state=None):
    net.train()
    opt_state = opt_state or {}
    opt_g = _make_optimizer(net.G, args.gen_lr, opt_state.get("G"), betas=(args.b1, args.b2))
    opt_d = _make_optimizer(net.D, args.gen_lr, opt_state.get("D"), betas=(args.b1, args.b2))

    total_g, total_d, n_batches = 0.0, 0.0, 0
    for _ in range(args.gen_local_ep):
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device)
            b = x.size(0)
            real_target = torch.ones(b, 1, device=args.device)
            fake_target = torch.zeros(b, 1, device=args.device)

            # ---- D step ----
            opt_d.zero_grad()
            z = torch.randn(b, net.latent_size, device=args.device)
            fake = net.G(z, y).detach()
            d_real = net.D(x, y)
            d_fake = net.D(fake, y)
            d_loss = F.binary_cross_entropy_with_logits(d_real, real_target) + \
                F.binary_cross_entropy_with_logits(d_fake, fake_target)
            d_loss.backward()
            opt_d.step()

            # ---- G step ----
            opt_g.zero_grad()
            z = torch.randn(b, net.latent_size, device=args.device)
            fake = net.G(z, y)
            g_loss = F.binary_cross_entropy_with_logits(net.D(fake, y), real_target)
            g_loss.backward()
            opt_g.step()

            total_g += g_loss.item()
            total_d += d_loss.item()
            n_batches += 1

    avg_loss = (total_g + total_d) / max(2 * n_batches, 1)
    return net.state_dict(), avg_loss, {"G": opt_g.state_dict(), "D": opt_d.state_dict()}


@LOCAL_GEN_UPDATE_REGISTRY.register("ddpm")
def local_update_ddpm(net, dataloader, args, opt_state=None):
    net.train()
    opt = _make_optimizer(net, args.gen_lr, opt_state)
    total_loss, n_batches = 0.0, 0
    for _ in range(args.gen_local_ep):
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = net(x, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    return net.state_dict(), avg_loss, opt.state_dict()


def get_local_gen_update(gen_model: str) -> Callable:
    return LOCAL_GEN_UPDATE_REGISTRY.get(gen_model)
