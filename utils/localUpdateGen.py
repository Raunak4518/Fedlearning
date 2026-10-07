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
from typing import Callable

import torch
import torch.nn.functional as F

from registry import Registry

LOCAL_GEN_UPDATE_REGISTRY = Registry("local_gen_update")


def _make_optimizer(net, lr, opt_state=None, betas=(0.9, 0.999), weight_decay=0.0):
    opt = torch.optim.Adam(net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    return opt


def _conditioning_params(net):
    """The class-indexed parameters ([num_classes, ...], row r = class r) the
    generator declares through conditioning_parameter_names()."""
    named = dict(net.named_parameters())
    return [named[k] for k in net.conditioning_parameter_names() if k in named]


def _make_lazy_optimizer(net, lr, opt_state, weight_decay):
    """Adam whose conditioning rows are excluded from the built-in (coupled)
    weight decay; LazyConditioningDecay applies it to them row by row."""
    cond_ids = {id(p) for p in _conditioning_params(net)}
    trunk = [p for p in net.parameters() if id(p) not in cond_ids]
    cond = [p for p in net.parameters() if id(p) in cond_ids]
    opt = torch.optim.Adam([{"params": trunk, "weight_decay": weight_decay},
                            {"params": cond, "weight_decay": 0.0}], lr=lr)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    return opt


class LazyConditioningDecay:
    """Lazy conditioning decay (experiment plan, section 2.1).

    With coupled L2 inside Adam, a class row that gets no data gradient in a
    step still receives g = lambda * w, which Adam normalises to a step of
    about lr * sign(w): rows of classes absent from a batch shrink at full
    learning-rate speed. Here a conditioning row is touched only in steps
    whose batch contains its class, in the spirit of LazyAdam (which updates
    only the embedding rows present in a batch):

      before opt.step(): add lambda * w_r to the gradient of present rows only
      after  opt.step(): restore absent rows, so neither decay nor leftover
                         Adam momentum moves them
    """

    def __init__(self, net, weight_decay: float):
        self.params = _conditioning_params(net)
        self.weight_decay = weight_decay
        self._saved = None
        self._absent = None

    def before_step(self, y: torch.Tensor):
        present = torch.zeros(self.params[0].size(0), dtype=torch.bool, device=y.device) if self.params else None
        if present is None:
            return
        present[y.unique()] = True
        self._absent = ~present
        self._saved = [p.detach()[self._absent].clone() for p in self.params]
        for p in self.params:
            if p.grad is not None and self.weight_decay > 0:
                p.grad[present] += self.weight_decay * p.detach()[present]

    def after_step(self):
        if self._saved is None:
            return
        with torch.no_grad():
            for p, saved in zip(self.params, self._saved):
                p[self._absent] = saved
        self._saved = None


@LOCAL_GEN_UPDATE_REGISTRY.register("vae")
def local_update_vae(net, dataloader, args, opt_state=None):
    """CVAE / CVAE-F. Paper Table XV: Adam lr 1e-3, weight decay 1e-3
    (reference utils/localUpdateGenF.py:41)."""
    net.train()
    wd = getattr(args, 'weight_decay_vae', 1e-3)
    lazy = None
    if getattr(args, 'lazy_cond_decay', 0):
        opt = _make_lazy_optimizer(net, args.gen_lr, opt_state, wd)
        lazy = LazyConditioningDecay(net, wd)
    else:
        opt = _make_optimizer(net, args.gen_lr, opt_state, weight_decay=wd)
    total_loss, n_batches = 0.0, 0
    for _ in range(args.gen_local_ep):
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            recon, mu, logvar = net(x, y)
            loss = net.loss_function(recon, x, mu, logvar)
            loss.backward()
            if lazy is not None:
                lazy.before_step(y)
            opt.step()
            if lazy is not None:
                lazy.after_step()
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    return net.state_dict(), avg_loss, opt.state_dict()


LOCAL_GEN_UPDATE_REGISTRY.register("cvae_paper")(local_update_vae)


@LOCAL_GEN_UPDATE_REGISTRY.register("gan")
def local_update_gan(net, dataloader, args, opt_state=None):
    net.train()
    opt_state = opt_state or {}
    lr = getattr(args, 'gen_lr_gan', args.gen_lr)  # paper Table XV: 2e-4
    opt_g = _make_optimizer(net.G, lr, opt_state.get("G"), betas=(args.b1, args.b2))
    opt_d = _make_optimizer(net.D, lr, opt_state.get("D"), betas=(args.b1, args.b2))

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
    lr = getattr(args, 'gen_lr_ddpm', args.gen_lr)  # paper Table XV: 1e-4
    wd = getattr(args, 'weight_decay_ddpm', 0.0)    # paper Table XV: none (reference GeFL_DDPM-F.py:60)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
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
