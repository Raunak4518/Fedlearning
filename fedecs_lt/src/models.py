"""Networks of the GeFL-F pipeline (copied from experiments/core.py so results pair with the earlier runs):
the frozen feature extractor (FE), the ten client head architectures CNN-1..CNN-10, the CVAE-F generator, the
mean-anchored PC-VAE generator of FedECS-LT and the Gaussian feature generator; plus CUDA-graph replay for speed."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from settings import HEADER_WIDTHS, N_ARCH


class PaperFE(nn.Module):
    """GeFL-F Tables XXI-XXIII: 3x16x16 features (MNIST-type), 10x16x16 (CIFAR-type)."""

    def __init__(self, in_ch, variant):
        super().__init__()
        if variant == "mnist":
            layers = [nn.Conv2d(in_ch, 3, 3, 1, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True)]
            self.out_ch = 3
        else:
            layers = [nn.Conv2d(in_ch, 3, 3, 1, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True),
                      nn.Conv2d(3, 10, 3, 1, 1), nn.BatchNorm2d(10), nn.ReLU(inplace=True)]
            self.out_ch = 10
        self.net = nn.Sequential(*layers, nn.MaxPool2d(2, 2))

    def forward(self, x):
        return self.net(x)


class PaperHeader(nn.Module):
    """Client head CNN-g: conv blocks of HEADER_WIDTHS[g] (body) and a linear last layer fc."""

    def __init__(self, widths, in_ch, bn, C, S=16):
        super().__init__()
        layers, ci, s = [], in_ch, S
        for w in widths:
            layers.append(nn.Conv2d(ci, w, 3, 1, 1))
            if bn:
                layers.append(nn.BatchNorm2d(w))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
            ci, s = w, s // 2
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(ci * s * s, C)
        self.emb_dim = ci * s * s

    def embed(self, h):
        return self.features(h).flatten(1)

    def forward(self, h):
        return self.fc(self.embed(h))


def build_headers(in_ch, bn, C, device):
    return [PaperHeader(HEADER_WIDTHS[g], in_ch, bn, C).to(device) for g in range(N_ARCH)]


class PaperCVAE(nn.Module):
    """CVAE-F (GeFL-F Table XX). Class identity enters only through label_proj (C x 512), one row per class."""
    kind = "cvae"
    cond = {"label_proj": 0}

    def __init__(self, in_ch, latent, C, S=16):
        super().__init__()
        n = int(round(math.log2(S)))
        widths = [min(64 * 2 ** i, 512) for i in range(n)]
        enc, ci = [], in_ch + 1
        for w in widths:
            enc += [nn.Conv2d(ci, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            ci = w
        self.enc = nn.Sequential(*enc)
        self.flat = widths[-1]
        self.latent = latent
        self.mu, self.logvar = nn.Linear(self.flat, latent), nn.Linear(self.flat, latent)
        self.dec_z = nn.Linear(latent, self.flat)
        self.label_proj = nn.Parameter(torch.empty(C, self.flat))
        b = 1.0 / math.sqrt(latent + C)
        for p in (self.dec_z.weight, self.dec_z.bias, self.label_proj):
            nn.init.uniform_(p, -b, b)
        dec, rev = [], list(reversed(widths))
        for i, w in enumerate(rev):
            co = rev[i + 1] if i + 1 < len(rev) else in_ch
            dec.append(nn.ConvTranspose2d(w, co, 4, 2, 1))
            if i + 1 < len(rev):
                dec += [nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.dec = nn.Sequential(*dec)

    def decode(self, z, y):
        h = F.relu(self.dec_z(z) + self.label_proj[y]).view(-1, self.flat, 1, 1)
        return F.relu(self.dec(h))

    def loss(self, x, y):
        ymap = y.float().view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        h = self.enc(torch.cat([x, ymap], 1)).flatten(1)
        mu, logvar = self.mu(h), self.logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = self.decode(z, y)
        return F.mse_loss(rec, x, reduction="sum") - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    @torch.no_grad()
    def sample(self, y, z=None):
        z = torch.randn(y.size(0), self.latent, device=y.device) if z is None else z
        return self.decode(z, y)


class ProtoCVAE(nn.Module):
    """PC-VAE, the mean-anchored generator of FedECS-LT: CVAE-F without class rows; the class enters through its
    exact federated mean mu_c (from secure-aggregated sums) and a class-shared projection,
    x = ReLU(mu_y + dec(z, mu_y)). No class-specific parameter exists, so averaging cannot shrink a class."""
    kind = "cvae"
    cond = {}

    def __init__(self, in_ch, latent, proto, S=16):
        super().__init__()
        n = int(round(math.log2(S)))
        widths = [min(64 * 2 ** i, 512) for i in range(n)]
        enc, ci = [], 2 * in_ch
        for w in widths:
            enc += [nn.Conv2d(ci, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            ci = w
        self.enc = nn.Sequential(*enc)
        self.flat, self.latent = widths[-1], latent
        self.mu, self.logvar = nn.Linear(self.flat, latent), nn.Linear(self.flat, latent)
        self.dec_z = nn.Linear(latent, self.flat)
        self.proto_proj = nn.Linear(proto[0].numel(), self.flat)
        dec, rev = [], list(reversed(widths))
        for i, w in enumerate(rev):
            co = rev[i + 1] if i + 1 < len(rev) else in_ch
            dec.append(nn.ConvTranspose2d(w, co, 4, 2, 1))
            if i + 1 < len(rev):
                dec += [nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.dec = nn.Sequential(*dec)
        self.register_buffer("proto", proto.clone())
        self.register_buffer("zm", torch.zeros(latent))
        self.register_buffer("zL", torch.eye(latent))

    def decode(self, z, y):
        p = self.proto[y]
        h = F.relu(self.dec_z(z) + self.proto_proj(p.flatten(1))).view(-1, self.flat, 1, 1)
        return F.relu(p + self.dec(h))

    def encode(self, x, y):
        h = self.enc(torch.cat([x, self.proto[y]], 1)).flatten(1)
        return self.mu(h), self.logvar(h)

    def loss(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        rec = self.decode(z, y)
        return F.mse_loss(rec, x, reduction="sum") - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    @torch.no_grad()
    def sample(self, y, z=None):
        z = torch.randn(y.size(0), self.latent, device=y.device) if z is None else z
        return self.decode(self.zm + z @ self.zL.T, y)


class GaussFeatureGen(nn.Module):
    """Class-conditional Gaussian with the pooled within-class covariance, from exact sums (control method)."""
    kind = "gauss"
    cond = {}

    def __init__(self, mu, chol, shape):
        super().__init__()
        self.register_buffer("mu", mu)
        self.register_buffer("chol", chol)
        self.shape = tuple(shape)

    @torch.no_grad()
    def sample(self, y, z=None):
        eps = torch.randn(len(y), self.mu.shape[1], device=self.mu.device) if z is None else z
        return (self.mu[y] + eps @ self.chol.T).clamp(min=0).view(len(y), *self.shape)


def make_generator(kind, in_ch, latent, C, proto=None):
    if kind == "pcvae":
        return ProtoCVAE(in_ch, latent, proto)
    if kind == "cvae":
        return PaperCVAE(in_ch, latent, C)
    raise ValueError(kind)


# ---------------------------------------------------------------- CUDA graphs (as core.py: same maths, ~2.6x faster)
class _Call(nn.Module):
    def __init__(self, fn, owner):
        super().__init__()
        self.owner = owner
        self.fn = fn

    def forward(self, *a):
        return self.fn(*a)


class GraphCache:
    """Records forward+backward of full-size batches once as a CUDA graph and replays it; other shapes run eagerly."""

    def __init__(self, enabled):
        self.store, self.enabled = {}, enabled

    def __call__(self, key, module, fn, *args):
        if not self.enabled or not module.training:
            return fn(*args)
        g = self.store.get(key)
        if g is None:
            saved = {k: v.detach().clone() for k, v in module.state_dict().items()}
            g = torch.cuda.make_graphed_callables(_Call(fn, module), tuple(a.detach().clone() for a in args))
            module.load_state_dict(saved)  # undo BN-statistics updates from the warm-up passes
            self.store[key] = g
        return g(*args)


def sd_clone(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def sd_to_numpy(sd):
    """State dict -> (names, numpy arrays) for Flower ArrayRecords; float16/32 kept, integer buffers kept."""
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def sd_from_numpy(arrs, device):
    return {k: torch.from_numpy(v.copy()).to(device) for k, v in arrs.items()}


def n_params(m):
    return sum(p.numel() for p in m.parameters())
