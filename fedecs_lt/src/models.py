"""Networks of the GeFL-F pipeline (copied from experiments/core.py so results pair with the earlier runs):
the frozen feature extractor (FE), the ten client head architectures CNN-1..CNN-10, the CVAE-F generator, the
mean-anchored PC-VAE generator of FedECS-LT and the Gaussian feature generator; plus CUDA-graph replay for speed."""
import math
import numpy as np

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


class GroupedHead(PaperHeader):
    """A client's CNN-g head for the baselines GeFL compares in its grouped setting (FedAvg, FedProx, FedALA, AvgKD,
    CCVR): the same network as GeFL-F's target network; base() is the embedding the HtFL code expects."""

    def base(self, h):
        return self.embed(h)


class ETFHead(nn.Module):
    """FedETF (Li et al., ICCV 2023) on a CNN-g body, as LongTailLib resnet_fedetf.py: linear projection of the
    embedding to C dimensions, L2 normalisation, fixed simplex-ETF classifier P (I - 11^T / C) sqrt(C / (C - 1)) with
    a random orthogonal P (QR of a uniform matrix, seeded; the same for every client), and a learnable logit scale
    initialised at 1."""

    def __init__(self, widths, in_ch, bn, C, seed=0, S=16):
        super().__init__()
        body = PaperHeader(widths, in_ch, bn, C, S)
        self.features = body.features
        self.emb_dim = body.emb_dim
        self.linear_proto = nn.Linear(body.emb_dim, C)
        a = np.random.RandomState(seed).random_sample((C, C))
        P, _ = np.linalg.qr(a)
        M = np.sqrt(C / (C - 1)) * (torch.tensor(P).float() @ (torch.eye(C) - torch.ones(C, C) / C))
        self.register_buffer("proto", M)
        self.scaling_train = nn.Parameter(torch.tensor(1.0))

    def embed(self, h):
        return self.features(h).flatten(1)

    base = embed

    def forward(self, h):
        f = self.linear_proto(self.embed(h))
        f = f / torch.norm(f, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return self.scaling_train * (f @ self.proto)


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

    def encode(self, x, y):
        ymap = y.float().view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        h = self.enc(torch.cat([x, ymap], 1)).flatten(1)
        return self.mu(h), self.logvar(h)

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


# ---------------------------------------------------------------- Track 2: ResNet-8 (FedLTLib / CReFF protocol)
class BasicBlock(nn.Module):
    """Standard CIFAR ResNet basic block (He et al., 2016), as in the ResNet-8 of CReFF / LongTailLib."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + (x if self.down is None else self.down(x)))


class ResNet8FE(nn.Module):
    """ResNet-8 (one block per stage, widths 64 / 128 / 256, i.e. scaling 4) up to the second stage: the shared,
    frozen feature extractor of FedECS-LT in Track 2; output 128 x 16 x 16."""

    def __init__(self, in_ch=3, scaling=4):
        super().__init__()
        w = 16 * scaling
        self.net = nn.Sequential(nn.Conv2d(in_ch, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                                 BasicBlock(w, w), BasicBlock(w, 2 * w, 2))
        self.out_ch = 2 * w

    def forward(self, x):
        return self.net(x)


class ResNet8Head(nn.Module):
    """The rest of ResNet-8 (third stage, global average pooling, linear layer): the head of every client."""

    def __init__(self, in_ch, C, scaling=4):
        super().__init__()
        w = 64 * scaling
        self.features = nn.Sequential(BasicBlock(in_ch, w, 2), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(w, C)
        self.emb_dim = w

    def embed(self, h):
        return self.features(h).flatten(1)

    def forward(self, h):
        return self.fc(self.embed(h))


def make_fe(spec):
    return ResNet8FE(spec["in_ch"]) if spec["fe"] == "resnet8" else PaperFE(spec["in_ch"], spec["fe"])


def make_head(spec, g, in_ch, C):
    if spec.get("heads") == "resnet8":
        return ResNet8Head(in_ch, C)
    return PaperHeader(HEADER_WIDTHS[g], in_ch, spec["header_bn"], C)


def make_headers(spec, in_ch, C, device):
    """All architectures' heads (Track 1: CNN-1..CNN-10; Track 2: one ResNet-8 head)."""
    n = 1 if spec.get("heads") == "resnet8" else N_ARCH
    return [make_head(spec, g, in_ch, C).to(device) for g in range(n)]
