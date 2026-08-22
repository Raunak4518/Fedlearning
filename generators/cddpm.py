"""
generators/cddpm.py

Conditional DDPM -- a real (if deliberately small) denoising diffusion
model: a proper linear beta schedule, the standard closed-form forward
process q(x_t | x_0), a UNet-lite epsilon-predictor conditioned on both
the diffusion timestep and the class label, and ancestral sampling with
classifier-free guidance. Resolution-agnostic via the same dynamic
down/up-sampling schedule used in CCVAE/CCGAN.

This is the generator the proposal itself flags as the expensive stretch
goal (~3 orders of magnitude more sampling cost than the VAE/GAN variants
per GeFL's own reported MACs) -- included for completeness of the
generator registry (`--gen_model ddpm`), but --gen_model vae or gan should
be the default for the main experiment sweep.
"""
import math
from typing import List

import torch
import torch.nn as nn

from generators.base import ConditionalGenerator, GEN_REGISTRY


def _make_beta_schedule(n_T: int, beta1: float = 1e-4, beta2: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta1, beta2, n_T)


class _SinusoidalTimeEmbed(nn.Module):
    """Standard transformer-style sinusoidal timestep embedding, dimension-agnostic."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device).float() / half)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


def _num_groups(ch: int) -> int:
    """Largest group count in {8,4,2,1} that evenly divides `ch`, so
    GroupNorm never crashes regardless of the (user-configurable) n_feat /
    channel-multiplier schedule."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class _ResBlock(nn.Module):
    """Conv block that additively injects a (timestep + label) embedding, at whatever
    spatial resolution it's called at -- no hardcoded size."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.norm1 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.cond_proj(cond)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class _UNetLite(nn.Module):
    """Down -> bottleneck -> up, with skip connections, resolution-agnostic via the
    same halve-until-4px schedule as CCVAE."""

    def __init__(self, in_channels: int, img_size: int, n_feat: int, cond_dim: int):
        super().__init__()
        sizes = [img_size]
        s = img_size
        while s > 4:
            s = s // 2
            sizes.append(s)
        self.sizes = sizes
        n_stages = len(sizes) - 1
        self.n_stages = n_stages

        self.stem = nn.Conv2d(in_channels, n_feat, 3, 1, 1)

        chans = [n_feat]
        for _ in range(n_stages):
            chans.append(min(chans[-1] * 2, n_feat * 8))

        self.down_blocks = nn.ModuleList([_ResBlock(chans[i], chans[i + 1], cond_dim) for i in range(n_stages)])
        self.downsample = nn.ModuleList([nn.Conv2d(chans[i + 1], chans[i + 1], 4, 2, 1) for i in range(n_stages)])

        self.bottleneck = _ResBlock(chans[-1], chans[-1], cond_dim)

        self.up_blocks = nn.ModuleList(
            [_ResBlock(chans[n_stages - i] + chans[n_stages - i], chans[n_stages - i - 1], cond_dim)
             for i in range(n_stages)]
        )
        self.out_norm = nn.GroupNorm(_num_groups(n_feat), n_feat)
        self.out_conv = nn.Conv2d(n_feat, in_channels, 3, 1, 1)
        self.act = nn.SiLU()

    def forward(self, x, cond):
        h = self.stem(x)
        skips = [h]
        for down_block, down in zip(self.down_blocks, self.downsample):
            h = down_block(h, cond)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h, cond)
        for i, up_block in enumerate(self.up_blocks):
            target_size = self.sizes[self.n_stages - i - 1]
            h = nn.functional.interpolate(h, size=(target_size, target_size), mode="nearest")
            skip = skips[self.n_stages - i]
            h = torch.cat([h, nn.functional.interpolate(skip, size=(target_size, target_size), mode="nearest")
                           if skip.shape[-1] != target_size else skip], dim=1)
            h = up_block(h, cond)
        return self.out_conv(self.act(self.out_norm(h)))


@GEN_REGISTRY.register("ddpm")
class CDDPM(ConditionalGenerator):
    def __init__(self, num_classes: int, in_channels: int, img_size: int, args):
        super().__init__(num_classes, in_channels, img_size, args)
        self.n_T = args.n_T
        self.guide_w = args.guide_w
        cond_dim = args.n_feat * 4

        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbed(args.n_feat), nn.Linear(args.n_feat, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # class index `num_classes` reserved as the "null" label for classifier-free guidance dropout
        self.label_embed = nn.Embedding(num_classes + 1, cond_dim)
        self.unet = _UNetLite(in_channels, img_size, args.n_feat, cond_dim)

        betas = _make_beta_schedule(args.n_T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

    def conditioning_parameter_names(self) -> List[str]:
        return ["label_embed.weight"]

    def _cond(self, t, y):
        return self.time_embed(t) + self.label_embed(y)

    def forward(self, x0: torch.Tensor, y: torch.Tensor, label_drop_prob: float = 0.1):
        """Training forward pass: sample a random timestep per example, add the
        corresponding noise, predict it back out, return the epsilon-prediction MSE loss."""
        b = x0.size(0)
        t = torch.randint(0, self.n_T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].view(b, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(b, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omab * noise

        y_in = y.clone()
        if self.training and label_drop_prob > 0:
            drop = torch.rand(b, device=x0.device) < label_drop_prob
            y_in = torch.where(drop, torch.full_like(y_in, self.num_classes), y_in)

        cond = self._cond(t, y_in)
        eps_hat = self.unet(x_t, cond)
        return nn.functional.mse_loss(eps_hat, noise)

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        b = labels.size(0)
        x = torch.randn(b, self.in_channels, self.img_size, self.img_size, device=device)
        null_labels = torch.full_like(labels, self.num_classes)

        for t_int in reversed(range(self.n_T)):
            t = torch.full((b,), t_int, device=device, dtype=torch.long)
            cond_c = self._cond(t, labels)
            eps_c = self.unet(x, cond_c)
            if self.guide_w != 0:
                cond_u = self._cond(t, null_labels)
                eps_u = self.unet(x, cond_u)
                eps = (1 + self.guide_w) * eps_c - self.guide_w * eps_u
            else:
                eps = eps_c

            alpha = self.alphas[t_int]
            alpha_bar = self.alpha_bar[t_int]
            beta = self.betas[t_int]
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps)
            if t_int > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x.clamp(-1, 1)
