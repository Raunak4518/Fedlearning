"""
generators/ccvae.py

Conditional Convolutional VAE -- resolution- and channel-agnostic.

Unlike the original GeFL repo (which hardcodes a separate generator class
per resolution in generators16/ generators32/ generators64/), this single
class builds its own encoder/decoder stage list at __init__ time from
whatever `img_size` and `in_channels` the active dataset provides, by
halving the spatial resolution (via stride-2 convs) until it reaches 4px,
then mirroring that exact size sequence back up in the decoder with
`nn.Upsample(size=...)` (which — unlike ConvTranspose2d — hits an exact
target size regardless of parity, so this works for MNIST's 28px, CIFAR's
32px, STL-10's 96px, or anything else without modification).

The class-conditioning pathway is a single nn.Embedding whose weight is
the only entry `conditioning_parameter_names()` returns -- every other
parameter (encoder + decoder convs) is the "trunk" Mechanism A leaves on
plain volume-weighted FedAvg.
"""
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from generators.base import ConditionalGenerator, GEN_REGISTRY


@GEN_REGISTRY.register("vae")
class CCVAE(ConditionalGenerator):
    def __init__(self, num_classes: int, in_channels: int, img_size: int, args, output_activation: str = "tanh"):
        super().__init__(num_classes, in_channels, img_size, args, output_activation)
        base = args.gen_channels
        self.embed_dim = max(8, base // 4)
        self.label_embed = nn.Embedding(num_classes, self.embed_dim)
        self.latent_size = args.latent_size

        # ---- dynamic spatial-size schedule: halve until <= 4px ----
        sizes = [img_size]
        s = img_size
        while s > 4:
            s = s // 2
            sizes.append(s)
        self.spatial_sizes = sizes
        n_stages = len(sizes) - 1

        enc_channels = []
        ch, out_ch = in_channels, base
        for _ in range(n_stages):
            enc_channels.append((ch, out_ch))
            ch, out_ch = out_ch, min(out_ch * 2, base * 8)
        self.enc_out_channels = enc_channels[-1][1] if enc_channels else in_channels
        self.enc_out_size = sizes[-1]

        enc_layers = []
        for (ci, co) in enc_channels:
            enc_layers += [nn.Conv2d(ci, co, 4, 2, 1), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc_layers)

        flat_dim = self.enc_out_channels * self.enc_out_size * self.enc_out_size
        self.fc_mu = nn.Linear(flat_dim + self.embed_dim, self.latent_size)
        self.fc_logvar = nn.Linear(flat_dim + self.embed_dim, self.latent_size)

        # ---- decoder: exact mirror of the encoder's size schedule ----
        self.dec_fc = nn.Linear(self.latent_size + self.embed_dim, flat_dim)
        dec_channels = [(o, i) for (i, o) in reversed(enc_channels)]
        rev_sizes = list(reversed(sizes))
        dec_layers = []
        for stage_idx, (ci, co) in enumerate(dec_channels):
            target_size = rev_sizes[stage_idx + 1]
            dec_layers.append(nn.Upsample(size=(target_size, target_size), mode="nearest"))
            dec_layers.append(nn.Conv2d(ci, co, 3, 1, 1))
            if stage_idx != len(dec_channels) - 1:
                dec_layers.append(nn.BatchNorm2d(co))
                dec_layers.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*dec_layers) if dec_layers else nn.Identity()

    def conditioning_parameter_names(self) -> List[str]:
        return ["label_embed.weight"]

    def encode(self, x, y):
        h = self.encoder(x).flatten(1)
        e = self.label_embed(y)
        h = torch.cat([h, e], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        e = self.label_embed(y)
        h = self.dec_fc(torch.cat([z, e], dim=1))
        h = h.view(-1, self.enc_out_channels, self.enc_out_size, self.enc_out_size)
        return self._apply_activation(self.decoder(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        z = torch.randn(labels.size(0), self.latent_size, device=labels.device)
        return self.decode(z, labels)

    @staticmethod
    def loss_function(recon, x, mu, logvar) -> torch.Tensor:
        # Paper / reference: sum-reduction ELBO, KL weight = 1.
        # Mean+0.1*KL (β-VAE style) under-regularizes the latent, degrades
        # conditional-generation fidelity, and is not what GEFL runs.
        # Loss is divided by dataset size at logging time (see localUpdateGen).
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
