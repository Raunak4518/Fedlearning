"""
generators/ccgan.py

Conditional DCGAN -- resolution-agnostic, mirroring CCVAE's dynamic
size-schedule construction. Holds both the generator (`G`) and
discriminator (`D`) as sub-modules of one CCGAN, since GeFL's federated
loop trains and aggregates both (a GAN can't be trained federated with
only the generator shared -- the discriminator has to travel with it or
every client's local discriminator instantly overfits to its own local
class distribution, which would silently reintroduce exactly the
class-imbalance problem this project targets).

Both G and D declare their own label-embedding row as a conditioning
parameter, so Mechanism A reweights both pathways per class.
"""
from typing import List

import torch
import torch.nn as nn

from generators.base import ConditionalGenerator, GEN_REGISTRY


class _Generator(nn.Module):
    def __init__(self, num_classes, in_channels, img_size, latent_size, base, embed_dim):
        super().__init__()
        self.latent_size = latent_size
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        sizes = [img_size]
        s = img_size
        while s > 4:
            s = s // 2
            sizes.append(s)
        self.start_size = sizes[-1]
        rev_sizes = list(reversed(sizes))
        n_stages = len(sizes) - 1

        # channel schedule mirrors the encoder-style progression, reversed
        chans = [base]
        for _ in range(n_stages - 1):
            chans.append(min(chans[-1] * 2, base * 8))
        chans = list(reversed(chans))  # widest first, narrowing toward output
        self.start_channels = chans[0]

        self.fc = nn.Linear(latent_size + embed_dim, self.start_channels * self.start_size * self.start_size)

        layers = []
        ch = self.start_channels
        for stage_idx in range(n_stages):
            out_ch = chans[stage_idx + 1] if stage_idx + 1 < len(chans) else base
            target_size = rev_sizes[stage_idx + 1]
            is_last = stage_idx == n_stages - 1
            layers.append(nn.Upsample(size=(target_size, target_size), mode="nearest"))
            layers.append(nn.Conv2d(ch, in_channels if is_last else out_ch, 3, 1, 1))
            if not is_last:
                layers += [nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
                ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, z, y):
        e = self.label_embed(y)
        h = self.fc(torch.cat([z, e], dim=1))
        h = h.view(-1, self.start_channels, self.start_size, self.start_size)
        return torch.tanh(self.net(h))


class _Discriminator(nn.Module):
    def __init__(self, num_classes, in_channels, img_size, base, embed_dim):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, embed_dim)

        sizes = [img_size]
        s = img_size
        while s > 4:
            s = s // 2
            sizes.append(s)
        n_stages = len(sizes) - 1

        layers = []
        ch, out_ch = in_channels, base
        for _ in range(n_stages):
            layers += [nn.Conv2d(ch, out_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            ch, out_ch = out_ch, min(out_ch * 2, base * 8)
        self.conv = nn.Sequential(*layers)
        self.out_channels = ch
        self.out_size = sizes[-1]
        flat_dim = self.out_channels * self.out_size * self.out_size
        self.fc = nn.Linear(flat_dim + embed_dim, 1)

    def forward(self, x, y):
        h = self.conv(x).flatten(1)
        e = self.label_embed(y)
        return self.fc(torch.cat([h, e], dim=1))


@GEN_REGISTRY.register("gan")
class CCGAN(ConditionalGenerator):
    def __init__(self, num_classes: int, in_channels: int, img_size: int, args):
        super().__init__(num_classes, in_channels, img_size, args)
        embed_dim = max(8, args.gen_channels // 4)
        self.latent_size = args.latent_size
        self.G = _Generator(num_classes, in_channels, img_size, args.latent_size, args.gen_channels, embed_dim)
        self.D = _Discriminator(num_classes, in_channels, img_size, args.gen_channels, embed_dim)

    def conditioning_parameter_names(self) -> List[str]:
        return ["G.label_embed.weight", "D.label_embed.weight"]

    @torch.no_grad()
    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        z = torch.randn(labels.size(0), self.latent_size, device=labels.device)
        return self.G(z, labels)
