"""
generators/ccvae_paper.py

FedCVAE-F exactly as the paper's Table XX and the reference repo's
generators16/CCVAE.py build it, for a C x S x S feature map with S a power
of two (S = 16 in every GeFL-F experiment):

Encoder: the label enters as ONE extra input channel holding the integer
class id (reference: `ones * argmax(y)`), then stride-2 conv(4x4) + BN +
ReLU blocks 64 -> 128 -> 256 -> 512 down to 1 x 1, then Linear(512, l) for
mu and for logvar.

Decoder: relu(Linear(l + C_cls -> 512)([z, onehot(y)])), reshaped to
512 x 1 x 1, then ConvTranspose(4x4, stride 2) + BN + ReLU blocks back up
to C x S x S, final ReLU.

The decoder's first Linear is stored in two pieces that are mathematically
the same layer: `dec_z` (the z columns, with the bias) and `label_proj`
(the one-hot columns, as a [C_cls, 512] matrix, row r = column l + r of
the original weight). Splitting it makes row r the ONLY parameter specific
to class r, which is what Mechanism A aggregates and what lazy conditioning
decay regularises. Both pieces are initialised with the bound the unsplit
Linear(l + C_cls) would use, 1/sqrt(l + C_cls).
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from generators.base import ConditionalGenerator, GEN_REGISTRY


@GEN_REGISTRY.register("cvae_paper")
class PaperCCVAE(ConditionalGenerator):
    def __init__(self, num_classes: int, in_channels: int, img_size: int, args, output_activation: str = "relu"):
        super().__init__(num_classes, in_channels, img_size, args, output_activation)
        n_stages = int(round(math.log2(img_size)))
        if 2 ** n_stages != img_size:
            raise ValueError(f"cvae_paper needs a power-of-two feature size, got {img_size}")
        self.latent_size = args.latent_size
        widths = [min(64 * 2 ** i, 512) for i in range(n_stages)]
        self.flat_dim = widths[-1]

        enc, ci = [], in_channels + 1
        for w in widths:
            enc += [nn.Conv2d(ci, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            ci = w
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(self.flat_dim, self.latent_size)
        self.fc_logvar = nn.Linear(self.flat_dim, self.latent_size)

        self.dec_z = nn.Linear(self.latent_size, self.flat_dim)
        self.label_proj = nn.Parameter(torch.empty(num_classes, self.flat_dim))
        bound = 1.0 / math.sqrt(self.latent_size + num_classes)
        for p in (self.dec_z.weight, self.dec_z.bias, self.label_proj):
            nn.init.uniform_(p, -bound, bound)

        dec, rev = [], list(reversed(widths))
        for i, w in enumerate(rev):
            co = rev[i + 1] if i + 1 < len(rev) else in_channels
            dec.append(nn.ConvTranspose2d(w, co, 4, 2, 1))
            if i + 1 < len(rev):
                dec += [nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.decoder = nn.Sequential(*dec)

    def conditioning_parameter_names(self) -> List[str]:
        return ["label_proj"]

    def encode(self, x, y):
        y_map = y.float().view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        h = self.encoder(torch.cat([x, y_map], dim=1)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, y):
        h = F.relu(self.dec_z(z) + self.label_proj[y])
        h = h.view(-1, self.flat_dim, 1, 1)
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
        # Reference utils/localUpdateGenF.py: sum-reduction MSE + KL.
        recon_loss = F.mse_loss(recon, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld
