"""
generators/base.py

Every generator (CCVAE, CCGAN, CDDPM) implements this interface. The one
method that matters for Mechanism A is `conditioning_parameter_names()`:
it tells the aggregator exactly which state_dict keys form the
class-conditioning pathway (as opposed to the unconditional trunk), so
`utils/avg.py` never has to hardcode a parameter name like
`"label_embed.weight"` for a specific architecture -- every registered
generator declares its own conditioning keys, and Mechanism A works
identically for a VAE, a GAN, or a diffusion model.
"""
from typing import List

import torch.nn as nn

from registry import Registry

GEN_REGISTRY = Registry("generator")


class ConditionalGenerator(nn.Module):
    """Abstract interface. `args` carries every architecture hyperparameter
    (latent_size, gen_channels, ...) so no generator subclass ever reads a
    hardcoded constant -- everything comes from the CLI/config."""

    def __init__(self, num_classes: int, in_channels: int, img_size: int, args):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = img_size

    def conditioning_parameter_names(self) -> List[str]:
        """state_dict keys (this module's own naming) that encode
        class-conditioning information -- these get Mechanism A's
        per-class effective-number weighting at aggregation time. All
        other keys are treated as the unconditional trunk and aggregated
        with plain volume-weighted FedAvg."""
        raise NotImplementedError

    def sample(self, labels):
        """labels: LongTensor (n,) -> images: FloatTensor (n, C, H, W) in
        the same normalized range the dataset's own tensors are in."""
        raise NotImplementedError
