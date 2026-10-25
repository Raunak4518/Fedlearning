"""Floating-point operations of a forward pass (convolutions, transposed convolutions and linear layers, the layers
that dominate the cost; 2 FLOPs per multiply-accumulate), counted with forward hooks on one input."""
import torch
import torch.nn as nn


def forward_flops(module, x, *args):
    """FLOPs of module(x, *args) per input row of x."""
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total[0] += 2 * out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose2d):
            total[0] += 2 * inp[0].numel() * (m.out_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += 2 * out.numel() * m.in_features
    hs = [m.register_forward_hook(hook) for m in module.modules()
          if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear))]
    modes = [(m, m.training) for m in module.modules()]   # restored one by one (BatchNorm statistics untouched)
    module.eval()
    try:
        with torch.no_grad():
            module(x, *args)
    finally:
        for m, t in modes:
            m.training = t
        for h in hs:
            h.remove()
    return total[0] / x.shape[0]
