"""
gefl_f/paper_nets.py

The GeFL-F target networks exactly as the paper's appendix specifies them
(Kang et al. 2025, Tables XXI-XXIII), split into the common feature
extractor and the ten heterogeneous headers.

Feature extractor (32x32 input -> 16x16 feature map):
    "mnist" (MNIST / FMNIST / SVHN, Tables XXI-XXII):
        conv(3, 3x3, pad 1) - bn(3) - relu - maxpool(2)           -> 3 x 16 x 16
    "cifar10" (Table XXIII):
        conv(3) - bn(3) - relu - conv(10) - bn(10) - relu - maxpool(2) -> 10 x 16 x 16

Headers CNN-1 ... CNN-10: a stack of conv(w, 3x3, pad 1) [- bn(w)] - relu -
maxpool(2) blocks, then one fc to the classes. Widths per header:

    CNN-1  16                 fc 1024      CNN-6  20 40 80          fc 320
    CNN-2  16 32              fc  512      CNN-7  10 20 40          fc 160
    CNN-3  20 40              fc  640      CNN-8  16 32 64 128      fc 128
    CNN-4  10 20              fc  320      CNN-9  20 40 80 100      fc 100
    CNN-5  16 32 64           fc  256      CNN-10 10 20 40 80       fc  80

Table XXII (FMNIST, SVHN) adds bn after every header conv: registered as
"cnnN_bn". Table XXIII lists 10 channels for CIFAR-10 CNN-5's first conv
where the other tables list 16; we treat that as a typo and use 16 (the fc
width does not depend on it). Order and widths match the reference repo's
targetNetModels/cnn.py (CNN2, CNN3, CNN3b, CNN3c, CNN4, CNN4b, CNN4c, CNN5,
CNN5b, CNN5c).
"""
import torch.nn as nn

from gefl_f.headers import HEADER_REGISTRY

PAPER_HEADER_WIDTHS = {
    1: (16,),
    2: (16, 32),
    3: (20, 40),
    4: (10, 20),
    5: (16, 32, 64),
    6: (20, 40, 80),
    7: (10, 20, 40),
    8: (16, 32, 64, 128),
    9: (20, 40, 80, 100),
    10: (10, 20, 40, 80),
}


class PaperFeatureExtractor(nn.Module):
    """Common feature extractor of Tables XXI-XXIII."""

    def __init__(self, in_channels: int, variant: str = "mnist"):
        super().__init__()
        if variant == "mnist":
            layers = [nn.Conv2d(in_channels, 3, 3, 1, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True)]
            self.fe_channels = 3
        elif variant == "cifar10":
            layers = [nn.Conv2d(in_channels, 3, 3, 1, 1), nn.BatchNorm2d(3), nn.ReLU(inplace=True),
                      nn.Conv2d(3, 10, 3, 1, 1), nn.BatchNorm2d(10), nn.ReLU(inplace=True)]
            self.fe_channels = 10
        else:
            raise ValueError(f"unknown paper FE variant {variant!r} (mnist|cifar10)")
        layers.append(nn.MaxPool2d(2, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    @property
    def out_channels(self):
        return self.fe_channels


class PaperHeader(nn.Module):
    """One of the ten heterogeneous headers CNN-1 ... CNN-10."""

    def __init__(self, widths, fe_channels: int, num_classes: int, fe_spatial: int = 16,
                 batch_norm: bool = False):
        super().__init__()
        layers, ci, s = [], fe_channels, fe_spatial
        for w in widths:
            layers.append(nn.Conv2d(ci, w, 3, 1, 1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(w))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2, 2)]
            ci, s = w, s // 2
        if s < 1:
            raise ValueError(f"fe_spatial={fe_spatial} too small for {len(widths)} pooling stages")
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(ci * s * s, num_classes)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def _register(idx: int, batch_norm: bool):
    name = f"cnn{idx}" + ("_bn" if batch_norm else "")

    @HEADER_REGISTRY.register(name)
    class _Header(PaperHeader):
        def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = 16):
            super().__init__(PAPER_HEADER_WIDTHS[idx], fe_channels, num_classes, fe_spatial, batch_norm)

    _Header.__name__ = _Header.__qualname__ = f"PaperCNN{idx}{'BN' if batch_norm else ''}"
    return _Header


for _i in PAPER_HEADER_WIDTHS:
    _register(_i, batch_norm=False)
    _register(_i, batch_norm=True)
