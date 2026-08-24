"""
gefl_f/headers.py

Heterogeneous header networks for GeFL-F. Each header takes the
feature extractor's output (a feature map) as input and produces
class logits. Multiple header architectures provide the model
heterogeneity — each client runs the shared FE + one of these headers.

The headers are structurally similar to the existing target nets but
operate on feature maps (fe_channels channels, half the spatial
resolution of input images due to the FE's MaxPool2d) rather than raw
images.
"""
import torch.nn as nn

from registry import Registry

HEADER_REGISTRY = Registry("header")


def _num_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


@HEADER_REGISTRY.register("header_small")
class HeaderSmall(nn.Module):
    """Shallow CNN header — 2 conv blocks on top of the feature extractor."""

    def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(fe_channels, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.classifier = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x):
        h = self.pool(self.features(x))
        return self.classifier(h.flatten(1))


@HEADER_REGISTRY.register("header_deep")
class HeaderDeep(nn.Module):
    """Deeper header with a residual skip connection."""

    def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = None):
        super().__init__()
        self.c1 = nn.Conv2d(fe_channels, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.c4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.classifier = nn.Linear(128 * 2 * 2, num_classes)

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.act(self.c2(h)) + h
        h = self.act(self.c3(h))
        h = self.act(self.c4(h))
        h = self.pool(h)
        return self.classifier(h.flatten(1))


@HEADER_REGISTRY.register("header_wide")
class HeaderWide(nn.Module):
    """Wide but shallow header — more channels, fewer layers."""

    def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(fe_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(2)
        self.classifier = nn.Linear(256 * 2 * 2, num_classes)

    def forward(self, x):
        h = self.pool(self.features(x))
        return self.classifier(h.flatten(1))


@HEADER_REGISTRY.register("header_residual")
class HeaderResidual(nn.Module):
    """Header with pre-activation residual blocks and GroupNorm."""

    class _Block(nn.Module):
        def __init__(self, ci, co, stride):
            super().__init__()
            self.n1 = nn.GroupNorm(_num_groups(ci), ci)
            self.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
            self.n2 = nn.GroupNorm(_num_groups(co), co)
            self.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
            self.skip = nn.Conv2d(ci, co, 1, stride, bias=False) if (ci != co or stride != 1) else nn.Identity()
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):
            h = self.c1(self.act(self.n1(x)))
            h = self.c2(self.act(self.n2(h)))
            return h + self.skip(x)

    def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = None):
        super().__init__()
        self.stage1 = self._Block(fe_channels, 64, stride=1)
        self.stage2 = self._Block(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        h = self.stage1(x)
        h = self.stage2(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


@HEADER_REGISTRY.register("header_mlp")
class HeaderMLP(nn.Module):
    """Flatten-and-MLP header — no convolutions, maximally different
    from the convolutional headers above."""

    def __init__(self, fe_channels: int, num_classes: int, fe_spatial: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(4)
        flat_dim = fe_channels * 4 * 4
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(self.pool(x))
