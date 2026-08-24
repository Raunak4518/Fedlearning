"""
targetNetModels/nets.py

The heterogeneous target-network pool. Every architecture here is
resolution- and channel-agnostic (via nn.AdaptiveAvgPool2d before the
final classifier head, so it never needs to know img_size in advance) and
takes (in_channels, num_classes, img_size) purely as constructor
arguments -- nothing is hardcoded to a specific dataset.

`--target_models` on the command line selects any comma-separated subset
of these by name (see NET_REGISTRY.names()); GeFL's model-heterogeneous
setting is realized by simply listing more than one.
"""
import torch.nn as nn

from registry import Registry

NET_REGISTRY = Registry("target_net")


def _num_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


@NET_REGISTRY.register("cnn_small")
class CNNSmall(nn.Module):
    """Plain shallow CNN -- 2 conv blocks."""

    def __init__(self, in_channels: int, num_classes: int, img_size: int = None):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        h = self.pool(self.features(x))
        return self.classifier(h.flatten(1))


@NET_REGISTRY.register("cnn_deep")
class CNNDeep(nn.Module):
    """Deeper, narrower CNN with a residual-style skip -- structurally
    different depth/width profile from CNNSmall."""

    def __init__(self, in_channels: int, num_classes: int, img_size: int = None):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, 12, 3, padding=1)
        self.c2 = nn.Conv2d(12, 12, 3, padding=1)
        self.c3 = nn.Conv2d(12, 24, 3, padding=1, stride=2)
        self.c4 = nn.Conv2d(24, 24, 3, padding=1, stride=2)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Linear(24 * 4 * 4, num_classes)

    def forward(self, x):
        h = self.act(self.c1(x))
        h = self.act(self.c2(h)) + h
        h = self.act(self.c3(h))
        h = self.act(self.c4(h))
        h = self.pool(h)
        return self.classifier(h.flatten(1))


@NET_REGISTRY.register("mobilenet_lite")
class MobileNetLite(nn.Module):
    """Depthwise-separable convolutions (MobileNet-style) -- a third,
    architecturally distinct family from the two plain-conv nets above."""

    def __init__(self, in_channels: int, num_classes: int, img_size: int = None, width: int = 24):
        super().__init__()

        def dw_block(ci, co, stride):
            return nn.Sequential(
                nn.Conv2d(ci, ci, 3, stride, 1, groups=ci, bias=False),
                nn.BatchNorm2d(ci), nn.ReLU(inplace=True),
                nn.Conv2d(ci, co, 1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.block1 = dw_block(width, width * 2, stride=2)
        self.block2 = dw_block(width * 2, width * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.block1(h)
        h = self.block2(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


@NET_REGISTRY.register("resnet_lite")
class ResNetLite(nn.Module):
    """Small pre-activation residual network -- 3 residual stages with
    downsampling, GroupNorm (batch-size-robust for small local client
    datasets, unlike BatchNorm)."""

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

    def __init__(self, in_channels: int, num_classes: int, img_size: int = None, width: int = 20):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, 3, 1, 1)
        self.stage1 = self._Block(width, width, stride=1)
        self.stage2 = self._Block(width, width * 2, stride=2)
        self.stage3 = self._Block(width * 2, width * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width * 4, num_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


@NET_REGISTRY.register("mlp_mixer_lite")
class MLPMixerLite(nn.Module):
    """Patch-based MLP-Mixer-style network -- no convolutions at all, the
    most architecturally distinct member of the pool. Patch size adapts to
    img_size so it never divides unevenly."""

    def __init__(self, in_channels: int, num_classes: int, img_size: int = 32, hidden: int = 64, depth: int = 2):
        super().__init__()
        patch = 4
        while img_size % patch != 0 and patch > 1:
            patch -= 1
        self.patch = patch
        n_patches = (img_size // patch) ** 2
        patch_dim = in_channels * patch * patch

        self.embed = nn.Linear(patch_dim, hidden)
        self.token_mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(n_patches, n_patches), nn.GELU(),
        ) for _ in range(depth)])
        self.channel_mix = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden),
        ) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.classifier = nn.Linear(hidden, num_classes)
        self.img_size = img_size
        self.in_channels = in_channels

    def _patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # (b, c, h/p, w/p, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, (h // p) * (w // p), c * p * p)
        return x

    def forward(self, x):
        if x.shape[-1] != self.img_size:
            x = nn.functional.interpolate(x, size=(self.img_size, self.img_size), mode="nearest")
        tokens = self.embed(self._patchify(x))  # (b, n_patches, hidden)
        for tmix, cmix in zip(self.token_mix, self.channel_mix):
            h = tmix[0](tokens).transpose(1, 2)
            h = tmix[2](tmix[1](h)).transpose(1, 2)
            tokens = tokens + h
            tokens = tokens + cmix(tokens)
        tokens = self.norm(tokens)
        return self.classifier(tokens.mean(dim=1))


# ============================================================
#  Paper's 10-CNN family (Table XXIII, Kang et al. 2025)
#
#  All share the same stem:
#    conv(3, 3×3, pad=1) → bn → relu → conv(10, 3×3, pad=1) → bn → relu → maxpool(2×2)
#
#  Then each CNN branches into a different number of additional
#  {conv → relu → maxpool} stages at different channel widths,
#  ending in one FC layer → num_classes.
#
#  CNN-1 (deepest) and CNN-4 (shallowest) are exact from the table.
#  CNN-2,3,5-10 are reasonable interpolations — the specific channel
#  numbers in a few cells were inconsistent in the extracted text.
#  [LIKELY, VERIFY]: cross-check against the actual PDF table.
# ============================================================

class PaperCNN(nn.Module):
    """Configurable CNN matching the paper's Table XXIII structure.

    Args:
        in_channels: image input channels
        num_classes: number of classes
        img_size: unused (AdaptiveAvgPool handles any size)
        post_stem_channels: list of channel widths for post-stem conv stages
            e.g. [16, 32, 64, 128] for CNN-1 (4 additional stages)
    """

    def __init__(self, in_channels: int, num_classes: int, img_size: int = None,
                 post_stem_channels: list = None):
        super().__init__()
        if post_stem_channels is None:
            post_stem_channels = [16, 32]

        # Shared stem: conv(3→3) → bn → relu → conv(3→10) → bn → relu → maxpool
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 10, 3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Post-stem stages: each is conv → relu → maxpool
        layers = []
        ch_in = 10
        for ch_out in post_stem_channels:
            layers.extend([
                nn.Conv2d(ch_in, ch_out, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
            ch_in = ch_out
        self.body = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch_in, num_classes)

    def forward(self, x):
        h = self.stem(x)
        h = self.body(h)
        h = self.pool(h).flatten(1)
        return self.classifier(h)


# Register 10 variants with the depth/width profiles from Table XXIII.
# CNN-1 is the deepest (4 post-stem stages), CNN-10 is the shallowest.

_PAPER_CNN_CONFIGS = {
    # name:            post_stem_channels
    "paper_cnn_1":     [16, 32, 64, 128],     # deepest: 4 stages, fc(128→C)
    "paper_cnn_2":     [16, 32, 64, 96],      # 4 stages, narrower final
    "paper_cnn_3":     [16, 32, 64],           # 3 stages
    "paper_cnn_4":     [10, 32],               # shallowest: 2 stages, fc(32*spatial→C)
    "paper_cnn_5":     [16, 48, 96],           # 3 stages, wider
    "paper_cnn_6":     [16, 32, 48],           # 3 stages, narrower
    "paper_cnn_7":     [20, 40, 80, 128],      # 4 stages, different widths
    "paper_cnn_8":     [12, 24, 48],           # 3 stages, narrow
    "paper_cnn_9":     [10, 20, 40, 80],       # 4 stages, narrow throughout
    "paper_cnn_10":    [16, 32, 64, 64],       # 4 stages, flat final
}


def _make_paper_cnn_factory(channels):
    """Create a factory function for a specific PaperCNN variant."""
    def factory(in_channels: int, num_classes: int, img_size: int = None):
        return PaperCNN(in_channels, num_classes, img_size, post_stem_channels=list(channels))
    return factory


for _name, _channels in _PAPER_CNN_CONFIGS.items():
    NET_REGISTRY.register(_name)(_make_paper_cnn_factory(_channels))

