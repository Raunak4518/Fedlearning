"""
gefl_f/feature_extractor.py

The common feature extractor (FE) shared by ALL clients in GeFL-F,
regardless of which heterogeneous header they run.

Section IV-A:
    "designed with a single convolutional layer followed by batch
    normalization and pooling layers"

This is deliberately minimal — the point is low-capacity shared
communication (the feature tensor is much smaller than a full image),
while all discriminative power lives in the per-architecture headers.

Architecture: Conv2d → BatchNorm2d → ReLU → MaxPool2d
    - Output spatial size = input_size / 2 (from the pool)
    - Output channels = fe_channels (default 32)

The paper doesn't specify exact channel count or kernel parameters;
fe_channels=32, kernel=3x3, stride=1, pad=1 is our documented choice.
"""
import torch.nn as nn


class CommonFeatureExtractor(nn.Module):
    """Single conv→bn→relu→pool block shared across all clients.

    Args:
        in_channels: image input channels (e.g. 3 for CIFAR10)
        fe_channels: output feature channels (default 32)
        kernel_size: conv kernel size (default 3)
    """

    def __init__(self, in_channels: int, fe_channels: int = 32,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.fe_channels = fe_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, fe_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(fe_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.net(x)

    @property
    def out_channels(self):
        return self.fe_channels
