"""
datasets/synthetic.py

A procedurally-generated image classification dataset, registered under
the exact same DATASET_REGISTRY interface as every real torchvision
dataset in datasets/vision.py. It exists for two reasons:

  1. CI / offline development: every module downstream (partitioner,
     generators, target nets, evaluator) can be exercised and unit-tested
     without a network connection or a multi-hundred-MB download.
  2. A configurable-classes debugging dataset: `--dataset synthetic
     --num_classes_synthetic N --img_size 32` lets you sanity-check the
     pipeline at any resolution/class-count combination before committing
     to a full CIFAR-10/100-LT run.

Each class gets a fixed, seeded 2D template (a sinusoidal field + a color
bias) so classes are separable but not trivially so once noise is added --
this gives target networks a non-trivial task and lets rare classes
actually be hard, which is what a long-tail benchmark needs.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.vision import DatasetMeta

# reuse the same registry instance as vision.py
from datasets.vision import DATASET_REGISTRY


def _class_template(class_id: int, img_size: int, in_channels: int, seed_base: int = 1000) -> np.ndarray:
    rng = np.random.RandomState(seed=seed_base + class_id)
    xx, yy = np.meshgrid(np.linspace(0, 1, img_size), np.linspace(0, 1, img_size))
    freq_x, freq_y = rng.uniform(1, 4, size=2)
    phase = rng.uniform(0, 2 * np.pi)
    field = np.sin(2 * np.pi * freq_x * xx + phase) * np.cos(2 * np.pi * freq_y * yy)
    color_bias = rng.uniform(-1, 1, size=in_channels)
    return np.stack([field * color_bias[c] for c in range(in_channels)], axis=0).astype(np.float32)


class SyntheticImageDataset(Dataset):
    def __init__(self, num_classes: int, samples_per_class: dict, img_size: int, in_channels: int,
                 noise_std: float = 0.55, seed: int = 0):
        self.num_classes = num_classes
        self.img_size = img_size
        self.in_channels = in_channels
        templates = [_class_template(c, img_size, in_channels) for c in range(num_classes)]
        rng = np.random.RandomState(seed)

        xs, ys = [], []
        for c in range(num_classes):
            n = samples_per_class.get(c, 0)
            if n == 0:
                continue
            noise = rng.normal(0, noise_std, size=(n, in_channels, img_size, img_size)).astype(np.float32)
            xs.append(templates[c][None] + noise)
            ys.append(np.full(n, c, dtype=np.int64))
        self.data = torch.from_numpy(np.concatenate(xs, axis=0))
        self.targets = torch.from_numpy(np.concatenate(ys, axis=0))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], int(self.targets[idx])


@DATASET_REGISTRY.register("synthetic")
def _synthetic(root, img_size, download, num_classes=10, in_channels=3,
               train_per_class=500, test_per_class=100, seed=0):
    """
    Signature intentionally matches the torchvision-backed entries
    (root, img_size, download) plus keyword-only extras so
    get_dataset.py can call every registry entry the same way; extra
    kwargs are supplied via args.synthetic_num_classes etc. when present.
    """
    size = img_size or 32
    train_counts = {c: train_per_class for c in range(num_classes)}
    test_counts = {c: test_per_class for c in range(num_classes)}
    train = SyntheticImageDataset(num_classes, train_counts, size, in_channels, seed=seed)
    test = SyntheticImageDataset(num_classes, test_counts, size, in_channels, seed=seed + 999)
    meta = DatasetMeta("synthetic", num_classes, in_channels, size,
                        mean=(0.0,) * in_channels, std=(1.0,) * in_channels)
    return train, test, meta
