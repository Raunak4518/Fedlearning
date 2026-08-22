"""
datasets/vision.py

Registers every torchvision-backed dataset under a single common
interface: DATASET_REGISTRY.get(name)(root, img_size, download) ->
(train_dataset, test_dataset, meta), where meta is a DatasetMeta describing
num_classes / in_channels / native img_size / normalization stats.

Every entry returns PLAIN (PIL-free) tensors so nothing downstream needs
to know which dataset it is looking at -- the whole rest of the codebase
(partitioner, generators, target nets, evaluator) only ever sees
`(image_tensor, int_label)` pairs of shape (in_channels, img_size, img_size).

To add a new torchvision (or custom) dataset, add one function here with
`@DATASET_REGISTRY.register("name")` -- nothing elsewhere changes.
"""
from dataclasses import dataclass

import torchvision.transforms as T
from torchvision import datasets as tvd

from registry import Registry

DATASET_REGISTRY = Registry("dataset")


@dataclass
class DatasetMeta:
    name: str
    num_classes: int
    in_channels: int
    native_img_size: int
    mean: tuple
    std: tuple


def _transform(img_size: int, in_channels: int, mean: tuple, std: tuple) -> T.Compose:
    ops = [T.Resize((img_size, img_size))]
    if in_channels == 3:
        ops.append(T.Lambda(lambda im: im.convert("RGB")))
    elif in_channels == 1:
        ops.append(T.Grayscale(num_output_channels=1))
    ops += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


def _build(cls, root, img_size, download, in_channels, num_classes, native_size, mean, std, name,
           train_kwargs=None, test_kwargs=None):
    train_kwargs = train_kwargs or {}
    test_kwargs = test_kwargs or {}
    tfm = _transform(img_size or native_size, in_channels, mean, std)
    train = cls(root=root, download=download, transform=tfm, **train_kwargs)
    test = cls(root=root, download=download, transform=tfm, **test_kwargs)
    meta = DatasetMeta(name, num_classes, in_channels, img_size or native_size, mean, std)
    return train, test, meta


@DATASET_REGISTRY.register("cifar10")
def _cifar10(root, img_size, download):
    return _build(tvd.CIFAR10, root, img_size, download, in_channels=3, num_classes=10, native_size=32,
                   mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), name="cifar10",
                   train_kwargs=dict(train=True), test_kwargs=dict(train=False))


@DATASET_REGISTRY.register("cifar100")
def _cifar100(root, img_size, download):
    return _build(tvd.CIFAR100, root, img_size, download, in_channels=3, num_classes=100, native_size=32,
                   mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762), name="cifar100",
                   train_kwargs=dict(train=True), test_kwargs=dict(train=False))


@DATASET_REGISTRY.register("mnist")
def _mnist(root, img_size, download):
    return _build(tvd.MNIST, root, img_size, download, in_channels=1, num_classes=10, native_size=28,
                   mean=(0.1307,), std=(0.3081,), name="mnist",
                   train_kwargs=dict(train=True), test_kwargs=dict(train=False))


@DATASET_REGISTRY.register("fmnist")
def _fmnist(root, img_size, download):
    return _build(tvd.FashionMNIST, root, img_size, download, in_channels=1, num_classes=10, native_size=28,
                   mean=(0.2860,), std=(0.3530,), name="fmnist",
                   train_kwargs=dict(train=True), test_kwargs=dict(train=False))


@DATASET_REGISTRY.register("svhn")
def _svhn(root, img_size, download):
    return _build(tvd.SVHN, root, img_size, download, in_channels=3, num_classes=10, native_size=32,
                   mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970), name="svhn",
                   train_kwargs=dict(split="train"), test_kwargs=dict(split="test"))


@DATASET_REGISTRY.register("stl10")
def _stl10(root, img_size, download):
    return _build(tvd.STL10, root, img_size, download, in_channels=3, num_classes=10, native_size=96,
                   mean=(0.4467, 0.4398, 0.4066), std=(0.2603, 0.2566, 0.2713), name="stl10",
                   train_kwargs=dict(split="train"), test_kwargs=dict(split="test"))
