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

# Transform options, set once per run: get_dataset() calls configure_transforms().
#   normalize: "dataset" (per-dataset mean/std), "half" ((0.5,), (0.5,)), or "none"
#   augment:   random crop (pad 4) + horizontal flip on the TRAIN split only
# The GeFL reference uses none for MNIST, "half" for FMNIST and crop+flip for
# CIFAR-10 (utils/getData.py), so paper-parity configs set these explicitly.
_TRANSFORM_OPTS = {"normalize": "dataset", "augment": False}


def configure_transforms(normalize: str = "dataset", augment: bool = False) -> None:
    if normalize not in ("dataset", "half", "none"):
        raise ValueError(f"normalize must be dataset|half|none, got {normalize!r}")
    _TRANSFORM_OPTS["normalize"] = normalize
    _TRANSFORM_OPTS["augment"] = bool(augment)


@dataclass
class DatasetMeta:
    name: str
    num_classes: int
    in_channels: int
    native_img_size: int
    mean: tuple
    std: tuple


def _transform(img_size: int, in_channels: int, mean: tuple, std: tuple, train: bool = False) -> T.Compose:
    ops = [T.Resize((img_size, img_size))]
    if in_channels == 3:
        ops.append(T.Lambda(lambda im: im.convert("RGB")))
    elif in_channels == 1:
        ops.append(T.Grayscale(num_output_channels=1))
    if train and _TRANSFORM_OPTS["augment"]:
        ops += [T.RandomCrop(img_size, padding=4), T.RandomHorizontalFlip()]
    ops.append(T.ToTensor())
    if _TRANSFORM_OPTS["normalize"] == "dataset":
        ops.append(T.Normalize(mean, std))
    elif _TRANSFORM_OPTS["normalize"] == "half":
        ops.append(T.Normalize((0.5,) * in_channels, (0.5,) * in_channels))
    return T.Compose(ops)


def _effective_stats(in_channels: int, mean: tuple, std: tuple):
    """(mean, std) that undo the active normalization, for code that denormalizes."""
    mode = _TRANSFORM_OPTS["normalize"]
    if mode == "half":
        return (0.5,) * in_channels, (0.5,) * in_channels
    if mode == "none":
        return (0.0,) * in_channels, (1.0,) * in_channels
    return mean, std


def _build(cls, root, img_size, download, in_channels, num_classes, native_size, mean, std, name,
           train_kwargs=None, test_kwargs=None):
    train_kwargs = train_kwargs or {}
    test_kwargs = test_kwargs or {}
    size = img_size or native_size
    train = cls(root=root, download=download, transform=_transform(size, in_channels, mean, std, train=True),
                **train_kwargs)
    test = cls(root=root, download=download, transform=_transform(size, in_channels, mean, std, train=False),
               **test_kwargs)
    mean, std = _effective_stats(in_channels, mean, std)
    meta = DatasetMeta(name, num_classes, in_channels, size, mean, std)
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
