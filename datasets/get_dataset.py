"""
datasets/get_dataset.py

The single function the rest of the codebase calls to obtain data:

    train_ds, test_ds, meta, train_labels = get_dataset(args)

Nothing outside this file (and the per-dataset registrations in vision.py
/ synthetic.py) knows or cares whether the dataset is CIFAR-10, CIFAR-100,
SVHN, MNIST, or the synthetic fallback -- the rest of the pipeline is
written entirely against `meta.num_classes`, `meta.in_channels`,
`meta.native_img_size` (or the user's --img_size override), and a plain
`train_labels` numpy array. Adding a new dataset never touches this file.
"""
import logging
import warnings

import numpy as np

import datasets.synthetic  # noqa: F401 -- registers "synthetic"
import datasets.vision  # noqa: F401 -- registers cifar10/100, mnist, fmnist, svhn, stl10
from datasets.vision import DATASET_REGISTRY
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)


def _extract_labels(dataset) -> np.ndarray:
    """Works across torchvision's inconsistent label attribute naming
    (CIFAR/MNIST/FashionMNIST use `.targets`, SVHN/STL10 use `.labels`)
    plus our own SyntheticImageDataset (`.targets`), without the caller
    needing to know which dataset this is."""
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            val = getattr(dataset, attr)
            return np.asarray(val).reshape(-1)
    # last resort: iterate (works for any Dataset, just slower)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def get_dataset(args):
    """
    Returns:
        train_ds, test_ds : torch.utils.data.Dataset  (image_tensor, int_label)
        meta               : datasets.vision.DatasetMeta
        train_labels        : np.ndarray, shape (len(train_ds),)
    """
    name = args.dataset.lower()
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown --dataset '{args.dataset}'. Registered: {DATASET_REGISTRY.names()}")

    builder = DATASET_REGISTRY.get(name)
    try:
        train_ds, test_ds, meta = builder(root=args.data_root, img_size=args.img_size, download=True)
    except Exception as e:  # noqa: BLE001 -- deliberately broad: any download/network failure
        if name == "synthetic":
            raise
        warnings.warn(
            f"Could not load real dataset '{name}' ({type(e).__name__}: {e}). "
            f"Falling back to --dataset synthetic so the pipeline can still run offline. "
            f"Re-run with network access to use the real dataset.",
            RuntimeWarning,
        )
        builder = DATASET_REGISTRY.get("synthetic")
        train_ds, test_ds, meta = builder(root=args.data_root, img_size=args.img_size, download=False)

    train_labels = _extract_labels(train_ds)
    logger.info("Loaded dataset=%s  num_classes=%d  in_channels=%d  img_size=%d  train_n=%d  test_n=%d",
                meta.name, meta.num_classes, meta.in_channels, meta.native_img_size, len(train_ds), len(test_ds))
    return train_ds, test_ds, meta, train_labels
