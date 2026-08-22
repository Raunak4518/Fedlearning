"""
sampling/partition.py

Everything here operates purely on a `labels: np.ndarray` and
`num_classes: int` -- no dataset-specific code, so it works identically
whether `labels` came from CIFAR-100, SVHN, or the synthetic fallback.

Three independent knobs, matching the controls the proposal specifies
(the same two CReFF itself uses, plus the long-tail severity applied
before partitioning):

    imbalance_factor  -- global long-tail severity (min/max class count)
    dir_param (alpha)  -- cross-client heterogeneity (Dirichlet concentration)
    noniid              -- if False, ignore alpha and use a plain IID shard split
"""
from typing import Dict, List

import numpy as np


def class_counts_long_tailed(labels: np.ndarray, num_classes: int, imbalance_factor: float,
                              max_per_class: int = None) -> Dict[int, int]:
    """
    Standard exponential long-tail schedule (Cui et al. 2019 / CReFF):
    imbalance_factor = min_count / max_count, classes 0..num_classes-1
    decay geometrically between them. imbalance_factor=1.0 means no long
    tail is applied at all -- every class keeps its full original count.
    """
    available = {c: int((labels == c).sum()) for c in range(num_classes)}
    natural_max = max(available.values()) if available else 0
    cap = min(max_per_class, natural_max) if max_per_class is not None else natural_max

    if imbalance_factor >= 1.0:
        return {c: available[c] for c in range(num_classes)}

    mu = imbalance_factor ** (1.0 / max(num_classes - 1, 1))
    target = {c: max(1, int(cap * (mu ** c))) for c in range(num_classes)}
    # never ask for more samples of a class than actually exist
    return {c: min(target[c], available[c]) for c in range(num_classes)}


def apply_long_tail(labels: np.ndarray, num_classes: int, imbalance_factor: float,
                     max_per_class: int = None, seed: int = 0) -> np.ndarray:
    """Returns a subset of sample INDICES (into the original `labels` array)
    realizing the long-tail class-count schedule above, via random
    subsampling of the majority classes -- the underlying dataset is never
    modified, only which indices get used downstream."""
    rng = np.random.RandomState(seed)
    counts = class_counts_long_tailed(labels, num_classes, imbalance_factor, max_per_class)
    keep = []
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        keep.append(idx_c[:counts[c]])
    keep = np.concatenate(keep) if keep else np.array([], dtype=int)
    rng.shuffle(keep)
    return keep


def dirichlet_partition(labels: np.ndarray, num_clients: int, num_classes: int,
                         alpha: float, seed: int = 0, min_size: int = 1) -> Dict[int, List[int]]:
    """
    Split sample indices across clients so each client's per-class share
    is drawn from Dirichlet(alpha) -- low alpha => clients hold skewed,
    heterogeneous class mixes; high alpha => close to IID. Indices in
    `labels` may already be a long-tail subset (apply_long_tail runs first);
    this function only ever sees whatever indices it's given.

    Retries with fresh Dirichlet draws if any client would end up with
    fewer than `min_size` samples, so no client is left with an unusable
    empty (or near-empty) local dataset.
    """
    rng = np.random.RandomState(seed)
    n = len(labels)
    for _attempt in range(50):
        client_indices = {i: [] for i in range(num_clients)}
        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            if len(idx_c) == 0:
                continue
            rng.shuffle(idx_c)
            proportions = rng.dirichlet(alpha=[alpha] * num_clients)
            split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            for i, split in enumerate(np.split(idx_c, split_points)):
                client_indices[i].extend(split.tolist())
        sizes = [len(v) for v in client_indices.values()]
        if min(sizes) >= min_size or n < num_clients * min_size:
            break
    for i in client_indices:
        rng.shuffle(client_indices[i])
    return client_indices


def iid_partition(labels: np.ndarray, num_clients: int, seed: int = 0) -> Dict[int, List[int]]:
    """Plain IID shard split -- equal-size random shards, ignoring class identity."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    shards = np.array_split(idx, num_clients)
    return {i: shard.tolist() for i, shard in enumerate(shards)}


def partition_dataset(labels: np.ndarray, num_classes: int, args) -> Dict[int, List[int]]:
    """Top-level entry point used by utils/setup.py. Applies the long-tail
    schedule first (if imbalance_factor < 1.0), then splits what remains
    across clients either IID or via Dirichlet, per args.noniid."""
    kept_idx = apply_long_tail(labels, num_classes, args.imbalance_factor,
                                args.max_per_class, seed=args.partition_seed)
    kept_labels = labels[kept_idx]

    if args.noniid:
        local_partition = dirichlet_partition(kept_labels, args.num_users, num_classes,
                                               args.dir_param, seed=args.partition_seed)
    else:
        local_partition = iid_partition(kept_labels, args.num_users, seed=args.partition_seed)

    # map back from indices-into-kept_labels to indices-into-the-original dataset
    return {cid: [int(kept_idx[j]) for j in local_idxs] for cid, local_idxs in local_partition.items()}


def client_class_counts(labels: np.ndarray, indices: List[int], num_classes: int) -> Dict[int, int]:
    """Per-class sample count for one client's index list -- used by Mechanism A's
    per-class effective-number aggregation weight."""
    sub = labels[indices]
    return {c: int((sub == c).sum()) for c in range(num_classes)}


def frequency_bucket_labels(class_counts: Dict[int, int], num_buckets: int = 3) -> Dict[int, str]:
    """Splits classes into 'head' / 'medium' / 'tail' (or more generally
    `num_buckets` buckets) by rank in `class_counts`, for bucketed
    evaluation -- purely rank-based, so it works for any num_classes."""
    names = ["head", "medium", "tail"] if num_buckets == 3 else [f"bucket_{i}" for i in range(num_buckets)]
    ordered = sorted(class_counts, key=lambda c: -class_counts[c])
    n = len(ordered)
    per_bucket = max(1, n // num_buckets)
    buckets = {}
    for rank, c in enumerate(ordered):
        b = min(rank // per_bucket, num_buckets - 1)
        buckets[c] = names[b]
    return buckets
