import numpy as np

from sampling.partition import (apply_long_tail, class_counts_long_tailed, client_class_counts,
                                 dirichlet_partition, frequency_bucket_labels, iid_partition,
                                 partition_dataset)


def _make_labels(num_classes=10, per_class=500, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.concatenate([np.full(per_class, c) for c in range(num_classes)])
    rng.shuffle(labels)
    return labels


def test_class_counts_long_tailed_monotonic_decrease():
    labels = _make_labels()
    counts = class_counts_long_tailed(labels, 10, imbalance_factor=0.01)
    values = [counts[c] for c in range(10)]
    assert values == sorted(values, reverse=True)
    assert values[0] > values[-1]
    assert min(values) >= 1


def test_class_counts_balanced_when_imbalance_factor_is_one():
    labels = _make_labels(per_class=500)
    counts = class_counts_long_tailed(labels, 10, imbalance_factor=1.0)
    assert all(v == 500 for v in counts.values())


def test_apply_long_tail_realizes_the_schedule():
    labels = _make_labels()
    idx = apply_long_tail(labels, 10, imbalance_factor=0.02, seed=0)
    kept_labels = labels[idx]
    counts = {c: int((kept_labels == c).sum()) for c in range(10)}
    expected = class_counts_long_tailed(labels, 10, 0.02)
    assert counts == expected


def test_dirichlet_partition_covers_every_sample_exactly_once():
    labels = _make_labels(per_class=200)
    parts = dirichlet_partition(labels, num_clients=5, num_classes=10, alpha=0.3, seed=1)
    all_idx = sorted(i for idxs in parts.values() for i in idxs)
    assert all_idx == list(range(len(labels)))


def test_dirichlet_partition_respects_min_size():
    labels = _make_labels(per_class=50)
    parts = dirichlet_partition(labels, num_clients=4, num_classes=10, alpha=0.1, seed=2, min_size=5)
    assert min(len(v) for v in parts.values()) >= 5


def test_iid_partition_roughly_equal_sizes():
    labels = _make_labels(per_class=100)
    parts = iid_partition(labels, num_clients=4, seed=0)
    sizes = [len(v) for v in parts.values()]
    assert max(sizes) - min(sizes) <= 1


def test_partition_dataset_indices_are_valid_and_unique():
    labels = _make_labels(per_class=300)

    class Args:
        imbalance_factor = 0.05
        max_per_class = None
        partition_seed = 0
        noniid = True
        dir_param = 0.3
        num_users = 6

    parts = partition_dataset(labels, 10, Args())
    all_idx = [i for idxs in parts.values() for i in idxs]
    assert len(all_idx) == len(set(all_idx))  # no client shares a sample with another
    assert max(all_idx) < len(labels)
    assert min(all_idx) >= 0


def test_client_class_counts_sums_to_client_size():
    labels = _make_labels(per_class=100)
    idxs = list(range(0, 1000, 3))
    counts = client_class_counts(labels, idxs, 10)
    assert sum(counts.values()) == len(idxs)


def test_frequency_bucket_labels_head_has_the_largest_counts():
    counts = {0: 500, 1: 300, 2: 200, 3: 100, 4: 50, 5: 30, 6: 20, 7: 10, 8: 5, 9: 2}
    buckets = frequency_bucket_labels(counts, num_buckets=3)
    assert buckets[0] == "head"
    assert buckets[9] == "tail"
    assert len(set(buckets.values())) == 3


def test_imbalance_factor_zero_does_not_crash_and_stays_within_available():
    labels = _make_labels()
    counts = class_counts_long_tailed(labels, 10, imbalance_factor=0.0001)
    assert all(v >= 1 for v in counts.values())
    assert counts[0] >= counts[9]
