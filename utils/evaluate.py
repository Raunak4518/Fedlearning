"""
utils/evaluate.py

- `evaluate_models`: per-architecture-group accuracy on the held-out test
  set, matching the original repo's `evaluate_models(local_models,
  ws_glob, dataset_test, args, ...)` convention.
- `bucketed_accuracy` / `average_client_bucketed_accuracy`: accuracy
  broken out by class-frequency bucket (head/medium/tail), the metric
  this project's proposal commits to in its evaluation plan.
- `train_centralized_upper_bound` / `gap_report`: the centralized-vs-
  federated comparison the proposal's Section 5 specifies.
- `mnd_ratio`: GeFL's own privacy metric -- the mean nearest-neighbor
  distance ratio (Kang et al. 2025, Section IV-C): for a batch of
  synthetic samples, how much closer they sit to the real TRAINING set
  than to a held-out VALIDATION set, in flattened pixel space. A ratio
  near 1 means synthetic samples are no more suspiciously close to
  training data than to unseen data (no detectable memorization signal);
  a ratio well below 1 indicates the generator may be memorizing and
  reproducing individual training samples -- a privacy leak. This mirrors
  Monte-Carlo membership-inference attacks based on proximity, and (per
  the original paper) is intentionally architecture-agnostic so it scores
  GANs, VAEs, and diffusion models on the same footing.
"""
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


@torch.no_grad()
def evaluate_models(models: List[torch.nn.Module], ws_glob: List[dict], dataset_test, args) -> List[float]:
    accs = []
    loader = DataLoader(dataset_test, batch_size=args.bs)
    for m, w in zip(models, ws_glob):
        m.load_state_dict(w)
        m.eval()
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            pred = m(x).argmax(dim=1)
            correct += int((pred == y).sum())
            total += y.size(0)
        accs.append(correct / max(total, 1))
    return accs


@torch.no_grad()
def bucketed_accuracy(model: torch.nn.Module, dataset_test, buckets: Dict[int, str], device: str,
                       batch_size: int = 128):
    model.eval()
    loader = DataLoader(dataset_test, batch_size=batch_size)
    correct = defaultdict(int)
    total = defaultdict(int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        for c in y.unique().tolist():
            mask = y == c
            correct[c] += int((preds[mask] == c).sum())
            total[c] += int(mask.sum())

    per_class = {c: correct[c] / total[c] for c in total if total[c] > 0}
    bucket_scores = defaultdict(list)
    for c, acc in per_class.items():
        bucket_scores[buckets[c]].append(acc)
    out = {b: sum(v) / len(v) for b, v in bucket_scores.items()}
    out["overall"] = sum(per_class.values()) / max(len(per_class), 1)
    return out, per_class


def average_client_bucketed_accuracy(client_models: Dict[int, torch.nn.Module], dataset_test,
                                      buckets: Dict[int, str], device: str) -> Dict[str, float]:
    """GeFL clients keep private, architecturally heterogeneous target
    nets, so there is no single shared model to evaluate -- report the
    mean of every client's own bucketed accuracy instead."""
    all_scores = []
    for m in client_models.values():
        scores, _ = bucketed_accuracy(m, dataset_test, buckets, device)
        all_scores.append(scores)
    keys = set().union(*[s.keys() for s in all_scores]) if all_scores else set()
    return {k: sum(s.get(k, 0.0) for s in all_scores) / len(all_scores) for k in keys}


def train_centralized_upper_bound(net_cls, train_ds, num_classes: int, in_channels: int, img_size: int,
                                   args, epochs: int = None) -> torch.nn.Module:
    """One model, trained once on the full pooled (still long-tailed)
    dataset, no federation at all -- the upper bound `gap_report` compares
    federated results against."""
    epochs = epochs if epochs is not None else args.centralized_epochs
    model = net_cls(in_channels=in_channels, num_classes=num_classes, img_size=img_size).to(args.device)
    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loader = DataLoader(train_ds, batch_size=args.local_bs, shuffle=True)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


def gap_report(federated_scores: Dict[str, float], centralized_scores: Dict[str, float]) -> Dict[str, float]:
    return {b: round(centralized_scores.get(b, 0) - federated_scores.get(b, 0), 4) for b in centralized_scores}


@torch.no_grad()
def mnd_ratio(synthetic_samples: torch.Tensor, train_samples: torch.Tensor,
              val_samples: torch.Tensor, max_ref: int = 2000) -> float:
    """
    GeFL's mean nearest-neighbor distance ratio (see module docstring).
    All three tensors are (N, C, H, W); compared in flattened, normalized
    pixel space, which is what makes it architecture-agnostic (works
    identically for GAN/VAE/DDPM output). `max_ref` caps how many
    reference (train/val) samples are used for the nearest-neighbor
    search, since it's O(N_syn * N_ref).
    """
    def _flat(t):
        return t.flatten(1).float()

    syn = _flat(synthetic_samples)
    tr = _flat(train_samples)[:max_ref]
    va = _flat(val_samples)[:max_ref]

    d_train = torch.cdist(syn, tr).min(dim=1).values.mean().item()
    d_val = torch.cdist(syn, va).min(dim=1).values.mean().item()
    if d_val == 0:
        return float("nan")
    return d_train / d_val


def make_held_out_val_split(train_ds, val_fraction: float, seed: int = 0):
    """Splits off a validation subset from the training set, disjoint from
    whatever indices were handed to clients -- used only for the MND
    privacy metric, never used in training."""
    import numpy as np
    n = len(train_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val].tolist()
    remaining_idx = idx[n_val:].tolist()
    return Subset(train_ds, val_idx), remaining_idx
