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
import numpy as np

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
def _get_lpips_fn(device: str):
    try:
        import lpips
        return lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    except ImportError:
        return None

@torch.no_grad()
def mnd_ratio(query_samples: torch.Tensor, ref_synth: torch.Tensor,
              ref_val: torch.Tensor, n_query: int = 1000,
              n_ref: int = 600, device: str = None,
              dataset_mean: tuple = None, dataset_std: tuple = None) -> float:
    if device is None:
        device = query_samples.device

    # Subsample to specified sizes
    n_q = min(n_query, query_samples.size(0))
    perm_q = torch.randperm(query_samples.size(0))[:n_q]
    query = query_samples[perm_q].to(device)

    n_s = min(n_ref, ref_synth.size(0))
    synth = ref_synth[torch.randperm(ref_synth.size(0))[:n_s]].to(device)

    n_v = min(n_ref, ref_val.size(0))
    val = ref_val[torch.randperm(ref_val.size(0))[:n_v]].to(device)

    # Scale real images (query and val) to [-1, 1] to match synth
    if dataset_mean is not None and dataset_std is not None:
        mean_t = torch.tensor(dataset_mean, device=device).view(1, -1, 1, 1)
        std_t = torch.tensor(dataset_std, device=device).view(1, -1, 1, 1)
        # Denormalize to [0, 1] then scale to [-1, 1]
        query = (query * std_t + mean_t) * 2.0 - 1.0
        val = (val * std_t + mean_t) * 2.0 - 1.0

    # LPIPS expects 3 channels
    if query.size(1) == 1:
        query = query.repeat(1, 3, 1, 1)
        synth = synth.repeat(1, 3, 1, 1)
        val = val.repeat(1, 3, 1, 1)

    # Try LPIPS, fall back to Euclidean
    lpips_fn = _get_lpips_fn(str(device))

    if lpips_fn is not None:
        # LPIPS works on (N, C, H, W) in [-1, 1]; compute pairwise in batches
        ratios = []
        batch_sz = 64
        for i in range(0, n_q, batch_sz):
            q_batch = query[i:i + batch_sz]  # (B, C, H, W)
            b = q_batch.size(0)

            # d(x_i, synth) for each query sample
            d_synth = torch.zeros(b, n_s, device=device)
            for j in range(0, n_s, batch_sz):
                s_batch = synth[j:j + batch_sz]
                s_len = s_batch.size(0)
                # Expand: (B,1,C,H,W) vs (1,S,C,H,W) -> pairwise
                q_exp = q_batch.unsqueeze(1).expand(-1, s_len, -1, -1, -1).reshape(-1, *q_batch.shape[1:])
                s_exp = s_batch.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(-1, *s_batch.shape[1:])
                d = lpips_fn(q_exp, s_exp).view(b, s_len)
                d_synth[:, j:j + s_len] = d

            # d(x_i, val) for each query sample
            d_val = torch.zeros(b, n_v, device=device)
            for j in range(0, n_v, batch_sz):
                v_batch = val[j:j + batch_sz]
                v_len = v_batch.size(0)
                q_exp = q_batch.unsqueeze(1).expand(-1, v_len, -1, -1, -1).reshape(-1, *q_batch.shape[1:])
                v_exp = v_batch.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(-1, *v_batch.shape[1:])
                d = lpips_fn(q_exp, v_exp).view(b, v_len)
                d_val[:, j:j + v_len] = d

            min_d_synth = d_synth.min(dim=1).values
            min_d_val = d_val.min(dim=1).values
            # ρ_i = min_d_val / min_d_synth (paper Eq. 1)
            valid = min_d_synth > 0
            if valid.any():
                ratios.append((min_d_val[valid] / min_d_synth[valid]).cpu())

        if not ratios:
            return float("nan")
        return float(torch.cat(ratios).mean().item())
    else:
        # Fallback: Euclidean in flattened pixel space
        def _flat(t):
            return t.flatten(1).float()

        q_flat = _flat(query)
        s_flat = _flat(synth)
        v_flat = _flat(val)

        min_d_synth = torch.cdist(q_flat, s_flat).min(dim=1).values
        min_d_val = torch.cdist(q_flat, v_flat).min(dim=1).values
        valid = min_d_synth > 0
        if not valid.any():
            return float("nan")
        return float((min_d_val[valid] / min_d_synth[valid]).mean().item())


def make_held_out_val_split(train_ds, val_fraction: float, seed: int = 0):
    import numpy as np
    n = len(train_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val].tolist()
    remaining_idx = idx[n_val:].tolist()
    from torch.utils.data import Subset
    return Subset(train_ds, val_idx), remaining_idx


@torch.no_grad()
def compute_confusion_matrix(model: torch.nn.Module, dataset_test, num_classes: int,
                             device: str, batch_size: int = 128) -> np.ndarray:
    """Returns a (num_classes, num_classes) confusion matrix.
    Row = true label, column = predicted label."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    loader = DataLoader(dataset_test, batch_size=batch_size)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        indices = num_classes * y + preds
        cm += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes).cpu().numpy()
    return cm


def classification_report_from_cm(cm: np.ndarray) -> Dict[str, dict]:
    """Computes per-class and aggregate precision/recall/F1 from a
    confusion matrix. Returns a dict with keys for each class index
    plus 'macro_avg', 'weighted_avg', and 'class_balanced_accuracy'."""
    num_classes = cm.shape[0]
    report = {}
    supports = cm.sum(axis=1)
    total_support = supports.sum()

    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        report[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(supports[c]),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    weights = supports / max(total_support, 1)

    report["macro_avg"] = {
        "precision": float(precisions.mean()),
        "recall": float(recalls.mean()),
        "f1": float(f1s.mean()),
    }
    report["weighted_avg"] = {
        "precision": float((precisions * weights).sum()),
        "recall": float((recalls * weights).sum()),
        "f1": float((f1s * weights).sum()),
    }
    report["class_balanced_accuracy"] = float(recalls.mean())
    return report


def per_class_accuracy(cm: np.ndarray) -> Dict[int, float]:
    """Per-class accuracy (recall) extracted from confusion matrix."""
    num_classes = cm.shape[0]
    result = {}
    for c in range(num_classes):
        total = cm[c, :].sum()
        result[c] = float(cm[c, c] / max(total, 1))
    return result


class ConvergenceTracker:
    """Tracks accuracy over rounds and detects plateau/divergence."""

    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.history: List[float] = []
        self.best_acc = -float("inf")
        self.rounds_without_improvement = 0

    def update(self, accuracy: float) -> Dict[str, object]:
        self.history.append(accuracy)
        delta_from_prev = accuracy - self.history[-2] if len(self.history) > 1 else 0.0

        if accuracy > self.best_acc + self.delta:
            self.best_acc = accuracy
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        return {
            "acc_delta": float(delta_from_prev),
            "best_acc": float(self.best_acc),
            "rounds_without_improvement": self.rounds_without_improvement,
            "is_plateau": self.rounds_without_improvement >= self.patience,
            "is_diverging": len(self.history) >= 3 and all(
                self.history[-(i + 1)] < self.history[-(i + 2)]
                for i in range(min(3, len(self.history) - 1))
            ),
        }


@torch.no_grad()
def per_client_accuracy(client_models: Dict[int, torch.nn.Module], dataset_test,
                        device: str, batch_size: int = 128) -> Dict[int, float]:
    """Evaluate each client's model on the full test set independently."""
    loader = DataLoader(dataset_test, batch_size=batch_size)
    results = {}

    # Pre-load test data to avoid re-iterating for each client
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x)
        all_y.append(y)

    for cid, model in client_models.items():
        model.eval()
        correct, total = 0, 0
        for x, y in zip(all_x, all_y):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum())
            total += y.size(0)
        results[cid] = correct / max(total, 1)

    return results


@torch.no_grad()
def generator_quality_metrics(gennet, target_model: torch.nn.Module,
                              real_dataset, num_classes: int,
                              device: str, n_samples: int = 200) -> Dict[str, float]:
    """Lightweight generator quality assessment using the target model's
    feature space (no InceptionV3 needed). Measures:
    - mean_confidence: average target-net softmax confidence on generated samples
    - label_accuracy: fraction where the target-net's argmax matches the conditioning label
    - per_class_confidence: mean confidence broken out by class
    """
    gennet.eval()
    target_model.eval()

    labels = torch.randint(0, num_classes, (n_samples,), device=device)
    synth = gennet.sample(labels)
    logits = target_model(synth)
    probs = F.softmax(logits, dim=1)
    predicted = logits.argmax(dim=1)

    mean_conf = probs.gather(1, labels.unsqueeze(1)).mean().item()
    label_acc = (predicted == labels).float().mean().item()

    per_class_conf = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_conf[c] = probs[mask, c].mean().item()

    return {
        "gen_mean_confidence": mean_conf,
        "gen_label_accuracy": label_acc,
        "gen_per_class_confidence": per_class_conf,
    }

def average_client_metrics(client_models, dataset_test, num_classes: int, buckets, device: str):
    all_scores = []
    
    from collections import defaultdict
    for m in client_models.values():
        cm = compute_confusion_matrix(m, dataset_test, num_classes, device)
        report = classification_report_from_cm(cm)
        
        recalls = [report[c]["recall"] for c in range(num_classes)]
        bucket_scores = defaultdict(list)
        for c, acc in enumerate(recalls):
            bucket_scores[buckets[c]].append(acc)
            
        scores = {b: sum(v) / max(len(v), 1) for b, v in bucket_scores.items()}
        scores["overall"] = report["class_balanced_accuracy"]
        scores["class_balanced_accuracy"] = report["class_balanced_accuracy"]
        scores["macro_f1"] = report["macro_avg"]["f1"]
        scores["weighted_f1"] = report["weighted_avg"]["f1"]
        scores["macro_precision"] = report["macro_avg"]["precision"]
        scores["macro_recall"] = report["macro_avg"]["recall"]
        
        all_scores.append(scores)
        
    keys = set().union(*[s.keys() for s in all_scores]) if all_scores else set()
    return {k: sum(s.get(k, 0.0) for s in all_scores) / len(all_scores) for k in keys}
