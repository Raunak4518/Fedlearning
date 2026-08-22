"""
utils/metrics.py

Extended evaluation metrics beyond bucketed accuracy. Designed for
class-imbalanced federated learning where overall accuracy is misleading:

- Confusion matrix (full NxN)
- Per-class precision, recall, F1
- Macro / weighted averages
- Class-balanced accuracy (mean of per-class accuracies)
- Convergence detection (plateau / divergence)
- Per-client evaluation (each client's model on the global test set)
- Generator quality via feature-space FID-lite (no InceptionV3 needed)
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


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
        for true_c, pred_c in zip(y.cpu().numpy(), preds.cpu().numpy()):
            cm[true_c, pred_c] += 1
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
