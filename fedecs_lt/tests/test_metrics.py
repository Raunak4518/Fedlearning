"""metrics.class_metrics agrees with scikit-learn and with experiments/core.py._class_metrics."""
import os
import sys

import numpy as np
import torch
from sklearn import metrics as skm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "experiments"))

import core  # noqa: E402
import metrics  # noqa: E402

core.DEV = "cpu"


def test_against_sklearn_and_core():
    rng = np.random.RandomState(0)
    for C in (10, 7, 100):
        N = 3000
        y = torch.from_numpy(rng.randint(0, C, N))
        logits = torch.from_numpy(rng.randn(N, C).astype(np.float32)) + 2.0 * torch.nn.functional.one_hot(y, C)
        m = metrics.class_metrics(logits, y, C, prior=np.full(C, 1.0 / C))
        yy, pred = y.numpy(), logits.argmax(1).numpy()
        prob = logits.softmax(1).double().numpy()
        assert abs(m["acc"] - skm.accuracy_score(yy, pred)) < 1e-9
        assert abs(m["bal"] - skm.balanced_accuracy_score(yy, pred)) < 1e-9
        assert abs(m["f1"] - skm.f1_score(yy, pred, average="macro", zero_division=0)) < 1e-9
        assert abs(m["precision"] - skm.precision_score(yy, pred, average="macro", zero_division=0)) < 1e-9
        assert abs(m["mcc"] - skm.matthews_corrcoef(yy, pred)) < 1e-9
        assert abs(m["kappa"] - skm.cohen_kappa_score(yy, pred)) < 1e-9
        assert np.allclose(m["recall_c"], skm.recall_score(yy, pred, average=None, zero_division=0))
        assert np.array_equal(np.array(m["confusion"]), skm.confusion_matrix(yy, pred, labels=range(C)))
        assert abs(m["top5"] - skm.top_k_accuracy_score(yy, prob, k=min(5, C), labels=range(C))) < 1e-9
        assert abs(m["auc_roc"] - skm.roc_auc_score(yy, prob, multi_class="ovr", average="macro", labels=range(C))) < 1e-6
        assert abs(m["brier"] - ((prob - np.eye(C)[yy]) ** 2).sum(1).mean()) < 1e-9
        assert abs(m["nll"] - skm.log_loss(yy, np.clip(prob, 1e-6, 1), labels=range(C))) < 1e-3
        core.NUM_CLASSES = C
        r = core._class_metrics(logits, y, C, prior=np.full(C, 1.0 / C))
        for k in ("acc", "precision", "f1", "gmean", "ece", "nll", "brier", "local"):
            assert abs(r[k] - m[k]) < 1e-5, (k, r[k], m[k])
    core.NUM_CLASSES = 10
