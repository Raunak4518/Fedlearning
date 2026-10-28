"""Fixes of 28 Oct 2026 that change only schedule or memory, never values: training images preprocessed on demand
equal the whole-set preprocessing; FedDF (which reuses GeFL-F's generator) is queued after every other run."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))
import data  # noqa: E402


def fake_cache(root, raw, n, side, ch):
    g = torch.Generator().manual_seed(0)
    d = dict(xtr=torch.randint(0, 256, (n, ch, side, side), generator=g, dtype=torch.uint8),
             ytr=torch.randint(0, 10, (n,), generator=g), xte=torch.randint(0, 256, (20, ch, side, side), generator=g,
                                                                            dtype=torch.uint8),
             yte=torch.randint(0, 10, (20,), generator=g))
    torch.save(d, os.path.join(root, f"_fedecs_tensors_{raw}.pt"))


def test_lazy_prep_equals_eager(tmp_path):
    for ds, side, ch in (("mnist", 28, 1), ("cifar10", 32, 3)):
        fake_cache(str(tmp_path), ds, 300, side, ch)
        eager = data.load_dataset(ds, str(tmp_path))
        lazy = data.load_dataset(ds, str(tmp_path), lazy=True)
        assert lazy["xtr"].dtype == torch.uint8
        idx = torch.tensor([5, 0, 299, 17, 17, 123])
        assert torch.equal(lazy["prep"](lazy["xtr"][idx]), eager["xtr"][idx])
        assert torch.equal(lazy["xte"], eager["xte"])


def test_feddf_after_its_generator():
    import build_notebooks as bn
    runs = [bn.run(m, "mnist", s) for s in (3, 4) for m in ("FedECS-LT", "GeFL-F", "FedDF", "Local", "FedLC")]
    out = bn.queue_order(runs)
    labels = [r["label"] for r in out]
    assert sorted(map(str, out), key=str) == sorted(map(str, runs), key=str)
    assert labels[-2:] == ["FedDF", "FedDF"]
    assert [r["seed"] for r in out[-2:]] == [3, 4]
    # alone (no GeFL-F in the queue) FedDF trains the generator itself and keeps its place
    alone = [bn.run(m, "mnist", 3) for m in ("FedDF", "Local")]
    assert bn.queue_order(alone) == alone
