"""Datasets, long-tail construction and client partition.

Tier-1 loading, preprocessing and make_partition are copied from experiments/core.py (load_dataset, make_partition,
assign_arch, permute_labels) so that the same seed gives the same clients as every earlier run; the unit test
tests/test_equivalence.py checks this against core.py. New datasets (tier 2) use the same preprocessing rules.
Only file formats that cannot run code are read (parquet, npz without pickle, idx, images).
"""
import gzip
import io
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

from settings import CAP_IMAGES, DATASETS


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _download(url, dst, tries=4):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for i in range(tries):
        try:
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, dst + ".part")
            os.replace(dst + ".part", dst)
            return dst
        except Exception as e:  # network hiccup: retry with backoff
            log(f"download failed ({e}); retry {i + 1}/{tries}")
            time.sleep(2 ** (i + 1))
    raise RuntimeError(f"could not download {url}")


def _read_parquet_images(path, img_col, label_col, mode):
    import pyarrow.parquet as pq
    from PIL import Image
    t = pq.read_table(path).to_pydict()
    x = np.stack([np.array(Image.open(io.BytesIO(im["bytes"])).convert(mode)) for im in t[img_col]])
    x = torch.from_numpy(x)
    x = x.unsqueeze(1) if mode == "L" else x.permute(0, 3, 1, 2)
    return x, torch.tensor(t[label_col])


HF = "https://huggingface.co/datasets/"


def _raw(name, root):
    """Raw uint8 images (N, ch, H, W) and int64 labels, train and test."""
    import torchvision.datasets as tvd
    if name in ("mnist", "fmnist"):
        cls = tvd.MNIST if name == "mnist" else tvd.FashionMNIST
        tr, te = cls(root, train=True, download=True), cls(root, train=False, download=True)
        return tr.data.unsqueeze(1), tr.targets, te.data.unsqueeze(1), te.targets
    if name == "svhn":
        base = HF + "ufldl-stanford/svhn/resolve/main/cropped_digits/"
        out = []
        for split in ("train", "test"):
            p = _download(base + f"{split}-00000-of-00001.parquet", os.path.join(root, "svhn_hf", f"{split}.parquet"))
            out += list(_read_parquet_images(p, "image", "label", "RGB"))
        return tuple(out)
    if name in ("cifar10", "cifar100"):
        sub, lab = ("cifar10_hf", "label") if name == "cifar10" else ("cifar100_hf", "fine_label")
        base = HF + ("uoft-cs/cifar10/resolve/main/plain_text/" if name == "cifar10" else
                     "uoft-cs/cifar100/resolve/main/cifar100/")
        out = []
        for split in ("train", "test"):
            p = _download(base + f"{split}-00000-of-00001.parquet", os.path.join(root, sub, f"{split}.parquet"))
            out += list(_read_parquet_images(p, "img", lab, "RGB"))
        return tuple(out)
    if name in ("dermamnist", "organamnist", "bloodmnist", "pathmnist"):
        p = _download(f"https://zenodo.org/records/10519652/files/{name}.npz?download=1",
                      os.path.join(root, "medmnist", f"{name}.npz"))
        z = np.load(p, allow_pickle=False)

        def conv(x):
            x = torch.from_numpy(x)
            return x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
        return (conv(z["train_images"]), torch.from_numpy(z["train_labels"]).flatten().long(),
                conv(z["test_images"]), torch.from_numpy(z["test_labels"]).flatten().long())
    if name == "gtsrb":
        tr, te = tvd.GTSRB(root, split="train", download=True), tvd.GTSRB(root, split="test", download=True)

        def conv(ds):
            xs, ys = [], []
            for img, y in ds:
                xs.append(np.array(img.convert("RGB").resize((32, 32))))
                ys.append(y)
            return torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2), torch.tensor(ys)
        return conv(tr) + conv(te)
    if name == "emnist":
        base = HF + "Royc30ne/emnist-byclass/resolve/main/"

        def idx(fn):
            p = _download(base + fn, os.path.join(root, "emnist_hf", fn))
            with gzip.open(p, "rb") as f:
                b = f.read()
            nd = b[3]
            dims = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(nd)]
            return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * nd).reshape(dims)
        # EMNIST images are stored transposed relative to MNIST
        xtr = torch.from_numpy(idx("emnist-byclass-train-images-idx3-ubyte.gz").copy()).transpose(1, 2).unsqueeze(1)
        xte = torch.from_numpy(idx("emnist-byclass-test-images-idx3-ubyte.gz").copy()).transpose(1, 2).unsqueeze(1)
        ytr = torch.from_numpy(idx("emnist-byclass-train-labels-idx1-ubyte.gz").astype(np.int64))
        yte = torch.from_numpy(idx("emnist-byclass-test-labels-idx1-ubyte.gz").astype(np.int64))
        return xtr.contiguous(), ytr, xte.contiguous(), yte
    if name == "fer2013":
        base = HF + "Aaryan333/fer2013_train_publicTest_privateTest/resolve/main/data/"
        ptr = _download(base + "train-00000-of-00001-5eab84e1c6a2fc27.parquet", os.path.join(root, "fer_hf", "train.parquet"))
        pte = _download(base + "privateTest-00000-of-00001-4b8a0715cf1b7560.parquet",
                        os.path.join(root, "fer_hf", "test.parquet"))
        return _read_parquet_images(ptr, "image", "label", "L") + _read_parquet_images(pte, "image", "label", "L")
    raise ValueError(f"unknown dataset {name}")


def load_dataset(name, root, device="cpu"):
    """Whole dataset as preprocessed tensors: x (N, ch, 32, 32) float, y (N,) long. Raw images are cached once as
    uint8 tensors in root (safe tensors-only file). Preprocessing as experiments/core.py: /255, 28 -> 32 bilinear
    (antialias) for small images, then the dataset's normalisation."""
    raw = DATASETS[name].get("raw", name)
    cache = os.path.join(root, f"_fedecs_tensors_{raw}.pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=True)
    else:
        os.makedirs(root, exist_ok=True)
        xtr, ytr, xte, yte = _raw(raw, root)
        d = dict(xtr=xtr.contiguous(), ytr=ytr.long(), xte=xte.contiguous(), yte=yte.long())
        torch.save(d, cache)
    spec = DATASETS[name]

    def prep(x):
        if x.shape[-1] != 32:
            x = torch.cat([F.interpolate(x[i:i + 8192].float().div(255.0), size=32, mode="bilinear",
                                         align_corners=False, antialias=True) for i in range(0, len(x), 8192)])
        else:
            x = x.float().div(255.0)
        if spec["norm"] == "half":
            x.sub_(0.5).div_(0.5)
        elif spec["norm"] != "none":
            m = torch.tensor(spec["norm"][0]).view(1, -1, 1, 1)
            s = torch.tensor(spec["norm"][1]).view(1, -1, 1, 1)
            x.sub_(m).div_(s)
        return x
    return dict(xtr=prep(d["xtr"]), ytr=d["ytr"], xte=prep(d["xte"]).to(device), yte=d["yte"].to(device))


def frac_of(name, n_train):
    spec = DATASETS[name]
    return spec["frac"] if spec["frac"] is not None else min(1.0, CAP_IMAGES / n_train)


def make_partition(ytr, C, K, frac, IF, alpha, seed, min_size=10):
    """Client index lists, as experiments/core.py make_partition (same RandomState calls in the same order).
    IF: long-tail imbalance factor in core's convention (0.01 = ratio 100; 1.0 = balanced); None = natural class
    sizes (each class keeps the fraction frac of its images). Also returns the validation set: 200 images per class
    (2000 // C when C > 10; fewer if a class is small) that no client holds."""
    rng = np.random.RandomState(seed)
    y = ytr.cpu().numpy()
    n_total = int(round(frac * len(y)))
    by_class = [rng.permutation(np.where(y == c)[0]) for c in range(C)]
    if IF is None:
        sizes = np.array([len(bc) for bc in by_class])
        want = np.floor(frac * sizes).astype(int)
    else:
        mu = IF ** (1.0 / (C - 1))
        prof = mu ** np.arange(C)
        want = np.floor(n_total * prof / prof.sum()).astype(int)
    nh = min(200, 2000 // C)
    if IF is None:  # natural imbalance: never take more than a quarter of a small class for validation
        nhs = [min(nh, len(bc) // 4) for bc in by_class]
    else:
        nhs = [nh] * C
    heldout = np.concatenate([bc[:h] for bc, h in zip(by_class, nhs)])
    pools = [bc[h:h + min(want[c], len(bc) - h)] for c, (bc, h) in enumerate(zip(by_class, nhs))]
    if alpha is None:
        allidx = rng.permutation(np.concatenate(pools))
        parts = [p.tolist() for p in np.array_split(allidx, K)]
    else:
        for _ in range(100):
            parts = [[] for _ in range(K)]
            for c in range(C):
                idx = pools[c]
                prop = rng.dirichlet([alpha] * K)
                cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(idx, cuts)):
                    parts[k].extend(chunk.tolist())
            if min(len(p) for p in parts) >= min_size:
                break
        while min(len(p) for p in parts) < min_size:
            small = int(np.argmin([len(p) for p in parts]))
            big = int(np.argmax([len(p) for p in parts]))
            j = rng.randint(len(parts[big]))
            parts[small].append(parts[big].pop(j))
        parts = [rng.permutation(p).tolist() for p in parts]
    counts = np.array([[int((y[p] == c).sum()) for c in range(C)] for p in parts])
    return dict(parts=[torch.tensor(p, dtype=torch.long) for p in parts], counts=counts,
                heldout=torch.tensor(heldout, dtype=torch.long), class_counts=counts.sum(0))


def natural_order(ytr, C):
    """For natural-imbalance datasets: relabelling that orders classes from most to least frequent, so that the
    head / medium / tail groups (by label index) mean the same as under the made long tail."""
    n = torch.bincount(ytr, minlength=C).numpy()
    order = np.argsort(-n, kind="stable")
    perm = np.empty(C, dtype=np.int64)
    perm[order] = np.arange(C)
    return torch.from_numpy(perm)


def relabel(data, perm):
    return dict(data, ytr=perm.to(data["ytr"].device)[data["ytr"]], yte=perm.to(data["yte"].device)[data["yte"]])


def permute_labels(data, seed):
    """Random rare classes (sweep): relabel by np.random.RandomState(1000 + seed).permutation(C), as core.py."""
    C = int(data["ytr"].max()) + 1
    perm = torch.from_numpy(np.random.RandomState(1000 + seed).permutation(C)).long()
    return relabel(data, perm)


def assign_arch(K, M=10):
    """Contiguous blocks of clients per architecture (reference GeFL_CVAE-F.py:35)."""
    blk = max(K // M, 1)
    return [min(k // blk, M - 1) for k in range(K)]


def active_clients(K, pfrac, seed, stage, r):
    """Clients taking part in round r of a stage (all, or the same seeded subset for every method), as core.py."""
    if pfrac >= 1.0:
        return list(range(K))
    m = max(1, int(round(pfrac * K)))
    rng = np.random.RandomState((seed * 1000003 + {"i": 1, "ii": 2, "iii": 3}[stage] * 10007 + r) % 2 ** 32)
    return sorted(rng.choice(K, m, replace=False).tolist())


def gpu_augment(x):
    """RandomCrop(32, padding=4) + RandomHorizontalFlip on a batch (as core.py)."""
    B, C, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(H, device=x.device)
    rows = (i[:, None] + ar)[:, None, :, None].expand(B, C, H, W)
    cols = (j[:, None] + ar)[:, None, None, :].expand(B, C, H, W)
    out = xp[torch.arange(B, device=x.device)[:, None, None, None],
             torch.arange(C, device=x.device)[None, :, None, None], rows, cols]
    flip = torch.rand(B, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def batches(n, B, device):
    perm = torch.randperm(n, device=device)
    for s in range(0, n, B):
        yield perm[s:s + B]


class Federation:
    """One dataset split into clients for one (setting, seed): every client's images and labels on the device,
    the validation set and the test set. Built identically by the server and by the client process."""

    def __init__(self, ds, root, IF, alpha, K, seed, device, tail_perm=False, frac_scale=1.0, clients=True):
        spec = DATASETS[ds]
        self.ds, self.C, self.K, self.seed, self.device = ds, spec["C"], K, seed, device
        data = load_dataset(ds, root, device)
        if spec["imbalance"] == "natural":
            data = relabel(data, natural_order(data["ytr"], self.C))
            IF = None
        if tail_perm:
            data = permute_labels(data, seed)
        if spec.get("partition") == "lt_standard":   # Track 2: the split LongTailLib's baselines get (track2.py)
            from track2 import lt_partition
            parts, heldout, counts = lt_partition(data["ytr"].numpy(), self.C, K, IF, alpha, seed)
            part = dict(parts=[torch.from_numpy(np.asarray(p)).long() for p in parts], counts=counts,
                        heldout=torch.from_numpy(heldout).long(), class_counts=counts.sum(0))
            if frac_scale < 1.0:   # smoke tests only
                part["parts"] = [p[:max(10, int(len(p) * frac_scale))] for p in part["parts"]]
                part["counts"] = np.array([[int((data["ytr"][p] == c).sum()) for c in range(self.C)]
                                           for p in part["parts"]])
                part["class_counts"] = part["counts"].sum(0)
        else:
            frac = frac_of(ds, len(data["ytr"])) * frac_scale
            part = make_partition(data["ytr"], self.C, K, frac, IF, alpha, seed)
        self.counts, self.class_counts = part["counts"], part["class_counts"]
        # the server process keeps no client data (clients=False): only sizes and, for logging, the class counts
        self.x = [data["xtr"][p].to(device) for p in part["parts"]] if clients else None
        self.y = [data["ytr"][p].to(device) for p in part["parts"]] if clients else None
        self.x_val, self.y_val = data["xtr"][part["heldout"]].to(device), data["ytr"][part["heldout"]].to(device)
        self.x_te, self.y_te = data["xte"].to(device), data["yte"].to(device)
        self.sizes = [len(p) for p in part["parts"]]
        self.arch = [0] * K if spec.get("single_arch") else assign_arch(K)
        self.groups = sorted(set(self.arch))
