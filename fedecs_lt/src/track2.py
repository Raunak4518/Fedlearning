"""Track 2 (one shared model; the FedLTLib benchmark protocol): the baselines run the authors' LongTailLib code
unchanged (Lin and Wang, FedLTLib, arXiv 2609.15625; https://github.com/gzhu-hcai/LongTailLib, MIT licence stated in
its README; downloaded at run time at a pinned commit, never stored in this repository), on this suite's client splits.

  prepare()  writes our split of CIFAR-10/100-LT into LongTailLib's dataset format (per-client train / test files and
             the balanced global test set)
  driver     runs LongTailLib's main.py with its default FedLTLib protocol (20 clients, 40% per round, 200 rounds,
             10 local epochs, batch 32, SGD 0.1 with momentum 0.9 and decay 0.99 per round), ResNet-8 for every
             method; keeps the server object, then re-runs the library's own global-test evaluation while recording
             the scores it computes, so that every metric of this suite is computed from exactly the predictions the
             library itself uses (checked: the recorded predictions give the library's own accuracy).

    python track2.py prepare DS IF SEED LT_DIR DATA_ROOT
    python track2.py run ALGO DS IF SEED LT_DIR OUT_DIR [ROUNDS]
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

LT_REPO = "https://github.com/gzhu-hcai/LongTailLib.git"
LT_COMMIT = "414167807aa244819bde33988ae36d07edfed645"
ALGOS = {"FedAvg": "FedAvg", "FedProx": "FedProx", "MOON": "MOON", "FedLC": "FedLC", "CCVR": "CCVR",
         "CReFF": "CReFF", "FedETF": "FedETF", "Fed-GraB": "FedGraB", "FedLoGe": "FedLoGe", "FedYoYo": "FedYoYo",
         "RUCR": "RUCR", "CLIP2FL": "CLIP2FL"}
K2, ALPHA2, JR2 = 20, 0.5, 0.4


def fetch(lt_dir):
    """The authors' code at the pinned commit (downloaded once per notebook)."""
    if not os.path.isdir(os.path.join(lt_dir, ".git")):
        subprocess.run(["git", "clone", "-q", LT_REPO, lt_dir], check=True)
    subprocess.run(["git", "-C", lt_dir, "fetch", "-q", "--depth", "50", "origin", LT_COMMIT], check=False)
    subprocess.run(["git", "-C", lt_dir, "checkout", "-q", LT_COMMIT], check=True)
    return lt_dir


def folder(ds, IF, seed):
    return f"{'Cifar100' if ds == 'cifar100' else 'Cifar10'}-IF{int(round(1 / IF))}-a{ALPHA2}-K{K2}-s{seed}"


def lt_partition(y, C, K, IF, alpha, seed, min_size=10):
    """Standard CIFAR-LT profile n_c = n_max IF^(c/(C-1)) (Cao et al., 2019; CReFF), after holding out the
    validation images (200 per class; 20 for C = 100), then a Dirichlet(alpha) split over K clients (same rule and
    random calls as data.make_partition)."""
    rng = np.random.RandomState(seed)
    by_class = [rng.permutation(np.where(y == c)[0]) for c in range(C)]
    nh = min(200, 2000 // C)
    heldout = np.concatenate([bc[:nh] for bc in by_class])
    n_max = min(len(bc) for bc in by_class) - nh
    want = [int(n_max * IF ** (c / (C - 1))) for c in range(C)]
    pools = [bc[nh:nh + want[c]] for c, bc in enumerate(by_class)]
    for _ in range(100):
        parts = [[] for _ in range(K)]
        for c in range(C):
            prop = rng.dirichlet([alpha] * K)
            cuts = (np.cumsum(prop) * len(pools[c])).astype(int)[:-1]
            for k, chunk in enumerate(np.split(pools[c], cuts)):
                parts[k].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            break
    while min(len(p) for p in parts) < min_size:
        small, big = int(np.argmin([len(p) for p in parts])), int(np.argmax([len(p) for p in parts]))
        parts[small].append(parts[big].pop(rng.randint(len(parts[big]))))
    parts = [rng.permutation(p) for p in parts]
    counts = np.array([[int((y[p] == c).sum()) for c in range(C)] for p in parts])
    return parts, heldout, counts


def raw_cifar(ds, data_root):
    """uint8 NHWC images and labels (the same Hugging Face copies data.py uses)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data import _raw
    xtr, ytr, xte, yte = _raw(ds, data_root)
    return (xtr.permute(0, 2, 3, 1).numpy(), ytr.numpy(), xte.permute(0, 2, 3, 1).numpy(), yte.numpy())


def prepare(ds, IF, seed, lt_dir, data_root):
    C = 100 if ds == "cifar100" else 10
    xtr, ytr, xte, yte = raw_cifar(ds, data_root)
    parts, heldout, counts = lt_partition(ytr, C, K2, IF, ALPHA2, seed)
    root = os.path.join(lt_dir, "dataset", folder(ds, IF, seed))
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "test"), exist_ok=True)
    rng = np.random.RandomState(10_000 + seed)
    for k, p in enumerate(parts):
        np.savez_compressed(os.path.join(root, "train", f"{k}.npz"), data=dict(x=xtr[p], y=ytr[p].astype(np.int64)))
        t = rng.choice(len(yte), 200, replace=False)  # small local test slice (the library averages client tests)
        np.savez_compressed(os.path.join(root, "test", f"{k}.npz"), data=dict(x=xte[t], y=yte[t].astype(np.int64)))
    np.savez_compressed(os.path.join(root, "global_test.npz"), data=dict(x=xte, y=yte.astype(np.int64)))
    json.dump(dict(num_clients=K2, num_classes=C, non_iid=True, balance=False, partition="dir", alpha=ALPHA2,
                   longtail=True, imbalance_factor=int(round(1 / IF)), seed=seed, source="FedECS-LT suite split"),
              open(os.path.join(root, "config.json"), "w"))
    np.savez_compressed(os.path.join(root, "suite_split.npz"), counts=counts, heldout=heldout)
    return root, counts


DRIVER = r'''
import json, os, runpy, sys, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
import flcore.servers.serverbase as sb
CAP = {}
def _wrap(cls):
    if "train" in cls.__dict__ and not getattr(cls.__dict__["train"], "_fedecs", False):
        orig = cls.__dict__["train"]
        def train(self, *a, **k):
            CAP["server"] = self
            return orig(self, *a, **k)
        train._fedecs = True
        cls.train = train
args, out = json.loads(sys.argv[1]), sys.argv[2]
# every global evaluation the library makes (every eval_gap rounds): record the scores it ranks and compute every
# per-class metric with the suite's metrics.py (wrappers chain, so the final re-evaluation below still records)
import importlib.util
sys.path.append(os.environ["FEDECS_SRC"])
_spec = importlib.util.spec_from_file_location("fedecs_metrics", os.path.join(os.environ["FEDECS_SRC"], "metrics.py"))
FM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FM)
_name = args[args.index("-data") + 1]
Y = torch.from_numpy(np.load(os.path.join("..", "dataset", _name, "global_test.npz"), allow_pickle=True)["data"]
                     .item()["y"]).long()
C_ = int(args[args.index("-ncl") + 1])
PER, DEPTH, ROUND = [], [0], [0]
def _wrap_round(cls):
    """Counts completed rounds (one aggregation per round; a subclass calling super() counts once)."""
    orig = cls.__dict__["aggregate_parameters"]
    if getattr(orig, "_fedecs", False):
        return
    def f(self, *a, **k):
        DEPTH[0] += 1
        try:
            return orig(self, *a, **k)
        finally:
            DEPTH[0] -= 1
            if DEPTH[0] == 0:
                ROUND[0] += 1
    f._fedecs = True
    setattr(cls, "aggregate_parameters", f)
def _wrap_eval(cls, name):
    orig = cls.__dict__[name]
    if getattr(orig, "_fedecs", False):
        return
    def f(self, *a, **k):
        if DEPTH[0]:
            return orig(self, *a, **k)
        rec, pmax, ptmax = [], torch.max, torch.Tensor.max
        def m1(x, *aa, **kk):
            if torch.is_tensor(x) and x.dim() == 2 and x.is_floating_point() and (aa[:1] == (1,) or kk.get("dim") == 1):
                rec.append(x.detach().float().cpu())
            return pmax(x, *aa, **kk)
        def m2(t, *aa, **kk):
            if t.dim() == 2 and t.is_floating_point() and (aa[:1] == (1,) or kk.get("dim") == 1):
                rec.append(t.detach().float().cpu())
            return ptmax(t, *aa, **kk)
        DEPTH[0] += 1
        torch.max, torch.Tensor.max = m1, m2
        try:
            res = orig(self, *a, **k)
        finally:
            torch.max, torch.Tensor.max = pmax, ptmax
            DEPTH[0] -= 1
        try:
            lg = torch.cat(rec)[-len(Y):] if rec else None
            if lg is not None and len(lg) == len(Y) and lg.shape[1] == C_:
                PER.append(dict(round=ROUND[0], post=FM.log_view(FM.class_metrics(lg, Y, C_))))
        except Exception as e:
            print("per-class evaluation not recorded:", repr(e))
        return res
    f._fedecs = True
    setattr(cls, name, f)
import importlib, pkgutil, flcore.servers as fs
for m in pkgutil.iter_modules(fs.__path__):
    mod = importlib.import_module("flcore.servers." + m.name)
    for v in list(vars(mod).values()):
        if isinstance(v, type) and issubclass(v, sb.Server):
            _wrap(v)
            for _n in ("compute_global_test_accuracy", "_evaluate_global"):
                if _n in v.__dict__:
                    _wrap_eval(v, _n)
            if "aggregate_parameters" in v.__dict__:
                _wrap_round(v)
sys.argv = ["main.py"] + args
t0 = time.time()
runpy.run_path("main.py", run_name="__main__")
srv = CAP["server"]
# re-run the library's own global evaluation while recording the scores it ranks
REC, ACC = [], None
_max, _tmax = torch.max, torch.Tensor.max
def rmax(x, *a, **k):
    if torch.is_tensor(x) and x.dim() == 2 and x.is_floating_point() and (a[:1] == (1,) or k.get("dim") == 1):
        REC.append(x.detach().float().cpu())
    return _max(x, *a, **k)
def rtmax(self, *a, **k):
    if self.dim() == 2 and self.is_floating_point() and (a[:1] == (1,) or k.get("dim") == 1):
        REC.append(self.detach().float().cpu())
    return _tmax(self, *a, **k)
torch.max, torch.Tensor.max = rmax, rtmax
with torch.no_grad():
    if hasattr(srv, "_evaluate_global"):
        res = srv._evaluate_global()
    else:
        res = srv.compute_global_test_accuracy()
torch.max, torch.Tensor.max = _max, _tmax
lib_acc = res[0] if isinstance(res, tuple) else res
n_test = len(srv.global_testloader.dataset)
logits = torch.cat(REC)[-n_test:] if REC else None
def _num_list(v):
    """The library's per-round history lists (rs_*) that hold numbers; some methods (Fed-GraB) also keep lists of
    dicts there, which are skipped (the per-class metrics above cover every evaluation)."""
    out = []
    for x in v:
        if torch.is_tensor(x) and x.numel() == 1:
            x = x.item()
        if not isinstance(x, (int, float, np.integer, np.floating)) or isinstance(x, bool):
            return None
        out.append(float(x))
    return out
hist = {}
for k in dir(srv):
    if k.startswith("rs_") and isinstance(getattr(srv, k), list):
        _l = _num_list(getattr(srv, k))
        if _l is not None:
            hist[k] = _l
emb = None
try:
    srv.global_model.eval()
    feats = []
    with torch.no_grad():
        for i, (xb, yb) in enumerate(srv.global_testloader):
            o = srv.global_model(xb.to(srv.device))
            if isinstance(o, tuple):
                feats.append(o[0].flatten(1).float().cpu())
            if sum(len(f) for f in feats) >= 2000:
                break
    emb = torch.cat(feats)[:2000].numpy() if feats else None
except Exception as e:
    print("embeddings not available:", repr(e))
np.savez_compressed(out, logits=logits.numpy() if logits is not None else np.zeros(0), lib_acc=np.array(
    [lib_acc if lib_acc is not None else np.nan]), time_s=np.array([time.time() - t0]),
    embed=emb if emb is not None else np.zeros(0),
    gpu_peak_mb=np.array([torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0]))
json.dump(hist, open(out + ".hist.json", "w"))
json.dump(PER, open(out + ".perclass.json", "w"))
'''


def run(algo, ds, IF, seed, lt_dir, out_dir, rounds=200, label=None, data_root=None, ds_key=None):
    """One LongTailLib run on our split; writes a suite-format record into out_dir/runs.jsonl."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from class_names import CLASS_NAMES
    from metrics import PER_CLASS, class_metrics, summarize
    import torch
    C = 100 if ds == "cifar100" else 10
    name = folder(ds, IF, seed)
    sysdir = os.path.join(lt_dir, "system")
    drv = os.path.join(sysdir, "_suite_driver.py")
    open(drv, "w").write(DRIVER)
    os.makedirs(os.path.join(out_dir, "predictions"), exist_ok=True)
    label = label or algo
    ds_key = ds_key or ds
    rid = f"{label}|{ds_key}|IF{IF}|a{ALPHA2}|K{K2}|s{seed}|p{JR2}"
    safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in rid)
    tmp = os.path.join(out_dir, "predictions", safe + "_lib.npz")
    args = ["-data", name, "-algo", ALGOS[algo], "-m", "ResNet8", "-gr", str(rounds), "-ncl", str(C), "-nc", str(K2),
            "-jr", str(JR2), "-seed", str(seed), "-did", "0", "-dev", "cuda" if torch.cuda.is_available() else "cpu"]
    # evaluate every 5th round (LongTailLib's default is every round): training is unchanged, the final model is
    # evaluated by the driver below; saves about 15% of a run
    args += ["-eg", "5"]
    args += os.environ.get("FEDECS_LT_TEST_ARGS", "").split()   # CPU tests only (e.g. "-ls 1"); unset on Kaggle
    t0 = time.time()
    os.makedirs(os.path.join(out_dir, "logs"), exist_ok=True)
    log = open(os.path.join(out_dir, "logs", safe + "_longtaillib.log"), "w")
    p = subprocess.run([sys.executable, drv, json.dumps(args), tmp], cwd=sysdir, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, FEDECS_SRC=os.path.dirname(os.path.abspath(__file__))))
    log.close()
    if p.returncode != 0:
        raise RuntimeError(f"LongTailLib {algo} failed (see logs/{safe}_longtaillib.log)")
    d = np.load(tmp)
    root = os.path.join(lt_dir, "dataset", name)
    y = np.load(os.path.join(root, "global_test.npz"), allow_pickle=True)["data"].item()["y"]
    logits = torch.from_numpy(d["logits"]).float()
    m = class_metrics(logits, torch.from_numpy(y).long(), C, keep_probs=True)
    probs = m.pop("probs")
    lib_acc = float(d["lib_acc"][0])
    summ = summarize({0: m})
    rec = dict(run_id=rid, label=label, dataset=ds_key, IF=IF, alpha=ALPHA2,
               K=K2, seed=seed, pfrac=JR2, tail_perm=False, track=2, framework="LongTailLib (authors' code)",
               method=dict(pipeline="longtaillib", algo=ALGOS[algo], model="ResNet8", commit=LT_COMMIT,
                           source="LongTailLib (FedLTLib benchmark), authors' code unchanged, downloaded at run time"),
               client_counts=np.load(os.path.join(root, "suite_split.npz"))["counts"].tolist(),
               class_order=list(range(C)), class_names=CLASS_NAMES.get(ds),
               test_class_counts=np.bincount(y, minlength=C).tolist(),
               library_accuracy=lib_acc, accuracy_check=abs(m["acc"] - lib_acc) < 1e-6 if lib_acc == lib_acc else None,
               time_s=time.time() - t0, rounds=dict(T=rounds))
    if "gpu_peak_mb" in d.files and d["gpu_peak_mb"][0] > 0:   # the library's process (a fresh one per run)
        rec["gpu_peak_mb"] = round(float(d["gpu_peak_mb"][0]))
    for k, v in summ.items():
        if isinstance(v, (int, float)):
            rec[f"final_{k}"] = v
        elif k in PER_CLASS:
            rec[f"final_{k}"] = [round(float(x), 4) for x in v]
        elif k == "confusion":
            rec["final_confusion"] = v
    hist = json.load(open(tmp + ".hist.json"))
    rec["history_library"] = hist
    # every evaluation of the library (every 5th round, and the final model): every metric for every class, as the
    # suite's per-round logs (event "round_eval", round = rounds of training completed)
    per = json.load(open(tmp + ".perclass.json")) if os.path.exists(tmp + ".perclass.json") else []
    with open(os.path.join(out_dir, "logs", safe + ".jsonl"), "w") as f:
        f.write(json.dumps(dict(t=time.time(), event="run_start", run_id=rid, label=label, dataset=ds_key)) + "\n")
        for e in per:
            f.write(json.dumps(dict(t=time.time(), event="round_eval", round=e["round"], post=e["post"])) + "\n")
    rec["evaluations_logged"] = len(per)
    np.savez_compressed(os.path.join(out_dir, "predictions", safe + ".npz"), y_full=y, probs_0=probs.astype(np.float16),
                        **({"embed_0": d["embed"].astype(np.float16), "y": y[:len(d["embed"])]}
                           if d["embed"].size else {}))
    for f_ in (tmp, tmp + ".hist.json", tmp + ".perclass.json"):   # temporary files of the driver
        if os.path.exists(f_):
            os.remove(f_)
    with open(os.path.join(out_dir, "runs.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def run_for_suite(R):
    """Runner entry: R["ds"] is a Track-2 dataset key (cifar10_t2 / cifar100_t2). Fetches the library and prepares the
    split once per notebook (lock file), then runs the method."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from settings import DATASETS
    raw = DATASETS[R["ds"]]["raw"]
    lt_dir = os.path.join(R["cache_dir"], "LongTailLib")
    lock = lt_dir + ".lock"
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
            break
        except FileExistsError:
            time.sleep(5)
    try:
        fetch(lt_dir)
        if not os.path.exists(os.path.join(lt_dir, "dataset", folder(raw, R["IF"], R["seed"]), "global_test.npz")):
            prepare(raw, R["IF"], R["seed"], lt_dir, R["data_root"])
    finally:
        os.remove(lock)
    rounds = (R.get("rounds") or {}).get("T", 200)
    return run(R["method"]["name"], raw, R["IF"], R["seed"], lt_dir, R["out_dir"], rounds=rounds, label=R["label"],
               ds_key=R["ds"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        ds, IF, seed, lt_dir, data_root = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
        fetch(lt_dir)
        print(prepare(ds, IF, seed, lt_dir, data_root)[0])
    elif cmd == "run":
        algo, ds, IF, seed, lt_dir, out = sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6], \
            sys.argv[7]
        rounds = int(sys.argv[8]) if len(sys.argv) > 8 else 200
        r = run(algo, ds, IF, seed, lt_dir, out, rounds)
        print(r["run_id"], "bal", round(r["final_bal"], 4), "library acc", r["library_accuracy"], "check",
              r["accuracy_check"])
