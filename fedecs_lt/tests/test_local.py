"""Local GPU runs (fedecs_lt/local/run_local.py, 29 Oct 2026): the job list is the notebooks' run lists, every run
has a kind that the memory check measures, the fit rules, Windows' exclusions, and the cross-platform file locks."""
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FL = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(FL, "src"))
sys.path.insert(0, os.path.join(FL, "local"))
sys.path.insert(0, FL)


def rl(windows=False):
    os.environ["FEDECS_LOCAL_AS_WINDOWS"] = "1" if windows else "0"
    import run_local
    return importlib.reload(run_local)


def test_jobs_are_the_notebooks():
    import build_notebooks as bn
    nbs = bn.all_notebooks()
    jobs = rl().jobs()
    assert set(jobs) == {n for n in nbs if not n.startswith("NB00")}
    for n, j in jobs.items():
        assert j["runs"] == bn.queue_order(nbs[n]["runs"]), n


def test_every_run_has_a_checked_kind():
    m = rl()
    runs = [r for j in m.jobs().values() for r in j["runs"]]
    kinds = {m.kind(r) for r in runs}
    assert kinds <= set(m.CHECK), kinds - set(m.CHECK)
    for k, entries in m.CHECK.items():   # every check run is a real run of the jobs, of its own kind
        for label, ds, K in entries:
            hit = [r for r in runs if r["label"] == label and r["ds"] == ds and r.get("K", 10) == K]
            assert hit and m.kind(hit[0]) == k, (k, label, ds, K)


def test_fit_rules():
    m = rl()
    runs = [dict(label="FedTGP", ds="cifar10", K=10), dict(label="FedECS-LT", ds="cifar10", K=10)]
    assert m.fits(runs, None) == (None, ["gen", "htfl"])
    chk = dict(kinds=dict(gen=dict(ok=True), htfl=dict(ok=True)))
    assert m.fits(runs, chk) == (True, [])
    chk["kinds"]["gen"]["ok"] = False
    assert m.fits(runs, chk) == (False, ["gen"])
    assert m.fits(runs, dict(kinds=dict(htfl=dict(ok=True)))) == (None, ["gen"])


def test_windows_leaves_one_shared_model_to_kaggle():
    m = rl(windows=True)
    t2 = [dict(label="FedLoGe (LongTailLib)", ds="cifar100_t2", K=20)]
    chk = dict(kinds=dict(t2_baseline=dict(ok=True)))
    assert m.fits(t2, chk)[0] is False
    assert m.fits([dict(label="FedTGP", ds="cifar10", K=10)], dict(kinds=dict(htfl=dict(ok=True)))) == (True, [])
    rl(windows=False)


def test_locks_and_alive(tmp_path):
    import oslock
    p = tmp_path / "x.lock"
    with open(p, "w") as f:
        oslock.lock(f)
        oslock.unlock(f)
        oslock.lock(f, shared=True)
        oslock.unlock(f)
    assert oslock.alive(os.getpid()) and oslock.alive(0) and not oslock.alive(2 ** 22 + 12345)


def test_no_posix_only_calls_left_in_src():
    """fcntl and os.kill(pid, 0) only inside oslock.py (os.kill(pid, 0) terminates a process on Windows)."""
    src = os.path.join(FL, "src")
    for f in os.listdir(src):
        if f.endswith(".py") and f != "oslock.py":
            text = open(os.path.join(src, f)).read()
            assert "import fcntl" not in text and "os.kill(" not in text, f


def test_check_record_roundtrip(tmp_path, monkeypatch):
    m = rl()
    monkeypatch.setattr(m, "CHECK_FILE", str(tmp_path / "gpu_check.json"))
    json.dump(dict(gpu="x", vram_mb=4096, kinds=dict(ce=dict(ok=True, peak_mb=1000))), open(m.CHECK_FILE, "w"))
    assert m.load_check()["kinds"]["ce"]["ok"]
