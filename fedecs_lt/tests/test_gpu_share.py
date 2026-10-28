"""runner.GpuShare: a run that needs the whole GPU waits for the runs already on it and keeps new ones from starting
until it has finished (FedECS-LT with one shared model ran out of memory next to another run on a T4)."""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import runner  # noqa: E402


def job(d, tag, alone, delay, hold, q):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    time.sleep(delay)
    with runner.GpuShare(d, alone):
        q.put((tag, "in", time.time()))
        time.sleep(hold)
        q.put((tag, "out", time.time()))


def test_whole_gpu_runs_alone(tmp_path):
    q = mp.Queue()
    jobs = [("A", False, 0.0, 1.5), ("X", True, 0.4, 1.5), ("B", False, 0.8, 0.5)]
    ps = [mp.Process(target=job, args=(str(tmp_path),) + j + (q,)) for j in jobs]
    for p in ps:
        p.start()
    for p in ps:
        p.join(30)
    t = {(a, b): c for a, b, c in (q.get(timeout=5) for _ in range(6))}
    assert t[("X", "in")] >= t[("A", "out")]          # waits for the run already on the GPU
    assert t[("B", "in")] >= t[("X", "out")]          # no new run starts while it waits or runs


def test_which_runs_need_the_whole_gpu():
    assert runner.needs_whole_gpu(dict(label="FedECS-LT", ds="cifar100_t2"))
    assert runner.needs_whole_gpu(dict(label="FedECS-LT (extractor 200 rounds)", ds="cifar10_t2"))
    assert not runner.needs_whole_gpu(dict(label="FedAvg (LongTailLib)", ds="cifar10_t2"))
    assert not runner.needs_whole_gpu(dict(label="FedECS-LT", ds="cifar10"))


def test_dead_waiter_does_not_block(tmp_path):
    """A .want file left by a process that was killed outright no longer keeps shared runs off the GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    share = runner.GpuShare(str(tmp_path), False)
    open(share.path + ".want.999999999", "w").close()
    t0 = time.time()
    with share:
        pass
    assert time.time() - t0 < 2
