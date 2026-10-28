"""File locks across processes on Linux (fcntl.flock, as on Kaggle) and Windows (msvcrt.locking; local GPU runs,
fedecs_lt/local/run_local.py). Windows has no shared lock here: a shared request takes the exclusive lock, which only
serialises what would otherwise share the GPU (local runs go one at a time anyway)."""
import os
import time

WINDOWS = os.name == "nt"


def lock(f, shared=False):
    """Blocks until the open file `f` is locked."""
    if WINDOWS:
        import msvcrt
        f.seek(0)
        while True:
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)   # the first byte stands for the whole file
                return
            except OSError:
                time.sleep(0.5)
    import fcntl
    fcntl.flock(f, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)


def unlock(f):
    if WINDOWS:
        import msvcrt
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl
    fcntl.flock(f, fcntl.LOCK_UN)


def alive(pid):
    """Whether process `pid` exists. Never os.kill(pid, 0) on Windows: there it terminates the process."""
    pid = int(pid)
    if pid == 0:   # a lock file whose owner has not written its pid yet (os.kill(0, 0) succeeds too): taken
        return True
    if pid < 0:
        return False
    if WINDOWS:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return k32.GetLastError() == 5   # access denied: the process exists
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259   # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
