"""Every method keeps one colour, marker and line style in every figure of every notebook (separate processes)."""
import json
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
NAMES = ["FedECS-LT", "FedLC", "FedLC (LongTailLib)", "FedAvg (LongTailLib)", "A method not in ORDER"]


def styles(hashseed):
    code = (f"import sys, json; sys.path.insert(0, {SRC!r}); import plot_style as ps; "
            f"print(json.dumps([ps.style(m) for m in {NAMES!r}], default=str))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env=dict(os.environ, PYTHONHASHSEED=str(hashseed)))
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_style_stable_across_processes():
    a, b = styles(1), styles(2)
    assert a == b
    ours, lc, lc_lib = a[0], a[1], a[2]
    assert ours["linestyle"] == "-" and ours["linewidth"] > lc["linewidth"]
    assert lc_lib["color"] == lc["color"] and lc_lib["linestyle"] != lc["linestyle"]


def test_log_axis_only_for_wide_ranges():
    """A log axis over less than a factor of 10 labels its minor ticks, which overlapped (Fig. 42, GTSRB, 27 Oct)."""
    sys.path.insert(0, SRC)
    import plot_style as ps
    narrow = [131.0, 138.0, 146.0, 152.0, 163.0]      # communication per client (MB) of five methods
    f, ax = ps.figure("column", 2.4)
    ax.scatter(narrow, [50, 52, 49, 55, 51])
    assert not ps.log_axis(ax, narrow) and ax.get_xscale() == "linear"
    assert ps.layout_problems(f) == []
    f, ax = ps.figure("column", 2.4)
    wide = [18.8, 131.0, 4739.0, 11268.0]
    ax.scatter(wide, [50, 52, 49, 55])
    assert ps.log_axis(ax, wide) and ax.get_xscale() == "log"
    assert ps.layout_problems(f) == []
