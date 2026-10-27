"""docs/papers/fill.py: runs are sorted into the paper's experiments by method and setting (a sweep run never
enters the main tables), and every table and number slot is written from them."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILL = os.path.join(HERE, "..", "..", "docs", "papers", "fill.py")


def load_fill():
    spec = importlib.util.spec_from_file_location("fill", FILL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(label, ds, seed, bal, IF=0.01, alpha=0.5, K=10):
    return dict(run_id=f"{label}|{ds}|IF{IF}|a{alpha}|K{K}|s{seed}", label=label, dataset=ds, IF=IF, alpha=alpha, K=K,
                seed=seed, final_bal=bal, er_bal=bal + 0.1, final_tail=bal / 2, client_counts=[[50, 5]] * 10,
                test_class_counts=[100, 100], n_train=550, class_sep_F=0.2, gate_branch="anchor",
                time_s=100.0, comm_mb_per_client=10.0)


def test_fill_sorts_and_writes(tmp_path, monkeypatch):
    fill = load_fill()
    res = tmp_path / "results"
    runs = {"NB10_main_cifar10_part1of3": [rec(m, "cifar10", s, b + 0.01 * s) for s in (3, 4, 5)
                                           for m, b in (("FedECS-LT", 0.40), ("GeFL-F", 0.30), ("FedDF", 0.35))],
            "NB31_components_cifar10": [rec("FedECS-LT w/o moment correction", "cifar10", s, 0.38) for s in (3, 4, 5)],
            "NB40_sweep_cifar10_a0.1": [rec(m, "cifar10", s, 0.2, alpha=0.1) for s in (3, 4, 5)
                                        for m in ("FedECS-LT", "GeFL-F")]}
    for name, rs in runs.items():
        (res / name).mkdir(parents=True)
        (res / name / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    monkeypatch.setattr(fill, "RESULTS", str(res))
    monkeypatch.setattr(fill, "GEN", str(tmp_path / "generated"))
    monkeypatch.setattr(sys, "argv", ["fill.py"])
    fill.main()
    nums = (tmp_path / "generated" / "numbers.tex").read_text()
    assert r"\defnum{main/cifar10/FedECS-LT/bal}{44.0 $\pm$ 1.0}" in nums     # the sweep run (20.0) did not enter
    assert r"\defnum{main/cifar10/margin}{+5.0}" in nums                     # over FedDF
    assert r"\defnum{main/cifar10/diff/GeFL-F}{+10.0}" in nums
    assert r"\defnum{sweep/Dirichlet0.1/FedECS-LT/bal}{20.0 $\pm$ 0.0}" in nums
    tables = tmp_path / "generated" / "tables"
    assert (tables / "main_bal.tex").exists() and (tables / "robust.tex").exists()
    comp = (tables / "abl_components.tex").read_text()
    assert "FedECS-LT (full)" in comp and "w/o moment correction" in comp
