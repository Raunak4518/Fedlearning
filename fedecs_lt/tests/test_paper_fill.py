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
            "cifar10_cifar100_components_of_fedecs_lt": [rec(m, "cifar10", s, 0.38) for s in (3, 4, 5) for m in
                                                         ("FedECS-LT w/o moment correction",
                                                          "FedECS-LT w/ Gaussian sampler")],
            "cifar10_cifar100_gefl_f_variants": [rec("GeFL-F + AdamW", "cifar10", s, 0.32) for s in (3, 4, 5)],
            "cifar10_cifar100_matched_synthetic_epochs": [rec("GeFL-F (10 synthetic epochs)", "cifar10", s, 0.33)
                                                          for s in (3, 4, 5)],
            "cifar10_cifar100_fresh_seeds_6_to_8": [rec(m, "cifar10", s, b) for s in (6, 7, 8)
                                                    for m, b in (("FedECS-LT", 0.10), ("GeFL-F", 0.05))],
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
    # FedECS-LT is reported with its readout (er_bal = final_bal + 0.1 here); its heads as trained are a second row.
    # Neither the sweep runs (20.0) nor the fresh seeds 6-8 (10.0) entered the main setting.
    assert r"\defnum{main/cifar10/FedECS-LT/bal}{54.0 $\pm$ 1.0}" in nums
    assert r"\defnum{main/cifar10/FedECS-LT (heads only)/bal}{44.0 $\pm$ 1.0}" in nums
    assert r"\defnum{main/cifar10/margin}{+15.0}" in nums                    # over FedDF as published
    assert r"\defnum{main/cifar10/heads_margin}{+5.0}" in nums               # the heads alone
    assert r"\defnum{main/cifar10/er_margin}{+5.0}" in nums                  # every method with the readout
    assert r"\defnum{main/cifar10/diff/GeFL-F}{+20.0}" in nums
    assert r"\defnum{main/cifar10/FedECS-LT/gain_er}{+10.0}" in nums
    assert r"\defnum{main/cifar10/strongest}{FedDF}" in nums                 # the heads-only row is not a rival
    assert r"\defnum{sweep/Dirichlet0.1/FedECS-LT/bal}{30.0 $\pm$ 0.0}" in nums
    assert r"\defnum{fresh/cifar10/FedECS-LT/bal}{20.0 $\pm$ 0.0}" in nums
    assert r"\defnum{fresh/cifar10/diff/GeFL-F}{+15.0}" in nums
    tables = tmp_path / "generated" / "tables"
    assert (tables / "main_bal.tex").exists() and (tables / "robust.tex").exists()
    comp = (tables / "abl_components.tex").read_text()
    assert "FedECS-LT (full)" in comp and "w/o moment correction" in comp and "w/ Gaussian sampler" in comp
    assert "44.0" in comp and "54.0" in comp                                    # full: as trained and with readout
    abl = (tables / "abl_bal.tex").read_text()
    assert "GeFL-F + AdamW" in abl and "GeFL-F (10 synthetic epochs)" in abl and "FedECS-LT (heads only)" in abl
    assert r"\defnum{abl/cifar10/GeFL-F + AdamW/gain}{-2.0}" in nums                # against GeFL-F (34.0)
    assert r"\defnum{abl/cifar10/heads_vs/GeFL-F (10 synthetic epochs)}{+11.0}" in nums
    assert r"\defnum{comp/cifar10/w/ Gaussian sampler/er_drop}{+6.0}" in nums       # full 54.0 with the readout
    assert r"\defnum{comp/cifar10/w/ Gaussian sampler/drop}{+6.0}" in nums          # heads 44.0 as trained
    assert r"\defnum{comp/n_variants}{two}" in nums
