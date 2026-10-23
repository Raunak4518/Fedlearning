"""Formal figure captions (IEEE style), built from the experiment's settings. A caption says what is plotted, on
which data and setting, over which seeds, and what the error bars mean; it never names notebooks, run numbers or
internal file names (those are listed in the report's appendix)."""
from settings import CITE, DATASETS


def mname(m):
    """Method name as written in the text (with its citation where it has one)."""
    base = m.split(" (")[0].split(" + ")[0]
    if m in CITE:
        return f"{m} [{CITE[m]}]"
    if base in CITE and m.startswith(base):
        return m
    return m


def setting_text(S, seeds=None):
    ds, IF, alpha, K, pfrac, tail_perm = S
    spec = DATASETS.get(ds, dict(name=ds, imbalance="IF"))
    if spec["imbalance"] == "natural":
        data = f"{spec['name']} (natural class imbalance)"
    elif IF is None or IF >= 1.0:
        data = f"{spec['name']} (balanced)"
    else:
        data = f"{spec['name']}-LT (imbalance ratio {int(round(1 / IF))})"
    part = "IID split" if alpha is None else f"Dirichlet label split with alpha = {alpha}"
    txt = f"{data}, {K} clients, {part}"
    if pfrac < 1.0:
        txt += f", {int(round(100 * pfrac))}% of the clients per round"
    if tail_perm:
        txt += ", rare classes chosen at random"
    if seeds:
        s = sorted(seeds)
        txt += "; seeds " + (", ".join(str(x) for x in s[:-1]) + f" and {s[-1]}" if len(s) > 1 else str(s[0]))
    return txt + "."


def _seeds(E, S):
    return sorted({r["seed"] for r in E.runs if (r["dataset"], r["IF"], r["alpha"], r["K"], r.get("pfrac", 1.0),
                                                bool(r.get("tail_perm", False))) == S})


def simple(num, E, S, text):
    return f"{text} {setting_text(S, _seeds(E, S))}"


def curve(num, E, S, what):
    return (f"Test-set {what}, per evaluation round (every 5 rounds), mean over seeds; the shaded band is one standard "
            f"deviation. {setting_text(S, _seeds(E, S))}")


def per_class(num, E, S, what):
    return (f"Per-class {what} on the test set after training (heads as trained, mean over the architectures' heads "
            f"and over seeds); classes are ordered from the most to the least frequent in the federation's training "
            f"data. {setting_text(S, _seeds(E, S))}")


def setting_slug(S):
    ds, IF, alpha, K, pfrac, tail_perm = S
    s = f"{ds}_IF{'nat' if IF is None else int(round(1 / IF))}_a{alpha}_K{K}"
    if pfrac < 1.0:
        s += f"_p{pfrac}"
    if tail_perm:
        s += "_randomrare"
    return s


NA = {
    7: "consensus soft labels are used only by FedECS-LT",
    11: "no saved confusion matrices for FedECS-LT or GeFL-F",
    19: "FedECS-LT is not part of this experiment", 20: "GeFL-F is not part of this experiment",
    22: "no generator in this experiment", 23: "no generator in this experiment",
    26: "made from the saved models (models.zip) by the repository script, not in the notebook",
    30: "no class-row generator in this experiment", 33: "no class-row generator in this experiment",
    34: "no class-row generator in this experiment", 35: "no class-row generator in this experiment",
    36: "no class-row generator in this experiment", 45: "ablation figure: made from the ablation experiment",
    46: "sensitivity figure: made from the sweep experiments", 50: "all clients take part in every round",
    52: "privacy figure: made from the differential-privacy experiment",
    53: "privacy figure: made from the membership-inference experiment",
    54: "privacy figure: made from the differential-privacy experiment",
    55: "the pipeline diagram is drawn once for the paper", 57: "no generator copies were evaluated",
    58: "no class-row generator in this experiment", 60: "local-only training is not part of this experiment",
}


def not_applicable(num):
    return NA.get(num, "the data this figure needs is not produced by this experiment")
