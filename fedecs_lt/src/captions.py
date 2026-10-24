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
    if spec.get("track") == 2:
        txt += ", one ResNet-8 shared by all clients (standard CIFAR-LT class profile)"
    if pfrac < 1.0:
        txt += f", {int(round(100 * pfrac))}% of the clients per round"
    if tail_perm:
        txt += ", rare classes chosen at random"
    if seeds:
        s = sorted(seeds)
        txt += ("; seeds " + ", ".join(str(x) for x in s[:-1]) + f" and {s[-1]}") if len(s) > 1 else f"; seed {s[0]}"
    return txt + "."


def _seeds(E, S):
    return sorted({r["seed"] for r in E.runs if (r["dataset"], r["IF"], r["alpha"], r["K"], r.get("pfrac", 1.0),
                                                bool(r.get("tail_perm", False))) == S})


def simple(num, E, S, text):
    return f"{text} {setting_text(S, _seeds(E, S))}"


def curve(num, E, S, what, split="Test-set"):
    """split: "Test-set" for evaluated quantities; None for training-time quantities (losses, drift), whose text
    starts with its own capital."""
    head = f"{split} {what}" if split else what
    return (f"{head}, per evaluation round (every 5 rounds), mean over seeds; the shaded band is one standard "
            f"deviation. {setting_text(S, _seeds(E, S))}")


PROTOTYPE_METHODS = ("FedProto", "FedTGP")


def proto_note(methods):
    """Methods that classify by the nearest prototype score classes by negative distances; a softmax of those is no
    probability estimate, so calibration, NLL and Brier score are not comparable with the other methods."""
    ps_ = [m for m in methods if m in PROTOTYPE_METHODS]
    if not ps_:
        return ""
    one = len(ps_) == 1
    return (" " + " and ".join(ps_) + (" classifies" if one else " classify") + " by the nearest prototype; "
            + ("its" if one else "their") + " scores are a softmax of negative distances, not probability estimates, "
            "so " + ("its" if one else "their") + " calibration, NLL and Brier values are not comparable with those of "
            "the other methods.")


def per_class(num, E, S, what, note=""):
    shared = DATASETS.get(S[0], {}).get("track") == 2
    model = "the global model, mean over seeds" if shared else "the clients' models as trained, mean over them and over seeds"
    return (f"Per-class {what} on the test set after training ({model}); classes are ordered from the most to the least frequent in the federation's training "
            f"data.{note} {setting_text(S, _seeds(E, S))}")


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
    56: "every head architecture has a single client here (10 clients, 10 architectures), so aggregating a head "
        "leaves it unchanged; the figure is made for the 50- and 100-client federations (or FedECS-LT and GeFL-F "
        "are not part of this experiment)",
    59: "every head architecture has a single client here (10 clients, 10 architectures), so aggregating a head "
        "leaves it unchanged; the figure is made for the 50- and 100-client federations (or FedECS-LT and GeFL-F "
        "are not part of this experiment)",
    58: "no class-row generator in this experiment", 60: "local-only training is not part of this experiment",
    69: "the exact readout is computed only for FedECS-LT and the GeFL-F family of methods",
    71: "a figure of the one-shared-model setting (Track 2)", 72: "a figure of the one-shared-model setting (Track 2)",
    73: "a figure of the one-shared-model setting (Track 2)",
    70: "FedECS-LT and GeFL-F are not part of this experiment (their per-round, per-class metrics are in the logs)",
}


def not_applicable(num):
    return NA.get(num, "the data this figure needs is not produced by this experiment")
