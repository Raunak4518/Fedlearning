"""Formal figure captions (IEEE style), built from the experiment's settings. A caption says what is plotted, on
which data and setting, over which seeds, and what the error bars mean; it never names notebooks, run numbers or
internal file names (those are listed in the report's appendix)."""
import re

from settings import CITE, DATASETS

# With one seed there is no spread over seeds: these phrases are rewritten (in this order) in every caption
ONE_SEED = [
    (r"; mean ± standard deviation over seeds", ""),
    (r"[,;] mean over seeds \(band: one standard deviation\)", ""),
    (r"[,;] mean over seeds[,;] (error bars|the shaded band is) one standard deviation", ""),
    (r"mean over them and over seeds", "mean over them"),
    (r"(averaged|mean) over architectures and seeds", r"\1 over architectures"),
    (r", summed over seeds", ""),
    (r" \(all seeds\)", ""),
    (r" \(mean over seeds\)", ""),
    (r"[,;] mean over seeds", ""),
]


# Track 2 has one model shared by all clients (one architecture): the stage figures' wording for it
TRACK2_WORDS = [
    (r"aggregated heads", "aggregated head"),
    (r"the clients' local heads", "the clients' local copies of the head"),
    (r"minus local heads", "minus local copies"),
    (r"the architecture heads", "its head"),
    (r"the clients' heads", "the clients' copies of the head"),
    (r"of the heads as trained", "of the head as trained"),
    (r"[,;] (averaged|mean) over architectures", ""),
]


def track2_words(text, S):
    if DATASETS.get(S[0], {}).get("track") != 2:
        return text
    for pat, rep in TRACK2_WORDS:
        text = re.sub(pat, rep, text)
    return text


def one_seed(text, seeds):
    """The caption text for the number of seeds it covers (unchanged for two or more)."""
    if seeds is None or len(seeds) != 1:
        return text
    for pat, rep in ONE_SEED:
        text = re.sub(pat, rep, text)
    return text


# Names as printed in reports and figures. Inside the suite the baselines of the one-shared-model setting carry the
# suffix " (LongTailLib)" (it selects their code, and pairs them with earlier runs); the reports print the method's
# name only, and the methods table names the code each method ran in.
DISPLAY = {"FedECS-LT (extractor 200 rounds)": "FedECS-LT (200 extractor rounds)"}
LT_SUFFIX = " (LongTailLib)"


# Messages of the federation, as named in the communication figure and table
MESSAGES = {
    "train.stage1": "Extractor training (Stage I)", "train.gen": "Generator training (Stage II)",
    "train.head": "Head training (Stage III)", "train.htfl": "Training rounds (method's own exchange)",
    "train.local": "Local training", "query.features": "Extractor broadcast", "query.heads": "Head broadcast",
    "query.load_gen": "Generator broadcast", "secsum.stats": "Secure sums: class statistics",
    "secsum.rows": "Secure sums: generator class rows", "secsum.consensus": "Secure sums: consensus soft labels",
    "secsum.readout": "Secure sums: exact readout", "sums.stats": "Sums: class statistics",
    "sums.gram": "Sums: class second moments", "sums.rows": "Sums: generator class rows",
    "sums.consensus": "Sums: consensus soft labels", "sums.readout": "Sums: exact readout",
}


def message_name(t):
    return MESSAGES.get(t, t)


def display(label):
    if label in DISPLAY:
        return DISPLAY[label]
    return label[:-len(LT_SUFFIX)] if label.endswith(LT_SUFFIX) else label


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
    seeds = _seeds(E, S)
    return track2_words(one_seed(f"{text} {setting_text(S, seeds)}", seeds), S)


def curve(num, E, S, what, split="Test-set", every=5):
    """split: "Test-set" for evaluated quantities; None for training-time quantities (losses, drift), whose text
    starts with its own capital. every: rounds between the plotted points (1: every round)."""
    head = f"{split} {what}" if split else what
    seeds = _seeds(E, S)
    when = "per round" if every == 1 else f"per evaluation round (every {every} rounds)"
    return track2_words(one_seed(f"{head}, {when}, mean over seeds; the shaded band is "
                                 f"one standard deviation. {setting_text(S, seeds)}", seeds), S)


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
    seeds = _seeds(E, S)
    return one_seed(f"Per-class {what} on the test set after training ({model}); classes are ordered from the most to "
                    f"the least frequent in the federation's training data.{note} {setting_text(S, seeds)}", seeds)


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
    26: "made from the saved models (models.zip) after the experiment",
    30: "no class-row generator in this experiment", 33: "no class-row generator in this experiment",
    34: "no class-row generator in this experiment", 35: "no class-row generator in this experiment",
    36: "no class-row generator in this experiment", 45: "ablation figure: made from the ablation experiment",
    46: "sensitivity figure: made from the sweep experiments", 50: "all clients take part in every round",
    52: "privacy figure: made from the differential-privacy experiment",
    53: "privacy figure: made from the membership-inference experiment",
    54: "privacy figure: made from the differential-privacy experiment",
    55: "the method diagram is drawn once for the paper", 57: "no generator copies were evaluated",
    56: "made for FedECS-LT and GeFL-F when several clients share a head architecture: either neither method is "
        "part of this experiment, or every architecture has a single client (10 clients, 10 architectures), so "
        "aggregating a head leaves it unchanged",
    59: "made for FedECS-LT and GeFL-F when several clients share a head architecture: either neither method is "
        "part of this experiment, or every architecture has a single client (10 clients, 10 architectures), so "
        "aggregating a head leaves it unchanged",
    58: "no class-row generator in this experiment", 60: "local-only training is not part of this experiment",
    69: "the exact readout is computed only for FedECS-LT and the GeFL-F family of methods",
    71: "made for the one-shared-model setting only", 72: "made for the one-shared-model setting only",
    73: "made for the one-shared-model setting only",
    70: "FedECS-LT and GeFL-F are not part of this experiment (their per-round, per-class metrics are in the logs)",
    74: "the head-training probes are logged by FedECS-LT and the GeFL-F family of methods only",
    75: "no method of this experiment has a synthetic phase in its head training",
    76: "no method of this experiment has a synthetic phase in its head training",
    77: "consensus soft labels are used only by FedECS-LT",
    78: "no saved test embeddings with last layers in this experiment",
    79: "CKA compares the embeddings of several head architectures of FedECS-LT or GeFL-F",
    80: "no generator in this experiment", 81: "no generator in this experiment",
    82: "no variational generator in this experiment",
}


def not_applicable(num):
    return NA.get(num, "the data this figure needs is not produced by this experiment")


# Words and patterns a caption in the paper must not contain (internal names, file names, run ids, raw keys)
LINT_BANNED = [r"\bsuite\b", r"\bTrack\b", r"\bnotebook", r"\bcache", r"\bpipeline\b", r"\bholders?\b",
               r"\bER\b", r"\(LongTailLib\)", r"\|(IF|a\d|K\d|s\d|p\d)", r"\bfinal_", r"\ber_", r"\bval_", r"_(?!\{)",
               r"\bOurs?\b", r"\bours\b", r"\bTODO\b", r"\bnan\b"]


def lint(text, figure=True):
    """Problems of a caption (figure or table): banned internal words; for figures, the setting and the seeds
    must be stated. Not-applicable reasons are checked with figure=False."""
    import re as _re
    out = [f"contains '{m.group(0)}'" for pat in LINT_BANNED for m in [_re.search(pat, text)] if m]
    if figure and not (_re.search(r"\bclients\b", text) and _re.search(r"\bseeds?\b", text)):
        out.append("does not state the setting and the seeds")
    return out
