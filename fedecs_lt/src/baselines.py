"""Registry of the baselines that have their own protocol (not the GeFL-F pipeline family of runner.PIPELINE).

Track 1 (different client models; frozen Stage-I extractor shared with FedECS-LT): the model-heterogeneous FL methods
ported from HtFLlib (Apache License 2.0), run in Flower by htfl_server.py / htfl_client.py. Track 2 (one shared model,
the FedLTLib benchmark) runs the authors' LongTailLib code unchanged on this suite's splits (track2.py)."""

HTFL = ["FedProto", "FedTGP", "FedGen", "FedGH", "LG-FedAvg", "FedMRL"]
SOURCE = {m: "HtFLlib (Zhang et al., KDD 2025), Apache-2.0, ported to Flower" for m in HTFL}
BASELINES = {m: dict(pipeline="htfl", name=m, source=SOURCE[m]) for m in HTFL}
# GeFL's grouped setting (Kang et al., IEEE TMC 2025: "clients are partitioned by model architecture"): FedAvg,
# FedProx and AvgKD as in GeFL's code (MIT), FedALA from its authors' code (Apache-2.0), and the long-tail methods CCVR
# and FedETF applied the same way (grouped_server.py / grouped_client.py)
GROUPED = ["FedAvg", "FedProx", "FedALA", "AvgKD", "CCVR", "FedETF"]
SOURCE.update({"FedAvg": "GeFL code (Kang et al., IEEE TMC 2025), MIT, grouped by architecture",
               "FedProx": "GeFL code (Kang et al., IEEE TMC 2025), MIT, grouped by architecture",
               "AvgKD": "GeFL code (Kang et al., IEEE TMC 2025), MIT",
               "FedALA": "authors' ALA module (Zhang et al., AAAI 2023), Apache-2.0, grouped by architecture",
               "CCVR": "LongTailLib CCVR calibration (FedLTLib), grouped by architecture",
               "FedETF": "LongTailLib FedETF (FedLTLib), grouped by architecture"})
for m in GROUPED:
    BASELINES[m] = dict(pipeline="grouped", name=m, source=SOURCE[m])
TRACK2 = ["FedAvg", "FedProx", "MOON", "FedLC", "CCVR", "CReFF", "FedETF", "Fed-GraB", "FedLoGe", "FedYoYo", "RUCR",
          "CLIP2FL"]
for m in TRACK2:
    BASELINES[f"{m} (LongTailLib)"] = dict(pipeline="longtaillib", name=m,
                                          source="LongTailLib (FedLTLib benchmark), authors' code unchanged, "
                                                 "downloaded at run time")


def run_baseline(grid, context, R):
    if R["method"]["pipeline"] == "htfl":
        from htfl_server import run_htfl
        return run_htfl(grid, context, R)
    if R["method"]["pipeline"] == "grouped":
        from grouped_server import run_grouped
        return run_grouped(grid, context, R)
    raise ValueError(f"unknown baseline pipeline {R['method']['pipeline']}")
