"""Registry of the baselines that have their own protocol (not the GeFL-F pipeline family of runner.PIPELINE).

Track 1 (different client models; frozen Stage-I extractor shared with FedECS-LT): the model-heterogeneous FL methods
ported from HtFLlib (Apache License 2.0), run in Flower by htfl_server.py / htfl_client.py. Track 2 (one shared model,
the FedLTLib benchmark) runs the authors' LongTailLib code unchanged on this suite's splits (track2.py)."""

HTFL = ["FedProto", "FedTGP", "FedGen", "FedGH", "LG-FedAvg", "FedMRL"]
SOURCE = {m: "HtFLlib (Zhang et al., KDD 2025), Apache-2.0, ported to Flower" for m in HTFL}
BASELINES = {m: dict(pipeline="htfl", name=m, source=SOURCE[m]) for m in HTFL}


def run_baseline(grid, context, R):
    if R["method"]["pipeline"] == "htfl":
        from htfl_server import run_htfl
        return run_htfl(grid, context, R)
    raise ValueError(f"unknown baseline pipeline {R['method']['pipeline']}")
