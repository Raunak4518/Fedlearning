"""Secure summation with Flower's SecAgg+ (Bell et al., CCS 2020), and the plain sum used by baseline protocols.

The server learns only the sum over the participating clients. Flower's SecAgg+ workflow returns a weighted mean of
quantised vectors in [-clip, clip]; every client therefore sends its contribution divided by a public bound (with
num_examples = 1) and the server multiplies the mean by (number of clients x bound). Values outside the bound are
clipped and counted (reported in the run log). Quantisation uses 30 bits (settings.SECAGG); the round-off is below
1e-7 of the bound per client.
"""
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message
from flwr.common import Code, FitIns, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.compat.common import recorddict_compat as compat
from flwr.server import ServerConfig
from flwr.server.compat import LegacyContext
from flwr.server.compat.grid_client_proxy import GridClientProxy
from flwr.server.strategy import Strategy
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD, Key as WK

from fl_common import jdump
from settings import SECAGG


class _SumStrategy(Strategy):
    """Legacy-strategy shim for SecAggPlusWorkflow: sends one config to every node, keeps the unmasked mean."""

    def __init__(self, cfg):
        self.cfg, self.result = cfg, None

    def initialize_parameters(self, client_manager):
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        return [(c, FitIns(ndarrays_to_parameters([np.zeros(1)]), {"j": jdump(self.cfg)}))
                for c in client_manager.all().values()]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"secure sum: {len(failures)} client failures: {failures[:1]}")
        self.result = parameters_to_ndarrays(results[0][1].parameters)
        return None, {}

    def configure_evaluate(self, *a):
        return []

    def aggregate_evaluate(self, *a):
        return None, {}

    def evaluate(self, *a):
        return None


def secure_sum(grid, context, node_ids, cfg, bounds, round_no=1):
    """Sum over node_ids of the arrays each client's contribution function returns for cfg; bounds: one public bound
    per array (client divides by it). Returns float64 numpy arrays."""
    node_ids = list(node_ids)
    n = len(node_ids)
    if n < 3:  # SecAgg+ needs at least three parties; with fewer, a sum reveals the summands anyway
        return None
    strat = _SumStrategy(dict(cfg, bounds=[float(b) for b in bounds]))
    lc = LegacyContext(context, config=ServerConfig(num_rounds=1), strategy=strat)
    for nid in node_ids:
        lc.client_manager.register(GridClientProxy(nid, grid, context.run_id))
    lc.state.config_records[MAIN_CONFIGS_RECORD] = ConfigRecord({WK.CURRENT_ROUND: int(round_no)})
    lc.state.array_records[MAIN_PARAMS_RECORD] = ArrayRecord([np.zeros(1)])
    wf = SecAggPlusWorkflow(num_shares=n, reconstruction_threshold=max(2, n // 2 + 1),
                            max_weight=SECAGG["max_weight"], clipping_range=SECAGG["clipping_range"],
                            quantization_range=SECAGG["quantization_range"], modulus_range=SECAGG["modulus_range"])
    wf(grid, lc)
    if strat.result is None:
        raise RuntimeError(f"secure sum of {cfg.get('what')} failed (a client raised an error: see the log above)")
    return [a.astype(np.float64) * n * b for a, b in zip(strat.result, bounds)]


def client_secsum(msg, contrib_fn):
    """Client side of a secure sum (registered with secaggplus_mod): scale by the public bounds, clip, reply in the
    legacy fit format the mod expects. contrib_fn(cfg) -> list of numpy arrays."""
    fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
    import json
    cfg = json.loads(fitins.config["j"])
    arrays = contrib_fn(cfg)
    lim = SECAGG["clipping_range"]
    out, clipped = [], 0
    for a, b in zip(arrays, cfg["bounds"]):
        s = np.asarray(a, dtype=np.float64) / b
        clipped += int((np.abs(s) > lim).sum())
        out.append(np.clip(s, -lim, lim))
    res = FitRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(out), num_examples=1,
                 metrics={"clipped": clipped})
    return Message(compat.fitres_to_recorddict(res, keep_input=True), reply_to=msg)
