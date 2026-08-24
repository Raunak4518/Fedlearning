"""
utils/avg.py

Aggregation rules. `FedAvg` (flat, unweighted) and `weighted_FedAvg`
(volume-weighted) are the two building blocks every strategy below is
built from -- matching the original GeFL repo's `FedAvg` /
`model_wise_FedAvg` naming.

`aggregate_generator` is the one that matters for this project: it
dispatches between GeFL's baseline (flat average over every parameter,
including the conditioning pathway -- the class-blind behaviour this
project's proposal identifies as the gap) and Mechanism A (trunk stays
volume-weighted FedAvg; the conditioning pathway gets a per-class
effective-number-of-samples weighting instead). It is written entirely
against `generator.conditioning_parameter_names()`, so it works
identically for CCVAE, CCGAN, or CDDPM without any architecture-specific
code here.
"""
from collections import OrderedDict
from typing import Dict, List

import torch


def FedAvg(state_dicts: List[OrderedDict]) -> OrderedDict:
    """Flat, unweighted average -- GeFL's own reference aggregation rule:
    wg <- (1/|C|) * sum_k w_k. Every client counts equally regardless of
    how much data it holds or which classes it has."""
    avg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        avg[key] = stacked.mean(dim=0).to(state_dicts[0][key].dtype)
    return avg


def weighted_FedAvg(state_dicts: List[OrderedDict], weights: List[float]) -> OrderedDict:
    """Standard volume-weighted FedAvg."""
    total = sum(weights)
    norm_w = [w / total for w in weights]
    avg = OrderedDict()
    for key in state_dicts[0].keys():
        stacked = torch.stack([sd[key].float() * w for sd, w in zip(state_dicts, norm_w)], dim=0)
        avg[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return avg


def model_wise_FedAvg(ws_glob: List[OrderedDict], ws_local: List[List[OrderedDict]],
                       sample_counts: List[List[int]] = None) -> List[OrderedDict]:
    """Target-network aggregation for the heterogeneous pool: `ws_local[m]`
    is the list of state_dicts uploaded this round by clients whose
    dev_spec_idx == m (only clients running the *same* architecture can be
    averaged together at all).

    Paper Algorithm 2: flat unweighted FedAvg for both generators and
    target nets — θ_g ← (1/|C_agg|) Σ θ_k. No volume weighting.
    Groups with no participants this round keep their previous global weights."""
    new_glob = []
    for m, group in enumerate(ws_local):
        if len(group) == 0:
            new_glob.append(ws_glob[m])
        else:
            new_glob.append(FedAvg(group))
    return new_glob


def _effective_num_weight(count: int, beta: float) -> float:
    """Effective number of samples, Cui et al. 2019: E_n = (1-beta^n)/(1-beta).
    Used here AS the aggregation weight (not its loss-reweighting inverse
    1/E_n -- see the note in generators/base.py's docstring and the
    project README for why the direction matters): it grows monotonically
    with a client's count of the class, with diminishing returns, so the
    client that actually holds more of a class earns proportionally more
    say over that class's conditioning row -- the same direction FedAvg
    already weights by total data volume."""
    if count <= 0:
        return 0.0
    return (1.0 - beta ** count) / (1.0 - beta)


def frequency_weighted_row_average(client_tensors: List[torch.Tensor], client_class_counts: List[Dict[int, int]],
                                    num_conditioning_classes: int, beta: float,
                                    fallback_weights: List[float]) -> torch.Tensor:
    """
    client_tensors: list of identically-shaped [R, ...] tensors (R rows),
                     one per client -- e.g. a label-embedding weight matrix.
    client_class_counts: list of {class_id: count} dicts, one per client,
                     over the TRUE data class ids (0..num_conditioning_classes-1).
    num_conditioning_classes: how many of the R rows correspond to a real,
                     countable class (e.g. num_classes for a VAE/GAN
                     embedding). Any additional rows (e.g. DDPM's reserved
                     "null" row for classifier-free guidance) fall back to
                     `fallback_weights` (ordinary volume weighting) since
                     they aren't tied to any single class's frequency.
    """
    R = client_tensors[0].shape[0]
    out = torch.zeros_like(client_tensors[0])
    fb_norm = torch.tensor(fallback_weights, dtype=torch.float32)
    fb_norm = fb_norm / fb_norm.sum()

    for r in range(R):
        if r < num_conditioning_classes:
            w_raw = torch.tensor([_effective_num_weight(counts.get(r, 0), beta) for counts in client_class_counts])
            if w_raw.sum() == 0:
                out[r] = client_tensors[0][r]  # no client had this class this round; keep prior value
                continue
            w = w_raw / w_raw.sum()
        else:
            w = fb_norm
        out[r] = sum(w[i] * client_tensors[i][r] for i in range(len(client_tensors)))
    return out


def aggregate_generator(client_state_dicts: List[OrderedDict], client_sample_counts: List[int],
                         client_class_counts: List[Dict[int, int]], num_classes: int,
                         conditioning_keys: List[str], mechanism_a: bool, beta: float = 0.999) -> OrderedDict:
    """
    Full-model generator aggregation used every communication round.

    mechanism_a=False -> GeFL baseline: FedAvg (flat) over every key,
                          including the conditioning pathway.
    mechanism_a=True  -> Mechanism A: every key NOT in `conditioning_keys`
                          gets weighted_FedAvg (volume-weighted, matching
                          how GeFL already treats the trunk); every key IN
                          `conditioning_keys` gets
                          frequency_weighted_row_average instead.
    """
    if not mechanism_a:
        return FedAvg(client_state_dicts)

    trunk_keys = [k for k in client_state_dicts[0].keys() if k not in conditioning_keys]
    trunk_sds = [{k: sd[k] for k in trunk_keys} for sd in client_state_dicts]
    avg = weighted_FedAvg(trunk_sds, client_sample_counts)

    for key in conditioning_keys:
        tensors = [sd[key] for sd in client_state_dicts]
        avg[key] = frequency_weighted_row_average(
            tensors, client_class_counts, num_classes, beta, client_sample_counts
        )
    return avg
