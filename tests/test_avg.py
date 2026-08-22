from collections import OrderedDict

import torch

from utils.avg import (FedAvg, _effective_num_weight, aggregate_generator, frequency_weighted_row_average,
                        model_wise_FedAvg, weighted_FedAvg)


def _sd(**kwargs):
    return OrderedDict({k: torch.as_tensor(v, dtype=torch.float32).reshape(-1) for k, v in kwargs.items()})


def test_fedavg_is_unweighted_mean():
    sds = [_sd(w=1.0), _sd(w=3.0), _sd(w=5.0)]
    out = FedAvg(sds)
    assert torch.isclose(out["w"], torch.tensor(3.0))


def test_weighted_fedavg_matches_manual_weighted_mean():
    sds = [_sd(w=1.0), _sd(w=3.0)]
    out = weighted_FedAvg(sds, weights=[1, 3])
    # (1*1 + 3*3) / 4 = 2.5
    assert torch.isclose(out["w"], torch.tensor(2.5))


def test_effective_num_weight_increases_with_count():
    """Regression test for the direction bug caught during development:
    a client holding MORE samples of a class must get a LARGER weight,
    not smaller."""
    w0 = _effective_num_weight(0, beta=0.999)
    w3 = _effective_num_weight(3, beta=0.999)
    w47 = _effective_num_weight(47, beta=0.999)
    assert w0 == 0.0
    assert w3 < w47, "effective-number weight must grow with the client's class count"


def test_frequency_weighted_row_average_dominated_by_majority_holder():
    # 3 clients' embedding rows for a single class-9-like row
    client_tensors = [torch.tensor([[9.0, 9.0]]), torch.tensor([[1.0, 1.0]]), torch.tensor([[100.0, 100.0]])]
    client_class_counts = [{0: 0}, {0: 3}, {0: 47}]
    out = frequency_weighted_row_average(client_tensors, client_class_counts, num_conditioning_classes=1,
                                          beta=0.999, fallback_weights=[1, 1, 1])
    # client 2 (count=47) should dominate -> result close to [100, 100]
    assert out[0, 0] > 50.0


def test_frequency_weighted_row_average_falls_back_for_rows_beyond_num_classes():
    # row index 1 is beyond num_conditioning_classes=1 -> should use fallback (volume) weights, not class counts
    client_tensors = [torch.tensor([[1.0], [10.0]]), torch.tensor([[2.0], [20.0]])]
    client_class_counts = [{0: 5}, {0: 5}]
    out = frequency_weighted_row_average(client_tensors, client_class_counts, num_conditioning_classes=1,
                                          beta=0.999, fallback_weights=[1, 3])
    # row 1 (the "null" row) should be volume-weighted: (1*10 + 3*20)/4 = 17.5
    assert torch.isclose(out[1, 0], torch.tensor(17.5))


def test_aggregate_generator_flat_when_mechanism_a_off():
    sds = [_sd(trunk=1.0, cond=1.0), _sd(trunk=3.0, cond=9.0)]
    out = aggregate_generator(sds, client_sample_counts=[1, 100], client_class_counts=[{0: 0}, {0: 10}],
                               num_classes=1, conditioning_keys=["cond"], mechanism_a=False)
    assert torch.isclose(out["trunk"], torch.tensor(2.0))  # flat mean, ignores the 1-vs-100 volume gap
    assert torch.isclose(out["cond"], torch.tensor(5.0))   # flat mean here too


def test_aggregate_generator_splits_trunk_and_conditioning_when_mechanism_a_on():
    sds = [_sd(trunk=1.0, cond=1.0), _sd(trunk=3.0, cond=9.0)]
    out = aggregate_generator(sds, client_sample_counts=[1, 99], client_class_counts=[{0: 0}, {0: 50}],
                               num_classes=1, conditioning_keys=["cond"], mechanism_a=True)
    # trunk: volume-weighted -> (1*1 + 99*3)/100 = 2.98
    assert torch.isclose(out["trunk"], torch.tensor(2.98), atol=1e-4)
    # cond: client 0 has zero of the class -> weight 0 -> result should equal client 1's value (9.0)
    assert torch.isclose(out["cond"], torch.tensor(9.0), atol=1e-3)


def test_model_wise_fedavg_groups_independently():
    ws_glob = [_sd(w=0.0), _sd(w=0.0)]
    ws_local = [[_sd(w=2.0), _sd(w=4.0)], []]  # group 1 has no participants this round
    out = model_wise_FedAvg(ws_glob, ws_local)
    assert torch.isclose(out[0]["w"], torch.tensor(3.0))
    assert torch.isclose(out[1]["w"], torch.tensor(0.0))  # unchanged, kept previous global
