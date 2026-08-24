import numpy as np
import torch

from utils.label_sampler import FidelityGatedSampler, UniformLabelSampler, build_label_sampler


def test_uniform_sampler_covers_all_classes_roughly_evenly():
    torch.manual_seed(0)
    sampler = UniformLabelSampler(10, np.ones(10))
    labels = sampler.sample(20000)
    counts = torch.bincount(labels, minlength=10).float()
    assert (counts.max() / counts.min()) < 1.3  # roughly uniform


def test_fidelity_gated_sampler_early_blend_is_far_from_full_inverse_frequency():
    """Right after init (before any fidelity update), the sampler should
    not yet be doing the FULL inverse-frequency correction -- it should sit
    somewhere between natural frequency and the pure inverse-frequency
    rate, not already at the pure-inverse-frequency extreme."""
    counts = np.array([500, 250, 10])  # heavily skewed
    sampler = FidelityGatedSampler(3, counts)
    labels = sampler.sample(50000)
    freq_rare = (labels == 2).float().mean().item()
    natural_rare = counts[2] / counts.sum()
    pure_inverse_freq_rare = (1 / counts[2]) / sum(1 / counts)
    assert natural_rare < freq_rare < pure_inverse_freq_rare


def test_fidelity_gated_sampler_shifts_toward_rare_class_as_fidelity_rises():
    counts = np.array([500, 250, 10])
    sampler = FidelityGatedSampler(3, counts, args=None)
    natural_rare = counts[2] / counts.sum()

    before = sampler.sample(50000)
    freq_before = (before == 2).float().mean().item()

    # simulate many rounds of high confidence on the rare class
    for _ in range(30):
        sampler.update_fidelity({0: 0.9, 1: 0.9, 2: 0.9})

    after = sampler.sample(50000)
    freq_after = (after == 2).float().mean().item()

    assert freq_after > freq_before
    assert freq_after > natural_rare


def test_fidelity_gated_sampler_does_not_oversample_unreliable_rare_class():
    """A rare class whose fidelity STAYS low should not be aggressively
    oversampled even though it's rare -- Mechanism B scales the
    inverse-frequency term by that class's own fidelity."""
    counts = np.array([500, 250, 10])
    sampler = FidelityGatedSampler(3, counts)
    # class 0 and 1 become reliable, class 2 (the rare one) stays unreliable
    for _ in range(30):
        sampler.update_fidelity({0: 0.9, 1: 0.9, 2: 0.05})
    labels = sampler.sample(50000)
    freq_rare = (labels == 2).float().mean().item()
    naive_inverse_freq_share = (1 / counts[2]) / sum(1 / counts)
    assert freq_rare < naive_inverse_freq_share


def test_build_label_sampler_selects_by_mechanism_b_flag():
    class Args:
        mechanism_b = 0
        mech_b_ema_decay = 0.6
        mech_b_init_fidelity = 0.3

    s = build_label_sampler(Args(), 5, np.ones(5))
    assert isinstance(s, UniformLabelSampler)

    Args.mechanism_b = 1
    s2 = build_label_sampler(Args(), 5, np.ones(5))
    assert isinstance(s2, FidelityGatedSampler)


def test_fidelity_state_is_reported_for_logging():
    sampler = FidelityGatedSampler(4, np.array([10, 10, 10, 10]))
    state = sampler.state()
    assert "mean_fidelity" in state
    assert len(state["fidelity_per_class"]) == 4

class DummyArgs:
    mech_b_inv_freq_power = 1.0

def test_fidelity_gated_sampler_power_1_0_is_pure_inverse_frequency():
    counts = np.array([500, 250, 10])
    # power=1.0 should reproduce the original behavior exactly
    args = DummyArgs()
    args.mech_b_inv_freq_power = 1.0
    sampler = FidelityGatedSampler(3, counts, args=args)
    
    inv_dist = sampler._inverse_freq_dist()
    pure_inv = (1 / (counts / counts.sum())) * sampler.fidelity
    
    np.testing.assert_allclose(inv_dist, pure_inv)

def test_fidelity_gated_sampler_lower_power_reduces_tail_oversampling():
    counts = np.array([500, 250, 10])
    
    args_full = DummyArgs()
    args_full.mech_b_inv_freq_power = 1.0
    sampler_full = FidelityGatedSampler(3, counts, args=args_full)
    
    args_reduced = DummyArgs()
    args_reduced.mech_b_inv_freq_power = 0.5
    sampler_reduced = FidelityGatedSampler(3, counts, args=args_reduced)
    
    # We expect the proportion allocated to class 2 (tail) to be smaller for reduced power
    # and proportion allocated to class 1 (medium) to be larger
    dist_full = sampler_full._inverse_freq_dist()
    dist_full /= dist_full.sum()
    
    dist_reduced = sampler_reduced._inverse_freq_dist()
    dist_reduced /= dist_reduced.sum()
    
    assert dist_reduced[2] < dist_full[2]
    assert dist_reduced[1] > dist_full[1]
