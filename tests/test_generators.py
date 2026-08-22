import torch
import pytest

from generators.base import GEN_REGISTRY
import generators.ccvae  # noqa: F401
import generators.ccgan  # noqa: F401
import generators.cddpm  # noqa: F401


class Args:
    gen_channels = 8
    latent_size = 6
    n_feat = 8
    n_T = 4
    guide_w = 0.3
    b1 = 0.5
    b2 = 0.999


@pytest.mark.parametrize("gen_name", ["vae", "gan", "ddpm"])
@pytest.mark.parametrize("img_size,in_channels", [(16, 1), (20, 3), (28, 1)])
def test_generator_sample_shape_matches_input(gen_name, img_size, in_channels):
    cls = GEN_REGISTRY.get(gen_name)
    gen = cls(num_classes=6, in_channels=in_channels, img_size=img_size, args=Args())
    labels = torch.randint(0, 6, (5,))
    samples = gen.sample(labels)
    assert samples.shape == (5, in_channels, img_size, img_size)


@pytest.mark.parametrize("gen_name", ["vae", "gan", "ddpm"])
def test_conditioning_parameter_names_exist_in_state_dict(gen_name):
    cls = GEN_REGISTRY.get(gen_name)
    gen = cls(num_classes=6, in_channels=3, img_size=16, args=Args())
    sd = gen.state_dict()
    for key in gen.conditioning_parameter_names():
        assert key in sd, f"{gen_name}: declared conditioning key '{key}' missing from state_dict"


def test_registry_rejects_unknown_generator():
    with pytest.raises(KeyError):
        GEN_REGISTRY.get("not_a_real_generator")


def test_registry_names_include_all_three():
    names = GEN_REGISTRY.names()
    assert {"vae", "gan", "ddpm"}.issubset(set(names))
