"""Tests for the paper-exact GeFL-F pieces: Tables XX-XXIII architectures,
the client -> architecture map, and lazy conditioning decay."""
from types import SimpleNamespace

import torch

from gefl_f.engine_f import _assign_architectures, _conditioning_row_norms
from gefl_f.headers import HEADER_REGISTRY
from gefl_f.paper_nets import PaperFeatureExtractor
from generators.ccvae_paper import PaperCCVAE
from utils.localUpdateGen import LazyConditioningDecay, _make_lazy_optimizer

PAPER_FC_IN = [1024, 512, 640, 320, 256, 320, 160, 128, 100, 80]  # Tables XXI-XXIII


def test_paper_fe_shapes():
    x1, x3 = torch.randn(2, 1, 32, 32), torch.randn(2, 3, 32, 32)
    assert PaperFeatureExtractor(1, "mnist")(x1).shape == (2, 3, 16, 16)
    assert PaperFeatureExtractor(3, "mnist")(x3).shape == (2, 3, 16, 16)
    assert PaperFeatureExtractor(3, "cifar10")(x3).shape == (2, 10, 16, 16)


def test_paper_header_fc_widths():
    for i, fc_in in enumerate(PAPER_FC_IN, start=1):
        for suffix, ch in (("", 3), ("_bn", 3), ("", 10)):
            h = HEADER_REGISTRY.get(f"cnn{i}{suffix}")(fe_channels=ch, num_classes=10, fe_spatial=16)
            assert h.fc.in_features == fc_in, (i, suffix)
            assert h(torch.randn(2, ch, 16, 16)).shape == (2, 10)
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in HEADER_REGISTRY.get("cnn5_bn")(3, 10).modules())
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in HEADER_REGISTRY.get("cnn5")(3, 10).modules())


def test_contiguous_assignment_matches_reference():
    assert _assign_architectures(10, 10, "contiguous") == list(range(10))
    assert _assign_architectures(50, 10, "contiguous") == [i // 5 for i in range(50)]
    assert _assign_architectures(11, 10, "contiguous")[-2:] == [9, 9]
    assert _assign_architectures(6, 3, "roundrobin") == [0, 1, 2, 0, 1, 2]


def _cvae():
    return PaperCCVAE(10, 3, 16, SimpleNamespace(latent_size=16))


def test_paper_cvae_shapes_and_conditioning():
    g = _cvae()
    x, y = torch.rand(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])
    recon, mu, logvar = g(x, y)
    assert recon.shape == x.shape and mu.shape == (4, 16)
    assert g.sample(y).shape == (4, 3, 16, 16)
    assert g.conditioning_parameter_names() == ["label_proj"]
    assert g.label_proj.shape == (10, 512)
    # split decoder == Linear(l + C) on [z, onehot(y)]
    z = torch.randn(4, 16)
    full_w = torch.cat([g.dec_z.weight, g.label_proj.t()], dim=1)
    onehot = torch.nn.functional.one_hot(y, 10).float()
    ref = torch.cat([z, onehot], 1) @ full_w.t() + g.dec_z.bias
    assert torch.allclose(g.dec_z(z) + g.label_proj[y], ref, atol=1e-6)


def test_absent_class_row_gets_no_data_gradient():
    g = _cvae()
    x, y = torch.rand(4, 3, 16, 16), torch.tensor([0, 0, 1, 1])
    recon, mu, logvar = g(x, y)
    g.loss_function(recon, x, mu, logvar).backward()
    assert g.label_proj.grad[2:].abs().sum() == 0
    assert g.label_proj.grad[:2].abs().sum() > 0


def test_coupled_decay_shrinks_absent_rows_and_lazy_decay_does_not():
    torch.manual_seed(0)
    x, y = torch.rand(8, 3, 16, 16), torch.tensor([0, 1] * 4)

    def run(lazy):
        g = _cvae()
        before = g.label_proj.detach().clone()
        if lazy:
            opt, ld = _make_lazy_optimizer(g, 1e-3, None, 1e-3), LazyConditioningDecay(g, 1e-3)
        else:
            opt, ld = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-3), None
        for _ in range(20):
            opt.zero_grad()
            recon, mu, logvar = g(x, y)
            g.loss_function(recon, x, mu, logvar).backward()
            if ld:
                ld.before_step(y)
            opt.step()
            if ld:
                ld.after_step()
        return before, g.label_proj.detach()

    before, after = run(lazy=False)
    # Adam normalises lambda*w, so absent rows move ~lr per step toward 0
    assert (after[2:].abs() < before[2:].abs()).float().mean() > 0.95
    shrink = (before[2:].abs() - after[2:].abs()).clamp(min=0)
    assert shrink.max() <= 20 * 1e-3 * 1.05
    before, after = run(lazy=True)
    assert torch.equal(after[2:], before[2:])
    assert not torch.equal(after[:2], before[:2])


def test_conditioning_row_norms():
    state = {"label_proj": torch.tensor([[3.0, 4.0], [0.0, 0.0]])}
    assert _conditioning_row_norms(state, ["label_proj"], 2) == [5.0, 0.0]
