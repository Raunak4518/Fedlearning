"""The consensus pool is generated in chunks (client.pool_chunk) so that no client holds a pool larger than 1 GiB.
Checks: a one-chunk pool is the same draw as the earlier one-shot generation (Track 1 unchanged); every slice of a
multi-chunk pool equals the same rows of the whole pool; chunked consensus probabilities equal the whole-pool ones."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import client  # noqa: E402


class FakeGen:
    """Draws from the global torch RNG like the real generators (class shift + noise)."""

    def sample(self, y):
        return y[:, None].float() + torch.randn(len(y), 6)


def sampler():
    return dict(G=FakeGen(), mc=None, pool={}, shape=(6,))


def one_shot(cfg, sm):   # the generation before chunking (single draw of the whole pool)
    torch.manual_seed(cfg["pool_seed"])
    py = torch.arange(cfg["C"])[torch.randint(0, cfg["C"], (cfg["P"],))]
    return client.draw(sm, py), py


def test_single_chunk_equals_one_shot():
    cfg = dict(round=0, pool_seed=123, P=500, C=4, heads_key="h")
    sm = sampler()
    x0, y0 = one_shot(cfg, sm)
    x1, y1 = client.pool_slice(cfg, sm, 0, cfg["P"])
    assert torch.equal(x0, x1) and torch.equal(y0, y1)


def test_multi_chunk_slices_and_probs(monkeypatch):
    monkeypatch.setattr(client, "POOL_CHUNK_BYTES", 4 * 6 * 64)     # 64 samples per chunk
    cfg = dict(round=1, pool_seed=7, P=300, C=4, heads_key="h")
    sm = sampler()
    L = client.pool_chunk_size(cfg, sm)
    assert L == 64
    chunks = [client.pool_chunk(cfg, sm, j) for j in range((cfg["P"] + L - 1) // L)]
    X = torch.cat([c[0] for c in chunks])
    Y = torch.cat([c[1] for c in chunks])
    assert len(Y) == cfg["P"]
    for o, n in [(0, 10), (60, 10), (63, 130), (250, 50), (0, 300)]:
        x, y = client.pool_slice(cfg, sm, o, n)
        assert torch.equal(x, X[o:o + n]) and torch.equal(y, Y[o:o + n])
    h = torch.nn.Linear(6, 4)
    p = client.pool_probs(cfg, sm, h, g=0)
    with torch.no_grad():
        assert torch.allclose(p, h(X).softmax(1))
    state = torch.random.get_rng_state()
    client.pool_chunk(cfg, sm, 2)
    assert torch.equal(state, torch.random.get_rng_state())   # the clients' own random stream is untouched


def test_draw_batched_equals_whole_draw():
    """draw() corrects batch by batch into one tensor; equal to drawing everything and correcting at once."""
    from fedecs import MCParams
    C, d = 4, 6
    mc = MCParams(torch.randn(C, d), torch.randn(C, d), torch.rand(C) + 0.5)
    sm = dict(G=FakeGen(), mc=mc, pool={}, shape=(d,))
    y = torch.randint(0, C, (9000,))
    torch.manual_seed(5)
    new = client.draw(sm, y)
    torch.manual_seed(5)
    old = torch.cat([sm["G"].sample(y[s:s + 4096]) for s in range(0, len(y), 4096)])
    old = mc.apply(old, y)
    assert torch.equal(new, old)
