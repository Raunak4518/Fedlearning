"""N4: Proposition 2.8 (additive per-client logit corrections are a bias reparametrisation) and
Corollary (FedRS = per-class learning-rate rescaling of the absent-class rows), checked to float precision.

Head f(x) = W body(x) + b (body: conv-free MLP with ReLU, as a stand-in for PaperHeader; the statement only needs a
free last-layer bias). Same batch sequence for every run, SGD lr 0.1, no momentum, no weight decay.
 run LA   : loss CE(f(x) + a, y), init (theta0, b0)
 run CE'  : loss CE(f(x), y),     init (theta0, b0 + a)
 claim    : body and W identical at every step; b_LA = b_CE' - a.
 run RS   : loss CE(where(absent, alpha f, f), y), init (theta0, W0, b0), lr eta
 run CE'' : loss CE(f, y) with absent rows initialised at alpha*(W0, b0) and their lr eta*alpha^2
 claim    : body identical; absent rows of CE'' = alpha * absent rows of RS.
Also: with a synthetic phase (uniform labels, no adjustment) between real phases, LA is no longer a
reparametrisation (the two phases see different biases): report the difference.
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.abspath(_os.path.join(_HERE, "..", "..", ".."))
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
C, d, n = 10, 20, 600
X = torch.randn(n, d)
counts = torch.tensor([300, 150, 80, 40, 20, 10, 0, 0, 0, 0])  # client lacks classes 6-9
Y = torch.cat([torch.full((int(k),), c) for c, k in enumerate(counts)])
X = X + 1.5 * F.one_hot(Y, d)[:, :d].double()  # class signal
prior = (counts.double() + 1) / (counts.sum() + C)
a = torch.log(prior)  # LA offset (tau = 1), as core.local_log_prior
absent = counts == 0


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.fc = nn.Linear(16, C)

    def forward(self, x):
        return self.fc(self.body(x))


h0 = Head()
batches = [torch.randperm(n)[:64] for _ in range(400)]


def run(loss_fn, init_fn, lr_fn=None, syn=None):
    h = Head()
    h.load_state_dict(h0.state_dict())
    init_fn(h)
    for t, b in enumerate(batches):
        if syn is not None and t % 50 == 0:  # synthetic phase: 10 steps, uniform labels, plain CE
            for _ in range(10):
                xs = torch.randn(64, d) + 1.5 * F.one_hot(ys := torch.randint(0, C, (64,)), d)[:, :d].double()
                l = F.cross_entropy(h(xs), ys)
                h.zero_grad(); l.backward()
                with torch.no_grad():
                    for p in h.parameters():
                        p -= 0.1 * p.grad
        l = loss_fn(h(X[b]), Y[b])
        h.zero_grad(); l.backward()
        with torch.no_grad():
            for name, p in h.named_parameters():
                g = p.grad if lr_fn is None else lr_fn(name, p.grad)
                p -= 0.1 * g
    return h


def diff(h1, h2, names):
    s1, s2 = dict(h1.named_parameters()), dict(h2.named_parameters())
    return max(float((s1[k] - s2[k]).abs().max()) for k in names)


body_names = [k for k, _ in h0.named_parameters() if k.startswith("body")]
la = run(lambda z, y: F.cross_entropy(z + a, y), lambda h: None)
ce = run(lambda z, y: F.cross_entropy(z, y), lambda h: h.fc.bias.data.add_(a))
print("LA vs CE with shifted bias init: max|body diff| = %.2e, max|W diff| = %.2e, max|b_LA - (b_CE - a)| = %.2e"
      % (diff(la, ce, body_names), diff(la, ce, ["fc.weight"]), float((la.fc.bias - (ce.fc.bias - a)).abs().max())))
plain = run(lambda z, y: F.cross_entropy(z, y), lambda h: None)
print("   (for scale) LA vs plain CE from the SAME init: max|body diff| = %.2e" % diff(la, plain, body_names))

alpha = 0.5


def rs_loss(z, y):
    return F.cross_entropy(torch.where(absent[None, :], alpha * z, z), y)


def init_scaled(h):
    with torch.no_grad():
        h.fc.weight[absent] *= alpha
        h.fc.bias[absent] *= alpha


def lr_scaled(name, g):
    if name.startswith("fc."):
        g = g.clone()
        g[absent] *= alpha ** 2
    return g


rs = run(rs_loss, lambda h: None)
ce2 = run(lambda z, y: F.cross_entropy(z, y), init_scaled, lr_scaled)
print("FedRS vs CE with absent rows scaled by alpha and lr*alpha^2: max|body diff| = %.2e, max|alpha*W_RS - W_CE| (absent) = %.2e"
      % (diff(rs, ce2, body_names), float((alpha * rs.fc.weight[absent] - ce2.fc.weight[absent]).abs().max())))

la_s = run(lambda z, y: F.cross_entropy(z + a, y), lambda h: None, syn=True)
ce_s = run(lambda z, y: F.cross_entropy(z, y), lambda h: h.fc.bias.data.add_(a), syn=True)
print("With a synthetic phase (uniform labels, no adjustment): LA vs shifted-init CE max|body diff| = %.2e (no longer 0)"
      % diff(la_s, ce_s, body_names))
