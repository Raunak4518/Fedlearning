"""S1: Lemma 2 audit. A non-holder's row under torch.optim.Adam with coupled decay
(weight_decay=lam, the code path of experiments/core.py GenClient, lr=1e-3, eps=1e-8,
persistent optimiser state across rounds, row reset to the global value every round).
(a) lambda-invariance of D (norm loss per round) incl. tiny lambda (eps regime);
(b) sign changes / non-contraction once |r_j| < eta (oscillation floor);
(c) per-coordinate step can exceed eta when the state persists (jump at round start).
"""
import math, torch
torch.manual_seed(0)
d, S, R, eta = 512, 47, 6, 1e-3
b = 1 / math.sqrt(16 + 10)                    # PaperCVAE init bound, latent 16
r0 = (torch.rand(d) * 2 - 1) * b

def run(lam, r_init, rounds=R, steps=S, reset=True, glob_seq=None):
    p = torch.nn.Parameter(r_init.clone())
    opt = torch.optim.Adam([p], lr=eta, weight_decay=lam)
    Ds, maxstep = [], 0.0
    for t in range(rounds):
        if reset:
            with torch.no_grad():
                p.copy_(r_init if glob_seq is None else glob_seq[t])
        start = p.detach().clone()
        for s in range(steps):
            prev = p.detach().clone()
            opt.zero_grad(); p.grad = torch.zeros_like(p)   # data gradient on an absent row is exactly 0
            opt.step()
            maxstep = max(maxstep, float((p.detach() - prev).abs().max()))
        Ds.append(float(start.norm() - p.detach().norm()))
    return Ds, maxstep, p.detach()

print("(a) D per round (norm loss) for a non-holder; ||r0|| = %.4f" % r0.norm())
for lam in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]:
    Ds, ms, _ = run(lam, r0)
    print(f"  lam={lam:8.0e}  D(round1..{R}) = " + " ".join(f"{x:.4f}" for x in Ds) + f"   max|step|/eta={ms/eta:.3f}")

print("\n(b) long run (2000 steps, no reset) from r0, lam=1e-3: oscillation floor")
p = torch.nn.Parameter(r0.clone()); opt = torch.optim.Adam([p], lr=eta, weight_decay=1e-3)
signs = torch.sign(p.detach()); flips = 0; hist = []
for s in range(2000):
    opt.zero_grad(); p.grad = torch.zeros_like(p); opt.step()
    sg = torch.sign(p.detach()); flips += int((sg != signs).sum()); signs = sg
    if s in (46, 99, 199, 499, 999, 1999):
        hist.append((s + 1, float(p.detach().norm()), float(p.detach().abs().max())))
for h in hist:
    print(f"  step {h[0]:5d}: ||r|| = {h[1]:.5f} ({h[1]/float(r0.norm()):.4f} of init), max|r_j| = {h[2]:.2e}")
print(f"  coordinate sign flips over 2000 steps: {flips} (d = {d})")

print("\n(c) persistent state: global row restored each round after the non-holder had shrunk it")
Ds, ms, _ = run(1e-3, r0, rounds=6)
print(f"  max per-coordinate |step| / eta over 6 rounds = {ms/eta:.3f}  (Adam bound (1-b1)/sqrt(1-b2) = {0.1/math.sqrt(0.001):.3f})")
# a non-holder whose row was near zero then receives a large global row (e.g. holders regrew it)
small = r0 * 0.01; big = r0
p = torch.nn.Parameter(small.clone()); opt = torch.optim.Adam([p], lr=eta, weight_decay=1e-3)
for s in range(500):
    opt.zero_grad(); p.grad = torch.zeros_like(p); opt.step()
with torch.no_grad(): p.copy_(big)
prev = p.detach().clone(); opt.zero_grad(); p.grad = torch.zeros_like(p); opt.step()
print(f"  after 500 steps near 0, global row x100 restored: first step max|step|/eta = {float((p.detach()-prev).abs().max())/eta:.3f}")
