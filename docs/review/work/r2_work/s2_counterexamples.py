"""S2: counterexamples.
(A) old Prop 1 (paper_single_idea.md A.1 'earlier form'):  Delta||w_c|| <= (2 m_c/K - 1) S eta.
(B) 'A class held by every client cannot be contracted by the aggregation' (paper_draft.md:125-126).
All local updates use torch.optim.Adam(lr=eta, weight_decay=lam), exactly as GenClient."""
import torch, math
eta = 1e-3

def local(r, grads, lam=1e-3):
    p = torch.nn.Parameter(r.clone()); opt = torch.optim.Adam([p], lr=eta, weight_decay=lam)
    for g in grads:
        opt.zero_grad(); p.grad = g.clone(); opt.step()
    return p.detach()

print("(A1) K=1, m=1 (holder only), d=2, S=1. Bound: Delta <= (2*1/1-1)*S*eta = eta")
r = torch.tensor([1.0, 1.0]); rk = local(r, [torch.tensor([-1.0, -1.0])])
print(f"    Delta = {float(rk.norm()-r.norm()):.6e}  vs bound {eta:.1e}  -> ratio {float(rk.norm()-r.norm())/eta:.3f} (= sqrt(d))")
print("    with d=512: Delta = %.4e (bound 1e-3)" % float(local(torch.ones(512), [-torch.ones(512)]).norm() - torch.ones(512).norm()))

print("(A2) K=1, m=0 (one non-holder), d=1, S=1, r=1e-4 < eta. Bound: Delta <= -S*eta = -1e-3")
r = torch.tensor([1e-4]); rk = local(r, [torch.zeros(1)])
print(f"    r_k = {float(rk):.3e}; Delta = {float(rk.abs()-r.abs()):+.3e}  (norm GROWS: Adam overshoots zero)")

print("(A3) K=2, m=1, d=1, S=1, r=0 (row at zero). Bound: Delta <= 0")
r = torch.zeros(1); r_h = local(r, [torch.tensor([-1.0])]); r_n = local(r, [torch.zeros(1)])
agg = (r_h + r_n) / 2
print(f"    holder {float(r_h):.1e}, non-holder {float(r_n):.1e}, aggregate {float(agg):.1e}: Delta = {float(agg.abs()):+.1e} > 0")

print("(B) m_c = K = 2, every holder GROWS the row, flat average contracts it")
r = torch.tensor([1.0, 0.0])
# each holder's data pulls the row towards its own target (different clients, different class sub-modes)
g1 = lambda p: p - torch.tensor([0.5, 1.2]); g2 = lambda p: p - torch.tensor([0.5, -1.2])
def local_fn(r, gfun, S=3000, lam=1e-3):
    p = torch.nn.Parameter(r.clone()); opt = torch.optim.Adam([p], lr=eta, weight_decay=lam)
    for _ in range(S):
        opt.zero_grad(); p.grad = gfun(p.detach()); opt.step()
    return p.detach()
r1, r2 = local_fn(r, g1), local_fn(r, g2)
agg = (r1 + r2) / 2
print(f"    ||r||=1.000, ||r_1||={float(r1.norm()):.3f}, ||r_2||={float(r2.norm()):.3f} (both grew: Gbar>0), "
      f"||r+||={float(agg.norm()):.3f}  -> contracted by the aggregation alone")
