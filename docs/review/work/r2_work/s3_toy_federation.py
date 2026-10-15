"""S3: toy federated row dynamics with the code's exact optimiser semantics
(torch Adam, coupled weight_decay, lr 1e-3, PERSISTENT per-client state, flat 1/K average
or HWA). One row r_c in R^d. A client holding class c with in-batch share pi sees, per
step, c_b ~ Binomial(B, pi) samples; data gradient c_b * (r - t_k) (sum-reduced loss,
as F.mse_loss(..., reduction='sum') in PaperCVAE.loss); otherwise exactly 0 data gradient.
Questions: (1) is the whole trajectory lambda-invariant (Lemma 2 claims only D-bar);
(2) do HOLDERS shrink the row on steps whose batch lacks c (G-bar small or negative)?
(3) can a class held by EVERY client contract? (4) does HWA prevent contraction?"""
import torch, math
torch.manual_seed(0)
K, d, B, S, R, eta = 10, 64, 64, 47, 40, 1e-3

def simulate(pis, lam, agg="flat", seed=0, tnorm=1.5):
    g = torch.Generator().manual_seed(seed)
    b = 1 / math.sqrt(26)
    r = (torch.rand(d, generator=g) * 2 - 1) * b
    t = torch.randn(d, generator=g); t = t / t.norm() * tnorm
    tk = [t + 0.3 * torch.randn(d, generator=g) for _ in range(K)]     # client-specific class sub-modes
    ps = [torch.nn.Parameter(r.clone()) for _ in range(K)]
    opts = [torch.optim.Adam([p], lr=eta, weight_decay=lam) for p in ps]
    n0 = float(r.norm()); hold = [k for k in range(K) if pis[k] > 0]; non = [k for k in range(K) if pis[k] == 0]
    Gs, Ds, shrink = [], [], 0
    for rd in range(R):
        outs = []
        for k in range(K):
            with torch.no_grad(): ps[k].copy_(r)
            for s in range(S):
                cb = float(torch.distributions.Binomial(B, torch.tensor(pis[k])).sample()) if pis[k] > 0 else 0.0
                opts[k].zero_grad()
                ps[k].grad = cb * (ps[k].detach() - tk[k])
                opts[k].step()
            outs.append(ps[k].detach().clone())
        nr = float(r.norm())
        dk = [float(o.norm()) - nr for o in outs]
        Gs.append(sum(dk[k] for k in hold) / max(len(hold), 1))
        if non: Ds.append(-sum(dk[k] for k in non) / len(non))
        if agg == "flat":
            rn = sum(outs) / K
        else:  # HWA with E(n) ~ pi (counts proportional to pi)
            w = torch.tensor(pis); rn = sum(w[k] * outs[k] for k in range(K)) / w.sum()
        if float(rn.norm()) < nr: shrink += 1
        r = rn
    return float(r.norm()) / n0, sum(Gs) / len(Gs), (sum(Ds) / len(Ds) if Ds else float("nan")), shrink

cases = {
    "head: all 10 hold, pi=0.20 (in-batch ~1.00)": [0.2] * 10,
    "rare, ALL 10 hold, pi=0.003 (in-batch ~0.17)": [0.003] * 10,
    "rare, ALL 10 hold, pi=0.001 (in-batch ~0.06)": [0.001] * 10,
    "rare, 8/10 hold, pi=0.003": [0.003] * 8 + [0, 0],
    "rare, 4/10 hold, pi=0.05 (in-batch ~0.96)": [0.05] * 4 + [0] * 6,
}
print(f"K={K}, d={d}, S={S} steps/round, {R} rounds; columns: final ||r||/||r0||, mean holder gain G, mean non-holder loss D, rounds shrunk")
for name, pis in cases.items():
    for lam in [1e-3, 1e-4, 0.0]:
        f, G, D, sh = simulate(pis, lam)
        print(f"  {name:48s} lam={lam:6.0e} flat: final {f:6.3f}  G={G:+.4f}  D={D:.4f}  shrunk {sh:2d}/{R}")
    f, G, D, sh = simulate(pis, 1e-3, agg="hwa")
    print(f"  {name:48s} lam= 1e-03 HWA : final {f:6.3f}  G={G:+.4f}  shrunk {sh:2d}/{R}")
