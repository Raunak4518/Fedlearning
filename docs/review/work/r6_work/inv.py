import numpy as np, torch, math
exec(open('mia_rff.py').read().split("for seed in (0,1,2):")[0])
torch.manual_seed(0)
for seed in (0,1):
  pools,held,rest=pools_for(seed)
  f=lambda i: torch.tensor(X[i]).flatten(1)
  V=[]
  for c in range(10):
      Z=f(pools[c]); V.append(float(((Z-Z.mean(0))**2).sum(1).mean()))
  s0=math.sqrt(2*np.mean(V)); phi=rff(784,6144,s0,1234+seed)
  c=9; mem=f(pools[c]); n=len(mem); K=phi(mem).sum(0); S1=mem.sum(0); Q=(mem**2).sum()
  ref=f(held[c])
  Z=(ref[torch.randint(0,200,(n,))]+0.05*torch.randn(n,784)).clamp(0,1).requires_grad_(True)
  opt=torch.optim.Adam([Z],lr=0.01)
  for t in range(3000):
      Zc=Z.clamp(0,1)
      loss=((phi(Zc).sum(0)-K)**2).sum()*100+((Zc.sum(0)-S1)**2).sum()/n+((Zc**2).sum()-Q)**2/n**2
      opt.zero_grad(); loss.backward(); opt.step()
  R=Z.detach().clamp(0,1)
  d_rec=torch.cdist(mem,R).min(1).values
  non=f(rest[c][:5000]); d_pub=torch.cdist(mem,non).min(1).values
  d_init=torch.cdist(mem,ref).min(1).values
  d_mean=(mem-mem.mean(0)).norm(dim=1)
  print(f"seed {seed}: median dist member->nearest recovered {d_rec.median():.2f}; ->nearest of 5000 other real class-9 imgs {d_pub.median():.2f}; ->nearest public held-out (init pool) {d_init.median():.2f}; ->exact mean {d_mean.median():.2f}; members closer to a recovered pt than to any other real image: {(d_rec<d_pub).sum().item()}/{n}; final loss {loss.item():.4f}")
