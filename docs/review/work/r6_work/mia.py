import numpy as np, torch, math, sys
torch.manual_seed(0)
root='/home/user/Fedlearning/experiments/data/MNIST/raw/'
def rd(p,off): return np.frombuffer(open(p,'rb').read(),dtype=np.uint8,offset=off)
y=rd(root+'train-labels-idx1-ubyte',8).astype(int)
X=rd(root+'train-images-idx3-ubyte',16).reshape(-1,28,28).astype(np.float32)/255.
C=10; IF=0.01; frac=0.1
def pools_for(seed):
    rng=np.random.RandomState(seed)
    n_total=int(round(frac*len(y))); mu=IF**(1/(C-1)); prof=mu**np.arange(C)
    want=np.floor(n_total*prof/prof.sum()).astype(int)
    by=[rng.permutation(np.where(y==c)[0]) for c in range(C)]
    held=[bc[:200] for bc in by]; pools=[bc[200:200+min(want[c],len(bc)-200)] for c,bc in enumerate(by)]
    rest=[bc[200+len(pools[c]):] for c,bc in enumerate(by)]
    return pools,held,rest
# random 1-conv-block FE -> 3x16x16=768, ReLU (surrogate for the frozen FE)
conv=torch.nn.Conv2d(1,3,4,2,1); 
def fe(idx):
    x=torch.tensor(X[idx])[:,None]; x=torch.nn.functional.pad(x,(2,2,2,2))
    with torch.no_grad(): return torch.relu(conv(x)).flatten(1)
def auc(pos,neg):
    pos=np.asarray(pos); neg=np.asarray(neg)
    r=np.concatenate([pos,neg]).argsort().argsort()+1
    return (r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
def rho(eps,d=1e-5): L=math.log(1/d); return (math.sqrt(L+eps)-math.sqrt(L))**2
out=[]
for seed in (0,1,2):
  pools,held,rest=pools_for(seed)
  for c in (9,8,6):
    for space in ('pixel','fe768'):
      f=(lambda i: torch.tensor(X[i]).flatten(1)) if space=='pixel' else fe
      mem=f(pools[c]); ref=f(held[c]); non=f(rest[c][:2000])
      clip=float(torch.quantile(ref.norm(dim=1),0.9))
      mref=ref.mean(0)
      for eps in (None,8,2):
        m=mem*(clip/mem.norm(dim=1,keepdim=True)).clamp(max=1.0) if eps else mem
        S=m.sum(0); n=float(len(m))
        if eps:
            s=math.sqrt(2/rho(eps)); g=torch.Generator().manual_seed(seed*7+c)
            S=S+s*clip*torch.randn(S.shape,generator=g); n=max(1.0,n+s*float(torch.randn(1,generator=g)))
        rel=S/n
        sc=lambda Z: ((Z-mref)@(rel-mref)).numpy()
        a=auc(sc(mem),sc(non))
        out.append((seed,c,len(mem),space,eps,a))
        print(f"seed {seed} class {c} n={len(mem)} {space:6s} eps={eps}: membership AUC={a:.3f}")
