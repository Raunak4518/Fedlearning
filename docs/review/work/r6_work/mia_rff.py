import numpy as np, torch, math
exec(open('mia.py').read().split("out=[]")[0])
def rff(d,D,s0,seed):
    g=torch.Generator().manual_seed(seed); parts=[torch.randn(D//3,d,generator=g)/(f*s0) for f in (0.5,1.,2.)]
    W=torch.cat(parts); b=torch.rand(W.shape[0],generator=g)*2*math.pi
    return lambda x: math.sqrt(2/W.shape[0])*torch.cos(x@W.T+b)
for seed in (0,1,2):
  pools,held,rest=pools_for(seed)
  allp=np.concatenate(pools)
  for space in ('pixel','fe768'):
    f=(lambda i: torch.tensor(X[i]).flatten(1)) if space=='pixel' else fe
    # bandwidth as in code: s0 = sqrt(2 * mean_c V_c) over the federation
    V=[]; 
    for c in range(10):
        Z=f(pools[c]); V.append(float(((Z-Z.mean(0))**2).sum(1).mean()))
    s0=math.sqrt(2*np.mean(V)); phi=rff(f(pools[0][:1]).shape[1],6144,s0,1234+seed)
    for c in (9,8):
        mem,ref,non=f(pools[c]),f(held[c]),f(rest[c][:2000])
        mu=phi(mem).mean(0); mref=phi(ref).mean(0)
        sc=lambda Z:((phi(Z)-mref)@(mu-mref)).numpy()
        pm,pn=sc(mem),sc(non); a=auc(pm,pn)
        thr=np.quantile(pn,0.99); tpr=(pm>thr).mean()
        print(f"seed {seed} {space} class {c} n={len(mem)} RFF-KME tracing AUC={a:.3f}  TPR@1%FPR={tpr:.2f}")
