import importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("core", "/home/user/Fedlearning/experiments/core.py")
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
cnts={"mnist":[5923,6742,5958,6131,5842,5421,5918,6265,5851,5949],"fmnist":[6000]*10,"cifar10":[5000]*10}
for ds,cc in cnts.items():
  y=torch.tensor(np.concatenate([[c]*n for c,n in enumerate(cc)]))
  spec_=core.DATASETS[ds]; B=spec_["B"]
  for (IF,alpha) in [(1.0,None),(100.0,0.5)]:
    for K in [10,50,100]:
      Ps=[]; 
      for seed in [0,1,2]:
        part=core.make_partition(y,K,spec_["frac"],IF,alpha,seed)
        sizes=[len(p) for p in part["parts"]]
        need1=[(sz//B if sz>=B else 1)*(B if sz>=B else sz) for sz in sizes]
        P1 = max(need1) if K==10 else sum(need1)
        Ps.append((sum(sizes),max(sizes),min(sizes),P1))
      a=np.array(Ps).mean(0)
      print(ds,"IF",IF,"K",K,"N=%.0f max=%.0f min=%.0f P(Ts=1)=%.0f"%tuple(a))
