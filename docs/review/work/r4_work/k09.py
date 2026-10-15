import numpy as np, itertools
from scipy import stats
from load import load
R=load()
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for i,j in enumerate(o):
        run=max(run,(m-i)*p[j]); adj[j]=min(1,run)
    return adj
def bh(p):
    p=np.asarray(p); m=len(p); o=np.argsort(p)[::-1]; adj=np.empty(m); run=1
    for i,j in enumerate(o):
        rank=m-i; run=min(run,p[j]*m/rank); adj[j]=run
    return adj
arms=["-PC (CVAE+HWA)","-LA","-CSL","-MC","-KH"]
D={}; P=[]; names=[]
for ds in ["mnist","fmnist"]:
    f=f"K09_kaggle_ablation_final__{ds}"
    ref={s:100*r["final_bal"] for s,r in R[(f,"Ours-A Ts=10",ds,0.01,10)].items()}
    for a in arms:
        x={s:100*r["final_bal"] for s,r in R[(f,a,ds,0.01,10)].items()}
        d=np.array([x[s]-ref[s] for s in sorted(ref)]); D[(ds,a)]=d
        p=stats.ttest_rel([x[s] for s in sorted(ref)],[ref[s] for s in sorted(ref)]).pvalue
        P.append(p); names.append((ds,a))
H=holm(P); B=bh(P)
print("| dataset | removed | per-seed diff | mean | paired p | Holm (10) | BH (10) |"); print("|---|---|---|---|---|---|---|")
for (ds,a),p,h,b in zip(names,P,H,B):
    d=D[(ds,a)]; print(f"| {ds} | {a} | {', '.join(f'{v:+.2f}' for v in d)} | {d.mean():+.2f} | {p:.3f} | {h:.3f} | {b:.3f} |")
allneg=sum((d<0).sum() for d in D.values()); print("negatives:",allneg,"of",sum(len(d) for d in D.values()))
print("naive sign test 30/30 two-sided p =",2*0.5**30)
print("cluster-level sign test (6 dataset x seed clusters, all-negative) p =",2*0.5**6)
# within-cluster correlation of the 5 differences: does the reference draw drive them?
for ds in ["mnist","fmnist"]:
    M=np.array([D[(ds,a)] for a in arms])  # 5 x 3
    print(ds,"per-seed mean diff over arms:",M.mean(0).round(2), " between-seed sd of the cluster means:",M.mean(0).std(ddof=1).round(3))
# robustness: swap reference to E25 Ours-A Ts=10 (same config, same seeds, separate run)
print("\nReference swap: removal arm (K09) minus Ours-A Ts=10 from E25 (same label, same seeds, different job)")
neg=0;tot=0
for ds in ["mnist","fmnist"]:
    ref2={s:100*r["final_bal"] for s,r in R[("E25_anchored_stack","Ours-A Ts=10",ds,0.01,10)].items()}
    for a in arms:
        x={s:100*r["final_bal"] for s,r in R[(f"K09_kaggle_ablation_final__{ds}",a,ds,0.01,10)].items()}
        d=np.array([x[s]-ref2[s] for s in sorted(ref2)]); neg+=(d<0).sum(); tot+=3
        print(f"  {ds} {a}: {', '.join(f'{v:+.2f}' for v in d)}  p={stats.ttest_rel([x[s] for s in sorted(ref2)],[ref2[s] for s in sorted(ref2)]).pvalue:.3f}")
print("  negatives with swapped reference:",neg,"of",tot)
# Intersection-union: claim "every component helps" needs every individual test -> p_IUT = max p
print("\nIUT p (every component helps, both datasets) = max paired p =",max(P).round(3))
print("IUT p per dataset:", {ds:round(max(p for (d,a),p in zip(names,P) if d==ds),3) for ds in ["mnist","fmnist"]})
# exact sign-flip permutation per arm within K09 (n=3 -> min p 0.25)
print("min attainable two-sided p for a sign-flip/Wilcoxon test with n=3: ",2/8)
