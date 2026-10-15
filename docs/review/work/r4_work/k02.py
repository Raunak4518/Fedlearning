import numpy as np
from scipy import stats
from load import load
R=load(); f="K02_kaggle_regimes_privacy_ablations"
def g(l,ds,IF=0.01): return {s:100*r["final_bal"] for s,r in R[(f,l,ds,IF,10)].items()}
for ds in ["mnist","fmnist"]:
    o=g("Ours",ds); ge=g("GeFL-F",ds)
    for l in ["Ours, DP eps=10","Ours, DP eps=1","Ours, DP eps=0.1","no HWA (flat averaging)","no LA (plain CE)","no CSL","HWA weights = counts n","HWA weights = uniform over holders"]:
        x=g(l,ds); S=sorted(x); d=[x[s]-o[s] for s in S]; d2=[x[s]-ge[s] for s in S]
        print(f"{ds} {l}: mean {np.mean(list(x.values())):.2f}; vs Ours {np.mean(d):+.2f} p={(stats.ttest_rel([x[s] for s in S],[o[s] for s in S]).pvalue if len(S)>1 else float("nan")):.3f} n={len(S)}; vs GeFL-F {np.mean(d2):+.2f}")
    # gain by regime
    gains=[]
    for IF in (1.0,0.1,0.01):
        o_=g("Ours",ds,IF); ge_=g("GeFL-F",ds,IF); d=np.array([o_[s]-ge_[s] for s in (0,1,2)]); gains.append(d)
        print(f"  regime IF={IF}: gain {d.mean():+.2f} per-seed {d.round(2)}")
    print("  gain(IF=100)-gain(IF=1) paired p:", stats.ttest_rel(gains[2],gains[0]).pvalue.round(4))
