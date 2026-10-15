import json, numpy as np
from scipy import stats
F="/home/user/Fedlearning/experiments/results/T03_exact_readout/runs.jsonl"
print("| run | n pairs | rho(B,dR) naive [p] | rho(1-a,dR) [p] | rho(B,dR) within-class (class-demeaned ranks) | rho(B,dR) within-head | class-mean rho (n=10) | partial rho(B,dR | 1-a) |")
print("|---|---|---|---|---|---|---|---|")
for l in open(F):
    r=json.loads(l); M=r["mechanism"]
    B=np.array([m["B"] for m in M]); dR=np.array([m["dR"] for m in M]); a=np.array([1-m["alpha"] for m in M])
    c=np.array([m["c"] for m in M]); g=np.array([m["g"] for m in M])
    rb=stats.spearmanr(B,dR); ra=stats.spearmanr(a,dR)
    def demean(x,grp):
        rx=stats.rankdata(x).astype(float)
        for u in np.unique(grp): rx[grp==u]-=rx[grp==u].mean()
        return rx
    wc=np.corrcoef(demean(B,c),demean(dR,c))[0,1]; wh=np.corrcoef(demean(B,g),demean(dR,g))[0,1]
    cm=stats.spearmanr([B[c==u].mean() for u in range(10)],[dR[c==u].mean() for u in range(10)])
    # partial Spearman of B and dR given (1-a)
    rB,rD,rA=[stats.rankdata(x) for x in (B,dR,a)]
    res=lambda y,x: y-np.polyval(np.polyfit(x,y,1),x)
    pr=np.corrcoef(res(rB,rA),res(rD,rA))[0,1]
    print(f"| {r['dataset']} {r['label']} | {len(M)} | {rb.correlation:.2f} [{rb.pvalue:.1e}] | {ra.correlation:.2f} [{ra.pvalue:.1e}] | {wc:.2f} | {wh:.2f} | {cm.correlation:.2f} [{cm.pvalue:.2f}] | {pr:.2f} |")
# T03b tau-norm recovery
print()
for l in open("/home/user/Fedlearning/experiments/results/T03_exact_readout/runs_b.jsonl"):
    r=json.loads(l)
    gap=r["oracle_bal"]-r["head_bal"]; rec=(r["tnorm_bal"]-r["head_bal"])/gap
    print(f"T03b {r['label']}: tau-norm recovers {100*rec:.0f}% of gap; ER-LDA {100*r['er_lda_bal']:.2f} vs tnorm {100*r['tnorm_bal']:.2f}; rho(wdef)={r['spearman_wdef']:.2f}")
