import numpy as np
from scipy import stats
from load import load
R=load()
LT,IID=0.01,1.0
def get(folder,label,ds,IF,K=10,metric="final_bal"):
    s=R[(folder,label,ds,float(IF),K)]
    return {sd:100*r[metric] for sd,r in s.items() if r.get(metric) is not None}
# K01 cifar summary-only values (mean, sd, n)
K01C={("GeFL-F",LT,"final_bal"):(33.74,0.51,3),("GeFL-F",IID,"best_mean_acc"):(59.10,0.63,3),
      ("GeFL-F",IID,"final_bal"):(56.68,0.85,3),("Ours (HWA+LA+CSL)",LT,"final_bal"):(42.10,0.74,3),
      ("FSG+LA",LT,"final_bal"):(44.17,0.87,3),("+HWA+LA",LT,"final_bal"):(40.73,0.78,3)}
def ms(x): v=np.array(list(x.values())); return v.mean(), v.std(ddof=1), len(v)
def comp(name, a, b, reported=None):
    """a = method, b = reference; dicts seed->value or (mean,sd,n) tuples"""
    out=f"| {name} | "
    if isinstance(a,dict): ma,sa,na=ms(a)
    else: ma,sa,na=a
    if isinstance(b,dict): mb,sb,nb=ms(b)
    else: mb,sb,nb=b
    out+=f"{ma:.2f} ± {sa:.2f} (n={na}) | {mb:.2f} ± {sb:.2f} (n={nb}) | {ma-mb:+.2f} | "
    if isinstance(a,dict) and isinstance(b,dict):
        sh=sorted(set(a)&set(b)); d=np.array([a[s]-b[s] for s in sh])
        t=stats.ttest_rel([a[s] for s in sh],[b[s] for s in sh])
        se=d.std(ddof=1)/np.sqrt(len(d)); tc=stats.t.ppf(0.975,len(d)-1)
        out+=f"{', '.join(f'{x:+.2f}' for x in d)} | paired p={t.pvalue:.3g}; 95% CI [{d.mean()-tc*se:+.2f}, {d.mean()+tc*se:+.2f}] | "
    else:
        out+="– | "
    w=stats.ttest_ind_from_stats(ma,sa,na,mb,sb,nb,equal_var=False)
    out+=f"Welch p={w.pvalue:.3g} | {reported or ''} |"
    print(out)
print("| comparison | method | reference | diff | per-seed diffs | paired | Welch | reported |")
print("|---|---|---|---|---|---|---|---|")
OA="Ours-A Ts=10"; OA1="Ours-A (PC+MC+KH+LA+CSL)"
comp("MNIST LT: Ours-A Ts10 (E25) vs GeFL-F (E20)", get("E25_anchored_stack",OA,"mnist",LT), get("E20_proto_generator","GeFL-F","mnist",LT), "94.82 vs 75.25, p=0.002")
comp("FMNIST LT: Ours-A Ts10 vs GeFL-F", get("E25_anchored_stack",OA,"fmnist",LT), get("E20_proto_generator","GeFL-F","fmnist",LT), "79.67 vs 58.47, p=0.011")
comp("MNIST LT BBC vs GeFL-F", get("E25_anchored_stack",OA,"mnist",LT,metric="bbc_bal"), get("E20_proto_generator","GeFL-F","mnist",LT), "95.17 (abstract 95.2)")
comp("FMNIST LT BBC vs GeFL-F", get("E25_anchored_stack",OA,"fmnist",LT,metric="bbc_bal"), get("E20_proto_generator","GeFL-F","fmnist",LT), "80.55 (abstract 80.6)")
comp("CIFAR LT: Ours-A Ts10 (K08) vs GeFL-F (K01 summary)", get("K08_kaggle_anchored_stack__cifar10_lt",OA,"cifar10",LT), K01C[("GeFL-F",LT,"final_bal")], "44.38 vs 33.74")
comp("CIFAR LT: Ours-A Ts1 vs GeFL-F", get("K08_kaggle_anchored_stack__cifar10_lt",OA1,"cifar10",LT), K01C[("GeFL-F",LT,"final_bal")], "46.06, +12.3 p=0.001 (abstract 46.1)")
comp("CIFAR LT: Ours-A Ts1 vs minimal", get("K08_kaggle_anchored_stack__cifar10_lt",OA1,"cifar10",LT), K01C[("Ours (HWA+LA+CSL)",LT,"final_bal")], "+4.0 p=0.017")
comp("CIFAR LT: Ours-A Ts1 vs FSG+LA", get("K08_kaggle_anchored_stack__cifar10_lt",OA1,"cifar10",LT), K01C[("FSG+LA",LT,"final_bal")], "+1.9 p=0.11")
comp("CIFAR LT: Ours-A Ts1+BBC vs FSG+LA", get("K08_kaggle_anchored_stack__cifar10_lt",OA1,"cifar10",LT,metric="bbc_bal"), K01C[("FSG+LA",LT,"final_bal")], "+4.8 p=0.003")
comp("CIFAR LT: Ts10 vs Ts1", get("K08_kaggle_anchored_stack__cifar10_lt",OA,"cifar10",LT), get("K08_kaggle_anchored_stack__cifar10_lt",OA1,"cifar10",LT), "44.4 vs 46.1 p=0.03")
comp("SVHN LT: Ours-A Ts10 (K08) vs GeFL-F (K01)", get("K08_kaggle_anchored_stack__svhn",OA,"svhn",LT), get("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",LT), "54.41 vs 50.49")
comp("SVHN LT: rows+MC+KH (K10) vs GeFL-F", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",LT), get("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",LT), "65.6 vs 50.5")
comp("SVHN LT: rows+MC+KH vs +HWA+LA", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",LT), get("K01_kaggle_cifar10_svhn__svhn","+HWA+LA (ours)","svhn",LT), "65.60 vs 62.45")
comp("SVHN LT: rows+MC+KH vs hybrid PCR+MC-S", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",LT), get("K11_kaggle_covariance_anchoring__svhn","Ours-R-S (PCR+MCS+KH+LA+CSL)","svhn",LT), "Welch p=0.07")
comp("SVHN LT: Ours-A-S vs Ours-A (K11)", get("K11_kaggle_covariance_anchoring__svhn","Ours-A-S (PC+MCS+KH+LA+CSL)","svhn",LT), get("K11_kaggle_covariance_anchoring__svhn",OA1,"svhn",LT), "+6.0 p=0.016")
print("| **IID (best_mean_acc)** |||||||")
B="best_mean_acc"
comp("MNIST IID: Ours-A Ts10 (E25) vs GeFL-F (F10)", get("E25_anchored_stack",OA,"mnist",IID,metric=B), get("F10_combined_method","GeFL-F","mnist",IID,metric=B), "97.62 vs 96.01, p=0.001")
comp("MNIST IID: Ours-A Ts10 vs minimal (F10)", get("E25_anchored_stack",OA,"mnist",IID,metric=B), get("F10_combined_method","+HWA+LA+CSL","mnist",IID,metric=B), "+0.85 p=0.001")
comp("FMNIST IID: Ours-A Ts10 vs GeFL-F (E20)", get("E25_anchored_stack",OA,"fmnist",IID,metric=B), get("E20_proto_generator","GeFL-F","fmnist",IID,metric=B), "83.96 vs 82.76 p=0.021")
comp("FMNIST IID: Ours-A Ts10 vs minimal Ts10 (E16)", get("E25_anchored_stack",OA,"fmnist",IID,metric=B), get("E16_csl_synthetic_budget","Ours (HWA+LA+CSL) Ts=10","fmnist",IID,metric=B), "-0.4 p=0.18")
comp("CIFAR IID: Ours-A Ts10 vs GeFL-F (K01 summary)", get("K08_kaggle_anchored_stack__cifar10_iid",OA,"cifar10",IID,metric=B), K01C[("GeFL-F",IID,"best_mean_acc")], "60.93 vs 59.10, p~0.1")
comp("CIFAR IID: final_bal Ours-A Ts10 vs GeFL-F", get("K08_kaggle_anchored_stack__cifar10_iid",OA,"cifar10",IID), K01C[("GeFL-F",IID,"final_bal")], "not reported")
comp("CIFAR IID: final_bal Ours-A Ts1 vs GeFL-F", get("K08_kaggle_anchored_stack__cifar10_iid",OA1,"cifar10",IID), K01C[("GeFL-F",IID,"final_bal")], "not reported")
comp("SVHN IID: rows+MC+KH vs GeFL-F", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",IID,metric=B), get("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",IID,metric=B), "77.07 vs 75.82")
comp("SVHN IID: rows+MC+KH vs +CSL", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",IID,metric=B), get("K01_kaggle_cifar10_svhn__svhn","+CSL (beta=0.5)","svhn",IID,metric=B), "")
comp("SVHN IID final_bal: rows+MC+KH vs GeFL-F", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",IID), get("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",IID), "not reported")
print("| **Ours-A vs minimal LT / BBC / MC-B / KH** |||||||")
comp("MNIST LT: Ours-A Ts10 vs minimal Ts10 (E16)", get("E25_anchored_stack",OA,"mnist",LT), get("E16_csl_synthetic_budget","Ours (HWA+LA+CSL) Ts=10","mnist",LT), "+4.7, positive every seed")
comp("FMNIST LT: Ours-A Ts10 vs minimal Ts10 (E16)", get("E25_anchored_stack",OA,"fmnist",LT), get("E16_csl_synthetic_budget","Ours (HWA+LA+CSL) Ts=10","fmnist",LT), "+0.5, positive every seed")
comp("FMNIST LT: MC-B (E32) vs Ours-A (K09) [as in T15]", get("E32_moment_anchor_bayes","Ours-A-B Ts=10","fmnist",LT), get("K09_kaggle_ablation_final__fmnist",OA,"fmnist",LT), "81.57, p=0.03")
comp("FMNIST LT: MC-B (E32) vs Ours-A (E25, same machine)", get("E32_moment_anchor_bayes","Ours-A-B Ts=10","fmnist",LT), get("E25_anchored_stack",OA,"fmnist",LT), "")
comp("MNIST LT: MC-B (E32) vs Ours-A (K09)", get("E32_moment_anchor_bayes","Ours-A-B Ts=10","mnist",LT), get("K09_kaggle_ablation_final__mnist",OA,"mnist",LT), "94.76 p=0.75")
comp("FMNIST LT: MC-B vs oracle (E25 oracle_bal)", get("E32_moment_anchor_bayes","Ours-A-B Ts=10","fmnist",LT), get("E25_anchored_stack",OA,"fmnist",LT,metric="oracle_bal"), "'= oracle 81.55'")
comp("FMNIST IID: MC-B vs Ours-A (E25)", get("E32_moment_anchor_bayes","Ours-A-B Ts=10","fmnist",IID,metric=B), get("E25_anchored_stack",OA,"fmnist",IID,metric=B), "85.75")
comp("FMNIST LT: Ours-A Ts10 vs GeFL-F oracle recalibration bound", get("E25_anchored_stack",OA,"fmnist",LT), get("E20_proto_generator","GeFL-F","fmnist",LT,metric="oracle_bal"), "readout report +0.5")
comp("MNIST LT: Ours-A Ts10 vs GeFL-F oracle", get("E25_anchored_stack",OA,"mnist",LT), get("E20_proto_generator","GeFL-F","mnist",LT,metric="oracle_bal"), "readout report +4.1")
comp("SVHN LT: rows+MC+KH vs GeFL-F oracle", get("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",LT), get("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",LT,metric="oracle_bal"), "readout report +0.8")
for ds,IF,m in [("mnist",LT,"final_bal"),("fmnist",LT,"final_bal"),("fmnist",IID,B)]:
    comp(f"KH: {ds} IF={IF} Ours-A Ts1 (E25) vs PC+MC (E20) [{m}]", get("E25_anchored_stack",OA1,ds,IF,metric=m), get("E20_proto_generator","Ours-PC+MC",ds,IF,metric=m), "+0.80/+0.66/+0.43")
print("| **DP (E26) vs exact (E25)** |||||||")
for ds in ["mnist","fmnist"]:
    for e in [8,2]:
        comp(f"DP {ds} eps={e}", get("E26_private_anchoring",f"Ours-A Ts=10, DP eps={e}",ds,LT), get("E25_anchored_stack",OA,ds,LT), "")
        comp(f"DP {ds} eps={e} vs GeFL-F", get("E26_private_anchoring",f"Ours-A Ts=10, DP eps={e}",ds,LT), get("E20_proto_generator","GeFL-F",ds,LT), "")
