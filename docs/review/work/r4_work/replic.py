import numpy as np
from load import load
R=load()
groups={
 "GeFL-F MNIST LT final_bal":[("F01_main_longtail","GeFL-F"),("F10_combined_method","GeFL-F"),("E16_csl_synthetic_budget","GeFL-F"),("E20_proto_generator","GeFL-F"),("K02_kaggle_regimes_privacy_ablations","GeFL-F")],
 "GeFL-F FMNIST LT final_bal":[("F01_main_longtail","GeFL-F"),("F10_combined_method","GeFL-F"),("E16_csl_synthetic_budget","GeFL-F"),("E20_proto_generator","GeFL-F"),("K02_kaggle_regimes_privacy_ablations","GeFL-F")],
 "+HWA+LA MNIST LT final_bal":[("F01_main_longtail","+HWA+LA"),("F10_combined_method","+HWA+LA")],
 "Ours-A Ts=10 MNIST LT final_bal":[("E25_anchored_stack","Ours-A Ts=10"),("K09_kaggle_ablation_final__mnist","Ours-A Ts=10")],
 "Ours-A Ts=10 FMNIST LT final_bal":[("E25_anchored_stack","Ours-A Ts=10"),("K09_kaggle_ablation_final__fmnist","Ours-A Ts=10")],
 "Ours-A Ts=1 SVHN LT final_bal":[("K08_kaggle_anchored_stack__svhn","Ours-A (PC+MC+KH+LA+CSL)"),("K11_kaggle_covariance_anchoring__svhn","Ours-A (PC+MC+KH+LA+CSL)")],
 "GeFL-F MNIST IID best_mean_acc":[("F09_consensus_paper_setting","GeFL-F"),("F10_combined_method","GeFL-F"),("E16_csl_synthetic_budget","GeFL-F")],
 "GeFL-F FMNIST IID best_mean_acc":[("F09_consensus_paper_setting","GeFL-F"),("F10_combined_method","GeFL-F"),("E16_csl_synthetic_budget","GeFL-F"),("E20_proto_generator","GeFL-F")],
}
print("| configuration | jobs | per-job means | range of job means | within-seed (replicate) sd | between-seed sd of seed means | share of variance due to seed |")
print("|---|---|---|---|---|---|---|")
for name,lst in groups.items():
    ds="mnist" if "MNIST" in name and "FMNIST" not in name else ("fmnist" if "FMNIST" in name else "svhn")
    IF=1.0 if "IID" in name else 0.01
    met="best_mean_acc" if "best_mean" in name else "final_bal"
    M=[]
    for f,l in lst:
        s=R[(f,l,ds,IF,10)]; M.append([100*s[k][met] for k in (0,1,2)])
    M=np.array(M)  # jobs x seeds
    within=np.sqrt(M.var(axis=0,ddof=1).mean()); seedmeans=M.mean(0); between=seedmeans.std(ddof=1)
    # crude variance components (two-way, no interaction): seed var = var(seed means) - within^2/J
    J=M.shape[0]; vs=max(between**2-within**2/J,0); share=vs/(vs+within**2)
    print(f"| {name} | {J} | {', '.join(f'{m:.2f}' for m in M.mean(1))} | {M.mean(1).max()-M.mean(1).min():.2f} | {within:.2f} | {between:.2f} | {share:.2f} |")
