import numpy as np
from load import load
R=load()
arms=[("F10_combined_method","GeFL-F","mnist",1.0),("F10_combined_method","+HWA+LA+CSL","mnist",1.0),("E25_anchored_stack","Ours-A Ts=10","mnist",1.0),
("E20_proto_generator","GeFL-F","fmnist",1.0),("E16_csl_synthetic_budget","Ours (HWA+LA+CSL) Ts=10","fmnist",1.0),("E25_anchored_stack","Ours-A Ts=10","fmnist",1.0),("E32_moment_anchor_bayes","Ours-A-B Ts=10","fmnist",1.0),
("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",1.0),("K01_kaggle_cifar10_svhn__svhn","+CSL (beta=0.5)","svhn",1.0),("K10_kaggle_hybrid_svhn_cifar10__svhn","Ours+MC+KH (CVAE+HWA)","svhn",1.0),("K08_kaggle_anchored_stack__svhn","Ours-A Ts=10","svhn",1.0),
("K08_kaggle_anchored_stack__cifar10_iid","Ours-A (PC+MC+KH+LA+CSL)","cifar10",1.0),("K08_kaggle_anchored_stack__cifar10_iid","Ours-A Ts=10","cifar10",1.0),
("E25_anchored_stack","Ours-A Ts=10","mnist",0.01),("E20_proto_generator","GeFL-F","mnist",0.01),("K08_kaggle_anchored_stack__svhn","Ours-A Ts=10","svhn",0.01),("K01_kaggle_cifar10_svhn__svhn","GeFL-F","svhn",0.01),("K08_kaggle_anchored_stack__cifar10_lt","Ours-A Ts=10","cifar10",0.01)]
print("| run | best_mean_acc | final_acc (last round) | best − last | #eval points | best-round location |"); print("|---|---|---|---|---|---|")
for f,l,ds,IF in arms:
    s=R[(f,l,ds,IF,10)]
    b=np.mean([100*r["best_mean_acc"] for r in s.values()]); fa=np.mean([100*r["final_acc"] for r in s.values()])
    h=list(s.values())[0].get("history",[])
    npts=len(h)
    # where does mean acc peak
    accs=[x.get("acc") for x in h]; st=[f'{x.get("stage")}:{x.get("round")}' for x in h]
    loc=st[int(np.argmax(accs))] if accs and None not in accs else "?"
    print(f"| {f} / {l} / {ds} IF={IF} | {b:.2f} | {fa:.2f} | {b-fa:+.2f} | {npts} | {loc} (seed {list(s)[0]}) |")
