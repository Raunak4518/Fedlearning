### T22_fair_retest: validation-selected comparison on new seeds 3-5 (K17, long tail)

Selection on validation only (pre-registered). Test-set balanced accuracy of each method's selected readout; difference = Ours - baseline, paired over seeds, 95% CI, Holm-adjusted p.

**mnist**: selected T_s = 10 (rule: 10); Ours readout = bbc; Ours test = 95.69

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | ccvr | 92.07 | +3.62 [+3.48, +3.76] | 0.000417 |
| Real only | ccvr | 91.19 | +4.51 [+3.33, +5.69] | 0.014 |
| Real only + LA | ccvr | 91.18 | +4.51 [+3.26, +5.77] | 0.014 |
| Real only + FedLC | ccvr | 91.13 | +4.56 [+3.34, +5.78] | 0.014 |
| Real only + FedRS | ccvr | 91.18 | +4.51 [+3.36, +5.67] | 0.014 |

**fmnist**: not run yet.

**svhn**: not run yet.

**cifar10**: not run yet.

**cifar100**: not run yet.

P4 (label-free gap): Spearman(ro_disagree, er_bal - final_bal) = 0.97 (p = 1.3e-13, n = 21 runs); prediction >= 0.7.
P2 (validation picks the same T_s as the old rule): 1 of 1 datasets; prediction >= 4 of 5.
