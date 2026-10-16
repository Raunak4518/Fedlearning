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

**svhn**: selected T_s = 10 (rule: 1); Ours readout = ccvr; Ours test = 65.56

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | ccvr | 64.02 | +1.54 [+0.63, +2.46] | 0.0371 |
| GeFL-F + AdamW + LA | ccvr | 64.08 | +1.48 [+0.38, +2.59] | 0.0371 |
| Real only | er | 54.34 | +11.22 [+8.77, +13.67] | 0.0154 |
| Real only + LA | er | 54.29 | +11.27 [+8.75, +13.80] | 0.0154 |
| Real only + FedLC | er | 54.32 | +11.24 [+8.74, +13.74] | 0.0154 |
| Real only + FedRS | er | 54.32 | +11.24 [+8.74, +13.73] | 0.0154 |

**cifar10**: not run yet.

**cifar100**: selected T_s = 10 (rule: 1); Ours readout = er; Ours test = 18.74

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | er | 14.75 | +3.98 [+2.59, +5.38] | 0.0197 |
| GeFL-F + AdamW + LA | er | 15.73 | +3.00 [+2.32, +3.69] | 0.0167 |
| Real only | er | 16.93 | +1.80 [+1.34, +2.26] | 0.0175 |
| Real only + LA | er | 17.10 | +1.64 [+0.65, +2.62] | 0.0197 |
| Real only + FedLC | er | 16.94 | +1.79 [+1.33, +2.26] | 0.0175 |
| Real only + FedRS | er | 17.02 | +1.72 [+1.05, +2.39] | 0.0197 |

P4 (label-free gap): Spearman(ro_disagree, er_bal - final_bal) = 0.22 (p = 0.061, n = 72 runs); prediction >= 0.7.
P2 (validation picks the same T_s as the old rule): 1 of 3 datasets; prediction >= 4 of 5.
