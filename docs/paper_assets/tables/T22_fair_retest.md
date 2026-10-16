### T22_fair_retest: validation-selected comparison on new seeds 3-5 (K17, long tail)

Selection on validation only (pre-registered). Test-set balanced accuracy of each method's selected readout; difference = Ours - baseline, paired over seeds, 95% CI, Holm-adjusted p.

**mnist**: selected T_s = 10 (rule: 10); Ours readout = ccvr; Ours test = 95.57

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | ccvr | 92.18 | +3.40 [+3.22, +3.58] | 0.000901 |
| GeFL-F + AdamW + LA | ccvr | 93.53 | +2.04 [+1.85, +2.24] | 0.00147 |
| Real only | ccvr | 90.88 | +4.70 [+4.35, +5.05] | 0.00138 |
| Real only + LA | ccvr | 90.92 | +4.65 [+3.62, +5.68] | 0.00264 |
| Real only + FedLC | ccvr | 90.92 | +4.65 [+4.19, +5.11] | 0.00147 |
| Real only + FedRS | ccvr | 90.88 | +4.70 [+4.36, +5.03] | 0.00138 |

**fmnist**: selected T_s = 10 (rule: 10); Ours readout = ccvr; Ours test = 82.08

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | ccvr | 81.07 | +1.01 [-0.04, +2.06] | 0.107 |
| GeFL-F + AdamW + LA | ccvr | 81.57 | +0.51 [-0.18, +1.20] | 0.107 |
| Real only | ccvr | 79.47 | +2.60 [+1.69, +3.52] | 0.0332 |
| Real only + LA | ccvr | 79.52 | +2.56 [+1.57, +3.55] | 0.0332 |
| Real only + FedLC | ccvr | 79.47 | +2.61 [+1.66, +3.56] | 0.0332 |
| Real only + FedRS | ccvr | 79.42 | +2.66 [+1.83, +3.50] | 0.0317 |

**svhn**: selected T_s = 10 (rule: 1); Ours readout = ccvr; Ours test = 65.56

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | ccvr | 64.02 | +1.54 [+0.63, +2.46] | 0.0371 |
| GeFL-F + AdamW + LA | ccvr | 64.08 | +1.48 [+0.38, +2.59] | 0.0371 |
| Real only | er | 54.34 | +11.22 [+8.77, +13.67] | 0.0154 |
| Real only + LA | er | 54.29 | +11.27 [+8.75, +13.80] | 0.0154 |
| Real only + FedLC | er | 54.32 | +11.24 [+8.74, +13.74] | 0.0154 |
| Real only + FedRS | er | 54.32 | +11.24 [+8.74, +13.73] | 0.0154 |

**cifar10**: selected T_s = 1 (rule: 1); Ours readout = er; Ours test = 53.14

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | er | 50.72 | +2.42 [+1.83, +3.01] | 0.0189 |
| GeFL-F + AdamW + LA | er | 51.64 | +1.50 [+0.84, +2.15] | 0.051 |
| Real only | er | 51.55 | +1.59 [-0.16, +3.35] | 0.239 |
| Real only + LA | er | 51.62 | +1.52 [-0.17, +3.21] | 0.239 |
| Real only + FedLC | er | 51.54 | +1.59 [-0.21, +3.40] | 0.239 |
| Real only + FedRS | er | 51.53 | +1.61 [-0.21, +3.43] | 0.239 |

CIFAR-10 gate check (P3): anchor branch minus forced row branch = -0.13 points.

**cifar100**: selected T_s = 10 (rule: 1); Ours readout = er; Ours test = 18.74

| baseline | selected readout | baseline test | Ours - baseline [95% CI] | p (Holm) |
|---|---|---|---|---|
| GeFL-F | er | 14.75 | +3.98 [+2.59, +5.38] | 0.0197 |
| GeFL-F + AdamW + LA | er | 15.73 | +3.00 [+2.32, +3.69] | 0.0167 |
| Real only | er | 16.93 | +1.80 [+1.34, +2.26] | 0.0175 |
| Real only + LA | er | 17.10 | +1.64 [+0.65, +2.62] | 0.0197 |
| Real only + FedLC | er | 16.94 | +1.79 [+1.33, +2.26] | 0.0175 |
| Real only + FedRS | er | 17.02 | +1.72 [+1.05, +2.39] | 0.0197 |

P4 (label-free gap): Spearman(ro_disagree, er_bal - final_bal) = 0.28 (p = 0.0018, n = 123 runs); prediction >= 0.7.
P2 (validation picks the same T_s as the old rule): 3 of 5 datasets; prediction >= 4 of 5.
