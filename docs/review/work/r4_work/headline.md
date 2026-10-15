| comparison | method | reference | diff | per-seed diffs | paired | Welch | reported |
|---|---|---|---|---|---|---|---|
| MNIST LT: Ours-A Ts10 (E25) vs GeFL-F (E20) | 94.82 ± 0.57 (n=3) | 75.25 ± 1.12 (n=3) | +19.57 | +21.18, +19.73, +17.80 | paired p=0.00249; 95% CI [+15.36, +23.78] | Welch p=0.00012 | 94.82 vs 75.25, p=0.002 |
| FMNIST LT: Ours-A Ts10 vs GeFL-F | 79.67 ± 0.54 (n=3) | 58.47 ± 3.59 (n=3) | +21.21 | +17.16, +21.54, +24.91 | paired p=0.011; 95% CI [+11.56, +30.85] | Welch p=0.00829 | 79.67 vs 58.47, p=0.011 |
| MNIST LT BBC vs GeFL-F | 95.17 ± 0.55 (n=3) | 75.25 ± 1.12 (n=3) | +19.93 | +21.50, +20.12, +18.16 | paired p=0.00234; 95% CI [+15.77, +24.09] | Welch p=0.00013 | 95.17 (abstract 95.2) |
| FMNIST LT BBC vs GeFL-F | 80.55 ± 0.65 (n=3) | 58.47 ± 3.59 (n=3) | +22.09 | +17.88, +22.44, +25.94 | paired p=0.011; 95% CI [+12.05, +32.12] | Welch p=0.00723 | 80.55 (abstract 80.6) |
| CIFAR LT: Ours-A Ts10 (K08) vs GeFL-F (K01 summary) | 44.38 ± 1.07 (n=3) | 33.74 ± 0.51 (n=3) | +10.64 | – | Welch p=0.000761 | 44.38 vs 33.74 |
| CIFAR LT: Ours-A Ts1 vs GeFL-F | 46.06 ± 1.29 (n=3) | 33.74 ± 0.51 (n=3) | +12.32 | – | Welch p=0.00123 | 46.06, +12.3 p=0.001 (abstract 46.1) |
| CIFAR LT: Ours-A Ts1 vs minimal | 46.06 ± 1.29 (n=3) | 42.10 ± 0.74 (n=3) | +3.96 | – | Welch p=0.0167 | +4.0 p=0.017 |
| CIFAR LT: Ours-A Ts1 vs FSG+LA | 46.06 ± 1.29 (n=3) | 44.17 ± 0.87 (n=3) | +1.89 | – | Welch p=0.113 | +1.9 p=0.11 |
| CIFAR LT: Ours-A Ts1+BBC vs FSG+LA | 48.71 ± 0.96 (n=3) | 44.17 ± 0.87 (n=3) | +4.54 | – | Welch p=0.00382 | +4.8 p=0.003 |
| CIFAR LT: Ts10 vs Ts1 | 44.38 ± 1.07 (n=3) | 46.06 ± 1.29 (n=3) | -1.67 | -1.19, -1.65, -2.18 | paired p=0.0277; 95% CI [-2.90, -0.45] | Welch p=0.162 | 44.4 vs 46.1 p=0.03 |
| SVHN LT: Ours-A Ts10 (K08) vs GeFL-F (K01) | 54.41 ± 1.13 (n=3) | 50.49 ± 2.24 (n=3) | +3.92 | +2.91, +1.32, +7.54 | paired p=0.17; 95% CI [-4.10, +11.95] | Welch p=0.0741 | 54.41 vs 50.49 |
| SVHN LT: rows+MC+KH (K10) vs GeFL-F | 65.60 ± 1.50 (n=3) | 50.49 ± 2.24 (n=3) | +15.11 | +12.34, +13.89, +19.09 | paired p=0.0178; 95% CI [+6.32, +23.89] | Welch p=0.00118 | 65.6 vs 50.5 |
| SVHN LT: rows+MC+KH vs +HWA+LA | 65.60 ± 1.50 (n=3) | 62.45 ± 1.76 (n=3) | +3.15 | +3.59, +2.51, +3.35 | paired p=0.0106; 95% CI [+1.74, +4.56] | Welch p=0.0794 | 65.60 vs 62.45 |
| SVHN LT: rows+MC+KH vs hybrid PCR+MC-S | 65.60 ± 1.50 (n=3) | 62.89 ± 1.07 (n=3) | +2.71 | +2.15, +1.78, +4.22 | paired p=0.0704; 95% CI [-0.56, +5.99] | Welch p=0.0696 | Welch p=0.07 |
| SVHN LT: Ours-A-S vs Ours-A (K11) | 61.80 ± 1.28 (n=3) | 55.76 ± 1.68 (n=3) | +6.03 | +4.54, +7.02, +6.54 | paired p=0.0155; 95% CI [+2.76, +9.31] | Welch p=0.00923 | +6.0 p=0.016 |
| **IID (best_mean_acc)** |||||||
| MNIST IID: Ours-A Ts10 (E25) vs GeFL-F (F10) | 97.62 ± 0.06 (n=3) | 96.01 ± 0.14 (n=3) | +1.60 | +1.69, +1.62, +1.49 | paired p=0.00138; 95% CI [+1.35, +1.86] | Welch p=0.000666 | 97.62 vs 96.01, p=0.001 |
| MNIST IID: Ours-A Ts10 vs minimal (F10) | 97.62 ± 0.06 (n=3) | 96.77 ± 0.06 (n=3) | +0.85 | +0.85, +0.91, +0.80 | paired p=0.0014; 95% CI [+0.71, +0.99] | Welch p=5.09e-05 | +0.85 p=0.001 |
| FMNIST IID: Ours-A Ts10 vs GeFL-F (E20) | 83.96 ± 0.39 (n=3) | 82.76 ± 0.33 (n=3) | +1.20 | +0.86, +1.30, +1.45 | paired p=0.0214; 95% CI [+0.43, +1.97] | Welch p=0.0166 | 83.96 vs 82.76 p=0.021 |
| FMNIST IID: Ours-A Ts10 vs minimal Ts10 (E16) | 83.96 ± 0.39 (n=3) | 84.34 ± 0.40 (n=3) | -0.38 | -0.75, -0.20, -0.18 | paired p=0.182; 95% CI [-1.18, +0.43] | Welch p=0.31 | -0.4 p=0.18 |
| CIFAR IID: Ours-A Ts10 vs GeFL-F (K01 summary) | 60.93 ± 1.27 (n=3) | 59.10 ± 0.63 (n=3) | +1.83 | – | Welch p=0.112 | 60.93 vs 59.10, p~0.1 |
| CIFAR IID: final_bal Ours-A Ts10 vs GeFL-F | 56.82 ± 2.16 (n=3) | 56.68 ± 0.85 (n=3) | +0.14 | – | Welch p=0.925 | not reported |
| CIFAR IID: final_bal Ours-A Ts1 vs GeFL-F | 58.62 ± 1.43 (n=3) | 56.68 ± 0.85 (n=3) | +1.94 | – | Welch p=0.129 | not reported |
| SVHN IID: rows+MC+KH vs GeFL-F | 77.07 ± 0.19 (n=3) | 75.82 ± 0.55 (n=3) | +1.25 | +2.04, +1.14, +0.58 | paired p=0.0981; 95% CI [-0.57, +3.08] | Welch p=0.0471 | 77.07 vs 75.82 |
| SVHN IID: rows+MC+KH vs +CSL | 77.07 ± 0.19 (n=3) | 76.30 ± 0.30 (n=3) | +0.77 | +1.26, +0.76, +0.29 | paired p=0.11; 95% CI [-0.43, +1.98] | Welch p=0.0266 |  |
| SVHN IID final_bal: rows+MC+KH vs GeFL-F | 75.89 ± 0.44 (n=3) | 74.54 ± 0.35 (n=3) | +1.35 | +2.10, +1.41, +0.56 | paired p=0.0927; 95% CI [-0.56, +3.26] | Welch p=0.016 | not reported |
| **Ours-A vs minimal LT / BBC / MC-B / KH** |||||||
| MNIST LT: Ours-A Ts10 vs minimal Ts10 (E16) | 94.82 ± 0.57 (n=3) | 90.15 ± 1.35 (n=3) | +4.66 | +6.57, +4.70, +2.72 | paired p=0.0522; 95% CI [-0.11, +9.44] | Welch p=0.0155 | +4.7, positive every seed |
| FMNIST LT: Ours-A Ts10 vs minimal Ts10 (E16) | 79.67 ± 0.54 (n=3) | 79.23 ± 0.51 (n=3) | +0.45 | +0.26, +0.41, +0.67 | paired p=0.0653; 95% CI [-0.07, +0.96] | Welch p=0.357 | +0.5, positive every seed |
| FMNIST LT: MC-B (E32) vs Ours-A (K09) [as in T15] | 81.57 ± 0.31 (n=3) | 79.79 ± 0.31 (n=3) | +1.78 | +1.99, +2.19, +1.16 | paired p=0.0298; 95% CI [+0.43, +3.13] | Welch p=0.00219 | 81.57, p=0.03 |
| FMNIST LT: MC-B (E32) vs Ours-A (E25, same machine) | 81.57 ± 0.31 (n=3) | 79.67 ± 0.54 (n=3) | +1.89 | +2.38, +1.69, +1.61 | paired p=0.0164; 95% CI [+0.84, +2.95] | Welch p=0.0118 |  |
| MNIST LT: MC-B (E32) vs Ours-A (K09) | 94.76 ± 1.95 (n=3) | 95.18 ± 0.08 (n=3) | -0.42 | +0.81, +0.63, -2.70 | paired p=0.748; 95% CI [-5.32, +4.48] | Welch p=0.745 | 94.76 p=0.75 |
| FMNIST LT: MC-B vs oracle (E25 oracle_bal) | 81.57 ± 0.31 (n=3) | 81.55 ± 0.76 (n=3) | +0.01 | +0.83, -0.14, -0.65 | paired p=0.976; 95% CI [-1.86, +1.89] | Welch p=0.978 | '= oracle 81.55' |
| FMNIST IID: MC-B vs Ours-A (E25) | 85.75 ± 0.45 (n=3) | 83.96 ± 0.39 (n=3) | +1.78 | +1.74, +1.77, +1.85 | paired p=0.000337; 95% CI [+1.64, +1.92] | Welch p=0.00688 | 85.75 |
| FMNIST LT: Ours-A Ts10 vs GeFL-F oracle recalibration bound | 79.67 ± 0.54 (n=3) | 78.94 ± 0.31 (n=3) | +0.74 | -0.11, +1.25, +1.08 | paired p=0.228; 95% CI [-1.11, +2.59] | Welch p=0.129 | readout report +0.5 |
| MNIST LT: Ours-A Ts10 vs GeFL-F oracle | 94.82 ± 0.57 (n=3) | 91.21 ± 0.50 (n=3) | +3.61 | +4.69, +3.60, +2.54 | paired p=0.0282; 95% CI [+0.95, +6.27] | Welch p=0.00129 | readout report +4.1 |
| SVHN LT: rows+MC+KH vs GeFL-F oracle | 65.60 ± 1.50 (n=3) | 64.82 ± 0.99 (n=3) | +0.78 | +0.30, +0.15, +1.88 | paired p=0.294; 95% CI [-1.60, +3.16] | Welch p=0.501 | readout report +0.8 |
| KH: mnist IF=0.01 Ours-A Ts1 (E25) vs PC+MC (E20) [final_bal] | 93.11 ± 0.42 (n=3) | 92.31 ± 0.61 (n=3) | +0.80 | +1.24, +0.65, +0.51 | paired p=0.0695; 95% CI [-0.16, +1.76] | Welch p=0.144 | +0.80/+0.66/+0.43 |
| KH: fmnist IF=0.01 Ours-A Ts1 (E25) vs PC+MC (E20) [final_bal] | 78.22 ± 0.69 (n=3) | 77.56 ± 0.76 (n=3) | +0.66 | +0.35, +0.43, +1.19 | paired p=0.133; 95% CI [-0.49, +1.80] | Welch p=0.33 | +0.80/+0.66/+0.43 |
| KH: fmnist IF=1.0 Ours-A Ts1 (E25) vs PC+MC (E20) [best_mean_acc] | 83.35 ± 0.22 (n=3) | 82.92 ± 0.66 (n=3) | +0.43 | +0.07, +0.96, +0.25 | paired p=0.254; 95% CI [-0.73, +1.59] | Welch p=0.381 | +0.80/+0.66/+0.43 |
| **DP (E26) vs exact (E25)** |||||||
| DP mnist eps=8 | 90.36 ± 0.30 (n=3) | 94.82 ± 0.57 (n=3) | -4.46 | -4.78, -4.81, -3.77 | paired p=0.00581; 95% CI [-5.92, -2.99] | Welch p=0.00124 |  |
| DP mnist eps=8 vs GeFL-F | 90.36 ± 0.30 (n=3) | 75.25 ± 1.12 (n=3) | +15.11 | +16.40, +14.91, +14.03 | paired p=0.00209; 95% CI [+12.14, +18.09] | Welch p=0.000997 |  |
| DP mnist eps=2 | 85.06 ± 0.54 (n=3) | 94.82 ± 0.57 (n=3) | -9.76 | -9.97, -10.41, -8.90 | paired p=0.0021; 95% CI [-11.68, -7.83] | Welch p=2.88e-05 |  |
| DP mnist eps=2 vs GeFL-F | 85.06 ± 0.54 (n=3) | 75.25 ± 1.12 (n=3) | +9.81 | +11.22, +9.32, +8.90 | paired p=0.00522; 95% CI [+6.75, +12.88] | Welch p=0.00104 |  |
| DP fmnist eps=8 | 72.77 ± 1.67 (n=3) | 79.67 ± 0.54 (n=3) | -6.90 | -7.12, -8.60, -5.00 | paired p=0.0221; 95% CI [-11.40, -2.41] | Welch p=0.0125 |  |
| DP fmnist eps=8 vs GeFL-F | 72.77 ± 1.67 (n=3) | 58.47 ± 3.59 (n=3) | +14.30 | +10.05, +12.95, +19.91 | paired p=0.0394; 95% CI [+1.71, +26.89] | Welch p=0.00984 |  |
| DP fmnist eps=2 | 65.59 ± 0.58 (n=3) | 79.67 ± 0.54 (n=3) | -14.09 | -13.38, -15.26, -13.62 | paired p=0.00176; 95% CI [-16.63, -11.54] | Welch p=6.81e-06 |  |
| DP fmnist eps=2 vs GeFL-F | 65.59 ± 0.58 (n=3) | 58.47 ± 3.59 (n=3) | +7.12 | +3.78, +6.28, +11.29 | paired p=0.0843; 95% CI [-2.38, +16.62] | Welch p=0.072 |  |
