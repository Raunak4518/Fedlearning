# T01: contraction theorem checked on the real CVAE-F

## GeFL-F wd=0  (mnist, seed 0, K=10, 40 rounds, IF=0.01, alpha=0.5)

holders m_c per class: 0:10 1:10 2:10 3:10 4:10 5:9 6:9 7:8 8:7 9:8
total samples n_c:     0:2417 1:1449 2:868 3:520 4:312 5:187 6:112 7:67 8:40 9:24
row norm end / init: head 1.414, tail 1.219; tail/head at end 0.848
per class end / init:  0:1.40 1:1.44 2:1.40 3:1.45 4:1.38 5:1.49 6:1.41 7:1.20 8:1.14 9:1.12

* theorem violations (||r+|| > bound): **0 / 400** row-rounds
* certified contractions (bound < ||r||): 2; actual contractions: 2; certified but did not shrink: 0
* alignment slack ||r+|| / bound: mean 0.9994, min 0.9970
* row-rounds with m_c >= K/2 (old theorem: no contraction): 400; of these shrank: 2; certified by the new theorem: 2

| class | m_c | m_c/K | mean holder class-in-batch freq | Gbar/Dbar (median) | theta (median) | rounds certified to contract | rounds shrank |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 1.0 | 0.994 | nan | 0.000 | 0/40 | 0/40 |
| 1 | 10 | 1.0 | 0.972 | nan | 0.000 | 0/40 | 0/40 |
| 2 | 10 | 1.0 | 0.883 | nan | 0.000 | 0/40 | 0/40 |
| 3 | 10 | 1.0 | 0.776 | nan | 0.000 | 0/40 | 0/40 |
| 4 | 10 | 1.0 | 0.711 | nan | 0.000 | 0/40 | 0/40 |
| 5 | 9 | 0.9 | 0.750 | nan | 0.000 | 0/40 | 0/40 |
| 6 | 9 | 0.9 | 0.640 | nan | 0.000 | 0/40 | 0/40 |
| 7 | 8 | 0.8 | 0.590 | nan | 0.000 | 0/40 | 0/40 |
| 8 | 7 | 0.7 | 0.578 | nan | 0.000 | 1/40 | 1/40 |
| 9 | 8 | 0.8 | 0.326 | nan | 0.000 | 1/40 | 1/40 |
* non-holder norm loss per local step, rounds 0-4: mean 0.00e+00

## GeFL-F wd=1e-3  (mnist, seed 0, K=10, 40 rounds, IF=0.01, alpha=0.5)

holders m_c per class: 0:10 1:10 2:10 3:10 4:10 5:9 6:9 7:8 8:7 9:8
total samples n_c:     0:2417 1:1449 2:868 3:520 4:312 5:187 6:112 7:67 8:40 9:24
row norm end / init: head 1.422, tail 0.219; tail/head at end 0.152
per class end / init:  0:1.42 1:1.44 2:1.40 3:1.42 4:1.33 5:0.75 6:0.65 7:0.10 8:0.04 9:0.09

* theorem violations (||r+|| > bound): **0 / 400** row-rounds
* certified contractions (bound < ||r||): 139; actual contractions: 197; certified but did not shrink: 0
* alignment slack ||r+|| / bound: mean 0.9828, min 0.8397
* row-rounds with m_c >= K/2 (old theorem: no contraction): 400; of these shrank: 197; certified by the new theorem: 139

| class | m_c | m_c/K | mean holder class-in-batch freq | Gbar/Dbar (median) | theta (median) | rounds certified to contract | rounds shrank |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 1.0 | 0.994 | nan | 0.000 | 0/40 | 0/40 |
| 1 | 10 | 1.0 | 0.972 | nan | 0.000 | 0/40 | 0/40 |
| 2 | 10 | 1.0 | 0.883 | nan | 0.000 | 0/40 | 0/40 |
| 3 | 10 | 1.0 | 0.776 | nan | 0.000 | 1/40 | 1/40 |
| 4 | 10 | 1.0 | 0.711 | nan | 0.000 | 1/40 | 1/40 |
| 5 | 9 | 0.9 | 0.750 | 0.094 | 0.914 | 38/40 | 40/40 |
| 6 | 9 | 0.9 | 0.640 | 0.074 | 0.931 | 39/40 | 40/40 |
| 7 | 8 | 0.8 | 0.590 | 0.201 | 0.833 | 23/40 | 40/40 |
| 8 | 7 | 0.7 | 0.578 | 0.638 | 0.610 | 13/40 | 35/40 |
| 9 | 8 | 0.8 | 0.326 | 0.203 | 0.831 | 24/40 | 40/40 |

* holder gain G_k / Dbar vs P(class in batch): Spearman rho = -0.181 over 1640 holder-rounds
  * P in [0, 0.1): median G_k/Dbar = 0.261 (n=80)
  * P in [0.1, 0.5): median G_k/Dbar = 0.150 (n=640)
  * P in [0.5, 0.9): median G_k/Dbar = 0.099 (n=520)
  * P in [0.9, 1): median G_k/Dbar = 0.073 (n=400)
* non-holder norm loss per local step, rounds 0-4: mean 1.49e-02

## GeFL-F wd=1e-4  (mnist, seed 0, K=10, 40 rounds, IF=0.01, alpha=0.5)

holders m_c per class: 0:10 1:10 2:10 3:10 4:10 5:9 6:9 7:8 8:7 9:8
total samples n_c:     0:2417 1:1449 2:868 3:520 4:312 5:187 6:112 7:67 8:40 9:24
row norm end / init: head 1.415, tail 0.223; tail/head at end 0.155
per class end / init:  0:1.42 1:1.44 2:1.39 3:1.41 4:1.33 5:0.75 6:0.66 7:0.10 8:0.04 9:0.09

* theorem violations (||r+|| > bound): **0 / 400** row-rounds
* certified contractions (bound < ||r||): 140; actual contractions: 196; certified but did not shrink: 0
* alignment slack ||r+|| / bound: mean 0.9827, min 0.8283
* row-rounds with m_c >= K/2 (old theorem: no contraction): 400; of these shrank: 196; certified by the new theorem: 140

| class | m_c | m_c/K | mean holder class-in-batch freq | Gbar/Dbar (median) | theta (median) | rounds certified to contract | rounds shrank |
|---|---|---|---|---|---|---|---|
| 0 | 10 | 1.0 | 0.994 | nan | 0.000 | 0/40 | 0/40 |
| 1 | 10 | 1.0 | 0.972 | nan | 0.000 | 0/40 | 0/40 |
| 2 | 10 | 1.0 | 0.883 | nan | 0.000 | 0/40 | 0/40 |
| 3 | 10 | 1.0 | 0.776 | nan | 0.000 | 1/40 | 1/40 |
| 4 | 10 | 1.0 | 0.711 | nan | 0.000 | 1/40 | 1/40 |
| 5 | 9 | 0.9 | 0.750 | 0.094 | 0.914 | 38/40 | 40/40 |
| 6 | 9 | 0.9 | 0.640 | 0.076 | 0.930 | 39/40 | 40/40 |
| 7 | 8 | 0.8 | 0.590 | 0.205 | 0.830 | 23/40 | 39/40 |
| 8 | 7 | 0.7 | 0.578 | 0.638 | 0.611 | 14/40 | 35/40 |
| 9 | 8 | 0.8 | 0.326 | 0.203 | 0.831 | 24/40 | 40/40 |

* holder gain G_k / Dbar vs P(class in batch): Spearman rho = -0.179 over 1640 holder-rounds
  * P in [0, 0.1): median G_k/Dbar = 0.251 (n=80)
  * P in [0.1, 0.5): median G_k/Dbar = 0.149 (n=640)
  * P in [0.5, 0.9): median G_k/Dbar = 0.105 (n=520)
  * P in [0.9, 1): median G_k/Dbar = 0.072 (n=400)
* non-holder norm loss per local step, rounds 0-4: mean 1.49e-02

## HWA wd=1e-3  (mnist, seed 0, K=10, 40 rounds, IF=0.01, alpha=0.5)

holders m_c per class: 0:10 1:10 2:10 3:10 4:10 5:9 6:9 7:8 8:7 9:8
total samples n_c:     0:2417 1:1449 2:868 3:520 4:312 5:187 6:112 7:67 8:40 9:24
row norm end / init: head 1.752, tail 1.280; tail/head at end 0.716
per class end / init:  0:1.72 1:2.03 2:1.51 3:1.50 4:1.46 5:1.55 6:1.51 7:1.25 8:1.19 9:1.17

