# K25_instrumented: Instrumented runs: per-round diagnostics of the three stages (MNIST, CIFAR-10; seed 3)

Mode: full

Hypothesis: (a)-(c) in the docstring.

## mnist  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | er_bal | final_tail | p vs GeFL-F [instrumented] (final_bal) |
|---|---|---|---|---|---|
| Ours Ts=10 [instrumented] | 1 | 95.47 ± 0.00 | 94.21 ± 0.00 | 92.12 ± 0.00 | n<2 |
| GeFL-F [instrumented] | 1 | 74.43 ± 0.00 | 90.72 ± 0.00 | 55.00 ± 0.00 | - |
| GeFL-F + AdamW + LA [instrumented] | 1 | 87.65 ± 0.00 | 92.19 ± 0.00 | 81.01 ± 0.00 | n<2 |

| label | seed | final_bal | er_bal | final_tail |
|---|---|---|---|---|
| Ours Ts=10 [instrumented] | 3 | 95.47 | 94.21 | 92.12 |
| GeFL-F [instrumented] | 3 | 74.43 | 90.72 | 55.00 |
| GeFL-F + AdamW + LA [instrumented] | 3 | 87.65 | 92.19 | 81.01 |

## cifar10  IF=0.01  alpha=0.5  K=10

| label | n | final_bal | er_bal | final_tail | p vs GeFL-F [instrumented] (final_bal) |
|---|---|---|---|---|---|
| Ours Ts=1 [instrumented] | 1 | 43.94 ± 0.00 | 53.19 ± 0.00 | 34.85 ± 0.00 | n<2 |
| GeFL-F [instrumented] | 1 | 33.15 ± 0.00 | 50.35 ± 0.00 | 14.53 ± 0.00 | - |
| GeFL-F + AdamW + LA [instrumented] | 1 | 39.31 ± 0.00 | 51.49 ± 0.00 | 27.02 ± 0.00 | n<2 |

| label | seed | final_bal | er_bal | final_tail |
|---|---|---|---|---|
| Ours Ts=1 [instrumented] | 3 | 43.94 | 53.19 | 34.85 |
| GeFL-F [instrumented] | 3 | 33.15 | 50.35 | 14.53 |
| GeFL-F + AdamW + LA [instrumented] | 3 | 39.31 | 51.49 | 27.02 |

