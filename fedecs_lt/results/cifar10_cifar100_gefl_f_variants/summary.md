# cifar10_cifar100_gefl_f_variants: INCOMPLETE

Kaggle: https://www.kaggle.com/code/deepakyadavcec/notebook20306486e6  
Downloaded: 2026-10-30  
Code: 342992a670fc / sources sha256 f246d5574d4f184c

Runs finished: 27 of 36 planned. Runs that failed on every attempt: 9.

Report checks: failures 9, figure_errors 0, table_errors 0, layout_problems 0, caption_problems 0, audit_problems 0

Failed runs by cause:

- Ray out of memory (host RAM): 8
- RuntimeError('secure sum of consensus failed (a client raised an error: see the log above)'): 1

Failed runs:

- FedDF + LA|cifar100|IF0.01|a0.5|K10|s4
- FedDF + LA|cifar100|IF0.01|a0.5|K10|s5
- FedECS-LT w/ Gaussian sampler|cifar100|IF0.01|a0.5|K10|s5
- GeFL-F + AdamW|cifar100|IF0.01|a0.5|K10|s3
- GeFL-F + AdamW|cifar100|IF0.01|a0.5|K10|s4
- GeFL-F + AdamW|cifar100|IF0.01|a0.5|K10|s5
- GeFL-F + LA|cifar100|IF0.01|a0.5|K10|s5
- GeFL-F w/ Gaussian generator + LA|cifar100|IF0.01|a0.5|K10|s5
- GeFL-F w/o generator|cifar100|IF0.01|a0.5|K10|s5
