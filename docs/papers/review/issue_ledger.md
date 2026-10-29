# Issue ledger

Statuses: OPEN, IN_PROGRESS, RESOLVED_UNVERIFIED, RESOLVED_VERIFIED, REJECTED_WITH_RATIONALE, BLOCKED,
AUTHOR_DECISION_REQUIRED. Severity P0 (validity/integrity) to P3 (editorial).

| ID | Cycle first/last | Severity | Location | Issue | Decision | Status | Verification |
|---|---|---|---|---|---|---|---|
| C-1 | 0/1 | P0 | abstract, sec1, sec7 | exact readout "for every method": only GeFL-F pipeline | accept | RESOLVED_VERIFIED (C cycle 1: PARTIAL, caption fixed in v03) | C_review |
| C-2 | 0/1 | P0 | sec6 | worst-class recall definition | accept | RESOLVED_VERIFIED | C_review PASS |
| C-3 | 0/1 | P0 | sec6 | hyperparameters "of each paper" | accept | RESOLVED_VERIFIED | C_review PASS |
| C-4 | 0/1 | P0 | sec6, appendix | reproduction vs "original implementation" | accept | RESOLVED_UNVERIFIED (appendix fixed in v03) | C_review PARTIAL |
| N-1 | 1 | P0 | sec6, sec8 | development used seeds 3-5 and the same splits and test sets (text said seeds 0-2, new splits) | accept | RESOLVED_UNVERIFIED (v03) | to verify |
| C-5..C-8 | 0/1 | P1 | sec4-6, abstract | no generator for BSM/FedLC/FedRS; T_s of GeFL-F; readout as evaluation; FedRS step size | accept | C-5/6/8 VERIFIED; C-7 RESOLVED_UNVERIFIED (abstract v03) | C_review |
| N-2 | 1 | P1 | sec6 | test sets not balanced on MNIST/SVHN | accept | RESOLVED_UNVERIFIED (v03) | to verify |
| D-21 | 1 | P1 | sec6 | "no method was tuned" vs development disclosure | accept | RESOLVED_UNVERIFIED (v03) | to verify |
| D-1, D-2 | 0 | P1 | tables | MNIST/FMNIST/SVHN, GTSRB, Track-2 seeds of FedECS-LT not returned | needs runs | BLOCKED (compute) | - |
| D-3..D-5, D-12, D-16, D-19, D-20 | 0/1 | P1-P3 | sec7, sec8, abstract | tail, published numbers, shared-model scope, natural caveats, 10-NN, readout, own mix | accept | RESOLVED_VERIFIED | D_review PASS |
| D-6, D-26 | 0/1 | P1 | sec6 threats | compute not matched; T_s=1 ablation does not answer it | accept wording; matched-compute arms need runs | PARTIAL: disclosed; experiment BLOCKED | - |
| D-7 | 0 | P1 | sec6 threats | HtFLlib ports peak at round 1 | disclose; sanity run needs compute | PARTIAL: disclosed; BLOCKED | - |
| D-8 | 0 | P1 | sec5/6 | gate near threshold on CIFAR-10; other branch | NB31 provides it | BLOCKED (NB31) | - |
| D-9, D-24 | 0/1 | P2 | sec6 stats | power, permutation floor, family, noise floor from the repeat | accept | RESOLVED_UNVERIFIED (v03, slots repeat/*/floor) | to verify |
| D-10, D-13, D-14, D-17 | 0 | P2-P3 | sec6, sec1, fill.py | non-determinism, seeds sentence, fill groups, realised ratio | accept | RESOLVED (D-13/14 PASS) | D_review |
| D-11, C-14 | 0/1 | P2 | tab:cost | time measured with a shared GPU | accept | RESOLVED_UNVERIFIED (v03 caption) | to verify |
| D-15 | 0 | P2 | appendix | GeFL-setting "FedECS-LT" is an older configuration (K16 Ours gated) | open | OPEN | - |
| N-3..N-13, D-22, D-23, D-25, D-27 | 1 | P2-P3 | sec4-6, appendix | Holm counts, grouped at K=10, tab:hyper, SecAgg bound, redraw, FedDF sentence, appendix widths/FedDF, fairness rows, determinism, FedRS counts, comm cause, RQ table, threats | accept | RESOLVED_UNVERIFIED (v03) | to verify |
| C-9, C-11, C-16, C-18, C-20, C-26, C-27, C-29..C-34, C-36 | 0 | P2-P3 | sec4, appendix | theory wording (Adam bound, alternation conditions), notation G/tau, misc | accept | OPEN (next revision) | - |
