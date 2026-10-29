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
| B-G1 | 0/1 | P1 | sec6 | GeFL ran FedDF on public data, not generated features | accept | RESOLVED_UNVERIFIED (v04) | - |
| B-G2, G3, G4, G5, G6, G8 | 0/1 | P2-P3 | sec6, appendix, bib | AvgKD is cross-architecture; long tail is our construction; GeFL ref details; GeFL-F number from Fig. 4; GeFL's measure; reproduction wording | accept | RESOLVED_UNVERIFIED (v04) | - |
| B-N1, C14 | 0 | P1 | sec1, sec2, sec4 | FedCOF (NeurIPS 2025) LDA-type federated classifier not cited | accept: cited, readout positioned as same family | RESOLVED_UNVERIFIED (v04) | - |
| B-N2..N7, C4..C12, C15..C17 | 0 | P2-P3 | sec2, sec4, sec5, sec6, sec7 | AFL, DFL-FS, Fisher, FedGen count weights, AvgKD/FedDF target form; invariance credited; CReFF settings and forced ResNet-8; HtFLlib, FedFTG, FedVTC, FedRS, FedLTLib, collapse, CAS wording; positioning table; "than ours" | accept | RESOLVED_UNVERIFIED (v04) | - |
| C-9, C-11, C-20, C-26, C-27, C-29, C-30, C-32, C-36, D-15 | 0 | P2-P3 | sec1, sec3-5, appendix | Adam bound, alternation conditions, Flower scope, contraction wording, holder gradient, clipping and <3 clients, prototype scores, sample size, spread definition, GeFL-setting configuration | accept | RESOLVED_UNVERIFIED (v04) | - |
| C-16 | 0 | P2 | appendix | HtFLlib port and FedDF deviations | accept | RESOLVED_UNVERIFIED (v04: appendix paragraphs) | - |
| E-1 | 1 | P0 | build | draft PDF prints conditional statements | build.sh final (\finalslotstrue via -jobname main_final) | RESOLVED_UNVERIFIED (final PDF: no slot names) | verifier |
| E-2, E-3, E-4 | 1 | P1 | abstract, sec5, sec7 | readout carries the gain; method vs analysis; statistics framing | readout is part of FedECS-LT (user); abstract lists which parts use statistics | RESOLVED_UNVERIFIED (v05) | verifier |
| E-5, E-6 | 1 | P1 | sec4, highlights | decay diagnosis not causal; fidelity unreported | fidelity table, softened text; AdamW-only arm built | text RESOLVED_UNVERIFIED; run BLOCKED (folder 8) | verifier |
| E-7, E-26 | 1 | P1 | sec6 | selection on the test sets | development count and untouched datasets stated; fresh seeds 6-8 built | text RESOLVED_UNVERIFIED; run BLOCKED (folder 8) | verifier |
| E-8, E-25 | 1 | P1 | sec7 | compute/distillation confound; gate branch | T_s = 10 arms, component notebook built | BLOCKED (folder 8) | - |
| E-9, E-10 | 1 | P1 | sec3, sec5 | every client runs every head; K = 10 one client per architecture | assumption paragraph in sec3; privacy sentence corrected | RESOLVED_UNVERIFIED | verifier |
| E-11..E-17, E-19, E-21..E-23 | 1 | P1-P3 | highlights, abstract, sec1, sec3, sec4, sec7, appendix, fill.py | highlights, one-seed note, conclusion, post-hoc LA, lemmas, "only", FedDF in GeFL measure, counts, ECE rank, spread caveat, number format | accept | RESOLVED_UNVERIFIED | verifier |
| E-18, E-20 | 1 | P2 | sec8 | weak baselines; small extractor | limitations | RESOLVED_UNVERIFIED (text); experiment not planned | - |
| E-24 | 1 | P3 | front matter | placeholders | user gives names and URL | AUTHOR_DECISION_REQUIRED | - |
| F-* | 1 | P1-P3 | all | editorial review | applied by the editor agent (skips listed) | RESOLVED_UNVERIFIED | F_applied.md |
| I (all PARTIAL, D-5 FAIL) | 2 | P1 | abstract, intro, conclusion, highlights, sec6, sec7 | shared-model wording, highlight 4, T_s in abstract, causal wording, grouped text without results, ratios 50/10 | accept | RESOLVED_UNVERIFIED (v06) | final checks |
| G-M1, G-M2, G-M3 | 2 | P1 | method, setting | components, K = G = 10, tuning on reported splits | folder 8 and folder 7 notebooks | BLOCKED (runs) | - |
| G-M4..G-M6 | 2 | P1 | intro, sec4, sec6, conclusion, abstract | causal claims, FedDF spread, MC fidelity, promised content, trade-offs | accept | RESOLVED_UNVERIFIED (v06) | final checks |
| G-M7 | 2 | P2 | sec5 | privacy of class means | note added; no inversion experiment | REJECTED_WITH_RATIONALE (separate study) | - |
| G-minor | 2 | P2-P3 | all | terms, symbols, fonts, citations, captions | accept, except symbol renames | RESOLVED_UNVERIFIED | final_review.md |

