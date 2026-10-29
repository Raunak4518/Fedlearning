# Final review report (cycle 2, version v06)

**Verdict of the fresh reviewer G (on v05): major revision.** Scores 2 / 3 / 2 / 3 / 2 (originality, soundness,
rigour, clarity, significance). Full report: reviews/cycle2/G_final_review.md. Verification of every accepted earlier
item: reviews/cycle2/I_verification.md.

## What the evidence supports now
- With ten architectures (one client each), FedECS-LT with its exact readout is 15.1 / 8.6 points above the strongest
  other method without the readout on CIFAR-10-LT / CIFAR-100-LT (Holm p 0.0052 / 0.0042).
- Most of that margin is the readout's. With the same readout for every GeFL-F-pipeline method, the lead over the
  strongest is +1.4 / +1.5 (not significant); the lead over GeFL-F is +2.3 / +3.8 (significant on CIFAR-100-LT only).
- On the six naturally imbalanced datasets (never used in development), the same-readout margins are -0.03 to +0.9.
- With one shared model, six methods are better on CIFAR-10-LT, among them FedAvg.

## Fixed in v06 (text)
Every G item that text can fix: causal wording (abstract, intro, conclusion, highlights), "without the readout"
instead of "as published", trade-offs in the abstract (communication) and conclusion (own-mix accuracy, cost, client
requirement), signed margins (BloodMNIST -0.03), conditional grouped/pooled/ratio-50-10 text, FedDF spread caveat,
moment-correction fidelity, privacy note, realised imbalance ratio, symbol definitions (LSE, d_e, B), noise-floor rule,
readout tables in the appendix, p-value precision, fonts, citations. All of I's PARTIAL and FAIL items.

## Open: needs runs (notebooks built, not run)
- G-M1 / E-R1 components → `cifar10_cifar100_components_of_fedecs_lt` (2 parts).
- G-M3 / E-R4 tuning → `cifar10_cifar100_fresh_seeds_6_to_8`.
- E-R3 decay cause, Gaussian sampler → `cifar10_cifar100_gefl_f_variants`.
- E-8 compute → `cifar10_cifar100_matched_synthetic_epochs`.
- G-M2 several clients per architecture → folder 7 (`*_50clients_all_methods`).

The paper text for each is written with conditional slots and appears when the results return.

## Not done, with reasons
- An inversion test or a DP variant of the statistics release (G-M7): a separate study. The privacy limit is stated
  in Sec. 5 and 8.
- A stronger or pre-trained extractor (E-20): outside GeFL-F's design, stated as a limitation.
- Renaming overloaded symbols (G, ρ, α, ω, u): kept, because the names follow GeFL, FedRS and the Dirichlet
  convention. Each is defined where it is used.
- The author names, email and repository URL: waiting for the user.

## Checks on v06
- Builds: draft and final, Elsevier and IEEE, all pass with 0 undefined references and 0 Type 3 fonts.
- 0 em dashes; abstract 249 words.
- Originality: 0 passages of 8 or more words shared with 42 sources.
- Claims: 35 in all; 27 Supported, 1 Partially, 7 Unresolved (all hidden in the final build until their runs return).
