# decision log

## Cycle 0-1 (29 Oct 2026)
- Agents B (literature), C (technical vs code), D (experiments, statistics) ran independently on v00; C and D
  re-reviewed v02. Every finding of C and D was checked against the code or LEDGER.md before it was applied; none was
  rejected so far. N-1 (C) overruled my own v01 text (seeds 0-2): LEDGER.md shows development on seeds 3-5 with the
  same splits; the paper now states that.
- Issues needing new runs (more seeds, matched-compute arms, port sanity check, F0 sweep, missing benchmarks) are
  BLOCKED on compute; their claims are narrowed in the text, not fabricated.
- The readout is an evaluation device, not a component: headline numbers stay "as trained" (C-7).

## Cycle 1, agents E and F (29 Oct 2026), v05
- **Reversal of C-7 by the user.** "If readout is improving result, why aren't u adding it in our main architecture":
  the exact readout is now the last step of FedECS-LT. This answers E's R2/E-3 with option (a): FedECS-LT is defined
  with the readout, the headline margins are over the published methods, and the same-readout comparison for every
  GeFL-F-pipeline method (+1.4 / +1.5 over the strongest, not significant) is reported next to it in the abstract.
  The heads alone stay as a row, so nothing is hidden. With one shared model the readout lowers FedECS-LT; reported.
- **Runs (user: new Flower notebooks only).** E's R1, R3, R4 and E-8 ask for runs: component ablation, AdamW-only
  GeFL-F, a Gaussian sampler with FedECS-LT heads, fresh seeds 6-8, GeFL-F and FedDF with T_s = 10. Built as four new
  notebooks (folder 8); their paper text is conditional. Not built: F0 sensitivity beyond the CCWA-rows arm (the
  component notebook runs both branches), a stronger extractor (E-20; a limitation), K = 50 (folder 7 already
  exists).
- **E-6 accepted:** the logged referee fidelity is reported (appendix table). It shows that GeFL-F's generated
  features are near chance for every class, so Section 4.1 now calls the row shrinkage a measured property, not a
  demonstrated cause, and highlight 1 says "shrinks" instead of "erases".
- **E-15 accepted:** propositions renamed lemmas; contribution 1 is "a measured diagnosis supported by elementary
  lemmas".
- **E-24 (front matter, repository URL):** waiting for the user ("I'll give names + URL").
- **F's editorial findings** are applied by a separate editor agent on the v05 text (report in
  reviews/cycle1/F_applied.md once it returns).

## Cycle 2 (29 Oct 2026), v06
- Verifier I and reviewer G read v05 independently. Every I PARTIAL/FAIL item and every G item that text can fix
  were applied (final_review.md). G's experiment requests map to the folder-8 notebooks and to folder 7 (K = 50); no
  new notebook was added.
- Not done, with reasons: inversion test / DP release (separate study), stronger extractor (outside GeFL-F's
  design), symbol renames (names follow the cited methods; each is defined where used).
- The labels of the matched-epoch arms became "GeFL-F (10 synthetic epochs)" and "FedDF (10 synthetic epochs)":
  the underscore of "T_s" failed the report's caption check in the CPU test. No run had used the old labels.

## Results of 17 notebooks (30 Oct 2026), v07
- **Returned:** MNIST, Fashion-MNIST, SVHN (main comparison); fresh seeds 6-8; GeFL-F variants (27 of 36 runs);
  matched synthetic epochs; components (24 of 36); pooled-data reference; GeFL's grouped baselines at 10 clients
  (CIFAR-10-LT); the HtFLlib and grouped methods at 50 clients (CIFAR-10-LT complete, CIFAR-100-LT 29 of 57). Every
  failure was host memory (Ray out of memory, or a secure sum failing when a client was killed): all these notebooks
  ran code 342992a, before the one-run-per-GPU change for 50 clients (15e0366). FedECS-LT, GeFL-F, FedDF and
  Balanced Softmax at 50 clients (NB40_sweep_cifar10_K50 and CIFAR-100 part 1) did not finish, so the paper shows no
  50-client table yet (fill.py writes it only when FedECS-LT has runs in every setting).
- **Gaussian sampler kept out of the paper (user):** "i am not in mood to make any changes so please don't keep it in
  paper, you give me a different pdf about it". FedECS-LT with its trained generator replaced by class Gaussians from
  exact statistics is +1.7 points with the readout on CIFAR-10-LT (Holm p = 0.041 over the seven changes) at 16 times
  less communication. fill.py excludes it from the components table, figure and counts (NOT_IN_PAPER) and writes its
  numbers under note/...; the finding is in docs/study/gaussian_sampler.pdf. Reviewer E asked for this arm (R1): a
  reviewer may ask for it again.
- **Text changed with the results:** the decay diagnosis now cites the AdamW arm (+1.4 points, a small part of the
  loss); the components paragraph names the consensus soft labels as the only resolved component and the moment
  correction as showing no benefit; MNIST is the one benchmark where the same-readout lead is significant
  (contribution 3 corrected); AvgKD collapses to one class (reported with a plausible reason, untested); the pooled
  reference and the grouped baselines enter tab:main.
