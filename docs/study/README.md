# Study text

`fedecs_lt_explained.tex` teaches the whole FedECS-LT project to a reader who did not do the research: the problem,
GeFL-F and where it fails, every component of FedECS-LT, the study design, every result and how to defend it. It is
written from the paper and uses the paper's number slots (`../papers/generated/numbers.tex`), so its numbers follow
the paper. `PROMPT.md` is the prompt it was written from. Build: `pdflatex fedecs_lt_explained.tex` (three passes).

`gaussian_sampler.tex` is a separate two-page note on one result kept out of the paper (user, 30 Oct 2026):
FedECS-LT with its trained generator replaced by class Gaussians from exact statistics. Build: `pdflatex
gaussian_sampler.tex` (two passes).
