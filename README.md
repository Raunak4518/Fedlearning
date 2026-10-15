# GeFL-F under long-tailed clients: exact-statistics anchoring (research code)

This repository studies **GeFL-F** (generator-based model-heterogeneous federated learning; Kang et al., IEEE TMC 2025)
when clients hold long-tailed, label-skewed data. Clients share a frozen feature extractor and a class-conditional
feature generator, and each trains its own head architecture. We diagnose why rare classes are lost and test repairs that
take class information from exactly aggregated statistics (class sums under secure aggregation) instead of from
FedAvg-trained weights.

**Status (15 Oct 2026):** under revision after an adversarial review. The reviews, and the plan that follows from them,
are the starting point for any new work:
- `docs/review/action_plan.md`: what to do next, in order, including the two experiments that decide the paper's framing;
- `docs/review/adversarial_review_dossier.md`: every surviving objection, with evidence.

## Where things are

| Path | What it is |
|---|---|
| `docs/paper_draft.md` | The manuscript (single source of the paper's text) |
| `docs/PROGRESS.md` | Each claim of the paper → its evidence → status, and the list of jobs still to run |
| `docs/related_work.md` | Literature notes |
| `docs/research/readout_report.md` | The readout-gap analysis and exact readout (T03) |
| `docs/paper_assets/` | Generated tables, figures and every number with its source (`numbers.csv`), and `CHECK.md` (draft numbers that disagree with the data) |
| `docs/review/` | The adversarial review: dossier, ten reviewer reports, recomputation scripts, the GPT review package and its comparison, and the action plan |
| `docs/professor_report/` | Professor handout (Word + PDF) and the scripts that build its figures |
| `docs/collapse_explainer/` | Plain-language PDF on the collapse result |
| `experiments/core.py` | **The whole pipeline**: data (MNIST, FashionMNIST, SVHN, CIFAR-10, CIFAR-100), partitions, feature extractor, generators, heads, every method component, metrics and post-hoc readouts |
| `experiments/specs/` | One file per experiment (arms, datasets, seeds) |
| `experiments/build.py` | Builds a standalone script `experiments/<name>.py` = `core.py` + `specs/<name>.py` (built scripts are not committed) |
| `experiments/results/<job>/` | Raw results: `runs.jsonl` (one record per run and seed) and `summary.md` |
| `models/<job>/` (not in git) | Trained models, kept apart from the results: `fe/`, `gen/`, `heads/`, `index.jsonl`. Kaggle notebooks zip them as `<folder>__models.zip`; re-evaluate them without training with `--from_models models/<job>` |
| `experiments/LEDGER.md`, `experiments/ledger.py` | Every completed run and every planned job's status; `--check` finds duplicates before anything new is run |
| `experiments/paper_assets.py` | Regenerates `docs/paper_assets/` from the raw results |
| `experiments/make_kaggle_notebooks.py` | Writes Kaggle notebooks for named jobs |
| `experiments/T01_*`, `T02_*`, `T03_*` | Small CPU mechanism tests (collapse bound, dilution rates, exact readout) |
| `experiments/D01_component_evidence.py` | Per-component mechanism measurements |
| `experiments/reference_check/` | Parity check of our GeFL-F against the authors' released code |
| `kaggle_notebooks/` | Notebooks that are ready but **not yet run** (K14: NB36–37; K15: NB38–39; K16: NB40–48) |

## Common commands

```bash
pip install -r requirements.txt

python experiments/build.py                                   # build all standalone experiment scripts
python experiments/build.py && python experiments/F01_main_longtail.py --list    # show a job's planned runs
python experiments/ledger.py                                  # refresh LEDGER.md after adding results
python experiments/ledger.py --check K15_kaggle_collapse_fixes --datasets mnist  # duplicate check for a job
python experiments/paper_assets.py                            # regenerate tables, figures, numbers.csv, CHECK.md
python experiments/make_kaggle_notebooks.py                   # list jobs; then name the ones to write, e.g. NB38 NB39
python experiments/K16_kaggle_recalibration.py --datasets mnist --from_models models/K16_kaggle_recalibration --out_dir experiments/results
                                                              # re-evaluate saved models (new readouts/metrics), no training
```

Results from Kaggle go into `experiments/results/<folder>/` (`summary.md` + `runs.jsonl`). After adding results, rerun
`ledger.py` and `paper_assets.py`.

## Removed in the 15 Oct 2026 clean-up

These are still in git history, but are no longer in the tree because they were superseded or would mislead:
- The first framework (root `GeFL_*.py`, `engine.py`, `args.py`, `configs/`, `utils/`, `gefl_f/`, `generators/`, `datasets/`, `sampling/`, `targetNetModels/`, `tests/`, `scripts/`, `baselines/`, `Baseline_CReFF.py`). No reported result used it; every experiment runs on `experiments/core.py`.
- The 59 generated experiment scripts: rebuild them with `build.py`.
- Old plans and pages: `docs/experiment_plan.html`, `improvement_derivations.html`, `technical_walkthrough.html`, `project_review.html`, `results.html`, `night_plan.md`, `paper_skeleton.md`, `diagnostics.md` (old K/2 bound), `kme_gen.md`, `archive/`.
- The second, shorter draft `docs/paper_single_idea.md`.
- Old reports: the v1 research report, the Hinglish notes (old theorem), deck and HTML samples.
- Queue and page scripts, and notebooks whose results have been ingested (NB32–NB35).
