# Notes for Claude

## Compute is limited: never propose a run that already exists

- `experiments/LEDGER.md` lists every completed run (method, setting, seeds) and every Kaggle
  notebook whose results never came back. Regenerate it with `python experiments/ledger.py`
  after adding results.
- Before writing any new spec or notebook, run
  `python experiments/ledger.py --check <built script> [--datasets ...]` and remove every
  DUPLICATE. Reuse existing seeds (same splits and seeds = valid pairing) instead of re-running
  a reference arm in the new job.
- Before suggesting a notebook from the "no results" list, ask whether it was already run and
  only the zip is missing.
- Results from Kaggle go in `experiments/results/<folder>/` (`summary.md` + `runs.jsonl`).
