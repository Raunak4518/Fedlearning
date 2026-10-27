# fedecs_lt: the Flower re-run of every experiment

FedECS-LT (Exact Class Statistics for Long-Tailed federated learning) and every comparison, rewritten for Flower
(1.42, simulation with the Ray backend). Decisions: `docs/review/action_plan.md` sections 16-22.

## Layout
- `src/`: the code. Every Kaggle notebook pastes all of it (`%%writefile` cells), so each notebook is standalone.
  - `settings.py` datasets, rounds, our method's settings, secure-aggregation settings, citations
  - `data.py` downloads, preprocessing, long tail, client split (identical to `experiments/core.py`)
  - `models.py` extractor, the ten head architectures, CVAE-F, PC-VAE, Gaussian generator, ResNet-8 (Track 2)
  - `fedecs.py` our method's components (statistics release, gate, CCWA, moment correction, consensus labels, exact
    readout); `local_train.py` each client's local update
  - `server.py`, `client.py` Flower ServerApp / ClientApp of the GeFL-F pipeline family; `secure_sum.py` SecAgg+
  - `htfl_server.py`, `htfl_client.py` Track-1 baselines ported from HtFLlib (Apache-2.0)
  - `track2.py` Track-2 baselines: the authors' LongTailLib code, downloaded at run time, on our splits
  - `metrics.py`, `monitor.py`, `plot_style.py` (IEEE), `plots.py` (70-figure catalogue), `captions.py`,
    `report.py` (report.pdf), `pack.py` (the three output files), `runner.py` (workers, resume, deadline),
    `acceptance.py` (NB00 check against the earlier results), `baselines.py` (registry)
- `tests/`: `python -m pytest fedecs_lt/tests -q` (partitions, networks, formulas and every client training step
  against `experiments/core.py`; metrics against scikit-learn); `run_notebook_on_cpu.py` runs a notebook end to end on
  CPU from the notebook file itself.
- `build_notebooks.py` writes `notebooks/<priority folder>/*.ipynb` (folders `0_` to `4_`, in run order) and
  `notebooks/START_HERE.md`. Notebooks are generated; edit `src/`
  or the builder, then rebuild.
- `merge_results.py` joins several notebooks' results (parts, seeds, ablation + reference) into one report;
  `paper_tables.py` writes the paper's numbers as LaTeX / CSV. How to use them for the paper: `PAPER_GUIDE.md`.
- `results/<notebook>/`: returned results (`<name>__results.zip` unpacked; the report PDF next to it).

## Every notebook returns
`NAME__results.zip` (every figure as PDF + PNG + CSV, `runs.jsonl`, `runs.csv`, per-round logs, TensorBoard,
predictions), `NAME__report.pdf` (every table and figure with captions; its cover shows the logging audit),
`NAME__models.zip` (weights) and `NAME__summary.zip` (a few MB: every metric and log as text, the figures' CSV data;
the file to send back when the others are too big to upload).
