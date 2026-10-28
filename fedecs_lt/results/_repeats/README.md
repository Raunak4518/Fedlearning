# Repeated notebooks (not used by the paper)

A notebook that was run a second time, on another Kaggle account, with the same run list (same client splits and
seeds). The first returned run stays in `results/<notebook>/` and is the one the paper uses; the repeat is kept here
as a measure of run-to-run variation (GPU non-determinism and code versions differing only in schedule and memory).
`fill.py`, `gefl_measure.py` and the notebook builder skip folders whose name starts with `_`.

| Repeat | Kaggle | First run | Same splits | Largest change of a run's balanced accuracy |
|---|---|---|---|---|
| NB10_main_bloodmnist__wizard345 | https://www.kaggle.com/code/wizard345/bloodmnist | https://www.kaggle.com/code/raunak45/main-comparison-on-bloodmnist | yes | 3.6 points (FedDF); method means change by at most 2.5 points (FedDF), FedECS-LT +0.6 |
