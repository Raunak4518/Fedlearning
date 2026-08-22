"""
utils/logger.py

Console + CSV logging by default (always works, no network / account
needed); Weights & Biases is opt-in via --wandb 1 and imported lazily so
its absence never breaks a run. Matches the original repo's convention of
an optional `run = wandb.init(...)` object threaded through training,
except here `ExperimentLogger` is always a valid object -- `args.wandb=0`
just makes its wandb calls no-ops -- so the training loop never has to
check `if run is not None`.
"""
import csv
import logging
import os
import sys
import time


def get_console_logger(name: str, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    return logger


class ExperimentLogger:
    def __init__(self, args):
        self.args = args
        self.console = get_console_logger(args.name, bool(args.verbose))
        os.makedirs(args.out_dir, exist_ok=True)
        self.csv_path = os.path.join(args.out_dir, f"{args.name}_{int(time.time())}.csv")
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = None

        self.wandb_run = None
        if getattr(args, "wandb", 0):
            try:
                import wandb
                self.wandb_run = wandb.init(project=args.wandb_proj_name, name=args.name, config=vars(args))
            except Exception as e:  # noqa: BLE001
                self.console.warning("W&B logging requested but unavailable (%s); continuing without it.", e)

    def log(self, metrics: dict, step: int = None) -> None:
        row = dict(metrics)
        if step is not None:
            row = {"step": step, **row}
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        if self.wandb_run is not None:
            self.wandb_run.log(row)

        msg = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items())
        self.console.info(msg)

    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
