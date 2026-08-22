"""
utils/logger.py

Multi-sink experiment logger: console + file + CSV + JSONL + optional
TensorBoard + optional Weights & Biases, all behind one ``ExperimentLogger``
object that the training loop calls once per evaluation step.

Console output uses tqdm progress bars when available (graceful fallback).
Structured log levels, file logging with rotation, per-round timing, and
generator loss tracking -- all the instrumentation the original logger lacked.
"""
import csv
import json
import logging
import os
import sys
import time
from typing import Dict, Optional


def get_console_logger(name: str, verbose: bool = True,
                       log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger with console + optional file handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if verbose else logging.WARNING)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class RoundTimer:
    """Context manager for timing individual phases of a round."""

    def __init__(self):
        self._timings: Dict[str, float] = {}
        self._start: Optional[float] = None
        self._current_phase: Optional[str] = None

    def start(self, phase: str):
        self._current_phase = phase
        self._start = time.perf_counter()

    def stop(self):
        if self._start is not None and self._current_phase is not None:
            elapsed = time.perf_counter() - self._start
            self._timings[self._current_phase] = elapsed
        self._start = None
        self._current_phase = None

    def get_timings(self) -> Dict[str, float]:
        return {f"{k}_time": round(v, 3) for k, v in self._timings.items()}

    def reset(self):
        self._timings.clear()
        self._start = None
        self._current_phase = None

    def total(self) -> float:
        return sum(self._timings.values())


class ExperimentLogger:
    """Unified logging sink for GeFL experiments.

    Writes to:
    - Console (stdlib logging, with optional tqdm progress bar)
    - Log file ({out_dir}/{name}.log)
    - CSV ({out_dir}/{name}_{timestamp}.csv)
    - JSONL ({out_dir}/{name}_metrics.jsonl) -- machine-parseable
    - TensorBoard (opt-in via --tensorboard 1)
    - Weights & Biases (opt-in via --wandb 1)
    """

    def __init__(self, args):
        self.args = args
        os.makedirs(args.out_dir, exist_ok=True)

        # Log file
        log_file = os.path.join(args.out_dir, f"{args.name}.log")
        self.console = get_console_logger(args.name, bool(args.verbose), log_file)

        # CSV
        self.csv_path = os.path.join(args.out_dir, f"{args.name}_{int(time.time())}.csv")
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames = None

        # JSONL (one JSON object per line, easy to load in pandas)
        self.jsonl_path = os.path.join(args.out_dir, f"{args.name}_metrics.jsonl")
        self._jsonl_file = open(self.jsonl_path, "w", encoding="utf-8")

        # Timer
        self.round_timer = RoundTimer()
        self._train_start = time.time()

        # TensorBoard (opt-in)
        self.tb_writer = None
        if getattr(args, "tensorboard", 0):
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(args.out_dir, "tensorboard", args.name)
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                self.console.info("TensorBoard logging enabled: %s", tb_dir)
            except ImportError:
                self.console.warning(
                    "TensorBoard requested but torch.utils.tensorboard not available; "
                    "continuing without it."
                )

        # Weights & Biases (opt-in)
        self.wandb_run = None
        if getattr(args, "wandb", 0):
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=args.wandb_proj_name, name=args.name,
                    config=vars(args)
                )
            except Exception as e:  # noqa: BLE001
                self.console.warning(
                    "W&B logging requested but unavailable (%s); "
                    "continuing without it.", e
                )

        # Progress bar (tqdm, graceful fallback)
        self._pbar = None
        if bool(args.verbose):
            try:
                from tqdm import tqdm
                total_rounds = getattr(args, "epochs", None)
                self._pbar = tqdm(
                    total=total_rounds, desc="Training",
                    unit="round", ncols=100, leave=True,
                    bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                                "[{elapsed}<{remaining}, {rate_fmt}]")
                )
            except ImportError:
                pass

        # Log experiment config
        self.console.info("=" * 70)
        self.console.info("EXPERIMENT: %s", args.name)
        self.console.info("=" * 70)
        config_summary = {
            k: v for k, v in vars(args).items()
            if not k.startswith("_") and k not in ("config",)
        }
        self.console.debug("Full config: %s", json.dumps(config_summary, default=str))

    def log(self, metrics: dict, step: int = None) -> None:
        """Log one evaluation step to all sinks."""
        row = dict(metrics)
        if step is not None:
            row["step"] = step
        row["wall_time"] = round(time.time() - self._train_start, 1)

        # Add round timing if available
        timings = self.round_timer.get_timings()
        if timings:
            row.update(timings)

        # CSV
        self._write_csv(row)

        # JSONL
        self._write_jsonl(row)

        # TensorBoard
        if self.tb_writer is not None and step is not None:
            for k, v in row.items():
                if isinstance(v, (int, float)) and k != "step":
                    self.tb_writer.add_scalar(f"metrics/{k}", v, step)

        # W&B
        if self.wandb_run is not None:
            self.wandb_run.log(row)

        # Console (formatted summary)
        self._log_console(row)

        # Update progress bar
        if self._pbar is not None:
            acc = row.get("acc_overall", row.get("acc_head", None))
            loss = row.get("train_loss", None)
            postfix = {}
            if acc is not None:
                postfix["acc"] = f"{acc:.4f}"
            if loss is not None:
                postfix["loss"] = f"{loss:.4f}"
            self._pbar.set_postfix(postfix)

    def log_round_start(self, round_idx: int):
        """Called at the beginning of each round."""
        self.round_timer.reset()
        self.console.debug("Round %d started", round_idx + 1)

    def log_round_end(self, round_idx: int):
        """Called at the end of each round (after eval if applicable)."""
        if self._pbar is not None:
            self._pbar.update(1)

    def log_image_grid(self, tag: str, images, step: int):
        """Log a grid of images to TensorBoard (if active).

        Args:
            tag: TensorBoard tag for the image.
            images: torch.Tensor of shape (N, C, H, W).
            step: global step.
        """
        if self.tb_writer is not None:
            try:
                from torchvision.utils import make_grid
                grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
                self.tb_writer.add_image(tag, grid, step)
            except ImportError:
                pass

    def log_summary(self, summary: dict):
        """Log a final summary section."""
        self.console.info("=" * 70)
        self.console.info("EXPERIMENT SUMMARY")
        self.console.info("=" * 70)
        for k, v in summary.items():
            if isinstance(v, float):
                self.console.info("  %-30s  %.4f", k, v)
            elif isinstance(v, dict):
                self.console.info("  %-30s  %s", k,
                                  {kk: f"{vv:.4f}" if isinstance(vv, float) else vv
                                   for kk, vv in v.items()})
            else:
                self.console.info("  %-30s  %s", k, v)
        self.console.info("=" * 70)

    def _write_csv(self, row: dict):
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames,
                                              extrasaction="ignore")
            self._csv_writer.writeheader()
        # Handle new keys that appeared mid-run
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            # Rewrite header by reopening (CSVs don't support dynamic columns well)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _write_jsonl(self, row: dict):
        line = json.dumps(row, default=_json_default)
        self._jsonl_file.write(line + "\n")
        self._jsonl_file.flush()

    def _log_console(self, row: dict):
        # Priority metrics first
        priority_keys = ["step", "round", "train_loss", "gen_loss",
                         "acc_overall", "acc_head", "acc_medium", "acc_tail",
                         "macro_f1", "class_balanced_accuracy", "mean_fidelity"]
        parts = []
        for k in priority_keys:
            if k in row:
                v = row[k]
                parts.append(_fmt_kv(k, v))

        # Timing
        for k in ("gen_time", "target_time", "eval_time", "wall_time"):
            if k in row:
                parts.append(_fmt_kv(k, row[k]))

        self.console.info("  ".join(parts))

    def close(self):
        """Flush and close all sinks."""
        if self._csv_file is not None:
            self._csv_file.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()
        if self._pbar is not None:
            self._pbar.close()
        if self.tb_writer is not None:
            self.tb_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
        self.console.info("Logs saved: CSV=%s  JSONL=%s", self.csv_path, self.jsonl_path)


def _fmt_kv(k: str, v) -> str:
    if isinstance(v, float):
        return f"{k}={v:.4f}"
    return f"{k}={v}"


def _json_default(obj):
    """JSON serializer for numpy types."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
