import os
import csv
import time
from typing import Dict, Any

class RunTracker:
    def __init__(self, index_path: str = "logs/runs/index.csv"):
        self.index_path = index_path
        self.fieldnames = [
            "run_id", "name", "status", "start_time", "end_time", 
            "git_commit", "git_dirty", "gen_model", "imbalance_factor"
        ]
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        if not os.path.exists(self.index_path):
            with open(self.index_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log_start(self, run_id: str, args: Any, git_commit: str, git_dirty: bool):
        row = {
            "run_id": run_id,
            "name": getattr(args, "name", "unknown"),
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": "",
            "git_commit": git_commit,
            "git_dirty": str(git_dirty),
            "gen_model": getattr(args, "gen_model", "none"),
            "imbalance_factor": getattr(args, "imbalance_factor", "")
        }
        with open(self.index_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)

    def log_finish(self, run_id: str, status: str = "completed"):
        # We need to update the row where run_id matches.
        # Since CSV is append-mostly, we can rewrite the whole file for small indexes,
        # or just append a new status row. For simplicity and reliability, we'll rewrite it.
        rows = []
        with open(self.index_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.fieldnames = reader.fieldnames
            for row in reader:
                if row["run_id"] == run_id:
                    row["status"] = status
                    row["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                rows.append(row)
                
        with open(self.index_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)
