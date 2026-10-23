"""Training monitor: every round's signals as JSON lines (one file per run), the bytes of every message type, time
per stage and GPU memory; TensorBoard scalars when the tensorboard package is available."""
import json
import os
import time

import numpy as np
import torch


def _clean(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if torch.is_tensor(v):
        return v.detach().cpu().tolist()
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


class Monitor:
    def __init__(self, log_dir, run_id, tensorboard=True):
        os.makedirs(log_dir, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_.=" else "_" for ch in run_id)
        self.path = os.path.join(log_dir, safe + ".jsonl")
        self.f = open(self.path, "a")
        self.bytes = {}          # message type -> [bytes server->client, bytes client->server, messages]
        self.stage_t = {}
        self._t = {}
        self.tb = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(os.path.join(log_dir, "tb", safe))
            except Exception:
                self.tb = None

    def log(self, event, **fields):
        rec = dict(t=round(time.time(), 3), event=event, **_clean(fields))
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        if self.tb is not None and "round" in fields:
            step = int(fields["round"])
            for k, v in fields.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "round":
                    self.tb.add_scalar(f"{event}/{k}", v, step)

    def count(self, mtype, down, up, n=1):
        b = self.bytes.setdefault(mtype, [0, 0, 0])
        b[0] += down
        b[1] += up
        b[2] += n

    def start(self, stage):
        self._t[stage] = time.time()

    def stop(self, stage):
        self.stage_t[stage] = self.stage_t.get(stage, 0.0) + time.time() - self._t.pop(stage)

    def close(self):
        self.f.close()
        if self.tb is not None:
            self.tb.close()
