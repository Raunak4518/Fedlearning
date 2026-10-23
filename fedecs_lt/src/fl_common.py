"""Helpers shared by the Flower server and client: message packing, seeding, cache keys, averaging."""
import hashlib
import json
import random

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict


def jdump(d):
    return json.dumps(d, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def key_of(*parts):
    return hashlib.sha1(jdump(parts).encode()).hexdigest()[:16]


def seed_all(*parts):
    """Deterministic seed for one client action (run seed, stage, round, client): results do not depend on the order
    in which the simulation engine executes clients."""
    s = int(key_of(*parts), 16) % (2 ** 31 - 1)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    return s


def pack(cfg=None, arrays=None, **named_arrays):
    """RecordDict with a JSON config and named ArrayRecords (dicts of tensors)."""
    rd = RecordDict()
    rd["cfg"] = ConfigRecord({"j": jdump(cfg or {})})
    for name, tensors in ({"w": arrays} if arrays is not None else {}).items():
        rd[name] = ArrayRecord({k: v.detach().cpu() for k, v in tensors.items()})
    for name, tensors in named_arrays.items():
        rd[name] = ArrayRecord({k: (v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v))
                                for k, v in tensors.items()})
    return rd


def cfg_of(content):
    return json.loads(content["cfg"]["j"])


def arrays_of(content, name="w", device="cpu"):
    if name not in content.array_records:
        return None
    return {k: v.to(device) for k, v in content[name].to_torch_state_dict().items()}


def reply(msg, cfg=None, arrays=None, **named):
    return Message(pack(cfg, arrays, **named), reply_to=msg)


def nbytes(content):
    return sum(len(a.data) for rec in content.array_records.values() for a in rec.values())


class Averager:
    """Unweighted mean of state dicts (as experiments/core.py: every participating client counts once); integer
    buffers (BN num_batches_tracked) take the last client's value."""

    def __init__(self):
        self.sum, self.n = None, 0

    def add(self, sd):
        if self.sum is None:
            self.sum = {k: (v.detach().clone().float() if v.is_floating_point() else v.clone()) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if v.is_floating_point():
                    self.sum[k] += v
                else:
                    self.sum[k] = v.clone()
        self.n += 1

    def result(self):
        return {k: (v / self.n if v.is_floating_point() else v) for k, v in self.sum.items()}
