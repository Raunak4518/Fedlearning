"""
utils/checkpoint.py

Round-level checkpointing so a long run can be resumed after interruption
(pre-emption, crash, manual stop) without retraining from round 0.
Everything needed to resume exactly is saved: target-net weights per
architecture group, the generator's weights, every client's label-sampler
state (Mechanism B fidelity must persist across rounds), and the round
counter + RNG state.
"""
import os
import pickle
from typing import Optional

import torch


def ckpt_path(args, tag: str = "latest") -> str:
    return os.path.join(args.ckpt_dir, f"{args.name}_{tag}.pt")


def save_checkpoint(path: str, round_idx: int, ws_glob, gen_state: Optional[dict],
                     gen_opt_state, label_sampler_states: dict, args) -> None:
    torch.save({
        "round_idx": round_idx,
        "ws_glob": ws_glob,
        "gen_state": gen_state,
        "gen_opt_state": gen_opt_state,
        "label_sampler_states": label_sampler_states,
        "torch_rng_state": torch.get_rng_state(),
        "args": vars(args),
    }, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def save_results(path: str, results: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(results, f)
