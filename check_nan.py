import torch
import os

def check_checkpoint(ckpt_path):
    print(f"--- Checking {ckpt_path} ---")
    if not os.path.exists(ckpt_path):
        print(f"File {ckpt_path} not found.")
        return
    
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    found_problem = False

    if "gen_state" in checkpoint and checkpoint["gen_state"] is not None:
        gen_state = checkpoint["gen_state"]
        for k, v in gen_state.items():
            if torch.isnan(v).any():
                print(f"Generator NaN found in {k}")
                found_problem = True
            if torch.isinf(v).any():
                print(f"Generator Inf found in {k}")
                found_problem = True
    
    if "target_states" in checkpoint and checkpoint["target_states"] is not None:
        target_states = checkpoint["target_states"]
        for idx, t_state in enumerate(target_states):
            for k, v in t_state.items():
                if torch.isnan(v).any():
                    print(f"Target model {idx} NaN found in {k}")
                    found_problem = True
                if torch.isinf(v).any():
                    print(f"Target model {idx} Inf found in {k}")
                    found_problem = True

    if not found_problem:
        print("No NaN/Inf found anywhere.")

check_checkpoint("checkpoint/gefl_cifar10lt_baseline_latest.pt")
check_checkpoint("checkpoint/gefl_cifar10lt_proposed_latest.pt")
