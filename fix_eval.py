import re
import sys

def main():
    with open('utils/evaluate.py', 'r', encoding='utf-8') as f:
        text = f.read()
    
    lpips_code = """def _get_lpips_fn(device: str):
    try:
        import lpips
        return lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    except ImportError:
        return None

@torch.no_grad()
def mnd_ratio(query_samples: torch.Tensor, ref_synth: torch.Tensor,
              ref_val: torch.Tensor, n_query: int = 1000,
              n_ref: int = 600, device: str = None,
              dataset_mean: tuple = None, dataset_std: tuple = None) -> float:
    if device is None:
        device = query_samples.device

    # Subsample to specified sizes
    n_q = min(n_query, query_samples.size(0))
    perm_q = torch.randperm(query_samples.size(0))[:n_q]
    query = query_samples[perm_q].to(device)

    n_s = min(n_ref, ref_synth.size(0))
    synth = ref_synth[torch.randperm(ref_synth.size(0))[:n_s]].to(device)

    n_v = min(n_ref, ref_val.size(0))
    val = ref_val[torch.randperm(ref_val.size(0))[:n_v]].to(device)

    # Scale real images (query and val) to [-1, 1] to match synth
    if dataset_mean is not None and dataset_std is not None:
        mean_t = torch.tensor(dataset_mean, device=device).view(1, -1, 1, 1)
        std_t = torch.tensor(dataset_std, device=device).view(1, -1, 1, 1)
        # Denormalize to [0, 1] then scale to [-1, 1]
        query = (query * std_t + mean_t) * 2.0 - 1.0
        val = (val * std_t + mean_t) * 2.0 - 1.0

    # LPIPS expects 3 channels
    if query.size(1) == 1:
        query = query.repeat(1, 3, 1, 1)
        synth = synth.repeat(1, 3, 1, 1)
        val = val.repeat(1, 3, 1, 1)

    # Try LPIPS, fall back to Euclidean
    lpips_fn = _get_lpips_fn(str(device))

    if lpips_fn is not None:
        # LPIPS works on (N, C, H, W) in [-1, 1]; compute pairwise in batches
        ratios = []
        batch_sz = 64
        for i in range(0, n_q, batch_sz):
            q_batch = query[i:i + batch_sz]  # (B, C, H, W)
            b = q_batch.size(0)

            # d(x_i, synth) for each query sample
            d_synth = torch.zeros(b, n_s, device=device)
            for j in range(0, n_s, batch_sz):
                s_batch = synth[j:j + batch_sz]
                s_len = s_batch.size(0)
                # Expand: (B,1,C,H,W) vs (1,S,C,H,W) -> pairwise
                q_exp = q_batch.unsqueeze(1).expand(-1, s_len, -1, -1, -1).reshape(-1, *q_batch.shape[1:])
                s_exp = s_batch.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(-1, *s_batch.shape[1:])
                d = lpips_fn(q_exp, s_exp).view(b, s_len)
                d_synth[:, j:j + s_len] = d

            # d(x_i, val) for each query sample
            d_val = torch.zeros(b, n_v, device=device)
            for j in range(0, n_v, batch_sz):
                v_batch = val[j:j + batch_sz]
                v_len = v_batch.size(0)
                q_exp = q_batch.unsqueeze(1).expand(-1, v_len, -1, -1, -1).reshape(-1, *q_batch.shape[1:])
                v_exp = v_batch.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(-1, *v_batch.shape[1:])
                d = lpips_fn(q_exp, v_exp).view(b, v_len)
                d_val[:, j:j + v_len] = d

            min_d_synth = d_synth.min(dim=1).values
            min_d_val = d_val.min(dim=1).values
            # ρ_i = min_d_val / min_d_synth (paper Eq. 1)
            valid = min_d_synth > 0
            if valid.any():
                ratios.append((min_d_val[valid] / min_d_synth[valid]).cpu())

        if not ratios:
            return float("nan")
        return float(torch.cat(ratios).mean().item())
    else:
        # Fallback: Euclidean in flattened pixel space
        def _flat(t):
            return t.flatten(1).float()

        q_flat = _flat(query)
        s_flat = _flat(synth)
        v_flat = _flat(val)

        min_d_synth = torch.cdist(q_flat, s_flat).min(dim=1).values
        min_d_val = torch.cdist(q_flat, v_flat).min(dim=1).values
        valid = min_d_synth > 0
        if not valid.any():
            return float("nan")
        return float((min_d_val[valid] / min_d_synth[valid]).mean().item())


def make_held_out_val_split(train_ds, val_fraction: float, seed: int = 0):
    import numpy as np
    n = len(train_ds)
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_fraction))
    val_idx = idx[:n_val].tolist()
    remaining_idx = idx[n_val:].tolist()
    from torch.utils.data import Subset
    return Subset(train_ds, val_idx), remaining_idx
"""
    # Replace mnd_ratio and whatever comes after it
    text = re.sub(r'def mnd_ratio\(.*', lpips_code, text, flags=re.DOTALL)
    
    with open('utils/metrics.py', 'r', encoding='utf-8') as f:
        metrics_lines = f.readlines()
        
    metrics_add = "\\n" + "".join(metrics_lines[23:])
    
    # Vectorize compute_confusion_matrix!
    metrics_add = metrics_add.replace(
        "for true_c, pred_c in zip(y.cpu().numpy(), preds.cpu().numpy()):\\n            cm[true_c, pred_c] += 1",
        "indices = num_classes * y + preds\\n        cm += torch.bincount(indices, minlength=num_classes**2).view(num_classes, num_classes).cpu().numpy()"
    )
    
    with open('utils/evaluate.py', 'w', encoding='utf-8') as f:
        f.write(text + metrics_add)
        
if __name__ == '__main__':
    main()
