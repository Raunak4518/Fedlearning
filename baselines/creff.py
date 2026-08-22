"""
baselines/creff.py

CReFF (Shang et al., IJCAI 2022) -- the classifier-side comparison
baseline the proposal's Section 5 methodology calls for. Unlike GeFL,
CReFF assumes every client can run the SAME backbone architecture (no
model heterogeneity), which is exactly the axis the main GeFL_*.py scripts
exist to support -- CReFF is included here purely as the reference point
that answers "how good is a long-tail fix when architectural heterogeneity
isn't a constraint", a strictly easier setting.

Phase 1: ordinary volume-weighted FedAvg on one shared backbone + head.
Phase 2: each client computes per-class feature mean/std from the final
         backbone and uploads only those statistics (never raw data or
         features); the server synthesizes a class-BALANCED set of
         features from them and retrains only the head.

Fully dataset-agnostic: `SharedBackbone` takes (in_channels, feat_dim,
img_size) and uses adaptive pooling, so it never hardcodes a resolution.
"""
import copy
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class SharedBackbone(nn.Module):
    def __init__(self, in_channels: int, feat_dim: int = 128, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(width * 4, feat_dim)

    def forward(self, x):
        h = self.pool(self.net(x)).flatten(1)
        return self.proj(h)


def _weighted_avg_state_dicts(state_dicts: List[dict], weights: List[int]) -> dict:
    total = sum(weights)
    out = {}
    for k in state_dicts[0]:
        out[k] = sum(sd[k].float() * (w / total) for sd, w in zip(state_dicts, weights))
    return out


def run_creff(exp, args) -> dict:
    device = args.device
    num_classes, in_channels, img_size = exp.meta.num_classes, exp.meta.in_channels, exp.meta.native_img_size
    client_ids = [cid for cid in exp.dict_users if len(exp.dict_users[cid]) > 0]
    loaders = {cid: DataLoader(Subset(exp.dataset_train, exp.dict_users[cid]), batch_size=args.local_bs, shuffle=True)
               for cid in client_ids}

    backbone = SharedBackbone(in_channels, args.creff_feat_dim).to(device)
    head = nn.Linear(args.creff_feat_dim, num_classes).to(device)

    # ---- Phase 1: ordinary FedAvg on backbone + head ----
    for _rnd in range(args.creff_rounds):
        bb_states, head_states, counts = [], [], []
        for cid in client_ids:
            local_bb, local_head = copy.deepcopy(backbone), copy.deepcopy(head)
            opt = torch.optim.Adam(list(local_bb.parameters()) + list(local_head.parameters()), lr=args.lr)
            local_bb.train(); local_head.train()
            for _ep in range(args.local_ep):
                for x, y in loaders[cid]:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    loss = F.cross_entropy(local_head(local_bb(x)), y)
                    loss.backward()
                    opt.step()
            bb_states.append(local_bb.state_dict())
            head_states.append(local_head.state_dict())
            counts.append(len(loaders[cid].dataset))
        backbone.load_state_dict(_weighted_avg_state_dicts(bb_states, counts))
        head.load_state_dict(_weighted_avg_state_dicts(head_states, counts))

    # ---- Phase 2: federated per-class feature statistics ----
    backbone.eval()
    sums: Dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(args.creff_feat_dim))
    sq_sums: Dict[int, torch.Tensor] = defaultdict(lambda: torch.zeros(args.creff_feat_dim))
    n_c: Dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for cid in client_ids:
            for x, y in loaders[cid]:
                feats = backbone(x.to(device)).cpu()
                for c in y.unique().tolist():
                    mask = y == c
                    f = feats[mask]
                    sums[c] += f.sum(dim=0)
                    sq_sums[c] += (f ** 2).sum(dim=0)
                    n_c[c] += int(mask.sum())

    means, stds = {}, {}
    for c in range(num_classes):
        if n_c[c] == 0:
            continue
        mean = sums[c] / n_c[c]
        var = (sq_sums[c] / n_c[c] - mean ** 2).clamp(min=1e-4)
        means[c] = mean
        stds[c] = var.sqrt()

    # ---- synthesize class-balanced features, retrain head only ----
    feats_syn, labels_syn = [], []
    for c, mean in means.items():
        z = torch.randn(args.creff_synth_per_class, args.creff_feat_dim) * stds[c] + mean
        feats_syn.append(z)
        labels_syn.append(torch.full((args.creff_synth_per_class,), c, dtype=torch.long))
    feats_syn = torch.cat(feats_syn, dim=0).to(device)
    labels_syn = torch.cat(labels_syn, dim=0).to(device)

    head = nn.Linear(args.creff_feat_dim, num_classes).to(device)  # re-init, retrain from scratch
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    for _ in range(args.creff_head_retrain_steps):
        perm = torch.randperm(len(labels_syn))
        for i in range(0, len(perm), args.local_bs):
            idx = perm[i:i + args.local_bs]
            opt.zero_grad()
            loss = F.cross_entropy(head(feats_syn[idx]), labels_syn[idx])
            loss.backward()
            opt.step()

    combined = nn.Sequential(backbone, head)
    return {"backbone": backbone, "head": head, "combined": combined}
