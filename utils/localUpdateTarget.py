"""
utils/localUpdateTarget.py

One client's local target-network training step, for one communication
round. Implements the paper's SEQUENTIAL (not joint) training design
(Algorithm 1, Table XIV):

    Phase 1 — Synthetic-only (T_s epochs):
        for t = 1...T_s:
            (x_i, y_i) ~ G(z|y, w_g)
            θ ← θ − α∇J(θ)

    Phase 2 — Real-only (T_r epochs):
        for t = 1...T_r:
            (x_i, y_i) ~ D_k (real local data)
            θ ← θ − α∇J(θ)

Real data gets T_r/T_s = 5x the epoch exposure synthetic data gets (by
default), and they NEVER share a batch. This is a deliberate design
choice from the paper: the 1:5 sequential split anchors training in real
data far more than a 1:1 joint mix would, diluting any miscalibration in
the generator's p(y).

- If `args.aid_by_gen` is 0, only the real-only phase runs (plain FedAvg).
- If Mechanism B is active, fidelity feedback is collected during the
  synthetic phase (same forward pass, no extra compute).
"""
from collections import defaultdict

import torch
import torch.nn.functional as F


class LocalUpdate:
    def __init__(self, args, dataloader):
        self.args = args
        self.dataloader = dataloader

    def _make_optimizer(self, net):
        if self.args.optimizer == "adam":
            return torch.optim.Adam(net.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        return torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                weight_decay=self.args.weight_decay)

    def train(self, net, gennet=None, label_sampler=None):
        args = self.args
        net.train()

        total_loss, n_batches = 0.0, 0
        fidelity_conf_sum = defaultdict(float)
        fidelity_conf_n = defaultdict(int)

        # ---- Phase 1: Synthetic-only (T_s epochs) ----
        # Paper Algorithm 1: train on generated samples ONLY, before real data.
        # Reference impl uses a fresh optimizer per phase; SGD momentum
        # carried over from a synthetic step corrupts the real-only phase.
        if args.aid_by_gen and gennet is not None:
            opt = self._make_optimizer(net)
            for _ in range(args.target_ts):
                for x_real, _ in self.dataloader:
                    # Use same batch count as real data, but draw synthetic samples
                    batch_size = x_real.size(0)
                    syn_y = label_sampler.sample(batch_size).to(args.device)
                    with torch.no_grad():
                        syn_x = gennet.sample(syn_y)

                    opt.zero_grad()
                    logits = net(syn_x)
                    loss = F.cross_entropy(logits, syn_y)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
                    n_batches += 1

                    # Collect fidelity feedback for Mechanism B (reusing this forward pass)
                    if args.mechanism_b:
                        with torch.no_grad():
                            probs = F.softmax(logits.detach(), dim=1)
                            conf = probs.gather(1, syn_y.unsqueeze(1)).squeeze(1)
                        for c in syn_y.unique().tolist():
                            mask = syn_y == c
                            fidelity_conf_sum[c] += conf[mask].sum().item()
                            fidelity_conf_n[c] += int(mask.sum())

        # ---- Phase 2: Real-only (T_r epochs) ----
        # Paper Algorithm 1: train on real local data ONLY, after synthetic.
        opt = self._make_optimizer(net)  # fresh optimizer, per reference impl
        for _ in range(args.target_tr):
            for x, y in self.dataloader:
                x, y = x.to(args.device), y.to(args.device)

                opt.zero_grad()
                logits = net(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        fidelity_feedback = {c: fidelity_conf_sum[c] / fidelity_conf_n[c] for c in fidelity_conf_sum}
        return net.state_dict(), avg_loss, fidelity_feedback


class LocalUpdate_onlyGen(LocalUpdate):
    """Ablation: train ONLY on synthetic samples (no real local data at
    all) -- useful for isolating how much a client's own real data
    contributes versus the shared generator alone."""

    def train(self, net, gennet=None, label_sampler=None):
        args = self.args
        assert gennet is not None and label_sampler is not None
        net.train()
        opt = self._make_optimizer(net)
        total_loss, n_batches = 0.0, 0
        n_real = sum(x.size(0) for x, _ in self.dataloader)
        for _ in range(args.target_ts):
            remaining = n_real
            while remaining > 0:
                b = min(args.local_bs, remaining)
                syn_y = label_sampler.sample(b).to(args.device)
                with torch.no_grad():
                    syn_x = gennet.sample(syn_y)
                opt.zero_grad()
                loss = F.cross_entropy(net(syn_x), syn_y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
                remaining -= b
        return net.state_dict(), total_loss / max(n_batches, 1), {}
