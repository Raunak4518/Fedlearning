"""
utils/localUpdateTarget.py

One client's local target-network training step, for one communication
round. Matches the original repo's `LocalUpdate` naming/interface:

    weight, loss, fidelity_feedback = LocalUpdate(args, dataloader).train(
        net, gennet=global_generator, label_sampler=...
    )

- If `args.aid_by_gen` is 0, this degrades to plain local SGD on real data
  only (the FedAvg-without-any-generator baseline).
- If `args.aid_by_gen` is 1 and `gennet` is provided, each round the
  client draws `args.synth_batch` synthetic labels from its label sampler
  (uniform under the GeFL baseline, fidelity-gated under Mechanism B),
  generates the matching synthetic images from the (already-aggregated)
  global generator, and trains on real-local ∪ synthetic together.
- If `label_sampler` is a FidelityGatedSampler (Mechanism B active), this
  also measures mean target-net confidence on the synthetic batch, per
  class, and returns it as `fidelity_feedback` so the caller can call
  `label_sampler.update_fidelity(...)` for next round -- reusing this same
  forward pass, no extra compute.
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
        opt = self._make_optimizer(net)

        total_loss, n_batches = 0.0, 0
        fidelity_conf_sum = defaultdict(float)
        fidelity_conf_n = defaultdict(int)

        for _ in range(args.local_ep):
            for x, y in self.dataloader:
                x, y = x.to(args.device), y.to(args.device)

                if args.aid_by_gen and gennet is not None:
                    syn_y = label_sampler.sample(args.synth_batch).to(args.device)
                    with torch.no_grad():
                        syn_x = gennet.sample(syn_y)
                    xb = torch.cat([x, syn_x], dim=0)
                    yb = torch.cat([y, syn_y], dim=0)
                else:
                    xb, yb = x, y

                opt.zero_grad()
                logits = net(xb)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1

                if args.aid_by_gen and gennet is not None and args.mechanism_b:
                    with torch.no_grad():
                        probs = F.softmax(net(syn_x), dim=1)
                        conf = probs.gather(1, syn_y.unsqueeze(1)).squeeze(1)
                    for c in syn_y.unique().tolist():
                        mask = syn_y == c
                        fidelity_conf_sum[c] += conf[mask].sum().item()
                        fidelity_conf_n[c] += int(mask.sum())

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
        for _ in range(args.local_ep):
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
