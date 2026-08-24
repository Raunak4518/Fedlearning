"""
utils/label_sampler.py

Controls p(y) when a client draws conditioning labels to generate
synthetic augmentation samples from the shared generator each round.

UniformLabelSampler   -- GeFL baseline / the reference implementation's
                          behaviour: p(y) uniform over classes, regardless
                          of true frequency or generator reliability.

FidelityGatedSampler  -- Mechanism B. Maintains a per-class fidelity score
                          (an EMA of target-network confidence on synthetic
                          samples of that class -- reused for free from the
                          same forward pass local training already runs).
                          p(y) blends from the natural class frequency
                          toward inverse-frequency rebalancing as mean
                          fidelity rises, with each class's own weight
                          inside the inverse-frequency term additionally
                          scaled by that class's own fidelity so a
                          still-unreliable rare class isn't aggressively
                          oversampled just because it's rare.

Both are registered under SAMPLER_REGISTRY so new sampling strategies can
be added (e.g. a pure inverse-frequency ablation) without touching the
training loop.
"""
import numpy as np
import torch

from registry import Registry

SAMPLER_REGISTRY = Registry("label_sampler")


class BaseLabelSampler:
    name = "base"

    def __init__(self, num_classes: int, natural_class_counts: np.ndarray, args=None):
        self.num_classes = num_classes
        self.natural_class_counts = natural_class_counts

    def sample(self, n: int) -> torch.Tensor:
        raise NotImplementedError

    def update_fidelity(self, class_confidences: dict) -> None:
        """class_confidences: {class_id: mean target-net softmax confidence
        on this round's synthetic samples of that class}."""
        pass

    def state(self) -> dict:
        """For logging / diagrams: whatever internal state is worth tracking."""
        return {}


@SAMPLER_REGISTRY.register("uniform")
class UniformLabelSampler(BaseLabelSampler):
    name = "uniform (GeFL baseline)"

    def sample(self, n: int) -> torch.Tensor:
        return torch.randint(0, self.num_classes, (n,))


@SAMPLER_REGISTRY.register("fidelity_gated")
class FidelityGatedSampler(BaseLabelSampler):
    name = "fidelity-gated adaptive (Mechanism B)"

    def __init__(self, num_classes: int, natural_class_counts: np.ndarray, args=None):
        super().__init__(num_classes, natural_class_counts, args)
        total = natural_class_counts.sum()
        self.natural_freq = (natural_class_counts / total) if total > 0 else np.full(num_classes, 1 / num_classes)
        init_fidelity = getattr(args, "mech_b_init_fidelity", 0.15) if args is not None else 0.15
        self.ema_decay = getattr(args, "mech_b_ema_decay", 0.6) if args is not None else 0.6
        self.inv_freq_power = getattr(args, "mech_b_inv_freq_power", 1.0) if args is not None else 1.0
        self.fidelity = np.full(num_classes, init_fidelity, dtype=np.float64)

    def update_fidelity(self, class_confidences: dict) -> None:
        for c, conf in class_confidences.items():
            self.fidelity[c] = self.ema_decay * self.fidelity[c] + (1 - self.ema_decay) * conf

    def _inverse_freq_dist(self) -> np.ndarray:
        inv = 1.0 / np.clip(self.natural_freq, 1e-6, None)
        inv = np.power(inv, self.inv_freq_power)
        inv = inv * self.fidelity  # don't oversample a rare-but-still-unreliable class
        if inv.sum() == 0:
            return self.natural_freq.copy()
        return inv / inv.sum()

    def sample(self, n: int) -> torch.Tensor:
        g = float(self.fidelity.mean())
        p = (1 - g) * self.natural_freq + g * self._inverse_freq_dist()
        p = p / p.sum()
        labels = np.random.choice(self.num_classes, size=n, p=p)
        return torch.from_numpy(labels).long()

    def state(self) -> dict:
        return {"mean_fidelity": float(self.fidelity.mean()), "fidelity_per_class": self.fidelity.tolist()}


def build_label_sampler(args, num_classes: int, natural_class_counts: np.ndarray) -> BaseLabelSampler:
    name = "fidelity_gated" if args.mechanism_b else "uniform"
    cls = SAMPLER_REGISTRY.get(name)
    return cls(num_classes, natural_class_counts, args)
