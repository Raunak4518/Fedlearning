"""
utils/user_sampling.py

Client participation for one communication round. Mirrors the original
GeFL repo's `user_select(args)` convention: args.frac controls what
fraction of args.num_users participate each round (1.0 = full
participation, every round). A dedicated RNG (seeded from args.seed, but
distinct per call via a running counter) is used so partial-participation
draws don't perturb model-initialization randomness or vice versa.
"""
import numpy as np


class ClientSampler:
    def __init__(self, num_users: int, frac: float, seed: int = 0):
        self.num_users = num_users
        self.num_selected = max(1, int(round(frac * num_users)))
        self.rng = np.random.RandomState(seed)

    def select(self):
        if self.num_selected >= self.num_users:
            return list(range(self.num_users))
        return sorted(self.rng.choice(self.num_users, self.num_selected, replace=False).tolist())


def user_select(args, sampler: ClientSampler = None):
    if sampler is None:
        sampler = ClientSampler(args.num_users, args.frac, args.seed)
    return sampler.select()
