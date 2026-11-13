"""Masked random policy — the sanity floor for all comparisons."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, info) -> int:
        return int(self.rng.choice(np.flatnonzero(info["action_mask"])))
