"""Baseline Policies for CVRP Local Search:
- RandomPolicy: Selects uniformly at random among {0, 1, 2, 3}
- CyclicPolicy: Cycles through 0 -> 1 -> 2 -> 3 -> 0
- GreedyPolicy: Evaluates operators and picks the one yielding the largest immediate improvement
"""
import random
from typing import List, Optional, Tuple
import numpy as np

from ..cvrp.instance import CVRPInstance
from ..cvrp.operators import best_move


class RandomPolicy:
    """Selects an operator uniformly at random, respecting action masks."""
    def __init__(self, seed: Optional[int] = 42):
        self.rng = random.Random(seed)

    def act(self, state: np.ndarray, action_mask: Optional[list] = None) -> Tuple[int, np.ndarray]:
        valid_actions = [0, 1, 2, 3]
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v] or [0, 1]
        action = self.rng.choice(valid_actions)
        probs = np.zeros(4, dtype=np.float32)
        for a in valid_actions:
            probs[a] = 1.0 / len(valid_actions)
        return action, probs


class CyclicPolicy:
    """Cycles through operators in round-robin fashion: 0 -> 1 -> 2 -> 3 -> 0."""
    def __init__(self):
        self.current = 0

    def act(self, state: np.ndarray, action_mask: Optional[list] = None) -> Tuple[int, np.ndarray]:
        valid_actions = [0, 1, 2, 3]
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v] or [0, 1]
        
        # Advance until valid action
        attempts = 0
        while attempts < 4 and self.current not in valid_actions:
            self.current = (self.current + 1) % 4
            attempts += 1
        
        action = self.current
        self.current = (self.current + 1) % 4
        pseudo_q = np.zeros(4, dtype=np.float32)
        pseudo_q[action] = 1.0
        return action, pseudo_q


class GreedyPolicy:
    """Greedy Policy: probes valid operators with a small sample (e.g. 16 candidates)
    and selects the operator with the largest immediate cost reduction.
    """
    def __init__(self, probe_candidates: int = 32, seed: Optional[int] = 42):
        self.probe_candidates = probe_candidates
        self.rng = random.Random(seed)

    def select_operator(
        self,
        routes: List[List[int]],
        instance: CVRPInstance,
        action_mask: Optional[list] = None
    ) -> Tuple[int, np.ndarray]:
        valid_actions = [0, 1, 2, 3]
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v] or [0, 1]

        deltas = np.full(4, -1e9, dtype=np.float32)
        for op in valid_actions:
            _, delta, _, _ = best_move(
                routes, op, instance, max_candidates=self.probe_candidates, rng=self.rng
            )
            deltas[op] = float(delta)

        best_op = int(np.argmax(deltas))
        if deltas[best_op] <= 1e-6:
            best_op = self.rng.choice(valid_actions)

        # Clean display deltas
        display_deltas = np.zeros(4, dtype=np.float32)
        for a in valid_actions:
            display_deltas[a] = max(0.0, deltas[a])

        return best_op, display_deltas

    def act(self, state: np.ndarray, action_mask: Optional[list] = None) -> Tuple[int, np.ndarray]:
        valid_actions = [0, 1, 2, 3]
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v] or [0, 1]
        return self.rng.choice(valid_actions), np.zeros(4, dtype=np.float32)

