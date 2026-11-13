"""Gymnasium-compatible Local Search Environment for CVRP.
State: 28-dimensional continuous vector (F28)
Action: Discrete(4) -> [0: 2-opt, 1: intra-relocate, 2: inter-relocate, 3: inter-swap]
Reward: +100 * delta_C / C if improved, -0.1 otherwise.
"""
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .instance import CVRPInstance
from .clarke_wright import clarke_wright_savings
from .operators import best_move, OPERATOR_NAMES
from .state_f28 import StateExtractorF28


class LocalSearchEnv(gym.Env):
    """Local Search Environment controlling CVRP operators via RL."""
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        instance: CVRPInstance,
        max_evals: int = 2560,
        patience: Optional[int] = None,
        feature_patience: float = 30.0,
        seed: Optional[int] = 42
    ):
        super().__init__()
        self.instance = instance
        self.max_evals = max_evals
        self.patience = patience
        self.feature_patience = feature_patience
        self.seed_val = seed
        self.rng = random.Random(seed)

        # Spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(28,), dtype=np.float32
        )

        self.state_extractor = StateExtractorF28(feature_patience=feature_patience)

        # Dynamic state
        self.routes: List[List[int]] = []
        self.initial_cost: float = 0.0
        self.current_cost: float = 0.0
        self.evals_used: int = 0
        self.steps_taken: int = 0
        self.consecutive_stagnation: int = 0
        self.history_records: List[Dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial Clarke-Wright solution."""
        super().reset(seed=seed)
        if seed is not None:
            self.rng = random.Random(seed)

        self.routes = clarke_wright_savings(self.instance)
        self.initial_cost = self.instance.solution_cost(self.routes)
        self.current_cost = self.initial_cost
        self.evals_used = 0
        self.steps_taken = 0
        self.consecutive_stagnation = 0
        self.history_records = []
        self.state_extractor.reset()

        state = self.state_extractor.extract(
            self.routes,
            self.instance,
            evals_used=0,
            max_evals=self.max_evals,
            initial_cost=self.initial_cost
        )

        info = {
            "initial_cost": self.initial_cost,
            "current_cost": self.current_cost,
            "routes": self.routes,
            "evals_used": 0,
            "is_feasible": self.instance.is_feasible(self.routes)[0]
        }
        return state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one local search operator selected by the agent."""
        if action not in (0, 1, 2, 3):
            raise ValueError(f"Invalid action {action}. Must be 0, 1, 2, or 3.")

        self.steps_taken += 1
        prev_cost = self.current_cost

        # Execute candidate search for chosen operator
        new_routes, delta_cost, cand_count, improved = best_move(
            self.routes,
            action,
            self.instance,
            max_candidates=128,
            rng=self.rng
        )

        self.evals_used += cand_count

        if improved:
            self.routes = new_routes
            self.current_cost = self.instance.solution_cost(self.routes)
            # Legacy reward formula from report: +100 * delta_C / C
            reward = float(100.0 * (delta_cost / max(1e-6, prev_cost)))
            self.consecutive_stagnation = 0
        else:
            reward = -0.1
            self.consecutive_stagnation += 1

        # Update history
        self.state_extractor.record_step(action, improved, delta_cost)

        # Check termination
        terminated = False
        truncated = False
        stop_reason = ""

        if self.evals_used >= self.max_evals:
            terminated = True
            stop_reason = "max_evals"
        elif self.patience is not None and self.consecutive_stagnation >= self.patience:
            terminated = True
            stop_reason = "patience"

        next_state = self.state_extractor.extract(
            self.routes,
            self.instance,
            evals_used=self.evals_used,
            max_evals=self.max_evals,
            initial_cost=self.initial_cost
        )

        impr_percent = ((self.initial_cost - self.current_cost) / max(1e-6, self.initial_cost)) * 100.0

        step_info = {
            "action": int(action),
            "operator_name": OPERATOR_NAMES[action],
            "improved": improved,
            "delta_cost": float(delta_cost),
            "current_cost": float(self.current_cost),
            "improvement_pct": float(impr_percent),
            "evals_used": int(self.evals_used),
            "candidates_evaluated": int(cand_count),
            "steps_taken": int(self.steps_taken),
            "stop_reason": stop_reason,
            "is_feasible": self.instance.is_feasible(self.routes)[0],
            "routes": self.routes
        }
        self.history_records.append(step_info)

        return next_state, reward, terminated, truncated, step_info

