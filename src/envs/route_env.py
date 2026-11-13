"""Static CVRP Gymnasium environment.

Formulation (constructive, single vehicle with depot returns — the standard
RL-for-VRP setting of Nazari et al. 2018):

* State: agent position (one-hot), visited mask, remaining capacity, plus
  the instance layout (normalised coordinates and demands).
* Action: index of the next node to visit (0 = depot, 1..n = customers).
  Invalid actions (visited customers, demand exceeding remaining capacity,
  depot self-loop) are exposed through an action mask in ``info``.
* Reward: negative travelled distance, normalised by the instance's max
  pairwise distance. Raw step distance is reported in ``info``.
* Episode ends when every customer is served and the vehicle is back at
  the depot. Returning to the depot refills capacity, so fleet size is
  implicitly unbounded; total distance is the objective, and optimality
  gaps vs. best-known solutions remain directly comparable.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .instances import CVRPInstance


class RouteEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, instance: CVRPInstance, seed: int | None = None):
        super().__init__()
        self.instance = instance
        self.n_nodes = instance.n_customers + 1
        self.dist_scale = float(instance.dist.max())

        span = instance.coords.max(axis=0) - instance.coords.min(axis=0)
        span[span == 0] = 1.0
        self._norm_coords = (instance.coords - instance.coords.min(axis=0)) / span
        self._norm_demands = instance.demands / instance.capacity

        self.action_space = spaces.Discrete(self.n_nodes)
        obs_dim = 5 * self.n_nodes + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.position = 0
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.remaining_capacity = instance.capacity
        self.route: list[int] = [0]

    # -- helpers -----------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        """Boolean mask of feasible next nodes."""
        mask = np.zeros(self.n_nodes, dtype=bool)
        unserved = ~self.visited[1:]
        feasible = unserved & (self.instance.demands[1:] <= self.remaining_capacity)
        mask[1:] = feasible
        # depot is a valid move whenever we are not already there;
        # if nothing else is feasible it is the only move.
        if self.position != 0:
            mask[0] = True
        return mask

    def _obs(self) -> np.ndarray:
        pos = np.zeros(self.n_nodes, dtype=np.float32)
        pos[self.position] = 1.0
        return np.concatenate(
            [
                self._norm_coords.ravel().astype(np.float32),
                self._norm_demands.astype(np.float32),
                self.visited.astype(np.float32),
                pos,
                np.array([self.remaining_capacity / self.instance.capacity], dtype=np.float32),
            ]
        )

    def _info(self) -> dict:
        return {"action_mask": self.action_mask(), "route": list(self.route)}

    def step_distance(self, a: int, b: int) -> float:
        """Distance for moving a->b; hook overridden by the dynamic env."""
        return float(self.instance.dist[a, b])

    # -- gym API ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.position = 0
        self.visited = np.zeros(self.n_nodes, dtype=bool)
        self.visited[0] = True
        self.remaining_capacity = self.instance.capacity
        self.route = [0]
        return self._obs(), self._info()

    def step(self, action: int):
        action = int(action)
        if not self.action_mask()[action]:
            raise ValueError(
                f"Infeasible action {action} at position {self.position} "
                f"(visited={self.visited[action]}, cap={self.remaining_capacity})"
            )

        distance = self.step_distance(self.position, action)
        reward = -distance / self.dist_scale

        self.position = action
        self.route.append(action)
        if action == 0:
            self.remaining_capacity = self.instance.capacity
        else:
            self.visited[action] = True
            self.remaining_capacity -= int(self.instance.demands[action])

        terminated = bool(self.visited.all() and self.position == 0)
        info = self._info()
        info["distance"] = distance
        return self._obs(), reward, terminated, False, info

    # -- reporting ----------------------------------------------------------

    def total_route_distance(self) -> float:
        """Recompute the executed route's distance from the static matrix."""
        return float(
            sum(self.instance.dist[a, b] for a, b in zip(self.route, self.route[1:]))
        )
