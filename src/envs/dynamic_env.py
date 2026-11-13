"""Dynamic CVRP environment with stochastic traffic disruption.

Extends :class:`RouteEnv` with edge-wise traffic multipliers drawn from a
log-normal distribution and re-sampled every ``resample_every`` steps —
modelling congestion that shifts while the vehicle is en route.

Dynamism levels (``delta``):
    static: multipliers pinned to 1 (regression-identical to RouteEnv)
    low:    sigma = 0.15  (multipliers mostly 0.75x–1.35x)
    high:   sigma = 0.40  (multipliers mostly 0.45x–2.2x)

The observation is extended with the current traffic multipliers from the
agent's position to every node, so a learning agent can condition its
next-node choice on live conditions. ``info['traffic_version']``
increments on every re-sample, giving re-planning baselines their
trigger, and ``info['effective_dist']`` exposes the full current
travel-cost matrix (what a dispatcher's live traffic feed would show).
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from .instances import CVRPInstance
from .route_env import RouteEnv

SIGMA = {"static": 0.0, "low": 0.15, "high": 0.40}
CLIP = (0.5, 3.0)


class DynamicRouteEnv(RouteEnv):
    def __init__(
        self,
        instance: CVRPInstance,
        delta: str = "low",
        resample_every: int = 3,
        seed: int | None = None,
    ):
        super().__init__(instance, seed=seed)
        if delta not in SIGMA:
            raise ValueError(f"delta must be one of {list(SIGMA)}")
        self.delta = delta
        self.sigma = SIGMA[delta]
        self.resample_every = resample_every

        obs_dim = 6 * self.n_nodes + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32)

        self.traffic = np.ones((self.n_nodes, self.n_nodes))
        self.traffic_version = 0
        self._steps_since_resample = 0

    # -- traffic ------------------------------------------------------------

    def _resample_traffic(self) -> None:
        if self.sigma == 0.0:
            return
        m = self._rng.lognormal(mean=0.0, sigma=self.sigma, size=self.traffic.shape)
        m = np.clip((m + m.T) / 2, *CLIP)  # symmetric, bounded
        np.fill_diagonal(m, 1.0)
        self.traffic = m
        self.traffic_version += 1

    def effective_dist(self) -> np.ndarray:
        return self.instance.dist * self.traffic

    def step_distance(self, a: int, b: int) -> float:
        return float(self.instance.dist[a, b] * self.traffic[a, b])

    # -- gym API overrides ----------------------------------------------------

    def _obs(self) -> np.ndarray:
        base = super()._obs()
        live = self.traffic[self.position] / CLIP[1]
        return np.concatenate([base, live.astype(np.float32)])

    def _info(self) -> dict:
        info = super()._info()
        info["traffic_version"] = self.traffic_version
        info["effective_dist"] = self.effective_dist()
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.traffic = np.ones((self.n_nodes, self.n_nodes))
        self.traffic_version = 0
        self._steps_since_resample = 0
        self._resample_traffic()
        return self._obs(), self._info()

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        self._steps_since_resample += 1
        if self._steps_since_resample >= self.resample_every:
            self._resample_traffic()
            self._steps_since_resample = 0
            obs, info_new = self._obs(), self._info()
            info_new["distance"] = info["distance"]
            info = info_new
        return obs, reward, terminated, truncated, info
