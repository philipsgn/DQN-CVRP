"""OR-Tools with event-triggered re-planning for the dynamic environment.

Re-solves from the vehicle's current state whenever the environment
re-samples traffic. This is the strongest fair classical competitor to a
learned adaptive policy — and its cumulative re-solve wall time is the
price it pays, which we record.

Pricing modes
-------------
Traffic multipliers are i.i.d. per re-sample with no persistence, so the
expected cost of any future leg is proportional to its static distance.

* ``expectation`` (default): plan on the static matrix. This is the
  expectation-optimal replanner for transient i.i.d. noise, and empirically
  reproduces the optimal tour exactly when re-solving mid-route.
* ``live_full``: plan on the full current effective matrix — the naive
  dispatcher that treats transient congestion as permanent.
* ``live_row``: only the departure row from the current position is
  live-priced (the information the DQN observes).

Both live modes empirically inflate realized tours by ~15–25% on
E-n22-k4: each re-plan front-loads a detour to grab a transiently
discounted leg into the executed 3-step prefix, while the compensating
efficient remainder is back-loaded and discarded at the next re-plan.
Kept as experimental conditions (negative result: when disruptions are
transient, chasing live prices with a full-horizon solver is harmful).
"""

from __future__ import annotations

import numpy as np

from .ortools_solver import solve

MODES = ("expectation", "live_row", "live_full")


class ORToolsReplanPolicy:
    name = "ortools_replan"

    def __init__(self, time_limit_s: int = 1, mode: str = "expectation"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.time_limit_s = time_limit_s
        self.mode = mode
        self.queue: list[int] = []
        self._version = -1
        self.replan_count = 0
        self.replan_time_s = 0.0

    def _planning_dist(self, env, info) -> np.ndarray:
        static = env.instance.dist.astype(float)
        if self.mode == "expectation":
            return static
        eff = np.asarray(info.get("effective_dist", static), dtype=float)
        if self.mode == "live_full":
            return eff
        dist = static.copy()
        dist[env.position] = eff[env.position]
        return dist

    def _hint(self, env) -> list[list[int]] | None:
        """Previous plan's remainder as a warm start: still feasible, so the
        re-solve is guaranteed no worse than just following the old plan."""
        if not self.queue or env.position == 0:
            return None
        routes, current = [], []
        for node in self.queue:
            if node == 0:
                routes.append(current)
                current = []
            else:
                current.append(node)
        if current:
            routes.append(current)
        return routes

    def _replan(self, env, info) -> None:
        serve_mask = ~env.visited.copy()
        serve_mask[0] = False
        routes, _, wall = solve(
            env.instance,
            dist=self._planning_dist(env, info),
            time_limit_s=self.time_limit_s,
            serve_mask=serve_mask,
            start_node=env.position,
            start_capacity=env.remaining_capacity,
            hint=self._hint(env),
        )
        self.replan_count += 1
        self.replan_time_s += wall
        self.queue = [node for route in routes for node in [*route, 0]]
        # drop a leading depot-return if we are already at the depot
        while self.queue and self.queue[0] == 0 and env.position == 0:
            self.queue.pop(0)

    def act(self, env, obs, info) -> int:
        version = info.get("traffic_version", 0)
        if version != self._version or not self.queue:
            self._version = version
            self._replan(env, info)
        if not self.queue:
            return 0  # everything served: head home
        nxt = self.queue[0]
        # If the solver's plan implies a refill we cannot express (e.g. its
        # first vehicle was left unused, dropping the start->depot leg),
        # detour via the depot before continuing the plan.
        if nxt != 0 and not info["action_mask"][nxt]:
            return 0
        return self.queue.pop(0)
