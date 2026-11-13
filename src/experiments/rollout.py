"""Shared episode runners used by baselines, evaluation, and experiments."""

from __future__ import annotations

import time

from ..envs.route_env import RouteEnv


def rollout_policy(env: RouteEnv, policy, seed: int = 0) -> dict:
    """Run one episode with a step-by-step policy; returns metrics."""
    obs, info = env.reset(seed=seed)
    total_distance, steps = 0.0, 0
    t0 = time.perf_counter()
    terminated = False
    while not terminated:
        action = policy.act(env, obs, info)
        obs, reward, terminated, _, info = env.step(action)
        total_distance += info["distance"]
        steps += 1
    return {
        "distance": total_distance,
        "static_distance": env.total_route_distance(),
        "wall_time_s": time.perf_counter() - t0,
        "steps": steps,
        "route": list(env.route),
    }


def execute_fixed_routes(env: RouteEnv, routes: list[list[int]], seed: int = 0) -> dict:
    """Execute a pre-computed plan verbatim (the 'static-then-fixed' mode).

    In the static environment this reproduces the plan's cost exactly; in
    the dynamic environment it pays whatever the disrupted conditions
    charge for the same node sequence.
    """
    obs, info = env.reset(seed=seed)
    total_distance = 0.0
    terminated = False
    for route in routes:
        for node in [*route, 0]:
            obs, reward, terminated, _, info = env.step(node)
            total_distance += info["distance"]
    assert terminated, "fixed plan did not serve all customers"
    return {
        "distance": total_distance,
        "static_distance": env.total_route_distance(),
        "route": list(env.route),
    }
