import numpy as np

from src.baselines.nearest_neighbour import NearestNeighbourPolicy
from src.baselines.ortools_replan import ORToolsReplanPolicy
from src.baselines.random_policy import RandomPolicy
from src.envs.dynamic_env import CLIP, DynamicRouteEnv
from src.envs.instances import load
from src.envs.route_env import RouteEnv
from src.experiments.cross_eval import TrafficBlindDQNPolicy
from src.experiments.rollout import execute_fixed_routes, rollout_policy
from src.experiments.verify_resolve_fix import ForcedReplanPolicy


def test_static_delta_reproduces_route_env():
    inst = load("E-n22-k4")
    static = rollout_policy(RouteEnv(inst), RandomPolicy(seed=7), seed=7)
    dynamic = rollout_policy(
        DynamicRouteEnv(inst, delta="static"), RandomPolicy(seed=7), seed=7
    )
    assert static["distance"] == dynamic["distance"]
    assert static["route"] == dynamic["route"]


def test_traffic_resamples_and_stays_bounded():
    inst = load("P-n16-k8")
    env = DynamicRouteEnv(inst, delta="high", resample_every=2)
    _, info = env.reset(seed=0)
    versions = {info["traffic_version"]}
    policy = NearestNeighbourPolicy()
    obs, terminated = None, False
    obs, info = env._obs(), info
    while not terminated:
        action = policy.act(env, obs, info)
        obs, _, terminated, _, info = env.step(action)
        versions.add(info["traffic_version"])
        off_diag = env.traffic[~np.eye(env.n_nodes, dtype=bool)]
        assert off_diag.min() >= CLIP[0] and off_diag.max() <= CLIP[1]
    assert len(versions) > 2, "traffic should re-sample repeatedly during episode"


def test_fixed_plan_costs_more_under_disruption_on_average():
    inst = load("A-n32-k5")
    plan = inst.best_known_routes
    static_cost = inst.best_known_cost
    costs = []
    for seed in range(5):
        env = DynamicRouteEnv(inst, delta="high")
        costs.append(execute_fixed_routes(env, plan, seed=seed)["distance"])
    assert np.mean(costs) > static_cost


def test_replan_policy_serves_all_customers():
    inst = load("P-n16-k8")
    env = DynamicRouteEnv(inst, delta="high")
    policy = ORToolsReplanPolicy(time_limit_s=1)
    result = rollout_policy(env, policy, seed=3)
    assert env.visited.all() and env.position == 0
    assert policy.replan_count >= 2
    assert result["distance"] > 0


def test_disruption_stream_is_deterministic_from_seed():
    """The comparative design's load-bearing fairness claim (§3.3.3/§4.8.1):
    re-running with the same seed must reproduce byte-identical traffic
    draws, not just similar ones. No test asserted this before (2026-08-05
    audit)."""
    inst = load("P-n16-k8")
    policy = NearestNeighbourPolicy()

    def run():
        env = DynamicRouteEnv(inst, delta="high", resample_every=2)
        obs, info = env.reset(seed=0)
        traffic_snapshots = [env.traffic.copy()]
        terminated = False
        while not terminated:
            action = policy.act(env, obs, info)
            obs, _, terminated, _, info = env.step(action)
            traffic_snapshots.append(env.traffic.copy())
        return traffic_snapshots

    run_a, run_b = run(), run()
    assert len(run_a) == len(run_b)
    for a, b in zip(run_a, run_b):
        assert np.array_equal(a, b)


def test_disruption_stream_identical_across_different_policies_at_same_seed():
    """Fairness-by-construction: two different methods facing the same
    (instance, delta, seed) must see the same traffic realisation at each
    resample checkpoint, regardless of which actions they took to get
    there — resampling is triggered purely by elapsed step count, not by
    position (see DynamicRouteEnv.step)."""
    inst = load("P-n16-k8")

    def traffic_history(policy):
        env = DynamicRouteEnv(inst, delta="high", resample_every=2)
        obs, info = env.reset(seed=42)
        history = [env.traffic.copy()]
        terminated = False
        while not terminated:
            action = policy.act(env, obs, info)
            obs, _, terminated, _, info = env.step(action)
            if info["traffic_version"] != len(history) - 1:
                history.append(env.traffic.copy())
        return history

    hist_nn = traffic_history(NearestNeighbourPolicy())
    hist_random = traffic_history(RandomPolicy(seed=99))
    for a, b in zip(hist_nn, hist_random):
        assert np.array_equal(a, b)


def test_traffic_blind_policy_truncation_drops_exactly_the_live_row():
    """TrafficBlindDQNPolicy (the decomposition experiment's core mechanism,
    src/experiments/cross_eval.py) must hand the static-trained network
    exactly the static observation layout — nothing more, nothing less."""
    inst = load("P-n16-k8")
    env = DynamicRouteEnv(inst, delta="high")
    obs, info = env.reset(seed=0)
    static_obs_dim = 5 * env.n_nodes + 1
    assert obs.shape[0] == 6 * env.n_nodes + 1  # sanity: dynamic obs is 1 row bigger

    # RouteEnv._obs, called unbound on the same (dynamic) env instance,
    # computes exactly the static-layout observation for the current state —
    # the ground truth the truncation must match.
    expected_static_obs = RouteEnv._obs(env)
    assert np.array_equal(obs[:static_obs_dim], expected_static_obs)

    received = {}

    class RecordingAgent:
        def select_action(self, obs, mask, greedy=True):
            received["obs"] = obs
            return int(np.flatnonzero(mask)[0])

    policy = TrafficBlindDQNPolicy(RecordingAgent(), static_obs_dim)
    policy.act(env, obs, info)
    assert received["obs"].shape[0] == static_obs_dim
    assert np.array_equal(received["obs"], expected_static_obs)


def test_forced_resolve_on_static_matrix_reproduces_bks():
    """Regression guard for the OR-Tools re-solve mispricing fix (Chapter 3
    §3.4.3): if a future refactor reintroduced the zero-priced-empty-vehicle
    defect, forced periodic re-solves on the unperturbed static matrix would
    stop reproducing the best-known solution. Runs only the smallest
    instance (fast) — see src/experiments/verify_resolve_fix.py for the full
    8-instance verification that backs the dissertation's "6 of 8 exact"
    claim."""
    inst = load("P-n16-k8")
    env = RouteEnv(inst, seed=0)
    policy = ForcedReplanPolicy(mode="expectation", time_limit_s=1)
    result = rollout_policy(env, policy, seed=0)
    assert result["distance"] == inst.best_known_cost
