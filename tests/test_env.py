import math

import numpy as np
import pytest

from src.envs.instances import INSTANCES, load, route_cost
from src.envs.route_env import RouteEnv


@pytest.fixture(scope="module", params=list(INSTANCES))
def instance(request):
    return load(request.param)


def random_rollout(env: RouteEnv, seed: int):
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total_reward, steps = 0.0, 0
    capacities, actions = [], []
    terminated = False
    while not terminated:
        mask = info["action_mask"]
        assert mask.any(), "no feasible action available mid-episode"
        action = int(rng.choice(np.flatnonzero(mask)))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        capacities.append(env.remaining_capacity)
        actions.append(action)
        steps += 1
        assert steps < 10 * env.n_nodes, "episode failed to terminate"
    return total_reward, actions, capacities


def test_rollout_serves_all_customers_exactly_once(instance):
    env = RouteEnv(instance)
    _, actions, _ = random_rollout(env, seed=0)
    customer_visits = [a for a in actions if a != 0]
    assert sorted(customer_visits) == list(range(1, env.n_nodes))


def test_capacity_never_violated(instance):
    env = RouteEnv(instance)
    _, _, capacities = random_rollout(env, seed=1)
    assert min(capacities) >= 0


def test_reward_matches_route_distance(instance):
    env = RouteEnv(instance)
    total_reward, _, _ = random_rollout(env, seed=2)
    assert math.isclose(-total_reward * env.dist_scale, env.total_route_distance(), rel_tol=1e-9)


def test_route_distance_matches_independent_recompute(instance):
    env = RouteEnv(instance)
    random_rollout(env, seed=3)
    # split executed route into depot-to-depot subroutes and score them
    # with the same function that validates best-known solutions
    routes, current = [], []
    for node in env.route[1:]:
        if node == 0:
            routes.append(current)
            current = []
        else:
            current.append(node)
    assert route_cost(routes, instance.dist) == env.total_route_distance()


def test_mask_blocks_visited_and_overweight(instance):
    env = RouteEnv(instance)
    obs, info = env.reset(seed=4)
    first = int(np.flatnonzero(info["action_mask"])[1])  # first feasible customer
    obs, _, _, _, info = env.step(first)
    assert not info["action_mask"][first], "visited customer must be masked"
    over = np.flatnonzero(instance.demands > env.remaining_capacity)
    for node in over:
        if node != 0 and not env.visited[node]:
            assert not info["action_mask"][node], "overweight customer must be masked"


def test_infeasible_action_raises(instance):
    env = RouteEnv(instance)
    env.reset(seed=5)
    with pytest.raises(ValueError):
        env.step(0)  # depot self-loop from depot


def test_best_known_solution_replays_to_bks_cost(instance):
    env = RouteEnv(instance)
    env.reset(seed=6)
    total = 0.0
    for route in instance.best_known_routes:
        for node in [*route, 0]:
            _, reward, terminated, _, _ = env.step(node)
            total += reward
    assert terminated
    assert math.isclose(-total * env.dist_scale, instance.best_known_cost, rel_tol=1e-9)
