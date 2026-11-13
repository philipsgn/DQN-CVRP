"""Test Suite for Neural LNS, Hard Fleet Cap (K_used <= K) and UI Contracts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from src.cvrp.motorbike_env import MotorbikeCVRPEnv, clarke_wright_asymmetric, consolidate_routes_to_k
from src.cvrp.neural_lns_env import NeuralLNSEnv
from src.web.dispatch_motorbike import (
    dispatch_motorbike,
    DispatchRequest,
    StoreItem,
    FleetConfig,
    get_stores,
    get_depots
)


def test_hard_fleet_cap_consolidation():
    """Verifies that clarke_wright_asymmetric and consolidate_routes_to_k enforce len(routes) <= K."""
    stores = get_stores()
    depots = get_depots()
    assert len(stores) >= 24
    assert len(depots) >= 1

    depot = depots[0]
    # Pick 24 stores with total demand around 315 kg
    # 24 stores * ~13 kg or varying packages
    sample_stores = stores[:24]

    # Target: K = 7 vehicles, Q = 50 kg (Total fleet capacity = 350 kg >= 315 kg)
    req = DispatchRequest(
        depot={"id": depot["id"], "lat": depot["lat"], "lon": depot["lon"]},
        stores=[
            StoreItem(id=s["id"], n_pkg=2 if idx < 15 else 3)
            for idx, s in enumerate(sample_stores)
        ],
        fleet=FleetConfig(k=7, q_kg=50),
        options={"solver_mode": "cw_greedy"}
    )

    res_cw = dispatch_motorbike(req)
    assert res_cw["k_used"] <= 7, f"CW + Greedy exceeded K=7: {res_cw['k_used']}"
    assert len(res_cw["vehicles"]) <= 7

    # Verify each route load <= 50 kg
    for v in res_cw["vehicles"]:
        assert v["load_kg"] <= 50.0 + 1e-3, f"Vehicle overloaded: {v['load_kg']} > 50"


def test_neural_lns_env_step():
    """Verifies NeuralLNSEnv initialization, action masking, step execution, and SA acceptance."""
    dist_m = np.random.uniform(500, 5000, size=(15, 15))
    np.fill_diagonal(dist_m, 0)
    demands = [0.0] + [random_val for random_val in [10.0, 15.0, 5.0, 20.0, 10.0, 15.0, 10.0, 5.0, 10.0, 15.0, 20.0, 5.0, 10.0, 15.0]]
    capacity = 50.0
    k_vehicles = 4

    env = NeuralLNSEnv(
        dist_m=dist_m,
        demands_kg=demands,
        capacity_kg=capacity,
        k_vehicles=k_vehicles,
        max_steps=10,
        seed=123
    )

    state, mask = env.reset()
    assert state.shape == (36,)
    assert len(mask) == 6
    assert len(env.routes) <= k_vehicles

    for action in range(6):
        if mask[action]:
            next_state, reward, done, info = env.step(action)
            assert next_state.shape == (36,)
            assert isinstance(reward, float)
            assert len(env.routes) <= k_vehicles
            assert env.best_cost <= env.initial_cost + 1e-4
            if done:
                break


def test_all_three_models_dispatch():
    """Tests that cw_greedy, ddqn_base, and neural_lns all dispatch successfully and respect K <= 7."""
    stores = get_stores()
    depots = get_depots()
    depot = depots[0]
    sample_stores = stores[:12]

    for mode in ["cw_greedy", "ddqn_base", "neural_lns"]:
        req = DispatchRequest(
            depot={"id": depot["id"], "lat": depot["lat"], "lon": depot["lon"]},
            stores=[StoreItem(id=s["id"], n_pkg=2) for s in sample_stores],
            fleet=FleetConfig(k=4, q_kg=50),
            options={"solver_mode": mode}
        )
        res = dispatch_motorbike(req)
        assert res["k_used"] <= 4, f"{mode} exceeded K=4: {res['k_used']}"
        assert res["total_distance_km"] > 0
        assert "vehicles" in res


if __name__ == "__main__":
    test_hard_fleet_cap_consolidation()
    print("[PASS] Hard fleet cap consolidation (K_used <= K)")
    test_neural_lns_env_step()
    print("[PASS] NeuralLNSEnv steps and state space")
    test_all_three_models_dispatch()
    print("[PASS] All 3 models dispatch correctly")
