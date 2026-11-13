"""
Unit tests for the 4-tier Feasibility Gate (MP-MOTO-CVRP V9.2).
"""

import pytest
from src.cvrp.feasibility import check_feasibility, first_fit_decreasing_bins, solve_exact_bin_packing

def test_tier1_overweight_item():
    demands = [10.0, 15.0, 65.0, 20.0]  # 65 > 50
    res = check_feasibility(demands, k_vehicles=3, capacity_kg=50.0)
    assert not res.is_feasible
    assert res.code == "ORDER_EXCEEDS_Q"
    assert res.metadata["splits_suggested"] == 2
    assert "vượt quá tải trọng" in res.reason

def test_tier2_total_demand_exceeded():
    demands = [25.0, 25.0, 30.0, 30.0]  # total 110 > 2 * 50 = 100
    res = check_feasibility(demands, k_vehicles=2, capacity_kg=50.0)
    assert not res.is_feasible
    assert res.code == "TOTAL_EXCEEDS_CAPACITY"
    assert res.metadata["k_min_by_weight"] == 3
    assert "Cần điều động ít nhất 3 xe" in res.reason

def test_tier3_binpacking_infeasible():
    # 3 items of 30kg with Q = 50kg, K = 2
    # Total demand = 90kg <= 2 * 50 = 100kg (passes Tier 2)
    # But max 1 item of 30kg per 50kg vehicle -> needs 3 vehicles!
    demands = [30.0, 30.0, 30.0]
    res = check_feasibility(demands, k_vehicles=2, capacity_kg=50.0)
    assert not res.is_feasible
    assert res.code == "BINPACKING_INFEASIBLE"
    assert res.metadata["k_min_binpacking"] == 3
    assert "Bin-Packing" in res.reason

def test_tier4_unreachable_node():
    demands = [10.0, 15.0, 12.0]
    node_ids = [100001, 100002, 999999] # 999999 is outside SCC
    valid_scc = {100001, 100002, 100003}
    res = check_feasibility(demands, k_vehicles=2, capacity_kg=50.0, node_ids=node_ids, valid_scc_nodes=valid_scc)
    assert not res.is_feasible
    assert res.code == "UNREACHABLE_NODE"
    assert "nằm ngoài vùng liên thông" in res.reason

def test_all_tiers_pass():
    demands = [10.0, 15.0, 20.0, 15.0, 18.0] # total 78kg, Q = 50kg, K = 2
    node_ids = [100001, 100002, 100003, 100004, 100005]
    valid_scc = set(node_ids)
    res = check_feasibility(demands, k_vehicles=2, capacity_kg=50.0, node_ids=node_ids, valid_scc_nodes=valid_scc)
    assert res.is_feasible
    assert res.code == "OK"

