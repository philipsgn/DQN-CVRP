"""Unit tests for the Clarke-Wright savings heuristic."""
from pathlib import Path
import pytest
from src.cvrp.instance import CVRPInstance
from src.cvrp.clarke_wright import clarke_wright_savings


def test_clarke_wright_synthetic():
    instance = CVRPInstance.generate_synthetic(
        name="TestCW",
        num_customers=20,
        capacity=30,
        max_vehicles=6,
        seed=100
    )
    routes = clarke_wright_savings(instance)
    assert len(routes) > 0
    is_feas, msg = instance.is_feasible(routes)
    assert is_feas, f"Clarke-Wright produced infeasible solution: {msg}"


def test_clarke_wright_hcmc():
    hcmc_path = Path("data/hcmc_instances/hcmc_district_20.json")
    if not hcmc_path.exists():
        pytest.skip("HCMC instance file not found")
    instance = CVRPInstance.from_json(hcmc_path)
    routes = clarke_wright_savings(instance)
    assert len(routes) > 0
    is_feas, msg = instance.is_feasible(routes)
    assert is_feas, f"HCMC Clarke-Wright infeasible: {msg}"

