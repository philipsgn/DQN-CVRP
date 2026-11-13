"""Unit tests for the independent validator."""
from src.cvrp.instance import CVRPInstance
from src.cvrp.clarke_wright import clarke_wright_savings
from src.evaluation.validator import IndependentValidator


def test_validator_on_valid_solution():
    instance = CVRPInstance.generate_synthetic(
        name="ValidTest",
        num_customers=12,
        capacity=30,
        max_vehicles=5,
        seed=12
    )
    routes = clarke_wright_savings(instance)
    res = IndependentValidator.validate(instance, routes)
    assert res["is_valid"] is True
    assert res["status"] == "PASS"
    assert res["total_customers_served"] == 12
    assert res["recomputed_cost"] > 0.0


def test_validator_detects_violations():
    instance = CVRPInstance.generate_synthetic(
        name="InvalidTest",
        num_customers=10,
        capacity=15,
        max_vehicles=2,
        seed=15
    )
    # Intentionally bad routes: missing node 5, customer 2 duplicated
    bad_routes = [[0, 1, 2, 3, 0], [0, 2, 4, 0]]
    res = IndependentValidator.validate(instance, bad_routes)
    assert res["is_valid"] is False
    assert res["status"] == "FAIL"
    assert len(res["errors"]) > 0

