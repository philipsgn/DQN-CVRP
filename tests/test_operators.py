"""Unit tests for the 4 CVRP local search operators."""
import random
import pytest
from src.cvrp.instance import CVRPInstance
from src.cvrp.clarke_wright import clarke_wright_savings
from src.cvrp.operators import (
    best_move,
    evaluate_2opt_candidates,
    evaluate_intra_relocate_candidates,
    evaluate_inter_relocate_candidates,
    evaluate_inter_swap_candidates
)


@pytest.fixture
def sample_instance():
    return CVRPInstance.generate_synthetic(
        name="TestInstance",
        num_customers=15,
        capacity=25,
        max_vehicles=5,
        seed=42
    )


def test_2opt_operator(sample_instance):
    rng = random.Random(42)
    routes = clarke_wright_savings(sample_instance)
    new_routes, delta, count, improved = evaluate_2opt_candidates(routes, sample_instance, rng=rng)
    assert count >= 0
    if improved:
        assert delta > 0.0
        is_feas, msg = sample_instance.is_feasible(new_routes)
        assert is_feas, f"2-opt caused infeasibility: {msg}"


def test_intra_relocate_operator(sample_instance):
    rng = random.Random(42)
    routes = clarke_wright_savings(sample_instance)
    new_routes, delta, count, improved = evaluate_intra_relocate_candidates(routes, sample_instance, rng=rng)
    assert count >= 0
    if improved:
        assert delta > 0.0
        is_feas, msg = sample_instance.is_feasible(new_routes)
        assert is_feas, f"intra-relocate caused infeasibility: {msg}"


def test_inter_relocate_operator(sample_instance):
    rng = random.Random(42)
    routes = clarke_wright_savings(sample_instance)
    new_routes, delta, count, improved = evaluate_inter_relocate_candidates(routes, sample_instance, rng=rng)
    assert count >= 0
    if improved:
        assert delta > 0.0
        is_feas, msg = sample_instance.is_feasible(new_routes)
        assert is_feas, f"inter-relocate caused infeasibility: {msg}"


def test_inter_swap_operator(sample_instance):
    rng = random.Random(42)
    routes = clarke_wright_savings(sample_instance)
    new_routes, delta, count, improved = evaluate_inter_swap_candidates(routes, sample_instance, rng=rng)
    assert count >= 0
    if improved:
        assert delta > 0.0
        is_feas, msg = sample_instance.is_feasible(new_routes)
        assert is_feas, f"inter-swap caused infeasibility: {msg}"


def test_best_move_dispatcher(sample_instance):
    rng = random.Random(42)
    routes = clarke_wright_savings(sample_instance)
    for op in range(4):
        _, delta, count, _ = best_move(routes, op, sample_instance, max_candidates=64, rng=rng)
        assert count >= 0
        assert delta >= 0.0

