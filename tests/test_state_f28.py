"""Unit tests for the F28 state extractor."""
import numpy as np
from src.cvrp.instance import CVRPInstance
from src.cvrp.clarke_wright import clarke_wright_savings
from src.cvrp.state_f28 import StateExtractorF28


def test_f28_shape_and_range():
    instance = CVRPInstance.generate_synthetic(
        name="TestF28",
        num_customers=18,
        capacity=28,
        max_vehicles=5,
        seed=77
    )
    routes = clarke_wright_savings(instance)
    extractor = StateExtractorF28(feature_patience=30.0)

    # Initial extraction
    s0 = extractor.extract(
        routes, instance, evals_used=0, max_evals=2560, initial_cost=100.0
    )
    assert isinstance(s0, np.ndarray)
    assert s0.shape == (28,)
    assert not np.isnan(s0).any()
    assert not np.isinf(s0).any()

    # Record some steps
    extractor.record_step(operator_id=0, improved=True, delta_cost=5.0)
    extractor.record_step(operator_id=2, improved=False, delta_cost=0.0)

    s1 = extractor.extract(
        routes, instance, evals_used=128, max_evals=2560, initial_cost=100.0
    )
    assert s1.shape == (28,)
    assert s1[12] > 0.0  # op 0 recent success
    assert s1[23] > 0.0  # op 2 streak / patience

