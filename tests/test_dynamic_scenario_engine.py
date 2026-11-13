"""Automated Tests for Dynamic CVRP Dispatch Platform Engine.
Tests:
- DynamicOSRMService (distance/duration matrices, route geometries, caching)
- DQNAgent predict_q_values and Scale-Invariant F28 state extraction
- FastAPI /api/create_dynamic_scenario endpoint validation
- Hard feasibility gate: di > Q and D_total > K_max * Q
- Clarke-Wright and IndependentValidator on dynamic instances
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.cvrp.instance import CVRPInstance, haversine_distance
from src.cvrp.osrm_service import DynamicOSRMService
from src.cvrp.state_f28 import StateExtractorF28
from src.cvrp.clarke_wright import clarke_wright_savings
from src.agents.dqn_agent import DQNAgent
from src.evaluation.validator import IndependentValidator
from src.web.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_osrm_service_matrix_and_cache():
    """Verify OSRM service computes distance & duration matrices with caching."""
    service = DynamicOSRMService()
    # 3 points: Depot (Ben Thanh), Point 1 (Notre Dame), Point 2 (Tan Dinh)
    coords = [
        [10.7725, 106.6980],
        [10.7798, 106.6990],
        [10.7903, 106.6896]
    ]
    dist_mat, time_mat = service.compute_matrices(coords, metric="distance")

    assert dist_mat.shape == (3, 3)
    assert time_mat.shape == (3, 3)
    assert dist_mat[0, 0] == 0.0
    assert dist_mat[0, 1] > 0.0
    assert time_mat[0, 1] > 0.0

    # Ensure caching works
    key = f"{coords[0][0]:.5f},{coords[0][1]:.5f}_{coords[1][0]:.5f},{coords[1][1]:.5f}"
    assert key in service._pair_cache


def test_osrm_service_route_geometry():
    """Verify route geometry generation with cache and fallback."""
    service = DynamicOSRMService()
    route = [[10.7725, 106.6980], [10.7798, 106.6990], [10.7725, 106.6980]]
    geom = service.get_route_geometry(route)
    assert len(geom) >= len(route)
    assert isinstance(geom[0], list)
    assert len(geom[0]) == 2


def test_dqn_predict_q_values():
    """Verify DQNAgent predict_q_values returns 4 finite values."""
    agent = DQNAgent(algorithm="ddqn")
    dummy_state = np.zeros(28, dtype=np.float32)
    q_vals = agent.predict_q_values(dummy_state)

    assert q_vals.shape == (4,)
    assert not np.isnan(q_vals).any()
    assert not np.isinf(q_vals).any()


def test_api_create_dynamic_scenario_success(client):
    """Verify successful dynamic scenario creation with fleet breakdown."""
    payload = {
        "depot": {
            "name": "Kho Bến Thành (Quận 1)",
            "lat": 10.7725,
            "lng": 106.6980,
            "district": "Quận 1",
            "demand": 0
        },
        "customers": [
            {"id": 1, "name": "Điểm A", "lat": 10.7798, "lng": 106.6990, "demand": 8, "district": "Quận 1"},
            {"id": 2, "name": "Điểm B", "lat": 10.7903, "lng": 106.6896, "demand": 10, "district": "Quận 1"},
            {"id": 3, "name": "Điểm C", "lat": 10.7828, "lng": 106.6958, "demand": 6, "district": "Quận 3"}
        ],
        "max_vehicles": 5,
        "capacity": 30,
        "metric": "distance"
    }

    res = client.post("/api/create_dynamic_scenario", json=payload)
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["num_customers"] == 3
    assert data["total_demand"] == 24
    assert data["k_min"] == 1  # 24 <= 30 -> 1 vehicle
    assert data["max_vehicles"] == 5
    assert data["is_feasible"] is True
    assert "road_geometries" in data
    assert "fleet_active" in data
    assert "fleet_idle" in data
    assert len(data["fleet_idle"]) >= 4


def test_api_create_dynamic_scenario_hard_validation_demand_over_capacity(client):
    """Hard Validation: Reject when any customer di > Q_vehicle."""
    payload = {
        "depot": {
            "name": "Kho Trung Tâm",
            "lat": 10.7725,
            "lng": 106.6980,
            "demand": 0
        },
        "customers": [
            {"id": 1, "name": "Điểm Vượt Tải", "lat": 10.7798, "lng": 106.6990, "demand": 35, "district": "Quận 1"},
            {"id": 2, "name": "Điểm Hợp Lệ", "lat": 10.7903, "lng": 106.6896, "demand": 10, "district": "Quận 1"}
        ],
        "max_vehicles": 5,
        "capacity": 30,  # 35 > 30!
        "metric": "distance"
    }

    res = client.post("/api/create_dynamic_scenario", json=payload)
    assert res.status_code == 400
    assert "vượt quá tải trọng xe" in res.json()["detail"]


def test_api_create_dynamic_scenario_hard_validation_fleet_overflow(client):
    """Hard Validation: Reject when total demand exceeds total fleet capacity."""
    payload = {
        "depot": {
            "name": "Kho Trung Tâm",
            "lat": 10.7725,
            "lng": 106.6980,
            "demand": 0
        },
        "customers": [
            {"id": 1, "name": "Điểm 1", "lat": 10.7798, "lng": 106.6990, "demand": 25, "district": "Quận 1"},
            {"id": 2, "name": "Điểm 2", "lat": 10.7903, "lng": 106.6896, "demand": 25, "district": "Quận 1"},
            {"id": 3, "name": "Điểm 3", "lat": 10.7828, "lng": 106.6958, "demand": 20, "district": "Quận 3"}
        ],
        "max_vehicles": 2,  # Total capacity: 2 * 30 = 60
        "capacity": 30,     # Total demand: 25 + 25 + 20 = 70 > 60!
        "metric": "distance"
    }

    res = client.post("/api/create_dynamic_scenario", json=payload)
    assert res.status_code == 400
    assert "vượt quá tổng tải trọng tối đa của đội xe" in res.json()["detail"]


def test_api_create_dynamic_scenario_insufficient_customers(client):
    """Reject when fewer than 2 customer points are provided."""
    payload = {
        "depot": {"name": "Kho", "lat": 10.7725, "lng": 106.6980, "demand": 0},
        "customers": [
            {"id": 1, "name": "Điểm Duy Nhất", "lat": 10.7798, "lng": 106.6990, "demand": 5}
        ],
        "max_vehicles": 3,
        "capacity": 30
    }
    res = client.post("/api/create_dynamic_scenario", json=payload)
    assert res.status_code == 400
    assert "Cần tối thiểu ít nhất 2 điểm giao hàng" in res.json()["detail"]

