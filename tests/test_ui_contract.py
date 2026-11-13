"""
MP-MOTO-CVRP (V9.2) — API & UI Contract Tests
Validates:
  1. All required fields in /api/dispatch-motorbike response are present and finite.
  2. Zero-Broken-Numbers: No NaN, Infinity, or null in mandatory numeric fields.
  3. Proper HTTP 400 and 422 error structures with clear resolution metadata.
"""

import math
import pytest
from fastapi import HTTPException

from src.web.dispatch_motorbike import (
    get_depots,
    get_stores,
    dispatch_motorbike,
    DispatchRequest,
    StoreItem,
    FleetConfig
)

def test_api_contract_success_schema():
    depots = get_depots()
    stores = get_stores()

    req = DispatchRequest(
        depot={"id": depots[0]["id"]},
        stores=[StoreItem(id=st["id"], n_pkg=2) for st in stores[:4]],
        fleet=FleetConfig(k=2, q_kg=50),
        options={"w_pkg_default_kg": 5.0}
    )

    data = dispatch_motorbike(req)

    # Mandatory root fields
    mandatory_fields = [
        "k_used", "vehicles", "total_distance_km",
        "baseline_cw_distance_km", "saving_vs_cw_pct",
        "makespan_min", "solver", "data_source"
    ]
    for field in mandatory_fields:
        assert field in data, f"Missing mandatory field '{field}' in response"
        val = data[field]
        if isinstance(val, (int, float)):
            assert not math.isnan(val), f"Field '{field}' is NaN"
            assert not math.isinf(val), f"Field '{field}' is Inf"

    # Mandatory vehicle fields
    for veh in data["vehicles"]:
        for v_field in ["vehicle_id", "color", "stops", "load_kg", "load_pkgs", "distance_km", "eta_min", "legs"]:
            assert v_field in veh, f"Missing vehicle field '{v_field}'"
            v_val = veh[v_field]
            if isinstance(v_val, (int, float)):
                assert not math.isnan(v_val), f"Vehicle {veh['vehicle_id']} field '{v_field}' is NaN"
                assert not math.isinf(v_val), f"Vehicle {veh['vehicle_id']} field '{v_field}' is Inf"

def test_api_contract_400_empty_stores():
    req = DispatchRequest(
        depot={"id": "D001"},
        stores=[],
        fleet=FleetConfig(k=3, q_kg=50)
    )
    with pytest.raises(HTTPException) as exc_info:
        dispatch_motorbike(req)
    assert exc_info.value.status_code == 400
    assert "Danh sách cửa hàng giao hàng đang trống" in exc_info.value.detail

def test_api_contract_422_feasibility_rejection():
    stores = get_stores()
    # Request with 10 stores of 10 packages = 500 kg, but only 2 vehicles of 50 kg = 100 kg capacity
    req = DispatchRequest(
        depot={"id": "D001"},
        stores=[StoreItem(id=st["id"], n_pkg=10) for st in stores[:10]],
        fleet=FleetConfig(k=2, q_kg=50)
    )
    with pytest.raises(HTTPException) as exc_info:
        dispatch_motorbike(req)
    assert exc_info.value.status_code == 422
    err = exc_info.value.detail
    assert err["is_feasible"] is False
    assert err["code"] == "TOTAL_EXCEEDS_CAPACITY"
    assert "metadata" in err
    assert err["metadata"]["k_min_by_weight"] >= 10

def test_api_contract_snap_endpoint():
    from src.web.dispatch_motorbike import snap_coordinate, SnapRequest

    # Valid coordinate in central HCMC (near Ben Thanh)
    res = snap_coordinate(SnapRequest(lat=10.7725, lon=106.6980))
    assert "snapped_node_id" in res
    assert res["snap_distance_m"] <= 150.0

    # Out-of-bounds coordinate (outside HCMC)
    with pytest.raises(HTTPException) as exc_info:
        snap_coordinate(SnapRequest(lat=21.0285, lon=105.8542))
    assert exc_info.value.status_code == 422
    err = exc_info.value.detail
    assert err["code"] == "SNAP_OUT_OF_BOUNDS"
    assert "150m" in err["reason"]
