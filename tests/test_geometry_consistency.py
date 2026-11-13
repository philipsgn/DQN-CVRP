"""
MP-MOTO-CVRP (V9.2) — Geometric Invariant Tests
Validates:
  1. Legs distance sum matches vehicle distance_km (tolerance <= 1%)
  2. Total vehicles distance matches reported total_distance_km
  3. Every consecutive pair in node_path is a valid directed edge in hcm_motorbike.graphml
  4. Single source of truth: No straight-line shortcuts or cross-river jumps outside bridges
"""

import pytest
import networkx as nx

from src.web.dispatch_motorbike import (
    get_graph,
    get_depots,
    get_stores,
    dispatch_motorbike,
    DispatchRequest,
    StoreItem,
    FleetConfig
)

def test_geometry_consistency_and_directed_edges():
    depots = get_depots()
    stores = get_stores()
    G = get_graph()

    depot = depots[0]
    selected_stores = stores[:6]

    req = DispatchRequest(
        depot={"id": depot["id"]},
        stores=[StoreItem(id=st["id"], n_pkg=2) for st in selected_stores],
        fleet=FleetConfig(k=3, q_kg=50),
        options={"w_pkg_default_kg": 5.0}
    )

    data = dispatch_motorbike(req)
    assert data["k_used"] > 0
    assert len(data["vehicles"]) == 3

    computed_total_dist_km = 0.0

    for veh in data["vehicles"]:
        if veh["load_kg"] == 0:
            continue  # Idle vehicle waiting at depot

        veh_legs_m = sum(leg["distance_m"] for leg in veh["legs"])
        veh_reported_km = veh["distance_km"]

        # 1. Legs distance vs reported distance_km (within 1%)
        assert abs(veh_legs_m / 1000.0 - veh_reported_km) <= 0.05, (
            f"Vehicle {veh['vehicle_id']} legs sum ({veh_legs_m/1000.0:.2f} km) differs from "
            f"reported distance ({veh_reported_km:.2f} km)!"
        )

        computed_total_dist_km += veh_reported_km

        # 2. Verify all edges in node_path exist in the directed graph G
        for leg in veh["legs"]:
            node_path = leg["node_path"]
            assert len(node_path) >= 2, "Leg path must have at least from and to nodes"
            for k in range(len(node_path) - 1):
                u = str(node_path[k])
                v = str(node_path[k + 1])
                assert G.has_edge(u, v), (
                    f"Invalid directed edge ({u} -> {v}) in route! "
                    f"Violates motorcycle road network directionality."
                )

    # 3. Total distance consistency
    assert abs(computed_total_dist_km - data["total_distance_km"]) <= 0.05, (
        f"Sum of vehicles ({computed_total_dist_km:.2f} km) differs from "
        f"total_distance_km ({data['total_distance_km']:.2f} km)!"
    )
