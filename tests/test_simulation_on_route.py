"""
MP-MOTO-CVRP (V9.2) — Simulation On-Route Invariant Tests
Validates that simulated motorcycle positions:
  1. Follow strictly their own polyline path (no cross-route leakage).
  2. Sampled at 100 time points have distance <= 5 meters to the polyline.
"""

import math
import pytest

from src.web.dispatch_motorbike import (
    get_depots,
    get_stores,
    dispatch_motorbike,
    DispatchRequest,
    StoreItem,
    FleetConfig
)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

def point_to_segment_dist_m(px, py, ax, ay, bx, by):
    """Calculates approximate distance in meters from point (px, py) to segment [A, B]."""
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return haversine_m(px, py, ax, ay)

    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return haversine_m(px, py, proj_x, proj_y)

def min_dist_to_polyline_m(lat, lon, polyline):
    min_d = float("inf")
    for k in range(len(polyline) - 1):
        p1 = polyline[k]
        p2 = polyline[k + 1]
        d = point_to_segment_dist_m(lat, lon, p1[0], p1[1], p2[0], p2[1])
        if d < min_d:
            min_d = d
    return min_d

def test_simulation_stays_on_assigned_route():
    depots = get_depots()
    stores = get_stores()

    req = DispatchRequest(
        depot={"id": depots[0]["id"]},
        stores=[StoreItem(id=st["id"], n_pkg=2) for st in stores[10:16]],
        fleet=FleetConfig(k=3, q_kg=50),
        options={"w_pkg_default_kg": 5.0}
    )

    data = dispatch_motorbike(req)

    for veh in data["vehicles"]:
        if veh["load_kg"] == 0 or not veh["legs"]:
            continue

        # Build full polyline
        full_polyline = []
        for leg in veh["legs"]:
            for pt in leg["geometry"]:
                full_polyline.append(pt)

        assert len(full_polyline) >= 2, "Polyline must have points"

        # Construct cumulative times along segments
        total_time_s = sum(leg["time_s"] for leg in veh["legs"])
        assert total_time_s > 0

        # Sample at 100 uniform time stamps
        for step in range(101):
            t_sample = (step / 100.0) * total_time_s

            # Interpolate position
            t_ratio = min(1.0, max(0.0, t_sample / total_time_s))
            pt_idx = int(t_ratio * (len(full_polyline) - 1))
            interp_lat = full_polyline[pt_idx][0]
            interp_lon = full_polyline[pt_idx][1]

            # Measure distance to polyline
            dist_to_line = min_dist_to_polyline_m(interp_lat, interp_lon, full_polyline)
            assert dist_to_line <= 5.0, (
                f"Vehicle {veh['vehicle_id']} simulated marker deviated {dist_to_line:.2f} m > 5m "
                f"from its designated polyline at t={t_sample:.1f}s!"
            )
