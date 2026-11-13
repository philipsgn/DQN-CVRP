"""Independent Solution Validator for CVRP.
Re-computes distances and rigorously checks all constraints:
1. Every customer visited exactly once.
2. Route starts and ends at Depot (0).
3. No route exceeds capacity Q.
4. Total vehicles <= K.
"""
from typing import Dict, List, Tuple
import numpy as np

from ..cvrp.instance import CVRPInstance, haversine_distance


class IndependentValidator:
    """Verifies feasibility and independently recomputes solution distance."""

    @staticmethod
    def validate(instance: CVRPInstance, routes: List[List[int]]) -> Dict[str, any]:
        """Perform comprehensive independent validation."""
        active_routes = [r for r in routes if len(r) > 2]
        n_cust = instance.num_customers
        capacity = instance.capacity
        k_max = instance.max_vehicles
        coords = instance.coords
        demands = instance.demands

        errors = []
        visited = []

        total_recomputed_dist = 0.0

        for idx, route in enumerate(active_routes):
            # Check endpoints
            if route[0] != 0 or route[-1] != 0:
                errors.append(f"Tuyến {idx} không xuất phát hoặc kết thúc tại Depot: {route}")

            # Check capacity
            route_demand = sum(demands[node] for node in route if node != 0)
            if route_demand > capacity:
                errors.append(f"Tuyến {idx} quá tải: {route_demand} > {capacity}")

            # Collect customers
            for node in route[1:-1]:
                visited.append(node)

            # Recompute distance from scratch
            route_dist = 0.0
            for i in range(len(route) - 1):
                u = route[i]
                v = route[i + 1]
                if instance.is_geo:
                    d = haversine_distance(coords[u, 0], coords[u, 1], coords[v, 0], coords[v, 1])
                else:
                    d = float(np.linalg.norm(coords[u] - coords[v]))
                route_dist += d

            total_recomputed_dist += route_dist

        # Check vehicle count
        if len(active_routes) > k_max:
            errors.append(f"Số xe vượt quá K: {len(active_routes)} > {k_max}")

        # Check customer visits
        unique_visited = set(visited)
        if len(visited) != len(unique_visited):
            errors.append("Có khách hàng bị trùng lặp trên các tuyến")

        expected_customers = set(range(1, n_cust + 1))
        missing = expected_customers - unique_visited
        if missing:
            errors.append(f"Có khách hàng chưa được phục vụ: {sorted(list(missing))}")

        is_valid = len(errors) == 0

        return {
            "is_valid": is_valid,
            "errors": errors,
            "num_active_routes": len(active_routes),
            "max_vehicles": k_max,
            "total_customers_served": len(unique_visited),
            "total_customers_expected": n_cust,
            "recomputed_cost": float(total_recomputed_dist),
            "status": "PASS" if is_valid else "FAIL"
        }

