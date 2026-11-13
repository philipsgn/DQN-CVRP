"""CVRP Instance Representation and Distance Computations.
Supports both real GPS coordinates (Haversine formula in km) and synthetic 2D coordinates.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


@dataclass
class CVRPInstance:
    """Represents a Capacitated Vehicle Routing Problem (CVRP) instance."""
    name: str
    num_customers: int
    capacity: int
    max_vehicles: int
    coords: np.ndarray  # Shape: (num_customers + 1, 2) -> (lat, lng) or (x, y)
    demands: np.ndarray  # Shape: (num_customers + 1,)
    distance_matrix: np.ndarray  # Shape: (num_customers + 1, num_customers + 1)
    node_names: List[str] = field(default_factory=list)
    is_geo: bool = True  # True if coordinates are GPS (lat, lng)

    @classmethod
    def from_json(cls, file_path: str | Path) -> "CVRPInstance":
        """Load an instance from a JSON file (e.g. HCMC instances)."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        name = data.get("name", "Unnamed")
        capacity = int(data.get("capacity", 30))
        max_vehicles = int(data.get("max_vehicles", 5))

        depot = data["depot"]
        customers = data["customers"]
        num_customers = len(customers)

        node_names = [depot.get("name", "Depot 0")]
        coords_list = [[float(depot["lat"]), float(depot["lng"])]]
        demands_list = [0]

        for cust in customers:
            node_names.append(cust.get("name", f"Customer {cust['id']}"))
            coords_list.append([float(cust["lat"]), float(cust["lng"])])
            demands_list.append(int(cust["demand"]))

        coords = np.array(coords_list, dtype=np.float64)
        demands = np.array(demands_list, dtype=np.int32)

        # Compute distance matrix using Haversine
        n_nodes = num_customers + 1
        dist_mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist_mat[i, j] = haversine_distance(
                        coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1]
                    )

        return cls(
            name=name,
            num_customers=num_customers,
            capacity=capacity,
            max_vehicles=max_vehicles,
            coords=coords,
            demands=demands,
            distance_matrix=dist_mat,
            node_names=node_names,
            is_geo=True
        )

    @classmethod
    def generate_synthetic(
        cls,
        name: str = "Synthetic-24",
        num_customers: int = 24,
        capacity: int = 30,
        max_vehicles: int = 5,
        distribution: str = "uniform",
        seed: Optional[int] = 42
    ) -> "CVRPInstance":
        """Generate synthetic instance as described in the benchmark study (n=20..50)."""
        rng = np.random.default_rng(seed)
        
        # Depot at (50, 50)
        coords = np.zeros((num_customers + 1, 2), dtype=np.float64)
        coords[0] = [50.0, 50.0]
        
        if distribution == "clustered":
            # 3 clusters
            cluster_centers = np.array([[25.0, 25.0], [75.0, 25.0], [50.0, 80.0]])
            assignments = rng.integers(0, 3, size=num_customers)
            offsets = rng.normal(0, 8.0, size=(num_customers, 2))
            coords[1:] = np.clip(cluster_centers[assignments] + offsets, 0.0, 100.0)
        else:
            # Uniform in [0, 100] x [0, 100]
            coords[1:] = rng.uniform(0.0, 100.0, size=(num_customers, 2))

        # Demands in [1, 9]
        demands = np.zeros(num_customers + 1, dtype=np.int32)
        demands[0] = 0
        demands[1:] = rng.integers(1, 10, size=num_customers)

        # Euclidean distances
        n_nodes = num_customers + 1
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))

        node_names = ["Depot 0"] + [f"Cust {i}" for i in range(1, num_customers + 1)]

        return cls(
            name=name,
            num_customers=num_customers,
            capacity=capacity,
            max_vehicles=max_vehicles,
            coords=coords,
            demands=demands,
            distance_matrix=dist_mat,
            node_names=node_names,
            is_geo=False
        )

    def route_distance(self, route: List[int]) -> float:
        """Compute the total travel distance of a single route [0, c1, ..., cp, 0]."""
        if len(route) <= 2:
            return 0.0
        dist = 0.0
        for i in range(len(route) - 1):
            dist += self.distance_matrix[route[i], route[i + 1]]
        return float(dist)

    def route_demand(self, route: List[int]) -> int:
        """Compute total demand of customers on this route."""
        return int(sum(self.demands[node] for node in route if node != 0))

    def solution_cost(self, routes: List[List[int]]) -> float:
        """Compute total distance of all active routes."""
        return float(sum(self.route_distance(r) for r in routes if len(r) > 2))

    def is_feasible(self, routes: List[List[int]]) -> Tuple[bool, str]:
        """Check strict feasibility: demand <= Q, vehicle count <= K, all visited once."""
        active_routes = [r for r in routes if len(r) > 2]
        if len(active_routes) > self.max_vehicles:
            return False, f"Vượt quá số xe tối đa: {len(active_routes)} > {self.max_vehicles}"

        visited = set()
        for idx, r in enumerate(active_routes):
            if r[0] != 0 or r[-1] != 0:
                return False, f"Tuyến {idx} không bắt đầu và kết thúc tại Depot: {r}"
            d = self.route_demand(r)
            if d > self.capacity:
                return False, f"Tuyến {idx} vượt tải trọng: {d} > {self.capacity}"
            for node in r[1:-1]:
                if node in visited:
                    return False, f"Khách hàng {node} bị phục vụ nhiều lần"
                visited.add(node)

        if len(visited) != self.num_customers:
            missing = set(range(1, self.num_customers + 1)) - visited
            return False, f"Thiếu khách hàng chưa được phục vụ: {missing}"

        return True, "Hợp lệ 100%"

