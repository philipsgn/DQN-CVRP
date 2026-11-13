"""
MP-MOTO-CVRP (V9.2) — OR-Tools Guided Local Search Solver
Solves CVRP on asymmetric distance matrices with uniform fleet capacity Q_kg.
Used as an exact/strong heuristic benchmark for validation and performance auditing.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def solve_cvrp_ortools(
    distance_matrix_m: np.ndarray,
    demands_kg: List[float],
    capacity_kg: float,
    num_vehicles: int,
    time_limit_s: float = 3.0,
    depot_index: int = 0
) -> Dict[str, Any]:
    """
    Solves CVRP using OR-Tools RoutingModel on asymmetric cost matrix.
    Args:
        distance_matrix_m: N x N float matrix (meters)
        demands_kg: List of demands in kg for node 0 (depot, demand 0) and nodes 1..N-1
        capacity_kg: Uniform vehicle capacity
        num_vehicles: Number of vehicles available
        time_limit_s: Wall time limit in seconds
        depot_index: Index of depot (default 0)
    """
    start_time = time.time()
    n_nodes = len(demands_kg)

    # Scale demands and capacity to integers
    int_demands = [int(round(d * 10)) for d in demands_kg]
    int_capacity = int(round(capacity_kg * 10))

    # Scale distances to integer meters
    int_dist_matrix = np.round(distance_matrix_m).astype(int).tolist()

    # Create routing index manager
    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demand callback & capacity constraint
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int_demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int_capacity] * num_vehicles,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity"
    )

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(int(time_limit_s))

    solution = routing.SolveWithParameters(search_parameters)
    elapsed = time.time() - start_time

    if not solution:
        return {
            "status": "INFEASIBLE",
            "routes": [],
            "total_distance_m": float("inf"),
            "total_distance_km": float("inf"),
            "k_used": 0,
            "elapsed_s": elapsed
        }

    # Extract routes
    routes = []
    total_dist_m = 0.0

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        route_dist = 0.0
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            if node_idx != depot_index:
                route.append(node_idx)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_dist += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        if route:
            routes.append(route)
            total_dist_m += route_dist

    return {
        "status": "OPTIMAL" if solution else "FEASIBLE",
        "routes": routes,
        "total_distance_m": float(total_dist_m),
        "total_distance_km": round(total_dist_m / 1000.0, 3),
        "k_used": len(routes),
        "elapsed_s": round(elapsed, 3)
    }

