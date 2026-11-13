"""Clarke-Wright Savings Algorithm for Initial CVRP Solution.
Produces a 100% capacity-feasible set of routes.
"""
from typing import List, Tuple
from .instance import CVRPInstance


def clarke_wright_savings(instance: CVRPInstance, k_max: int | None = None) -> List[List[int]]:
    """Compute the initial CVRP solution using the Clarke-Wright Savings heuristic, enforcing k_max."""
    n = instance.num_customers
    dist = instance.distance_matrix
    demands = instance.demands
    capacity = instance.capacity

    # Initial solution: each customer has its own route [0, i, 0]
    routes: List[List[int]] = [[0, i, 0] for i in range(1, n + 1)]

    # Compute savings for all pairs (i, j) where i != j
    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                s = dist[i, 0] + dist[0, j] - dist[i, j]
                savings.append((float(s), i, j))

    # Sort savings in descending order
    savings.sort(key=lambda x: x[0], reverse=True)

    # Helper: find which route contains customer node
    def find_route(node: int) -> int:
        for r_idx, r in enumerate(routes):
            if node in r:
                return r_idx
        raise ValueError(f"Node {node} not found in any route.")

    # Try merging routes based on savings
    for s_val, i, j in savings:
        r1_idx = find_route(i)
        r2_idx = find_route(j)

        if r1_idx == r2_idx:
            continue  # Already in same route

        r1 = routes[r1_idx]
        r2 = routes[r2_idx]

        # Check combined capacity
        dem1 = sum(demands[u] for u in r1[1:-1])
        dem2 = sum(demands[v] for v in r2[1:-1])
        if dem1 + dem2 > capacity:
            continue

        # Check merge orientations
        merged: List[int] | None = None
        # Case 1: i is end of r1, j is start of r2
        if r1[-2] == i and r2[1] == j:
            merged = r1[:-1] + r2[1:]
        # Case 2: j is end of r2, i is start of r1
        elif r2[-2] == j and r1[1] == i:
            merged = r2[:-1] + r1[1:]
        # Case 3: i is end of r1, j is end of r2 (reverse r2)
        elif r1[-2] == i and r2[-2] == j:
            rev_r2 = [0] + list(reversed(r2[1:-1])) + [0]
            merged = r1[:-1] + rev_r2[1:]
        # Case 4: i is start of r1, j is start of r2 (reverse r1)
        elif r1[1] == i and r2[1] == j:
            rev_r1 = [0] + list(reversed(r1[1:-1])) + [0]
            merged = rev_r1[:-1] + r2[1:]

        if merged is not None:
            # Replace r1 with merged, remove r2
            routes[r1_idx] = merged
            routes.pop(r2_idx)

    # Clean up any empty routes
    clean_routes = [r for r in routes if len(r) > 2]

    # Enforce k_max if provided
    if k_max is not None and len(clean_routes) > k_max:
        max_attempts = len(clean_routes) * 3
        attempts = 0
        while len(clean_routes) > k_max and attempts < max_attempts:
            attempts += 1
            loads = [sum(demands[u] for u in r[1:-1]) for r in clean_routes]
            sorted_idx = sorted(range(len(clean_routes)), key=lambda x: loads[x])
            consolidated = False
            for target_idx in sorted_idx:
                target_route = clean_routes[target_idx]
                customers_to_move = target_route[1:-1]
                remaining_routes = [r for idx, r in enumerate(clean_routes) if idx != target_idx]
                rem_loads = [sum(demands[u] for u in r[1:-1]) for r in remaining_routes]
                can_fit_all = True
                temp_routes = [list(r) for r in remaining_routes]
                temp_loads = list(rem_loads)
                sorted_custs = sorted(customers_to_move, key=lambda c: demands[c], reverse=True)
                for u in sorted_custs:
                    u_demand = demands[u]
                    best_cost_inc = float("inf")
                    best_r_idx = -1
                    best_pos = -1
                    for r_idx, r in enumerate(temp_routes):
                        if temp_loads[r_idx] + u_demand <= capacity + 1e-5:
                            for pos in range(1, len(r)):
                                pred = r[pos - 1]
                                succ = r[pos]
                                cost_inc = dist[pred, u] + dist[u, succ] - dist[pred, succ]
                                if cost_inc < best_cost_inc:
                                    best_cost_inc = cost_inc
                                    best_r_idx = r_idx
                                    best_pos = pos
                    if best_r_idx != -1:
                        temp_routes[best_r_idx].insert(best_pos, u)
                        temp_loads[best_r_idx] += u_demand
                    else:
                        can_fit_all = False
                        break
                if can_fit_all:
                    clean_routes = temp_routes
                    consolidated = True
                    break
            if not consolidated:
                break

    return clean_routes

