"""Local Search Operators for CVRP:
0: 2-opt (intra-route segment reversal)
1: intra-relocate (intra-route node shift)
2: inter-relocate (cross-route customer shift with capacity <= Q check)
3: inter-swap (cross-route customer swap with capacity <= Q check)

Candidate evaluation follows the report:
- Subsample up to 128 candidates without replacement.
- Strictly accepts the best improving move (Delta C > 0).
"""
import copy
import random
from typing import List, Tuple, Optional
from .instance import CVRPInstance

OPERATOR_NAMES = {
    0: "2-opt",
    1: "intra-relocate",
    2: "inter-relocate",
    3: "inter-swap"
}


def evaluate_2opt_candidates(
    routes: List[List[int]],
    instance: CVRPInstance,
    max_candidates: int = 128,
    rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], float, int, bool]:
    """Action 0: 2-opt intra-route segment reversal."""
    if rng is None:
        rng = random.Random()

    candidates = []
    for r_idx, route in enumerate(routes):
        m = len(route)
        if m < 4:  # At least 2 customers [0, c1, c2, 0] needed for 2-opt
            continue
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                candidates.append((r_idx, i, j))

    if not candidates:
        return routes, 0.0, 0, False

    num_evals = min(len(candidates), max_candidates)
    sampled = rng.sample(candidates, num_evals) if len(candidates) > max_candidates else candidates

    best_delta = 0.0
    best_move_params = None

    dist = instance.distance_matrix
    for r_idx, i, j in sampled:
        route = routes[r_idx]
        u = route[i - 1]
        v = route[i]
        w = route[j]
        x = route[j + 1]

        # Cost difference: (old edges) - (new edges)
        # Note: for symmetric distance, reversing segment (v..w) keeps internal segment cost same
        old_cost = dist[u, v] + dist[w, x]
        new_cost = dist[u, w] + dist[v, x]
        delta = old_cost - new_cost

        if delta > best_delta:
            best_delta = delta
            best_move_params = (r_idx, i, j)

    if best_delta > 1e-6 and best_move_params is not None:
        r_idx, i, j = best_move_params
        new_routes = [list(r) for r in routes]
        route = new_routes[r_idx]
        route[i:j + 1] = list(reversed(route[i:j + 1]))
        return new_routes, best_delta, len(sampled), True

    return routes, 0.0, len(sampled), False


def evaluate_intra_relocate_candidates(
    routes: List[List[int]],
    instance: CVRPInstance,
    max_candidates: int = 128,
    rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], float, int, bool]:
    """Action 1: intra-relocate node repositioning in the same route."""
    if rng is None:
        rng = random.Random()

    candidates = []
    for r_idx, route in enumerate(routes):
        m = len(route)
        if m < 4:
            continue
        for i in range(1, m - 1):
            for j in range(1, m - 1):
                if i != j:
                    candidates.append((r_idx, i, j))

    if not candidates:
        return routes, 0.0, 0, False

    num_evals = min(len(candidates), max_candidates)
    sampled = rng.sample(candidates, num_evals) if len(candidates) > max_candidates else candidates

    best_delta = 0.0
    best_move_params = None

    for r_idx, i, j in sampled:
        route = routes[r_idx]
        cur_cost = instance.route_distance(route)
        # Create candidate route
        new_route = list(route)
        node = new_route.pop(i)
        new_route.insert(j, node)
        new_cost = instance.route_distance(new_route)
        delta = cur_cost - new_cost

        if delta > best_delta:
            best_delta = delta
            best_move_params = (r_idx, i, j)

    if best_delta > 1e-6 and best_move_params is not None:
        r_idx, i, j = best_move_params
        new_routes = [list(r) for r in routes]
        node = new_routes[r_idx].pop(i)
        new_routes[r_idx].insert(j, node)
        return new_routes, best_delta, len(sampled), True

    return routes, 0.0, len(sampled), False


def evaluate_inter_relocate_candidates(
    routes: List[List[int]],
    instance: CVRPInstance,
    max_candidates: int = 128,
    rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], float, int, bool]:
    """Action 2: inter-relocate node shift from route A to route B (capacity <= Q)."""
    if rng is None:
        rng = random.Random()

    candidates = []
    demands = instance.demands
    capacity = instance.capacity

    active_routes = [idx for idx, r in enumerate(routes) if len(r) > 2]
    if len(active_routes) < 2:
        return routes, 0.0, 0, False

    route_demands = {idx: instance.route_demand(routes[idx]) for idx in active_routes}

    for r_from in active_routes:
        for r_to in active_routes:
            if r_from == r_to:
                continue
            r_from_len = len(routes[r_from])
            r_to_len = len(routes[r_to])
            for i in range(1, r_from_len - 1):
                node = routes[r_from][i]
                node_dem = demands[node]
                # Fast capacity check
                if route_demands[r_to] + node_dem <= capacity:
                    for j in range(1, r_to_len):
                        candidates.append((r_from, r_to, i, j))

    if not candidates:
        return routes, 0.0, 0, False

    num_evals = min(len(candidates), max_candidates)
    sampled = rng.sample(candidates, num_evals) if len(candidates) > max_candidates else candidates

    best_delta = 0.0
    best_move_params = None

    for r_from, r_to, i, j in sampled:
        rf = routes[r_from]
        rt = routes[r_to]
        cur_cost = instance.route_distance(rf) + instance.route_distance(rt)

        new_rf = list(rf)
        new_rt = list(rt)
        node = new_rf.pop(i)
        new_rt.insert(j, node)

        new_cost = instance.route_distance(new_rf) + instance.route_distance(new_rt)
        delta = cur_cost - new_cost

        if delta > best_delta:
            best_delta = delta
            best_move_params = (r_from, r_to, i, j)

    if best_delta > 1e-6 and best_move_params is not None:
        r_from, r_to, i, j = best_move_params
        new_routes = [list(r) for r in routes]
        node = new_routes[r_from].pop(i)
        new_routes[r_to].insert(j, node)
        # Clean up route if emptied
        new_routes = [r for r in new_routes if len(r) > 2]
        return new_routes, best_delta, len(sampled), True

    return routes, 0.0, len(sampled), False


def evaluate_inter_swap_candidates(
    routes: List[List[int]],
    instance: CVRPInstance,
    max_candidates: int = 128,
    rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], float, int, bool]:
    """Action 3: inter-swap swap customers between route A and route B (both capacities <= Q)."""
    if rng is None:
        rng = random.Random()

    candidates = []
    demands = instance.demands
    capacity = instance.capacity

    active_routes = [idx for idx, r in enumerate(routes) if len(r) > 2]
    if len(active_routes) < 2:
        return routes, 0.0, 0, False

    route_demands = {idx: instance.route_demand(routes[idx]) for idx in active_routes}

    for idx_a in range(len(active_routes)):
        r_a = active_routes[idx_a]
        for idx_b in range(idx_a + 1, len(active_routes)):
            r_b = active_routes[idx_b]
            for i in range(1, len(routes[r_a]) - 1):
                node_a = routes[r_a][i]
                dem_a = demands[node_a]
                for j in range(1, len(routes[r_b]) - 1):
                    node_b = routes[r_b][j]
                    dem_b = demands[node_b]

                    # Capacity checks for both routes
                    if (route_demands[r_a] - dem_a + dem_b <= capacity and
                            route_demands[r_b] - dem_b + dem_a <= capacity):
                        candidates.append((r_a, r_b, i, j))

    if not candidates:
        return routes, 0.0, 0, False

    num_evals = min(len(candidates), max_candidates)
    sampled = rng.sample(candidates, num_evals) if len(candidates) > max_candidates else candidates

    best_delta = 0.0
    best_move_params = None

    for r_a, r_b, i, j in sampled:
        ra = routes[r_a]
        rb = routes[r_b]
        cur_cost = instance.route_distance(ra) + instance.route_distance(rb)

        new_ra = list(ra)
        new_rb = list(rb)
        node_a = new_ra[i]
        node_b = new_rb[j]
        new_ra[i] = node_b
        new_rb[j] = node_a

        new_cost = instance.route_distance(new_ra) + instance.route_distance(new_rb)
        delta = cur_cost - new_cost

        if delta > best_delta:
            best_delta = delta
            best_move_params = (r_a, r_b, i, j)

    if best_delta > 1e-6 and best_move_params is not None:
        r_a, r_b, i, j = best_move_params
        new_routes = [list(r) for r in routes]
        node_a = new_routes[r_a][i]
        node_b = new_routes[r_b][j]
        new_routes[r_a][i] = node_b
        new_routes[r_b][j] = node_a
        return new_routes, best_delta, len(sampled), True

    return routes, 0.0, len(sampled), False


def best_move(
    routes: List[List[int]],
    operator_id: int,
    instance: CVRPInstance,
    max_candidates: int = 128,
    rng: Optional[random.Random] = None
) -> Tuple[List[List[int]], float, int, bool]:
    """Execute operator on routes and return best improving move.
    
    Returns:
        new_routes: Updated routes (or unchanged if no improving move)
        delta_cost: Improvement in cost (>= 0)
        num_candidates: Number of candidates evaluated
        improved: True if an improving move was found and applied
    """
    if operator_id == 0:
        return evaluate_2opt_candidates(routes, instance, max_candidates, rng)
    elif operator_id == 1:
        return evaluate_intra_relocate_candidates(routes, instance, max_candidates, rng)
    elif operator_id == 2:
        return evaluate_inter_relocate_candidates(routes, instance, max_candidates, rng)
    elif operator_id == 3:
        return evaluate_inter_swap_candidates(routes, instance, max_candidates, rng)
    else:
        raise ValueError(f"Unknown operator ID: {operator_id}")

