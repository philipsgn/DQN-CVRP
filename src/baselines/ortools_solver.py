"""Google OR-Tools CVRP solver — the strong classical benchmark.

Solves the capacitated VRP over a given (possibly traffic-adjusted)
distance matrix and returns depot-free routes. Used two ways:

* static: solve once on the true matrix, execute the plan as-is;
* replan: re-solve on the current traffic-adjusted matrix each time the
  dynamic environment re-samples conditions (cost of re-solving is
  recorded by the caller).
"""

from __future__ import annotations

import re
import time

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from ..envs.instances import CVRPInstance

# Extra vehicles offered to the solver beyond each instance's best-known
# fleet size k, so it can trade an extra (near-empty) route for a shorter
# overall solution if that's cheaper — pure slack, not a capacity constraint.
FLEET_SLACK = 2


def solve(
    instance: CVRPInstance,
    dist: np.ndarray | None = None,
    time_limit_s: int = 10,
    serve_mask: np.ndarray | None = None,
    start_node: int = 0,
    start_capacity: int | None = None,
    hint: list[list[int]] | None = None,
) -> tuple[list[list[int]], float, float]:
    """Solve a CVRP; returns (routes, cost, wall_time_s).

    ``serve_mask``/``start_node``/``start_capacity`` support mid-episode
    re-planning in the dynamic environment: only unserved customers are
    routed and the first vehicle starts from the current position with
    the current remaining capacity.

    ``hint`` warm-starts the search from a known-feasible plan (same
    format as the returned routes: hint[0] is vehicle 0's continuation,
    possibly empty). Guarantees the result is no worse than the hint.
    """
    if dist is None:
        dist = instance.dist
    demands = instance.demands
    n_nodes = len(demands)
    if serve_mask is None:
        serve_mask = np.ones(n_nodes, dtype=bool)
        serve_mask[0] = False

    # nodes in the reduced model: depot + unserved customers (+ current pos)
    customers = [i for i in range(1, n_nodes) if serve_mask[i]]
    if not customers:
        return [], 0.0, 0.0
    nodes = [0, *customers]
    virtual_start = None
    if start_node != 0 and start_node not in nodes:
        nodes.append(start_node)
        virtual_start = len(nodes) - 1
    elif start_node != 0:
        virtual_start = nodes.index(start_node)

    k = int(re.search(r"-k(\d+)", instance.name).group(1))
    n_vehicles = k + FLEET_SLACK

    idx_of = {node: i for i, node in enumerate(nodes)}
    starts = [idx_of[0]] * n_vehicles
    if virtual_start is not None:
        starts[0] = virtual_start
    ends = [idx_of[0]] * n_vehicles

    def attempt(strategy, warm_routes=None):
        """Build a fresh model and solve with the given first-solution
        strategy (or from a warm-start assignment). A RoutingModel must not
        be reused across solves with different parameters — it is closed by
        the first solve and later calls inherit its state."""
        manager = pywrapcp.RoutingIndexManager(len(nodes), n_vehicles, starts, ends)
        routing = pywrapcp.RoutingModel(manager)

        def dist_cb(i, j):
            return int(dist[nodes[manager.IndexToNode(i)], nodes[manager.IndexToNode(j)]])

        transit = routing.RegisterTransitCallback(dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        def demand_cb(i):
            node = nodes[manager.IndexToNode(i)]
            return int(demands[node]) if serve_mask[node] else 0

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        caps = [int(instance.capacity)] * n_vehicles
        if start_capacity is not None:
            caps[0] = int(start_capacity)
        routing.AddDimensionWithVehicleCapacity(demand_idx, 0, caps, True, "Capacity")

        if virtual_start is not None:
            # OR-Tools prices an unused vehicle's start->end arc as free, so
            # without this the model thinks abandoning the current position
            # (park vehicle 0, refill, restart from depot) costs nothing,
            # while reality charges d(position, depot). That mispricing makes
            # re-solves systematically over-prefer park-and-refill plans.
            routing.SetVehicleUsedWhenEmpty(True, 0)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = strategy
        params.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        params.time_limit.FromSeconds(time_limit_s)
        params.log_search = False
        if warm_routes is not None:
            # pad/trim to fleet size; vehicle 0 keeps the continuation route
            per_vehicle = [[idx_of[n] for n in r] for r in warm_routes]
            per_vehicle += [[] for _ in range(n_vehicles - len(per_vehicle))]
            routing.CloseModelWithParameters(params)
            initial = routing.ReadAssignmentFromRoutes(per_vehicle[:n_vehicles], True)
            if initial is None:
                return None  # hint infeasible in this model; skip
            solution = routing.SolveFromAssignmentWithParameters(initial, params)
        else:
            solution = routing.SolveWithParameters(params)
        if solution is None:
            return None

        routes, cost = [], 0.0
        for v in range(n_vehicles):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                node = nodes[manager.IndexToNode(index)]
                if node != 0 and serve_mask[node]:
                    route.append(node)
                prev, index = index, solution.Value(routing.NextVar(index))
                cost += dist[nodes[manager.IndexToNode(prev)],
                             nodes[manager.IndexToNode(index)]]
            # When re-solving from a mid-route position, always keep vehicle
            # 0's continuation route — even empty. An empty first route means
            # the solver decided to return to depot (refill) before serving
            # anyone, and dropping it would make the caller execute a
            # depot-planned, full-capacity route from the current position
            # and capacity.
            if route or (v == 0 and virtual_start is not None):
                routes.append(route)
        return routes, float(cost)

    # Mid-route re-solves (virtual start + partial capacity) are prone to
    # first-solution-dependent local optima that GLS does not escape within
    # the time budget: PATH_CHEAPEST_ARC gets trapped in park-and-refill
    # plans on tight-capacity instances, PARALLEL_CHEAPEST_INSERTION on
    # loose-capacity ones. Neither dominates, so mid-route solves try both
    # (fresh model each) and keep the cheaper plan; from-depot solves keep
    # PATH_CHEAPEST_ARC (verified against best-known solutions).
    def hint_cost(routes: list[list[int]]) -> float | None:
        """Cost of executing a warm-start plan verbatim, or None if invalid.

        Time-limited local search is not guaranteed to return a solution at
        least as good as its warm start (it can stop before reaching the
        hint's own cost). Keeping the hint as an explicit candidate makes
        the documented "no worse than the hint" guarantee hold regardless
        of how the solver's wall-clock budget is spent.
        """
        routes = routes[:n_vehicles]
        seen: set[int] = set()
        total = 0.0
        for v, route in enumerate(routes):
            load = 0
            prev = start_node if (v == 0 and start_node != 0) else 0
            for node in route:
                if node == 0 or node not in idx_of or not serve_mask[node] or node in seen:
                    return None
                seen.add(node)
                load += int(demands[node])
                total += float(dist[prev, node])
                prev = node
            cap = start_capacity if (v == 0 and start_capacity is not None) else instance.capacity
            if load > cap:
                return None
            total += float(dist[prev, 0])
        if seen != set(customers):
            return None
        return total

    FS = routing_enums_pb2.FirstSolutionStrategy
    strategies = [FS.PATH_CHEAPEST_ARC]
    if virtual_start is not None:
        strategies.append(FS.PARALLEL_CHEAPEST_INSERTION)

    t0 = time.perf_counter()
    best = None
    hcost = hint_cost(hint) if hint is not None else None
    if hcost is not None:
        best = (hint[:n_vehicles], hcost)
    attempts = [(s, None) for s in strategies]
    if hcost is not None:
        attempts.append((strategies[0], hint))
    for strategy, warm in attempts:
        cand = attempt(strategy, warm_routes=warm)
        if cand is not None and (best is None or cand[1] < best[1]):
            best = cand
    wall = time.perf_counter() - t0
    if best is None:
        raise RuntimeError(f"OR-Tools found no solution for {instance.name}")
    routes, cost = best
    return routes, cost, wall
