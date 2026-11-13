"""
MP-MOTO-CVRP (V9.2) — Motorbike CVRP Environment
Reinforcement Learning Environment for Local Search Operator Selection on Asymmetric Road Networks.
Supports:
  - Asymmetric route cost computation
  - 4 Actions: 0: 2-opt, 1: intra-relocate, 2: inter-relocate, 3: inter-swap
  - External termination (when no operator improves or step budget reached; no internal 'stop' action)
  - Strict Action Masking (capacity Q_kg check, positive delta C > 0, 1-vehicle inter-op mask)
  - 28-dimensional normalized state vector F28
  - Reward formulation: -Delta C / C_0 + new-record bonus (+0.05)
"""

import copy
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

OPERATOR_NAMES = {
    0: "2-opt",
    1: "intra-relocate",
    2: "inter-relocate",
    3: "inter-swap"
}

def compute_route_cost(route: List[int], dist_m: np.ndarray) -> float:
    """Computes directed cost of a single route [0, c1, c2, ..., cp, 0]."""
    if len(route) < 2:
        return 0.0
    cost = 0.0
    for k in range(len(route) - 1):
        cost += dist_m[route[k], route[k + 1]]
    return float(cost)

def compute_total_cost(routes: List[List[int]], dist_m: np.ndarray) -> float:
    """Computes total cost of all routes."""
    return sum(compute_route_cost(r, dist_m) for r in routes)

def clarke_wright_asymmetric(
    num_customers: int,
    demands: List[float],
    capacity: float,
    dist_m: np.ndarray,
    k_max: Optional[int] = None
) -> List[List[int]]:
    """
    Clarke-Wright Savings heuristic on asymmetric distance matrix.
    Initial solution has 1 vehicle per customer: [0, i, 0].
    Merges routes based on s_ij = dist[i, 0] + dist[0, j] - dist[i, j].
    Strictly enforces k_max fleet limit via Route Consolidation (Regret-2).
    """
    # Dict-based route bookkeeping avoids O(N^3) linear scans for large N (e.g. 570 stores).
    routes: Dict[int, List[int]] = {i: [0, i, 0] for i in range(1, num_customers + 1)}
    route_of: Dict[int, int] = {i: i for i in range(1, num_customers + 1)}
    route_demands: Dict[int, float] = {i: demands[i] for i in range(1, num_customers + 1)}

    if num_customers > 50:
        # Fast Neighborhood mode: only keep the k-best savings per customer.
        # Vectorized NumPy avoids a 570^2 Python loop and keeps large batches responsive.
        sub = dist_m[1:num_customers + 1, 1:num_customers + 1]
        s_mat = dist_m[1:num_customers + 1, 0][:, None] + dist_m[0, 1:num_customers + 1][None, :] - sub
        np.fill_diagonal(s_mat, -np.inf)
        k_neighbors = min(24, max(1, num_customers - 1))
        top_idx = np.argpartition(s_mat, -k_neighbors, axis=1)[:, -k_neighbors:]

        savings = []
        for i in range(num_customers):
            row = s_mat[i]
            for j in top_idx[i]:
                s_val = float(row[j])
                if s_val > 0:
                    savings.append((s_val, i + 1, int(j) + 1))
    else:
        savings = []
        for i in range(1, num_customers + 1):
            for j in range(1, num_customers + 1):
                if i != j:
                    s = dist_m[i, 0] + dist_m[0, j] - dist_m[i, j]
                    savings.append((s, i, j))

    savings.sort(key=lambda x: x[0], reverse=True)

    for s_val, i, j in savings:
        if s_val <= 0:
            continue

        r1_idx = route_of.get(i)
        r2_idx = route_of.get(j)
        if r1_idx is None or r2_idx is None or r1_idx == r2_idx:
            continue

        r1 = routes[r1_idx]
        r2 = routes[r2_idx]

        # Check capacity
        merged_demand = route_demands[r1_idx] + route_demands[r2_idx]
        if merged_demand > capacity + 1e-5:
            continue

        # Check end of r1 and start of r2
        if r1[-2] == i and r2[1] == j:
            merged_route = r1[:-1] + r2[1:]
            routes[r1_idx] = merged_route
            route_demands[r1_idx] = merged_demand
            for node in r2[1:-1]:
                route_of[node] = r1_idx
            del routes[r2_idx]
            del route_demands[r2_idx]

    route_list = list(routes.values())

    # Hard Fleet Cap Consolidation (if len(routes) > k_max)
    if k_max is not None and len(route_list) > k_max:
        route_list = consolidate_routes_to_k(route_list, demands, capacity, dist_m, k_max)

    return route_list


def consolidate_routes_to_k(
    routes: List[List[int]],
    demands: List[float],
    capacity: float,
    dist_m: np.ndarray,
    k_max: int
) -> List[List[int]]:
    """
    Consolidates routes so that len(routes) <= k_max without violating vehicle capacity.
    Repeatedly picks the route with the smallest load and redistributes its customers
    into the remaining routes using Regret-2 / Best-Insertion heuristic.
    """
    if len(routes) <= k_max:
        return routes

    routes = [list(r) for r in routes if len(r) > 2]
    max_attempts = len(routes) * 3
    attempts = 0

    while len(routes) > k_max and attempts < max_attempts:
        attempts += 1
        loads = [sum(demands[u] for u in r[1:-1]) for r in routes]
        sorted_route_indices = sorted(range(len(routes)), key=lambda idx: loads[idx])
        consolidated = False

        for target_idx in sorted_route_indices:
            target_route = routes[target_idx]
            customers_to_move = target_route[1:-1]
            remaining_routes = [r for idx, r in enumerate(routes) if idx != target_idx]
            rem_loads = [sum(demands[u] for u in r[1:-1]) for r in remaining_routes]

            can_fit_all = True
            temp_routes = [list(r) for r in remaining_routes]
            temp_loads = list(rem_loads)

            # Sort customers by demand descending
            sorted_custs = sorted(customers_to_move, key=lambda c: demands[c], reverse=True)

            for u in sorted_custs:
                u_demand = demands[u]
                best_cost_inc = float("inf")
                second_best_cost_inc = float("inf")
                best_route_idx = -1
                best_insert_pos = -1

                for r_idx, r in enumerate(temp_routes):
                    if temp_loads[r_idx] + u_demand <= capacity + 1e-5:
                        for pos in range(1, len(r)):
                            pred = r[pos - 1]
                            succ = r[pos]
                            cost_inc = dist_m[pred, u] + dist_m[u, succ] - dist_m[pred, succ]
                            if cost_inc < best_cost_inc:
                                second_best_cost_inc = best_cost_inc
                                best_cost_inc = cost_inc
                                best_route_idx = r_idx
                                best_insert_pos = pos
                            elif cost_inc < second_best_cost_inc:
                                second_best_cost_inc = cost_inc

                if best_route_idx != -1:
                    temp_routes[best_route_idx].insert(best_insert_pos, u)
                    temp_loads[best_route_idx] += u_demand
                else:
                    can_fit_all = False
                    break

            if can_fit_all:
                routes = temp_routes
                consolidated = True
                break

        if not consolidated:
            break

    return routes

class MotorbikeCVRPEnv:
    def __init__(
        self,
        dist_m: np.ndarray,
        demands_kg: List[float],
        capacity_kg: float,
        k_vehicles: int,
        max_steps: int = 25,
        feature_patience: int = 30,
        seed: Optional[int] = None
    ):
        self.dist_m = dist_m
        self.demands = demands_kg
        self.num_customers = len(demands_kg) - 1  # index 0 is depot
        self.capacity = capacity_kg
        self.k_vehicles = k_vehicles
        self.max_steps = max_steps
        self.feature_patience = feature_patience
        self.rng = random.Random(seed)

        # State tracking
        self.routes: List[List[int]] = []
        self.initial_cost: float = 0.0
        self.current_cost: float = 0.0
        self.best_cost: float = 0.0
        self.step_count: int = 0
        self.evals_used: int = 0

        # Operator statistics (for F28 state)
        self.history_calls = [0, 0, 0, 0]
        self.history_successes = [0, 0, 0, 0]
        self.history_total_deltas = [0.0, 0.0, 0.0, 0.0]
        self.history_streaks = [0, 0, 0, 0]

    def reset(self) -> Tuple[np.ndarray, List[bool]]:
        """Initializes environment with Clarke-Wright solution (enforcing k_vehicles limit)."""
        self.routes = clarke_wright_asymmetric(
            self.num_customers,
            self.demands,
            self.capacity,
            self.dist_m,
            k_max=self.k_vehicles
        )
        self.initial_cost = compute_total_cost(self.routes, self.dist_m)
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost
        self.step_count = 0
        self.evals_used = 0

        self.history_calls = [0, 0, 0, 0]
        self.history_successes = [0, 0, 0, 0]
        self.history_total_deltas = [0.0, 0.0, 0.0, 0.0]
        self.history_streaks = [0, 0, 0, 0]

        state = self._extract_state()
        action_mask = self.get_action_mask()
        return state, action_mask

    def _extract_state(self) -> np.ndarray:
        """Constructs 28-dimensional normalized state vector F28."""
        feat = np.zeros(28, dtype=np.float32)

        # Global features (12 features)
        num_routes = len(self.routes)
        loads = [sum(self.demands[c] for c in r[1:-1]) for r in self.routes]
        load_ratios = [l / self.capacity for l in loads]

        avg_load = float(np.mean(load_ratios)) if load_ratios else 0.0
        std_load = float(np.std(load_ratios)) if load_ratios else 0.0
        max_load = float(np.max(load_ratios)) if load_ratios else 0.0
        min_load = float(np.min(load_ratios)) if load_ratios else 0.0

        lengths = [compute_route_cost(r, self.dist_m) for r in self.routes]
        avg_len = float(np.mean(lengths)) if lengths else 0.0
        std_len = float(np.std(lengths)) if lengths else 0.0
        max_len = float(np.max(lengths)) if lengths else 0.0
        min_len = float(np.min(lengths)) if lengths else 0.0

        feat[0] = avg_load
        feat[1] = std_load
        feat[2] = max_load
        feat[3] = min_load
        feat[4] = num_routes / max(1, self.k_vehicles)
        feat[5] = avg_len / max(1e-3, self.initial_cost)
        feat[6] = std_len / max(1e-3, self.initial_cost)
        feat[7] = max_len / max(1e-3, self.initial_cost)
        feat[8] = min_len / max(1e-3, self.initial_cost)
        feat[9] = self.step_count / max(1, self.max_steps)
        feat[10] = (self.initial_cost - self.current_cost) / max(1e-3, self.initial_cost)
        feat[11] = 1.0 if self.current_cost < self.best_cost else 0.0

        # Operator history features (16 features: 4 stats x 4 operators)
        for op in range(4):
            calls = self.history_calls[op]
            succ = self.history_successes[op]
            total_delta = self.history_total_deltas[op]
            streak = self.history_streaks[op]

            base_idx = 12 + op * 4
            feat[base_idx] = (succ / calls) if calls > 0 else 0.0
            feat[base_idx + 1] = (total_delta / max(1, succ)) / max(1e-3, self.initial_cost) if succ > 0 else 0.0
            feat[base_idx + 2] = calls / max(1, self.step_count)
            feat[base_idx + 3] = min(1.0, streak / self.feature_patience)

        return np.clip(feat, 0.0, 1.0)

    def get_action_mask(self) -> List[bool]:
        """
        Returns boolean mask of 4 valid operators [op0, op1, op2, op3].
        Locks inter-route operators when only 1 route is active.
        """
        mask = [False, False, False, False]
        num_routes = len(self.routes)

        # Operator 0: 2-opt
        for r in self.routes:
            if len(r) >= 4:
                mask[0] = True
                break

        # Operator 1: intra-relocate
        for r in self.routes:
            if len(r) >= 4:
                mask[1] = True
                break

        # Operators 2 & 3: require >= 2 routes
        if num_routes >= 2:
            mask[2] = True
            mask[3] = True

        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes chosen operator (0..3).
        Terminates externally when max_steps reached or when no operator can improve.
        """
        self.step_count += 1

        new_routes, delta_cost, improved = self._evaluate_operator(action)

        self.history_calls[action] += 1
        if improved:
            self.history_successes[action] += 1
            self.history_total_deltas[action] += delta_cost
            self.history_streaks[action] = 0
            self.routes = new_routes
            self.current_cost -= delta_cost

            # Reward formulation: -Delta C / C_0 + record-breaker bonus
            reward = (delta_cost / max(1e-3, self.initial_cost))
            if self.current_cost < self.best_cost - 1e-4:
                self.best_cost = self.current_cost
                reward += 0.05  # Bonus for new episode best
        else:
            self.history_streaks[action] += 1
            reward = -0.005  # Step penalty for unproductive move

        # External stopping condition
        consecutive_idle = all(s >= 2 for s in self.history_streaks)
        done = (self.step_count >= self.max_steps) or consecutive_idle

        info = {
            "action_name": OPERATOR_NAMES[action],
            "delta_cost": delta_cost,
            "improved": improved,
            "current_cost": self.current_cost,
            "routes_count": len(self.routes)
        }

        next_state = self._extract_state()
        return next_state, reward, done, info

    def _evaluate_operator(self, op: int, max_candidates: int = 64) -> Tuple[List[List[int]], float, bool]:
        """Evaluates candidates for operator `op` and returns best improving move."""
        if op == 0:
            return self._eval_2opt(max_candidates)
        elif op == 1:
            return self._eval_intra_relocate(max_candidates)
        elif op == 2:
            return self._eval_inter_relocate(max_candidates)
        elif op == 3:
            return self._eval_inter_swap(max_candidates)
        return self.routes, 0.0, False

    def _eval_2opt(self, max_candidates: int) -> Tuple[List[List[int]], float, bool]:
        best_delta = 0.0
        best_move = None
        candidates = []

        for r_idx, route in enumerate(self.routes):
            m = len(route)
            if m < 4:
                continue
            for i in range(1, m - 2):
                for j in range(i + 1, m - 1):
                    candidates.append((r_idx, i, j))

        if not candidates:
            return self.routes, 0.0, False

        sampled = self.rng.sample(candidates, min(len(candidates), max_candidates))

        for r_idx, i, j in sampled:
            route = self.routes[r_idx]
            cand_route = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
            old_c = compute_route_cost(route, self.dist_m)
            new_c = compute_route_cost(cand_route, self.dist_m)
            delta = old_c - new_c
            if delta > best_delta:
                best_delta = delta
                best_move = (r_idx, cand_route)

        if best_delta > 1e-4 and best_move is not None:
            new_routes = [list(r) for r in self.routes]
            new_routes[best_move[0]] = best_move[1]
            return new_routes, best_delta, True

        return self.routes, 0.0, False

    def _eval_intra_relocate(self, max_candidates: int) -> Tuple[List[List[int]], float, bool]:
        best_delta = 0.0
        best_move = None
        candidates = []

        for r_idx, route in enumerate(self.routes):
            m = len(route)
            if m < 4:
                continue
            for i in range(1, m - 1):
                for j in range(1, m - 1):
                    if i != j:
                        candidates.append((r_idx, i, j))

        if not candidates:
            return self.routes, 0.0, False

        sampled = self.rng.sample(candidates, min(len(candidates), max_candidates))

        for r_idx, i, j in sampled:
            route = self.routes[r_idx]
            cand_route = list(route)
            cust = cand_route.pop(i)
            cand_route.insert(j, cust)
            old_c = compute_route_cost(route, self.dist_m)
            new_c = compute_route_cost(cand_route, self.dist_m)
            delta = old_c - new_c
            if delta > best_delta:
                best_delta = delta
                best_move = (r_idx, cand_route)

        if best_delta > 1e-4 and best_move is not None:
            new_routes = [list(r) for r in self.routes]
            new_routes[best_move[0]] = best_move[1]
            return new_routes, best_delta, True

        return self.routes, 0.0, False

    def _eval_inter_relocate(self, max_candidates: int) -> Tuple[List[List[int]], float, bool]:
        if len(self.routes) < 2:
            return self.routes, 0.0, False

        best_delta = 0.0
        best_move = None
        candidates = []

        for r1_idx in range(len(self.routes)):
            for r2_idx in range(len(self.routes)):
                if r1_idx != r2_idx:
                    r1 = self.routes[r1_idx]
                    r2 = self.routes[r2_idx]
                    for i in range(1, len(r1) - 1):
                        for j in range(1, len(r2)):
                            candidates.append((r1_idx, r2_idx, i, j))

        if not candidates:
            return self.routes, 0.0, False

        sampled = self.rng.sample(candidates, min(len(candidates), max_candidates))

        for r1_idx, r2_idx, i, j in sampled:
            r1 = self.routes[r1_idx]
            r2 = self.routes[r2_idx]
            cust = r1[i]
            cust_demand = self.demands[cust]

            current_r2_demand = sum(self.demands[c] for c in r2[1:-1])
            if current_r2_demand + cust_demand > self.capacity + 1e-5:
                continue

            cand_r1 = list(r1)
            cand_r1.pop(i)
            cand_r2 = list(r2)
            cand_r2.insert(j, cust)

            old_c = compute_route_cost(r1, self.dist_m) + compute_route_cost(r2, self.dist_m)
            new_c = compute_route_cost(cand_r1, self.dist_m) + compute_route_cost(cand_r2, self.dist_m)
            delta = old_c - new_c

            if delta > best_delta:
                best_delta = delta
                best_move = (r1_idx, r2_idx, cand_r1, cand_r2)

        if best_delta > 1e-4 and best_move is not None:
            new_routes = [list(r) for r in self.routes]
            r1_idx, r2_idx, cand_r1, cand_r2 = best_move
            new_routes[r1_idx] = cand_r1
            new_routes[r2_idx] = cand_r2
            new_routes = [r for r in new_routes if len(r) > 2]
            return new_routes, best_delta, True

        return self.routes, 0.0, False

    def _eval_inter_swap(self, max_candidates: int) -> Tuple[List[List[int]], float, bool]:
        if len(self.routes) < 2:
            return self.routes, 0.0, False

        best_delta = 0.0
        best_move = None
        candidates = []

        for r1_idx in range(len(self.routes)):
            for r2_idx in range(r1_idx + 1, len(self.routes)):
                r1 = self.routes[r1_idx]
                r2 = self.routes[r2_idx]
                for i in range(1, len(r1) - 1):
                    for j in range(1, len(r2) - 1):
                        candidates.append((r1_idx, r2_idx, i, j))

        if not candidates:
            return self.routes, 0.0, False

        sampled = self.rng.sample(candidates, min(len(candidates), max_candidates))

        for r1_idx, r2_idx, i, j in sampled:
            r1 = self.routes[r1_idx]
            r2 = self.routes[r2_idx]
            c1 = r1[i]
            c2 = r2[j]

            d1 = self.demands[c1]
            d2 = self.demands[c2]
            r1_dem = sum(self.demands[c] for c in r1[1:-1]) - d1 + d2
            r2_dem = sum(self.demands[c] for c in r2[1:-1]) - d2 + d1

            if r1_dem > self.capacity + 1e-5 or r2_dem > self.capacity + 1e-5:
                continue

            cand_r1 = list(r1)
            cand_r2 = list(r2)
            cand_r1[i] = c2
            cand_r2[j] = c1

            old_c = compute_route_cost(r1, self.dist_m) + compute_route_cost(r2, self.dist_m)
            new_c = compute_route_cost(cand_r1, self.dist_m) + compute_route_cost(cand_r2, self.dist_m)
            delta = old_c - new_c

            if delta > best_delta:
                best_delta = delta
                best_move = (r1_idx, r2_idx, cand_r1, cand_r2)

        if best_delta > 1e-4 and best_move is not None:
            new_routes = [list(r) for r in self.routes]
            r1_idx, r2_idx, cand_r1, cand_r2 = best_move
            new_routes[r1_idx] = cand_r1
            new_routes[r2_idx] = cand_r2
            return new_routes, best_delta, True

        return self.routes, 0.0, False
