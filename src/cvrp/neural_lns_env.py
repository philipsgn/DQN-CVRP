"""Neural Large Neighborhood Search (Neural LNS) Environment for CVRP on HCMC Graph.
Integrates Destroy & Repair Operators, Simulated Annealing Acceptance, and Shaped Rewards.
Enforces hard vehicle capacity Q and fleet limit K_used <= K.
"""
from typing import List, Tuple, Dict, Any, Optional
import math
import random
import numpy as np

from .motorbike_env import compute_total_cost, consolidate_routes_to_k, clarke_wright_asymmetric


class NeuralLNSEnv:
    """
    Gym-like Environment for Neural Large Neighborhood Search.
    State: S_t in R^36
    Action: 6 Macro Destroy-Repair Strategies:
      0: Worst-Cost Removal + Greedy Best Insertion
      1: Worst-Cost Removal + Regret-2 Insertion
      2: Shaw Relatedness Removal + Greedy Best Insertion
      3: Shaw Relatedness Removal + Regret-2 Insertion
      4: Route-Squeeze Removal + Regret-2 Insertion (Consolidation)
      5: Worst-Cost Removal + Regret-2 Insertion + Post-Repair 2-opt Polish
    """

    def __init__(
        self,
        dist_m: np.ndarray,
        demands_kg: List[float],
        capacity_kg: float,
        k_vehicles: int,
        max_steps: int = 30,
        initial_temp: float = 0.02,
        cooling_rate: float = 0.94,
        seed: Optional[int] = None
    ):
        self.dist_m = dist_m
        self.demands = demands_kg
        self.num_customers = len(demands_kg) - 1
        self.capacity = capacity_kg
        self.k_vehicles = k_vehicles
        self.max_steps = max_steps
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.rng = random.Random(seed)

        self.routes: List[List[int]] = []
        self.best_routes: List[List[int]] = []
        self.initial_cost: float = 0.0
        self.current_cost: float = 0.0
        self.best_cost: float = 0.0

        self.step_count: int = 0
        self.temp: float = initial_temp
        self.steps_since_improvement: int = 0
        self.cum_improvement: float = 0.0

        # Action history (6 actions)
        self.action_calls = [0] * 6
        self.action_successes = [0] * 6
        self.action_deltas = [0.0] * 6

    def reset(self) -> Tuple[np.ndarray, List[bool]]:
        """Initializes environment with Clarke-Wright solution (strictly <= k_vehicles)."""
        self.routes = clarke_wright_asymmetric(
            self.num_customers,
            self.demands,
            self.capacity,
            self.dist_m,
            k_max=self.k_vehicles
        )
        # Ensure k_vehicles constraint
        if len(self.routes) > self.k_vehicles:
            self.routes = consolidate_routes_to_k(
                self.routes, self.demands, self.capacity, self.dist_m, self.k_vehicles
            )

        self.initial_cost = compute_total_cost(self.routes, self.dist_m)
        self.current_cost = self.initial_cost
        self.best_cost = self.initial_cost
        self.best_routes = [list(r) for r in self.routes]

        self.step_count = 0
        self.temp = self.initial_temp
        self.steps_since_improvement = 0
        self.cum_improvement = 0.0

        self.action_calls = [0] * 6
        self.action_successes = [0] * 6
        self.action_deltas = [0.0] * 6

        state = self._extract_state()
        mask = self.get_action_mask()
        return state, mask

    def get_action_mask(self) -> List[bool]:
        """Returns boolean mask for 6 actions."""
        # Route squeeze (action 4) requires at least 2 routes
        num_routes = len(self.routes)
        mask = [True, True, True, True, (num_routes >= 2), True]
        return mask

    def _extract_state(self) -> np.ndarray:
        """Constructs 36-dimensional normalized state vector S_t."""
        feat = np.zeros(36, dtype=np.float32)

        # 1. Load statistics (4 features)
        loads = [sum(self.demands[u] for u in r[1:-1]) for r in self.routes] if self.routes else [0.0]
        mean_load = float(np.mean(loads)) / self.capacity
        std_load = float(np.std(loads)) / self.capacity if len(loads) > 1 else 0.0
        max_load = float(np.max(loads)) / self.capacity
        min_load = float(np.min(loads)) / self.capacity

        feat[0] = min(1.0, mean_load)
        feat[1] = min(1.0, std_load)
        feat[2] = min(1.0, max_load)
        feat[3] = min(1.0, min_load)

        # 2. Fleet & Route statistics (4 features)
        num_routes = len(self.routes)
        cust_counts = [len(r) - 2 for r in self.routes] if self.routes else [0]
        feat[4] = min(1.0, num_routes / max(1, self.k_vehicles))
        feat[5] = min(1.0, float(np.mean(cust_counts)) / max(1, self.num_customers))
        feat[6] = min(1.0, float(np.max(cust_counts)) / max(1, self.num_customers))
        feat[7] = min(1.0, float(np.min(cust_counts)) / max(1, self.num_customers))

        # 3. Distance & Geometry features (4 features)
        feat[8] = min(2.0, self.current_cost / max(1e-3, self.initial_cost))
        feat[9] = min(1.0, self.best_cost / max(1e-3, self.current_cost))
        feat[10] = min(1.0, self.cum_improvement / max(1e-3, self.initial_cost))
        feat[11] = min(1.0, self.temp / max(1e-4, self.initial_temp))

        # 4. Search progress features (8 features)
        feat[12] = min(1.0, self.step_count / max(1, self.max_steps))
        feat[13] = min(1.0, self.steps_since_improvement / max(1, self.max_steps))
        feat[14] = 1.0 if num_routes <= self.k_vehicles else 0.0
        feat[15] = min(1.0, sum(self.action_successes) / max(1, self.step_count))
        feat[16] = min(1.0, sum(self.action_deltas) / max(1e-3, self.initial_cost))
        feat[17] = min(1.0, float(self.num_customers) / 50.0)
        feat[18] = min(1.0, self.capacity / 80.0)
        feat[19] = min(1.0, float(self.k_vehicles) / 15.0)

        # 5. Operator statistics (16 features for the 6 destroy-repair actions)
        # Action calls ratio (6 features: indices 20..25)
        for a in range(6):
            feat[20 + a] = min(1.0, self.action_calls[a] / max(1, self.step_count))

        # Action success ratio (6 features: indices 26..31)
        for a in range(6):
            calls = self.action_calls[a]
            succ = self.action_successes[a]
            feat[26 + a] = (succ / calls) if calls > 0 else 0.0

        # Action average improvement ratio (4 features for top actions: indices 32..35)
        for a in range(4):
            succ = self.action_successes[a]
            delta = self.action_deltas[a]
            feat[32 + a] = min(1.0, (delta / max(1, succ)) / max(1e-3, self.initial_cost)) if succ > 0 else 0.0

        return np.clip(feat, 0.0, 1.0)

    def _destroy_worst_cost(self, routes: List[List[int]], p: int) -> Tuple[List[List[int]], List[int]]:
        """Removes p customers contributing highest cost increase to their routes."""
        candidates = []
        for r_idx, r in enumerate(routes):
            for pos in range(1, len(r) - 1):
                u = r[pos]
                pred = r[pos - 1]
                succ = r[pos + 1]
                cost_saving = self.dist_m[pred, u] + self.dist_m[u, succ] - self.dist_m[pred, succ]
                candidates.append((cost_saving, u, r_idx, pos))

        candidates.sort(key=lambda x: x[0], reverse=True)
        to_remove = set()
        for _, u, _, _ in candidates:
            to_remove.add(u)
            if len(to_remove) >= p:
                break

        new_routes = []
        for r in routes:
            new_r = [0] + [u for u in r[1:-1] if u not in to_remove] + [0]
            if len(new_r) > 2:
                new_routes.append(new_r)

        return new_routes, list(to_remove)

    def _destroy_shaw(self, routes: List[List[int]], p: int) -> Tuple[List[List[int]], List[int]]:
        """Removes p geographically/demand-related customers."""
        all_nodes = [u for r in routes for u in r[1:-1]]
        if not all_nodes:
            return routes, []

        seed_node = self.rng.choice(all_nodes)
        related = []
        for u in all_nodes:
            if u != seed_node:
                dist_score = (self.dist_m[seed_node, u] + self.dist_m[u, seed_node]) / 2.0
                dem_score = abs(self.demands[seed_node] - self.demands[u]) * 100.0
                related.append((dist_score + dem_score, u))

        related.sort(key=lambda x: x[0])
        to_remove = {seed_node}
        for _, u in related:
            to_remove.add(u)
            if len(to_remove) >= p:
                break

        new_routes = []
        for r in routes:
            new_r = [0] + [u for u in r[1:-1] if u not in to_remove] + [0]
            if len(new_r) > 2:
                new_routes.append(new_r)

        return new_routes, list(to_remove)

    def _destroy_route_squeeze(self, routes: List[List[int]], p: int) -> Tuple[List[List[int]], List[int]]:
        """Squeezes out the route with smallest load to eliminate a vehicle."""
        if len(routes) <= 1:
            return self._destroy_worst_cost(routes, p)

        loads = [sum(self.demands[u] for u in r[1:-1]) for r in routes]
        min_idx = int(np.argmin(loads))
        target_route = routes[min_idx]
        to_remove = set(target_route[1:-1])

        # If target route has fewer than p, add random nodes from other routes
        other_nodes = [u for idx, r in enumerate(routes) if idx != min_idx for u in r[1:-1]]
        self.rng.shuffle(other_nodes)
        for u in other_nodes:
            if len(to_remove) >= p:
                break
            to_remove.add(u)

        new_routes = []
        for idx, r in enumerate(routes):
            if idx == min_idx:
                continue
            new_r = [0] + [u for u in r[1:-1] if u not in to_remove] + [0]
            if len(new_r) > 2:
                new_routes.append(new_r)

        return new_routes, list(to_remove)

    def _repair_greedy(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Inserts removed nodes into the best positions minimizing cost increase."""
        temp_routes = [list(r) for r in routes]
        loads = [sum(self.demands[u] for u in r[1:-1]) for r in temp_routes]

        # Sort removed customers by demand descending
        sorted_nodes = sorted(removed, key=lambda u: self.demands[u], reverse=True)

        for u in sorted_nodes:
            u_dem = self.demands[u]
            best_cost_inc = float("inf")
            best_r_idx = -1
            best_pos = -1

            for r_idx, r in enumerate(temp_routes):
                if loads[r_idx] + u_dem <= self.capacity + 1e-5:
                    for pos in range(1, len(r)):
                        pred = r[pos - 1]
                        succ = r[pos]
                        inc = self.dist_m[pred, u] + self.dist_m[u, succ] - self.dist_m[pred, succ]
                        if inc < best_cost_inc:
                            best_cost_inc = inc
                            best_r_idx = r_idx
                            best_pos = pos

            # If can't fit into existing routes and fleet has room, open new route
            if best_r_idx == -1 and len(temp_routes) < self.k_vehicles:
                temp_routes.append([0, u, 0])
                loads.append(u_dem)
            elif best_r_idx != -1:
                temp_routes[best_r_idx].insert(best_pos, u)
                loads[best_r_idx] += u_dem
            else:
                # Fallback: force insertion into least loaded route that fits
                for r_idx, r in enumerate(temp_routes):
                    if loads[r_idx] + u_dem <= self.capacity + 1e-5:
                        temp_routes[r_idx].insert(len(r) - 1, u)
                        loads[r_idx] += u_dem
                        break

        return [r for r in temp_routes if len(r) > 2]

    def _repair_regret2(self, routes: List[List[int]], removed: List[int]) -> List[List[int]]:
        """Regret-2 insertion: prioritizes nodes with largest difference between 2nd best and best routes."""
        temp_routes = [list(r) for r in routes]
        loads = [sum(self.demands[u] for u in r[1:-1]) for r in temp_routes]
        unassigned = list(removed)

        while unassigned:
            best_regret = -float("inf")
            best_node = None
            chosen_r_idx = -1
            chosen_pos = -1
            chosen_inc = float("inf")

            for u in unassigned:
                u_dem = self.demands[u]
                best_inc = float("inf")
                second_inc = float("inf")
                best_r = -1
                best_p = -1

                for r_idx, r in enumerate(temp_routes):
                    if loads[r_idx] + u_dem <= self.capacity + 1e-5:
                        for pos in range(1, len(r)):
                            pred = r[pos - 1]
                            succ = r[pos]
                            inc = self.dist_m[pred, u] + self.dist_m[u, succ] - self.dist_m[pred, succ]
                            if inc < best_inc:
                                second_inc = best_inc
                                best_inc = inc
                                best_r = r_idx
                                best_p = pos
                            elif inc < second_inc:
                                second_inc = inc

                # If only fits in 1 route or can open new route
                if second_inc == float("inf") and len(temp_routes) < self.k_vehicles:
                    new_route_inc = self.dist_m[0, u] + self.dist_m[u, 0]
                    second_inc = max(second_inc, new_route_inc)

                regret = (second_inc - best_inc) if (best_inc != float("inf") and second_inc != float("inf")) else 0.0
                if regret > best_regret:
                    best_regret = regret
                    best_node = u
                    chosen_r_idx = best_r
                    chosen_pos = best_p
                    chosen_inc = best_inc

            if best_node is not None and chosen_r_idx != -1:
                temp_routes[chosen_r_idx].insert(chosen_pos, best_node)
                loads[chosen_r_idx] += self.demands[best_node]
                unassigned.remove(best_node)
            elif len(temp_routes) < self.k_vehicles and unassigned:
                u = unassigned.pop(0)
                temp_routes.append([0, u, 0])
                loads.append(self.demands[u])
            elif unassigned:
                # Greedy fallback
                temp_routes = self._repair_greedy(temp_routes, unassigned)
                break

        return [r for r in temp_routes if len(r) > 2]

    def _intra_2opt_polish(self, routes: List[List[int]]) -> List[List[int]]:
        """Polishes each route with intra-route 2-opt."""
        polished = []
        for r in routes:
            route = list(r)
            improved = True
            while improved:
                improved = False
                n = len(route)
                for i in range(1, n - 2):
                    for j in range(i + 1, n - 1):
                        a, b = route[i - 1], route[i]
                        c, d = route[j], route[j + 1]
                        current_seg = self.dist_m[a, b] + self.dist_m[c, d]
                        new_seg = self.dist_m[a, c] + self.dist_m[b, d]
                        if new_seg < current_seg - 1e-4:
                            route[i:j + 1] = reversed(route[i:j + 1])
                            improved = True
                            break
                    if improved:
                        break
            polished.append(route)
        return polished

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes chosen Destroy-Repair strategy (0..5).
        Evaluates candidate solution with Simulated Annealing acceptance.
        """
        self.step_count += 1
        self.action_calls[action] += 1

        # Calculate p (15% - 30% of customers, min 2, max min(8, n/2))
        p = max(2, min(8, int(self.num_customers * 0.25)))

        # 1. Execute Destroy
        if action in [0, 1, 5]:
            destroyed_routes, removed = self._destroy_worst_cost(self.routes, p)
        elif action in [2, 3]:
            destroyed_routes, removed = self._destroy_shaw(self.routes, p)
        else:  # action 4: route squeeze
            destroyed_routes, removed = self._destroy_route_squeeze(self.routes, p)

        # 2. Execute Repair
        if action in [0, 2]:
            repaired_routes = self._repair_greedy(destroyed_routes, removed)
        elif action in [1, 3, 4]:
            repaired_routes = self._repair_regret2(destroyed_routes, removed)
        else:  # action 5: regret2 + 2-opt polish
            temp_r = self._repair_regret2(destroyed_routes, removed)
            repaired_routes = self._intra_2opt_polish(temp_r)

        # 3. Consolidate to enforce hard fleet cap <= k_vehicles
        if len(repaired_routes) > self.k_vehicles:
            repaired_routes = consolidate_routes_to_k(
                repaired_routes, self.demands, self.capacity, self.dist_m, self.k_vehicles
            )

        cand_cost = compute_total_cost(repaired_routes, self.dist_m)
        delta = self.current_cost - cand_cost

        # 4. Simulated Annealing Acceptance
        accepted = False
        if delta > 1e-4:
            accepted = True
        else:
            # Acceptance probability for slightly worse solutions
            deg_pct = abs(delta) / max(1.0, self.current_cost)
            if deg_pct <= 0.02:  # Max 2% degradation allowed
                prob = math.exp(delta / max(1e-4, self.temp * self.current_cost))
                if self.rng.random() < prob:
                    accepted = True

        reward = 0.0
        new_record = False

        if accepted:
            self.routes = repaired_routes
            self.current_cost = cand_cost

            # Check new best record
            if self.current_cost < self.best_cost - 1e-4:
                cost_saving = self.best_cost - self.current_cost
                self.best_cost = self.current_cost
                self.best_routes = [list(r) for r in self.routes]
                self.cum_improvement += cost_saving
                self.steps_since_improvement = 0
                new_record = True

                self.action_successes[action] += 1
                self.action_deltas[action] += cost_saving

                # Reward proportional to record improvement
                reward = (cost_saving / max(1e-3, self.initial_cost)) * 100.0 + 1.0
            else:
                self.steps_since_improvement += 1
                reward = 0.05
        else:
            self.steps_since_improvement += 1
            reward = -0.02

        # Cool down temperature
        self.temp *= self.cooling_rate

        done = (self.step_count >= self.max_steps) or (self.steps_since_improvement >= 15)

        info = {
            "action": action,
            "accepted": accepted,
            "new_record": new_record,
            "current_cost": self.current_cost,
            "best_cost": self.best_cost,
            "k_used": len(self.routes),
            "step": self.step_count
        }

        state = self._extract_state()
        mask = self.get_action_mask()
        return state, reward, done, info

