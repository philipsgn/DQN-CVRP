"""State Extractor for F28 Feature Vector (28 dimensions).
12 Global Features + 16 Operator History Features.
Includes normalization for streak with feature_patience = 30 (Protocol A3).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from .instance import CVRPInstance


@dataclass
class OperatorStats:
    """Historical tracking for a single local search operator."""
    calls: int = 0
    successes: int = 0
    total_delta_cost: float = 0.0
    streak: int = 0  # Consecutive calls without improvement
    recent_outcomes: List[int] = field(default_factory=list)  # 1 for success, 0 for failure (window 10)


class StateExtractorF28:
    """Extracts the 28-dimensional state vector from current routes and search history."""

    def __init__(self, feature_patience: float = 30.0, window_size: int = 10):
        self.feature_patience = feature_patience
        self.window_size = window_size
        self.operator_stats: Dict[int, OperatorStats] = {
            op: OperatorStats() for op in range(4)
        }
        self.total_steps: int = 0

    def reset(self):
        """Reset historical statistics."""
        self.operator_stats = {op: OperatorStats() for op in range(4)}
        self.total_steps = 0

    def record_step(self, operator_id: int, improved: bool, delta_cost: float):
        """Update operator history after taking a step."""
        self.total_steps += 1
        stats = self.operator_stats[operator_id]
        stats.calls += 1
        if improved:
            stats.successes += 1
            stats.total_delta_cost += delta_cost
            stats.streak = 0
            stats.recent_outcomes.append(1)
        else:
            stats.streak += 1
            stats.recent_outcomes.append(0)

        if len(stats.recent_outcomes) > self.window_size:
            stats.recent_outcomes.pop(0)

    def extract(
        self,
        routes: List[List[int]],
        instance: CVRPInstance,
        evals_used: int,
        max_evals: int,
        initial_cost: float
    ) -> np.ndarray:
        """Compute the 28-dimensional state vector."""
        features = np.zeros(28, dtype=np.float32)
        active_routes = [r for r in routes if len(r) > 2]
        num_routes = max(1, len(active_routes))
        capacity = float(instance.capacity)
        k_max = float(instance.max_vehicles)

        # Route load statistics
        loads = [instance.route_demand(r) / capacity for r in active_routes]
        if not loads:
            loads = [0.0]
        loads_arr = np.array(loads, dtype=np.float32)

        # Route distance statistics
        dists = [instance.route_distance(r) for r in active_routes]
        if not dists:
            dists = [0.0]
        dists_arr = np.array(dists, dtype=np.float32)
        current_cost = float(np.sum(dists_arr))

        # Customer counts per route
        cust_counts = [len(r) - 2 for r in active_routes]
        if not cust_counts:
            cust_counts = [0]
        counts_arr = np.array(cust_counts, dtype=np.float32)

        # --- 12 Global Features ---
        features[0] = float(np.mean(loads_arr))                   # f1: mean vehicle load ratio
        features[1] = float(np.std(loads_arr))                    # f2: std dev of vehicle load ratio
        features[2] = float(np.max(loads_arr))                    # f3: max vehicle load ratio
        features[3] = float(np.min(loads_arr))                    # f4: min vehicle load ratio
        features[4] = float(num_routes / max(1.0, k_max))        # f5: normalized vehicle count
        features[5] = float(np.mean(counts_arr) / max(1.0, instance.num_customers))  # f6: mean cust / total
        features[6] = float(np.std(counts_arr))                   # f7: std dev cust per route
        features[7] = float(np.mean(dists_arr) / max(1.0, current_cost)) # f8: relative mean route distance
        features[8] = float(np.max(dists_arr) / max(1.0, current_cost))  # f9: relative max route distance
        features[9] = float(np.std(dists_arr) / max(1.0, np.mean(dists_arr))) # f10: CV of route distances
        features[10] = float(min(1.0, evals_used / max(1, max_evals))) # f11: budget progress
        # f12: cumulative relative improvement
        features[11] = float(max(0.0, (initial_cost - current_cost) / max(1e-6, initial_cost)))

        # --- 16 Operator History Features ---
        idx = 12
        total_calls = max(1, self.total_steps)
        patience = max(1.0, self.feature_patience)

        for op in range(4):
            stats = self.operator_stats[op]
            # Recent success rate (window)
            if stats.recent_outcomes:
                recent_succ = sum(stats.recent_outcomes) / len(stats.recent_outcomes)
            else:
                recent_succ = 0.0

            # Mean relative improvement
            if stats.calls > 0:
                mean_impr = (stats.total_delta_cost / stats.calls) / max(1e-6, current_cost)
            else:
                mean_impr = 0.0

            # Call frequency
            freq = stats.calls / total_calls

            # Stagnation streak normalized
            streak_norm = min(1.0, stats.streak / patience)

            features[idx] = float(recent_succ)
            features[idx + 1] = float(mean_impr)
            features[idx + 2] = float(freq)
            features[idx + 3] = float(streak_norm)
            idx += 4

        return np.clip(features, -10.0, 10.0)

