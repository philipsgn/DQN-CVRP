"""Capacity-aware nearest-neighbour construction heuristic.

The classical greedy baseline (Dijkstra-style myopic shortest-step
choice): from the current position always move to the closest feasible
unserved customer, returning to the depot to refill only when no
customer fits the remaining capacity.

Because it re-evaluates distances at every step through
``env.step_distance``, the same policy is inherently *reactive* in the
dynamic environment — it sees current traffic conditions when choosing
the next node.
"""

from __future__ import annotations

import numpy as np


class NearestNeighbourPolicy:
    name = "nearest_neighbour"

    def act(self, env, obs, info) -> int:
        mask = info["action_mask"].copy()
        mask[0] = False  # only return to depot when forced
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return 0
        dists = [env.step_distance(env.position, int(c)) for c in candidates]
        return int(candidates[int(np.argmin(dists))])
