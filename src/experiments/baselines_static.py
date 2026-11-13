"""M3 verification: run all baselines on all static instances.

Produces results/baselines_static.csv and prints a sanity table. Random
should be far worse than nearest-neighbour, which should be worse than
OR-Tools, which should sit within a few percent of best-known.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..baselines.nearest_neighbour import NearestNeighbourPolicy
from ..baselines.ortools_solver import solve as ortools_solve
from ..baselines.random_policy import RandomPolicy
from ..envs.instances import INSTANCES, load
from ..envs.route_env import RouteEnv
from .rollout import execute_fixed_routes, rollout_policy

RESULTS = Path(__file__).resolve().parents[2] / "results"
N_RANDOM_SEEDS = 5


def main() -> pd.DataFrame:
    rows = []
    for name in INSTANCES:
        inst = load(name)
        env = RouteEnv(inst)

        for seed in range(N_RANDOM_SEEDS):
            r = rollout_policy(env, RandomPolicy(seed=seed), seed=seed)
            rows.append(dict(method="random", instance=name, seed=seed,
                             distance=r["distance"], wall_time_s=r["wall_time_s"]))

        r = rollout_policy(env, NearestNeighbourPolicy(), seed=0)
        rows.append(dict(method="nearest_neighbour", instance=name, seed=0,
                         distance=r["distance"], wall_time_s=r["wall_time_s"]))

        routes, planned_cost, wall = ortools_solve(inst, time_limit_s=10)
        executed = execute_fixed_routes(env, routes, seed=0)
        assert executed["distance"] == planned_cost, "env/planner cost mismatch"
        rows.append(dict(method="ortools", instance=name, seed=0,
                         distance=planned_cost, wall_time_s=wall))

    df = pd.DataFrame(rows)
    bks = {name: load(name).best_known_cost for name in INSTANCES}
    df["gap_pct"] = df.apply(
        lambda r: 100 * (r.distance - bks[r.instance]) / bks[r.instance], axis=1
    )
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "baselines_static.csv", index=False)

    summary = df.groupby(["instance", "method"]).agg(
        distance=("distance", "mean"), gap_pct=("gap_pct", "mean")
    ).round(1)
    print(summary.to_string())
    return df


if __name__ == "__main__":
    main()
