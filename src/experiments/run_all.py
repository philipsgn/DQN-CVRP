"""Master evaluation sweep -> results/results_master.csv.

Protocol
--------
For every instance and dynamism level (static / low / high):

* random             — masked random policy, one rollout per eval seed
* nearest_neighbour  — reactive greedy, one rollout per eval seed
* ortools_fixed      — solve once on the true static matrix (10 s),
                       execute that plan verbatim under each eval seed
* ortools_replan     — re-solve (1 s) on the static matrix every traffic
                       re-sample (expectation-optimal for transient i.i.d.
                       noise; dynamic only — on static it equals
                       ortools_fixed)
* ortools_replan_live— naive re-solve on the full live-priced matrix
                       (negative-result condition: plan churn)
* dqn                — final checkpoint per training seed, greedy rollout
                       per eval seed

Static environments are deterministic, so deterministic methods run once.
Optimality gap vs. best-known is reported for static; dynamic rows keep
raw distance (degradation analysis happens in src.analysis.stats).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ..agents.dqn import DQNAgent, DQNConfig
from ..baselines.nearest_neighbour import NearestNeighbourPolicy
from ..baselines.ortools_replan import ORToolsReplanPolicy
from ..baselines.ortools_solver import solve as ortools_solve
from ..baselines.random_policy import RandomPolicy
from ..envs.instances import INSTANCES, load
from .rollout import execute_fixed_routes, rollout_policy
from .train import GreedyDQNPolicy, make_env

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
TRAINING = RESULTS / "training"

EVAL_SEEDS = list(range(1000, 1010))  # 10 evaluation episodes
DELTAS = ["static", "low", "high"]


def row(method, inst, delta, eval_seed, distance, wall, train_seed=None,
        replan_time=0.0, replan_count=0):
    return dict(
        method=method, instance=inst.name, n_customers=inst.n_customers,
        delta=delta, train_seed=train_seed, eval_seed=eval_seed,
        distance=distance, wall_time_s=wall,
        replan_time_s=replan_time, replan_count=replan_count,
        bks=inst.best_known_cost,
        gap_pct=100 * (distance - inst.best_known_cost) / inst.best_known_cost,
    )


def eval_instance(name: str, train_seeds: range) -> list[dict]:
    inst = load(name)
    rows = []
    static_routes, static_cost, static_wall = ortools_solve(inst, time_limit_s=10)

    for delta in DELTAS:
        deterministic_env = delta == "static"
        seeds = EVAL_SEEDS[:1] if deterministic_env else EVAL_SEEDS

        for es in EVAL_SEEDS:  # random policy is stochastic even on static env
            env = make_env(inst, delta, seed=es)
            r = rollout_policy(env, RandomPolicy(seed=es), seed=es)
            rows.append(row("random", inst, delta, es, r["distance"], r["wall_time_s"]))

        for es in seeds:
            env = make_env(inst, delta, seed=es)
            r = rollout_policy(env, NearestNeighbourPolicy(), seed=es)
            rows.append(row("nearest_neighbour", inst, delta, es,
                            r["distance"], r["wall_time_s"]))

        for es in seeds:
            env = make_env(inst, delta, seed=es)
            r = execute_fixed_routes(env, static_routes, seed=es)
            rows.append(row("ortools_fixed", inst, delta, es,
                            r["distance"], static_wall))

        if not deterministic_env:
            for method, mode in [("ortools_replan", "expectation"),
                                 ("ortools_replan_live", "live_full")]:
                for es in seeds:
                    env = make_env(inst, delta, seed=es)
                    policy = ORToolsReplanPolicy(time_limit_s=1, mode=mode)
                    r = rollout_policy(env, policy, seed=es)
                    rows.append(row(method, inst, delta, es,
                                    r["distance"], r["wall_time_s"],
                                    replan_time=policy.replan_time_s,
                                    replan_count=policy.replan_count))

        for ts in train_seeds:
            ckpt = TRAINING / f"{name}_{delta}_s{ts}_final.pt"
            if not ckpt.exists():
                continue
            env = make_env(inst, delta, seed=0)
            agent = DQNAgent(env.observation_space.shape[0], env.action_space.n,
                             DQNConfig(), seed=ts)
            agent.load(ckpt)
            policy = GreedyDQNPolicy(agent)
            for es in seeds:
                env = make_env(inst, delta, seed=es)
                r = rollout_policy(env, policy, seed=es)
                rows.append(row("dqn", inst, delta, es, r["distance"],
                                r["wall_time_s"], train_seed=ts))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-seeds", type=int, default=5)
    p.add_argument("--instances", nargs="+", default=list(INSTANCES))
    p.add_argument("--out", default="results_master.csv")
    args = p.parse_args()

    all_rows = []
    t0 = time.perf_counter()
    for name in args.instances:
        print(f"evaluating {name} ...", flush=True)
        all_rows += eval_instance(name, range(args.train_seeds))
    df = pd.DataFrame(all_rows)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / args.out
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} in {(time.perf_counter() - t0) / 60:.1f} min")

    check = df.groupby(["method", "delta"]).size().unstack(fill_value=0)
    print(check.to_string())


if __name__ == "__main__":
    main()
