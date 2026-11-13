"""Decomposition experiment: static-trained DQN policies on dynamic episodes.

The degradation metric in the main sweep conflates two effects for the DQN:
(a) execution under traffic noise, and (b) training under noisy rewards
converging to a worse policy. Rolling the *static*-trained checkpoints out on
low/high dynamic environments isolates (a): any degradation beyond the
i.i.d.-noise floor (~E[m]-1, cf. random/ortools_fixed) is execution cost, and
the gap between dyn-trained and static-trained policies on the same dynamic
episodes is the training-under-noise loss (b).

Writes results/cross_eval.csv with method = dqn_static_policy.
"""

from __future__ import annotations

import pandas as pd

from ..agents.dqn import DQNAgent, DQNConfig
from ..envs.instances import INSTANCES, load
from .rollout import rollout_policy
from .run_all import EVAL_SEEDS, RESULTS, TRAINING, row
from .train import GreedyDQNPolicy, make_env


class TrafficBlindDQNPolicy(GreedyDQNPolicy):
    """Static-trained network on a dynamic env: drop the appended live-traffic
    row so the observation matches the 5n+1 layout the network was trained on."""

    def __init__(self, agent: DQNAgent, static_obs_dim: int):
        super().__init__(agent)
        self.static_obs_dim = static_obs_dim

    def act(self, env, obs, info) -> int:
        return super().act(env, obs[: self.static_obs_dim], info)


def main():
    rows = []
    for name in INSTANCES:
        inst = load(name)
        for ts in range(5):
            ckpt = TRAINING / f"{name}_static_s{ts}_final.pt"
            if not ckpt.exists():
                print(f"missing {ckpt}, skipping")
                continue
            env = make_env(inst, "static", seed=0)
            static_obs_dim = env.observation_space.shape[0]
            agent = DQNAgent(static_obs_dim, env.action_space.n,
                             DQNConfig(), seed=ts)
            agent.load(ckpt)
            policy = TrafficBlindDQNPolicy(agent, static_obs_dim)
            for delta in ["low", "high"]:
                for es in EVAL_SEEDS:
                    env = make_env(inst, delta, seed=es)
                    r = rollout_policy(env, policy, seed=es)
                    rows.append(row("dqn_static_policy", inst, delta, es,
                                    r["distance"], r["wall_time_s"], train_seed=ts))
        print(f"{name} done", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / "cross_eval.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
