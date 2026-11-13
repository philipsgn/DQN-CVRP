"""DQN training entry point.

Examples:
    python -m src.experiments.train --instance P-n16-k8 --episodes 500 --seed 0
    python -m src.experiments.train --all --delta static          # full static sweep
    python -m src.experiments.train --all --delta low high        # dynamic sweeps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.dqn import DQNAgent, DQNConfig
from ..envs.dynamic_env import DynamicRouteEnv
from ..envs.instances import INSTANCES, SMALL, load
from ..envs.route_env import RouteEnv
from .rollout import rollout_policy

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Training-episode environment seeds must never collide with the evaluation
# seeds in run_all.EVAL_SEEDS (range(1000, 1010)). The previous formula
# (seed * 100_000 + ep) collided for train_seed=0 on any instance with
# episodes >= 1000: episodes 1000-1009 reused environment seeds 1000-1009,
# identical to all 10 held-out evaluation seeds, on medium instances under
# dynamic conditions (found in the 2026-08-05 audit; see RESULTS_NOTES.md's
# limitations note — the affected runs were not re-trained, since the
# collision only reused 10 of ~2000 episodes' worth of disruption draws per
# affected run and impact on reported means was judged small). This offset
# pushes every training-episode seed to >= 10,000,000, far above any eval
# seed or any (train_seed, episode) combination this project will ever run,
# so the two seed spaces can no longer overlap by construction.
TRAIN_SEED_OFFSET = 10_000_000


def make_env(instance, delta: str, seed: int):
    if delta == "static":
        return RouteEnv(instance, seed=seed)
    return DynamicRouteEnv(instance, delta=delta, seed=seed)


def episodes_for(name: str) -> int:
    return 1000 if name in SMALL else 2000


class GreedyDQNPolicy:
    """Adapter so a trained agent runs through the shared rollout helper."""

    name = "dqn"

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def act(self, env, obs, info) -> int:
        return self.agent.select_action(obs, info["action_mask"], greedy=True)


def train(
    instance_name: str,
    delta: str = "static",
    seed: int = 0,
    episodes: int | None = None,
    config: DQNConfig | None = None,
    log_every: int = 50,
    quiet: bool = False,
) -> Path:
    instance = load(instance_name)
    env = make_env(instance, delta, seed)
    episodes = episodes or episodes_for(instance_name)
    config = config or DQNConfig()

    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim, env.action_space.n, config, seed=seed)

    run = f"{instance_name}_{delta}_s{seed}"
    out_dir = RESULTS / "training"
    out_dir.mkdir(parents=True, exist_ok=True)

    logs, best_distance = [], float("inf")
    best_path = out_dir / f"{run}_best.pt"
    t0 = time.perf_counter()

    for ep in range(episodes):
        agent.set_epsilon(ep, episodes)
        obs, info = env.reset(seed=TRAIN_SEED_OFFSET + seed * 100_000 + ep)
        ep_reward, ep_distance, losses = 0.0, 0.0, []
        terminated = False
        while not terminated:
            mask = info["action_mask"]
            action = agent.select_action(obs, mask)
            next_obs, reward, terminated, _, info = env.step(action)
            loss = agent.observe(
                obs, action, reward, next_obs, info["action_mask"], terminated
            )
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            ep_reward += reward
            ep_distance += info["distance"]

        if ep_distance < best_distance:
            best_distance = ep_distance
            agent.save(best_path)

        logs.append(
            dict(episode=ep, reward=ep_reward, distance=ep_distance,
                 epsilon=agent.eps, loss=float(np.mean(losses)) if losses else np.nan)
        )
        if not quiet and (ep + 1) % log_every == 0:
            recent = np.mean([l["distance"] for l in logs[-log_every:]])
            print(f"[{run}] ep {ep + 1}/{episodes}  eps={agent.eps:.2f}  "
                  f"mean_dist={recent:.0f}  best={best_distance:.0f}", flush=True)

    agent.save(out_dir / f"{run}_final.pt")
    df = pd.DataFrame(logs)
    df["train_time_s"] = time.perf_counter() - t0
    df.to_csv(out_dir / f"{run}_log.csv", index=False)

    # quick greedy check with the final policy
    final = rollout_policy(env, GreedyDQNPolicy(agent), seed=99_999)
    if not quiet:
        print(f"[{run}] done in {time.perf_counter() - t0:.0f}s — "
              f"greedy={final['distance']:.0f}  BKS={instance.best_known_cost:.0f}")
    return best_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instance", choices=list(INSTANCES), default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--delta", nargs="+", default=["static"],
                   choices=["static", "low", "high"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", type=int, default=None, help="train seeds 0..N-1")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    names = list(INSTANCES) if args.all else [args.instance or "P-n16-k8"]
    seeds = range(args.seeds) if args.seeds else [args.seed]
    cfg = DQNConfig(lr=args.lr)
    for name in names:
        for delta in args.delta:
            for seed in seeds:
                train(name, delta=delta, seed=seed, episodes=args.episodes,
                      config=cfg, quiet=args.quiet)


if __name__ == "__main__":
    main()
