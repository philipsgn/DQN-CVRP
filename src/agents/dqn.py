"""DQN / Double DQN agent with invalid-action masking.

Masking is applied in three places — action selection, greedy evaluation,
and, critically, the bootstrap target (an infeasible next action must
never contribute to max_a' Q(s',a'); missing this is the classic bug that
stops masked DQN from converging).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .networks import QNetwork, masked_q
from .replay import ReplayBuffer


@dataclass
class DQNConfig:
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 50_000
    warmup_steps: int = 500
    train_every: int = 1
    target_update_every: int = 500     # hard update period (gradient steps)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.6        # fraction of total episodes to decay over
    hidden: int = 256
    depth: int = 3
    double_dqn: bool = True
    grad_clip: float = 10.0


class DQNAgent:
    name = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig, seed: int = 0):
        self.cfg = config
        self.n_actions = n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = QNetwork(obs_dim, n_actions, config.hidden, config.depth)
        self.target = QNetwork(obs_dim, n_actions, config.hidden, config.depth)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, n_actions, seed=seed)
        self.eps = config.eps_start
        self.grad_steps = 0
        self.env_steps = 0

    # -- acting --------------------------------------------------------------

    def select_action(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(np.flatnonzero(mask)))
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            q = masked_q(q, torch.as_tensor(mask).unsqueeze(0))
            return int(q.argmax(dim=1).item())

    def set_epsilon(self, episode: int, total_episodes: int) -> None:
        decay_len = max(1, int(self.cfg.eps_decay_frac * total_episodes))
        frac = min(1.0, episode / decay_len)
        self.eps = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # -- learning --------------------------------------------------------------

    def observe(self, obs, action, reward, next_obs, next_mask, done) -> float | None:
        self.buffer.push(obs, action, reward, next_obs, next_mask, done)
        self.env_steps += 1
        if len(self.buffer) < self.cfg.warmup_steps:
            return None
        if self.env_steps % self.cfg.train_every != 0:
            return None
        return self._train_step()

    def _train_step(self) -> float:
        batch = self.buffer.sample(self.cfg.batch_size)
        obs = torch.as_tensor(batch["obs"])
        action = torch.as_tensor(batch["action"])
        reward = torch.as_tensor(batch["reward"])
        next_obs = torch.as_tensor(batch["next_obs"])
        next_mask = torch.as_tensor(batch["next_mask"])
        done = torch.as_tensor(batch["done"])

        q_sa = self.q(obs).gather(1, action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            target_next = masked_q(self.target(next_obs), next_mask)
            if self.cfg.double_dqn:
                online_next = masked_q(self.q(next_obs), next_mask)
                best = online_next.argmax(dim=1, keepdim=True)
                next_q = target_next.gather(1, best).squeeze(1)
            else:
                next_q = target_next.max(dim=1).values
            # terminal states have no next mask/action: zero out via (1-done)
            next_q = torch.where(done.bool(), torch.zeros_like(next_q), next_q)
            target = reward + self.cfg.gamma * next_q

        loss = self.loss_fn(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # -- persistence -------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(self.q.state_dict(), path)

    def load(self, path) -> None:
        state = torch.load(path, map_location="cpu")
        self.q.load_state_dict(state)
        self.target.load_state_dict(state)
