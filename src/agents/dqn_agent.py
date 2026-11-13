"""DQN and Double DQN Agent for CVRP Local Search Operator Selection.
Matches the methodology and loss target formulations in the report.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import QNetworkMLP


class DQNAgent:
    """Deep Q-Network Agent supporting both Standard DQN and Double DQN."""

    def __init__(
        self,
        algorithm: str = "ddqn",  # "dqn" or "ddqn"
        state_dim: int = 28,
        action_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        target_update_freq: int = 250,
        grad_clip: float = 10.0,
        device: Optional[str] = None
    ):
        self.algorithm = algorithm.lower()
        if self.algorithm not in ("dqn", "ddqn"):
            raise ValueError(f"Unknown algorithm: {self.algorithm}. Must be 'dqn' or 'ddqn'.")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.grad_clip = grad_clip

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.online_net = QNetworkMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetworkMLP(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        self.train_steps: int = 0

        # Auto-load pre-trained checkpoint if available
        ckpt_path = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "ddqn_cvrp_hcmc.pt"
        if ckpt_path.exists():
            try:
                self.load(ckpt_path)
            except Exception:
                pass

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        """Compute raw Q-values for all actions without exploration."""
        with torch.no_grad():
            st_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online_net(st_t).cpu().numpy().squeeze(0).copy()

    def act(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: Optional[Union[list, np.ndarray]] = None
    ) -> Tuple[int, np.ndarray]:
        """Select action via epsilon-greedy policy with optional action masking."""
        with torch.no_grad():
            st_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.online_net(st_t).cpu().numpy().squeeze(0).copy()

        # Apply action mask if provided (e.g. [True, True, False, False] when K <= 1)
        valid_actions = list(range(self.action_dim))
        if action_mask is not None:
            valid_actions = [i for i, valid in enumerate(action_mask) if valid]
            if not valid_actions:
                valid_actions = list(range(self.action_dim))

        if np.random.rand() < epsilon:
            action = int(np.random.choice(valid_actions))
            return action, q_vals

        # For greedy selection, mask invalid actions to -infinity
        masked_q = q_vals.copy()
        if action_mask is not None:
            for i, valid in enumerate(action_mask):
                if not valid:
                    masked_q[i] = -1e9

        action = int(np.argmax(masked_q))
        return action, q_vals

    def compute_td_target(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Bellman TD target for either DQN or Double DQN."""
        with torch.no_grad():
            if self.algorithm == "dqn":
                # Standard DQN: max_a' Q_target(s', a')
                target_q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            else:
                # Double DQN: Q_target(s', argmax_a' Q_online(s', a'))
                best_online_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                target_q_next = self.target_net(next_states).gather(1, best_online_actions)

            targets = rewards + self.gamma * (1.0 - dones) * target_q_next
            return targets

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> float:
        """Perform one gradient update step on the online network."""
        self.online_net.train()

        states = states.to(self.device)
        actions = actions.to(self.device).long().unsqueeze(1)
        rewards = rewards.to(self.device).unsqueeze(1)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device).unsqueeze(1)

        # Predicted Q-value for chosen action
        current_q = self.online_net(states).gather(1, actions)

        # TD target
        target_q = self.compute_td_target(rewards, next_states, dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.sync_target_network()

        return float(loss.item())

    def sync_target_network(self):
        """Synchronize weights of target network with online network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: Union[str, Path]):
        """Save network weights to disk."""
        torch.save({
            "algorithm": self.algorithm,
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "train_steps": self.train_steps,
        }, str(path))

    def load(self, path: Union[str, Path]):
        """Load network weights from disk."""
        checkpoint = torch.load(str(path), map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.train_steps = checkpoint.get("train_steps", 0)
        self.algorithm = checkpoint.get("algorithm", self.algorithm)

