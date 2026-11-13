"""Neural Network Architectures for DQN-CVRP.
MLP: 28 -> 128 -> 128 -> 4 with ReLU activations.
"""
import torch
import torch.nn as nn


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set Q-values of infeasible actions to -inf before argmax."""
    return q.masked_fill(~mask.to(dtype=torch.bool), float("-inf"))


class QNetwork(nn.Module):
    """MLP mapping an observation to a Q-value per action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QNetworkMLP(nn.Module):
    """3-layer Multi-Layer Perceptron for Q-value estimation.
    Input: F28 state vector (28-dim continuous)
    Output: 4 Q-values corresponding to the 4 local search operators.
    """
    def __init__(self, state_dim: int = 28, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values for all 4 operators."""
        return self.net(state)
