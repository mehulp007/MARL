"""Q-networks.

``DuelingQNet``: obs -> 256 -> 256 -> { V: 128 -> 1, A: 128 -> n_actions },
``Q = V + A - mean(A)``.

``QEnsemble`` hides the choice between one network shared by all intersections
(parameter sharing, the default) and one separate network per intersection. In
both cases an agent's Q-values depend only on its own observation row.
"""
from __future__ import annotations

import torch
from torch import nn


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, head_hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, head_hidden), nn.ReLU(),
                                       nn.Linear(head_hidden, n_actions))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.trunk(obs)
        a = self.advantage(z)
        return self.value(z) + a - a.mean(dim=-1, keepdim=True)


class QEnsemble(nn.Module):
    """``forward(obs, agent)`` -> (B, n_actions). ``agent`` (B,) int64 selects the network
    when parameters are not shared and is ignored when they are."""

    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, hidden: int = 256,
                 head_hidden: int = 128, shared: bool = True):
        super().__init__()
        self.shared = shared
        self.n_agents = n_agents
        self.n_actions = n_actions
        count = 1 if shared else n_agents
        self.nets = nn.ModuleList(DuelingQNet(obs_dim, n_actions, hidden, head_hidden) for _ in range(count))

    def forward(self, obs: torch.Tensor, agent: torch.Tensor | None = None) -> torch.Tensor:
        if self.shared:
            return self.nets[0](obs)
        if agent is None:
            raise ValueError("agent ids are required when parameters are not shared")
        out = obs.new_empty((obs.shape[0], self.n_actions))
        for i in torch.unique(agent).tolist():
            m = agent == i
            out[m] = self.nets[i](obs[m])
        return out
