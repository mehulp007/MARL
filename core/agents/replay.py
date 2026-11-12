"""n-step experience replay.

``NStepAssembler`` turns the stream of joint environment steps into n-step
transitions for all agents at once:

    G    = sum_{k=0}^{m-1} gamma^k * r_{t+k}
    disc = gamma^m                      (m = n, or fewer steps at the end of an episode)
    next = observation and action mask at t + m

There are no terminal states (time-limit truncation is not termination), so
every transition bootstraps; ``disc`` carries the right power of gamma.

``ReplayBuffer`` stores per-agent transitions in a ring. Observations are kept
as float16 (they are normalised to [0, 1]; this halves memory) and returned as
float32.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class NStepAssembler:
    def __init__(self, n: int, gamma: float, n_agents: int):
        self.n = int(n)
        self.gamma = float(gamma)
        self.n_agents = int(n_agents)
        self._agent = np.arange(n_agents, dtype=np.int16)
        self._q: deque = deque()
        self._next_obs = None
        self._next_mask = None

    def reset(self) -> None:
        self._q.clear()
        self._next_obs = None
        self._next_mask = None

    def _emit(self) -> dict:
        obs, action, _ = self._q[0]
        g = np.zeros(self.n_agents, dtype=np.float64)
        w = 1.0
        for _, _, r in self._q:
            g += w * r
            w *= self.gamma
        self._q.popleft()
        return {"obs": obs, "action": action, "reward": g.astype(np.float32),
                "next_obs": self._next_obs, "next_mask": self._next_mask,
                "disc": np.full(self.n_agents, w, dtype=np.float32), "agent": self._agent}

    def push(self, obs, action, reward, next_obs, next_mask):
        """Add one joint step. Returns an n-step transition batch once n steps are available."""
        self._q.append((np.asarray(obs), np.asarray(action), np.asarray(reward, dtype=np.float64)))
        self._next_obs = np.asarray(next_obs)
        self._next_mask = np.asarray(next_mask)
        return self._emit() if len(self._q) == self.n else None

    def flush(self) -> list:
        """End of episode: emit the remaining steps with shortened horizons."""
        out = []
        while self._q:
            out.append(self._emit())
        return out


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, n_actions: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float16)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float16)
        self.next_mask = np.zeros((self.capacity, n_actions), dtype=bool)
        self.action = np.zeros(self.capacity, dtype=np.int8)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.disc = np.zeros(self.capacity, dtype=np.float32)
        self.agent = np.zeros(self.capacity, dtype=np.int16)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng([int(seed), 11])

    def __len__(self) -> int:
        return self.size

    def add(self, tr: dict) -> None:
        """Add a batch of per-agent transitions (the dict produced by ``NStepAssembler``)."""
        k = len(tr["action"])
        idx = (self.ptr + np.arange(k)) % self.capacity
        self.obs[idx] = tr["obs"]
        self.next_obs[idx] = tr["next_obs"]
        self.next_mask[idx] = tr["next_mask"]
        self.action[idx] = tr["action"]
        self.reward[idx] = tr["reward"]
        self.disc[idx] = tr["disc"]
        self.agent[idx] = tr["agent"]
        self.ptr = int((self.ptr + k) % self.capacity)
        self.size = min(self.capacity, self.size + k)

    def sample(self, batch_size: int) -> dict:
        idx = self.rng.integers(0, self.size, batch_size)
        return {"obs": self.obs[idx].astype(np.float32),
                "action": self.action[idx].astype(np.int64),
                "reward": self.reward[idx],
                "next_obs": self.next_obs[idx].astype(np.float32),
                "next_mask": self.next_mask[idx],
                "disc": self.disc[idx],
                "agent": self.agent[idx].astype(np.int64)}
