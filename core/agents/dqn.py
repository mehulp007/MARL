"""Independent Double-Dueling DQN for all intersections.

Every intersection is an independent learner: it acts on its own observation
row and is updated only from its own transitions and its own (mixed) reward.
By default the agents share one set of weights; ``dqn.shared_parameters=False``
gives each its own network.

Target (masked Double DQN, n-step, no terminal states):

    a* = argmax_{a valid in s'} Q_online(s', a)
    y  = G + disc * Q_target(s', a*)
    loss = Huber(Q_online(s, a) - y)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..config import Config, N_ACTIONS, OBS_DIM
from .networks import QEnsemble


class MADQN:
    def __init__(self, cfg: Config, n_agents: int | None = None, seed: int = 0, device: str = "cpu"):
        self.cfg = cfg
        d = cfg.dqn
        self.n_agents = int(cfg.n_agents if n_agents is None else n_agents)
        self.device = torch.device(device)
        if d.torch_threads > 0:
            torch.set_num_threads(d.torch_threads)
        torch.manual_seed(int(seed))
        kw = dict(n_agents=self.n_agents, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=d.hidden,
                  head_hidden=d.head_hidden, shared=d.shared_parameters)
        self.q = QEnsemble(**kw).to(self.device)
        self.q_target = QEnsemble(**kw).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=d.lr)
        self._ids = torch.arange(self.n_agents, device=self.device)
        self.updates = 0

    # ---- acting --------------------------------------------------------
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """(N, n_actions) Q-values for one joint observation (row i -> agent i)."""
        x = torch.as_tensor(np.ascontiguousarray(obs, dtype=np.float32), device=self.device)
        return self.q(x, self._ids).cpu().numpy()

    def act(self, obs: np.ndarray, mask: np.ndarray, eps: float = 0.0,
            rng: np.random.Generator | None = None) -> np.ndarray:
        """Masked epsilon-greedy. Each agent explores independently."""
        q = np.where(mask, self.q_values(obs), -np.inf)
        action = q.argmax(axis=1)
        if eps > 0.0:
            if rng is None:
                raise ValueError("rng is required when eps > 0")
            explore = rng.random(self.n_agents) < eps
            random_valid = np.where(mask, rng.random(mask.shape), -1.0).argmax(axis=1)
            action = np.where(explore, random_valid, action)
        return action

    # ---- learning ------------------------------------------------------
    def _tensors(self, batch: dict) -> dict:
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    @torch.no_grad()
    def compute_target(self, b: dict) -> torch.Tensor:
        """``y`` for a batch of tensors (see module docstring)."""
        q_next = self.q(b["next_obs"], b["agent"]).masked_fill(~b["next_mask"], -torch.inf)
        a_star = q_next.argmax(dim=1, keepdim=True)
        q_tgt = self.q_target(b["next_obs"], b["agent"]).gather(1, a_star).squeeze(1)
        return b["reward"] + b["disc"] * q_tgt

    def update(self, batch: dict) -> dict:
        """One gradient step on a replay batch, then a soft target update."""
        d = self.cfg.dqn
        b = self._tensors(batch)
        y = self.compute_target(b)
        q = self.q(b["obs"], b["agent"]).gather(1, b["action"].unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), d.grad_clip)
        self.opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1.0 - d.tau).add_(p, alpha=d.tau)
        self.updates += 1
        return {"loss": float(loss.item()), "q_mean": float(q.mean().item()),
                "td_abs": float((q - y).abs().mean().item())}

    # ---- persistence (no file I/O here; the caller saves the dict) ------
    def state_dict(self, with_optimizer: bool = False) -> dict:
        out = {
            "q": {k: v.detach().cpu().clone() for k, v in self.q.state_dict().items()},
            "q_target": {k: v.detach().cpu().clone() for k, v in self.q_target.state_dict().items()},
            "updates": self.updates,
            "config": self.cfg.to_dict(),
            "format": "marl-traffic-madqn-v1",
        }
        if with_optimizer:
            out["opt"] = self.opt.state_dict()
        return out

    def load_state_dict(self, state: dict) -> None:
        self.q.load_state_dict(state["q"])
        self.q_target.load_state_dict(state.get("q_target", state["q"]))
        self.updates = int(state.get("updates", 0))
        if "opt" in state:
            self.opt.load_state_dict(state["opt"])

    @classmethod
    def from_state_dict(cls, state: dict, cfg: Config | None = None, device: str = "cpu") -> "MADQN":
        """Rebuild a learner from a checkpoint dict (uses the config stored in it by default)."""
        cfg = cfg if cfg is not None else Config.from_dict(state["config"])
        learner = cls(cfg, device=device)
        learner.load_state_dict(state)
        return learner


class DQNController:
    """Greedy policy of a trained learner behind the common ``Controller`` protocol."""

    def __init__(self, learner: MADQN):
        self.learner = learner

    @classmethod
    def from_state_dict(cls, state: dict, cfg: Config | None = None) -> "DQNController":
        return cls(MADQN.from_state_dict(state, cfg))

    def reset(self, env) -> None:
        if env.n_agents != self.learner.n_agents:
            raise ValueError("checkpoint was trained for a different number of intersections")

    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> np.ndarray:
        return self.learner.act(obs, mask, eps=0.0)
