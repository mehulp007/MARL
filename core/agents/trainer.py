"""Algorithmic training loop. No file I/O, no printing.

The caller (``training/train.py``) receives everything through callbacks:

    on_episode_end(stats: EpisodeStats)
    on_eval(result: dict)                       # greedy evaluation on fixed seeds
    on_snapshot(episode: int, state: dict)      # learner.state_dict() at scheduled episodes

Per episode the trainer samples a demand level, side bias and profile, and with
probability ``train.p_scenario`` a random perturbation scenario (domain
randomisation), then runs masked epsilon-greedy control with one gradient step
per decision step once the buffer holds ``dqn.learn_starts`` transitions.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..baselines import run_episode
from ..config import Config, N_ACTIONS, OBS_DIM
from ..demand import DemandSpec
from ..env import TrafficEnv
from ..network import Network
from ..scenarios import EMPTY, Scenario, random_scenario
from .dqn import DQNController, MADQN
from .replay import NStepAssembler, ReplayBuffer


@dataclass
class EpisodeStats:
    episode: int                 # 1-based index of the episode just finished
    total_steps: int             # decision steps so far
    epsilon: float               # epsilon at the end of the episode
    ret: float                   # mean over agents of the summed (mixed) reward, unscaled
    ret_local: float             # same for the local reward
    loss: float                  # mean Huber loss over the episode's updates (nan if none)
    q_mean: float
    td_abs: float
    buffer: int
    demand_rate: float
    profile: str
    n_events: int
    wall_time: float             # seconds for this episode
    summary: dict = field(default_factory=dict)   # env.summary() of the training episode

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    def __init__(self, cfg: Config, env: TrafficEnv | None = None, learner: MADQN | None = None,
                 on_episode_end: Callable | None = None, on_eval: Callable | None = None,
                 on_snapshot: Callable | None = None):
        self.cfg = cfg
        d, t = cfg.dqn, cfg.train
        self.env = env if env is not None else TrafficEnv(cfg, seed=t.seed, net=Network(cfg))
        self.net = self.env.net
        self.learner = learner if learner is not None else MADQN(cfg, self.env.n_agents, seed=t.seed)
        self.buffer = ReplayBuffer(d.buffer_size, OBS_DIM, N_ACTIONS, seed=t.seed)
        self.assembler = NStepAssembler(d.n_step, d.gamma, self.env.n_agents)
        self.rng = np.random.default_rng([int(t.seed), 10])          # curriculum + exploration
        self.on_episode_end = on_episode_end
        self.on_eval = on_eval
        self.on_snapshot = on_snapshot
        self.episode = 0
        self.total_steps = 0
        self.steps_per_episode = int(round(cfg.env.episode_len / cfg.signal.decision_dt))
        self.planned_episodes = t.episodes
        self.history: list = []
        self.eval_history: list = []

    # ------------------------------------------------------------------
    def epsilon(self, step: int | None = None) -> float:
        """Linear decay from eps_start to eps_end over the first eps_frac of training."""
        d = self.cfg.dqn
        step = self.total_steps if step is None else step
        horizon = max(1.0, d.eps_frac * self.planned_episodes * self.steps_per_episode)
        frac = min(1.0, step / horizon)
        return d.eps_start + (d.eps_end - d.eps_start) * frac

    def sample_episode(self):
        """Curriculum: (seed, DemandSpec, Scenario) for the next training episode."""
        t, rng = self.cfg.train, self.rng
        rate = float(rng.uniform(*t.demand_range))
        bias = tuple(float(x) for x in rng.uniform(t.side_bias_range[0], t.side_bias_range[1], 4))
        profile = str(t.profiles[int(rng.integers(len(t.profiles)))])
        spec = DemandSpec.from_config(self.cfg, base_rate=rate, side_bias=bias, profile=profile)
        scenario = random_scenario(rng, self.cfg, self.net) if rng.random() < t.p_scenario else EMPTY
        seed = int(t.seed) * 100_003 + self.episode + 1
        return seed, spec, scenario

    def _shape(self, r: np.ndarray) -> np.ndarray:
        d = self.cfg.dqn
        return np.clip(r, -d.reward_clip, d.reward_clip) * d.reward_scale

    # ------------------------------------------------------------------
    def run_episode(self) -> EpisodeStats:
        d = self.cfg.dqn
        env, learner, buf, asm = self.env, self.learner, self.buffer, self.assembler
        t0 = time.perf_counter()
        seed, spec, scenario = self.sample_episode()
        obs, info = env.reset(seed=seed, demand=spec, scenario=scenario)
        mask = info["action_mask"]
        asm.reset()
        ret = np.zeros(env.n_agents)
        ret_local = np.zeros(env.n_agents)
        losses, qs, tds = [], [], []
        done = False
        while not done:
            action = learner.act(obs, mask, self.epsilon(), self.rng)
            next_obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            tr = asm.push(obs, info["action_executed"], self._shape(r), next_obs, info["action_mask"])
            if tr is not None:
                buf.add(tr)
            self.total_steps += 1
            if len(buf) >= d.learn_starts:
                for _ in range(d.updates_per_step):
                    m = learner.update(buf.sample(d.batch_size))
                    losses.append(m["loss"])
                    qs.append(m["q_mean"])
                    tds.append(m["td_abs"])
            ret += r
            ret_local += info["r_local"]
            obs, mask = next_obs, info["action_mask"]
        for tr in asm.flush():
            buf.add(tr)
        self.episode += 1
        nan = float("nan")
        stats = EpisodeStats(
            episode=self.episode, total_steps=self.total_steps, epsilon=self.epsilon(),
            ret=float(ret.mean()), ret_local=float(ret_local.mean()),
            loss=float(np.mean(losses)) if losses else nan, q_mean=float(np.mean(qs)) if qs else nan,
            td_abs=float(np.mean(tds)) if tds else nan, buffer=len(buf),
            demand_rate=spec.base_rate, profile=spec.profile, n_events=len(scenario.events),
            wall_time=time.perf_counter() - t0, summary=env.summary())
        self.history.append(stats)
        return stats

    def evaluate(self, seeds=None, scenario: Scenario | None = None) -> dict:
        """Greedy policy on fixed seeds at the config's default demand. Mean of each KPI."""
        seeds = tuple(self.cfg.train.eval_seeds if seeds is None else seeds)
        ctrl = DQNController(self.learner)
        spec = DemandSpec.from_config(self.cfg)
        runs = [run_episode(self.env, ctrl, seed=int(s), demand=None if scenario is not None else spec,
                            scenario=scenario) for s in seeds]
        out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], (int, float))}
        out["episode"] = self.episode
        out["n_seeds"] = len(seeds)
        return out

    def run(self, episodes: int | None = None) -> list:
        """Train for ``episodes`` (default ``train.episodes``). Returns the list of EpisodeStats."""
        t = self.cfg.train
        if episodes is not None:
            self.planned_episodes = max(self.planned_episodes if self.episode else 0, self.episode + episodes)
        target = self.planned_episodes if episodes is None else self.episode + episodes
        snaps = set(int(e) for e in t.snapshot_episodes)
        if self.episode == 0 and 0 in snaps and self.on_snapshot:
            self.on_snapshot(0, self.learner.state_dict())
        while self.episode < target:
            stats = self.run_episode()
            if self.on_episode_end:
                self.on_episode_end(stats)
            if self.episode in snaps and self.on_snapshot:
                self.on_snapshot(self.episode, self.learner.state_dict())
            if t.eval_every > 0 and (self.episode % t.eval_every == 0 or self.episode == target):
                result = self.evaluate()
                self.eval_history.append(result)
                if self.on_eval:
                    self.on_eval(result)
        return self.history
