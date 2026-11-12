"""TrafficEnv: the multi-agent decision process.

One ``step(actions)`` = one decision interval (``signal.decision_dt`` seconds,
``cfg.substeps`` simulation steps). The same thing is available in three calls
for a real-time server that wants to render every simulation step:

    env.apply_actions(actions)
    while not env.tick():          # one simulation step each
        render(env.sim.snapshot())
    obs, rewards, terminated, truncated, info = env.collect()

There are no terminal states: ``terminated`` is always False and ``truncated``
only marks the time limit, so learners must keep bootstrapping.
"""
from __future__ import annotations

import numpy as np

from .config import Config, N_ACTIONS, OBS_DIM
from .demand import DemandSpec
from .metrics import MetricsTracker
from .network import Network
from .observation import build_observation, phase_demand, phase_pressure, phase_wait
from .reward import compute_reward
from .scenarios import EMPTY, Scenario
from .simulator import TrafficSim


class TrafficEnv:
    obs_dim = OBS_DIM
    n_actions = N_ACTIONS

    def __init__(self, cfg: Config, seed: int = 0, demand: DemandSpec | None = None,
                 scenario: Scenario | None = None, net: Network | None = None,
                 episode_len: float | None = None):
        self.cfg = cfg
        self.net = net if net is not None else Network(cfg)
        self.n_agents = self.net.n
        self.episode_len = float(cfg.env.episode_len if episode_len is None else episode_len)
        self._default_scenario: Scenario = scenario if scenario is not None else EMPTY
        self._default_demand: DemandSpec | None = demand
        self.scenario: Scenario = self._default_scenario
        self.demand: DemandSpec | None = demand          # None -> config defaults + scenario overrides
        self.seed = int(seed)
        self.sim = TrafficSim(cfg, seed=self.seed, demand=self._spec(), net=self.net)
        self.metrics = MetricsTracker(cfg)
        self._cursor = 0
        self._ticks = 0
        self._executed = np.zeros(self.n_agents, dtype=np.int8)
        self._needs_reset = False

    def _spec(self) -> DemandSpec:
        if self.demand is not None:
            return self.demand
        return DemandSpec.from_config(self.cfg, horizon=self.episode_len, **self.scenario.demand)

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, demand: DemandSpec | None = None,
              scenario: Scenario | None = None):
        """Start a new episode.

        ``seed=None`` repeats the previous seed. ``demand`` and ``scenario`` apply to
        this episode only; ``None`` means the constructor's value. With no explicit
        demand the spec is the config default plus the scenario's demand overrides.
        """
        if seed is not None:
            self.seed = int(seed)
        self.scenario = scenario if scenario is not None else self._default_scenario
        self.demand = demand if demand is not None else self._default_demand
        self.sim.reset(self.seed, self._spec())
        self.metrics.reset()
        self._cursor = self.scenario.apply_due(self.sim, 0)
        self._ticks = 0
        self._executed[:] = self.sim.signals.phase
        ls = self.sim.lane_stats()
        dem = phase_demand(self.sim, ls)
        info = {
            "t": self.sim.t,
            "action_mask": self.sim.signals.action_mask(dem),
            "phase_demand": dem,
            "phase_pressure": phase_pressure(self.sim, ls),
            "phase_wait": phase_wait(self.sim, ls),
            "metrics": self.metrics.live(),
        }
        return build_observation(self.sim, ls), info

    # ------------------------------------------------------------------
    def apply_actions(self, actions: np.ndarray) -> np.ndarray:
        """Submit one target phase per agent; returns the phases actually executed."""
        self._executed = self.sim.request_phases(np.asarray(actions))
        return self._executed

    def tick(self) -> bool:
        """Advance one simulation step. True when the next decision is due."""
        self._cursor = self.scenario.apply_due(self.sim, self._cursor)
        self.sim.step()
        self._ticks += 1
        return self._ticks % self.cfg.substeps == 0

    def collect(self):
        """Observation, rewards and info for the interval that just ended."""
        sim = self.sim
        stats = sim.drain_interval_stats()
        r, r_local, terms = compute_reward(stats, self.net, self.cfg)
        self.metrics.update(sim, stats, sim.trip_log())
        ls = sim.lane_stats()
        dem = phase_demand(sim, ls)
        truncated = sim.t >= self.episode_len - 1e-9
        info = {
            "t": sim.t,
            "action_mask": sim.signals.action_mask(dem),
            "action_executed": self._executed,
            "r_local": r_local,
            "reward_terms": terms,
            "phase_demand": dem,
            "phase_pressure": phase_pressure(sim, ls),
            "phase_wait": phase_wait(sim, ls),
            "metrics": self.metrics.live(),
        }
        return build_observation(sim, ls), r, False, truncated, info

    def step(self, actions: np.ndarray):
        self.apply_actions(actions)
        while not self.tick():
            pass
        return self.collect()

    # ------------------------------------------------------------------
    def action_mask(self) -> np.ndarray:
        return self.sim.action_mask()

    def summary(self) -> dict:
        """End-of-episode KPIs, including ``time_in_system``."""
        return self.metrics.summary(self.sim)
