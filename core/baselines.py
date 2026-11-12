"""Non-learning controllers and the episode runner.

Every controller implements the same two-method protocol as the learned agent,
so the server, the evaluator and the adversary can swap them freely.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import N_PHASES
from .env import TrafficEnv


class Controller(Protocol):
    def reset(self, env: TrafficEnv) -> None: ...
    def act(self, obs: np.ndarray, mask: np.ndarray, info: dict) -> np.ndarray: ...


class FixedTimeController:
    """The rigid timer the project argues against: a fixed cycle 0 -> 1 -> 2 -> 3.

    ``greens`` are green seconds per phase. Signals can only change at decision
    instants, so each phase gets ``round((green + yellow + all_red) / decision_dt)``
    decision slots; with the defaults the plan is exact. ``offset_fn(net, cfg)``
    may return a per-intersection time offset in seconds (zero by default, i.e.
    every light runs the same clock).
    """

    def __init__(self, greens=None, offset_fn=None):
        self.greens = greens
        self.offset_fn = offset_fn

    def reset(self, env: TrafficEnv) -> None:
        cfg, sg = env.cfg, env.cfg.signal
        lost = sg.yellow + sg.all_red
        d = sg.decision_dt
        greens = self.greens if self.greens is not None else cfg.baseline.fixed_greens
        min_slots = int(np.ceil((sg.g_min + lost) / d - 1e-9))
        slots = [max(min_slots, int(round((g + lost) / d))) for g in greens]
        self.table = np.repeat(np.arange(N_PHASES), slots)
        self.d = d
        self.actual_greens = tuple(s * d - lost for s in slots)
        self.cycle_time = len(self.table) * d
        off = np.zeros(env.n_agents) if self.offset_fn is None else np.asarray(self.offset_fn(env.net, cfg))
        self.offset_slots = np.round(off / d).astype(np.int64)

    def act(self, obs, mask, info) -> np.ndarray:
        k = int(round(info["t"] / self.d))
        return self.table[(k - self.offset_slots) % len(self.table)]


def progression_offsets(net, cfg) -> np.ndarray:
    """Hand-tuned reference offsets: a green wave for eastbound and southbound traffic."""
    travel = cfg.block / cfg.vehicle.v0
    return (net.node_r + net.node_c) * travel


class MaxPressureController:
    """Serve the phase with the highest pressure (queued demand minus downstream load).

    Decentralised and non-learning. ``min_green`` (seconds) keeps a phase that still
    has pressure for at least that long; ``margin`` is the pressure advantage another
    phase needs before the controller pays the switching loss.

    ``max_wait`` (seconds) is a starvation guard: pure max-pressure can leave a lone
    vehicle on a side movement waiting for many minutes, so a phase holding a vehicle
    that has waited longer than this is served next (longest wait first). Pass
    ``max_wait=float("inf")`` for textbook max-pressure.
    """

    def __init__(self, min_green: float | None = None, margin: float | None = None,
                 max_wait: float | None = None):
        self.min_green = min_green
        self.margin = margin
        self.max_wait = max_wait

    def reset(self, env: TrafficEnv) -> None:
        self.env = env
        b = env.cfg.baseline
        self._min_green = b.mp_min_green if self.min_green is None else self.min_green
        self._margin = b.mp_margin if self.margin is None else self.margin
        self._max_wait = b.mp_max_wait if self.max_wait is None else self.max_wait
        self._ar = np.arange(env.n_agents)

    def act(self, obs, mask, info) -> np.ndarray:
        sg = self.env.sim.signals
        ar, cur = self._ar, sg.phase.astype(np.intp)
        pr = info["phase_pressure"]
        best = np.where(mask, pr, -np.inf).argmax(axis=1)
        p_cur, p_best = pr[ar, cur], pr[ar, best]
        hold = (sg.green_elapsed < self._min_green) & (p_cur > 0)
        keep = mask[ar, cur] & (hold | (p_cur >= p_best - self._margin))
        action = np.where(keep, cur, best)
        wait = np.where(mask, info["phase_wait"], -1.0)
        wait[ar, cur] = -1.0                              # the current phase is already being served
        starving = wait.max(axis=1) > self._max_wait
        if starving.any():
            action = np.where(starving & ~hold, wait.argmax(axis=1), action)
        return action


class RandomController:
    """Uniformly random valid action: the 'Initial Chaos' reference."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def reset(self, env: TrafficEnv) -> None:
        self.rng = np.random.default_rng([int(self.seed), int(env.seed), 3])

    def act(self, obs, mask, info) -> np.ndarray:
        return np.where(mask, self.rng.random(mask.shape), -1.0).argmax(axis=1)


def run_episode(env: TrafficEnv, controller: Controller, seed: int | None = None,
                demand=None, scenario=None) -> dict:
    """Run one full episode and return ``env.summary()`` plus the mean return per agent."""
    obs, info = env.reset(seed=seed, demand=demand, scenario=scenario)
    controller.reset(env)
    total = np.zeros(env.n_agents)
    local = np.zeros(env.n_agents)
    done = False
    while not done:
        obs, r, terminated, truncated, info = env.step(controller.act(obs, info["action_mask"], info))
        total += r
        local += info["r_local"]
        done = terminated or truncated
    out = env.summary()
    out["return"] = float(total.mean())
    out["return_local"] = float(local.mean())
    return out
