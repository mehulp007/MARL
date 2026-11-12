"""Signal state machines for all intersections, vectorised.

GREEN(p) --request q != p--> YELLOW(p) --> ALL_RED --> GREEN(q)

Timing is counted in whole simulation steps so durations are exact. The lane
signal returned by ``lane_signal()`` is the state that holds for the *coming*
step; ``advance()`` is called once at the end of every simulation step.
"""
from __future__ import annotations

import numpy as np

from .config import (
    Config,
    N_PHASES,
    SIG_GREEN,
    SIG_RED,
    SIG_YELLOW,
    STAGE_ALL_RED,
    STAGE_GREEN,
    STAGE_YELLOW,
)
from .network import Network


class Signals:
    def __init__(self, cfg: Config, net: Network):
        s, dt = cfg.signal, cfg.sim.dt
        self.net = net
        self.n = net.n
        self.dt = dt
        self.yellow_steps = int(round(s.yellow / dt))
        self.all_red_steps = int(round(s.all_red / dt))
        self.g_min = s.g_min
        self.g_max = s.g_max
        self.decision_dt = s.decision_dt
        self._ar = np.arange(self.n)
        self._base = np.where(net.lane_is_exit, SIG_GREEN, SIG_RED).astype(np.int8)
        self._sig = self._base.copy()
        self.phase = np.zeros(self.n, dtype=np.int8)      # green phase (old phase while YELLOW / ALL_RED)
        self.pending = np.zeros(self.n, dtype=np.int8)    # phase that will be green next
        self.stage = np.zeros(self.n, dtype=np.int8)
        self.timer = np.zeros(self.n, dtype=np.int32)     # steps left in YELLOW / ALL_RED
        self.green_steps = np.zeros(self.n, dtype=np.int32)
        self.reset()

    def reset(self) -> None:
        self.phase[:] = 0
        self.pending[:] = 0
        self.stage[:] = STAGE_GREEN
        self.timer[:] = 0
        self.green_steps[:] = 0

    # ---- read-outs -----------------------------------------------------
    @property
    def green_elapsed(self) -> np.ndarray:
        """Seconds of green so far (0 during YELLOW / ALL_RED)."""
        return self.green_steps * self.dt

    @property
    def target_phase(self) -> np.ndarray:
        """Current green phase, or the phase being switched to."""
        return np.where(self.stage == STAGE_GREEN, self.phase, self.pending).astype(np.int8)

    def lane_signal(self) -> np.ndarray:
        """(n_lanes,) int8 of SIG_*; exit lanes are always GREEN. Internal buffer."""
        sig = self._sig
        sig[:] = self._base
        lanes = self.net.phase_lanes[self._ar, self.phase]          # (N, 2)
        g = self.stage == STAGE_GREEN
        y = self.stage == STAGE_YELLOW
        sig[lanes[g].ravel()] = SIG_GREEN
        if y.any():
            sig[lanes[y].ravel()] = SIG_YELLOW
        return sig

    def action_mask(self, demand: np.ndarray) -> np.ndarray:
        """(N, 4) bool of valid actions; ``demand`` is (N, 4) vehicles per phase.

        * switching needs ``green_elapsed >= g_min``
        * keeping is invalid once another decision interval would exceed ``g_max``
          while some other phase has demand; then only phases with demand are valid
        * during YELLOW / ALL_RED only the pending phase is valid
        """
        ar = self._ar
        el = self.green_elapsed
        green = self.stage == STAGE_GREEN
        cur = self.phase
        mask = np.repeat((green & (el >= self.g_min))[:, None], N_PHASES, axis=1)
        other = demand.copy()
        other[ar, cur] = 0
        must = green & (el + self.decision_dt > self.g_max) & (other.sum(axis=1) > 0) & (el >= self.g_min)
        mask[ar, cur] = green & ~must
        if must.any():
            mask[must] = other[must] > 0
        tr = ~green
        if tr.any():
            mask[tr] = False
            mask[tr, self.pending[tr]] = True
        return mask

    # ---- control -------------------------------------------------------
    def request(self, phases: np.ndarray, demand: np.ndarray):
        """Ask for a target phase per intersection.

        Invalid requests are replaced by "keep" if valid, else by the valid phase
        with the highest demand. Returns ``(executed, switched)``.
        """
        ar = self._ar
        req = np.clip(np.asarray(phases, dtype=np.int64), 0, N_PHASES - 1)
        mask = self.action_mask(demand)
        best = np.where(mask, demand, -1).argmax(axis=1)
        executed = np.where(mask[ar, req], req, np.where(mask[ar, self.phase], self.phase, best)).astype(np.int8)
        sw = (self.stage == STAGE_GREEN) & (executed != self.phase)
        if sw.any():
            self.pending[sw] = executed[sw]
            self.stage[sw] = STAGE_YELLOW
            self.timer[sw] = self.yellow_steps
            self.green_steps[sw] = 0
            if self.yellow_steps == 0:
                self._end_yellow(sw)
        return executed, sw

    def advance(self) -> None:
        """Advance every state machine by one simulation step."""
        stage = self.stage
        g = stage == STAGE_GREEN
        y = stage == STAGE_YELLOW
        r = stage == STAGE_ALL_RED
        self.green_steps[g] += 1
        if y.any():
            self.timer[y] -= 1
            self._end_yellow(y & (self.timer <= 0))
        if r.any():
            self.timer[r] -= 1
            self._end_all_red(r & (self.timer <= 0))

    def _end_yellow(self, m: np.ndarray) -> None:
        if m.any():
            self.stage[m] = STAGE_ALL_RED
            self.timer[m] = self.all_red_steps
            if self.all_red_steps == 0:
                self._end_all_red(m)

    def _end_all_red(self, m: np.ndarray) -> None:
        if m.any():
            self.stage[m] = STAGE_GREEN
            self.phase[m] = self.pending[m]
            self.timer[m] = 0
            self.green_steps[m] = 0
