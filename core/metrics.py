"""Key performance indicators.

``MetricsTracker`` is fed once per decision interval with the simulator's
``IntervalStats`` and ``TripLog``. ``live()`` is cheap and meant for the UI;
``summary(sim)`` is the end-of-episode report.

Two travel-time numbers are reported on purpose:

* ``att`` - mean travel time of *completed* trips. Intuitive, but it flatters a
  controller that gridlocks: stuck vehicles never complete and never count.
* ``time_in_system`` - vehicle-seconds spent by *every* generated vehicle up to
  now (finished or not) divided by the number generated. Because demand is
  identical across controllers (CRN), this is the fair number to compare.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import Config, ETYPE_EMERGENCY
from .simulator import IntervalStats, TrafficSim, TripLog


def _ratio(a: float, b: float, default: float = 0.0) -> float:
    return float(a) / float(b) if b else default


class MetricsTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.n_done = 0
        self.tt_sum = 0.0
        self.ff_sum = 0.0
        self.wait_sum = 0.0
        self.stops_sum = 0
        self.prog_sum = 0
        self.prog_runs = 0
        self.ev_n = 0
        self.ev_tt = 0.0
        self.ev_ff = 0.0
        self.ev_stops = 0
        self.halt_sum = 0
        self.steps = 0
        self.cross = [0, 0, 0, 0]            # all, nonstop, arterial, arterial nonstop
        self.speed_sum = 0.0
        self.veh_steps = 0
        self.backlog = 0
        self.n_vehicles = 0
        self.n_spawned = 0
        self.gridlock = False
        self.wait_max_sum = 0.0              # sum over decisions of the mean (over agents) longest wait
        self.wait_max_peak = 0.0
        self.n_updates = 0
        self._waits: list = []               # per-trip total halted time, for percentiles
        self._slow_time = 0.0
        self._win: deque = deque()
        self.series: list = []               # (t, mean halted, mean speed, completed, backlog) per decision

    def update(self, sim: TrafficSim, stats: IntervalStats, trips: TripLog) -> None:
        mc = self.cfg.metrics
        self.t = sim.t
        n = len(trips)
        tt = float(trips.travel_time.sum()) if n else 0.0
        stops = int(trips.n_stops.sum()) if n else 0
        if n:
            self.n_done += n
            self.tt_sum += tt
            self.ff_sum += float(trips.ff_time.sum())
            self.wait_sum += float(trips.wait_total.sum())
            self._waits.append(trips.wait_total)
            self.stops_sum += stops
            self.prog_sum += int(trips.prog_sum.sum())
            self.prog_runs += int(trips.prog_runs.sum())
            ev = trips.etype == ETYPE_EMERGENCY
            if ev.any():
                self.ev_n += int(ev.sum())
                self.ev_tt += float(trips.travel_time[ev].sum())
                self.ev_ff += float(trips.ff_time[ev].sum())
                self.ev_stops += int(trips.n_stops[ev].sum())
        halt = int(stats.halt_sum.sum())
        self.wait_max_sum += float(stats.wait_max.mean())
        self.wait_max_peak = max(self.wait_max_peak, float(stats.wait_max.max()))
        self.n_updates += 1
        self.halt_sum += halt
        self.steps += stats.steps
        c = (stats.cross_all, stats.cross_nonstop, stats.art_all, stats.art_nonstop)
        for i in range(4):
            self.cross[i] += c[i]
        self.speed_sum += stats.speed_sum
        self.veh_steps += stats.veh_steps
        self.backlog = sim.n_backlog
        self.n_vehicles = sim.n_vehicles
        self.n_spawned = sim.n_spawned
        mean_speed = _ratio(stats.speed_sum, stats.veh_steps, self.cfg.vehicle.v0)
        if stats.veh_steps and mean_speed < mc.gridlock_speed:
            self._slow_time += stats.steps * self.cfg.sim.dt
        else:
            self._slow_time = 0.0
        self.gridlock = self.gridlock or self._slow_time >= mc.gridlock_time
        self._win.append((sim.t, n, tt, stops, halt, stats.steps) + c)
        while self._win and sim.t - self._win[0][0] >= mc.rolling_window:
            self._win.popleft()
        self.series.append((sim.t, _ratio(halt, stats.steps), mean_speed, n, sim.n_backlog))

    # ------------------------------------------------------------------
    def live(self) -> dict:
        """Rolling values over the last ``metrics.rolling_window`` seconds (cheap)."""
        w = np.asarray(self._win, dtype=np.float64) if self._win else np.zeros((1, 10))
        s = w.sum(axis=0)
        span = max(self.cfg.signal.decision_dt, min(self.t, self.cfg.metrics.rolling_window))
        return {
            "att": _ratio(s[2], s[1]),
            "throughput": s[1] * 3600.0 / span,
            "mean_queue": _ratio(s[4], s[5]),
            "stops_per_veh": _ratio(s[3], s[1]),
            "gws": _ratio(s[7], s[6]),
            "gws_arterial": _ratio(s[9], s[8]),
            "backlog": int(self.backlog),
            "n_vehicles": int(self.n_vehicles),
            "gridlock": bool(self.gridlock),
        }

    def summary(self, sim: TrafficSim | None = None) -> dict:
        """Whole-episode KPIs. Pass ``sim`` to include ``time_in_system``."""
        out = {
            "t": float(self.t),
            "n_spawned": int(self.n_spawned),
            "n_completed": int(self.n_done),
            "completion_ratio": _ratio(self.n_done, self.n_spawned),
            "att": _ratio(self.tt_sum, self.n_done),
            "delay": _ratio(self.tt_sum - self.ff_sum, self.n_done),
            "wait_per_veh": _ratio(self.wait_sum, self.n_done),
            "wait_p95": float(np.percentile(np.concatenate(self._waits), 95)) if self._waits else 0.0,
            "longest_wait_mean": _ratio(self.wait_max_sum, self.n_updates),   # fairness: typical longest wait at a junction
            "longest_wait_peak": float(self.wait_max_peak),
            "throughput": _ratio(self.n_done * 3600.0, self.t),
            "mean_queue": _ratio(self.halt_sum, self.steps),
            "mean_speed": _ratio(self.speed_sum, self.veh_steps, self.cfg.vehicle.v0),
            "stops_per_veh": _ratio(self.stops_sum, self.n_done),
            "gws": _ratio(self.cross[1], self.cross[0]),
            "gws_arterial": _ratio(self.cross[3], self.cross[2]),
            "progression_len": _ratio(self.prog_sum, self.prog_runs),
            "ev_count": int(self.ev_n),
            "ev_delay": _ratio(self.ev_tt - self.ev_ff, self.ev_n),
            "ev_stops": _ratio(self.ev_stops, self.ev_n),
            "backlog": int(self.backlog),
            "gridlock": bool(self.gridlock),
        }
        if sim is not None:
            out["time_in_system"] = _ratio(self.tt_sum + sim.unfinished_time(), self.n_spawned)
        return out

    def recovery_time(self, t_event: float, tol: float = 0.2, pre_window: float = 120.0,
                      hold: float = 60.0) -> float:
        """Seconds after ``t_event`` until the mean queue is back within ``tol`` of its
        pre-event level and stays there for ``hold`` seconds (inf if it never does).

        WARNING: only meaningful if the network was in steady state before the event.
        In this project's scenarios the grid is still filling at t = 300 s, so this
        reports a long "recovery" even with no event at all. For comparing controllers
        use ``excess_recovery`` below, which subtracts a no-event run on the same seed.
        """
        if not self.series:
            return float("inf")
        s = np.asarray(self.series, dtype=np.float64)
        t, q = s[:, 0], s[:, 1]
        pre = q[(t > t_event - pre_window) & (t <= t_event)]
        if not pre.size:
            return float("inf")
        limit = max(pre.mean() * (1.0 + tol), 1.0)
        after = t > t_event
        good = q <= limit
        need = max(1, int(round(hold / self.cfg.signal.decision_dt)))
        idx = np.flatnonzero(after)
        for i in idx:
            if i + need <= len(q) and good[i:i + need].all():
                return float(t[i] - t_event)
        return float("inf")


def excess_recovery(series_event, series_base, t_event: float, decision_dt: float,
                    hold: float = 60.0, floor: float = 5.0, frac: float = 0.10) -> dict:
    """Effect of a perturbation, measured against the SAME seed without it.

    ``series_event`` and ``series_base`` are ``MetricsTracker.series`` lists from two
    episodes that differ only in the scenario's events (common random numbers make the
    traffic otherwise identical). With ``excess(t) = queue_event(t) - queue_base(t)``:

    * ``excess_veh_hours``  extra queueing caused by the event, integrated from the event
      to the end of the episode (vehicle-hours; can be slightly negative by chance)
    * ``peak_excess``       largest extra queue (vehicles)
    * ``recovery_time``     seconds from the event until the excess queue is back under
      ``max(floor, frac * mean base queue)`` and stays there for ``hold`` seconds; capped
      at the time remaining in the episode when it never recovers
    """
    a = np.asarray(series_event, dtype=np.float64)
    b = np.asarray(series_base, dtype=np.float64)
    n = min(len(a), len(b))
    t, excess = a[:n, 0], a[:n, 1] - b[:n, 1]
    after = t > t_event
    remaining = float(t[-1] - t_event) if n else 0.0
    if not after.any():
        return {"excess_veh_hours": 0.0, "peak_excess": 0.0, "recovery_time": 0.0}
    limit = max(floor, frac * float(b[:n, 1][after].mean()))
    need = max(1, int(round(hold / decision_dt)))
    idx = np.flatnonzero(after)
    disturbed = np.flatnonzero(after & (excess > limit))
    recovery = 0.0
    if disturbed.size:                                   # search only after the disturbance began
        recovery = remaining
        for i in idx[idx > disturbed[0]]:
            if i + need <= n and (excess[i:i + need] <= limit).all():
                recovery = float(t[i] - t_event)
                break
    return {"excess_veh_hours": float(excess[after].sum() * decision_dt / 3600.0),
            "peak_excess": float(excess[after].max()), "recovery_time": recovery}
