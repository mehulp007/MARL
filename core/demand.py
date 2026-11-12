"""Traffic demand: Poisson arrivals, origin-destination model, profiles.

Common random numbers: everything here is drawn from one seeded stream that
depends only on time and on the rate history (base rate, surges) — never on the
state of the traffic or the signals. Two simulators with the same seed and the
same perturbations therefore receive identical vehicles (ids, entries,
destinations, driver parameters) whatever their controllers do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .config import Config, ETYPE_CAR, ETYPE_EMERGENCY
from .network import Network


@dataclass(frozen=True)
class DemandSpec:
    base_rate: float                 # veh/h per entry link
    side_bias: tuple                 # multiplier per city edge (N, E, S, W)
    profile: str                     # "flat" | "rush" | "waves"
    horizon: float                   # seconds the profile is shaped over
    p_straight: float
    ev_rate: float                   # emergency vehicles per hour, network-wide
    ev_p_straight: float
    rush_peak: float
    rush_center_frac: float
    rush_width_frac: float
    wave_period: float
    wave_amp: float

    @classmethod
    def from_config(cls, cfg: Config, **overrides) -> "DemandSpec":
        d = cfg.demand
        spec = cls(base_rate=d.base_rate, side_bias=tuple(d.side_bias), profile=d.profile,
                   horizon=cfg.env.episode_len, p_straight=d.p_straight, ev_rate=d.ev_rate,
                   ev_p_straight=d.ev_p_straight, rush_peak=d.rush_peak,
                   rush_center_frac=d.rush_center_frac, rush_width_frac=d.rush_width_frac,
                   wave_period=d.wave_period, wave_amp=d.wave_amp)
        return replace(spec, **overrides) if overrides else spec


@dataclass
class Arrivals:
    """Vehicles generated in one step (parallel arrays)."""
    entry: np.ndarray        # entry link id
    dest: np.ndarray         # exit link id
    v0_factor: np.ndarray    # driver desired-speed factor
    etype: np.ndarray        # ETYPE_CAR / ETYPE_EMERGENCY

    def __len__(self) -> int:
        return int(self.entry.size)


class DemandModel:
    def __init__(self, cfg: Config, net: Network, spec: DemandSpec, seed: int):
        self.cfg = cfg
        self.net = net
        self.spec = spec
        self.dt = cfg.sim.dt
        self.entries = net.entry_links
        self.n_entries = int(self.entries.size)
        self._side = net.link_side[self.entries].astype(np.intp)
        self._ns_side = (self._side % 2) == 0            # entries on the north/south edges
        self._entry_index = {int(l): i for i, l in enumerate(self.entries)}
        # per entry: every exit except straight-across and the one beside the entry
        ex = net.exit_links
        self._straight = net.straight_exit[self.entries]
        self._others = np.stack([
            ex[(ex != net.straight_exit[l]) & (ex != net.uturn_exit[l])] for l in self.entries])
        self._surges: list = []                          # (entry mask, multiplier, t_end)
        self._surge_mult = np.ones(self.n_entries)
        self.rng = np.random.default_rng([int(seed), 1])

    # ---- rates ---------------------------------------------------------
    def profile(self, t: float) -> np.ndarray:
        """Time-of-episode multiplier per entry."""
        s = self.spec
        if s.profile == "flat":
            return np.ones(self.n_entries)
        if s.profile == "rush":
            half = max(0.5 * s.rush_width_frac * s.horizon, 1e-9)
            u = min(1.0, abs(t - s.rush_center_frac * s.horizon) / half)
            return np.full(self.n_entries, 1.0 + (s.rush_peak - 1.0) * 0.5 * (1.0 + math.cos(math.pi * u)))
        if s.profile == "waves":
            w = s.wave_amp * math.sin(2.0 * math.pi * t / s.wave_period)
            return np.where(self._ns_side, 1.0 + w, 1.0 - w)
        raise ValueError(f"unknown demand profile {s.profile!r}")

    def rates(self, t: float) -> np.ndarray:
        """Arrival rate per entry link in vehicles/hour at time ``t``."""
        s = self.spec
        bias = np.asarray(s.side_bias, dtype=np.float64)[self._side]
        return s.base_rate * bias * self.profile(t) * self._surge_mult

    # ---- perturbations -------------------------------------------------
    def add_surge(self, entry_links, multiplier: float, t_end: float) -> None:
        m = np.zeros(self.n_entries, dtype=bool)
        for l in entry_links:
            m[self._entry_index[int(l)]] = True
        self._surges.append((m, float(multiplier), float(t_end)))
        self._refresh_surge()

    def clear_surges(self) -> None:
        self._surges = []
        self._refresh_surge()

    def expire(self, t: float) -> None:
        if self._surges and any(s[2] <= t for s in self._surges):
            self._surges = [s for s in self._surges if s[2] > t]
            self._refresh_surge()

    def _refresh_surge(self) -> None:
        mult = np.ones(self.n_entries)
        for m, k, _ in self._surges:
            mult[m] *= k
        self._surge_mult = mult

    def set_base_rate(self, base_rate: float) -> None:
        self.spec = replace(self.spec, base_rate=float(base_rate))

    # ---- generation ----------------------------------------------------
    def step(self, t: float) -> Arrivals:
        """Draw the arrivals of the step starting at ``t``."""
        s, vc, rng = self.spec, self.cfg.vehicle, self.rng
        counts = rng.poisson(self.rates(t) * (self.dt / 3600.0))
        n_ev = int(rng.poisson(s.ev_rate * self.dt / 3600.0))
        n_car = int(counts.sum())
        n = n_car + n_ev
        if n == 0:
            return _EMPTY
        ei = np.repeat(np.arange(self.n_entries), counts)
        if n_ev:
            ei = np.concatenate([ei, rng.integers(0, self.n_entries, n_ev)])
        etype = np.full(n, ETYPE_CAR, dtype=np.int8)
        etype[n_car:] = ETYPE_EMERGENCY
        p_straight = np.where(etype == ETYPE_EMERGENCY, s.ev_p_straight, s.p_straight)
        straight = rng.random(n) < p_straight
        col = rng.integers(0, self._others.shape[1], n)
        dest = np.where(straight, self._straight[ei], self._others[ei, col])
        f = np.clip(rng.normal(1.0, vc.v0_sigma, n), vc.v0_clip[0], vc.v0_clip[1])
        return Arrivals(entry=self.entries[ei], dest=dest.astype(np.int32), v0_factor=f, etype=etype)


_EMPTY = Arrivals(entry=np.zeros(0, np.int32), dest=np.zeros(0, np.int32),
                  v0_factor=np.zeros(0), etype=np.zeros(0, np.int8))
