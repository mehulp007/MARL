"""Perturbation events and scenarios.

A ``Scenario`` is an immutable, JSON-serialisable list of timed ``Event``s plus
optional demand overrides. Applying the same scenario to two simulators gives
identical perturbations, which keeps side-by-side comparisons fair.

Event kinds and their params
----------------------------
incident   link, lane (0 outer / 1 inner), pos, duration (None = until cleared)
closure    link, duration (None = until reopened)
surge      side ("N"|"E"|"S"|"W"|"all") or entries [link, ...]; multiplier; duration
emergency  entry (entry link id); dest (optional exit link id)
jam        link, n
demand     base_rate (veh/h per entry)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config, HEADING_NAMES
from .network import Network
from .simulator import TrafficSim

EVENT_KINDS = ("incident", "closure", "surge", "emergency", "jam", "demand")


@dataclass(frozen=True)
class Event:
    t: float
    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")

    def apply(self, sim: TrafficSim):
        p, net = self.params, sim.net
        if self.kind == "incident":
            return sim.add_incident(p["link"], p.get("lane", 0), p["pos"], p.get("duration"))
        if self.kind == "closure":
            return sim.close_link(p["link"], p.get("duration"))
        if self.kind == "surge":
            return sim.set_surge(_entries(net, p), p["multiplier"], p["duration"])
        if self.kind == "emergency":
            return sim.spawn_emergency(p["entry"], p.get("dest"))
        if self.kind == "jam":
            return sim.inject_jam(p["link"], p["n"])
        if self.kind == "demand":
            return sim.set_demand_level(p["base_rate"])

    def to_dict(self) -> dict:
        return {"t": float(self.t), "kind": self.kind, "params": _plain(self.params)}

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(t=float(d["t"]), kind=d["kind"], params=dict(d.get("params", {})))


@dataclass(frozen=True)
class Scenario:
    name: str = "empty"
    events: tuple = ()
    demand: dict = field(default_factory=dict)      # DemandSpec overrides, e.g. {"base_rate": 700}
    description: str = ""

    def __post_init__(self):
        object.__setattr__(self, "events", tuple(sorted(self.events, key=lambda e: e.t)))

    def apply_due(self, sim: TrafficSim, cursor: int = 0) -> int:
        """Apply every event with ``t <= sim.t`` starting at ``cursor``; returns the new cursor."""
        ev = self.events
        while cursor < len(ev) and ev[cursor].t <= sim.t + 1e-9:
            ev[cursor].apply(sim)
            cursor += 1
        return cursor

    def with_events(self, *events: Event) -> "Scenario":
        return Scenario(self.name, self.events + tuple(events), dict(self.demand), self.description)

    def first_event_time(self) -> float:
        return self.events[0].t if self.events else float("inf")

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "demand": _plain(self.demand),
                "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(name=d.get("name", "scenario"), events=tuple(Event.from_dict(e) for e in d.get("events", [])),
                   demand=dict(d.get("demand", {})), description=d.get("description", ""))


EMPTY = Scenario()


# --------------------------------------------------------------------------
def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def _entries(net: Network, p: dict) -> list:
    if "entries" in p:
        return [int(l) for l in p["entries"]]
    side = p.get("side", "all")
    if side == "all":
        return [int(l) for l in net.entry_links]
    return [int(l) for l in net.entries_by_side[HEADING_NAMES.index(side)]]


def _centre_link(net: Network, heading: int, dr: int = 0, dc: int = 0) -> int:
    """An incoming link near the middle of the grid with the given heading."""
    r = min(max(net.rows // 2 + dr, 0), net.rows - 1)
    c = min(max(net.cols // 2 + dc, 0), net.cols - 1)
    return int(net.in_link[r * net.cols + c, heading])


# --------------------------------------------------------------------------
# Scripted library. Each builder takes (cfg, net) and returns a Scenario.
# --------------------------------------------------------------------------
def _calm_low(cfg, net):
    return Scenario("calm_low", demand={"base_rate": cfg.demand.low},
                    description="Light, steady traffic.")


def _steady_medium(cfg, net):
    return Scenario("steady_medium", demand={"base_rate": cfg.demand.medium},
                    description="Steady traffic near the capacity of a fixed timer.")


def _heavy_high(cfg, net):
    return Scenario("heavy_high", demand={"base_rate": cfg.demand.high},
                    description="Heavy, steady traffic.")


def _rush_hour_wave(cfg, net):
    return Scenario("rush_hour_wave", demand={"base_rate": 450.0, "profile": "rush"},
                    description="Demand builds to a peak mid-episode and falls away.")


def _arterial_incident(cfg, net):
    link = _centre_link(net, heading=2, dr=-1)
    ev = Event(300.0, "incident", {"link": link, "lane": 0, "pos": 0.6 * cfg.grid.link_len, "duration": 600.0})
    return Scenario("arterial_incident", (ev,), {"base_rate": cfg.demand.medium},
                    "An accident blocks the through lane of a central arterial for 10 minutes.")


def _center_closure(cfg, net):
    ev = Event(300.0, "closure", {"link": _centre_link(net, heading=1), "duration": 600.0})
    return Scenario("center_closure", (ev,), {"base_rate": cfg.demand.medium},
                    "A central road is closed for 10 minutes; traffic must re-route.")


def _stadium_surge(cfg, net):
    ev = Event(300.0, "surge", {"side": "N", "multiplier": 2.5, "duration": 300.0})
    return Scenario("stadium_surge", (ev,), {"base_rate": cfg.demand.medium},
                    "Arrivals from the north edge jump to 2.5x for 5 minutes.")


def _emergency_corridor(cfg, net):
    west, north = net.entries_by_side[3], net.entries_by_side[0]
    evs = []
    for i, t in enumerate(np.arange(120.0, cfg.env.episode_len - 180.0, 120.0)):
        src = west if i % 2 == 0 else north
        evs.append(Event(float(t), "emergency", {"entry": int(src[(i // 2) % len(src)])}))
    return Scenario("emergency_corridor", tuple(evs), {"base_rate": cfg.demand.medium, "ev_rate": 0.0},
                    "An emergency vehicle crosses the city every two minutes.")


def _compound_chaos(cfg, net):
    east = net.entries_by_side[1]
    evs = [
        Event(200.0, "incident", {"link": _centre_link(net, heading=0), "lane": 0,
                                  "pos": 0.5 * cfg.grid.link_len, "duration": 500.0}),
        Event(400.0, "surge", {"side": "E", "multiplier": 2.5, "duration": 240.0}),
        Event(500.0, "closure", {"link": _centre_link(net, heading=3, dr=-1), "duration": 400.0}),
        Event(700.0, "jam", {"link": _centre_link(net, heading=2, dc=-1), "n": 25}),
    ]
    evs += [Event(t, "emergency", {"entry": int(east[i % len(east)])})
            for i, t in enumerate((300.0, 600.0, 900.0, 1200.0))]
    return Scenario("compound_chaos", tuple(evs), {"base_rate": 550.0, "ev_rate": 0.0},
                    "Accident, surge, closure, jam and emergency vehicles in one episode.")


LIBRARY = {
    "calm_low": _calm_low,
    "steady_medium": _steady_medium,
    "heavy_high": _heavy_high,
    "rush_hour_wave": _rush_hour_wave,
    "arterial_incident": _arterial_incident,
    "center_closure": _center_closure,
    "stadium_surge": _stadium_surge,
    "emergency_corridor": _emergency_corridor,
    "compound_chaos": _compound_chaos,
}


def get_scenario(name: str, cfg: Config, net: Network) -> Scenario:
    return LIBRARY[name](cfg, net)


# --------------------------------------------------------------------------
def random_event(rng: np.random.Generator, cfg: Config, net: Network, t: float) -> Event:
    """One random perturbation at time ``t`` (used for domain randomisation)."""
    kind = ("incident", "closure", "surge", "emergency", "jam")[
        int(rng.choice(5, p=(0.25, 0.15, 0.25, 0.20, 0.15)))]
    incoming = np.flatnonzero(~net.is_exit)
    if kind == "incident":
        link = int(rng.choice(incoming))
        return Event(t, kind, {"link": link, "lane": int(rng.random() < 0.25),
                               "pos": float(rng.uniform(40.0, cfg.grid.link_len - 20.0)),
                               "duration": float(rng.uniform(120.0, 480.0))})
    if kind == "closure":
        pool = net.internal_links if net.internal_links.size else incoming
        return Event(t, kind, {"link": int(rng.choice(pool)), "duration": float(rng.uniform(120.0, 420.0))})
    if kind == "surge":
        return Event(t, kind, {"side": HEADING_NAMES[int(rng.integers(4))],
                               "multiplier": float(rng.uniform(1.5, 2.5)),
                               "duration": float(rng.uniform(120.0, 360.0))})
    if kind == "emergency":
        return Event(t, kind, {"entry": int(rng.choice(net.entry_links))})
    return Event(t, kind, {"link": int(rng.choice(incoming)), "n": int(rng.integers(10, 31))})


def random_scenario(rng: np.random.Generator, cfg: Config, net: Network, max_events: int = 3) -> Scenario:
    """1..max_events random perturbations in the first 70% of the episode. No demand overrides."""
    n = int(rng.integers(1, max_events + 1))
    T = cfg.env.episode_len
    times = np.sort(rng.uniform(0.1 * T, 0.7 * T, n))
    return Scenario("random", tuple(random_event(rng, cfg, net, float(t)) for t in times))
