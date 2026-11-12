"""Adversarial scenario search.

Finds bounded perturbation scenarios on which a policy does worst *relative to
a reference controller*:

    J(scenario) = mean over seeds [ metric(policy) - metric(reference) ]      (regret)

Using regret instead of the raw metric makes the adversary look for scenarios
where the policy is specifically weak, not scenarios that are merely heavy for
everyone. The search is the Cross-Entropy Method over a mixed distribution:

* per event slot, a categorical over the event kind (including "none");
* per event slot, a clipped Gaussian over 6 numbers in [0, 1]:
  time, x, y, magnitude, duration, aux.

Locations are searched as a point (x, y) in the city and snapped to the nearest
suitable link, so "nearby" in parameter space means nearby on the map.

Every candidate is evaluated on the same seeds (common random numbers), and a
budget projection keeps the adversary bounded (``cfg.adversary``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .baselines import Controller, run_episode
from .config import Config, HEADING_NAMES
from .env import TrafficEnv
from .network import Network
from .scenarios import Event, Scenario

KINDS = ("none", "incident", "closure", "surge", "emergency", "jam")
N_CONT = 6                       # time, x, y, magnitude, duration, aux
_T, _X, _Y, _MAG, _DUR, _AUX = range(N_CONT)

# search ranges
T_RANGE = (0.05, 0.60)           # event time as a fraction of the episode
INCIDENT_DUR = (120.0, 600.0)
CLOSURE_DUR_MIN = 120.0
SURGE_MULT_MIN = 1.5
SURGE_DUR = (120.0, 480.0)
JAM_N = (10, 40)
SURGE_SIDES = HEADING_NAMES + ("all",)


@dataclass(frozen=True)
class AdversaryResult:
    scenario: Scenario
    regret: float                # policy metric - reference metric (higher = worse for the policy)
    policy_value: float
    reference_value: float
    metric: str
    seeds: tuple

    def __iter__(self):          # lets callers write: for scenario, regret in results
        yield self.scenario
        yield self.regret

    def to_dict(self) -> dict:
        return {"scenario": self.scenario.to_dict(), "regret": float(self.regret),
                "policy_value": float(self.policy_value), "reference_value": float(self.reference_value),
                "metric": self.metric, "seeds": [int(s) for s in self.seeds]}

    @classmethod
    def from_dict(cls, d: dict) -> "AdversaryResult":
        return cls(Scenario.from_dict(d["scenario"]), float(d["regret"]), float(d["policy_value"]),
                   float(d["reference_value"]), d["metric"], tuple(d["seeds"]))


def project_budget(events, cfg: Config) -> tuple:
    """Clamp a list of events to the adversary's budget (``cfg.adversary``).

    * at most ``max_events`` events (earliest first) and ``max_incidents`` incidents
    * surge multiplier <= ``max_surge``
    * total closure time <= ``max_closure_time`` (durations scaled down proportionally)
    * jam size <= 40 vehicles
    """
    a = cfg.adversary
    out, n_inc = [], 0
    for e in sorted(events, key=lambda e: e.t):
        if len(out) >= a.max_events:
            break
        p = dict(e.params)
        if e.kind == "incident":
            if n_inc >= a.max_incidents:
                continue
            n_inc += 1
            if p.get("duration") is None:
                p["duration"] = INCIDENT_DUR[1]
            p["duration"] = float(min(p["duration"], INCIDENT_DUR[1]))
        elif e.kind == "surge":
            p["multiplier"] = float(min(p["multiplier"], a.max_surge))
            p["duration"] = float(min(p["duration"], SURGE_DUR[1]))
        elif e.kind == "jam":
            p["n"] = int(min(p["n"], JAM_N[1]))
        elif e.kind == "closure":
            p["duration"] = float(a.max_closure_time if p.get("duration") is None else p["duration"])
        out.append(Event(e.t, e.kind, p))
    total = sum(e.params["duration"] for e in out if e.kind == "closure")
    if total > a.max_closure_time:
        k = a.max_closure_time / total
        out = [Event(e.t, e.kind, {**e.params, "duration": float(np.floor(e.params["duration"] * k))})
               if e.kind == "closure" else e for e in out]
    return tuple(out)


class Adversary:
    def __init__(self, policy: Controller, reference: Controller, cfg: Config,
                 rng: np.random.Generator, net: Network | None = None,
                 metric: str = "time_in_system", base_demand: dict | None = None,
                 kinds: tuple = KINDS, seeds=None):
        self.policy = policy
        self.reference = reference
        self.cfg = cfg
        self.rng = rng
        self.net = net if net is not None else Network(cfg)
        self.metric = metric
        self.base_demand = dict(base_demand or {})
        self.kinds = tuple(kinds)
        a = cfg.adversary
        self.episode_len = float(a.episode_len)
        self.env = TrafficEnv(cfg, net=self.net, episode_len=self.episode_len)
        self.seeds = tuple(int(s) for s in (seeds if seeds is not None
                                            else rng.integers(1, 1_000_000, a.n_seeds)))
        self._cache: dict = {}
        self.n_evaluations = 0
        self.history: list = []          # per iteration: dict(best, mean, elite_mean, probs)
        net_ = self.net
        g = cfg.grid
        # link "handles" for snapping a point to a link: carriageway midpoints / entry mouths
        mid = 0.5 * net_.link_len
        hx = net_.link_x0 + mid * net_.link_ux + g.lane_w * net_.link_rx
        hy = net_.link_y0 + mid * net_.link_uy + g.lane_w * net_.link_ry
        self._incoming = np.flatnonzero(~net_.is_exit)
        self._internal = net_.internal_links if net_.internal_links.size else self._incoming
        self._entries = net_.entry_links
        self._hx, self._hy = hx, hy
        self._ex = net_.link_x0[self._entries] + g.lane_w * net_.link_rx[self._entries]
        self._ey = net_.link_y0[self._entries] + g.lane_w * net_.link_ry[self._entries]
        pad = g.box + g.link_len
        self._x_range = (-pad, float(net_.node_x.max()) + pad)
        self._y_range = (-pad, float(net_.node_y.max()) + pad)

    # ------------------------------------------------------------------
    def _nearest(self, pool: np.ndarray, x: float, y: float) -> int:
        return int(pool[np.hypot(self._hx[pool] - x, self._hy[pool] - y).argmin()])

    def decode(self, kind_idx: np.ndarray, u: np.ndarray) -> Scenario:
        """Turn one sample (kinds (E,), numbers (E, 6) in [0, 1]) into a budget-feasible Scenario."""
        cfg, a, g = self.cfg, self.cfg.adversary, self.cfg.grid
        lerp = lambda r, s: r[0] + s * (r[1] - r[0])
        events = []
        for k, v in zip(kind_idx, u):
            kind = self.kinds[int(k)]
            if kind == "none":
                continue
            t = float(np.round(lerp(T_RANGE, v[_T]) * self.episode_len))
            x, y = lerp(self._x_range, v[_X]), lerp(self._y_range, v[_Y])
            if kind == "incident":
                p = {"link": self._nearest(self._incoming, x, y), "lane": int(v[_AUX] > 0.75),
                     "pos": float(np.round(lerp((40.0, g.link_len - 20.0), v[_MAG]))),
                     "duration": float(np.round(lerp(INCIDENT_DUR, v[_DUR])))}
            elif kind == "closure":
                p = {"link": self._nearest(self._internal, x, y),
                     "duration": float(np.round(lerp((CLOSURE_DUR_MIN, a.max_closure_time), v[_DUR])))}
            elif kind == "surge":
                p = {"side": SURGE_SIDES[min(len(SURGE_SIDES) - 1, int(v[_AUX] * len(SURGE_SIDES)))],
                     "multiplier": float(np.round(lerp((SURGE_MULT_MIN, a.max_surge), v[_MAG]), 2)),
                     "duration": float(np.round(lerp(SURGE_DUR, v[_DUR])))}
            elif kind == "emergency":
                p = {"entry": int(self._entries[np.hypot(self._ex - x, self._ey - y).argmin()])}
            else:  # jam
                p = {"link": self._nearest(self._incoming, x, y),
                     "n": int(np.round(lerp(JAM_N, v[_MAG])))}
            events.append(Event(t, kind, p))
        return Scenario("adversarial", project_budget(events, cfg), dict(self.base_demand),
                        "Found by cross-entropy search against a reference controller.")

    def evaluate(self, scenario: Scenario):
        """Returns ``(regret, policy_value, reference_value)`` averaged over the fixed seeds."""
        key = json.dumps(scenario.to_dict(), sort_keys=True)
        if key not in self._cache:
            pv = np.mean([run_episode(self.env, self.policy, seed=s, scenario=scenario)[self.metric]
                          for s in self.seeds])
            rv = np.mean([run_episode(self.env, self.reference, seed=s, scenario=scenario)[self.metric]
                          for s in self.seeds])
            self._cache[key] = (float(pv - rv), float(pv), float(rv))
            self.n_evaluations += 1
        return self._cache[key]

    def baseline(self):
        """Regret with no perturbation at all (what the adversary has to beat)."""
        return self.evaluate(Scenario("unperturbed", (), dict(self.base_demand)))

    # ------------------------------------------------------------------
    def search(self, top_k: int = 5, on_iteration=None, smoothing: float = 0.7) -> list:
        """Run CEM; returns up to ``top_k`` distinct scenarios, worst for the policy first."""
        a, rng = self.cfg.adversary, self.rng
        E, K, D = a.max_events, len(self.kinds), N_CONT
        probs = np.full((E, K), 1.0 / K)
        mean = np.full((E, D), 0.5)
        std = np.full((E, D), 0.3)
        archive: dict = {}
        for it in range(a.iterations):
            kinds = np.stack([rng.choice(K, size=a.population, p=probs[e]) for e in range(E)], axis=1)
            u = np.clip(rng.normal(mean, std, size=(a.population, E, D)), 0.0, 1.0)
            scores = np.empty(a.population)
            for i in range(a.population):
                sc = self.decode(kinds[i], u[i])
                reg, pv, rv = self.evaluate(sc)
                scores[i] = reg
                archive[json.dumps(sc.to_dict(), sort_keys=True)] = (sc, reg, pv, rv)
            elite = np.argsort(scores)[-a.elites:]
            for e in range(E):
                freq = np.bincount(kinds[elite, e], minlength=K) / a.elites
                probs[e] = (1.0 - smoothing) * probs[e] + smoothing * freq
                probs[e] = np.maximum(probs[e], 0.02)
                probs[e] /= probs[e].sum()
            mean = (1.0 - smoothing) * mean + smoothing * u[elite].mean(axis=0)
            std = np.maximum((1.0 - smoothing) * std + smoothing * u[elite].std(axis=0), 0.05)
            rec = {"iteration": it + 1, "best": float(scores.max()), "mean": float(scores.mean()),
                   "elite_mean": float(scores[elite].mean()),
                   "best_so_far": float(max(v[1] for v in archive.values())),
                   "evaluations": self.n_evaluations}
            self.history.append(rec)
            if on_iteration:
                on_iteration(rec)
        ranked = sorted(archive.values(), key=lambda v: -v[1])[:top_k]
        return [AdversaryResult(sc, reg, pv, rv, self.metric, self.seeds) for sc, reg, pv, rv in ranked]


def search(policy: Controller, reference: Controller, cfg: Config, rng: np.random.Generator,
           net: Network | None = None, top_k: int = 5, on_iteration=None, **kwargs) -> list:
    """Convenience wrapper: ``[(Scenario, regret), ...]`` (as ``AdversaryResult`` objects)."""
    return Adversary(policy, reference, cfg, rng, net=net, **kwargs).search(top_k=top_k,
                                                                           on_iteration=on_iteration)
