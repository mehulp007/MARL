"""The twin simulation behind the UI.

A ``Session`` owns two ``TrafficEnv`` instances, "A" and "B", created with the
same seed and demand. They are advanced in lockstep, each by its own
controller, and every perturbation is applied to both at the same simulation
time, so the only difference between the two cities is who controls the lights.

Everything here is synchronous and runs on the server's event-loop thread;
``app.py`` adds the timing and the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import Config, DemandSpec, Network, TrafficEnv
from core.baselines import FixedTimeController, MaxPressureController, RandomController
from core.config import HEADING_NAMES
from core.scenarios import LIBRARY, Event, Scenario, get_scenario

from . import protocol
from .protocol import ProtocolError

INJECT_LEAD = 10.0        # an injected scenario's first event happens this many seconds from now
LABELS = {"fixed": "Fixed timer", "maxpressure": "Max-pressure", "random": "Random"}


class Side:
    def __init__(self, name: str, env: TrafficEnv):
        self.name = name
        self.env = env
        self.controller = None
        self.kind = ""
        self.checkpoint = None
        self.obs = None
        self.info = None
        self.metrics: dict = {}

    @property
    def label(self) -> str:
        return LABELS.get(self.kind, f"Agents ({self.checkpoint})")


class Session:
    def __init__(self, cfg: Config | None = None, root: str | Path = ".", seed: int = 7, speed: float = 4.0):
        self.cfg = cfg if cfg is not None else Config()
        self.root = Path(root).resolve()
        self.net = Network(self.cfg)
        self.sides = {n: Side(n, TrafficEnv(self.cfg, seed=seed, net=self.net)) for n in protocol.SIDES}
        self.seed = int(seed)
        self.speed = float(speed)
        self.playing = True
        self.demand_level = float(self.cfg.demand.base_rate)
        self.scenario_name: str | None = None
        self._pending: list = []            # (time, Event, source) applied to both sides when due
        self._outbox: list = []             # event messages waiting to be broadcast
        self._controllers: dict = {}        # checkpoint path -> loaded DQNController
        self._next_entry = 0
        self._assign("A", "fixed", None)
        default = "cooperative.pt" if (self.root / "checkpoints" / "cooperative.pt").is_file() else None
        self._assign("B", "dqn" if default else "maxpressure", default)
        self.reset()

    # ------------------------------------------------------------------
    # controllers
    # ------------------------------------------------------------------
    def _checkpoint_path(self, rel: str) -> Path:
        """Resolve a client-supplied name strictly inside ``checkpoints/``."""
        base = (self.root / "checkpoints").resolve()
        path = (base / rel).resolve()
        if base not in path.parents or path.suffix != ".pt" or path.name.startswith("resume") or not path.is_file():
            raise ProtocolError(f"no such checkpoint: {rel}")
        return path

    def _load_dqn(self, rel: str):
        path = self._checkpoint_path(rel)
        key = (str(path), path.stat().st_mtime_ns)
        if key not in self._controllers:
            import torch
            from core.agents import DQNController
            torch.set_num_threads(1)
            # pickle-based format: only files inside this project's checkpoints/ are ever loaded
            state = torch.load(str(path), map_location="cpu", weights_only=False)
            self._controllers[key] = DQNController.from_state_dict(state)
        return self._controllers[key]

    def _assign(self, side: str, kind: str, checkpoint: str | None) -> None:
        s = self.sides[side]
        if kind == "fixed":
            ctrl = FixedTimeController()
        elif kind == "maxpressure":
            ctrl = MaxPressureController()
        elif kind == "random":
            ctrl = RandomController(seed=self.seed)
        else:
            ctrl = self._load_dqn(checkpoint)
            checkpoint = checkpoint.replace("\\", "/")
        s.controller, s.kind, s.checkpoint = ctrl, kind, (checkpoint if kind == "dqn" else None)

    def set_controller(self, side: str, kind: str, checkpoint: str | None = None) -> None:
        """Hot swap: the new controller takes over at the next decision, the traffic is untouched."""
        self._assign(side, kind, checkpoint)
        s = self.sides[side]
        s.controller.reset(s.env)

    def checkpoints(self) -> dict:
        base = self.root / "checkpoints"
        finals = sorted(p.name for p in base.glob("*.pt")) if base.is_dir() else []
        timeline = {}
        for d in sorted(p for p in base.glob("*") if p.is_dir()) if base.is_dir() else []:
            eps = sorted(int(p.stem.split("_")[1]) for p in d.glob("ep_*.pt"))
            if eps:
                timeline[d.name] = eps
        return {"finals": finals, "timeline": timeline}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, scenario: Scenario | None = None) -> None:
        """Restart both cities from t = 0 with identical traffic."""
        if seed is not None:
            self.seed = int(seed)
        overrides = dict(scenario.demand) if scenario is not None else {}
        if "base_rate" in overrides:
            self.demand_level = float(overrides.pop("base_rate"))
        spec = DemandSpec.from_config(self.cfg, base_rate=self.demand_level, **overrides)
        self.scenario_name = scenario.name if scenario is not None else None
        self._pending = [(e.t, e, "scenario") for e in (scenario.events if scenario is not None else ())]
        self._outbox = []
        for s in self.sides.values():
            s.obs, s.info = s.env.reset(seed=self.seed, demand=spec)
            s.controller.reset(s.env)
            s.env.apply_actions(s.controller.act(s.obs, s.info["action_mask"], s.info))
            s.metrics = self._metrics(s)

    @property
    def t(self) -> float:
        return self.sides["A"].env.sim.t

    def tick(self) -> None:
        """Advance both cities by one simulation step."""
        now = self.t
        while self._pending and self._pending[0][0] <= now + 1e-9:
            _, event, source = self._pending.pop(0)
            self._apply_event(event, source)
        for s in self.sides.values():
            if s.env.tick():
                s.obs, _, _, _, s.info = s.env.collect()
                s.env.apply_actions(s.controller.act(s.obs, s.info["action_mask"], s.info))
                s.metrics = self._metrics(s)

    def _metrics(self, s: Side) -> dict:
        m, sim = s.env.metrics, s.env.sim
        out = dict(m.live())
        out["time_in_system"] = (m.tt_sum + sim.unfinished_time()) / max(1, sim.n_spawned)
        out["ev_stops"] = m.ev_stops / m.ev_n if m.ev_n else 0.0
        out["n_completed"] = m.n_done
        out["mean_speed"] = m.series[-1][2] if m.series else self.cfg.vehicle.v0
        # plain Python numbers only: NumPy scalars are not JSON-serialisable by orjson
        clean = {}
        for k, v in out.items():
            if isinstance(v, (bool, np.bool_)):
                clean[k] = bool(v)
            elif isinstance(v, (int, np.integer)):
                clean[k] = int(v)
            else:
                clean[k] = round(float(v), 2)
        return clean

    # ------------------------------------------------------------------
    # perturbations (always applied to both sides)
    # ------------------------------------------------------------------
    def _where(self, link: int, pos: float | None = None) -> dict:
        p = 0.5 * float(self.net.link_len[link]) if pos is None else pos
        x, y, _ = self.net.lane_xy(2 * int(link), p)
        return {"link": int(link), "label": self.net.link_label(int(link)),
                "x": round(float(x), 1), "y": round(float(y), 1)}

    def _apply_event(self, event: Event, source: str) -> dict:
        results = [event.apply(s.env.sim) for s in self.sides.values()]
        p = event.params
        detail = {k: v for k, v in p.items() if k in ("lane", "pos", "duration", "multiplier", "side", "n", "base_rate")}
        if "link" in p:
            detail.update(self._where(p["link"], p.get("pos")))
        if event.kind == "emergency":
            detail.update(self._where(p["entry"], 0.0))
        if event.kind == "jam":
            detail["placed"] = results
        if event.kind == "demand":
            self.demand_level = float(p["base_rate"])
        msg = {"type": "event", "t": round(self.t, 1), "kind": event.kind, "source": source, "detail": detail}
        self._outbox.append(msg)
        return msg

    def _pick(self, x: float, y: float):
        hit = self.net.pick(x, y)
        if hit is None:
            raise ProtocolError("there is no road at that point")
        return hit

    def perturb(self, cmd: dict) -> dict:
        kind, net = cmd["kind"], self.net
        if kind == "incident":
            link, lane, pos = self._pick(cmd["x"], cmd["y"])
            if net.is_exit[link]:
                raise ProtocolError("accidents can only be placed on roads that lead to an intersection")
            params = {"link": link, "lane": lane, "pos": round(pos, 1), "duration": cmd["duration"]}
        elif kind == "closure":
            link, _, _ = self._pick(cmd["x"], cmd["y"])
            params = {"link": link, "duration": cmd["duration"]}
        elif kind == "jam":
            link, _, _ = self._pick(cmd["x"], cmd["y"])
            params = {"link": link, "n": cmd["n"]}
        elif kind == "surge":
            params = {"side": cmd["side"], "multiplier": cmd["multiplier"], "duration": cmd["duration"]}
        else:  # emergency
            entries = [int(l) for l in net.entry_links]
            if "entry" in cmd:
                entry = cmd["entry"]
            elif "label" in cmd:
                by_label = {net.link_label(l): l for l in entries}
                entry = by_label.get(cmd["label"], -1)
            elif "x" in cmd:
                d = [np.hypot(net.link_x0[l] - cmd["x"], net.link_y0[l] - cmd["y"]) for l in entries]
                entry = entries[int(np.argmin(d))]
            else:
                entry = entries[self._next_entry % len(entries)]
                self._next_entry += 1
            if entry not in entries:
                raise ProtocolError("unknown entry; use an entry link id or a label such as 'N2'")
            params = {"entry": int(entry)}
        return self._apply_event(Event(self.t, kind, params), "user")

    def clear(self) -> None:
        for s in self.sides.values():
            s.env.sim.clear_perturbations()
        self._pending = []
        self._outbox.append({"type": "event", "t": round(self.t, 1), "kind": "clear", "source": "user", "detail": {}})

    # ------------------------------------------------------------------
    # scenarios
    # ------------------------------------------------------------------
    def scenarios(self) -> dict:
        adv = self.root / "scenarios" / "adversarial"
        files = sorted(p.stem for p in adv.glob("adv_*.json")) if adv.is_dir() else []
        return {"library": list(LIBRARY), "adversarial": files}

    def _scenario(self, name: str) -> Scenario:
        if name in LIBRARY:
            return get_scenario(name, self.cfg, self.net)
        adv = (self.root / "scenarios" / "adversarial").resolve()
        path = (adv / f"{name}.json").resolve()
        if adv in path.parents and path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Scenario.from_dict(data.get("scenario", data))
        raise ProtocolError(f"unknown scenario: {name}")

    def run_scenario(self, name: str, mode: str) -> dict:
        sc = self._scenario(name)
        if mode == "reset":
            self.reset(scenario=sc)
            return {"mode": "reset", "events": len(sc.events)}
        # inject: keep the current traffic, shift the events so the first one happens shortly
        if "base_rate" in sc.demand:
            self._apply_event(Event(self.t, "demand", {"base_rate": float(sc.demand["base_rate"])}), "scenario")
        if sc.events:
            shift = self.t + INJECT_LEAD - sc.events[0].t
            self._pending += [(e.t + shift, e, "scenario") for e in sc.events]
            self._pending.sort(key=lambda x: x[0])
        self.scenario_name = sc.name
        return {"mode": "inject", "events": len(sc.events),
                "first_event_at": round(self.t + INJECT_LEAD, 1) if sc.events else None}

    # ------------------------------------------------------------------
    # commands and messages
    # ------------------------------------------------------------------
    def handle(self, message) -> dict:
        """Execute one client message. Returns an ``ack``; raises ``ProtocolError`` if rejected."""
        cmd = protocol.parse(message)
        kind = cmd["type"]
        if kind == "control":
            action = cmd["action"]
            if action == "play":
                self.playing = True
            elif action == "pause":
                self.playing = False
            elif action == "speed":
                self.speed = cmd["value"]
            elif action == "reset":
                self.reset(seed=cmd.get("seed"))
            elif action == "step":
                for _ in range(cmd["value"]):
                    self.tick()
            return protocol.ack(cmd, t=round(self.t, 1))
        if kind == "controller":
            self.set_controller(cmd["side"], cmd["kind"], cmd.get("checkpoint"))
            return protocol.ack(cmd, label=self.sides[cmd["side"]].label)
        if kind == "demand":
            self._apply_event(Event(self.t, "demand", {"base_rate": cmd["level"]}), "user")
            return protocol.ack(cmd)
        if kind == "perturb":
            return protocol.ack(cmd, event=self.perturb(cmd))
        if kind == "scenario":
            return protocol.ack(cmd, **self.run_scenario(cmd["name"], cmd["mode"]))
        self.clear()
        return protocol.ack(cmd)

    def drain_events(self) -> list:
        out, self._outbox = self._outbox, []
        return out

    def static(self) -> dict:
        return {"type": "static", **self.net.static_geometry()}

    def state(self) -> dict:
        ck = self.checkpoints()
        return {
            "type": "state", "t": round(self.t, 1), "playing": self.playing, "speed": self.speed,
            "seed": self.seed, "demand": self.demand_level, "scenario": self.scenario_name,
            "pending_events": len(self._pending),
            "sides": {n: {"kind": s.kind, "checkpoint": s.checkpoint, "label": s.label}
                      for n, s in self.sides.items()},
            "controllers": list(protocol.CONTROLLER_KINDS),
            "checkpoints": ck["finals"], "timeline": ck["timeline"],
            "scenarios": self.scenarios(),
            "limits": {"speed": list(protocol.SPEED_RANGE), "demand": list(protocol.DEMAND_RANGE),
                       "surge": list(protocol.SURGE_RANGE), "jam": list(protocol.JAM_RANGE),
                       "duration": list(protocol.DURATION_RANGE)},
            "sides_of_city": list(HEADING_NAMES),
        }

    def frame(self) -> dict:
        sides = {}
        for n, s in self.sides.items():
            snap = s.env.sim.snapshot()
            sides[n] = {"controller": s.label, "kind": s.kind, "veh": snap["veh"], "sig": snap["sig"],
                        "incidents": snap["incidents"], "closed": snap["closed"], "metrics": s.metrics}
        return {"type": "frame", "t": round(self.t, 3), "speed": self.speed, "playing": self.playing,
                "sides": sides}
