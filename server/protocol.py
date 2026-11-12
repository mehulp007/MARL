"""WebSocket protocol: every message is one JSON object with a ``type``.

Server -> client
    static   sent once on connect: everything drawn once (``Network.static_geometry()``)
    state    session settings; sent on connect and after every accepted command
    frame    one per simulation step (or fewer at high speed): both sides' vehicles, signals, metrics
    event    a perturbation that just happened (user, scenario or adversary)
    ack      reply to the sender of an accepted command
    error    reply to the sender of a rejected command

Client -> server (validated and normalised by ``parse``)
    {"type": "control", "action": "play" | "pause" | "reset" | "speed" | "step", "value": .., "seed": ..}
    {"type": "controller", "side": "A" | "B", "kind": "fixed" | "maxpressure" | "random" | "dqn",
     "checkpoint": "cooperative.pt" | "cooperative/ep_0040.pt"}
    {"type": "demand", "level": 500}
    {"type": "perturb", "kind": "incident" | "closure" | "jam", "x": .., "y": .., "duration": .., "n": ..}
    {"type": "perturb", "kind": "surge", "side": "N" | "E" | "S" | "W" | "all", "multiplier": .., "duration": ..}
    {"type": "perturb", "kind": "emergency", "entry": <entry link id> | "label": "N2" | "x": .., "y": ..}
    {"type": "scenario", "name": "compound_chaos" | "adv_01", "mode": "reset" | "inject"}
    {"type": "clear"}
"""
from __future__ import annotations

CONTROLLER_KINDS = ("fixed", "maxpressure", "random", "dqn")
SIDES = ("A", "B")
CONTROL_ACTIONS = ("play", "pause", "reset", "speed", "step")
PERTURB_KINDS = ("incident", "closure", "surge", "emergency", "jam")
SURGE_SIDES = ("N", "E", "S", "W", "all")
SCENARIO_MODES = ("reset", "inject")

SPEED_RANGE = (0.25, 20.0)
DEMAND_RANGE = (0.0, 1200.0)
DURATION_RANGE = (5.0, 3600.0)
SURGE_RANGE = (1.0, 5.0)
JAM_RANGE = (1, 60)
STEP_RANGE = (1, 400)
MAX_FPS = 30.0


class ProtocolError(ValueError):
    """A client message that cannot be executed; its text is sent back as an ``error``."""


def _number(msg: dict, key: str, lo: float, hi: float, default=None, integer: bool = False):
    v = msg.get(key, default)
    if v is None:
        raise ProtocolError(f"'{key}' is required")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ProtocolError(f"'{key}' must be a number")
    if not lo <= v <= hi:
        raise ProtocolError(f"'{key}' must be between {lo:g} and {hi:g}")
    return int(round(v)) if integer else float(v)


def _choice(msg: dict, key: str, options, default=None):
    v = msg.get(key, default)
    if v not in options:
        raise ProtocolError(f"'{key}' must be one of {', '.join(map(str, options))}")
    return v


def _duration(msg: dict, default):
    """None means 'until cleared'."""
    if msg.get("duration", default) is None:
        return None
    return _number(msg, "duration", *DURATION_RANGE, default=default)


def _point(msg: dict):
    if "x" not in msg or "y" not in msg:
        raise ProtocolError("'x' and 'y' (world coordinates in metres) are required")
    return _number(msg, "x", -1e5, 1e5), _number(msg, "y", -1e5, 1e5)


def parse(msg) -> dict:
    """Validate a client message and return it normalised. Raises ``ProtocolError``."""
    if not isinstance(msg, dict):
        raise ProtocolError("a message must be a JSON object")
    kind = msg.get("type")
    if kind == "control":
        action = _choice(msg, "action", CONTROL_ACTIONS)
        out = {"type": "control", "action": action}
        if action == "speed":
            out["value"] = _number(msg, "value", *SPEED_RANGE)
        elif action == "step":
            out["value"] = _number(msg, "value", *STEP_RANGE, default=1, integer=True)
        elif action == "reset" and msg.get("seed") is not None:
            out["seed"] = _number(msg, "seed", 0, 2**31 - 1, integer=True)
        return out
    if kind == "controller":
        out = {"type": "controller", "side": _choice(msg, "side", SIDES),
               "kind": _choice(msg, "kind", CONTROLLER_KINDS)}
        if out["kind"] == "dqn":
            ck = msg.get("checkpoint")
            if not isinstance(ck, str) or not ck:
                raise ProtocolError("'checkpoint' is required for kind 'dqn'")
            out["checkpoint"] = ck
        return out
    if kind == "demand":
        return {"type": "demand", "level": _number(msg, "level", *DEMAND_RANGE)}
    if kind == "perturb":
        pk = _choice(msg, "kind", PERTURB_KINDS)
        out = {"type": "perturb", "kind": pk}
        if pk in ("incident", "closure", "jam"):
            out["x"], out["y"] = _point(msg)
        if pk == "incident":
            out["duration"] = _duration(msg, 180.0)
        elif pk == "closure":
            out["duration"] = _duration(msg, None)
        elif pk == "jam":
            out["n"] = _number(msg, "n", *JAM_RANGE, default=20, integer=True)
        elif pk == "surge":
            out["side"] = _choice(msg, "side", SURGE_SIDES, default="all")
            out["multiplier"] = _number(msg, "multiplier", *SURGE_RANGE, default=2.5)
            out["duration"] = _number(msg, "duration", *DURATION_RANGE, default=180.0)
        elif pk == "emergency":
            if "entry" in msg:
                out["entry"] = _number(msg, "entry", 0, 10**6, integer=True)
            elif isinstance(msg.get("label"), str):
                out["label"] = msg["label"]
            elif "x" in msg and "y" in msg:
                out["x"], out["y"] = _point(msg)
            # none of them: the session picks the next entry in rotation
        return out
    if kind == "scenario":
        name = msg.get("name")
        if not isinstance(name, str) or not name:
            raise ProtocolError("'name' is required")
        return {"type": "scenario", "name": name, "mode": _choice(msg, "mode", SCENARIO_MODES, default="reset")}
    if kind == "clear":
        return {"type": "clear"}
    raise ProtocolError(f"unknown message type {kind!r}")


def ack(cmd: dict, **detail) -> dict:
    return {"type": "ack", "ok": True, "cmd": cmd, **detail}


def error(message: str) -> dict:
    return {"type": "error", "ok": False, "message": message}
