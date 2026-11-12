"""Server tests: protocol validation, the twin session, and the WebSocket application."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import protocol
from server.app import create_app
from server.protocol import ProtocolError
from server.session import Session


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------
def test_parse_normalises_valid_messages():
    p = protocol.parse
    assert p({"type": "control", "action": "speed", "value": 8}) == {"type": "control", "action": "speed", "value": 8.0}
    assert p({"type": "control", "action": "reset", "seed": 3})["seed"] == 3
    assert p({"type": "control", "action": "step"})["value"] == 1
    assert p({"type": "controller", "side": "B", "kind": "dqn", "checkpoint": "selfish.pt"})["checkpoint"] == "selfish.pt"
    assert "checkpoint" not in p({"type": "controller", "side": "A", "kind": "fixed", "checkpoint": "x.pt"})
    assert p({"type": "perturb", "kind": "incident", "x": 1, "y": 2})["duration"] == 180.0
    assert p({"type": "perturb", "kind": "closure", "x": 1, "y": 2})["duration"] is None
    assert p({"type": "perturb", "kind": "surge"}) == {"type": "perturb", "kind": "surge", "side": "all",
                                                       "multiplier": 2.5, "duration": 180.0}
    assert p({"type": "perturb", "kind": "emergency"}) == {"type": "perturb", "kind": "emergency"}
    assert p({"type": "scenario", "name": "compound_chaos"})["mode"] == "reset"
    assert p({"type": "clear", "junk": 1}) == {"type": "clear"}


@pytest.mark.parametrize("bad", [
    "text", {}, {"type": "nope"},
    {"type": "control", "action": "fly"},
    {"type": "control", "action": "speed", "value": 500},
    {"type": "control", "action": "speed", "value": "fast"},
    {"type": "control", "action": "speed", "value": float("nan")},
    {"type": "controller", "side": "C", "kind": "fixed"},
    {"type": "controller", "side": "A", "kind": "dqn"},
    {"type": "demand", "level": -5},
    {"type": "perturb", "kind": "incident", "x": 1},
    {"type": "perturb", "kind": "jam", "x": 1, "y": 2, "n": 5000},
    {"type": "perturb", "kind": "surge", "side": "up"},
    {"type": "scenario"},
    {"type": "scenario", "name": "x", "mode": "sideways"},
])
def test_parse_rejects_invalid_messages(bad):
    with pytest.raises(ProtocolError):
        protocol.parse(bad)


# --------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def session():
    return Session(seed=5)


def _point(session, link, pos=100.0, lane=0):
    x, y, _ = session.net.lane_xy(2 * link + lane, pos)
    return float(x), float(y)


def test_same_controller_on_both_sides_gives_identical_cities(session):
    session.reset(seed=5)
    session.set_controller("A", "maxpressure")
    session.set_controller("B", "maxpressure")
    session.reset(seed=5)
    for _ in range(300):
        session.tick()
    f = session.frame()
    assert f["sides"]["A"]["veh"] == f["sides"]["B"]["veh"] and len(f["sides"]["A"]["veh"]) > 50
    assert f["sides"]["A"]["sig"] == f["sides"]["B"]["sig"]
    assert json.dumps(f) and f["t"] == 150.0


def test_different_controllers_receive_identical_traffic_and_perturbations(session):
    session.set_controller("A", "fixed")
    session.set_controller("B", "dqn", "cooperative.pt")
    session.reset(seed=6)
    net = session.net
    link = int(net.in_link[5, 1])
    x, y = _point(session, link)
    for _ in range(100):
        session.tick()
    a = session.handle({"type": "perturb", "kind": "incident", "x": x, "y": y, "duration": 120})
    assert a["ok"] and a["event"]["detail"]["link"] == link and a["event"]["source"] == "user"
    cx, cy = _point(session, int(net.in_link[10, 0]))
    session.handle({"type": "perturb", "kind": "closure", "x": cx, "y": cy})
    session.handle({"type": "perturb", "kind": "surge", "side": "N", "multiplier": 3, "duration": 60})
    session.handle({"type": "perturb", "kind": "emergency", "label": "W1"})
    jam = session.handle({"type": "perturb", "kind": "jam", "x": cx, "y": cy, "n": 10})
    assert len(jam["event"]["detail"]["placed"]) == 2
    for _ in range(200):
        session.tick()
    sa, sb = session.sides["A"].env.sim, session.sides["B"].env.sim
    assert len(sa.incidents) == len(sb.incidents) == 1
    assert sa.closed.tolist() == sb.closed.tolist() and sa.closed.sum() == 1
    f = session.frame()
    assert f["sides"]["A"]["veh"] != f["sides"]["B"]["veh"]              # different control ...
    assert f["sides"]["A"]["closed"] == f["sides"]["B"]["closed"] == [int(net.in_link[10, 0])]
    # ... but the same demand: arrivals are independent of the controller
    assert sa.n_spawned - jam["event"]["detail"]["placed"][0] == sb.n_spawned - jam["event"]["detail"]["placed"][1]
    kinds = [e["kind"] for e in session.drain_events()]
    assert kinds == ["incident", "closure", "surge", "emergency", "jam"]
    session.handle({"type": "clear"})
    assert not sa.incidents and not sb.incidents and not sa.closed.any() and not sb.closed.any()
    assert session.drain_events()[-1]["kind"] == "clear"


def test_controls_and_hot_swap_do_not_restart_the_traffic(session):
    session.reset(seed=7)
    session.handle({"type": "control", "action": "step", "value": 40})
    t, n = session.t, session.sides["B"].env.sim.n_spawned
    assert t == 20.0 and n > 0
    for kind, ck in (("random", None), ("maxpressure", None), ("dqn", "selfish.pt"),
                     ("dqn", "cooperative/ep_0000.pt"), ("fixed", None)):
        msg = {"type": "controller", "side": "B", "kind": kind}
        if ck:
            msg["checkpoint"] = ck
        assert session.handle(msg)["ok"]
        assert session.t == t and session.sides["B"].env.sim.n_spawned == n
        session.handle({"type": "control", "action": "step", "value": 20})
        t, n = session.t, session.sides["B"].env.sim.n_spawned
    assert session.state()["sides"]["B"] == {"kind": "fixed", "checkpoint": None, "label": "Fixed timer"}
    session.handle({"type": "control", "action": "speed", "value": 12})
    session.handle({"type": "control", "action": "pause"})
    session.handle({"type": "demand", "level": 650})
    st = session.state()
    assert st["speed"] == 12.0 and st["playing"] is False and st["demand"] == 650.0
    assert session.sides["A"].env.sim.demand.spec.base_rate == 650.0
    session.handle({"type": "control", "action": "play"})
    session.handle({"type": "control", "action": "reset", "seed": 11})
    assert session.t == 0.0 and session.seed == 11 and session.playing is True
    assert json.dumps(st) and "cooperative.pt" in st["checkpoints"] and st["timeline"]["cooperative"][0] == 0


@pytest.mark.parametrize("name", ["../requirements.txt", "..\\..\\x.pt", "C:/Windows/x.pt", "nope.pt",
                                  "cooperative/resume.pt", "cooperative"])
def test_checkpoint_names_cannot_escape_the_checkpoints_folder(session, name):
    with pytest.raises(ProtocolError):
        session.handle({"type": "controller", "side": "B", "kind": "dqn", "checkpoint": name})


def test_scenarios_reset_or_inject(session):
    st = session.state()
    assert "compound_chaos" in st["scenarios"]["library"] and "adv_01" in st["scenarios"]["adversarial"]
    a = session.handle({"type": "scenario", "name": "stadium_surge"})
    assert a["mode"] == "reset" and session.t == 0.0 and session.scenario_name == "stadium_surge"
    assert session.state()["pending_events"] == 1
    session.handle({"type": "control", "action": "step", "value": 100})
    b = session.handle({"type": "scenario", "name": "adv_01", "mode": "inject"})
    assert b["mode"] == "inject" and b["events"] == 3 and b["first_event_at"] == 60.0 and session.t == 50.0
    session.drain_events()
    session.handle({"type": "control", "action": "step", "value": 24})
    fired = [e for e in session.drain_events() if e["source"] == "scenario"]
    assert len(fired) == 1 and fired[0]["t"] == 60.0
    with pytest.raises(ProtocolError):
        session.handle({"type": "scenario", "name": "../../secrets"})
    with pytest.raises(ProtocolError):
        session.handle({"type": "perturb", "kind": "incident", "x": -9999, "y": -9999})


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
def _until(ws, kind, limit=400):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind!r} message within {limit} messages")


def test_websocket_handshake_commands_and_errors():
    app = create_app(seed=9, speed=20.0)
    with TestClient(app) as client:
        assert client.get("/api/state").json()["type"] == "state"
        assert client.get("/api/static").json()["rows"] == 4
        assert "MARL Traffic Optimizer" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            static, state, frame = ws.receive_json(), ws.receive_json(), ws.receive_json()
            assert static["type"] == "static" and len(static["links"]) == 80
            assert state["type"] == "state" and state["sides"]["A"]["kind"] == "fixed"
            assert frame["type"] == "frame" and set(frame["sides"]) == {"A", "B"}
            m = frame["sides"]["B"]["metrics"]
            assert {"time_in_system", "mean_queue", "throughput", "gws_arterial", "backlog"} <= set(m)
            later = _until(ws, "frame")                      # the loop is streaming on its own
            later = _until(ws, "frame")
            assert later["t"] > frame["t"]
            ws.send_json({"type": "control", "action": "pause"})
            assert _until(ws, "ack")["cmd"]["action"] == "pause"
            assert _until(ws, "state")["playing"] is False
            t0 = _until(ws, "frame")["t"]
            ws.send_json({"type": "control", "action": "step", "value": 20})
            assert _until(ws, "ack")["t"] == pytest.approx(t0 + 10.0)
            assert _until(ws, "frame")["t"] == pytest.approx(t0 + 10.0)
            ws.send_json({"type": "perturb", "kind": "surge", "side": "E", "multiplier": 2})
            assert _until(ws, "ack")["event"]["kind"] == "surge"
            assert _until(ws, "event")["detail"]["side"] == "E"
            ws.send_json({"type": "control", "action": "warp"})
            assert "action" in _until(ws, "error")["message"]
            ws.send_text("not json")
            assert _until(ws, "error")["ok"] is False
            ws.send_json({"type": "controller", "side": "A", "kind": "dqn", "checkpoint": "../x.pt"})
            assert "checkpoint" in _until(ws, "error")["message"]
        assert not app.state.hub.clients


def test_static_files_are_served_from_web_and_nothing_else(tmp_path):
    # no web/ folder yet: a placeholder page, and missing files are a clean 404 (not a server error)
    with TestClient(create_app(root=tmp_path, seed=1)) as client:
        assert "has not been built yet" in client.get("/").text
        assert client.get("/favicon.ico").status_code == 404
        assert client.get("/js/main.js").status_code == 404
        # with no checkpoints folder the right-hand side falls back to max-pressure
        assert client.get("/api/state").json()["sides"]["B"]["kind"] == "maxpressure"
    web = tmp_path / "web"
    (web / "js").mkdir(parents=True)
    (web / "index.html").write_text("<h1>ui</h1>", encoding="utf-8")
    (web / "js" / "main.js").write_text("export const x = 1;", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("no", encoding="utf-8")
    with TestClient(create_app(root=tmp_path, seed=1)) as client:
        assert client.get("/").text == "<h1>ui</h1>"
        r = client.get("/js/main.js")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]
        assert r.headers["cache-control"] == "no-cache"
        for escape in ("/../secret.txt", "/%2e%2e/secret.txt", "/js/../../secret.txt", "/..%5csecret.txt"):
            assert client.get(escape).status_code == 404
        assert client.get("/api/static").json()["cols"] == 4          # API routes still win
