"""FastAPI application: static files, one WebSocket, and the real-time loop.

    python -m server.app [--host 127.0.0.1] [--port 8000] [--seed 7] [--speed 4]

The loop advances the session ``speed`` simulated seconds per wall-clock second
and broadcasts a frame per simulation step, capped at ``protocol.MAX_FPS``.
With no client connected the simulation waits. The server binds to localhost
by default: checkpoints are loaded with pickle, so it is not meant to be
exposed to a network.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

import orjson
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response

from . import protocol
from .session import Session

_PLACEHOLDER = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MARL Traffic Optimizer</title>
<style>body{font:15px system-ui,sans-serif;margin:2rem;max-width:44rem;line-height:1.5}
code,pre{background:#f1f1ee;padding:.1rem .3rem;border-radius:4px}pre{padding:.8rem}</style></head><body>
<h1>MARL Traffic Optimizer server</h1>
<p>The server is running. The visual interface (<code>web/</code>) has not been built yet;
this page only proves the live connection.</p>
<pre id="out">connecting...</pre>
<script>
const out = document.getElementById("out"); let frames = 0, state = null;
const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
ws.onmessage = (e) => { const m = JSON.parse(e.data);
  if (m.type === "state") state = m;
  if (m.type === "frame") { frames++; const s = m.sides;
    out.textContent = `frames ${frames}\\nsim time ${m.t.toFixed(1)} s  speed ${m.speed}x  playing ${m.playing}\\n` +
      Object.keys(s).map(k => `${k}: ${s[k].controller} | vehicles ${s[k].veh.length} | time in system ` +
      `${s[k].metrics.time_in_system} s | queue ${s[k].metrics.mean_queue}`).join("\\n"); } };
ws.onclose = () => { out.textContent += "\\n(disconnected)"; };
</script></body></html>"""


def _dumps(message: dict) -> bytes:
    return orjson.dumps(message, option=orjson.OPT_SERIALIZE_NUMPY)


def _json(message: dict) -> Response:
    return Response(content=_dumps(message), media_type="application/json")


class Hub:
    """The set of connected clients."""

    def __init__(self):
        self.clients: set = set()

    async def send(self, ws: WebSocket, message: dict) -> None:
        await ws.send_text(_dumps(message).decode())

    async def broadcast(self, message: dict) -> None:
        if not self.clients:
            return
        text = _dumps(message).decode()
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:                              # noqa: BLE001 - a closed socket is not an error
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


async def run_loop(session: Session, hub: Hub) -> None:
    """Advance the session in real time and broadcast frames until cancelled."""
    loop = asyncio.get_running_loop()
    dt = session.cfg.sim.dt
    next_tick = loop.time()
    last_sent = 0.0
    while True:
        if not session.playing or not hub.clients:
            await asyncio.sleep(0.05)
            next_tick = loop.time()
            continue
        session.tick()
        for event in session.drain_events():
            await hub.broadcast(event)
        now = loop.time()
        if now - last_sent >= 1.0 / protocol.MAX_FPS - 1e-3:
            await hub.broadcast(session.frame())
            last_sent = now
        next_tick += dt / session.speed
        delay = next_tick - loop.time()
        if delay < -0.5:                                    # fell behind (slow machine): do not try to catch up
            next_tick = loop.time()
            delay = 0.0
        await asyncio.sleep(max(0.0, delay))


def create_app(root: str | Path = ".", seed: int = 7, speed: float = 4.0, session: Session | None = None) -> FastAPI:
    root = Path(root).resolve()
    session = session if session is not None else Session(root=root, seed=seed, speed=speed)
    hub = Hub()
    web = root / "web"

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(run_loop(session, hub))
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="MARL Traffic Optimizer", lifespan=lifespan)
    app.state.session = session
    app.state.hub = hub

    @app.get("/api/state")
    async def api_state():
        return _json(session.state())

    @app.get("/api/static")
    async def api_static():
        return _json(session.static())

    @app.get("/api/frame")
    async def api_frame():
        return _json(session.frame())

    @app.get("/")
    async def index():
        page = web / "index.html"
        if page.is_file():
            return FileResponse(page, headers={"Cache-Control": "no-cache"})
        return HTMLResponse(_PLACEHOLDER)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await hub.send(ws, session.static())
        await hub.send(ws, session.state())
        await hub.send(ws, session.frame())
        hub.clients.add(ws)
        try:
            while True:
                text = await ws.receive_text()
                try:
                    reply = session.handle(json.loads(text))
                except (protocol.ProtocolError, json.JSONDecodeError) as exc:
                    await hub.send(ws, protocol.error(str(exc)))
                    continue
                await hub.send(ws, reply)
                for event in session.drain_events():
                    await hub.broadcast(event)
                await hub.broadcast(session.state())
                await hub.broadcast(session.frame())        # so a paused view updates immediately
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    no_cache = {"Cache-Control": "no-cache"}            # edits to web/ show up on reload

    @app.get("/{path:path}")
    async def web_file(path: str):
        """Everything else is a file under web/. Missing files, a missing
        web/ folder and anything outside it are a plain 404."""
        target = (web / path).resolve()
        if web not in target.parents:
            raise HTTPException(status_code=404)
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(target, headers=no_cache)

    return app


def main(argv=None) -> int:
    import uvicorn
    p = argparse.ArgumentParser(description="Run the MARL Traffic Optimizer server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=4.0)
    p.add_argument("--root", default=".")
    a = p.parse_args(argv)
    uvicorn.run(create_app(a.root, a.seed, a.speed), host=a.host, port=a.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
