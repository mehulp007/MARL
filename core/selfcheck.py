"""Invariant and numerical sanity suite.

    python -m core.selfcheck              # full suite
    python -m core.selfcheck --quick      # shorter runs
    python -m core.selfcheck signals reward   # selected groups
    python -m core.selfcheck --list

Exit code 0 when every check passes, 1 otherwise. ``run()`` returns the results
as data for test wrappers; only ``main()`` prints (the one place in ``core``
that does, because this module is a command-line tool).

Group -> numbered invariant it covers
    network       topology, routing tables, geometry            (supports 9)
    physics       1 conservation, 2 no overlap, 3 red-light compliance
    signals       4 conflict-freedom and exact timing, 5 mask soundness
    determinism   6 determinism and common random numbers
    observation   7 observation bounds
    reward        8 reward identity
    routing       9 destinations reached; closures respected
    perturbations incidents, jams, emergency vehicles, surges
    scenarios     serialisation, replay, library
    performance   10 simulation speed
    baselines     11 MaxPressure < FixedTime < Random
    agents        learner maths (skipped when PyTorch is not installed)
    adversary     budget projection and decoding
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import numpy as np

from .baselines import FixedTimeController, MaxPressureController, RandomController, run_episode
from .config import (
    Config, ETYPE_EMERGENCY, ETYPE_OBSTACLE, LANE_OF_MOVE, LEFT, OBS_DIM, RIGHT, SIG_RED,
    STAGE_GREEN, STAGE_YELLOW, THROUGH,
)
from .demand import DemandModel, DemandSpec
from .env import TrafficEnv
from .network import Network
from .scenarios import LIBRARY, Scenario, get_scenario, random_scenario
from .simulator import TrafficSim


@dataclass
class Check:
    group: str
    name: str
    passed: bool
    detail: str = ""


class _Ctx:
    """Shared objects and the result list for one suite run."""

    def __init__(self, quick: bool):
        self.quick = quick
        self.cfg = Config()
        self.net = Network(self.cfg)
        self.results: list = []
        self.group = ""

    def check(self, name: str, cond, detail: str = "") -> bool:
        self.results.append(Check(self.group, name, bool(cond), detail))
        return bool(cond)

    def dur(self, full: float, quick: float) -> float:
        return quick if self.quick else full


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _cycle(sim: TrafficSim, greens=(25.0, 8.0, 25.0, 8.0)) -> np.ndarray:
    """A simple fixed-cycle request, used to drive the simulator in the checks."""
    s = sim.signals
    return np.where(s.green_elapsed >= np.asarray(greens)[s.phase] - 1e-9, (s.phase + 1) % 4, s.phase)


def _min_gap(sim: TrafficSim) -> float:
    """Smallest bumper-to-bumper gap between consecutive vehicles on any lane."""
    idx = np.flatnonzero(sim.active[: sim.hi])
    if idx.size < 2:
        return np.inf
    ln, p = sim.lane[idx], sim.pos[idx]
    o = np.lexsort((-p, ln))
    ln, p, rear = ln[o], p[o], (p - sim.length[idx])[o]
    same = ln[1:] == ln[:-1]
    gap = rear[:-1][same] - p[1:][same]
    return float(gap.min()) if gap.size else np.inf


def _monitored_run(sim: TrafficSim, seconds: float, control=_cycle, hook=None) -> dict:
    """Step the simulator while checking the per-step invariants independently of its own counters."""
    net, cfg = sim.net, sim.cfg
    K = cfg.substeps
    st = {"conserved": True, "min_gap": np.inf, "red_crossings": 0, "v_max": 0.0, "beyond": -np.inf,
          "timing_exact": True, "conflict_free": True, "finite": True, "peak": 0}
    sg = sim.signals
    run_len = np.zeros(sim.n, dtype=int)
    prev = sg.stage.copy()
    for _ in range(int(round(seconds / cfg.sim.dt))):
        if control is not None and sim.step_count % K == 0:
            sim.request_phases(control(sim))
        if hook is not None:
            hook(sim)
        sig = sg.lane_signal().copy()
        # 4. at most one phase shows green/yellow per intersection
        lit = (sig != SIG_RED) & ~net.lane_is_exit
        ph = np.where(lit, net.lane_phase, -1)[: 8 * sim.n].reshape(sim.n, 8)
        st["conflict_free"] &= bool(((ph == -1) | (ph == ph.max(axis=1, keepdims=True))).all())
        # 4. yellow / all-red last exactly their configured number of steps
        changed = sg.stage != prev
        ended = changed & (prev != STAGE_GREEN)
        expect = np.where(prev == STAGE_YELLOW, sg.yellow_steps, sg.all_red_steps)
        st["timing_exact"] &= not bool((ended & (run_len != expect)).any())
        run_len = np.where(changed, 1, run_len + 1)
        prev = sg.stage.copy()
        hi0 = sim.hi
        lane0, vid0, act0 = sim.lane[:hi0].copy(), sim.vid[:hi0].copy(), sim.active[:hi0].copy()
        sim.step()
        # 3. nobody left a link whose lane showed red at the start of the step
        moved = act0 & sim.active[:hi0] & (sim.vid[:hi0] == vid0) & ((sim.lane[:hi0] >> 1) != (lane0 >> 1))
        st["red_crossings"] += int(np.count_nonzero(sig[lane0[moved]] == SIG_RED))
        st["conserved"] &= sim.n_spawned == sim.n_vehicles + sim.n_completed + sim.n_backlog
        st["min_gap"] = min(st["min_gap"], _min_gap(sim))
        a = sim.active[: sim.hi]
        if a.any():
            p, v = sim.pos[: sim.hi][a], sim.vel[: sim.hi][a]
            st["finite"] &= bool(np.isfinite(p).all() and np.isfinite(v).all() and (v >= 0).all())
            st["v_max"] = max(st["v_max"], float(v.max()))
            st["beyond"] = max(st["beyond"], float((p - net.lane_len[sim.lane[: sim.hi][a]]).max()))
            st["peak"] = max(st["peak"], int(a.sum()))
    st["red_crossings"] += sim.n_red_cross
    return st


def _drain(sim: TrafficSim, limit: float = 3600.0, keep: int = 0, hook=None) -> dict:
    """Monitored run with demand already switched off, until at most ``keep`` vehicles remain
    (or ``limit`` seconds pass). The naive test cycle clears a long turn queue only a few
    vehicles per cycle, so a fixed duration would be a test of patience, not of the simulator."""
    total = None
    spent = 0.0
    while spent < limit and (sim.n_vehicles > keep or sim.n_backlog > 0):
        st = _monitored_run(sim, 100.0, hook=hook)
        spent += 100.0
        if total is None:
            total = st
        else:
            for k in ("conserved", "timing_exact", "conflict_free", "finite"):
                total[k] = total[k] and st[k]
            total["min_gap"] = min(total["min_gap"], st["min_gap"])
            total["red_crossings"] = max(total["red_crossings"], st["red_crossings"])
            total["v_max"] = max(total["v_max"], st["v_max"])
            total["beyond"] = max(total["beyond"], st["beyond"])
            total["peak"] = max(total["peak"], st["peak"])
    if total is None:
        total = _monitored_run(sim, 0.0)
    total["drain_time"] = spent
    return total


def _physics_ok(st: dict) -> bool:
    return (st["conserved"] and st["min_gap"] >= -1e-6 and st["red_crossings"] == 0
            and st["finite"] and st["beyond"] < 0.0)


def _fingerprint(sim: TrafficSim):
    a = sim.active[: sim.hi]
    return (float(sim.pos[: sim.hi][a].sum()), float(sim.vel[: sim.hi][a].sum()),
            int(sim.lane[: sim.hi][a].sum()), sim.n_completed, sim.n_backlog, sim.n_spawned)


def _no_ev(cfg: Config, **kw) -> DemandSpec:
    return DemandSpec.from_config(cfg, ev_rate=0.0, **kw)


# --------------------------------------------------------------------------
# check groups
# --------------------------------------------------------------------------
def check_network(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    N = net.n
    c.check("counts: 80 links, 160 lanes, 16 entries, 16 exits, obs 81",
            net.n_links == 80 and net.n_lanes == 160 and net.entry_links.size == 16
            and net.exit_links.size == 16 and OBS_DIM == 81 and cfg.cap_lane == 30 and cfg.substeps == 10)
    good = True
    for i in range(N):
        for h in range(4):
            l, o, j = net.in_link[i, h], net.out_link[i, h], net.nbr[i, h]
            good &= l == 4 * i + h and net.link_dst[l] == i and net.link_src[o] == i
            good &= net.is_exit[o] if j < 0 else net.link_dst[o] == j
    c.check("in_link / out_link / nbr are consistent", good)
    good = True
    for i in range(N):
        for p in range(4):
            targets = []
            for ln in net.phase_lanes[i, p]:
                good &= net.lane_phase[ln] == p and net.lane_node[ln] == i
                moves = (LEFT,) if net.lane_k[ln] == 1 else (THROUGH, RIGHT)
                targets += [int(net.move_target[net.lane_link[ln], m]) for m in moves]
            good &= len(set(targets)) == len(targets)
    c.check("every phase is conflict-free by construction (distinct target links)", good)
    rt = net.base_routes
    inc, ex = np.flatnonzero(~net.is_exit), net.exit_links
    c.check("a finite route exists from every link to every exit", np.isfinite(rt.cost_to_go[np.ix_(inc, ex)]).all())
    v0 = cfg.vehicle.v0
    pen = np.asarray(cfg.grid.turn_penalty) + np.asarray(cfg.grid.conn_len) / v0
    good = True
    for l in inc:
        for d in ex:
            path = net.route_links(int(l), int(d), rt)
            cost = sum(net.link_len[p] / v0 for p in path)
            for a, b in zip(path[:-1], path[1:]):
                m = rt.next_move[a, d]
                good &= net.move_target[a, m] == b
                cost += pen[m]
            good &= path[-1] == d and len(set(path)) == len(path) and abs(cost - rt.cost_to_go[l, d]) < 1e-6
    c.check("walking next_move reaches the destination at the tabulated cost, without loops", good)
    lanes_k = np.tile([0, 1], net.n_links)
    nm = rt.next_move[np.repeat(np.arange(net.n_links), 2)]
    right = (nm >= 0) & (np.asarray(LANE_OF_MOVE)[np.maximum(nm, 0)] == lanes_k[:, None])
    c.check("lane_move equals next_move in the required lane and respects lane permissions",
            (rt.lane_move[right] == nm[right]).all() and np.isin(rt.lane_move[0::2], [-1, THROUGH, RIGHT]).all()
            and np.isin(rt.lane_move[1::2], [-1, LEFT]).all())
    worst_p = worst_a = 0.0
    eps = 1e-4
    wrap = lambda u, w: abs((u - w + np.pi) % (2 * np.pi) - np.pi)
    for l in inc:
        for k in (0, 1):
            pl = 2 * l + k
            for m in ((LEFT,) if k == 1 else (THROUGH, RIGHT)):
                for k2 in (0, 1):
                    nl = 2 * net.move_target[l, m] + k2
                    cl = cfg.grid.conn_len[m]
                    xa, ya, aa = net.lane_xy(pl, net.lane_len[pl] - eps)
                    xb, yb, ab = net.lane_xy(nl, -cl + eps, pl)
                    xc, yc, ac = net.lane_xy(nl, -eps, pl)
                    xd, yd, ad = net.lane_xy(nl, eps, pl)
                    worst_p = max(worst_p, float(np.hypot(xa - xb, ya - yb)), float(np.hypot(xc - xd, yc - yd)))
                    worst_a = max(worst_a, float(wrap(aa, ab)), float(wrap(ac, ad)))
    c.check("lane_xy is continuous in position and angle through every junction path",
            worst_p < 1e-2 and worst_a < 1e-2, f"max jump {worst_p:.1e} m, {worst_a:.1e} rad")
    rng = np.random.default_rng(0)
    good = True
    for _ in range(300):
        ln = int(rng.integers(net.n_lanes))
        p = float(rng.uniform(0, net.lane_len[ln]))
        x, y, _ = net.lane_xy(ln, p)
        r = net.pick(float(x) + rng.uniform(-1, 1), float(y) + rng.uniform(-1, 1))
        good &= r is not None and r[0] == ln // 2 and r[1] == ln % 2 and abs(r[2] - p) < 1.5
    c.check("pick() inverts lane_xy(); static_geometry() is JSON-serialisable",
            good and net.pick(-5e3, -5e3) is None and len(json.dumps(net.static_geometry())) > 1000)
    bad = 0
    for o in ({"nope": 1}, {"reward": {"coop_beta": "x"}}, {"signal": {"yellow": 5.0}}, {"dqn": {"n_step": 2.5}}):
        try:
            Config.from_dict(o)
        except (KeyError, TypeError, ValueError):
            bad += 1
    c.check("config: round trip, overrides, invalid overrides rejected",
            bad == 4 and Config.from_dict(cfg.to_dict()) == cfg
            and Config.from_dict({"reward": {"coop_beta": 0}}).reward.coop_beta == 0.0)


def check_physics(c: _Ctx) -> None:
    cfg = c.cfg
    T = c.dur(1200.0, 400.0)
    sim = TrafficSim(cfg, seed=1, net=c.net)
    st = _monitored_run(sim, T)
    c.check("1. conservation: spawned == on road + completed + backlog, every step", st["conserved"],
            f"spawned {sim.n_spawned}, completed {sim.n_completed}, on road {sim.n_vehicles}, backlog {sim.n_backlog}")
    c.check("2. no overlap: bumper gap >= 0 on every lane, every step", st["min_gap"] >= -1e-6,
            f"smallest gap {st['min_gap']:.2f} m, peak {st['peak']} vehicles")
    c.check("3. red-light compliance: no vehicle leaves a link on red", st["red_crossings"] == 0)
    c.check("states stay finite, speeds bounded, nobody past the end of a lane",
            st["finite"] and st["beyond"] < 0 and st["v_max"] <= cfg.vehicle.v0_ev * cfg.vehicle.v0_clip[1] + 1e-6,
            f"max speed {st['v_max']:.1f} m/s")
    trips = sim.trip_log()
    c.check("trips complete in plausible time (>= 75% of free-flow time)",
            len(trips) > 100 and (trips.travel_time >= 0.75 * trips.ff_time).all(),
            f"{len(trips)} trips, mean {trips.travel_time.mean():.0f} s vs free-flow {trips.ff_time.mean():.0f} s")
    heavy = TrafficSim(cfg, seed=2, demand=DemandSpec.from_config(cfg, base_rate=900.0), net=c.net)
    st = _monitored_run(heavy, c.dur(900.0, 300.0))
    c.check("invariants 1-3 also hold when oversaturated (900 veh/h per entry)", _physics_ok(st),
            f"on road {heavy.n_vehicles}, backlog {heavy.n_backlog}")
    # queue discharge rate
    s = TrafficSim(cfg, seed=3, demand=_no_ev(cfg, base_rate=1500.0, p_straight=1.0), net=c.net)
    for _ in range(2 * cfg.substeps):
        s.step()
    s.request_phases(np.full(s.n, 2))
    for _ in range(int(90 / cfg.sim.dt)):
        s.step()
    s.set_demand_level(0.0)
    for _ in range(int(20 / cfg.sim.dt)):
        s.step()
    lanes = [2 * int(l) for l in c.net.entry_links if c.net.link_heading[l] in (0, 2)]
    s.request_phases(np.zeros(s.n, dtype=int))
    times = {l: [] for l in lanes}
    for _ in range(int(60 / cfg.sim.dt)):
        before = s.lane_stats().count[lanes]
        s.step()
        after = s.lane_stats().count[lanes]
        for l, d in zip(lanes, before - after):
            times[l] += [s.t] * int(d)
    hw = float(np.mean([np.diff(t[3:15]).mean() for t in times.values() if len(t) >= 15]))
    c.check("queue discharge headway is realistic (1.7-2.3 s per vehicle)", 1.7 <= hw <= 2.3,
            f"{hw:.2f} s -> {3600 / hw:.0f} veh/h/lane")


def check_signals(c: _Ctx) -> None:
    cfg = c.cfg
    K, sgc = cfg.substeps, cfg.signal
    sim = TrafficSim(cfg, seed=4, net=c.net)
    st = _monitored_run(sim, c.dur(600.0, 300.0))
    c.check("4. never two phases green/yellow at one intersection", st["conflict_free"])
    c.check("4. yellow lasts exactly %.0f s and all-red exactly %.0f s" % (sgc.yellow, sgc.all_red), st["timing_exact"])
    s = TrafficSim(cfg, seed=5, net=c.net)
    m0 = s.action_mask()
    ok = m0[:, 0].all() and not m0[:, 1:].any()                       # green_elapsed 0 < g_min
    ok &= bool((s.request_phases(np.full(s.n, 2)) == 0).all())        # invalid request -> keep
    for _ in range(2 * K):
        s.step()
    ok &= bool(s.action_mask().all())
    ex = s.request_phases(np.full(s.n, 2))
    ok &= bool((ex == 2).all() and (s.signals.stage == STAGE_YELLOW).all())
    ok &= bool((s.action_mask() == (np.arange(4) == 2)).all())        # in transition: only the pending phase
    for _ in range(K):
        s.step()
    ok &= bool((s.signals.stage == STAGE_GREEN).all() and (s.signals.phase == 2).all()
               and np.allclose(s.signals.green_elapsed, sgc.decision_dt - sgc.yellow - sgc.all_red))
    c.check("5. masks: g_min blocks early switches; invalid requests are replaced; every intersection is green at decisions", ok)
    s = TrafficSim(cfg, seed=6, net=c.net)
    ok, forced, longest = True, 0, 0.0
    ar = np.arange(s.n)
    for _ in range(int(round(c.dur(1500.0, 600.0) / cfg.sim.dt))):
        if s.step_count % K == 0:
            mask = s.action_mask()
            cur, el, dem = s.signals.phase.copy(), s.signals.green_elapsed.copy(), s.phase_demand()
            other = dem.sum(axis=1) - dem[ar, cur]
            must = (el + sgc.decision_dt > sgc.g_max) & (other > 0)
            ex = s.request_phases(cur)                                # always ask to keep
            ok &= bool(mask.any(axis=1).all() and (ex[must] != cur[must]).all()
                       and (dem[must, ex[must]] > 0).all() and (ex[~must] == cur[~must]).all())
            forced += int(must.sum())
            if (other > 0).any():
                longest = max(longest, float(el[other > 0].max()))
        s.step()
    c.check("5. masks: every row has a valid action; a green is cut at g_max when others wait",
            ok and forced > 0, f"{forced} forced switches")


def check_determinism(c: _Ctx) -> None:
    cfg, T = c.cfg, c.dur(600.0, 300.0)
    a = TrafficSim(cfg, seed=7, net=c.net)
    _monitored_run(a, T)
    b = TrafficSim(cfg, seed=7, net=c.net)
    _monitored_run(b, T)
    c.check("6. same seed + same controller -> identical trajectories", _fingerprint(a) == _fingerprint(b))
    d = TrafficSim(cfg, seed=7, net=c.net)
    _monitored_run(d, T, control=lambda s: _cycle(s, (10.0, 10.0, 40.0, 10.0)))

    def arrivals(sim):
        tr = sim.trip_log()
        out = {int(v): (int(o), float(t)) for v, o, t in zip(tr.veh_id, tr.origin, tr.t_spawn)}
        for i in np.flatnonzero(sim.active[: sim.hi] & ~sim.phantom[: sim.hi]):
            out[int(sim.vid[i])] = (int(sim.origin[i]), float(sim.t_spawn[i]))
        for link, q in sim._backlog.items():
            out.update({int(e[0]): (link, float(e[4])) for e in q})
        return out
    A, D = arrivals(a), arrivals(d)
    c.check("6. common random numbers: a different signal plan receives exactly the same vehicles",
            A == D and _fingerprint(a) != _fingerprint(d), f"{len(A)} vehicles")
    a.reset(7)
    _monitored_run(a, T)
    e = TrafficSim(cfg, seed=8, net=c.net)
    _monitored_run(e, T)
    c.check("reset(seed) reproduces a run; a different seed gives different traffic",
            _fingerprint(a) == _fingerprint(b) and _fingerprint(e) != _fingerprint(b))
    e1, e2 = TrafficEnv(cfg, seed=9, net=c.net), TrafficEnv(cfg, seed=9, net=c.net)
    o1, i1 = e1.reset()
    o2, i2 = e2.reset()
    c1, c2 = RandomController(5), RandomController(5)
    c1.reset(e1)
    c2.reset(e2)
    same, ticks = True, 0
    for _ in range(40):
        a1, a2 = c1.act(o1, i1["action_mask"], i1), c2.act(o2, i2["action_mask"], i2)
        o1, r1, _, _, i1 = e1.step(a1)
        e2.apply_actions(a2)
        while True:
            ticks += 1
            if e2.tick():
                break
        o2, r2, _, _, i2 = e2.collect()
        same &= np.array_equal(o1, o2) and np.array_equal(r1, r2)
    c.check("env.step() equals apply_actions + tick x substeps + collect", same and ticks == 40 * cfg.substeps)


def check_observation(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    env = TrafficEnv(cfg, seed=3, net=net, episode_len=c.dur(1800.0, 900.0))
    obs, info = env.reset(scenario=get_scenario("compound_chaos", cfg, net))
    ctrl = RandomController(1)
    ctrl.reset(env)
    ok_shape = ok_bounds = ok_hot = True
    ev_seen, done = 0.0, False
    while not done:
        obs, r, term, trunc, info = env.step(ctrl.act(obs, info["action_mask"], info))
        done = term or trunc
        ok_shape &= obs.shape == (net.n, OBS_DIM) and obs.dtype == np.float32
        ok_bounds &= bool(np.isfinite(obs).all() and obs.min() >= 0.0 and obs.max() <= 1.0)
        ok_hot &= bool((obs[:, 44:48].sum(axis=1) == 1).all() and info["action_mask"].any(axis=1).all())
        ev_seen = max(ev_seen, float(obs[:, 32:40].max()))
    c.check("7. observations have shape (16, 81), dtype float32", ok_shape)
    c.check("7. observations are finite and inside [0, 1] through a chaotic episode", ok_bounds,
            f"emergency feature reached {ev_seen:.2f}")
    c.check("own-phase block is one-hot; terminated is never set; truncated at the time limit",
            ok_hot and not term and trunc)
    sim = env.sim
    o = obs
    hi = sim.hi
    act = sim.active[:hi] & (sim.etype[:hi] != ETYPE_OBSTACLE)
    ph, el, cap = sim.signals.target_phase, np.minimum(sim.signals.green_elapsed / cfg.signal.g_max, 1.0), float(cfg.cap_lane)
    good = True
    for i in range(net.n):
        for d in range(4):
            blk, j = o[i, 49 + 8 * d: 57 + 8 * d], net.nbr[i, d]
            if j < 0:
                good &= not blk.any()
                continue
            link_ji = net.in_link[i, (d + 2) % 4]
            idx = np.flatnonzero(act & (net.lane_node[sim.lane[:hi]] == j))
            mv = sim.routes.lane_move[sim.lane[idx], sim.dest[idx]]
            feed = sum(1 for a_, m_ in zip(idx, mv) if m_ >= 0 and net.move_target[sim.lane[a_] >> 1, m_] == link_ji)
            exp = np.zeros(8)
            exp[0], exp[1 + ph[j]], exp[5] = 1.0, 1.0, el[j]
            exp[6] = min(feed / cap, 1.0)
            exp[7] = min(int((sim.vel[idx] < cfg.vehicle.v_halt).sum()) / (2.0 * cap), 1.0)
            good &= bool(np.allclose(blk, exp, atol=1e-6))
        for jn in range(8):
            on = act & (sim.lane[:hi] == 8 * i + jn)
            good &= abs(o[i, jn] - min((sim.vel[:hi][on] < cfg.vehicle.v_halt).sum() / cap, 1.0)) < 1e-6
    c.check("neighbour blocks and own queues match a direct recomputation from the vehicle arrays", good)


def check_reward(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    rc, beta = cfg.reward, cfg.reward.coop_beta
    env = TrafficEnv(cfg, seed=3, net=net, episode_len=c.dur(900.0, 400.0))
    obs, info = env.reset(scenario=get_scenario("compound_chaos", cfg, net))
    ctrl = RandomController(2)
    ctrl.reset(env)
    ok, done = True, False
    while not done:
        obs, r, term, trunc, info = env.step(ctrl.act(obs, info["action_mask"], info))
        done = term or trunc
        t = info["reward_terms"]
        rl = -rc.w_h * t["h"] + rc.w_x * t["x"] - rc.w_e * t["e"] - rc.w_m * t["m"] - rc.w_b * t["b"] - rc.w_s * t["s"]
        nb = np.array([np.mean([rl[j] for j in net.nbr[i] if j >= 0]) for i in range(net.n)])
        ok &= bool(np.allclose(rl, info["r_local"], atol=1e-5) and np.allclose((1 - beta) * rl + beta * nb, r, atol=1e-5)
                   and r.dtype == np.float32 and r.shape == (net.n,))
    c.check("8. r equals the documented formula recomputed from reward_terms (local, then neighbour mix)", ok)
    c0 = cfg.override({"reward": {"coop_beta": 0.0}})
    e0 = TrafficEnv(c0, seed=3, net=Network(c0), episode_len=300.0)
    obs, info = e0.reset()
    ctrl.reset(e0)
    eq, done = True, False
    while not done:
        obs, r, term, trunc, info = e0.step(ctrl.act(obs, info["action_mask"], info))
        done = term or trunc
        eq &= bool(np.array_equal(r, info["r_local"]))
    c.check("8. with beta = 0 the reward equals the local reward exactly", eq)


def check_routing(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    spec = _no_ev(cfg)
    sim = TrafficSim(cfg, seed=11, demand=spec, net=net)
    T = c.dur(600.0, 300.0)
    _monitored_run(sim, T)
    sim.set_demand_level(0.0)
    st = _drain(sim)
    # regenerate the same arrivals independently of the simulator
    dm = DemandModel(cfg, net, spec, 11)
    dest, origin = {}, {}
    vid = 1
    for k in range(int(round(T / cfg.sim.dt))):
        arr = dm.step(k * cfg.sim.dt)
        for e_, d_ in zip(arr.entry.tolist(), arr.dest.tolist()):
            dest[vid], origin[vid] = d_, e_
            vid += 1
    tr = sim.trip_log()
    good = len(tr) == len(dest) == sim.n_spawned and sim.n_vehicles == 0 and sim.n_backlog == 0
    good &= all(dest[int(v)] == int(d) and origin[int(v)] == int(o) for v, d, o in zip(tr.veh_id, tr.dest, tr.origin))
    c.check("9. with no closure every vehicle exits at its own destination", good and st["conserved"],
            f"{len(tr)} trips")
    s = TrafficSim(cfg, seed=12, demand=spec, net=net)
    _monitored_run(s, 100.0)
    cl, ex_cl = int(net.in_link[5, 1]), int(net.exit_links[2])
    s.close_link(cl)
    s.close_link(ex_cl, duration=300.0)
    already = set(s.vid[: s.hi][s.active[: s.hi] & ((s.lane[: s.hi] >> 1) == cl)].tolist())
    entered = [0]

    def watch(sim_):
        m = sim_.active[: sim_.hi] & ((sim_.lane[: sim_.hi] >> 1) == cl)
        entered[0] += len(set(sim_.vid[: sim_.hi][m].tolist()) - already)
    st1 = _monitored_run(s, c.dur(500.0, 350.0), hook=watch)
    s.set_demand_level(0.0)
    st2 = _drain(s, hook=watch)
    c.check("9. nobody enters a closed link", entered[0] == 0)
    c.check("9. with a closed link all vehicles still exit; a timed closure reopens",
            s.n_vehicles == 0 and s.n_backlog == 0 and s.n_spawned == s.n_completed and not s.closed[ex_cl] and s.closed[cl]
            and _physics_ok(st1) and _physics_ok(st2), f"{s.n_completed} trips, drained in {st2['drain_time']:.0f} s")


def check_perturbations(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    s = TrafficSim(cfg, seed=13, demand=_no_ev(cfg), net=net)
    _monitored_run(s, 200.0)
    link = int(net.in_link[6, 2])
    i1 = s.add_incident(link, 0, 120.0, duration=None)
    i2 = s.add_incident(int(net.in_link[9, 1]), 1, 150.0, duration=200.0)
    i3 = s.add_incident(link, 0, 121.0)
    st1 = _monitored_run(s, c.dur(600.0, 300.0))
    lane_changes = s.n_lane_changes
    s.set_demand_level(0.0)
    rec = s.incidents.get(i1)
    broken = 0 if rec is None or rec["phantom"] else 1       # a broken-down vehicle stays until cleared
    st2 = _drain(s, keep=broken)
    left = s.n_vehicles - broken
    c.check("incident: vehicles change lanes around it; traffic drains past a permanent blockage",
            lane_changes > 0 and left == 0 and s.n_backlog == 0 and _physics_ok(st1) and _physics_ok(st2),
            f"{lane_changes} lane changes")
    c.check("incident: same spot returns the same id; a timed incident expires", i3 == i1 and i2 not in s.incidents)
    s.clear_incident(i1)
    _monitored_run(s, 120.0)
    c.check("incident: after clearing, the road is empty and counts balance",
            s.n_vehicles == 0 and not s.incidents and s.n_spawned == s.n_completed)
    s = TrafficSim(cfg, seed=14, demand=_no_ev(cfg), net=net)
    _monitored_run(s, 60.0)
    placed = s.inject_jam(int(net.in_link[10, 0]), 40)
    vid = s.spawn_emergency(int(net.entry_links[1]))
    n0 = s.n_spawned
    s.set_surge([int(l) for l in net.entries_by_side[0]], 3.0, 120.0)
    st1 = _monitored_run(s, 120.0)
    surge = s.n_spawned - n0
    st2 = _monitored_run(s, 120.0)
    normal = s.n_spawned - n0 - surge
    st3 = _monitored_run(s, 400.0)
    tr = s.trip_log()
    ev = tr.etype == ETYPE_EMERGENCY
    c.check("jam: stopped vehicles are placed without breaking invariants", 20 <= placed <= 40 and _physics_ok(st1),
            f"{placed} placed")
    c.check("emergency vehicle: reaches the straight-across exit",
            ev.sum() == 1 and int(tr.veh_id[ev][0]) == vid
            and int(tr.dest[ev][0]) == int(net.straight_exit[net.entry_links[1]]))
    c.check("surge: arrivals rise for its duration, then return to normal",
            surge > 1.25 * normal and _physics_ok(st2) and _physics_ok(st3), f"{surge} vs {normal} per 120 s")


def check_scenarios(c: _Ctx) -> None:
    cfg, net = c.cfg, c.net
    rng = np.random.default_rng(4)
    good = True
    for _ in range(50):
        sc = random_scenario(rng, cfg, net)
        good &= sc == Scenario.from_dict(json.loads(json.dumps(sc.to_dict()))) and 1 <= len(sc.events) <= 3
    lib = {n: get_scenario(n, cfg, net) for n in LIBRARY}
    good &= all(s == Scenario.from_dict(json.loads(json.dumps(s.to_dict()))) for s in lib.values())
    c.check("scenarios survive a JSON round trip (library and random)", good and len(lib) == 9)
    T = c.dur(1800.0, 600.0)
    env = TrafficEnv(cfg, seed=5, net=net, episode_len=T)
    mp = MaxPressureController()
    names = list(LIBRARY) if not c.quick else ["steady_medium", "compound_chaos"]
    ok, t0 = True, time.perf_counter()
    res = {}
    for n in names:
        res[n] = run_episode(env, mp, seed=5, scenario=lib[n])
        ok &= bool(np.isfinite(res[n]["time_in_system"]) and res[n]["n_completed"] > 50
                   and isinstance(json.dumps(res[n]), str))
    c.check(f"{len(names)} library scenarios run to the end with finite, JSON-serialisable summaries", ok,
            f"{(time.perf_counter() - t0) / len(names):.1f} s per {T:.0f} s episode")
    again = run_episode(TrafficEnv(cfg, seed=5, net=net, episode_len=T), mp, seed=5, scenario=lib["compound_chaos"])
    plain = run_episode(env, mp, seed=5)
    c.check("a scenario replays identically; reset arguments apply to one episode only",
            again == res["compound_chaos"] and plain != res["compound_chaos"]
            and plain == run_episode(TrafficEnv(cfg, seed=5, net=net, episode_len=T), mp, seed=5))


def check_performance(c: _Ctx) -> None:
    cfg = c.cfg
    K = cfg.substeps
    s = TrafficSim(cfg, seed=21, net=c.net)
    for _ in range(int(600 / cfg.sim.dt)):
        if s.step_count % K == 0:
            s.request_phases(_cycle(s))
        s.step()
    n0, steps = s.n_vehicles, int(c.dur(4000, 1500))
    t0 = time.perf_counter()
    for _ in range(steps):
        if s.step_count % K == 0:
            s.request_phases(_cycle(s))
        s.step()
    sps = steps / (time.perf_counter() - t0)
    c.check("10. simulation speed >= 800 steps/s at medium demand (target 1500)", sps >= 800.0,
            f"{sps:.0f} steps/s with ~{(n0 + s.n_vehicles) // 2} vehicles")
    env = TrafficEnv(cfg, seed=21, net=c.net, episode_len=600.0)
    t0 = time.perf_counter()
    run_episode(env, MaxPressureController(), seed=21)
    dps = 120 / (time.perf_counter() - t0)
    c.check("environment speed >= 100 decisions/s (observation, reward and metrics included)", dps >= 100.0,
            f"{dps:.0f} decisions/s")


def check_baselines(c: _Ctx) -> None:
    cfg = c.cfg
    env = TrafficEnv(cfg, net=c.net, episode_len=c.dur(1800.0, 900.0))
    seeds = (101, 102) if not c.quick else (101,)
    out = {}
    for name, ctrl in (("MaxPressure", MaxPressureController()), ("FixedTime", FixedTimeController()),
                       ("Random", RandomController())):
        rs = [run_episode(env, ctrl, seed=s) for s in seeds]
        out[name] = {k: float(np.mean([r[k] for r in rs])) for k in ("att", "time_in_system", "return")}
    show = lambda k: ", ".join(f"{n} {v[k]:.0f}" for n, v in out.items())
    c.check("11. ATT ordering at medium demand: MaxPressure < FixedTime < Random",
            out["MaxPressure"]["att"] < out["FixedTime"]["att"] < out["Random"]["att"], show("att"))
    c.check("the same ordering holds for time in system",
            out["MaxPressure"]["time_in_system"] < out["FixedTime"]["time_in_system"] < out["Random"]["time_in_system"],
            show("time_in_system"))
    c.check("the reward ranks the controllers the same way (MaxPressure > FixedTime > Random)",
            out["MaxPressure"]["return"] > out["FixedTime"]["return"] > out["Random"]["return"], show("return"))


def check_agents(c: _Ctx) -> None:
    try:
        import torch
    except ImportError:
        c.check("learner checks skipped: PyTorch is not installed", True, "install torch to run them")
        return
    from .agents import DQNController, MADQN, NStepAssembler, ReplayBuffer, Trainer
    cfg, N = c.cfg, c.net.n
    asm = NStepAssembler(3, 0.9, 2)
    R = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]], dtype=float)
    obs = [np.full((2, 4), t, dtype=np.float32) for t in range(6)]
    masks = [np.array([[True, t % 2 == 0, False, True]] * 2) for t in range(6)]
    out = []
    for t in range(5):
        tr = asm.push(obs[t], np.array([t, t]), R[t], obs[t + 1], masks[t + 1])
        if tr is not None:
            out.append(tr)
    out += asm.flush()
    exp = [(0, 1 + 0.9 * 2 + 0.81 * 3, 0.729, 3), (1, 2 + 0.9 * 3 + 0.81 * 4, 0.729, 4),
           (2, 3 + 0.9 * 4 + 0.81 * 5, 0.729, 5), (3, 4 + 0.9 * 5, 0.81, 5), (4, 5.0, 0.9, 5)]
    good = len(out) == 5
    for tr, (t, g, disc, nx) in zip(out, exp):
        good &= bool(tr["obs"][0, 0] == t and abs(tr["reward"][0] - g) < 1e-5 and abs(tr["reward"][1] - 10 * g) < 1e-4
                     and abs(tr["disc"][0] - disc) < 1e-6 and tr["next_obs"][0, 0] == nx
                     and np.array_equal(tr["next_mask"], masks[nx]))
    c.check("n-step returns, discounts and bootstrap states match a hand-computed example", good)
    buf = ReplayBuffer(40, 4, 4, seed=0)
    for tr in out * 5:
        buf.add(tr)
    b = buf.sample(32)
    c.check("replay buffer wraps at capacity and returns float32 / int64 / bool batches",
            len(buf) == 40 and buf.ptr == 10 and b["obs"].dtype == np.float32 and b["action"].dtype == np.int64
            and b["next_mask"].dtype == bool)
    L = MADQN(cfg, N, seed=0)
    with torch.no_grad():
        for p in L.q_target.parameters():
            p.add_(0.05 * torch.randn_like(p))
    rng = np.random.default_rng(0)
    B = 32
    batch = {"obs": rng.random((B, OBS_DIM), dtype=np.float32), "action": rng.integers(0, 4, B),
             "reward": rng.normal(size=B).astype(np.float32), "next_obs": rng.random((B, OBS_DIM), dtype=np.float32),
             "next_mask": rng.random((B, 4)) < 0.6, "disc": np.full(B, cfg.dqn.gamma ** 3, dtype=np.float32),
             "agent": rng.integers(0, N, B)}
    batch["next_mask"][:, 0] = True
    y = L.compute_target(L._tensors(batch)).numpy()
    with torch.no_grad():
        qo = L.q(torch.from_numpy(batch["next_obs"]), None).numpy()
        qt = L.q_target(torch.from_numpy(batch["next_obs"]), None).numpy()
    man = np.empty(B)
    for i in range(B):
        valid = np.flatnonzero(batch["next_mask"][i])
        man[i] = batch["reward"][i] + batch["disc"][i] * qt[i, valid[np.argmax(qo[i, valid])]]
    c.check("training target equals G + disc * Q_target(s', best valid action of Q_online)", np.allclose(y, man, atol=1e-5))
    first = L.update(batch)["loss"]
    for _ in range(200):
        last = L.update(batch)["loss"]
    c.check("gradient steps on one batch reduce the loss", last < 0.2 * first, f"{first:.3f} -> {last:.5f}")
    o1 = rng.random((N, OBS_DIM), dtype=np.float32)
    mask = rng.random((N, 4)) < 0.5
    mask[np.arange(N), rng.integers(0, 4, N)] = True
    r1 = np.random.default_rng(1)
    acts = np.stack([L.act(o1, mask, eps=1.0, rng=r1) for _ in range(200)])
    c.check("acting: greedy is the best valid action; exploration never picks an invalid one",
            np.array_equal(L.act(o1, mask), np.where(mask, L.q_values(o1), -np.inf).argmax(axis=1))
            and mask[np.arange(N)[None, :], acts].all())
    L2 = MADQN.from_state_dict(L.state_dict())
    env = TrafficEnv(cfg, seed=1, net=c.net, episode_len=200.0)
    s = run_episode(env, DQNController(L2), seed=1)
    c.check("checkpoint dict rebuilds an identical policy that runs as a Controller",
            np.allclose(L.q_values(o1), L2.q_values(o1), atol=1e-6) and np.isfinite(s["time_in_system"]))
    tc = cfg.override({"train": {"episodes": 10, "eval_every": 0}})
    T = Trainer(tc, env=TrafficEnv(tc, net=Network(tc)))
    tot = 10 * T.steps_per_episode
    eps = [T.epsilon(0), T.epsilon(int(0.3 * tot)), T.epsilon(int(0.6 * tot)), T.epsilon(tot)]
    d = tc.dqn
    c.check("epsilon decays linearly from eps_start to eps_end over the first eps_frac of training",
            abs(eps[0] - d.eps_start) < 1e-9 and abs(eps[1] - 0.5 * (d.eps_start + d.eps_end)) < 2e-3
            and abs(eps[2] - d.eps_end) < 1e-9 and abs(eps[3] - d.eps_end) < 1e-9)


def check_adversary(c: _Ctx) -> None:
    from .adversary import KINDS, N_CONT, Adversary, project_budget
    from .scenarios import Event
    cfg, net = c.cfg, c.net
    ev = [Event(50, "closure", {"link": 5, "duration": 800.0}), Event(60, "closure", {"link": 9, "duration": 400.0}),
          Event(10, "incident", {"link": 1, "lane": 0, "pos": 100.0, "duration": None}),
          Event(20, "incident", {"link": 2, "lane": 0, "pos": 100.0, "duration": 300.0}),
          Event(30, "incident", {"link": 3, "lane": 0, "pos": 100.0, "duration": 300.0}),
          Event(40, "surge", {"side": "N", "multiplier": 9.0, "duration": 5000.0}),
          Event(70, "jam", {"link": 6, "n": 500})]
    c6 = cfg.override({"adversary": {"max_events": 6}})
    pb = project_budget(ev, c6)
    a = cfg.adversary
    kinds = [e.kind for e in pb]
    c.check("budget projection enforces every limit and is idempotent",
            len(pb) == 6 and kinds.count("incident") == a.max_incidents
            and sum(e.params["duration"] for e in pb if e.kind == "closure") <= a.max_closure_time
            and all(e.params["multiplier"] <= a.max_surge for e in pb if e.kind == "surge")
            and all(e.params["n"] <= 40 for e in pb if e.kind == "jam")
            and project_budget(pb, c6) == pb and len(project_budget(ev, cfg)) == a.max_events)
    adv = Adversary(FixedTimeController(), MaxPressureController(), cfg, np.random.default_rng(0), net=net)
    rng = np.random.default_rng(1)
    good, seen = True, set()
    for _ in range(200):
        sc = adv.decode(rng.integers(0, len(KINDS), a.max_events), rng.random((a.max_events, N_CONT)))
        seen |= {e.kind for e in sc.events}
        good &= sc == Scenario.from_dict(json.loads(json.dumps(sc.to_dict()))) and project_budget(sc.events, cfg) == sc.events
        for e in sc.events:
            if e.kind in ("incident", "jam"):
                good &= not net.is_exit[e.params["link"]]
            if e.kind == "closure":
                good &= e.params["link"] in net.internal_links
            if e.kind == "emergency":
                good &= e.params["entry"] in net.entry_links
    c.check("every decoded sample is within budget, uses valid links and serialises", good and seen == set(KINDS) - {"none"})
    if not c.quick:
        tiny = cfg.override({"adversary": {"population": 4, "iterations": 1, "elites": 2, "episode_len": 200.0, "n_seeds": 1}})
        mk = lambda: Adversary(FixedTimeController(), MaxPressureController(), tiny, np.random.default_rng(3), net=Network(tiny))
        r1, r2 = mk().search(top_k=2), mk().search(top_k=2)
        c.check("a small search is reproducible and returns results sorted by regret",
                [r.to_dict() for r in r1] == [r.to_dict() for r in r2] and r1[0].regret >= r1[-1].regret)


GROUPS = {
    "network": check_network,
    "physics": check_physics,
    "signals": check_signals,
    "determinism": check_determinism,
    "observation": check_observation,
    "reward": check_reward,
    "routing": check_routing,
    "perturbations": check_perturbations,
    "scenarios": check_scenarios,
    "performance": check_performance,
    "baselines": check_baselines,
    "agents": check_agents,
    "adversary": check_adversary,
}


def run(groups=None, quick: bool = False) -> list:
    """Run the selected groups (all by default) and return a list of ``Check`` results.

    An exception inside a group is recorded as a failed check instead of aborting the suite.
    """
    ctx = _Ctx(quick)
    for name in (list(GROUPS) if not groups else list(groups)):
        if name not in GROUPS:
            raise KeyError(f"unknown check group {name!r}; choose from {list(GROUPS)}")
        ctx.group = name
        try:
            GROUPS[name](ctx)
        except Exception as exc:                      # noqa: BLE001 - report, do not hide
            ctx.check("group raised an exception", False, f"{type(exc).__name__}: {exc}")
    return ctx.results


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv:
        print("\n".join(GROUPS))
        return 0
    quick = "--quick" in argv
    names = [a for a in argv if not a.startswith("--")]
    t0 = time.perf_counter()
    results = []
    for name in (names or list(GROUPS)):
        t1 = time.perf_counter()
        part = run([name], quick)
        results += part
        print(f"\n[{name}]  ({time.perf_counter() - t1:.1f} s)")
        for r in part:
            print(f"  {'PASS' if r.passed else 'FAIL'}  {r.name}" + (f"   ({r.detail})" if r.detail else ""))
        sys.stdout.flush()
    failed = [r for r in results if not r.passed]
    print(f"\n{'=' * 72}\n{len(results) - len(failed)}/{len(results)} checks passed in "
          f"{time.perf_counter() - t0:.0f} s" + (" (quick mode)" if quick else ""))
    for r in failed:
        print(f"  FAILED  [{r.group}] {r.name}   ({r.detail})")
    print("SELFCHECK " + ("PASSED" if not failed else "FAILED"))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
