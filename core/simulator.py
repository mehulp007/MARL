"""TrafficSim: vectorised microscopic traffic simulator.

Vehicles live in struct-of-arrays storage; the hot path has no per-vehicle
Python loop. One ``step()`` advances the world by ``cfg.sim.dt`` seconds:

  1. expire timed perturbations; draw arrivals into the entry backlogs
  2. read the lane signals that hold for this step
  3. sort vehicles by (lane, position) -> leaders, lane fronts, lane tails
  4. stop-line logic for lane fronts (red / yellow / closed / don't-block-the-box)
  5. IDM acceleration, ballistic integration, hard projections
  6. bookkeeping (waits, stops, reward accumulators)
  7. transfers across stop lines and despawns at exits
  8. spawn from backlogs; lane changes (only while incidents/closures exist)
  9. advance the signal state machines and the clock

Conventions: ``pos`` is the front bumper; a vehicle inside a junction sits on
its *target* lane with a negative ``pos``.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import (
    Config,
    ETYPE_CAR,
    ETYPE_EMERGENCY,
    ETYPE_OBSTACLE,
    LEFT,
    LINK_INTERNAL,
    N_PHASES,
    SIG_GREEN,
    SIG_RED,
    SIG_YELLOW,
    THROUGH,
)
from .demand import DemandModel, DemandSpec
from .kinematics import idm_accel, integrate, lane_change_ok, project
from .network import Network, RoutingTables
from .signals import Signals

_BIG = 1e9
_HALF_BIG = 5e8


@dataclass
class LaneStats:
    """Per-lane read-out of the current state (obstacles excluded)."""
    halted: np.ndarray       # (n_lanes,) vehicles with v < v_halt
    moving: np.ndarray       # (n_lanes,)
    speed_sum: np.ndarray    # (n_lanes,) sum of speeds of moving vehicles
    wait_sum: np.ndarray     # (n_lanes,) sum of wait_link (veh*s)
    wait_max: np.ndarray     # (n_lanes,) longest wait_link on the lane (s)
    ev_prox: np.ndarray      # (n_lanes,) emergency proximity in [0, 1], 0 = none
    link_count: np.ndarray   # (n_links,) vehicles on the link
    inbound: np.ndarray      # (n_links,) vehicles whose next move leads into the link
    move_count: np.ndarray   # (n_lanes, 3) vehicles per lane by the movement they will execute

    @property
    def count(self) -> np.ndarray:
        return self.halted + self.moving

    @property
    def mean_speed(self) -> np.ndarray:
        return self.speed_sum / np.maximum(self.moving, 1)


@dataclass
class IntervalStats:
    """Accumulators since the previous ``drain_interval_stats()`` call."""
    steps: int
    halt_sum: np.ndarray      # (N,) sum over steps of halted vehicles on incoming lanes
    ev_halt_sum: np.ndarray   # (N,) same for emergency vehicles
    crossings: np.ndarray     # (N,) stop-line crossings
    blocked_sum: np.ndarray   # (N,) sum over steps of green lanes held by don't-block-the-box
    wait_max: np.ndarray      # (N,) longest wait_link on incoming lanes right now
    switched: np.ndarray      # (N,) bool, a phase change was initiated
    cross_all: int            # network totals for the Green Wave Score
    cross_nonstop: int
    art_all: int              # THROUGH crossings arriving from an internal link
    art_nonstop: int
    speed_sum: float          # sum over steps and vehicles of speed
    veh_steps: int            # sum over steps of vehicles present


@dataclass
class TripLog:
    """Completed trips since the previous ``trip_log()`` call (parallel arrays)."""
    veh_id: np.ndarray
    etype: np.ndarray
    origin: np.ndarray
    dest: np.ndarray
    t_spawn: np.ndarray       # arrival time at the city edge (backlog time included)
    t_exit: np.ndarray
    ff_time: np.ndarray       # free-flow time of the route
    wait_total: np.ndarray
    n_stops: np.ndarray
    n_cross: np.ndarray
    prog_sum: np.ndarray      # sum of lengths of non-stop THROUGH runs
    prog_runs: np.ndarray     # number of such runs

    def __len__(self) -> int:
        return int(self.veh_id.size)

    @property
    def travel_time(self) -> np.ndarray:
        return self.t_exit - self.t_spawn


_TRIP_FIELDS = ("veh_id", "etype", "origin", "dest", "t_spawn", "t_exit", "ff_time",
                "wait_total", "n_stops", "n_cross", "prog_sum", "prog_runs")


class TrafficSim:
    def __init__(self, cfg: Config, seed: int = 0, demand: DemandSpec | None = None,
                 net: Network | None = None):
        self.cfg = cfg
        self.net = net if net is not None else Network(cfg)
        self.signals = Signals(cfg, self.net)
        self._spec0 = demand if demand is not None else DemandSpec.from_config(cfg)
        self.seed = int(seed)
        net_, vc = self.net, cfg.vehicle
        self.n = net_.n
        self.dt = cfg.sim.dt
        M = vc.max_veh
        self.max_veh = M
        # ---- vehicle store (struct of arrays)
        self.active = np.zeros(M, dtype=bool)
        self.phantom = np.zeros(M, dtype=bool)        # obstacle that is not a vehicle
        self.vid = np.zeros(M, dtype=np.int64)
        self.lane = np.zeros(M, dtype=np.int32)
        self.pos = np.zeros(M)
        self.vel = np.zeros(M)
        self.v0 = np.ones(M)
        self.a_max = np.ones(M)
        self.length = np.ones(M)
        self.dest = np.zeros(M, dtype=np.int32)
        self.prev_lane = np.zeros(M, dtype=np.int32)
        self.conn_move = np.zeros(M, dtype=np.int8)   # movement being executed while pos < 0
        self.etype = np.zeros(M, dtype=np.int8)
        self.origin = np.zeros(M, dtype=np.int32)
        self.wait_link = np.zeros(M)
        self.wait_total = np.zeros(M)
        self.n_stops = np.zeros(M, dtype=np.int32)
        self.n_cross = np.zeros(M, dtype=np.int32)
        self.was_halted = np.zeros(M, dtype=bool)
        self.stopped_on_link = np.zeros(M, dtype=bool)
        self.t_spawn = np.zeros(M)
        self.lc_until = np.zeros(M)
        self.lat = np.zeros(M)
        self.prog_cur = np.zeros(M, dtype=np.int32)
        self.prog_sum = np.zeros(M, dtype=np.int32)
        self.prog_runs = np.zeros(M, dtype=np.int32)
        # ---- per-lane scratch
        self._tail_rear = np.full(net_.n_lanes, _BIG)
        self._tail_v = np.zeros(net_.n_lanes)
        self._lane_of_move = net_.lane_of_move.astype(np.intp)
        self._entry_set = set(int(l) for l in net_.entry_links)
        self.reset(self.seed)

    # ==================================================================
    # lifecycle
    # ==================================================================
    def reset(self, seed: int | None = None, demand: DemandSpec | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        if demand is not None:
            self._spec0 = demand
        net = self.net
        self.active[:] = False
        self.phantom[:] = False
        self.lat[:] = 0.0
        self.hi = 0                                     # slots >= hi have never been used
        self._free = list(range(self.max_veh - 1, -1, -1))
        self.signals.reset()
        self.demand = DemandModel(self.cfg, net, self._spec0, self.seed)
        self._prng = np.random.default_rng([self.seed, 2])   # perturbation stream (jam destinations)
        self.closed = np.zeros(net.n_links, dtype=bool)
        self._closure_until = np.full(net.n_links, np.inf)
        self.routes: RoutingTables = net.base_routes
        self.incidents: dict = {}                       # id -> record
        self._next_incident = 1
        self._n_obst = 0
        self._lc_watch = False
        self._lat_active = False
        self._backlog = {int(l): deque() for l in net.entry_links}
        self._backlog_links: set = set()
        self.n_backlog = 0
        self._next_id = 1
        self.step_count = 0
        self.t = 0.0
        self.n_spawned = 0                              # vehicles generated (incl. still in backlog)
        self.n_completed = 0
        self.n_lane_changes = 0
        self.n_red_cross = 0                            # must stay 0 (red-light compliance)
        self._trips = {k: [] for k in _TRIP_FIELDS}
        self._reset_acc()

    def _reset_acc(self) -> None:
        N = self.n
        self._acc_steps = 0
        self._acc_halt = np.zeros(N, dtype=np.int64)
        self._acc_ev = np.zeros(N, dtype=np.int64)
        self._acc_cross = np.zeros(N, dtype=np.int64)
        self._acc_blocked = np.zeros(N, dtype=np.int64)
        self._acc_switched = np.zeros(N, dtype=bool)
        self._acc_gws = [0, 0, 0, 0]
        self._acc_speed = 0.0
        self._acc_vehsteps = 0

    # ==================================================================
    # control
    # ==================================================================
    def phase_demand(self) -> np.ndarray:
        """(N, 4) vehicles (halted + moving) on the lanes of each phase."""
        hi = self.hi
        m = self.active[:hi] & (self.etype[:hi] != ETYPE_OBSTACLE)
        cnt = np.bincount(self.lane[:hi][m], minlength=self.net.n_lanes)[: 8 * self.n].reshape(self.n, 8)
        return cnt[:, :N_PHASES] + cnt[:, N_PHASES:]

    def action_mask(self) -> np.ndarray:
        return self.signals.action_mask(self.phase_demand())

    def request_phases(self, phases: np.ndarray) -> np.ndarray:
        """Request a target phase per intersection; returns the phases actually executed."""
        executed, switched = self.signals.request(phases, self.phase_demand())
        self._acc_switched |= switched
        return executed

    # ==================================================================
    # one simulation step
    # ==================================================================
    def step(self) -> None:
        net, routes, vc, dt = self.net, self.routes, self.cfg.vehicle, self.dt
        self._expire()
        self._draw_arrivals()
        sig = self.signals.lane_signal()
        tail_rear, tail_v = self._tail_rear, self._tail_v
        tail_rear.fill(_BIG)

        hi = self.hi
        idx = np.flatnonzero(self.active[:hi])
        n = idx.size
        if n:
            # ---- 3. sort by lane, front first
            ln = self.lane[idx]
            p = self.pos[idx]
            order = np.lexsort((-p, ln))
            si = idx[order]
            ln = ln[order]
            p = p[order]
            v = self.vel[si]
            rear = p - self.length[si]
            dest = self.dest[si]
            et = self.etype[si]
            L = net.lane_len[ln]
            same = ln[1:] == ln[:-1]
            has_lead = np.empty(n, dtype=bool)
            has_lead[0] = False
            has_lead[1:] = same
            is_tail = np.empty(n, dtype=bool)
            is_tail[-1] = True
            is_tail[:-1] = ~same
            lead_rear = np.full(n, _BIG)
            lead_v = v.copy()
            lead_rear[1:][same] = rear[:-1][same]
            lead_v[1:][same] = v[:-1][same]
            x_lim = lead_rear - vc.overlap_margin
            tl = ln[is_tail]
            tail_rear[tl] = rear[is_tail]
            tail_v[tl] = v[is_tail]

            # ---- 4. stop-line logic for lane fronts
            mv = routes.lane_move[ln, dest]                 # movement this lane allows, -1 if none
            fi = np.flatnonzero(~has_lead)
            fl = ln[fi]
            fp = p[fi]
            fv = v[fi]
            fL = L[fi]
            fm = mv[fi]
            f_exit = net.lane_is_exit[fl]
            has_move = fm >= 0
            ms = np.where(has_move, fm, THROUGH).astype(np.intp)
            flink = fl >> 1
            tgt = net.move_target[flink, ms]
            tgt_s = np.where(tgt >= 0, tgt, 0)
            nm = routes.next_move[tgt_s, dest[fi]]
            kk = np.where(nm >= 0, self._lane_of_move[np.maximum(nm, 0)], ms == LEFT)
            tlane = 2 * tgt_s + kk
            dist = fL - fp
            sg = sig[fl]
            stop_ok = dist >= fv * fv / (2.0 * vc.decel_yellow)
            tr = tail_rear[tlane]
            tv = tail_v[tlane]
            t_has = tr < _HALF_BIG
            box = t_has & (tv < vc.box_block_speed) & (tr < vc.length + vc.s0)
            wall = ~f_exit & ((sg == SIG_RED) | ((sg == SIG_YELLOW) & stop_ok) | ~has_move | box)
            go_tail = ~f_exit & ~wall & t_has
            conn = net.conn_len[ms]
            f_rear = np.full(fi.size, _BIG)
            f_v = fv.copy()
            f_rear[wall] = fL[wall] - vc.stop_margin
            f_v[wall] = 0.0
            f_rear[go_tail] = fL[go_tail] + conn[go_tail] + tr[go_tail]
            f_v[go_tail] = tv[go_tail]
            lead_rear[fi] = f_rear
            lead_v[fi] = f_v
            x_lim[fi] = np.where(wall, fL - vc.wall_clamp, f_rear - vc.overlap_margin)

            # ---- 5. dynamics
            v_turn = net.v_turn
            mvs = np.where(mv >= 0, mv, THROUGH).astype(np.intp)
            cap = np.where(L - p < self.cfg.grid.turn_slow_dist, v_turn[mvs], np.inf)
            in_conn = p < 0.0
            if in_conn.any():
                cap = np.where(in_conn, v_turn[self.conn_move[si]], cap)
            v0e = np.minimum(self.v0[si], cap)
            acc = idm_accel(v, v0e, lead_rear - p, v - lead_v, self.a_max[si],
                            vc.headway, vc.decel, vc.s0, vc.decel_max)
            is_car = None
            if self._n_obst:
                is_car = et != ETYPE_OBSTACLE
                acc[~is_car] = 0.0
            x_new, v_new = integrate(p, v, acc, dt)
            x_new, v_new = project(p, x_new, v_new, x_lim, lead_v)
            self.pos[si] = x_new
            self.vel[si] = v_new

            # ---- 6. bookkeeping
            halted = v_new < vc.v_halt
            if is_car is not None:
                halted &= is_car
                self._acc_speed += float(v_new[is_car].sum())
                self._acc_vehsteps += int(is_car.sum())
            else:
                self._acc_speed += float(v_new.sum())
                self._acc_vehsteps += n
            if halted.any():
                hs = si[halted]
                self.wait_link[hs] += dt
                self.wait_total[hs] += dt
                self.stopped_on_link[hs] = True
                new_stop = halted & ~self.was_halted[si]
                if new_stop.any():
                    self.n_stops[si[new_stop]] += 1
                node = net.lane_node[ln]
                hm = halted & (node >= 0)                   # exit lanes belong to no agent
                if hm.any():
                    self._acc_halt += np.bincount(node[hm], minlength=self.n)
                    ev = hm & (et == ETYPE_EMERGENCY)
                    if ev.any():
                        self._acc_ev += np.bincount(node[ev], minlength=self.n)
            self.was_halted[si] = halted
            held = box & (sg == SIG_GREEN) & has_move & ~f_exit & (dist < vc.box_block_dist)
            if held.any():
                if is_car is not None:
                    held &= is_car[fi]
                self._acc_blocked += np.bincount(net.lane_node[fl[held]], minlength=self.n)

            # ---- 7. transfers and despawns (only lane fronts can reach the end)
            fx = x_new[fi]
            reach = fx >= fL
            if reach.any():
                cross = reach & ~f_exit & ~wall
                if cross.any():
                    self._transfer(si[fi[cross]], fl[cross], ms[cross], tlane[cross],
                                   fx[cross] - fL[cross] - conn[cross], sg[cross])
                done = reach & f_exit
                if done.any():
                    self._despawn(si[fi[done]])

        # ---- 8. spawns and lane changes
        if self._backlog_links:
            self._spawn_from_backlog()
        if self._lc_watch:
            self._lane_change_pass()
        if self._lat_active:
            lat = self.lat[: self.hi]
            np.subtract(lat, np.sign(lat) * min(1.0, dt / vc.lat_time), out=lat)
            lat[np.abs(lat) < 1e-9] = 0.0
            self._lat_active = bool(lat.any())

        # ---- 9. clock
        self.signals.advance()
        self.step_count += 1
        self.t = self.step_count * dt
        self._acc_steps += 1

    # ------------------------------------------------------------------
    def _transfer(self, vi, old_lane, move, new_lane, new_pos, sg) -> None:
        net = self.net
        node = net.lane_node[old_lane]
        self._acc_cross += np.bincount(node, minlength=self.n)
        self.n_red_cross += int(np.count_nonzero(sg == SIG_RED))
        nonstop = ~self.stopped_on_link[vi]
        through = move == THROUGH
        art = through & (net.link_kind[old_lane >> 1] == LINK_INTERNAL)
        g = self._acc_gws
        g[0] += int(vi.size)
        g[1] += int(nonstop.sum())
        g[2] += int(art.sum())
        g[3] += int((art & nonstop).sum())
        prog = nonstop & through
        cur = self.prog_cur[vi]
        ended = ~prog & (cur > 0)
        if ended.any():
            self.prog_sum[vi[ended]] += cur[ended]
            self.prog_runs[vi[ended]] += 1
        self.prog_cur[vi] = np.where(prog, cur + 1, 0)
        self.n_cross[vi] += 1
        self.stopped_on_link[vi] = False
        self.wait_link[vi] = 0.0
        self.prev_lane[vi] = old_lane
        self.conn_move[vi] = move
        self.lane[vi] = new_lane
        self.pos[vi] = new_pos

    def _despawn(self, vi) -> None:
        net, vc = self.net, self.cfg.vehicle
        cur = self.prog_cur[vi]
        prog_sum = self.prog_sum[vi] + cur
        prog_runs = self.prog_runs[vi] + (cur > 0)
        et = self.etype[vi]
        origin = self.origin[vi]
        dest = self.dest[vi]
        v_class = np.where(et == ETYPE_EMERGENCY, vc.v0_ev, vc.v0)
        tr = self._trips
        tr["veh_id"].append(self.vid[vi])
        tr["etype"].append(et)
        tr["origin"].append(origin)
        tr["dest"].append(dest)
        tr["t_spawn"].append(self.t_spawn[vi])
        tr["t_exit"].append(np.full(vi.size, self.t + self.dt))
        tr["ff_time"].append(net.base_routes.dist_to_go[origin, dest] / v_class)
        tr["wait_total"].append(self.wait_total[vi])
        tr["n_stops"].append(self.n_stops[vi])
        tr["n_cross"].append(self.n_cross[vi])
        tr["prog_sum"].append(prog_sum)
        tr["prog_runs"].append(prog_runs)
        self.active[vi] = False
        self._free.extend(int(i) for i in vi)
        self.n_completed += int(vi.size)

    # ------------------------------------------------------------------
    def _place(self, vid, lane, pos, vel, dest, v0_factor, etype, t_arr, origin) -> int:
        vc = self.cfg.vehicle
        s = self._free.pop()
        if s >= self.hi:
            self.hi = s + 1
        ev = etype == ETYPE_EMERGENCY
        self.active[s] = True
        self.phantom[s] = False
        self.vid[s] = vid
        self.lane[s] = lane
        self.pos[s] = pos
        self.v0[s] = (vc.v0_ev if ev else vc.v0) * v0_factor
        self.vel[s] = min(vel, self.v0[s])
        self.a_max[s] = vc.accel_ev if ev else vc.accel
        self.length[s] = vc.length
        self.dest[s] = dest
        self.prev_lane[s] = -1
        self.conn_move[s] = THROUGH
        self.etype[s] = etype
        self.origin[s] = origin
        self.wait_link[s] = 0.0
        self.wait_total[s] = 0.0
        self.n_stops[s] = 0
        self.n_cross[s] = 0
        self.was_halted[s] = False
        self.stopped_on_link[s] = False
        self.t_spawn[s] = t_arr
        self.lc_until[s] = 0.0
        self.lat[s] = 0.0
        self.prog_cur[s] = 0
        self.prog_sum[s] = 0
        self.prog_runs[s] = 0
        return s

    def _draw_arrivals(self) -> None:
        arr = self.demand.step(self.t)
        k = len(arr)
        if not k:
            return
        t = self.t
        entry, dest, f, et = arr.entry.tolist(), arr.dest.tolist(), arr.v0_factor.tolist(), arr.etype.tolist()
        for i in range(k):
            self._backlog[entry[i]].append((self._next_id, dest[i], f[i], et[i], t))
            self._next_id += 1
            self._backlog_links.add(entry[i])
        self.n_spawned += k
        self.n_backlog += k

    def _spawn_from_backlog(self) -> None:
        """Move waiting vehicles onto their entry lanes where there is room (FIFO per entry)."""
        net, routes, vc = self.net, self.routes, self.cfg.vehicle
        tail_rear, tail_v = self._tail_rear, self._tail_v
        for link in sorted(self._backlog_links):
            q = self._backlog[link]
            if self.closed[link]:
                continue
            used = 0                                      # bit mask of lanes filled this step
            while q and self._free:
                vid, dest, f, et, t_arr = q[0]
                if not np.isfinite(routes.cost_to_go[link, dest]):
                    dest = int(routes.fallback_dest[link])
                    if dest < 0:
                        break
                k = int(self._lane_of_move[routes.next_move[link, dest]])
                if used & (1 << k):
                    break
                ln = 2 * link + k
                gap = float(tail_rear[ln])                # rear of the last vehicle; huge if empty
                if gap < vc.s0:
                    break
                v_init = min(float(tail_v[ln]) if gap < _HALF_BIG else _BIG, (gap - vc.s0) / vc.headway)
                self._place(vid, ln, 0.0, v_init, dest, f, et, t_arr, link)
                q.popleft()
                self.n_backlog -= 1
                used |= 1 << k
            if not q:
                self._backlog_links.discard(link)

    # ------------------------------------------------------------------
    def _lane_change_pass(self) -> None:
        net, routes, vc = self.net, self.routes, self.cfg.vehicle
        hi = self.hi
        act = self.active[:hi]
        lane = self.lane[:hi]
        pos = self.pos[:hi]
        is_obst = act & (self.etype[:hi] == ETYPE_OBSTACLE)
        cars = act & ~is_obst
        want = np.zeros(hi, dtype=bool)
        obstacle_on_required = np.zeros(hi, dtype=bool)
        for o in np.flatnonzero(is_obst):
            ol, op = lane[o], pos[o]
            o_rear = op - self.length[o]
            want |= cars & (lane == ol) & (pos <= o_rear) & (pos >= o_rear - vc.lc_lookahead)
            obstacle_on_required |= cars & (lane == (ol ^ 1)) & (pos < op)
        link = lane >> 1
        nm = routes.next_move[link, self.dest[:hi]]
        req_k = self._lane_of_move[np.maximum(nm, 0)]
        wrong = cars & (nm >= 0) & (req_k != (lane & 1)) & ~net.lane_is_exit[lane]
        want |= wrong & ~obstacle_on_required
        if not is_obst.any() and not wrong.any():
            self._lc_watch = False
        want &= (pos >= 0.0) & (self.t >= self.lc_until[:hi])
        cand = np.flatnonzero(want)
        if not cand.size:
            return
        cand = cand[np.argsort(-pos[cand], kind="stable")]
        length, vel = self.length[:hi], self.vel[:hi]
        for c in cand:
            tl = int(lane[c]) ^ 1
            on = act & (lane == tl)
            x = pos[c]
            ahead = on & (pos > x)
            if ahead.any():
                j = np.flatnonzero(ahead)
                r = pos[j] - length[j]
                a = int(r.argmin())
                gap_front, v_lead = float(r[a] - x), float(vel[j[a]])
            else:
                gap_front, v_lead = _BIG, float(vel[c])
            behind = on & (pos <= x)
            if behind.any():
                j = np.flatnonzero(behind)
                b = int(pos[j].argmax())
                gap_back, v_foll = float(x - length[c] - pos[j[b]]), float(vel[j[b]])
            else:
                gap_back, v_foll = _BIG, 0.0
            if lane_change_ok(gap_front, float(vel[c]), v_lead, gap_back, v_foll, vc):
                self.lat[c] = 1.0 if (lane[c] & 1) == 0 else -1.0
                lane[c] = tl
                self.lc_until[c] = self.t + vc.lc_cooldown
                self.n_lane_changes += 1
                self._lat_active = True

    # ==================================================================
    # perturbations (call between steps)
    # ==================================================================
    def _expire(self) -> None:
        t = self.t
        self.demand.expire(t)
        if self.incidents:
            for iid in [i for i, r in self.incidents.items() if r["t_end"] <= t]:
                self.clear_incident(iid)
        if self.closed.any():
            due = np.flatnonzero(self.closed & (self._closure_until <= t))
            if due.size:
                self.closed[due] = False
                self._closure_until[due] = np.inf
                self._reroute()

    def _reroute(self) -> None:
        self.routes = self.net.compute_routes(self.closed) if self.closed.any() else self.net.base_routes
        idx = np.flatnonzero(self.active[: self.hi] & ~self.phantom[: self.hi])
        if idx.size:
            new = self.routes.resolve_dest(self.lane[idx] >> 1, self.dest[idx])
            self.dest[idx] = np.where(new >= 0, new, self.dest[idx])
        self._lc_watch = True

    def add_incident(self, link: int, lane: int, pos: float, duration: float | None = None) -> int:
        """Block one lane at ``pos``. Returns an incident id (-1 if no storage is left).

        If a vehicle occupies the spot it breaks down and becomes the obstacle
        (it drives on when the incident clears); otherwise a phantom obstacle of
        ``vehicle.incident_len`` is placed.
        """
        vc, net = self.cfg.vehicle, self.net
        ln = 2 * int(link) + int(lane)
        pos = float(np.clip(pos, vc.incident_len, net.lane_len[ln] - vc.stop_margin))
        hi = self.hi
        on = np.flatnonzero(self.active[:hi] & (self.lane[:hi] == ln))
        p = self.pos[on]
        hit = on[(p > pos - vc.incident_len) & (p - self.length[on] < pos)]
        t_end = np.inf if duration is None else self.t + float(duration)
        if hit.size:
            obst = hit[self.etype[hit] == ETYPE_OBSTACLE]
            if obst.size:                                   # already blocked here
                for iid, r in self.incidents.items():
                    if r["slot"] == int(obst[0]):
                        return iid
            cars = hit[self.pos[hit] >= 0.0]
            if not cars.size:
                cars = hit
            s = int(cars[np.abs(self.pos[cars] - pos).argmin()])
            rec = {"slot": s, "phantom": False, "etype": int(self.etype[s]), "t_end": t_end}
            self.etype[s] = ETYPE_OBSTACLE
            self.vel[s] = 0.0
        else:
            if not self._free:
                return -1
            s = self._place(0, ln, pos, 0.0, int(net.exit_links[0]), 1.0, ETYPE_OBSTACLE, self.t, link)
            self.phantom[s] = True
            self.length[s] = vc.incident_len
            rec = {"slot": s, "phantom": True, "etype": ETYPE_OBSTACLE, "t_end": t_end}
        iid = self._next_incident
        self._next_incident += 1
        self.incidents[iid] = rec
        self._n_obst += 1
        self._lc_watch = True
        return iid

    def clear_incident(self, incident_id: int) -> None:
        rec = self.incidents.pop(incident_id, None)
        if rec is None:
            return
        s = rec["slot"]
        if rec["phantom"]:
            self.active[s] = False
            self.phantom[s] = False
            self._free.append(s)
        else:
            self.etype[s] = rec["etype"]
            self.was_halted[s] = True
        self._n_obst -= 1

    def close_link(self, link: int, duration: float | None = None) -> None:
        """Forbid entering ``link``; vehicles already on it drain. Routes are recomputed."""
        link = int(link)
        self.closed[link] = True
        self._closure_until[link] = np.inf if duration is None else self.t + float(duration)
        self._reroute()

    def open_link(self, link: int) -> None:
        link = int(link)
        if self.closed[link]:
            self.closed[link] = False
            self._closure_until[link] = np.inf
            self._reroute()

    def clear_perturbations(self) -> None:
        """Remove every incident, reopen every closed link and cancel every surge.
        Vehicles already placed by a jam or sent as emergency vehicles stay; they are traffic."""
        for iid in list(self.incidents):
            self.clear_incident(iid)
        if self.closed.any():
            self.closed[:] = False
            self._closure_until[:] = np.inf
            self._reroute()
        self.demand.clear_surges()

    def set_surge(self, entries, multiplier: float, duration: float) -> None:
        """Multiply the arrival rate of the given entry links for ``duration`` seconds."""
        self.demand.add_surge(entries, multiplier, self.t + float(duration))

    def set_demand_level(self, base_rate: float) -> None:
        self.demand.set_base_rate(base_rate)

    def spawn_emergency(self, entry_link: int, dest_exit: int | None = None) -> int:
        """Queue an emergency vehicle at the head of an entry's backlog. Returns its id."""
        entry_link = int(entry_link)
        if entry_link not in self._entry_set:
            raise ValueError(f"link {entry_link} is not an entry link")
        dest = int(self.net.straight_exit[entry_link]) if dest_exit is None else int(dest_exit)
        vid = self._next_id
        self._next_id += 1
        self._backlog[entry_link].appendleft((vid, dest, 1.0, ETYPE_EMERGENCY, self.t))
        self._backlog_links.add(entry_link)
        self.n_spawned += 1
        self.n_backlog += 1
        return vid

    def inject_jam(self, link: int, n: int) -> int:
        """Place up to ``n`` stopped vehicles at the back of the queues on ``link``."""
        net, routes, vc = self.net, self.routes, self.cfg.vehicle
        link = int(link)
        L = float(net.link_len[link])
        is_exit = bool(net.is_exit[link])
        hi = self.hi
        top, floor = [], []
        for k in (0, 1):
            on = np.flatnonzero(self.active[:hi] & (self.lane[:hi] == 2 * link + k))
            p = self.pos[on]
            body = p >= 0.0
            head = L - vc.stop_margin - vc.s0
            top.append(min(head, float((p[body] - self.length[on][body]).min()) - vc.s0) if body.any() else head)
            floor.append(max(0.0, float(p[~body].max()) + vc.s0) if (~body).any() else 0.0)
        placed = attempts = 0
        while placed < n and attempts < 4 * n and self._free:
            attempts += 1
            if is_exit:
                dest, k = link, int(top[1] > top[0])
            else:
                dest = int(net.exit_links[self._prng.integers(net.exit_links.size)])
                if not np.isfinite(routes.cost_to_go[link, dest]):
                    continue
                k = int(self._lane_of_move[routes.next_move[link, dest]])
            x = top[k]
            if x - vc.length < floor[k]:
                continue
            self._place(self._next_id, 2 * link + k, x, 0.0, dest, 1.0, ETYPE_CAR, self.t, link)
            self._next_id += 1
            top[k] = x - vc.length - vc.s0
            placed += 1
        self.n_spawned += placed
        return placed

    # ==================================================================
    # read-outs
    # ==================================================================
    @property
    def n_vehicles(self) -> int:
        """Vehicles currently on the road (phantom obstacles excluded)."""
        hi = self.hi
        return int(np.count_nonzero(self.active[:hi] & ~self.phantom[:hi]))

    def mean_speed(self) -> float:
        hi = self.hi
        m = self.active[:hi] & (self.etype[:hi] != ETYPE_OBSTACLE)
        return float(self.vel[:hi][m].mean()) if m.any() else self.cfg.vehicle.v0

    def lane_stats(self) -> LaneStats:
        net, vc = self.net, self.cfg.vehicle
        hi = self.hi
        idx = np.flatnonzero(self.active[:hi] & (self.etype[:hi] != ETYPE_OBSTACLE))
        ln = self.lane[idx]
        v = self.vel[idx]
        nl, nk = net.n_lanes, net.n_links
        h = v < vc.v_halt
        halted = np.bincount(ln[h], minlength=nl)
        moving = np.bincount(ln[~h], minlength=nl)
        speed_sum = np.bincount(ln[~h], weights=v[~h], minlength=nl)
        wl = self.wait_link[idx]
        wait_sum = np.bincount(ln, weights=wl, minlength=nl)
        wait_max = np.zeros(nl)
        waiting = wl > 0.0
        if waiting.any():
            np.maximum.at(wait_max, ln[waiting], wl[waiting])
        ev_prox = np.zeros(nl)
        e = self.etype[idx] == ETYPE_EMERGENCY
        if e.any():
            floor = self.cfg.obs.ev_floor
            frac = np.clip(self.pos[idx][e] / net.lane_len[ln[e]], 0.0, 1.0)
            np.maximum.at(ev_prox, ln[e], floor + (1.0 - floor) * frac)
        link = ln >> 1
        link_count = np.bincount(link, minlength=nk)
        mv = self.routes.lane_move[ln, self.dest[idx]]
        ok = mv >= 0
        inbound = np.bincount(net.move_target[link[ok], mv[ok]], minlength=nk)
        move_count = np.bincount(3 * ln[ok] + mv[ok], minlength=3 * nl).reshape(nl, 3)
        return LaneStats(halted=halted, moving=moving, speed_sum=speed_sum, wait_sum=wait_sum,
                         wait_max=wait_max, ev_prox=ev_prox, link_count=link_count, inbound=inbound, move_count=move_count)

    def unfinished_time(self) -> float:
        """Vehicle-seconds spent so far by every vehicle that has not finished its trip
        (on the road or waiting in a backlog). Used for the censoring-free ``time_in_system``."""
        hi = self.hi
        m = self.active[:hi] & ~self.phantom[:hi]
        total = float((self.t - self.t_spawn[:hi][m]).sum())
        for link in self._backlog_links:
            total += sum(self.t - e[4] for e in self._backlog[link])
        return total

    def drain_interval_stats(self) -> IntervalStats:
        """Reward accumulators since the last call; resets them."""
        net = self.net
        hi = self.hi
        idx = np.flatnonzero(self.active[:hi] & (self.etype[:hi] != ETYPE_OBSTACLE))
        node = net.lane_node[self.lane[idx]]
        ok = node >= 0
        wait_max = np.zeros(self.n)
        if ok.any():
            np.maximum.at(wait_max, node[ok], self.wait_link[idx][ok])
        g = self._acc_gws
        out = IntervalStats(steps=self._acc_steps, halt_sum=self._acc_halt, ev_halt_sum=self._acc_ev,
                            crossings=self._acc_cross, blocked_sum=self._acc_blocked, wait_max=wait_max,
                            switched=self._acc_switched, cross_all=g[0], cross_nonstop=g[1],
                            art_all=g[2], art_nonstop=g[3], speed_sum=self._acc_speed,
                            veh_steps=self._acc_vehsteps)
        self._reset_acc()
        return out

    def trip_log(self) -> TripLog:
        """Trips completed since the last call; clears the buffer."""
        tr = self._trips
        if tr["veh_id"]:
            out = TripLog(**{k: np.concatenate(v) for k, v in tr.items()})
        else:
            out = TripLog(**{k: np.zeros(0) for k in _TRIP_FIELDS})
        self._trips = {k: [] for k in _TRIP_FIELDS}
        return out

    def snapshot(self) -> dict:
        """Render frame, JSON-serialisable.

        ``veh`` rows: ``[id, x, y, angle, etype, v_norm, lat]`` where (x, y) is the
        FRONT bumper, ``v_norm = v / V0`` and ``lat`` is a lateral offset in lane
        widths (positive = to the right) that decays to 0 after a lane change.
        ``sig`` rows: ``[phase, stage, green_elapsed]`` (phase = the old phase
        while YELLOW / ALL_RED). ``incidents`` rows: ``[id, x, y, angle]`` at the
        obstacle centre.
        """
        net, sg = self.net, self.signals
        hi = self.hi
        idx = np.flatnonzero(self.active[:hi] & ~self.phantom[:hi])
        x, y, ang = net.lane_xy(self.lane[idx], self.pos[idx], self.prev_lane[idx])
        vn = np.clip(self.vel[idx] / self.cfg.vehicle.v0, 0.0, 2.0)
        veh = [list(r) for r in zip(self.vid[idx].tolist(), np.round(x, 1).tolist(), np.round(y, 1).tolist(),
                                    np.round(ang, 2).tolist(), self.etype[idx].tolist(),
                                    np.round(vn, 2).tolist(), np.round(self.lat[idx], 2).tolist())]
        sig = [list(r) for r in zip(sg.phase.tolist(), sg.stage.tolist(),
                                    np.round(sg.green_elapsed, 1).tolist())]
        inc = []
        for iid, r in self.incidents.items():
            s = r["slot"]
            ix, iy, ia = net.lane_xy(self.lane[s], self.pos[s] - 0.5 * self.length[s], self.prev_lane[s])
            inc.append([iid, round(float(ix), 1), round(float(iy), 1), round(float(ia), 2)])
        return {"t": round(self.t, 3), "veh": veh, "sig": sig, "incidents": inc,
                "closed": np.flatnonzero(self.closed).tolist(),
                "n_vehicles": len(veh), "backlog": self.n_backlog}
