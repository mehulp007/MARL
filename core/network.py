"""Grid topology, lane/link indexing, routing tables and geometry.

``Network`` is immutable after construction: it holds topology and geometry
only. Closures live in the simulator; routing for a given closure set is a
pure function, ``Network.compute_routes(closed) -> RoutingTables``.

Indexing recap
--------------
* intersection ``i = r * cols + c`` (row 0 = north edge, col 0 = west edge)
* heading ``h``: 0=N, 1=E, 2=S, 3=W (direction of travel); y grows SOUTH
* incoming link of ``i`` with heading ``h``: ``in_link[i, h] = 4*i + h``
* exit links: ids ``4*N ...``
* lane id ``= 2*link + k`` with ``k=0`` outer (THROUGH+RIGHT), ``k=1`` inner (LEFT)
* the 8 incoming lanes of agent ``i`` are exactly ``8*i ... 8*i + 7`` in
  observation order ``j = 2*h + k``
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    Config,
    HEADING_DX,
    HEADING_DY,
    HEADING_NAMES,
    LANES,
    LANE_OF_MOVE,
    LEFT,
    LINK_ENTRY,
    LINK_EXIT,
    LINK_INTERNAL,
    MOVE_DH,
    N_PHASES,
    PHASE_NAMES,
    RIGHT,
    THROUGH,
)

_INF = np.inf


@dataclass(frozen=True)
class RoutingTables:
    """Destination-based routing for one closure set.

    All tables are indexed ``[link, dest]`` (or ``[lane, dest]``) where ``dest``
    is an exit *link id*; columns of non-exit links are inf / -1.
    """

    closed: np.ndarray        # (L,) bool   links that may not be entered
    cost_to_go: np.ndarray    # (L, L) f64  seconds from the start of link to the end of dest
    dist_to_go: np.ndarray    # (L, L) f64  metres along the chosen route (incl. junction paths)
    next_move: np.ndarray     # (L, L) i8   best movement at the end of link, -1 if none
    lane_move: np.ndarray     # (2L, L) i8  best movement *permitted by this lane*, -1 if none
    fallback_dest: np.ndarray  # (L,) i32   cheapest reachable exit from link, -1 if none

    def resolve_dest(self, link: np.ndarray, dest: np.ndarray) -> np.ndarray:
        """Keep ``dest`` where reachable from ``link``; otherwise the fallback exit."""
        link = np.asarray(link)
        dest = np.asarray(dest)
        ok = np.isfinite(self.cost_to_go[link, dest])
        return np.where(ok, dest, self.fallback_dest[link])


class Network:
    """Immutable description of the city grid."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        g = cfg.grid
        R, C = g.rows, g.cols
        N = R * C
        self.rows, self.cols, self.n = R, C, N
        self.n_links = cfg.n_links
        self.n_lanes = cfg.n_lanes
        self.n_exits = cfg.n_exits
        self.exit_base = 4 * N
        L = self.n_links

        idx = np.arange(N)
        self.node_r = idx // C
        self.node_c = idx % C
        self.node_x = self.node_c * cfg.block
        self.node_y = self.node_r * cfg.block

        # ---- neighbours: nbr[i, d] = intersection located in direction d of i
        nbr = -np.ones((N, 4), dtype=np.int32)
        m = self.node_r > 0
        nbr[m, 0] = idx[m] - C
        m = self.node_c < C - 1
        nbr[m, 1] = idx[m] + 1
        m = self.node_r < R - 1
        nbr[m, 2] = idx[m] + C
        m = self.node_c > 0
        nbr[m, 3] = idx[m] - 1
        self.nbr = nbr
        self.nbr_mask = nbr >= 0
        self.n_nbr = self.nbr_mask.sum(axis=1)

        # ---- links
        self.in_link = (4 * idx[:, None] + np.arange(4)[None, :]).astype(np.int32)
        out_link = np.empty((N, 4), dtype=np.int32)
        link_heading = np.empty(L, dtype=np.int8)
        link_src = -np.ones(L, dtype=np.int32)
        link_dst = -np.ones(L, dtype=np.int32)
        link_kind = np.empty(L, dtype=np.int8)
        link_side = -np.ones(L, dtype=np.int8)     # city edge of entry/exit links

        next_exit = self.exit_base
        for i in range(N):
            for h in range(4):
                l = 4 * i + h
                up = nbr[i, (h + 2) % 4]           # upstream node sits opposite to the heading
                link_heading[l] = h
                link_dst[l] = i
                link_src[l] = up
                if up >= 0:
                    link_kind[l] = LINK_INTERNAL
                else:
                    link_kind[l] = LINK_ENTRY
                    link_side[l] = (h + 2) % 4
                j = nbr[i, h]
                if j >= 0:
                    out_link[i, h] = 4 * j + h
                else:
                    e = next_exit
                    next_exit += 1
                    out_link[i, h] = e
                    link_heading[e] = h
                    link_src[e] = i
                    link_kind[e] = LINK_EXIT
                    link_side[e] = h
        assert next_exit == L
        self.out_link = out_link
        self.link_heading = link_heading
        self.link_src = link_src
        self.link_dst = link_dst
        self.link_kind = link_kind
        self.link_side = link_side
        self.link_len = np.where(link_kind == LINK_EXIT, g.exit_len, g.link_len).astype(np.float64)
        self.is_exit = link_kind == LINK_EXIT
        self.entry_links = np.flatnonzero(link_kind == LINK_ENTRY).astype(np.int32)
        self.exit_links = np.flatnonzero(link_kind == LINK_EXIT).astype(np.int32)
        self.internal_links = np.flatnonzero(link_kind == LINK_INTERNAL).astype(np.int32)
        self.entries_by_side = tuple(
            self.entry_links[link_side[self.entry_links] == s] for s in range(4))
        self.exits_by_side = tuple(
            self.exit_links[link_side[self.exit_links] == s] for s in range(4))

        # ---- movements: move_target[link, m] = link reached by movement m (-1 on exit links)
        move_target = -np.ones((L, 3), dtype=np.int32)
        for l in range(4 * N):
            i, h = link_dst[l], link_heading[l]
            for mv in range(3):
                move_target[l, mv] = out_link[i, (h + MOVE_DH[mv]) % 4]
        self.move_target = move_target
        self.lane_of_move = np.asarray(LANE_OF_MOVE, dtype=np.int8)

        # ---- straight-across and U-turn exits (used by the demand model)
        straight_exit = -np.ones(L, dtype=np.int32)
        for l in range(4 * N):
            cur = l
            while not self.is_exit[cur]:
                cur = move_target[cur, THROUGH]
            straight_exit[l] = cur
        self.straight_exit = straight_exit
        uturn_exit = -np.ones(L, dtype=np.int32)
        for l in self.entry_links:
            uturn_exit[l] = out_link[link_dst[l], (link_heading[l] + 2) % 4]
        self.uturn_exit = uturn_exit

        # ---- lanes
        lane = np.arange(self.n_lanes)
        self.lane_link = (lane // LANES).astype(np.int32)
        self.lane_k = (lane % LANES).astype(np.int8)
        self.lane_len = self.link_len[self.lane_link]
        self.lane_heading = link_heading[self.lane_link]
        self.lane_node = link_dst[self.lane_link]            # controlling intersection, -1 on exits
        self.lane_is_exit = self.is_exit[self.lane_link]
        self.lane_phase = np.where(
            self.lane_node >= 0, 2 * (self.lane_heading % 2) + self.lane_k, -1).astype(np.int8)
        # incoming lanes of agent i in observation order j = 2*h + k
        self.agent_lanes = (8 * idx[:, None] + np.arange(8)[None, :]).astype(np.int32)
        # phase_lanes[i, p] = the two lanes that are green in phase p
        phase_lanes = np.empty((N, N_PHASES, 2), dtype=np.int32)
        for p in range(N_PHASES):
            hs = (0, 2) if p < 2 else (1, 3)
            for a, h in enumerate(hs):
                phase_lanes[:, p, a] = 8 * idx + 2 * h + (p % 2)
        self.phase_lanes = phase_lanes

        # ---- geometry
        ux = np.asarray(HEADING_DX)[link_heading]
        uy = np.asarray(HEADING_DY)[link_heading]
        rx, ry = -uy, ux                                   # right-hand side of the heading
        sx = np.empty(L)
        sy = np.empty(L)
        inc = ~self.is_exit
        # incoming: stop line on the box edge, start one link length upstream
        sx[inc] = self.node_x[link_dst[inc]] - (g.box + g.link_len) * ux[inc]
        sy[inc] = self.node_y[link_dst[inc]] - (g.box + g.link_len) * uy[inc]
        ex = self.is_exit
        sx[ex] = self.node_x[link_src[ex]] + g.box * ux[ex]
        sy[ex] = self.node_y[link_src[ex]] + g.box * uy[ex]
        self.link_x0, self.link_y0 = sx, sy                 # on the road centre line
        self.link_ux, self.link_uy = ux, uy
        self.link_rx, self.link_ry = rx, ry
        off = (LANES - 0.5 - self.lane_k) * g.lane_w        # k=0 -> 1.5w, k=1 -> 0.5w
        ll = self.lane_link
        self.lane_ux, self.lane_uy = ux[ll], uy[ll]
        self.lane_x0 = sx[ll] + off * rx[ll]
        self.lane_y0 = sy[ll] + off * ry[ll]
        self.lane_x1 = self.lane_x0 + self.lane_len * self.lane_ux
        self.lane_y1 = self.lane_y0 + self.lane_len * self.lane_uy
        self.lane_angle = np.arctan2(self.lane_uy, self.lane_ux)
        self.conn_len = np.asarray(g.conn_len, dtype=np.float64)
        # speed cap per movement (THROUGH is uncapped)
        self.v_turn = np.array([g.v_turn_left, _INF, g.v_turn_right])
        # movement implied by a heading change (h_new - h_old) % 4
        self._move_of_dh = np.array([THROUGH, RIGHT, THROUGH, LEFT], dtype=np.int8)

        self._base_routes: RoutingTables | None = None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @property
    def base_routes(self) -> RoutingTables:
        """Routing with no closures (cached)."""
        if self._base_routes is None:
            self._base_routes = self.compute_routes(None)
        return self._base_routes

    def compute_routes(self, closed: np.ndarray | None = None) -> RoutingTables:
        """Shortest-time routing tables for a set of closed links.

        A closed link may not be *entered*; vehicles already on it drain normally.
        Edge cost of taking movement m from link l into link t:
        ``turn_penalty[m] + conn_len[m]/v0``; node cost of a link: ``len/v0``.
        """
        g, v0 = self.cfg.grid, self.cfg.vehicle.v0
        L = self.n_links
        closed = np.zeros(L, dtype=bool) if closed is None else np.asarray(closed, dtype=bool).copy()
        t_link = self.link_len / v0
        pen = np.asarray(g.turn_penalty, dtype=np.float64) + self.conn_len / v0   # (3,)
        tgt = self.move_target
        valid = tgt >= 0
        tgt_safe = np.where(valid, tgt, 0)
        edge_open = valid & ~closed[tgt_safe]                                       # (L, 3)
        ex = self.exit_links

        def expand(c: np.ndarray) -> np.ndarray:
            cand = pen[None, :, None] + c[tgt_safe]                                 # (L, 3, L)
            return np.where(edge_open[:, :, None], cand, _INF)

        cost = np.full((L, L), _INF)
        cost[ex, ex] = t_link[ex]
        for _ in range(4 * L + 4):                    # value iteration; exact fixed point
            new = t_link[:, None] + expand(cost).min(axis=1)
            new[ex, :] = _INF
            new[ex, ex] = t_link[ex]
            if np.array_equal(new, cost):
                break
            cost = new
        cand = np.round(expand(cost), 6)              # round so equal-cost routes tie exactly

        def argmin_moves(moves: tuple) -> np.ndarray:
            sub = cand[:, list(moves), :]                                           # (L, k, L)
            best = np.asarray(moves, dtype=np.int8)[sub.argmin(axis=1)]
            return np.where(np.isfinite(sub.min(axis=1)), best, -1).astype(np.int8)

        # ties prefer THROUGH (keeps platoons on their arterial), then LEFT, then RIGHT
        next_move = argmin_moves((THROUGH, LEFT, RIGHT))
        lane_move = np.empty((self.n_lanes, L), dtype=np.int8)
        lane_move[0::2] = argmin_moves((THROUGH, RIGHT))      # outer lane
        lane_move[1::2] = argmin_moves((LEFT,))               # inner lane

        col = cost[:, ex]
        fallback = np.where(np.isfinite(col.min(axis=1)), ex[col.argmin(axis=1)], -1).astype(np.int32)

        # route length along next_move
        has = next_move >= 0
        mv = np.where(has, next_move, 0).astype(np.intp)
        # tgt_safe is (L, 3) and mv is (L, L): gather the target link per (link, dest)
        nxt = tgt_safe[np.arange(L)[:, None], mv]                                    # (L, L)
        step = self.link_len[:, None] + self.conn_len[mv]
        cols = np.arange(L)[None, :]
        dist = np.full((L, L), _INF)
        dist[ex, ex] = self.link_len[ex]
        for _ in range(4 * L + 4):
            new = np.where(has, step + dist[nxt, cols], _INF)
            new[ex, ex] = self.link_len[ex]
            if np.array_equal(new, dist):
                break
            dist = new

        for a in (closed, cost, dist, next_move, lane_move, fallback):
            a.setflags(write=False)
        return RoutingTables(closed=closed, cost_to_go=cost, dist_to_go=dist,
                             next_move=next_move, lane_move=lane_move, fallback_dest=fallback)

    def route_links(self, link: int, dest: int, routes: RoutingTables | None = None) -> list:
        """Sequence of links from ``link`` to ``dest`` following ``next_move`` ([] if unreachable)."""
        routes = routes or self.base_routes
        if not np.isfinite(routes.cost_to_go[link, dest]):
            return []
        path = [int(link)]
        cur = int(link)
        while cur != dest:
            cur = int(self.move_target[cur, routes.next_move[cur, dest]])
            path.append(cur)
            if len(path) > self.n_links:
                raise RuntimeError("routing loop")
        return path

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def move_between(self, prev_lane: np.ndarray, lane: np.ndarray) -> np.ndarray:
        """Movement that leads from ``prev_lane`` onto ``lane`` (by heading change)."""
        dh = (self.lane_heading[lane].astype(np.int64) - self.lane_heading[prev_lane]) % 4
        return self._move_of_dh[dh]

    def lane_xy(self, lane, pos, prev_lane=None):
        """World position and heading angle of points on lanes (vectorised).

        ``pos >= 0``: straight along the lane. ``pos < 0`` (inside the junction,): a cubic Bezier from the end of ``prev_lane`` to the start of
        ``lane``, tangent to both, parameterised by ``1 + pos / CONN_LEN[move]``.
        Where ``prev_lane`` is missing (< 0 or None) the lane is extended straight
        backwards. Returns ``(x, y, angle)`` arrays shaped like the broadcast input.
        """
        lane = np.asarray(lane, dtype=np.intp)
        pos = np.asarray(pos, dtype=np.float64)
        lane, pos = np.broadcast_arrays(lane, pos)
        shape = lane.shape
        lane, pos = lane.reshape(-1), pos.reshape(-1)
        x = self.lane_x0[lane] + pos * self.lane_ux[lane]
        y = self.lane_y0[lane] + pos * self.lane_uy[lane]
        ang = self.lane_angle[lane].copy()
        if prev_lane is None:
            return x.reshape(shape), y.reshape(shape), ang.reshape(shape)
        prev = np.broadcast_to(np.asarray(prev_lane, dtype=np.intp), shape).reshape(-1)
        inj = (pos < 0) & (prev >= 0)
        if not inj.any():
            return x.reshape(shape), y.reshape(shape), ang.reshape(shape)
        ln, pl, ps = lane[inj], prev[inj], pos[inj]
        mv = self.move_between(pl, ln)
        s = np.clip(1.0 + ps / self.conn_len[mv], 0.0, 1.0)
        p0x, p0y = self.lane_x1[pl], self.lane_y1[pl]
        p3x, p3y = self.lane_x0[ln], self.lane_y0[ln]
        d = np.hypot(p3x - p0x, p3y - p0y)
        c = np.where(mv == THROUGH, d / 3.0, 0.39 * d)        # 0.39*d ~ circular arc
        p1x, p1y = p0x + c * self.lane_ux[pl], p0y + c * self.lane_uy[pl]
        p2x, p2y = p3x - c * self.lane_ux[ln], p3y - c * self.lane_uy[ln]
        q = 1.0 - s
        b0, b1, b2, b3 = q ** 3, 3 * q * q * s, 3 * q * s * s, s ** 3
        x[inj] = b0 * p0x + b1 * p1x + b2 * p2x + b3 * p3x
        y[inj] = b0 * p0y + b1 * p1y + b2 * p2y + b3 * p3y
        tx = 3 * q * q * (p1x - p0x) + 6 * q * s * (p2x - p1x) + 3 * s * s * (p3x - p2x)
        ty = 3 * q * q * (p1y - p0y) + 6 * q * s * (p2y - p1y) + 3 * s * s * (p3y - p2y)
        ang[inj] = np.arctan2(ty, tx)
        return x.reshape(shape), y.reshape(shape), ang.reshape(shape)

    def pick(self, x: float, y: float, max_dist: float | None = None):
        """Nearest lane to a world point -> ``(link, lane_k, pos)`` or ``None``."""
        if max_dist is None:
            max_dist = self.cfg.grid.pick_radius
        t = np.clip((x - self.lane_x0) * self.lane_ux + (y - self.lane_y0) * self.lane_uy,
                    0.0, self.lane_len)
        d = np.hypot(x - (self.lane_x0 + t * self.lane_ux), y - (self.lane_y0 + t * self.lane_uy))
        j = int(d.argmin())
        if d[j] > max_dist:
            return None
        return int(self.lane_link[j]), int(self.lane_k[j]), float(t[j])

    def link_label(self, link: int) -> str:
        """Short human label, e.g. entry 'N2' = second entry on the north edge."""
        kind = int(self.link_kind[link])
        if kind == LINK_INTERNAL:
            return f"{int(self.link_src[link])}>{int(self.link_dst[link])}"
        node = int(self.link_dst[link] if kind == LINK_ENTRY else self.link_src[link])
        side = int(self.link_side[link])
        along = self.node_c[node] if side in (0, 2) else self.node_r[node]
        return f"{HEADING_NAMES[side]}{int(along) + 1}"

    def static_geometry(self) -> dict:
        """Everything the frontend draws once. JSON-serialisable, metres, y south."""
        g, cfg = self.cfg.grid, self.cfg
        r2 = lambda v: round(float(v), 2)
        road_w = LANES * g.lane_w
        links, roads = [], []
        for l in range(self.n_links):
            x0, y0 = self.link_x0[l], self.link_y0[l]
            x1 = x0 + self.link_len[l] * self.link_ux[l]
            y1 = y0 + self.link_len[l] * self.link_uy[l]
            ox, oy = road_w * self.link_rx[l], road_w * self.link_ry[l]
            links.append({
                "id": l, "kind": int(self.link_kind[l]), "heading": int(self.link_heading[l]),
                "src": int(self.link_src[l]), "dst": int(self.link_dst[l]),
                "side": int(self.link_side[l]), "len": r2(self.link_len[l]),
                "label": self.link_label(l),
                "x0": r2(x0), "y0": r2(y0), "x1": r2(x1), "y1": r2(y1),
            })
            roads.append({"link": l, "poly": [[r2(x0), r2(y0)], [r2(x1), r2(y1)],
                                              [r2(x1 + ox), r2(y1 + oy)], [r2(x0 + ox), r2(y0 + oy)]]})
        lanes, stop_lines = [], []
        hw = 0.5 * g.lane_w
        for j in range(self.n_lanes):
            l = int(self.lane_link[j])
            lanes.append({"id": j, "link": l, "k": int(self.lane_k[j]),
                          "phase": int(self.lane_phase[j]), "node": int(self.lane_node[j]),
                          "x0": r2(self.lane_x0[j]), "y0": r2(self.lane_y0[j]),
                          "x1": r2(self.lane_x1[j]), "y1": r2(self.lane_y1[j])})
            if not self.lane_is_exit[j]:
                ox, oy = hw * self.link_rx[l], hw * self.link_ry[l]
                stop_lines.append({"lane": j, "node": int(self.lane_node[j]),
                                   "phase": int(self.lane_phase[j]),
                                   "x0": r2(self.lane_x1[j] - ox), "y0": r2(self.lane_y1[j] - oy),
                                   "x1": r2(self.lane_x1[j] + ox), "y1": r2(self.lane_y1[j] + oy)})
        inter = [{"id": i, "r": int(self.node_r[i]), "c": int(self.node_c[i]),
                  "x": r2(self.node_x[i]), "y": r2(self.node_y[i]), "half": r2(g.box)}
                 for i in range(self.n)]

        def terminal(l: int, at_start: bool) -> dict:
            px = self.link_x0[l] + (0.0 if at_start else self.link_len[l]) * self.link_ux[l]
            py = self.link_y0[l] + (0.0 if at_start else self.link_len[l]) * self.link_uy[l]
            px += 0.5 * road_w * self.link_rx[l]
            py += 0.5 * road_w * self.link_ry[l]
            node = int(self.link_dst[l] if at_start else self.link_src[l])
            return {"link": int(l), "node": node, "heading": int(self.link_heading[l]),
                    "side": int(self.link_side[l]), "label": self.link_label(l),
                    "x": r2(px), "y": r2(py)}

        margin = g.box + max(g.link_len, g.exit_len)
        return {
            "rows": self.rows, "cols": self.cols,
            "bounds": [r2(-margin), r2(-margin),
                       r2(self.node_x.max() + margin), r2(self.node_y.max() + margin)],
            "constants": {
                "lane_w": g.lane_w, "box": g.box, "block": cfg.block, "lanes": LANES,
                "link_len": g.link_len, "exit_len": g.exit_len,
                "veh_len": cfg.vehicle.length, "incident_len": cfg.vehicle.incident_len,
                "v0": cfg.vehicle.v0, "dt": cfg.sim.dt, "decision_dt": cfg.signal.decision_dt,
                "g_max": cfg.signal.g_max, "heading_names": list(HEADING_NAMES),
                "phase_names": list(PHASE_NAMES),
            },
            "intersections": inter,
            "links": links,
            "roads": roads,
            "lanes": lanes,
            "stop_lines": stop_lines,
            "entries": [terminal(int(l), True) for l in self.entry_links],
            "exits": [terminal(int(l), False) for l in self.exit_links],
        }
